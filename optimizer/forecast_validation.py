"""Blocked model selection, separate calibration, and a final temporal test."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def fit_ridge(x, y, regularization):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1])*regularization
    penalty[0, 0] = 0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return dict(mean=mean, scale=scale, coef=coef)


def fit_candidate(spec, x, y, cache=None):
    cache = {} if cache is None else cache
    key = repr(spec)
    if key in cache:
        return cache[key]
    model = dict(spec)
    if spec['method'] == 'ridge':
        model['fit'] = fit_ridge(x[:, :spec['features']], y, spec['regularization'])
    elif spec['method'] == 'boosted_trees':
        with threadpool_limits(limits=1):
            model['fit'] = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.05, max_iter=120,
                                                        max_leaf_nodes=spec['leaves'], min_samples_leaf=30,
                                                        l2_regularization=10, early_stopping=False, random_state=88).fit(x, y)
    elif spec['method'] == 'ensemble':
        model['members'] = [fit_candidate(member, x, y, cache) for member in spec['components']]
    cache[key] = model
    return model


def predict(model, x):
    if model['method'] == 'recent_mean':
        raw = x[:, 0]
    elif model['method'] == 'ridge':
        fitted = model['fit']; x = x[:, :model['features']]
        design = np.column_stack([np.ones(len(x)), (x-fitted['mean'])/fitted['scale']])
        raw = design @ fitted['coef']
    elif model['method'] == 'ensemble':
        raw = np.average([predict(member, x) for member in model['members']], axis=0, weights=model['weights'])
    else:
        with threadpool_limits(limits=1):
            raw = model['fit'].predict(x)
    return np.maximum(raw, 0)


def residual_sample(model, projection):
    """Prefer score-dependent calibration bins when they have at least 30 games."""
    bounds = model.get('residual_bounds', [])
    key = int(np.searchsorted(bounds, projection))
    return np.asarray(model.get('residual_bins', {}).get(key, model['residuals']), dtype=float)


def prediction_interval(model, projection, coverage=.8):
    """Finite-sample absolute-error order statistic; temporal coverage is measured, not guaranteed."""
    errors = np.abs(residual_sample(model, projection))
    if not len(errors) or not np.isfinite(errors).all() or not 0 < coverage < 1:
        raise ValueError('Intervals require finite held-out residuals and coverage between zero and one.')
    if model.get('interval_method') == 'equal_tail':
        return tuple(projection+np.quantile(residual_sample(model, projection), [(1-coverage)/2, (1+coverage)/2]))
    order = min(len(errors), int(np.ceil((len(errors)+1)*coverage)))
    radius = float(np.partition(errors, order-1)[order-1])
    return projection-radius, projection+radius


def train_models(records, include_ensembles=True):
    if records.empty:
        raise ValueError('No pregame historical feature records can be constructed.')
    models, metrics, fold_metrics, test_predictions, calibration_records = {}, [], [], [], []
    dates = sorted(records.game_date.unique())
    for position, group in records.groupby('position'):
        group = group.sort_values('game_date').reset_index(drop=True)
        if len(dates) < 20:
            continue
        # 50% initial training, 20% expanding tuning, 10% calibration, 20% untouched test.
        tuning_start, calibration_start, test_start = [dates[int(frac*len(dates))] for frac in [.5, .7, .8]]
        if len(group[group.game_date < tuning_start]) < 50 or len(group[group.game_date >= test_start]) < 20:
            continue
        x, y = np.stack(group.vector), group.actual.to_numpy(float)
        specs = [dict(method='recent_mean', regularization=0, features=6)]
        specs += [dict(method='ridge', regularization=lam, features=n) for n in sorted({6, x.shape[1]}) for lam in [10., 100.]]
        specs += [dict(method='boosted_trees', leaves=n, features=x.shape[1]) for n in [7, 15]]
        if include_ensembles:
            # Weights and members are fixed before evaluating the final period.
            members = [dict(method='ridge', regularization=100., features=x.shape[1]),
                       dict(method='boosted_trees', leaves=7, features=x.shape[1])]
            specs += [dict(method='ensemble', features=x.shape[1], components=members, weights=[1-w, w])
                      for w in [.25, .5, .75]]
        tuning_dates = [d for d in dates if tuning_start <= d < calibration_start]
        blocks = [list(block) for block in np.array_split(tuning_dates, 3) if len(block)]
        losses = [[] for _ in specs]
        fold_errors = [[] for _ in specs]
        for block_id, block in enumerate(blocks):
            fitted_candidates = {}
            train_mask = (group.game_date < block[0]).to_numpy()
            val_mask = group.game_date.isin(block).to_numpy()
            for i, spec in enumerate(specs):
                if not val_mask.any():
                    continue
                fitted = fit_candidate(spec, x[train_mask], y[train_mask], fitted_candidates)
                errors = y[val_mask]-predict(fitted, x[val_mask])
                losses[i].extend(errors.tolist())
                fold_errors[i].append(errors)
                fold_metrics.append(dict(position=position, candidate=spec['method'], features=spec['features'],
                                         regularization=spec.get('regularization', np.nan), leaves=spec.get('leaves', np.nan),
                                         tree_weight=spec.get('weights', [np.nan, np.nan])[1],
                                         fold=block_id+1, validation_start=str(block[0]), training_last_date=str(group.loc[train_mask, 'game_date'].max()),
                                         rows=len(errors), rmse=float(np.sqrt(np.mean(errors**2))), mae=float(np.mean(np.abs(errors)))))
        # Squared loss targets conditional means, the optimizer's expected-points objective.
        best = min(range(len(specs)), key=lambda i: np.mean(np.square(losses[i])))
        selected = specs[best]
        interval_method = 'equal_tail'
        # Choose interval shape on tuning folds, never on the final test or
        # on the separate residual-calibration period.
        errors_by_fold = fold_errors[best]
        if len(errors_by_fold) >= 2:
            prior_errors = np.concatenate(errors_by_fold[:-1])
            check_errors = errors_by_fold[-1]
            if len(prior_errors) >= 30 and len(check_errors) >= 10:
                interval_losses = {}
                for method in ['equal_tail', 'absolute']:
                    lower, upper = prediction_interval(dict(residuals=prior_errors, interval_method=method), 0.)
                    interval_losses[method] = np.mean(upper-lower + 10*np.maximum(lower-check_errors, 0) + 10*np.maximum(check_errors-upper, 0))
                interval_method = min(interval_losses, key=interval_losses.get)
        before_cal = (group.game_date < calibration_start).to_numpy()
        cal_mask = ((group.game_date >= calibration_start) & (group.game_date < test_start)).to_numpy()
        if cal_mask.sum() < 20:
            continue
        fixed = fit_candidate(selected, x[before_cal], y[before_cal])
        cal_predictions = predict(fixed, x[cal_mask]); residuals = y[cal_mask]-cal_predictions
        bounds = np.quantile(cal_predictions, [1/3, 2/3])
        bins = np.searchsorted(bounds, cal_predictions)
        calibration_rows = group.loc[cal_mask, ['player_id', 'position', 'game_date', 'team', 'opponent', 'game_id']].copy()
        calibration_rows['projection'] = cal_predictions; calibration_rows['residual'] = residuals
        calibration_records.append(calibration_rows)
        fixed['residuals'] = residuals.tolist(); fixed['residual_bounds'] = bounds.tolist()
        fixed['interval_method'] = interval_method
        fixed['residual_bins'] = {i: residuals[bins == i].tolist() for i in range(3) if (bins == i).sum() >= 30}
        # Freeze fitted model and calibrated intervals at test start; final outcomes never select it.
        test_mask = (group.game_date >= test_start).to_numpy()
        preds = predict(fixed, x[test_mask]); actual = y[test_mask]; errors = actual-preds
        base = np.maximum(x[test_mask, 0], 0)
        intervals = np.asarray([prediction_interval(fixed, p) for p in preds])
        coverage = (actual >= intervals[:, 0]) & (actual <= intervals[:, 1])
        widths = intervals[:, 1]-intervals[:, 0]
        # Proper interval score penalizes both width and misses (alpha = .2).
        interval_scores = widths + 10*np.maximum(intervals[:, 0]-actual, 0) + 10*np.maximum(actual-intervals[:, 1], 0)
        legacy_coverage = [np.quantile(residual_sample(fixed, p), .1) <= e <= np.quantile(residual_sample(fixed, p), .9) for p, e in zip(preds, errors)]
        gain = (actual-base)**2-errors**2
        evaluated = group.loc[test_mask, ['player_id', 'position', 'game_date', 'actual']].copy()
        evaluated['projection'] = preds; evaluated['baseline'] = base
        evaluated['interval_lower'] = intervals[:, 0]; evaluated['interval_upper'] = intervals[:, 1]
        evaluated['squared_error_gain'] = gain; test_predictions.append(evaluated)
        # Bootstrap game dates, not individual players, to retain within-date dependence.
        day_gain = evaluated.groupby('game_date').squared_error_gain.mean().to_numpy()
        rng = np.random.default_rng(88)
        bootstrap = rng.choice(day_gain, (500, len(day_gain)), replace=True).mean(axis=1)
        high_volume = x[test_mask, 3] >= np.quantile(x[before_cal, 3], .75)
        metrics.append(dict(position=position, selected_model=selected['method'], selected_features=selected['features'],
                            train_rows=int(before_cal.sum()), calibration_rows=int(cal_mask.sum()), test_rows=int(test_mask.sum()),
                            test_start=str(test_start), test_mae=float(np.abs(errors).mean()), test_rmse=float(np.sqrt(np.mean(errors**2))),
                            baseline_test_mae=float(np.abs(actual-base).mean()), baseline_test_rmse=float(np.sqrt(np.mean((actual-base)**2))),
                            rmse_improvement_pct=float(100*(1-np.sqrt(np.mean(errors**2))/max(np.sqrt(np.mean((actual-base)**2)), 1e-9))),
                            test_bias=float(errors.mean()), interval_80_coverage=float(np.mean(coverage)),
                            interval_80_mean_width=float(widths.mean()), interval_80_score=float(interval_scores.mean()),
                            legacy_interval_80_coverage=float(np.mean(legacy_coverage)),
                            interval_method=interval_method,
                            high_volume_test_rows=int(high_volume.sum()), high_volume_test_rmse=float(np.sqrt(np.mean(errors[high_volume]**2))) if high_volume.any() else np.nan,
                            mse_gain_ci_low=float(np.quantile(bootstrap,.025)), mse_gain_ci_high=float(np.quantile(bootstrap,.975)),
                            evidence='Historical reconstruction, not archived as-published pregame forecasts'))
        deployed = fit_candidate(selected, x, y)
        for field in ['residuals', 'residual_bounds', 'residual_bins', 'interval_method']:
            deployed[field] = fixed[field]
        models[position] = deployed
    if not models:
        raise ValueError('Not enough history for chronological model validation.')
    results = pd.DataFrame(metrics)
    results.attrs['calibration_records'] = pd.concat(calibration_records, ignore_index=True)
    results.attrs['fold_metrics'] = pd.DataFrame(fold_metrics)
    results.attrs['test_predictions'] = pd.concat(test_predictions, ignore_index=True)
    return models, results
