"""Temporal holdout diagnostics, not a claim of verified profitability."""
import numpy as np
import pandas as pd
try:
    from .scenarios import sample_outcomes
except ImportError:
    from scenarios import sample_outcomes
from datetime import datetime, timezone


def calibrate(history):
    required = {'game_date', 'position', 'projection', 'actual'}
    if not required.issubset(history):
        raise ValueError('History requires game_date, position, projection, actual. Use saved pregame forecasts and final DK scores.')
    data = history.copy()
    data['game_date'] = pd.to_datetime(data.game_date, errors='raise', utc=True)
    if data.game_date.isna().any() or (data.game_date >= datetime.now(timezone.utc)).any():
        raise ValueError('History must contain completed past game dates.')
    if 'forecast_at' in data:
        forecast_at = pd.to_datetime(data.forecast_at, errors='raise', utc=True)
        if forecast_at.isna().any() or (forecast_at >= data.game_date).any():
            raise ValueError('forecast_at must precede its game_date to avoid leakage.')
    if {'player_id', 'game_date'}.issubset(data) and data.duplicated(['player_id', 'game_date']).any():
        raise ValueError('Duplicate player/game rows would distort calibration.')
    for col in ['projection', 'actual']:
        data[col] = pd.to_numeric(data[col], errors='raise')
        if not np.isfinite(data[col]).all():
            raise ValueError('Historical projections and actuals must be finite.')
    dates = sorted(data.game_date.unique())
    if len(dates) < 8:
        raise ValueError('Calibration needs at least eight distinct game dates. Hold out the most recent 25%.')
    cutoff = dates[max(1, int(len(dates) * .75))]
    train, test = data[data.game_date < cutoff], data[data.game_date >= cutoff]
    models, metrics = {}, []
    for position in data.position.unique():
        training, testing = train[train.position.eq(position)], test[test.position.eq(position)]
        if len(training) < 30 or len(testing) < 10:
            raise ValueError(f'{position} needs at least 30 training and 10 holdout observations.')
        errors = (training.actual - training.projection).to_numpy()
        holdout = (testing.actual - testing.projection).to_numpy()
        models[position] = {'residuals': errors.tolist()}
        lower, upper = np.quantile(errors, [.1, .9])
        metrics.append(dict(position=position, train_rows=len(training), holdout_rows=len(testing),
                            holdout_mae=float(np.abs(holdout).mean()), holdout_rmse=float(np.sqrt((holdout ** 2).mean())),
                            holdout_bias=float(holdout.mean()), interval_80_coverage=float(((holdout >= lower) & (holdout <= upper)).mean()),
                            training_cutoff=str(cutoff)))
    return models, pd.DataFrame(metrics)


def evaluate_portfolio(lineups, calibration, n_sims=2000, seed=88):
    """Conditional residual/correlation score distribution, never a win probability."""
    players = pd.concat(lineups).drop_duplicates('player_key').reset_index(drop=True)
    if not set(players.position).issubset(calibration):
        raise ValueError('Calibration does not cover every selected position.')
    rng = np.random.default_rng(seed)
    outcomes = sample_outcomes(players, calibration, n_sims, rng)
    results = []
    for i, lineup in enumerate(lineups):
        points = sum(outcomes[row.player_key] * (1.5 if row.slot == 'CPT' else 1) for _, row in lineup.iterrows())
        results.append(dict(lineup=i + 1, model_mean=float(points.mean()), model_p10=float(np.quantile(points, .1)),
                            model_p90=float(np.quantile(points, .9))))
    return pd.DataFrame(results)
