"""Downloadable forecast evidence, without treating data coverage as accuracy."""
import hashlib
import json

import pandas as pd

try:
    from .forecast_features import FEATURES
except ImportError:
    from forecast_features import FEATURES


def evidence_report(data, available, *, pool_sha256, contest, contest_verified, release):
    snapshot = data.to_csv(index=False)
    metrics = data.attrs.get('free_model_metrics', pd.DataFrame())
    validation = json.loads(metrics.to_json(orient='records'))
    sources = []
    source_labels = data.get('source', pd.Series('Unknown', index=data.index))
    for source, group in data.groupby(source_labels, dropna=False):
        sources.append(dict(source=str(source), pool_rows=len(group),
                            urls=sorted(group.source_url.dropna().astype(str).unique()) if 'source_url' in group else [],
                            published_at=sorted(group.source_updated.dropna().astype(str).unique()) if 'source_updated' in group else [],
                            fetched_at=sorted(group.fetched_at.dropna().astype(str).unique()) if 'fetched_at' in group else []))
    eligible_ids = set(available.player_id)
    coverage = []
    for position, group in data.groupby('position'):
        usable = group[group.player_id.isin(eligible_ids)].drop_duplicates('player_key')
        reported = usable.get('injury_report_status', pd.Series('Unknown', index=usable.index))
        official = usable.get('official_status', pd.Series('Unknown', index=usable.index))
        coverage.append(dict(position=position, pool_athletes=group.player_key.nunique(),
                             eligible_athletes=usable.player_key.nunique(),
                             injury_report_unknown=int((reported.isna() | reported.eq('Unknown')).sum()),
                             official_report_listed=int(official.notna().sum()-official.eq('Unknown').sum()),
                             official_negative_designations=group.loc[group.get('official_status', pd.Series('', index=group.index)).isin(['OUT', 'DOUBTFUL', 'INACTIVE', 'IR', 'SUSPENDED']), 'player_key'].nunique(),
                             ownership_known=int(usable.ownership.notna().sum()) if 'ownership' in usable else 0))
    report = dict(schema_version=1, app_release=release,
                  generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
                  pool_sha256=pool_sha256,
                  forecast_csv_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
                  contest=dict(input=contest, membership_verified=contest_verified),
                  sources=sources, upstream=data.attrs.get('upstream_evidence', {}),
                  models=data.attrs.get('model_specifications', {}), features=FEATURES,
                  coverage=coverage, historical_validation=validation,
                  equations=dict(error='actual - projection',
                                 mae='mean(abs(actual - projection))',
                                 rmse='sqrt(mean((actual - projection)^2))',
                                 bias='mean(actual - projection)',
                                 interval_80_coverage='mean(q10 <= actual - projection <= q90)',
                                 recency='sum(0.7^(n-1-j) * score[j]) / sum(0.7^(n-1-j)), n <= 6',
                                 ridge='beta = pinv(Z.T Z + lambda P) Z.T y; P[0,0] = 0; standardized features',
                                 ensemble='weighted mean of ridge and boosted-tree predictions; weights selected on chronological validation folds',
                                 interval='shape selected using tuning-fold interval score; equal-tail residual quantiles or radius at order ceil((n+1)*0.8) of absolute calibration errors',
                                 cash='maximize sum(projection[i] * selected[i]) subject to roster, salary and eligibility constraints',
                                 showdown='CPT score = 1.5 * base projection; use uploaded CPT salary'),
                  limitations=['Coverage is not forecast accuracy.',
                               'The historical evaluation slice has been used in prior development; future archived live forecasts are needed for new independent evidence.',
                               'Historical reconstruction may include corrected statistics; it is not archived live forecast performance.',
                               'Injury reports and depth charts do not establish final participation.',
                               'Official report observation time is not publication time; blank designations and missing reports do not mean healthy.',
                               'Model scores are conditional on playing; missing box scores are not assumed zero.',
                               'GPP scenarios, stacks and diversity do not establish win probability or profit.'])
    return json.dumps(report, indent=2, allow_nan=False), pd.DataFrame(coverage)
