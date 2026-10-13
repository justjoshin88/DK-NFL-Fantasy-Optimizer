# Forecast validation snapshot

Measured 2026-10-13 with public nflverse 2023–2026 data through NFL Week 5.
The six positions contain 4,402 final test observations; the test period begins
2025-11-23. The baseline uses the same prior six games and 0.7 decay as the candidates.
Selection uses three earlier expanding validation blocks, and a separate earlier
calibration period supplies the intervals. See README for the exact procedure.

| Position | Selected model | Test observations | RMSE | Baseline RMSE | Change | 80% interval coverage |
|---|---|---:|---:|---:|---:|---:|
| DST | ridge | 356 | 5.19 | 5.63 | 7.8% lower | 79.2% |
| K | ridge | 339 | 4.88 | 5.15 | 5.1% lower | 70.8% |
| QB | boosted_trees | 404 | 8.34 | 8.94 | 6.7% lower | 79.5% |
| RB | ridge | 955 | 6.59 | 6.89 | 4.3% lower | 81.4% |
| TE | ridge | 810 | 5.45 | 5.75 | 5.2% lower | 80.1% |
| WR | ridge | 1538 | 6.49 | 6.78 | 4.3% lower | 81.2% |

QB/RB/WR/TE mean squared error gains have positive 95% game-date bootstrap bounds
on this sample. DST and K bounds cross zero, so their apparent improvement is
less certain. K interval coverage is about 71%, below the nominal 80%; intervals
must not be interpreted as reliable betting or contest-winning probabilities.

The test reconstructs prior features from currently available historical boxes.
It is not an as-published pregame projection archive, a complete participation
model, or a test of injury filters, lineup payouts or ROI. Outcomes missing from
stat boxes, corrected stats, new roles and out-of-sample news limit interpretation.

Fresh-download integration with Josh's supplied 727-row pool yielded 356 eligible
model rows when the public expert pages were unavailable. This count includes
conditional estimates whose exact-week injury status is unknown. Thirty-one backup
QB/K rows were excluded by current depth charts. Three legal Cash and three GPP
lineups exported using the exact supplied roster header, IDs and salaries.

Measured cold download + training + forecast + both three-lineup builds: 140 seconds.
Cash solving: 3.2 seconds. GPP candidate generation/selection: 21 seconds.
These are single-run timings, not performance guarantees. History and training
cache for one hour; unchanged forecasts cache for five minutes.

Reproduction: run `load_datasets(pool)`, `prepare_model(datasets)` and
`fetch_free(pool, datasets, artifacts=artifacts)` from `optimizer.free_model`.
Download test predictions, validation folds and forecasts through the app for
independent inspection. Dataset revisions and later dates can change these metrics.

Data: [nflverse](https://github.com/nflverse/nflverse-data),
[CC BY 4.0](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md).
Data are transformed into scoring totals, features and forecasts.
