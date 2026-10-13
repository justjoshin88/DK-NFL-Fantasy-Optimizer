# DK NFL Fantasy Optimizer

A Streamlit app for native DraftKings NFL player pools, automatic pregame
projection retrieval, legal lineups, and ID-only roster CSV exports.

## Run

```sh
python -m pip install -r requirements.txt
python -m streamlit run optimizer/optimizer.py
python -m unittest discover -s tests -v
```

Streamlit Community Cloud: repository `justjoshin88/DK-NFL-Fantasy-Optimizer`,
branch `main`, entrypoint `optimizer/optimizer.py`. Root requirements declare
all Python dependencies. Keep `.streamlit/secrets.toml` outside git.

## Workflow

1. Enter a DraftKings contest URL/ID and select NFL Classic or Showdown.
2. Upload the exact contest player pool CSV. Optionally upload its blank roster
   template. Do not upload a roster-only file as the player pool.
3. Free automatic forecasts retrieve for the pool's game dates. Review
   matching coverage, exclusions, source timestamps, injury flags and ownership
   availability. Unknown values are not replaced by averages or invented data.
4. Set lineup count (1–20), diversity, exposure, athlete locks/exclusions and
   Cash/GPP strategy. Generate and inspect lineups.
5. Download `DK_rosters.csv`. Repeated position columns, numeric IDs, and any
   additional supplied template columns are preserved. This is a roster-upload
   file, not an entry-edit template. No entries are placed or paid for.

Classic requires at least two NFL games. Showdown requires its own CPT/FLEX
pool, both teams, six distinct athletes, and a $50,000 cap. CPT scoring is 1.5x
base projection; the CPT salary and ID already encoded by DraftKings are used
unchanged. Started games are excluded; existing-entry late swaps are not supported.

## Data providers

Default **Free automatic** needs no API key, paid scraper or subscription. It
uses exact-game public forecasts when available, then fills missing coverage with
**our own model estimates** from [nflverse](https://nflreadr.nflverse.com/).
Downloads include current rosters, timestamped depth charts, exact-week injury
reports, schedules and up to four seasons of player/team box scores. History and fitted models cache for one hour and are shared by NFL season across
player pools. Rosters, injuries, depth charts, schedules and forecast pages cache
for five minutes. Cache entries are bounded to limit Community Cloud memory.
Refresh projections updates current status/pages without discarding fitted models.
Reload NFL history and retrain also downloads box scores and rebuilds validation.
Every historical row contributes to the model cache fingerprint, preventing changes
outside a sampled DataFrame hash from being missed. A first load can take a few minutes; public file
downloads retry transient network/server failures. Later refreshes reuse fitted
models until inputs change. Raw player stat columns, unused intermediate history columns and old depth snapshots
are discarded; player groups and matchup context are cached to control memory use on Streamlit Community Cloud.

The model reconstructs DraftKings offense, kicker and DST points, including
reception scoring and yardage bonuses. DST reconstruction excludes opposing defensive TD/safety points from allowed
points, counts conversion returns and extra points, and uses team boxes/scoreboards; rare scoring attribution may differ from official contest totals.
Player features use at most six prior games: exponentially weighted points (0.7
decay), carries, targets, passing/rushing/receiving yards, touchdowns, recent
volume, trends and team workload shares. Matchup features use only prior completed
games for team offense, points allowed by position, rest and observed home venue
roof. Target-game final weather, closing odds and roof decisions are excluded
because their pregame availability is not archived. Historical timestamps are
explicitly converted to nanoseconds before cutoff comparisons across pandas versions.

The oldest 50% of game dates provide initial training. Three expanding validation
blocks over the next 20% compare the recency baseline, ridge (6 or 22 features,
regularization 10/100) and shallow gradient boosted trees (7/15 leaves). Selection
uses RMSE, which targets the conditional mean used by expected-points optimization.
A separate 10% calibrates score-dependent error intervals and correlations; the
final 20% evaluates frozen fitted models and intervals. The selected model then
refits on available past outcomes for deployment. Test predictions and validation
comparisons can be downloaded. Metrics include baseline errors, high-volume player
errors, bias, interval coverage and game-date bootstrap uncertainty in error gains.

These are **historical reconstructions**, not saved as-published pregame forecasts.
The underlying boxes may contain later stat corrections, and players without a
statistical appearance are incompletely observed. Neither current injury/depth
filters nor cold-start exclusions have been proven to improve live contest ROI.
Improvement on a limited historical test does not establish superior contest profit.

Identity requires a unique roster ID, exact normalized name/team/position,
matching scheduled opponent/date/kickoff and active roster status. Generational
name suffixes are normalized; fuzzy nicknames are not guessed. Players need at
least three observed games and an appearance in the previous 21 days. Unsupported
or unmatched players remain excluded. Estimates are conditional on playing;
roster active status does not verify game-day injuries, starting role or workload.
Automatic snapshots expire after 15 minutes; published updates also must remain
within 24 hours, and fitted models within two hours. Browser-session data are
rechecked on use, and the download section rechecks expiry/game locks every 30
seconds while the page is active. This clock check does not fetch new injury news. A current roster publisher update must be verified
within 36 hours for own estimates. Recent workload from a different team is flagged
for role review. These checks improve input reliability, not demonstrated live ROI.
Published unavailable flags override the model. Fresh exact-week injury reports
also exclude Out/Doubtful/inactive athletes and cannot be overwritten by a positive
public forecast. A fresh complete depth snapshot excludes backup QB/K rows.
Questionable reports remain visible without an arbitrary point deduction. Old,
future-dated, wrong-week or absent reports do not verify health or participation.
Depth chart ranks do not guarantee final starters. Ownership remains unknown.
DraftKings `AvgPointsPerGame` is never substituted as a current forecast.

Data attribution: [nflverse datasets](https://github.com/nflverse/nflverse-data),
[CC BY 4.0](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md).
This app transforms those data into scores, features and model estimates.
Schedules: [nfldata](https://github.com/nflverse/nfldata/blob/master/data/games.csv).
Public source availability and schema changes can interrupt retrieval; the app
reports failure without inventing replacement records.

Public source: [Daily Fantasy Fuel](https://www.dailyfantasyfuel.com/nfl/projections/draftkings/).
Use exact game-date pages and match normalized name + team + position + opponent
+ date (DST by team). Source timestamps must be at most 24 hours old. The source
may publish future-slate forecasts only closer to game day, may omit players or
kickers, and can change its HTML format. In the Public forecasts-only mode, no positive current forecast means no
eligible player. The app lists missing coverage instead of using historical
`AvgPointsPerGame` from a DraftKings salary file.

Optional [SportsDataIO](https://sportsdata.io/developers/api-documentation/nfl)
real-data subscription: in Streamlit Secrets add:

```toml
SPORTSDATAIO_API_KEY = "your licensed key"
SPORTSDATAIO_PRODUCTION = true
```

Free trial data is scrambled and is not appropriate for real lineup selection.
The production flag is an operator attestation; upstream test/scramble indicators
are rejected where present. This adapter requires authorized schedule and
projection endpoints. It has fixture tests but was not live-tested with a
licensed key. Source timestamps are separate from retrieval timestamps. Records without a
verified timezone-aware update within 24 hours are excluded rather than treated
as fresh; missing or naive timestamps may make this feed ineligible. No key is shipped or purchased.

Projection-upload alternative: CSV with `player_id` (or `ID`) and `projection`.
IDs must belong to the uploaded pool. Optional: `ownership` (0–100 percent),
`injury_status`, `projection_ceiling`, `projection_std`. For Showdown, enter
base FLEX projections even on CPT ID rows. Uploaded freshness is user supplied,
not independently verified.

Contest metadata/ID checks use read-only public DraftKings endpoints. These may
be unavailable or change without notice. If lookup fails, contest membership is
explicitly unverified and the pool is used as supplied. A demonstrated ID mismatch
blocks the workflow. Contest lookup does not authorize account access or entries.

## Saved forecast scoring

Download forecast audit **before kickoff**. After games are final, open **Check
saved forecasts against actual results** and upload that CSV. The scorer matches
historical GSIS IDs (DST by canonical team), exact opponent/date/kickoff and final
NFL box scores, using the same DraftKings scoring rules. It reports observed-game
MAE/RMSE/bias and exports a calibration-ready `scored_forecast_history.csv`.

Pregame timestamps require timezones and must precede kickoff. Missing/after-kickoff
records, excluded rows, wrong identities, unfinished games and absent player boxes
remain unscored. CPT/FLEX rows are deduplicated as one athlete's **base** projection;
conflicting base values stop scoring. Timestamps in uploaded files are user editable,
so this does not independently prove as-published provenance. Box-derived final
points may differ in rare scoring attribution from official contest scores.
This is observed forecast evaluation, not a contest-payout or ROI evaluation.

Uploaded projection files retain the same import timestamp on reruns so built
rosters and downloads persist. Changing forecast contents invalidates old lineups;
changing lineup count/strategy keeps manually retrieved forecast data available.

## Model and evidence

Cash solves a mixed-integer slot assignment maximizing summed projected points
under exact salary, slot, athlete and game constraints. Every returned solution
must have proven optimal solver status for its given objective. Later lineups
also obey diversity and exposure constraints; the portfolio is a sequential
heuristic, not a jointly proven optimum.

GPP uses QB/pass-catcher stacks in Classic and athlete diversity. When every
eligible forecast comes from our free model, its separate calibration errors
provide automatic scenarios without another CSV upload. Same-team and opposing-team
rank-error correlations are estimated by projected role (QB1, RB1/2, WR1/2/3,
TE1, K1, DST1) with at least 30 observed pairs. Pair correlations shrink toward
independence using 100 prior observations. A Gaussian copula maps correlated
uniform draws to empirical residual quantiles; each matrix is repaired to be
positive definite. Unknown roles/pairs stay independent. CPT/FLEX rows share one
athlete outcome, and candidate/evaluation draws are separate. This approximation
omits unexpected role changes, correlated injury events and actual opponent fields.

Mixed expert/model coverage does not silently reuse our model's residuals for
expert predictions. A user archive remains an independent-residual alternative. If neither source of
empirical errors is usable, GPP uses fully supplied ceilings or mean projections.
With usable errors, it creates scenario-optimized candidates and selects them
greedily by sampled expected best portfolio score. This is a diversification
model, not a complete payout or opponent-field model.

Historical archive: `game_date,position,projection,actual`, saved pregame
forecasts and final DraftKings points; optional `forecast_at` (must precede game
start) and `player_id` (for duplicate checks). At least eight distinct dates,
30 training and 10 holdout rows per position. Train on the oldest 75% of dates,
validate on the newest 25%; report MAE, RMSE, bias and 80% interval coverage.
When `forecast_at` is absent, pregame provenance is user attested, not verified.
Position errors are resampled independently. This omits game correlations,
changing roles and news-driven distribution shifts. Do not interpret score
quantiles as calibrated odds of winning a contest.

No algorithm can guarantee a winning lineup. Calling this the best or estimating
profit needs broader comparative out-of-sample evidence, calibrated correlated
outcomes, ownership/duplicate-lineup modeling, field simulation, contest payout
structures and entry fees. Those are not inferred from a salary CSV or silently
manufactured by this app. Provider availability and input quality limit results.

Primary rules and methodological context:
- https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html
- https://nflreadr.nflverse.com/reference/load_depth_charts.html
- https://nflreadr.nflverse.com/reference/load_injuries.html
- Mahoney and Paniak, Method and Validation for Optimal Lineup Creation for Daily
  Fantasy Football Using Machine Learning and Linear Programming,
  https://arxiv.org/abs/2309.15253
  (supports comparative forecasting/optimization validation; does not establish
  that this app or a neural network is a universally superior method).
- https://www.draftkings.com/help/rules/1
- https://www.draftkings.com/help/rules/1/96
- Haugh and Singal, How to Play Fantasy Sports Strategically (and Win),
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393127

Latest measured comparison and timing: [model validation snapshot](docs/model-validation.md).

## Tests

Includes exhaustive small-pool comparison for objective optimality, BOM and exact
repeated roster headers, ID preservation, Classic two-game rules, CPT salary and
scoring, duplicate-athlete exclusion, exposure rounding, exact source dates and
teams, injury/lock exclusions, stale sources, malformed projections, wrong
contest pools, chronological holdout, Streamlit upload/build/rerun behavior, matchup leakage, timestamp units, depth snapshot replacement, exact-week injury freshness, separate tuning/calibration/test boundaries, correlated CPT outcome identity, cache reuse, freshness expiry, stable upload reruns, current schedule checks with cached models, and saved pregame snapshot scoring.
Synthetic fixtures exist only in tests; the app ships no fake live projections.
