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
Downloads include current rosters, schedules and the previous/current season's
player and team box scores. Downloads cache for one hour; projections for five
minutes. A first load may take longer while those public files download.

The model reconstructs DraftKings offense, kicker and DST points, including
reception scoring and yardage bonuses. DST reconstruction uses team boxes and
scoreboards; rare scoring attribution may differ from official contest totals.
Features use at most six previous games: exponentially weighted score (0.7
decay), last score, variability, recent volume, trend and time since playing.
For each position, the oldest 60% of game dates train a recency baseline and
ridge candidates (regularization 1, 10, 100). The next 20% select by MAE; the
last 20% report untouched test MAE, RMSE, bias and interval coverage. The selected
method refits on available history for future forecasts. Intervals use validation
residuals. This is a tested free baseline, not evidence of superior contest profit.

Identity requires a unique roster ID, exact normalized name/team/position,
matching scheduled opponent/date/kickoff and active roster status. Generational
name suffixes are normalized; fuzzy nicknames are not guessed. Players need at
least three observed games and an appearance in the previous 21 days. Unsupported
or unmatched players remain excluded. Estimates are conditional on playing;
roster active status does not verify game-day injuries, starting role or workload.
Published unavailable flags override the model. Ownership remains unknown.
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

## Model and evidence

Cash solves a mixed-integer slot assignment maximizing summed projected points
under exact salary, slot, athlete and game constraints. Every returned solution
must have proven optimal solver status for its given objective. Later lineups
also obey diversity and exposure constraints; the portfolio is a sequential
heuristic, not a jointly proven optimum.

GPP uses QB/pass-catcher stacks in Classic and athlete diversity. Without a
historical archive, it uses fully supplied ceilings or mean projections; no
arbitrary volatility or win probability is invented. With a supported archive,
it creates scenario-optimized candidates using empirical position forecast errors
and selects candidates greedily by sampled expected best portfolio score. Candidate
and evaluation draws are separate. This is a diversification model, not a complete
payout or opponent-field model.

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
- https://www.draftkings.com/help/rules/1
- https://www.draftkings.com/help/rules/1/96
- Haugh and Singal, How to Play Fantasy Sports Strategically (and Win),
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393127

## Tests

Includes exhaustive small-pool comparison for objective optimality, BOM and exact
repeated roster headers, ID preservation, Classic two-game rules, CPT salary and
scoring, duplicate-athlete exclusion, exposure rounding, exact source dates and
teams, injury/lock exclusions, stale sources, malformed projections, wrong
contest pools, chronological holdout, and Streamlit upload/build/rerun behavior.
Synthetic fixtures exist only in tests; the app ships no fake live projections.
