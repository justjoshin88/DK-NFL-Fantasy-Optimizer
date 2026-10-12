# DK NFL Fantasy Optimizer

Streamlit interface for the DraftKings NFL Classic optimizer.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run optimizer/optimizer.py
```

## Streamlit Community Cloud

Use repository `justjoshin88/DK-NFL-Fantasy-Optimizer`, branch `main`, and
entry point `optimizer/optimizer.py`. Dependencies are declared in the root
`requirements.txt`. If the deployment retains the old environment after an
update, reboot the app from its management menu.

Upload CSV or JSON records with `player_name`, `position`, `salary`,
`projection`, `team`, and `opponent`. Positions: QB, RB, WR, TE, DST.
Optional numeric fields include `projection_floor`, `projection_ceiling`,
`ownership` (0–100 percent), and `projection_std`.
DraftKings salaries alone do not provide forecasts: supply your own projections.
No live data feed or historical calibration is included.

Cash uses floor projections when available. GPP uses ceiling projections when
available (or supplied optimal percentages), requires a QB/pass catcher stack,
and caps total ownership at 130% if ownership is supplied. These are optimization
heuristics, not calibrated probabilities of winning a contest.

The app supports NFL Classic, not Showdown. Exported CSVs are lineup summaries,
not DraftKings bulk entry templates.

## Verify

```sh
python -m unittest discover -s tests -v
```
