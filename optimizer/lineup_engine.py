"""Slot-based integer optimization for Classic and CPT/FLEX Showdown."""
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp

try:
    from .dk_csv import CLASSIC, SHOWDOWN
    from .scenarios import sample_outcomes
    from .freshness import automatic_stale
except ImportError:
    from dk_csv import CLASSIC, SHOWDOWN
    from scenarios import sample_outcomes
    from freshness import automatic_stale


def eligible_pool(data, now=None):
    now = now or datetime.now(timezone.utc)
    # Forecast audit tables belong to the UI, not per-row solver operations.
    data = pd.DataFrame(data).copy()
    if 'exclusion_reason' not in data:
        data['exclusion_reason'] = ''
    for i, row in data.iterrows():
        reason = automatic_stale(row, now)
        if reason and not data.loc[i, 'exclusion_reason']:
            data.loc[i, 'exclusion_reason'] = reason
    locked = pd.to_datetime(data.kickoff, utc=True) <= now
    data.loc[locked, 'exclusion_reason'] = 'Game has started; new lineups cannot add this player'
    invalid = ~np.isfinite(pd.to_numeric(data.projection, errors='coerce')) | (data.projection <= 0)
    data.loc[invalid & data.exclusion_reason.eq(''), 'exclusion_reason'] = 'Missing positive projection'
    return data[data.exclusion_reason.eq('')].reset_index(drop=True), data[~data.exclusion_reason.eq('')].copy()


def solve_lineup(data, game_type='Classic', objective=None, previous=(), min_unique=1,
                 locked_keys=(), forbidden_keys=(), stack=False, time_limit=20):
    data = pd.DataFrame(data).reset_index(drop=True)
    slots = CLASSIC if game_type == 'Classic' else SHOWDOWN
    if game_type not in {'Classic', 'Showdown'}:
        raise ValueError('Only NFL Classic and Showdown are supported.')
    if data.empty:
        raise ValueError('No eligible players with current projections.')
    if not 1 <= min_unique <= len(slots):
        raise ValueError('Invalid minimum unique player count.')
    if not set(locked_keys).issubset(set(data.player_key)):
        raise ValueError('A locked player is unavailable or lacks a projection.')
    objective = np.asarray(data.projection if objective is None else objective, dtype=float)
    if len(objective) != len(data) or not np.isfinite(objective).all():
        raise ValueError('Every optimization coefficient must be finite.')
    problem = pulp.LpProblem('DK_NFL_Slots', pulp.LpMaximize)
    variables = {(i, slot): pulp.LpVariable(f'p_{i}_s_{slot}', cat=pulp.LpBinary)
                 for i, player in data.iterrows() for slot, label in enumerate(slots)
                 if label in player.eligible_slots}
    if any(not any(s == slot for _, s in variables) for slot in range(len(slots))):
        raise ValueError('Player pool does not cover every required roster slot.')
    problem += pulp.lpSum(objective[i] * (1.5 if slots[s] == 'CPT' else 1) * v for (i, s), v in variables.items())
    # CPT rows carry their actual higher salary in the original DK export.
    problem += pulp.lpSum(int(data.loc[i, 'salary']) * v for (i, _), v in variables.items()) <= 50000
    for slot in range(len(slots)):
        problem += pulp.lpSum(v for (_, s), v in variables.items() if s == slot) == 1
    # Build athlete sums once instead of repeatedly looking up every variable row.
    athlete_variables = {}
    by_game = {}
    by_team = {}
    keys = data.player_key.to_numpy()
    games_for_row = data.game_id.to_numpy()
    teams_for_row = data.team.to_numpy()
    for (i, slot), variable in variables.items():
        athlete_variables.setdefault(keys[i], []).append(variable)
        by_game.setdefault(games_for_row[i], []).append(variable)
        by_team.setdefault(teams_for_row[i], []).append(variable)
    by_player = {}
    for key in data.player_key.unique():
        selection = pulp.lpSum(athlete_variables.get(key, []))
        by_player[key] = selection
        problem += selection <= 1
        if key in locked_keys:
            problem += selection == 1
        if key in forbidden_keys:
            problem += selection == 0
    if game_type == 'Classic':
        games = data.game_id.unique()
        if len(games) < 2:
            raise ValueError('NFL Classic requires players from at least two NFL games.')
        for game in games:
            problem += pulp.lpSum(by_game.get(game, [])) <= 8
        if stack:
            for i, qb in data[data.position.eq('QB')].iterrows():
                catches = [by_player[p.player_key] for _, p in data.iterrows()
                           if p.team == qb.team and p.position in {'WR', 'TE'}]
                problem += pulp.lpSum(catches) >= by_player[qb.player_key]
    else:
        if data.game_id.nunique() != 1 or data.team.nunique() != 2:
            raise ValueError('Showdown requires one game and players from both teams.')
        for team in data.team.unique():
            problem += pulp.lpSum(by_team.get(team, [])) <= 5
    for keys in previous:
        problem += pulp.lpSum(by_player[key] for key in keys if key in by_player) <= len(slots) - min_unique
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=1))
    if status != pulp.LpStatusOptimal or problem.sol_status != pulp.LpSolutionOptimal:
        raise ValueError(f'No proven optimal lineup ({pulp.LpStatus[status]}). Check projections, salary, locks, and diversity constraints.')
    selected = []
    for slot, label in enumerate(slots):
        indices = [i for (i, s), v in variables.items() if s == slot and (pulp.value(v) or 0) > 0.5]
        if len(indices) != 1:
            raise ValueError('Solver returned an incomplete roster.')
        player = data.loc[indices[0]].to_dict()
        player['slot'] = label
        player['projected_points'] = player['projection'] * (1.5 if label == 'CPT' else 1)
        selected.append(player)
    lineup = pd.DataFrame(selected)
    validate_lineup(lineup, game_type)
    return lineup


def validate_lineup(lineup, game_type):
    slots = CLASSIC if game_type == 'Classic' else SHOWDOWN
    if list(lineup.slot) != slots or lineup.player_key.nunique() != len(slots):
        raise ValueError('Invalid roster slots or duplicate athlete.')
    if lineup.salary.sum() > 50000:
        raise ValueError('Salary cap exceeded.')
    for _, player in lineup.iterrows():
        if player.slot not in player.eligible_slots:
            raise ValueError('Player selected in an ineligible slot.')
    if game_type == 'Classic' and lineup.game_id.nunique() < 2:
        raise ValueError('Classic roster must span at least two games.')
    if game_type == 'Showdown' and (lineup.game_id.nunique() != 1 or lineup.team.nunique() != 2):
        raise ValueError('Showdown roster must include both teams from one game.')


def build_portfolio(data, game_type='Classic', count=3, min_unique=2, max_exposure=1.0,
                    strategy='Cash', locked_keys=(), forbidden_keys=(), calibration=None, seed=88):
    if not 1 <= count <= 20:
        raise ValueError('Generate between 1 and 20 lineups per run.')
    if not 0 < max_exposure <= 1:
        raise ValueError('Exposure must be greater than zero and at most 100%.')
    limit = int(np.floor(count * max_exposure))
    if limit < 1:
        raise ValueError(f'{count} lineups require at least {100 / count:.1f}% exposure for any athlete. Raise exposure or lineup count.')
    if set(locked_keys).intersection(forbidden_keys):
        raise ValueError('An athlete cannot be both locked and excluded.')
    if locked_keys and limit < count:
        raise ValueError('Locked players need 100% maximum exposure.')
    if strategy == 'GPP' and calibration is not None:
        return scenario_portfolio(data, game_type, count, min_unique, limit,
                                  locked_keys, forbidden_keys, calibration, seed)
    rng = np.random.default_rng(seed)
    lineups, previous, exposures = [], [], Counter()
    for number in range(count):
        objective = data.projection.to_numpy(dtype=float).copy()
        if strategy == 'GPP':
            if calibration is not None:
                # Scenario perturbations use actual historical forecast residuals.
                # Means remain visible separately from this candidate objective.
                for i, row in data.reset_index(drop=True).iterrows():
                    objective[i] += rng.choice(calibration[row.position]['residuals'])
            elif 'projection_ceiling' in data and np.isfinite(data.projection_ceiling).all():
                objective = data.projection_ceiling.to_numpy(dtype=float)
            # Without defensible uncertainty, optimize means with diversity rather than invent it.
        banned = set(forbidden_keys) | {key for key, n in exposures.items() if n >= limit}
        try:
            lineup = solve_lineup(data, game_type, objective, previous, min_unique,
                                  locked_keys, banned, stack=strategy == 'GPP')
        except ValueError as exc:
            if not lineups:
                raise
            if max(exposures.values()) / len(lineups) > max_exposure:
                raise ValueError('A partial portfolio would exceed the requested exposure percentage. Raise exposure or reduce lineup count.') from exc
            return lineups, f'Only {len(lineups)} of {count} lineups are feasible: {exc}'
        keys = set(lineup.player_key)
        previous.append(keys)
        exposures.update(keys)
        lineups.append(lineup)
    return lineups, None


def scenario_portfolio(data, game_type, count, min_unique, exposure_limit, locks, bans, calibration, seed):
    """Generate candidate solutions, then maximize their sampled expected best score.

    This approximates a diversification objective; it is not a payout/field model.
    Candidate and evaluation draws are independent to reduce selection overfitting.
    """
    data = pd.DataFrame(data).reset_index(drop=True)
    if not set(data.position).issubset(calibration):
        raise ValueError('Calibration must cover every eligible position.')
    rng = np.random.default_rng(seed)
    candidates, keys_seen = [], set()
    candidate_outcomes = sample_outcomes(data, calibration, max(12, count * 3), rng)
    for iteration in range(max(12, count * 3)):
        objective = data.projection.to_numpy(dtype=float).copy()
        if iteration:
            objective = np.array([candidate_outcomes[row.player_key][iteration] for _, row in data.iterrows()])
        lineup = solve_lineup(data, game_type, objective, locked_keys=locks, forbidden_keys=bans, stack=True)
        # Reordering equivalent RB/WR/FLEX slots is not a new candidate.
        captain = tuple(lineup.loc[lineup.slot.eq('CPT'),'player_key'])
        identity = (captain,tuple(sorted(lineup.player_key)))
        if identity not in keys_seen:
            keys_seen.add(identity)
            candidates.append(lineup)
    # Reuse the same underlying athlete outcome for CPT and FLEX rows.
    outcomes = sample_outcomes(data, calibration, 2000, rng)
    candidate_scores = [sum(outcomes[row.player_key] * (1.5 if row.slot == 'CPT' else 1)
                            for _, row in lineup.iterrows()) for lineup in candidates]
    selected, selected_keys, appearances = [], [], Counter()
    best_scores = np.full(2000, -np.inf)
    while len(selected) < count:
        valid = []
        for index, lineup in enumerate(candidates):
            keys = set(lineup.player_key)
            if any(appearances[key] >= exposure_limit for key in keys):
                continue
            if any(len(keys - prior) < min_unique for prior in selected_keys):
                continue
            gain = np.maximum(best_scores, candidate_scores[index]).mean()
            valid.append((gain, index))
        if not valid:
            break
        _, index = max(valid, key=lambda item: item[0])
        selected.append(candidates[index])
        keys = set(candidates[index].player_key)
        selected_keys.append(keys)
        appearances.update(keys)
        best_scores = np.maximum(best_scores, candidate_scores[index])
    if not selected:
        raise ValueError('No calibrated candidate satisfies the requested exposure constraints.')
    if max(appearances.values()) / len(selected) > exposure_limit / count:
        raise ValueError('Not enough candidates to meet the requested exposure percentage. Raise exposure or reduce lineup count.')
    message = None if len(selected) == count else f'Only {len(selected)} diversified candidates meet the requested constraints. Lower diversity or raise exposure to allow more.'
    return selected, message
