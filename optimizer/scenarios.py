"""Empirical marginal errors and a shrunk game/role Gaussian copula.

A score-diversification model, not a contest-field or expected-payout model.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
try:
    from .forecast_validation import residual_sample
except ImportError:
    from forecast_validation import residual_sample

ROLE_LIMITS = {'QB':1, 'RB':2, 'WR':3, 'TE':1, 'K':1, 'DST':1}


def add_roles(frame):
    data = frame.sort_values(['projection', 'player_id'], ascending=[False, True]).copy()
    rank = data.groupby(['game_id', 'team', 'position']).cumcount()+1
    data['role'] = data.position + rank.astype(str)
    data['known_role'] = [r <= ROLE_LIMITS.get(pos, 0) for pos, r in zip(data.position, rank)]
    return data


def learn_dependence(calibration):
    if calibration.empty:
        return {}, pd.DataFrame()
    ranked = add_roles(calibration)
    ranked = ranked[ranked.known_role].copy()
    # Rank normal scores estimate dependence without assuming Gaussian point errors.
    ranked['normal_score'] = ranked.groupby('role').residual.transform(
        lambda values: norm.ppf(values.rank(method='average')/(len(values)+1)))
    pairs = {}
    for _, game in ranked.groupby('game_id'):
        rows = list(game.itertuples())
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                relation = 'same' if a.team == b.team else 'opponent'
                roles = tuple(sorted([a.role, b.role]))
                key = (relation, *roles)
                vals = (a.normal_score, b.normal_score) if a.role <= b.role else (b.normal_score, a.normal_score)
                pairs.setdefault(key, []).append(vals)
    lookup, audit = {}, []
    for key, observations in pairs.items():
        sample = np.asarray(observations)
        if len(sample) < 30 or np.any(sample.std(axis=0) == 0):
            continue
        raw = float(np.corrcoef(sample.T)[0, 1])
        # Shrink noisy pair estimates toward independence with 100 prior observations.
        coefficient = raw*len(sample)/(len(sample)+100)
        lookup[key] = coefficient
        audit.append(dict(relation=key[0], role_a=key[1], role_b=key[2], observations=len(sample),
                          raw_rank_correlation=raw, shrunk_correlation=coefficient))
    return lookup, pd.DataFrame(audit)


def sample_outcomes(data, calibration, n_sims, rng):
    players = add_roles(data.drop_duplicates('player_key').rename(columns={'player_key':'athlete_key'}))
    players = players.reset_index(drop=True)
    count = len(players)
    covariance = np.eye(count)
    lookup = calibration.get('_dependence', {})
    rows = list(players.itertuples())
    for i, a in enumerate(rows):
        for j in range(i+1, count):
            b = rows[j]
            if a.game_id != b.game_id or not a.known_role or not b.known_role:
                continue
            relation = 'same' if a.team == b.team else 'opponent'
            key = (relation, *sorted([a.role, b.role]))
            covariance[i, j] = covariance[j, i] = lookup.get(key, 0.)
    with threadpool_limits(limits=1):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        repaired = (eigenvectors*np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired /= scale[:, None]*scale[None, :]
        factor = np.linalg.cholesky(repaired)
        normal = rng.standard_normal((n_sims, count)) @ factor.T
    uniforms = norm.cdf(normal)
    outcomes = {}
    for i, player in enumerate(rows):
        model = calibration[player.position]
        residuals = residual_sample(model, player.projection)
        points = player.projection + np.quantile(residuals, uniforms[:, i])
        if player.position != 'DST':
            points = np.maximum(points, 0.)
        outcomes[player.athlete_key] = points
    return outcomes
