"""Features built from completed games strictly before each forecast cutoff."""
from bisect import bisect_left
import numpy as np
import pandas as pd

BASE_FEATURES = ['recent_mean', 'last_score', 'recent_std', 'recent_volume', 'trend', 'gap_days']
FEATURES = BASE_FEATURES + ['recent_carries', 'recent_targets', 'recent_pass_yards', 'recent_rush_yards',
                            'recent_rec_yards', 'recent_tds', 'last_volume', 'volume_trend',
                            'target_share', 'carry_share', 'home', 'prior_home_indoor', 'team_points',
                            'opponent_position_points', 'opponent_rest', 'team_rest']
WORKLOAD = ['carries', 'targets', 'passing_yards', 'rushing_yards', 'receiving_yards', 'tds']


def recency_vector(scores, volumes, times, target):
    if len(scores) < 3:
        return None
    scores, volumes, times = scores[-6:], volumes[-6:], times[-6:]
    weights = .7 ** np.arange(len(scores)-1, -1, -1)
    gap = (pd.Timestamp(target).value - times[-1])/86_400_000_000_000
    return np.array([np.average(scores, weights=weights), scores[-1], scores.std(), volumes[-3:].mean(),
                     scores[-3:].mean()-scores.mean(), min(gap, 180)], dtype=float)


class MatchupContext:
    def __init__(self, history, games):
        self.games = games.set_index('game_id').to_dict('index')
        self.venues = {}
        if 'roof' in games:
            completed_venues = games[games.home_score.notna() & games.away_score.notna() & games.roof.notna()]
            for team, rows in completed_venues.groupby('home_team'):
                rows = rows.sort_values('kickoff_time')
                self.venues[team] = (rows.kickoff_time.dt.as_unit('ns').astype('int64').to_numpy(), rows.roof.isin(['dome', 'closed']).to_numpy(float))
        # Team offense and points allowed by position, with zero totals for absent position rows.
        self.offense, self.allowed, self.cache = {}, {}, {}
        offense = history[history.position.ne('DST')].groupby(['team', 'game_id']).agg(
            kickoff_time=('kickoff_time', 'first'), points=('dk_score', 'sum'),
            targets=('targets', 'sum'), carries=('carries', 'sum')).reset_index()
        self.denominators = {(r.team, r.game_id): (r.targets, r.carries) for r in offense.itertuples()}
        for team, rows in offense.groupby('team'):
            rows = rows.sort_values('kickoff_time')
            self.offense[team] = (rows.kickoff_time.dt.as_unit('ns').astype('int64').to_numpy(), rows[['points', 'targets', 'carries']].to_numpy(float))
        # Opposing DST totals are allowed by a team's offense; all positions use exact opponent identity.
        totals = history.groupby(['opponent_team', 'game_id', 'position']).dk_score.sum()
        team_games = history[['team', 'game_id', 'kickoff_time']].drop_duplicates()
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        for team, rows in team_games.groupby('team'):
            rows = rows.sort_values('kickoff_time')
            times = rows.kickoff_time.dt.as_unit('ns').astype('int64').to_numpy()
            for pos in positions:
                values = np.array([totals.get((team, gid, pos), 0.) for gid in rows.game_id], float)
                self.allowed[(team, pos)] = (times, values)

    def recent(self, lookup, key, target):
        entry = lookup.get(key)
        if entry is None:
            return np.empty(0), np.empty(0)
        times, values = entry
        # At least six hours after kickoff, matching the training data completion guard.
        index = bisect_left(times, pd.Timestamp(target).value - 6*3_600_000_000_000)
        return times[max(0, index-6):index], values[max(0, index-6):index]

    def team_vector(self, target, team, opponent, position, game_id):
        key = (game_id, team, opponent, position)
        if key in self.cache:
            return self.cache[key]
        own_times, own = self.recent(self.offense, team, target)
        opp_times, allowed = self.recent(self.allowed, (opponent, position), target)
        game = self.games[game_id]
        now = pd.Timestamp(target).value
        rest = lambda times: min((now-times[-1])/86_400_000_000_000, 30) if len(times) else 7.
        _, prior_venue = self.recent(self.venues, game['home_team'], target)
        vector = np.array([float(team == game['home_team']), prior_venue[-1] if len(prior_venue) else 0.,
                           own[:, 0].mean() if len(own) else 0., allowed.mean() if len(allowed) else 0.,
                           rest(opp_times), rest(own_times)])
        self.cache[key] = vector
        return vector

    def vector_arrays(self, workload, shares, volumes, target, team, opponent, position, game_id):
        weights = .7 ** np.arange(len(workload)-1, -1, -1)
        weighted = np.average(workload, axis=0, weights=weights)
        recent = volumes[-3:]
        return np.r_[weighted, recent[-1], recent[-1]-recent.mean(), shares[-3:].mean(axis=0),
                     self.team_vector(target, team, opponent, position, game_id)]

    def vector(self, prior, target, team, opponent, position, game_id):
        prior = prior.tail(6)
        shares = np.array([[row.targets/max(self.denominators.get((row.team, row.game_id), (0, 0))[0], 1),
                            row.carries/max(self.denominators.get((row.team, row.game_id), (0, 0))[1], 1)] for row in prior.itertuples()])
        return self.vector_arrays(prior[WORKLOAD].to_numpy(float), shares, prior.volume.to_numpy(float),
                                  target, team, opponent, position, game_id)
