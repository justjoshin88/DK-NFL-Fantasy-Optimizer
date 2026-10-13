"""No-key nflverse forecasts with chronological validation and explicit provenance.

These are our estimates from prior box scores, not scraped expert projections.
Data attribution: nflverse, CC BY 4.0. See README for sources and limitations.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import re

import numpy as np
import pandas as pd
import requests

try:
    from .dk_csv import normalize_name, normalize_team
    from .providers import ProviderError, fetch_public, match_projections, BAD_STATUS
except ImportError:
    from dk_csv import normalize_name, normalize_team
    from providers import ProviderError, fetch_public, match_projections, BAD_STATUS

GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
BASE_URL = 'https://github.com/nflverse/nflverse-data/releases/download/'
LICENSE_URL = 'https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md'
POSITIONS = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
FEATURES = ['recent_mean', 'last_score', 'recent_std', 'recent_volume', 'trend', 'gap_days']


def download_csv(url):
    try:
        response = requests.get(url, timeout=(5, 30), headers={'User-Agent': 'DK-NFL-Optimizer/3.0'})
        response.raise_for_status()
        if len(response.content) > 30_000_000:
            raise ProviderError('Free NFL dataset is unexpectedly large.')
        frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
        if frame.empty:
            raise ProviderError('Free NFL dataset is empty.')
        return frame
    except (requests.RequestException, pd.errors.ParserError, UnicodeError) as exc:
        raise ProviderError('Free nflverse data is unavailable. No synthetic replacement was generated.') from exc


def load_datasets(pool):
    seasons = {int(date[:4]) - (int(date[5:7]) < 3) for date in pool.game_date}
    if len(seasons) != 1:
        raise ProviderError('The free model requires one NFL season per pool.')
    season = seasons.pop()
    urls = dict(games=GAMES_URL, roster=BASE_URL + f'rosters/roster_{season}.csv',
                players_old=BASE_URL + f'stats_player/stats_player_week_{season-1}.csv',
                players_current=BASE_URL + f'stats_player/stats_player_week_{season}.csv',
                teams_old=BASE_URL + f'stats_team/stats_team_week_{season-1}.csv',
                teams_current=BASE_URL + f'stats_team/stats_team_week_{season}.csv')
    with ThreadPoolExecutor(max_workers=3) as executor:
        frames = dict(zip(urls, executor.map(download_csv, urls.values())))
    return dict(games=frames['games'], roster=frames['roster'],
                players=pd.concat([frames['players_old'], frames['players_current']], ignore_index=True),
                teams=pd.concat([frames['teams_old'], frames['teams_current']], ignore_index=True), urls=urls)


def require(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing:
        raise ProviderError(f'{label} schema changed; missing: {", ".join(sorted(missing))}.')


def player_scores(frame):
    """DK offense and kicker box-score totals, including yardage bonuses."""
    required = ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds',
                'receptions', 'receiving_yards', 'receiving_tds', 'fumbles_lost_total', 'special_teams_tds',
                'fumble_recovery_tds', 'passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions',
                'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_', 'pat_made']
    require(frame, required, 'NFL player stats')
    numbers = frame[required].apply(pd.to_numeric, errors='raise').fillna(0)
    scores = (.04*numbers.passing_yards + 4*numbers.passing_tds - numbers.passing_interceptions
              + .1*numbers.rushing_yards + 6*numbers.rushing_tds + numbers.receptions
              + .1*numbers.receiving_yards + 6*numbers.receiving_tds - numbers.fumbles_lost_total
              + 6*(numbers.special_teams_tds + numbers.fumble_recovery_tds)
              + 2*(numbers.passing_2pt_conversions + numbers.rushing_2pt_conversions + numbers.receiving_2pt_conversions)
              + 3*(numbers.passing_yards >= 300) + 3*(numbers.rushing_yards >= 100) + 3*(numbers.receiving_yards >= 100))
    kicker = frame.position.eq('K')
    scores.loc[kicker] = (3*(numbers.fg_made_0_19 + numbers.fg_made_20_29 + numbers.fg_made_30_39)
                          + 4*numbers.fg_made_40_49 + 5*(numbers.fg_made_50_59 + numbers.fg_made_60_) + numbers.pat_made)[kicker]
    return scores


def defense_scores(teams, games):
    required = ['game_id', 'team', 'opponent_team', 'def_sacks', 'def_interceptions', 'fumble_recovery_opp',
                'def_tds', 'special_teams_tds', 'def_safeties', 'def_punt_blocks', 'def_fg_blocks', 'def_pat_blocks', 'def_2pt_made']
    require(teams, required, 'NFL team stats')
    require(games, ['game_id', 'home_team', 'away_team', 'home_score', 'away_score'], 'NFL schedules')
    if teams.duplicated(['game_id', 'team']).any():
        raise ProviderError('Duplicate team/game records in NFL stats.')
    merged = teams.merge(games[['game_id', 'home_team', 'home_score', 'away_score']], on='game_id', validate='many_to_one')
    opponent = teams[['game_id', 'team', 'def_tds', 'def_2pt_made']].rename(
        columns={'team': 'opponent_team', 'def_tds': 'opponent_def_tds', 'def_2pt_made': 'opponent_def_2pt'})
    merged = merged.merge(opponent, on=['game_id', 'opponent_team'], validate='one_to_one')
    numeric = required[3:] + ['opponent_def_tds', 'opponent_def_2pt', 'home_score', 'away_score']
    for col in numeric:
        merged[col] = pd.to_numeric(merged[col], errors='raise')
    merged = merged.dropna(subset=['home_score', 'away_score'])
    allowed = np.where(merged.team.eq(merged.home_team), merged.away_score, merged.home_score)
    # Defensive TD points and defensive 2pt returns are not points allowed by DST.
    # The extra point following a defensive TD remains counted by DK's rule.
    allowed = allowed - 6*merged.opponent_def_tds - 2*merged.opponent_def_2pt
    tier = np.select([allowed == 0, allowed <= 6, allowed <= 13, allowed <= 20,
                      allowed <= 27, allowed <= 34], [10, 7, 4, 1, 0, -1], default=-4)
    merged['dk_score'] = (merged.def_sacks + 2*(merged.def_interceptions + merged.fumble_recovery_opp)
                          + 6*(merged.def_tds + merged.special_teams_tds) + 2*merged.def_safeties
                          + 2*(merged.def_punt_blocks + merged.def_fg_blocks + merged.def_pat_blocks + merged.def_2pt_made) + tier)
    merged['position'] = 'DST'
    merged['player_id'] = 'DST:' + merged.team
    merged['player_display_name'] = merged.team
    merged['volume'] = 1.
    return merged


def schedule_times(games):
    require(games, ['game_id', 'gameday', 'gametime', 'away_team', 'home_team', 'season', 'week', 'game_type'], 'NFL schedules')
    games = games.copy()
    dates = pd.to_datetime(games.gameday.astype(str) + ' ' + games.gametime.fillna('13:00').astype(str), errors='coerce')
    if dates.isna().any():
        raise ProviderError('Schedule contains invalid kickoff timestamps.')
    games['kickoff_time'] = dates.dt.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise').dt.tz_convert('UTC')
    for col in ['away_team', 'home_team']:
        games[col] = games[col].map(normalize_team)
    return games


def feature_vector(prior, target_date):
    prior = prior.sort_values('kickoff_time').tail(6)
    if len(prior) < 3:
        return None
    scores = prior.dk_score.to_numpy(dtype=float)
    volumes = prior.volume.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(volumes).all():
        raise ProviderError('Invalid numerical NFL history.')
    weights = .7 ** np.arange(len(scores)-1, -1, -1)
    mean = float(np.average(scores, weights=weights))
    gap = (target_date - prior.kickoff_time.iloc[-1]).total_seconds() / 86400
    return np.array([mean, scores[-1], scores.std(), volumes[-3:].mean(),
                     scores[-3:].mean() - scores.mean(), min(gap, 180)], dtype=float)


def prepare_history(datasets, as_of):
    games = schedule_times(datasets['games'])
    completed = games[(games.kickoff_time < as_of - pd.Timedelta(hours=6)) & games.home_score.notna() & games.away_score.notna()]
    players = datasets['players'].copy()
    require(players, ['player_id', 'position', 'player_display_name', 'team', 'game_id', 'season_type', 'attempts', 'carries', 'targets', 'fg_att'], 'NFL player stats')
    players = players[players.position.isin(POSITIONS - {'DST'}) & players.season_type.eq('REG')].copy()
    players['dk_score'] = player_scores(players)
    players['volume'] = np.select([players.position.eq('QB'), players.position.eq('K')],
                                  [players.attempts.fillna(0), players.fg_att.fillna(0)],
                                  default=players.carries.fillna(0) + players.targets.fillna(0))
    players = players.merge(completed[['game_id', 'kickoff_time']], on='game_id', validate='many_to_one')
    if players.duplicated(['player_id', 'game_id']).any():
        raise ProviderError('Duplicate NFL player/game records.')
    teams = datasets['teams'][datasets['teams'].season_type.eq('REG')].copy()
    for column in ['team', 'opponent_team']:
        teams[column] = teams[column].map(normalize_team)
    defense = defense_scores(teams, completed).merge(completed[['game_id', 'kickoff_time']], on='game_id', validate='many_to_one')
    history = pd.concat([players, defense], ignore_index=True)
    history['team'] = history.team.map(normalize_team)
    records = []
    for (player_id, position), group in history.groupby(['player_id', 'position'], sort=False):
        group = group.sort_values('kickoff_time').reset_index(drop=True)
        for i in range(3, len(group)):
            row = group.iloc[i]
            vector = feature_vector(group.iloc[:i], row.kickoff_time)
            records.append(dict(player_id=player_id, position=position, game_date=row.kickoff_time,
                                actual=row.dk_score, vector=vector))
    return history, pd.DataFrame(records), games


def fit_ridge(x, y, regularization):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1])*regularization
    penalty[0, 0] = 0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return dict(mean=mean, scale=scale, coef=coef)


def predict(model, x):
    if model['method'] == 'recent_mean':
        return np.maximum(x[:, 0], 0)
    fitted = model['fit']
    design = np.column_stack([np.ones(len(x)), (x-fitted['mean'])/fitted['scale']])
    return np.maximum(design @ fitted['coef'], 0)


def train_models(records):
    if records.empty:
        raise ProviderError('No pregame historical feature records can be constructed.')
    models, metrics = {}, []
    for position, group in records.groupby('position'):
        dates = sorted(group.game_date.unique())
        if len(dates) < 12:
            continue
        train_end, validation_end = dates[int(.6*len(dates))], dates[int(.8*len(dates))]
        train = group[group.game_date < train_end]
        validation = group[(group.game_date >= train_end) & (group.game_date < validation_end)]
        test = group[group.game_date >= validation_end]
        if len(train) < 50 or len(validation) < 20 or len(test) < 20:
            continue
        xt, yt = np.stack(train.vector), train.actual.to_numpy(float)
        xv, yv = np.stack(validation.vector), validation.actual.to_numpy(float)
        candidates = [dict(method='recent_mean', regularization=0)]
        candidates += [dict(method='ridge', regularization=lam, fit=fit_ridge(xt, yt, lam)) for lam in [1., 10., 100.]]
        chosen = min(candidates, key=lambda model: np.abs(predict(model, xv)-yv).mean())
        validation_errors = yv-predict(chosen, xv)
        # Refit the selected model before the untouched test period.
        before_test = group[group.game_date < validation_end]
        fixed = dict(chosen)
        if chosen['method'] == 'ridge':
            fixed['fit'] = fit_ridge(np.stack(before_test.vector), before_test.actual.to_numpy(float), chosen['regularization'])
        xp, yp = np.stack(test.vector), test.actual.to_numpy(float)
        errors = yp-predict(fixed, xp)
        lower, upper = np.quantile(validation_errors, [.1, .9])
        metrics.append(dict(position=position, selected_model=chosen['method'], train_rows=len(train),
                            validation_rows=len(validation), test_rows=len(test), test_start=str(validation_end),
                            test_mae=float(np.abs(errors).mean()), test_rmse=float(np.sqrt((errors**2).mean())),
                            baseline_test_mae=float(np.abs(yp-xp[:, 0]).mean()), test_bias=float(errors.mean()),
                            interval_80_coverage=float(((errors>=lower)&(errors<=upper)).mean())))
        deployed = dict(chosen)
        if chosen['method'] == 'ridge':
            deployed['fit'] = fit_ridge(np.stack(group.vector), group.actual.to_numpy(float), chosen['regularization'])
        deployed['residuals'] = validation_errors.tolist()
        models[position] = deployed
    if not models:
        raise ProviderError('Not enough history for chronological model validation.')
    return models, pd.DataFrame(metrics)


def name_key(value):
    # Remove generational suffixes only; no fuzzy name or team guessing.
    value = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', str(value).strip(), flags=re.I)
    return normalize_name(value)


def generate_forecasts(pool, datasets, as_of=None):
    as_of = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if as_of.tzinfo is None:
        raise ValueError('Forecast cutoff must be timezone-aware.')
    history, records, games = prepare_history(datasets, as_of)
    models, metrics = train_models(records)
    roster = datasets['roster'].copy()
    require(roster, ['full_name', 'team', 'position', 'status', 'gsis_id'], 'NFL rosters')
    roster['team'] = roster.team.map(normalize_team)
    roster['name_key'] = roster.full_name.map(name_key)
    roster_lookup = {key: group for key, group in roster.groupby(['name_key', 'team', 'position'])}
    histories = {(pid, pos): group for (pid, pos), group in history.groupby(['player_id', 'position'])}
    output, missing = [], []
    for _, player in pool.drop_duplicates('player_key').iterrows():
        if player.position not in models:
            missing.append((player.player_key, 'No validated model for position'))
            continue
        valid_game = games[(games.gameday == player.game_date) &
                           (((games.away_team == player.team) & (games.home_team == player.opponent)) |
                            ((games.home_team == player.team) & (games.away_team == player.opponent)))]
        if len(valid_game) != 1:
            missing.append((player.player_key, 'Pool date/team/opponent does not match free NFL schedule'))
            continue
        game = valid_game.iloc[0]
        if abs((pd.Timestamp(player.kickoff)-game.kickoff_time).total_seconds()) > 60:
            missing.append((player.player_key, 'Pool kickoff differs from NFL schedule'))
            continue
        if player.position == 'DST':
            player_id = 'DST:' + player.team
        else:
            matching = roster_lookup.get((name_key(player.player_name), player.team, player.position))
            if matching is None or matching.gsis_id.dropna().nunique() != 1:
                missing.append((player.player_key, 'No unique exact current-roster identity'))
                continue
            if not matching.status.eq('ACT').any():
                missing.append((player.player_key, 'Current roster does not report active status'))
                continue
            player_id = matching.gsis_id.dropna().iloc[0]
        prior = histories.get((player_id, player.position))
        if prior is None or len(prior) < 3:
            missing.append((player.player_key, 'Fewer than three observed NFL games; no cold-start guess'))
            continue
        prior = prior[prior.kickoff_time < min(as_of, game.kickoff_time)].sort_values('kickoff_time')
        vector = feature_vector(prior, game.kickoff_time)
        if vector is None or (as_of - prior.kickoff_time.iloc[-1]).days > 21:
            missing.append((player.player_key, 'Insufficient or stale playing history'))
            continue
        model = models[player.position]
        projection = float(predict(model, vector.reshape(1, -1))[0])
        residuals = np.asarray(model['residuals'])
        output.append(dict(player_name=player.player_name, position=player.position, team=player.team, opponent=player.opponent,
                           game_date=player.game_date, projection=projection,
                           projection_std=float(residuals.std()), projection_ceiling=projection+float(np.quantile(residuals, .9)),
                           injury_status='Roster active; game-day injury status unverified', ownership=np.nan,
                           source='Our free nflverse model', source_url='https://nflreadr.nflverse.com/',
                           source_updated=as_of.isoformat(), fetched_at=as_of.isoformat(), model_type=model['method'],
                           training_last_game=str(history.kickoff_time.max()), last_played=str(prior.kickoff_time.iloc[-1]),
                           history_games=len(prior), historical_player_id=player_id))
    forecasts = pd.DataFrame(output)
    return forecasts, metrics, dict(missing), models


def fetch_free(pool, datasets=None, as_of=None):
    messages = []
    public = pd.DataFrame()
    try:
        public, messages = fetch_public(pool)
    except ProviderError as exc:
        messages = [f'Expert forecast source unavailable: {exc}']
    data = match_projections(pool, public) if not public.empty else None
    if data is not None and data.exclusion_reason.eq('').all():
        return data, messages
    datasets = datasets or load_datasets(pool)
    generated, metrics, missing, models = generate_forecasts(pool, datasets, as_of)
    if generated.empty:
        raise ProviderError('Free model found no eligible validated player histories for this slate. Check schedule and roster coverage.')
    model_data = match_projections(pool, generated)
    # Preserve model audit columns and residual-derived ceilings after the generic join.
    extra = generated.set_index(['player_name', 'team', 'position', 'game_date'])
    for column in ['projection_std', 'projection_ceiling', 'model_type', 'training_last_game', 'last_played', 'history_games', 'historical_player_id']:
        model_data[column] = [extra.loc[(p.player_name, p.team, p.position, p.game_date), column]
                              if (p.player_name, p.team, p.position, p.game_date) in extra.index else np.nan for _, p in pool.iterrows()]
    for i, player in pool.reset_index(drop=True).iterrows():
        if player.player_key in missing:
            model_data.loc[i, 'exclusion_reason'] = missing[player.player_key]
        if data is not None:
            original = data.iloc[i]
            if str(original.injury_status).upper() in BAD_STATUS:
                model_data.loc[i, 'exclusion_reason'] = original.exclusion_reason
                model_data.loc[i, 'injury_status'] = original.injury_status
            elif not original.exclusion_reason:
                for column in original.index:
                    model_data.loc[i, column] = original[column]
                # Own-model uncertainty must not be attached to expert projections.
                for column in ['projection_std', 'projection_ceiling', 'model_type', 'training_last_game', 'last_played', 'history_games', 'historical_player_id']:
                    model_data.loc[i, column] = np.nan
    model_data.attrs['free_model_metrics'] = metrics
    model_data.attrs['free_model_notes'] = 'Own forecasts from real past box scores; conditional on playing. Injury changes, cold-start players, correlations, ownership and payouts are not fully modeled.'
    messages.append('Free model estimates fill missing expert forecasts. Review chronological test errors and confirm final injury/role news before using these estimates.')
    return model_data, messages
