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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .forecast_features import MatchupContext, FEATURES, WORKLOAD, recency_vector
    from .forecast_validation import train_models, predict, fit_ridge, residual_sample
    from .availability import depth_snapshot, player_availability
    from .scenarios import learn_dependence
    from .dk_csv import normalize_name, normalize_team
    from .providers import ProviderError, fetch_public, match_projections, BAD_STATUS
except ImportError:
    from forecast_features import MatchupContext, FEATURES, WORKLOAD, recency_vector
    from forecast_validation import train_models, predict, fit_ridge, residual_sample
    from availability import depth_snapshot, player_availability
    from scenarios import learn_dependence
    from dk_csv import normalize_name, normalize_team
    from providers import ProviderError, fetch_public, match_projections, BAD_STATUS

GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
BASE_URL = 'https://github.com/nflverse/nflverse-data/releases/download/'
LICENSE_URL = 'https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md'
POSITIONS = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
MODEL_AUDIT = ['projection_std', 'projection_floor', 'projection_ceiling', 'model_type', 'training_last_game', 'last_played',
               'history_games', 'historical_player_id', 'model_built_at', 'recent_opportunities', 'depth_rank', 'depth_updated',
               'injury_report_status', 'injury_report_week', 'availability_note']

SCORING_COLUMNS = ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds',
                'receptions', 'receiving_yards', 'receiving_tds', 'fumbles_lost_total', 'special_teams_tds',
                'fumble_recovery_tds', 'passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions',
                'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_', 'pat_made']
PLAYER_COLUMNS = list(dict.fromkeys(SCORING_COLUMNS + ['player_id', 'player_display_name', 'position', 'team', 'game_id', 'season_type', 'attempts', 'carries', 'targets', 'fg_att']))


def download_csv(url):
    try:
        with requests.Session() as session:
            retry = Retry(total=2, connect=2, read=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, timeout=(10, 45), headers={'User-Agent': 'DK-NFL-Optimizer/4.0'})
        response.raise_for_status()
        if len(response.content) > 30_000_000:
            raise ProviderError('Free NFL dataset is unexpectedly large.')
        frame = pd.read_csv(io.BytesIO(response.content), compression='gzip' if url.endswith('.gz') else None, low_memory=False)
        if frame.empty:
            raise ProviderError('Free NFL dataset is empty.')
        return frame
    except (requests.RequestException, pd.errors.ParserError, UnicodeError) as exc:
        raise ProviderError(f'Free NFL file {url.rsplit("/", 1)[-1]} is unavailable. No synthetic replacement was generated.') from exc


def load_datasets(pool):
    seasons = {int(date[:4]) - (int(date[5:7]) < 3) for date in pool.game_date}
    if len(seasons) != 1:
        raise ProviderError('The free model requires one NFL season per pool.')
    season = seasons.pop()
    urls = dict(games=GAMES_URL, roster=BASE_URL + f'rosters/roster_{season}.csv')
    for year in range(max(1999, season-3), season+1):
        urls[f'players_{year}'] = BASE_URL + f'stats_player/stats_player_week_{year}.csv'
        urls[f'teams_{year}'] = BASE_URL + f'stats_team/stats_team_week_{year}.csv'
    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = dict(zip(urls, executor.map(download_csv, urls.values())))
    for name, frame in frames.items():
        if name.startswith('players_'):
            require(frame, PLAYER_COLUMNS, 'NFL player stats')
            frames[name] = frame[PLAYER_COLUMNS].copy()
    datasets = dict(games=frames['games'], roster=frames['roster'],
                    players=pd.concat([v for k, v in frames.items() if k.startswith('players_')], ignore_index=True),
                    teams=pd.concat([v for k, v in frames.items() if k.startswith('teams_')], ignore_index=True), urls=urls, source_notes=[])
    optional = dict(depth=BASE_URL + f'depth_charts/depth_charts_{season}.csv.gz',
                    injuries=BASE_URL + f'injuries/injuries_{season}.csv')
    def optional_download(item):
        name, url = item
        try:
            return name, download_csv(url), None
        except ProviderError as exc:
            return name, None, str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for name, data, error in executor.map(optional_download, optional.items()):
            if data is not None:
                if name == 'depth':
                    data, _ = depth_snapshot(data, pd.Timestamp.now(tz='UTC'))
                datasets[name] = data
                urls[name] = optional[name]
            else:
                datasets['source_notes'].append(f'{name}: {error}; participation remains unverified.')
    # Injury files lack row update times. Parse the publisher timestamp, not our fetch time.
    try:
        response = requests.get(BASE_URL+'injuries/timestamp.json', timeout=(5, 15))
        response.raise_for_status()
        raw = response.json()['last_updated']
        match = re.fullmatch(r'(.*) (EDT|EST|UTC)', raw)
        if match:
            offset = {'EDT':'-04:00', 'EST':'-05:00', 'UTC':'+00:00'}[match[2]]
            datasets['injuries_updated'] = pd.Timestamp(match[1]+offset)
        else:
            stamp = pd.Timestamp(raw)
            if stamp.tzinfo is not None:
                datasets['injuries_updated'] = stamp
    except (requests.RequestException, ValueError, KeyError):
        datasets['source_notes'].append('Injury publisher freshness unavailable; report flags are not applied.')
    return datasets


def require(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing:
        raise ProviderError(f'{label} schema changed; missing: {", ".join(sorted(missing))}.')


def player_scores(frame):
    """DK offense and kicker box-score totals, including yardage bonuses."""
    required = SCORING_COLUMNS
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
    opponent = teams[['game_id', 'team', 'def_tds', 'def_safeties']].rename(
        columns={'team': 'opponent_team', 'def_tds': 'opponent_def_tds', 'def_safeties': 'opponent_safeties'})
    merged = merged.merge(opponent, on=['game_id', 'opponent_team'], validate='one_to_one')
    numeric = required[3:] + ['opponent_def_tds', 'opponent_safeties', 'home_score', 'away_score']
    for col in numeric:
        merged[col] = pd.to_numeric(merged[col], errors='raise')
    merged = merged.dropna(subset=['home_score', 'away_score'])
    allowed = np.where(merged.team.eq(merged.home_team), merged.away_score, merged.home_score)
    # Opposing defensive TDs and safeties are scored against our offense.
    # Conversion/extra-point returns and PATs do count as DST points allowed.
    # Official NFL Classic scoring notes: https://www.draftkings.com/help/rules/1
    allowed = allowed - 6*merged.opponent_def_tds - 2*merged.opponent_safeties
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
    games['kickoff_verified'] = games.gametime.notna()
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
    players['team'] = players.team.map(normalize_team)
    players['dk_score'] = player_scores(players)
    players['tds'] = players.passing_tds + players.rushing_tds + players.receiving_tds
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
    for column in WORKLOAD:
        defense[column] = 0.
    history = pd.concat([players, defense], ignore_index=True)
    history['team'] = history.team.map(normalize_team)
    # Resolve opponents from the exact schedule rather than relying on optional stat fields.
    schedule = completed.set_index('game_id')
    home = history.game_id.map(schedule.home_team)
    away = history.game_id.map(schedule.away_team)
    history['opponent_team'] = np.where(history.team.eq(home), away, home)
    context = MatchupContext(history, games)
    denominators = np.array([context.denominators.get((r.team, r.game_id), (0, 0)) for r in history.itertuples()], float)
    history['target_share_feature'] = history.targets.to_numpy(float)/np.maximum(denominators[:, 0], 1)
    history['carry_share_feature'] = history.carries.to_numpy(float)/np.maximum(denominators[:, 1], 1)
    records = []
    for (player_id, position), group in history.groupby(['player_id', 'position'], sort=False):
        group = group.sort_values('kickoff_time').reset_index(drop=True)
        scores, volumes, times = group.dk_score.to_numpy(float), group.volume.to_numpy(float), group.kickoff_time.dt.as_unit('ns').astype('int64').to_numpy()
        workload = group[WORKLOAD].to_numpy(float)
        shares = group[['target_share_feature', 'carry_share_feature']].to_numpy(float)
        for i, row in enumerate(group.itertuples()):
            if i < 3:
                continue
            start = max(0, i-6)
            vector = recency_vector(scores[:i], volumes[:i], times[:i], row.kickoff_time)
            vector = np.r_[vector, context.vector_arrays(workload[start:i], shares[start:i], volumes[start:i],
                                                        row.kickoff_time, row.team, row.opponent_team, position, row.game_id)]
            if not np.isfinite(vector).all():
                raise ProviderError('Invalid historical forecast features.')
            records.append(dict(player_id=player_id, position=position, game_date=row.kickoff_time,
                                actual=row.dk_score, vector=vector, team=row.team, opponent=row.opponent_team,
                                game_id=row.game_id))
    return history, pd.DataFrame(records), games


def name_key(value):
    # Remove generational suffixes only; no fuzzy name or team guessing.
    value = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', str(value).strip(), flags=re.I)
    return normalize_name(value)


def prepare_model(datasets, as_of=None):
    as_of = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if as_of.tzinfo is None:
        raise ValueError('Forecast cutoff must be timezone-aware.')
    history, records, games = prepare_history(datasets, as_of)
    models, metrics = train_models(records)
    return dict(history=history, games=games, models=models, metrics=metrics, as_of=as_of)


def generate_forecasts(pool, datasets, as_of=None, artifacts=None):
    as_of = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if as_of.tzinfo is None:
        raise ValueError('Forecast cutoff must be timezone-aware.')
    artifacts = artifacts or prepare_model(datasets, as_of)
    if artifacts['as_of'] > as_of:
        raise ValueError('Cached model contains history after the forecast cutoff.')
    history, games, models, metrics = [artifacts[k] for k in ['history', 'games', 'models', 'metrics']]
    context = MatchupContext(history, games)
    depth, depth_note = depth_snapshot(datasets.get('depth'), as_of)
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
        if not game.kickoff_verified:
            missing.append((player.player_key, 'NFL schedule kickoff is unverified'))
            continue
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
        unavailable, availability = player_availability(player_id, player.team, player.position, game, datasets, as_of, depth)
        if unavailable:
            missing.append((player.player_key, unavailable))
            continue
        prior = histories.get((player_id, player.position))
        if prior is None or len(prior) < 3:
            missing.append((player.player_key, 'Fewer than three observed NFL games; no cold-start guess'))
            continue
        prior = prior[prior.kickoff_time < min(as_of, game.kickoff_time)].sort_values('kickoff_time')
        vector = feature_vector(prior, game.kickoff_time)
        if vector is None or (as_of - prior.kickoff_time.iloc[-1]).days > 21:
            missing.append((player.player_key, 'Insufficient or stale playing history'))
            continue
        vector = np.r_[vector, context.vector(prior.tail(6), game.kickoff_time, player.team, player.opponent, player.position, game.game_id)]
        model = models[player.position]
        projection = float(predict(model, vector.reshape(1, -1))[0])
        residuals = residual_sample(model, projection)
        output.append(dict(player_name=player.player_name, position=player.position, team=player.team, opponent=player.opponent,
                           game_date=player.game_date, projection=projection,
                           projection_std=float(residuals.std()), projection_floor=max(0., projection+float(np.quantile(residuals, .1))),
                           projection_ceiling=max(projection, projection+float(np.quantile(residuals, .9))),
                           injury_status='Roster active; game-day injury status unverified', ownership=np.nan,
                           source='Our free nflverse model', source_url='https://nflreadr.nflverse.com/',
                           source_updated=as_of.isoformat(), fetched_at=as_of.isoformat(), model_type=model['method'],
                           training_last_game=str(history.kickoff_time.max()), last_played=str(prior.kickoff_time.iloc[-1]),
                           history_games=len(prior), historical_player_id=player_id, model_built_at=artifacts['as_of'].isoformat(),
                           recent_opportunities=float(prior.volume.tail(3).mean()), **availability))
    forecasts = pd.DataFrame(output)
    return forecasts, metrics, dict(missing), models


def fetch_free(pool, datasets=None, as_of=None, artifacts=None):
    messages = []
    public = pd.DataFrame()
    try:
        public, messages = fetch_public(pool)
    except ProviderError as exc:
        messages = [f'Expert forecast source unavailable: {exc}']
    data = match_projections(pool, public) if not public.empty else None
    datasets = datasets or load_datasets(pool)
    messages.extend(datasets.get('source_notes', []))
    injury_data = datasets.get('injuries')
    if injury_data is None or injury_data.empty:
        messages.append('Current injury source unavailable; final participation remains unverified.')
    else:
        latest_week = injury_data.week.max() if 'week' in injury_data else 'unknown'
        messages.append(f'Injury source latest reported week: {latest_week}. Only exact-week fresh reports are applied; absent reports do not mean healthy.')
    generated, metrics, missing, models = generate_forecasts(pool, datasets, as_of, artifacts)
    if generated.empty and data is not None and data.exclusion_reason.eq('').any():
        for i, player in pool.reset_index(drop=True).iterrows():
            reason = missing.get(player.player_key, '')
            if reason.startswith(('Unavailable on exact-week', 'Conflicting exact-week', 'Current depth chart')):
                data.loc[i, 'exclusion_reason'] = reason
        messages.append('Own model has no eligible histories; only covered public forecasts remain.')
        return data, messages
    if generated.empty:
        raise ProviderError('Free model found no eligible validated player histories for this slate. Check schedule and roster coverage.')
    model_data = match_projections(pool, generated)
    # Preserve model audit columns and residual-derived ceilings after the generic join.
    extra = generated.set_index(['player_name', 'team', 'position', 'game_date'])
    for column in MODEL_AUDIT:
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
                for column in MODEL_AUDIT:
                    model_data.loc[i, column] = np.nan
        reason = missing.get(player.player_key, '')
        if reason.startswith(('Unavailable on exact-week', 'Conflicting exact-week', 'Current depth chart')):
            model_data.loc[i, 'exclusion_reason'] = reason
    dependence, dependence_audit = learn_dependence(metrics.attrs.get('calibration_records', pd.DataFrame()))
    model_data.attrs['free_model_calibration'] = {pos: {k: model[k] for k in ['residuals', 'residual_bounds', 'residual_bins']} for pos, model in models.items()}
    model_data.attrs['free_model_calibration']['_dependence'] = dependence
    model_data.attrs['free_model_dependence'] = dependence_audit
    model_data.attrs['free_model_metrics'] = metrics
    model_data.attrs['free_model_folds'] = metrics.attrs.get('fold_metrics', pd.DataFrame())
    model_data.attrs['free_model_test_predictions'] = metrics.attrs.get('test_predictions', pd.DataFrame())
    model_data.attrs['free_model_notes'] = 'Own forecasts from real historical box scores; conditional on playing. Backtests reconstruct history and are not archived as-published forecasts. Exact-week injury/depth checks apply only when fresh and available; final participation, changing roles, ownership and payouts remain unverified.'
    messages.append('Free model estimates fill missing expert forecasts. Review chronological test errors and confirm final injury/role news before using these estimates.')
    return model_data, messages
