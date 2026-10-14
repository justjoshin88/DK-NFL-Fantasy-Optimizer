"""Read-only providers with exact game matching and explicit missing-data outcomes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import math
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

try:
    from .dk_csv import normalize_name, normalize_team
except ImportError:
    from dk_csv import normalize_name, normalize_team

DFF = 'https://www.dailyfantasyfuel.com/nfl/projections/draftkings/'
BAD_STATUS = {'O', 'OUT', 'D', 'DOUBTFUL', 'INACTIVE', 'IR', 'SUSPENDED'}


class ProviderError(ValueError):
    pass


def request(url, *, key=None, expect_json=False):
    headers = {'User-Agent': 'DK-NFL-Fantasy-Optimizer/2.0', 'Accept': 'application/json' if expect_json else 'text/html'}
    if key:
        headers['Ocp-Apim-Subscription-Key'] = key
    try:
        response = requests.get(url, headers=headers, timeout=(5, 15))
        response.raise_for_status()
        if len(response.content) > 8_000_000:
            raise ProviderError('Provider response exceeds the allowed size.')
        return response.json() if expect_json else response.text
    except (requests.RequestException, ValueError) as exc:
        response = getattr(exc, 'response', None)
        detail = f'HTTP {response.status_code}' if response is not None else type(exc).__name__
        # Report host and status only; never echo query parameters or credentials.
        raise ProviderError(f'{urlparse(url).hostname} request failed ({detail}). The source could not be verified.') from exc


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_dff(html, url, fetched_at=None):
    soup = BeautifulSoup(html, 'html.parser')
    updated = soup.select_one('time[itemprop="dateModified"]')
    timestamp = updated.get('datetime') if updated else None
    if timestamp is None:
        raise ProviderError('Public projections have no update timestamp; freshness cannot be verified.')
    fetched_at = fetched_at or datetime.now(timezone.utc)
    try:
        published = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if published.tzinfo is None:
            raise ValueError('Missing timezone')
    except ValueError as exc:
        raise ProviderError('Invalid source update timestamp.') from exc
    if published > fetched_at + timedelta(minutes=10) or fetched_at - published > timedelta(hours=24):
        raise ProviderError('Public projections are over 24 hours old or have an invalid future timestamp. Refresh later or use a current projection file.')
    records = []
    for row in soup.select('tr.projections-listing'):
        a = row.attrs
        if not a.get('data-start_date') or not a.get('data-name'):
            continue
        own = row.select_one('[data-own]')
        position = a.get('data-pos', '').upper()
        if position not in {'QB', 'RB', 'WR', 'TE', 'DST', 'K'}:
            continue
        records.append(dict(player_name=a['data-name'].strip(), position=position,
                            team=normalize_team(a.get('data-team', '')), opponent=normalize_team(a.get('data-opp', '')),
                            game_date=a['data-start_date'], projection=numeric(a.get('data-ppg_proj')),
                            injury_status=a.get('data-inj', '').upper() or 'No injury flag reported',
                            ownership=numeric(own.get('data-own')) if own else np.nan,
                            spread=numeric(a.get('data-spread')), game_total=numeric(a.get('data-ou')),
                            implied_team_total=numeric(a.get('data-proj_score')), source='Daily Fantasy Fuel',
                            source_url=url, source_updated=published.isoformat(), fetched_at=fetched_at.isoformat()))
    return pd.DataFrame(records)


def fetch_public(pool):
    dates = sorted(pool.game_date.unique())
    def fetch_date(date):
        url = DFF + date
        try:
            return parse_dff(request(url), url), None
        except ProviderError as exc:
            return pd.DataFrame(), f'{date}: {exc}'
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_date, dates))
    frames = [frame for frame, _ in results if not frame.empty]
    messages = [message for _, message in results if message]
    if not frames:
        raise ProviderError('No current public projections are published for these game dates. ' + ' '.join(messages))
    return pd.concat(frames, ignore_index=True), messages


def fetch_sportsdata(pool, key, production_confirmed=False):
    if not production_confirmed:
        raise ProviderError('SportsDataIO free trials scramble data. Enable SPORTSDATAIO_PRODUCTION only for a licensed real-data subscription.')
    dates = set(pool.game_date)
    seasons = {int(date[:4]) - (1 if int(date[5:7]) < 3 else 0) for date in dates}
    games = []
    for season in seasons:
        schedule = request(f'https://api.sportsdata.io/v3/nfl/scores/json/Schedules/{season}', key=key, expect_json=True)
        if not isinstance(schedule, list):
            raise ProviderError('Unexpected SportsDataIO schedule format.')
        games.extend(g for g in schedule if str(g.get('Date', ''))[:10] in dates)
    groups = {(g['Season'], g['SeasonType'], g['Week']) for g in games}
    if not groups:
        raise ProviderError('SportsDataIO could not match the game dates to an NFL week.')
    records = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for season, season_type, week in groups:
        suffix = {1: 'REG', 2: 'PRE', 3: 'POST'}.get(season_type)
        if not suffix:
            raise ProviderError('Unknown NFL season type.')
        for endpoint, defense in [('PlayerGameProjectionStatsByWeek', False), ('FantasyDefenseProjectionsByGame', True)]:
            url = f'https://api.sportsdata.io/v3/nfl/projections/json/{endpoint}/{season}{suffix}/{week}'
            payload = request(url, key=key, expect_json=True)
            if not isinstance(payload, list):
                raise ProviderError('Unexpected SportsDataIO projection format.')
            for p in payload:
                date = str(p.get('GameDate') or p.get('Date') or '')[:10]
                if date not in dates:
                    continue
                if p.get('IsScrambled') or p.get('IsTest'):
                    raise ProviderError('SportsDataIO returned test/scrambled data; optimization is blocked.')
                source_updated = p.get('Updated') or p.get('LastUpdated')
                freshness_verified = False
                if source_updated:
                    try:
                        timestamp = datetime.fromisoformat(str(source_updated).replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        freshness_verified = timestamp.tzinfo is not None and timedelta(minutes=-10) <= now - timestamp <= timedelta(hours=24)
                    except ValueError:
                        pass
                records.append(dict(player_name=p.get('Name') or p.get('Team'), position='DST' if defense else p.get('Position'),
                                    team=normalize_team(p.get('Team')), opponent=normalize_team(p.get('Opponent')),
                                    game_date=date, projection=numeric(p.get('FantasyPointsDraftKings')),
                                    injury_status=str(p.get('InjuryStatus') or 'Not supplied').upper(), ownership=np.nan,
                                    spread=np.nan, game_total=np.nan, implied_team_total=np.nan,
                                    source='SportsDataIO', source_url=url,
                                    source_updated=source_updated or 'Not supplied by provider',
                                    freshness_verified=freshness_verified, fetched_at=fetched_at))
    if not records:
        raise ProviderError('No SportsDataIO forecasts matched the pool dates.')
    return pd.DataFrame(records)


def match_projections(pool, forecasts):
    """Conservative exact identity joins. DST identity is its team, never a guessed name."""
    forecast_fields = ['projection', 'exclusion_reason', 'injury_status', 'source', 'source_url', 'source_updated',
                       'fetched_at', 'ownership', 'spread', 'game_total', 'implied_team_total']
    data = pool.drop(columns=forecast_fields, errors='ignore').copy()
    forecasts = pd.DataFrame(forecasts).copy()
    def key(row):
        name = normalize_team(row['team']) if row['position'] == 'DST' else normalize_name(row['player_name'])
        return (name, normalize_team(row['team']), row['position'], normalize_team(row['opponent']), str(row['game_date']))
    lookup = {}
    for _, row in forecasts.iterrows():
        lookup.setdefault(key(row), []).append(row)
    values = []
    for _, player in data.iterrows():
        matches = lookup.get(key(player), [])
        if len(matches) > 1:
            # Repeated identical page rows are harmless; conflicting rows are not.
            unique = pd.DataFrame(matches).drop_duplicates(subset=['projection', 'injury_status', 'source'])
            if len(unique) != 1:
                matches = []
                reason = 'Conflicting source records'
            else:
                matches = [unique.iloc[0]]
        else:
            reason = 'No exact player/team/opponent/date forecast'
        if not matches:
            values.append(dict(projection=np.nan, exclusion_reason=reason, injury_status='Unknown', source='Unavailable'))
            continue
        match = matches[0]
        payload = {c: match.get(c, np.nan) for c in ['projection', 'injury_status', 'ownership', 'spread', 'game_total',
                                                   'implied_team_total', 'source', 'source_url', 'source_updated', 'fetched_at']}
        status = str(payload['injury_status']).upper()
        if match.get('freshness_verified', True) is not True and not bool(match.get('freshness_verified', False)):
            reason = 'Source freshness cannot be verified (missing timezone/update time or stale data)'
        elif status in BAD_STATUS:
            reason = f'Unavailable: {status}'
        elif not np.isfinite(payload['projection']) or payload['projection'] <= 0:
            reason = 'No positive published forecast'
        else:
            reason = ''
        payload['exclusion_reason'] = reason
        values.append(payload)
    # Pool fields and prices always remain authoritative.
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(values)], axis=1)


def contest_id(value):
    value = value.strip()
    if value.isdigit():
        return value
    parsed = urlparse(value)
    if parsed.hostname not in {'www.draftkings.com', 'draftkings.com'}:
        raise ValueError('Enter a DraftKings contest URL or numeric contest ID.')
    match = re.search(r'/contest/(\d+)', parsed.path)
    if not match:
        raise ValueError('Contest ID was not found in this DraftKings URL.')
    return match.group(1)


def fetch_contest(value):
    cid = contest_id(value)
    payload = request(f'https://api.draftkings.com/contests/v1/contests/{cid}?format=json', expect_json=True)
    if not isinstance(payload, dict):
        raise ProviderError('Unexpected DraftKings contest response: expected an object.')
    contest = payload.get('contestDetail') or payload.get('contest') or payload
    if not isinstance(contest, dict):
        raise ProviderError('Unexpected DraftKings contest response.')
    nested_group = contest.get('draftGroup')
    draft_group = contest.get('draftGroupId') or (nested_group.get('draftGroupId') if isinstance(nested_group, dict) else None)
    if draft_group is None:
        raise ProviderError('DraftKings did not return a verifiable draft group for this contest.')
    return dict(contest_id=cid, name=contest.get('name') or contest.get('contestName') or f'Contest {cid}',
                draft_group_id=draft_group, entry_fee=contest.get('entryFee'),
                field_size=contest.get('maximumEntries') or contest.get('entriesMax'), raw=contest)


def load_projection_file(source, pool):
    """ID-matched user projections are an explicit source, never a salary-average fallback."""
    try:
        from .dk_csv import read_text
    except ImportError:
        from dk_csv import read_text
    import io
    frame = pd.read_csv(io.StringIO(read_text(source)), dtype={'player_id': str, 'ID': str})
    frame = frame.rename(columns={'ID': 'player_id', 'Projection': 'projection'})
    if not {'player_id', 'projection'}.issubset(frame):
        raise ValueError('Projection file needs player_id (or ID) and projection columns. Optional: ownership, projection_ceiling, projection_std, injury_status.')
    if frame.player_id.duplicated().any():
        raise ValueError('Projection file has duplicate player IDs.')
    if not set(frame.player_id).issubset(set(pool.player_id)):
        raise ValueError('Projection file contains IDs outside the uploaded DraftKings pool.')
    allowed = ['player_id', 'projection', 'ownership', 'injury_status', 'projection_ceiling', 'projection_std', 'source_updated']
    frame = frame[[col for col in allowed if col in frame]].copy()
    for col in ['projection', 'ownership', 'projection_ceiling', 'projection_std']:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
            if not np.isfinite(frame[col]).all():
                raise ValueError(f'{col} values must be finite.')
    if 'ownership' in frame and not frame.ownership.between(0, 100).all():
        raise ValueError('ownership must be a percentage between 0 and 100.')
    if 'projection_std' in frame and (frame.projection_std < 0).any():
        raise ValueError('projection_std must not be negative.')
    frame['source'] = 'Uploaded pregame projections'
    frame['fetched_at'] = datetime.now(timezone.utc).isoformat()
    if 'injury_status' not in frame:
        frame['injury_status'] = 'Not supplied'
    data = pool.merge(frame, on='player_id', how='left', validate='one_to_one')
    data['exclusion_reason'] = ''
    data.loc[data.projection.isna() | data.projection.le(0), 'exclusion_reason'] = 'No positive uploaded forecast'
    unavailable = data.injury_status.fillna('Unknown').str.upper().isin(BAD_STATUS)
    data.loc[unavailable, 'exclusion_reason'] = 'Unavailable according to uploaded injury status'
    return data


def verify_contest_pool(contest, pool):
    group = int(contest['draft_group_id'])
    payload = request(f'https://api.draftkings.com/draftgroups/v1/draftgroups/{group}/draftables?format=json', expect_json=True)
    entries = payload.get('draftables')
    if not isinstance(entries, list) or not entries:
        raise ProviderError('Unable to verify contest player IDs against its draft group.')
    ids = {str(p['draftableId']) for p in entries if 'draftableId' in p}
    wrong = set(pool.player_id) - ids
    if wrong:
        raise ValueError(f'{len(wrong)} pool IDs do not belong to this contest. Download its exact player pool CSV.')
    prices = {str(p['draftableId']): p.get('salary') for p in entries if 'draftableId' in p}
    for _, row in pool.iterrows():
        if prices.get(row.player_id) is not None and numeric(prices[row.player_id]) != row.salary:
            raise ValueError('Uploaded salaries differ from the contest draft group. Download a fresh contest pool CSV.')
    return True
