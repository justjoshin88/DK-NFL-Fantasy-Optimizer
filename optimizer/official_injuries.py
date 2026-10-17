"""Free, read-only official injury reports; exact schedule and identity matching.

NFL pages expose current designations but no publisher update clock. Observation
time is recorded separately and is never presented as publication time.
"""
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from bs4 import BeautifulSoup

try:
    from .dk_csv import normalize_name, normalize_team
    from .providers import ProviderError, request
    from .freshness import recent_time
except ImportError:
    from dk_csv import normalize_name, normalize_team
    from providers import ProviderError, request
    from freshness import recent_time


def parse_official(html, games, season, week, game_type, observed_at):
    url = f'https://www.nfl.com/injuries/league/{season}/{game_type.lower()}{week}'
    soup = BeautifulSoup(html, 'html.parser')
    canonical = soup.select_one('meta[property="og:url"]')
    if canonical is None or canonical.get('content', '').rstrip('/') != url:
        raise ProviderError('Official injury page season/week could not be verified.')
    heading = soup.select_one('h2.nfl-c-content-header__roofline')
    if heading is None or not re.search(rf'\bWEEK\s+{week}\b', heading.get_text(' ', strip=True), re.I):
        raise ProviderError('Official injury page week heading does not match its URL.')
    schedule = games[(games.season == season) & (games.week == week) & (games.game_type == game_type)]
    rows = []
    for section in soup.select('section.nfl-o-injury-report__unit'):
        teams = [normalize_team(t.get_text(strip=True)) for t in section.select('.nfl-c-matchup-strip__team-abbreviation')]
        tables = section.select('table')
        date_heading = section.find_previous('h2', class_='d3-o-section-title')
        date_text = date_heading.get_text(' ', strip=True).upper() if date_heading else ''
        date_match = re.fullmatch(r'[A-Z]+,\s+([A-Z]+)\s+(\d{1,2})(?:ST|ND|RD|TH)', date_text)
        if len(teams) != 2 or len(set(teams)) != 2 or len(tables) != 2 or date_match is None:
            continue  # Partial sections cannot safely assign a table to a club.
        game = schedule[(schedule.away_team == teams[0]) & (schedule.home_team == teams[1])]
        if len(game) != 1:
            continue
        date = pd.Timestamp(game.iloc[0].gameday)
        if date.strftime('%B').upper() != date_match[1] or date.day != int(date_match[2]):
            continue
        for team, opponent, table in [(teams[0], teams[1], tables[0]), (teams[1], teams[0], tables[1])]:
            headers = [cell.get_text(' ', strip=True) for cell in table.select('thead th')]
            if headers != ['Player', 'Position', 'Injuries', 'Practice Status', 'Game Status']:
                raise ProviderError('Official injury table schema changed; reports not applied.')
            for row in table.select('tbody tr'):
                cells = [cell.get_text(' ', strip=True) for cell in row.select('td')]
                if len(cells) != 5:
                    raise ProviderError('Incomplete official injury row; reports not applied.')
                name, position, injury, practice, status = cells
                if not name or position not in {'QB', 'RB', 'WR', 'TE', 'K'}:
                    continue
                rows.append(dict(player_name=name, name_key=normalize_name(name), team=team,
                                 opponent=opponent, position=position, game_date=str(date.date()),
                                 season=season, week=week, official_status=status.upper() or 'Not designated',
                                 official_practice=practice or 'Unknown', official_injury=injury,
                                 official_observed_at=str(observed_at), official_source_url=url))
    if not rows:
        raise ProviderError('Official injury page has no supported exact-schedule records.')
    return pd.DataFrame(rows)


def load_official(games, now):
    # Only upcoming games; no archive fetched now is represented as historical pregame evidence.
    future = games[(games.kickoff_time > now) & (games.kickoff_time < now+pd.Timedelta(days=8))]
    groups = future[['season', 'week', 'game_type']].drop_duplicates()
    def load(row):
        url = f'https://www.nfl.com/injuries/league/{row.season}/{row.game_type.lower()}{row.week}'
        try:
            result = parse_official(request(url), games, row.season, row.week, row.game_type, now.isoformat())
            return result, url, None
        except ProviderError as exc:
            return pd.DataFrame(), url, str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(load, groups.itertuples(index=False)))
    frames = [r[0] for r in results if not r[0].empty]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            {f'official_injuries_{i}': r[1] for i, r in enumerate(results)},
            [f'Official injury report: {r[2]}' for r in results if r[2]])


def apply_official(data, reports, now):
    """Negative designations exclude; absence or a blank designation never means healthy."""
    data = data.copy()
    audits = ['official_status', 'official_practice', 'official_injury', 'official_observed_at', 'official_source_url']
    for column in audits:
        data[column] = 'Unknown' if column in {'official_status', 'official_practice'} else ''
    if reports is None or reports.empty:
        return data
    keys = ['name_key', 'team', 'position', 'opponent', 'game_date']
    lookup = {key: group for key, group in reports.groupby(keys)}
    for i, player in data.iterrows():
        key = (normalize_name(player.player_name), player.team, player.position, player.opponent, player.game_date)
        matching = lookup.get(key)
        if matching is None:
            continue
        fresh = matching[matching.official_observed_at.map(lambda t: recent_time(t, now, 15*60))]
        if fresh.empty:
            continue
        # Conflicting duplicate records are excluded rather than arbitrarily picked.
        if fresh[audits].drop_duplicates().shape[0] != 1:
            data.at[i, 'exclusion_reason'] = 'Conflicting exact-game official injury reports'
            continue
        report = fresh.iloc[0]
        for column in audits:
            data.at[i, column] = report[column]
        status = report.official_status
        if status in {'OUT', 'DOUBTFUL', 'INACTIVE', 'IR', 'SUSPENDED'}:
            data.at[i, 'exclusion_reason'] = f'Unavailable on official NFL exact-game report: {status}'
        # Existing provider/roster exclusions are never cleared by another source.
    return data
