"""DraftKings input and output contracts. Never substitute historic averages for forecasts."""
import csv
import io
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CLASSIC = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
SHOWDOWN = ['CPT', 'FLEX', 'FLEX', 'FLEX', 'FLEX', 'FLEX']
TEAM_ALIASES = {'JAC': 'JAX', 'WSH': 'WAS', 'LA': 'LAR', 'SD': 'LAC', 'OAK': 'LV', 'AZ': 'ARI'}


def normalize_name(value):
    name = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]', '', name)


def normalize_team(value):
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def read_text(source):
    if isinstance(source, str):
        return source.lstrip('\ufeff')
    value = source.getvalue() if hasattr(source, 'getvalue') else source.read()
    return value.decode('utf-8-sig') if isinstance(value, bytes) else value.lstrip('\ufeff')


def parse_game(value, team):
    match = re.fullmatch(r'([A-Z]+)@([A-Z]+)\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}[AP]M)\s+ET', str(value).strip())
    if not match:
        raise ValueError(f'Unrecognized Game Info: {value}. Use an original DraftKings pool export.')
    away, home, date, time = match.groups()
    away, home = normalize_team(away), normalize_team(home)
    if team not in (away, home):
        raise ValueError(f'Team {team} does not match Game Info {value}.')
    kickoff = datetime.strptime(f'{date} {time}', '%m/%d/%Y %I:%M%p').replace(tzinfo=ZoneInfo('America/New_York'))
    return dict(opponent=home if team == away else away, game_id=f'{away}@{home}:{kickoff.date()}',
                game_date=str(kickoff.date()), kickoff=kickoff.isoformat())


def load_pool(source, game_type='Classic'):
    rows = list(csv.reader(io.StringIO(read_text(source))))
    header_index = next((i for i, row in enumerate(rows) if {'Position', 'Name', 'ID', 'Salary', 'Game Info', 'TeamAbbrev'}.issubset(row)), None)
    if header_index is None:
        if rows and rows[0][:9] == CLASSIC:
            raise ValueError('This is a roster upload template. Upload the player pool CSV under Player pool and this file under Roster template.')
        raise ValueError('Player pool must contain Position, Name, ID, Roster Position, Salary, Game Info, and TeamAbbrev.')
    header = rows[header_index]
    records = []
    for row in rows[header_index + 1:]:
        if not row or not any(x.strip() for x in row):
            continue
        raw = dict(zip(header, row))
        if not str(raw.get('ID', '')).strip().isdigit():
            continue  # Instructions / blank lines following an embedded player pool.
        position = raw['Position'].strip().upper().replace('D/ST', 'DST')
        team = normalize_team(raw['TeamAbbrev'])
        name = raw['Name'].strip()
        salary = float(raw['Salary'])
        if not name or not team or not np.isfinite(salary) or salary <= 0 or salary % 1:
            raise ValueError('Player names, teams, and positive whole dollar salaries are required.')
        if position not in {'QB', 'RB', 'WR', 'TE', 'DST', 'K'}:
            raise ValueError(f'Unsupported position: {position}. NFL Classic and Showdown only.')
        eligibility = str(raw.get('Roster Position', position)).strip().upper().split('/')
        if game_type == 'Showdown':
            if not set(eligibility).intersection({'CPT', 'FLEX'}):
                raise ValueError('Showdown requires its own CPT/FLEX player pool CSV. Classic IDs cannot be reused.')
            eligibility = [s for s in eligibility if s in {'CPT', 'FLEX'}]
        else:
            if 'CPT' in eligibility or position == 'K':
                raise ValueError('This pool is not NFL Classic. Choose the matching game type.')
            eligibility = [position] + (['FLEX'] if position in {'RB', 'WR', 'TE'} else [])
        record = dict(player_name=name, player_id=raw['ID'].strip(), position=position, salary=int(salary),
                      team=team, player_key=f'{normalize_name(name)}:{team}:{position}',
                      eligible_slots=eligibility, avg_points_per_game=raw.get('AvgPointsPerGame', ''),
                      **parse_game(raw['Game Info'], team))
        records.append(record)
    if not records:
        raise ValueError('No players found in the pool CSV.')
    data = pd.DataFrame(records)
    if data.player_id.duplicated().any():
        raise ValueError('Duplicate DraftKings player IDs in the pool.')
    if game_type == 'Classic' and data.player_key.duplicated().any():
        raise ValueError('Duplicate players in Classic pool.')
    if game_type == 'Showdown' and data.game_id.nunique() != 1:
        raise ValueError('Showdown must contain exactly one NFL game.')
    if data.game_date.nunique() > 7:
        raise ValueError('Pool spans more than seven game dates. Upload a single contest pool.')
    return data


def roster_template(source, game_type):
    expected = CLASSIC if game_type == 'Classic' else SHOWDOWN
    if source is None:
        return [expected]
    rows = list(csv.reader(io.StringIO(read_text(source))))
    if not rows or rows[0][:len(expected)] != expected:
        raise ValueError(f'Roster template must start with {",".join(expected)}. Repeated columns must be preserved.')
    if any(any(c.strip() for c in row[:len(expected)]) for row in rows[1:]):
        raise ValueError('Use a blank roster template so existing entries are not overwritten.')
    return rows


def export_rosters(lineups, game_type, template=None):
    rows = roster_template(template, game_type)
    expected = CLASSIC if game_type == 'Classic' else SHOWDOWN
    if not 1 <= len(lineups) <= 500:
        raise ValueError('DraftKings roster files accept 1 to 500 lineups.')
    width = len(rows[0])
    output = [rows[0]]
    for lineup in lineups:
        if list(lineup.slot) != expected:
            raise ValueError('Lineup slots do not match the roster template.')
        ids = lineup.player_id.astype(str).tolist()
        if any(not x.isdigit() for x in ids):
            raise ValueError('DraftKings export requires numeric player IDs.')
        output.append(ids + [''] * (width - len(ids)))
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\r\n').writerows(output)
    return stream.getvalue().encode('utf-8-sig')
