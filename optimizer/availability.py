"""Exact current depth and injury checks; missing reports stay explicitly unknown."""
import numpy as np
import pandas as pd
try:
    from .dk_csv import normalize_team
except ImportError:
    from dk_csv import normalize_team


def depth_snapshot(frame, as_of):
    columns = {'dt', 'team', 'gsis_id', 'pos_abb', 'pos_rank'}
    if frame is None or not columns.issubset(frame):
        return pd.DataFrame(), 'Current depth chart unavailable or schema unsupported'
    data = frame.copy()
    data['updated'] = pd.to_datetime(data.dt, utc=True, errors='coerce')
    data = data[data.updated.notna() & (data.updated <= as_of)]
    if data.empty:
        return data, 'Depth chart has no snapshot preceding the forecast'
    data['team'] = data.team.map(normalize_team)
    # A complete team snapshot, rather than the most recent row per athlete, removes obsolete entries.
    latest = data.groupby('team').updated.transform('max')
    data = data[data.updated.eq(latest) & ((as_of-data.updated) <= pd.Timedelta(hours=48))]
    return data, 'Depth chart only verifies listed rank, not final game-day participation'


def player_availability(player_id, team, position, game, datasets, as_of, depth=None):
    audit = dict(depth_rank=np.nan, depth_updated='', injury_report_status='Unknown',
                 injury_report_week=np.nan, availability_note='Game-day participation unverified')
    reason = ''
    if position != 'DST':
        matching = depth[(depth.gsis_id == player_id) & (depth.team == team) & (depth.pos_abb == position)] if depth is not None and not depth.empty else pd.DataFrame()
        if not matching.empty:
            rank = pd.to_numeric(matching.pos_rank, errors='coerce').min()
            audit.update(depth_rank=rank, depth_updated=str(matching.updated.max()))
            if position in {'QB', 'K'} and rank > 1:
                reason = 'Current depth chart lists backup QB/K; starting role not verified'
            audit['availability_note'] = 'Current depth rank verified; final participation unverified'
        elif position in {'QB', 'K'}:
            audit['availability_note'] = 'Starting QB/K role unknown; depth source does not verify it'
    injuries = datasets.get('injuries')
    updated = datasets.get('injuries_updated')
    required = {'season', 'week', 'team', 'gsis_id', 'report_status'}
    if injuries is not None and required.issubset(injuries) and updated is not None:
        updated = pd.Timestamp(updated)
        if updated.tzinfo is not None and pd.Timedelta(0) <= as_of-updated <= pd.Timedelta(hours=36):
            matching = injuries[(injuries.gsis_id == player_id) & (injuries.team.map(normalize_team) == team) &
                                (injuries.season == game.season) & (injuries.week == game.week)]
            if 'season_type' in matching:
                matching = matching[matching.season_type == game.game_type]
            statuses = matching.report_status.dropna().astype(str).str.upper().unique()
            if len(statuses) == 1:
                status = statuses[0]
                audit.update(injury_report_status=status, injury_report_week=game.week,
                             injury_report_updated=str(updated), availability_note='Exact-week injury report; final participation unverified')
                if status in {'OUT', 'DOUBTFUL', 'INACTIVE', 'IR', 'SUSPENDED'}:
                    reason = 'Unavailable on exact-week injury report: '+status
            elif len(statuses) > 1:
                reason = 'Conflicting exact-week injury report statuses'
    return reason, audit
