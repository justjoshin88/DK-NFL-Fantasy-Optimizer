"""Snapshot clocks and deterministic fingerprints for public-data caches."""
import hashlib
import pandas as pd

AUTOMATIC_SOURCES = {'Our free nflverse model', 'Daily Fantasy Fuel', 'SportsDataIO'}


def aware_time(value):
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert('UTC') if stamp.tzinfo is not None and not pd.isna(stamp) else None
    except (ValueError, TypeError):
        return None


def recent_time(value, now, seconds):
    stamp = aware_time(value)
    return stamp is not None and pd.Timedelta(0) <= pd.Timestamp(now)-stamp <= pd.Timedelta(seconds=seconds)


def frame_revision(frame):
    """Hash every row, including schema, rather than a sample of a large DataFrame."""
    digest = hashlib.sha256()
    digest.update(str([(str(c), str(t)) for c,t in zip(frame.columns,frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def history_revision(datasets):
    return hashlib.sha256(''.join(frame_revision(datasets[k]) for k in ['players','teams']).encode()).hexdigest()


def automatic_stale(row, now):
    if row.get('source') not in AUTOMATIC_SOURCES:
        return ''
    if not recent_time(row.get('fetched_at'), now, 15*60):
        return 'Automatic snapshot expired or retrieval time is unverified; refresh projections'
    if row.get('source') != 'Our free nflverse model' and not recent_time(row.get('source_updated'), now, 24*3600):
        return 'Published forecast is stale or its update time is unverified'
    if row.get('source') == 'Our free nflverse model' and not recent_time(row.get('model_built_at'), now, 2*3600):
        return 'Fitted model has expired; refresh NFL history and retrain'
    return ''


def portfolio_is_current(lineups, available):
    ids = set(available.player_id.astype(str))
    return all(set(lineup.player_id.astype(str)).issubset(ids) for lineup in lineups)
