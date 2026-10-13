"""Score user-saved pregame snapshots against exact observed NFL box scores."""
import numpy as np
import pandas as pd
try:
    from .freshness import aware_time
    from .dk_csv import normalize_team
    from .free_model import schedule_times, player_scores, defense_scores, require
except ImportError:
    from freshness import aware_time
    from dk_csv import normalize_team
    from free_model import schedule_times, player_scores, defense_scores, require


def score_snapshot(snapshot, datasets, as_of=None):
    now = pd.Timestamp(as_of or pd.Timestamp.now(tz='UTC'))
    if now.tzinfo is None:
        raise ValueError('Evaluation cutoff must be timezone-aware.')
    required = ['player_id','position','team','opponent','kickoff','game_date','projection','fetched_at','historical_player_id']
    require(snapshot,required,'Saved forecast audit')
    data = snapshot.copy()
    for column in ['team','opponent']:
        data[column] = data[column].map(normalize_team)
    data['projection'] = pd.to_numeric(data.projection,errors='raise')
    data['kickoff_time'] = data.kickoff.map(aware_time)
    data['recorded_at'] = data.fetched_at.map(aware_time)
    # Each input is a snapshot, not a post-hoc forecast fitted to the scored results.
    games = schedule_times(datasets['games'])
    final_games = games[games.home_score.notna() & games.away_score.notna() & (games.kickoff_time < now-pd.Timedelta(hours=6))]
    players = datasets['players'].copy()
    players['team'] = players.team.map(normalize_team)
    players['actual'] = player_scores(players)
    if players.duplicated(['player_id','game_id']).any():
        raise ValueError('Duplicate player/game actuals cannot be scored reliably.')
    teams = datasets['teams'].copy()
    for col in ['team','opponent_team']:
        teams[col] = teams[col].map(normalize_team)
    defense = defense_scores(teams,final_games).rename(columns={'dk_score':'actual'})
    actuals = pd.concat([players[['player_id','game_id','position','team','actual']],
                         defense[['player_id','game_id','position','team','actual']]],ignore_index=True)
    lookup = actuals.set_index(['player_id','game_id','position','team']).actual.to_dict()
    results,skipped = [],[]
    # CPT and FLEX rows have distinct DK IDs but share an athlete's base forecast.
    seen = {}
    for row in data.itertuples():
        reason = ''
        if row.kickoff_time is None or row.recorded_at is None:
            reason = 'Missing timezone-aware pregame timestamps'
        elif row.recorded_at >= row.kickoff_time:
            reason = 'Snapshot was recorded at/after kickoff'
        elif not np.isfinite(row.projection) or row.projection <= 0:
            reason = 'No positive saved projection'
        elif str(getattr(row,'exclusion_reason','') or '') not in {'','nan'}:
            reason = 'Player was excluded in the saved snapshot'
        elif pd.isna(row.historical_player_id) or not str(row.historical_player_id).strip():
            reason = 'No exact historical NFL player identity in snapshot'
        if not reason:
            game = final_games[(final_games.gameday.astype(str)==str(row.game_date)) &
                               (((final_games.home_team==row.team)&(final_games.away_team==row.opponent)) |
                                ((final_games.away_team==row.team)&(final_games.home_team==row.opponent)))]
            if len(game)!=1 or not game.iloc[0].kickoff_verified:
                reason = 'No exact completed schedule match'
            elif abs((game.iloc[0].kickoff_time-row.kickoff_time).total_seconds())>60:
                reason = 'Saved kickoff differs from NFL schedule'
        if not reason:
            value = lookup.get((str(row.historical_player_id),game.iloc[0].game_id,row.position,row.team))
            if value is None or not np.isfinite(value):
                reason = 'No observed final box score; absence is not scored as zero'
        if reason:
            skipped.append(dict(player_id=row.player_id,player=getattr(row,'player_name',row.historical_player_id),reason=reason))
            continue
        key = (str(row.historical_player_id),row.position,row.team,row.kickoff_time)
        if key in seen:
            if not np.isclose(seen[key],row.projection,rtol=0,atol=1e-8):
                raise ValueError('Conflicting base forecasts for the same athlete/game. Upload one pregame snapshot.')
            continue
        seen[key] = row.projection
        results.append(dict(player_id=str(row.player_id),historical_player_id=row.historical_player_id,
                            player=getattr(row,'player_name',row.historical_player_id),position=row.position,
                            team=row.team,opponent=row.opponent,game_date=row.kickoff_time.isoformat(),
                            forecast_at=row.recorded_at.isoformat(),projection=row.projection,actual=float(value),
                            error=float(value-row.projection),source=getattr(row,'source','User-saved snapshot')))
    scored = pd.DataFrame(results)
    metrics = []
    if not scored.empty:
        for pos,group in scored.groupby('position'):
            errors=group.error.to_numpy()
            metrics.append(dict(position=pos,observed_games=len(group),mae=float(np.abs(errors).mean()),
                                rmse=float(np.sqrt(np.mean(errors**2))),bias=float(errors.mean())))
    return scored,pd.DataFrame(metrics),pd.DataFrame(skipped)
