import io
import sys
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from optimizer.freshness import frame_revision,automatic_stale,portfolio_is_current
from optimizer.free_model import (prepare_history,generate_forecasts,fetch_free,pool_season,schedule_times,load_status,ProviderError)
from optimizer.forecast_features import MatchupContext
from optimizer.forecast_audit import score_snapshot
from optimizer.lineup_engine import eligible_pool,solve_lineup,build_portfolio
from test_free_model import fixture
from test_draftkings import classic_pool,classic_csv

ROOT=Path(__file__).resolve().parents[1]


def ui_module():
    module_path=str(ROOT/'optimizer')
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
    return importlib.import_module('streamlit_ui')


def artifacts(data,as_of):
    history,_,games=prepare_history(data,as_of)
    models={pos:dict(method='recent_mean',residuals=[-1,0,1],residual_bounds=[5,15],residual_bins={}) for pos in history.position.unique()}
    return dict(history=history,games=games,models=models,metrics=pd.DataFrame(),as_of=as_of)


class ReliabilityTests(unittest.TestCase):
    def test_full_revision_detects_changes_outside_typical_samples(self):
        frame=pd.DataFrame({'value':np.arange(50000)})
        first=frame_revision(frame);frame.loc[39877,'value']=-1
        self.assertNotEqual(first,frame_revision(frame))

    def test_automatic_snapshot_age_and_timezone_are_checked_on_use(self):
        now=pd.Timestamp('2026-10-13T12:00Z')
        row=dict(source='Our free nflverse model',fetched_at=now,model_built_at=now)
        self.assertEqual(automatic_stale(row,now),'')
        row['fetched_at']=now-pd.Timedelta(minutes=16)
        self.assertIn('expired',automatic_stale(row,now))
        row['fetched_at']='2026-10-13 12:00'
        self.assertIn('unverified',automatic_stale(row,now))
        row['fetched_at']=now;row['model_built_at']=now-pd.Timedelta(hours=3)
        self.assertIn('Fitted model',automatic_stale(row,now))
        row.update(source='Daily Fantasy Fuel',source_updated=now-pd.Timedelta(hours=25))
        self.assertIn('stale',automatic_stale(row,now))

    def test_expired_players_remove_existing_portfolio_eligibility(self):
        pool=classic_pool();now=pd.Timestamp.now(tz='UTC')
        pool['source']='Our free nflverse model';pool['fetched_at']=now.isoformat();pool['model_built_at']=now.isoformat()
        lineup=solve_lineup(pool)
        available,_=eligible_pool(pool,now)
        self.assertTrue(portfolio_is_current([lineup],available))
        available,excluded=eligible_pool(pool,now+pd.Timedelta(minutes=16))
        self.assertFalse(portfolio_is_current([lineup],available));self.assertEqual(len(excluded),len(pool))

    def test_lock_and_exclusion_conflict_is_actionable(self):
        pool=classic_pool();key=pool.player_key.iloc[0]
        with self.assertRaisesRegex(ValueError,'both locked and excluded'):
            build_portfolio(pool,locked_keys=[key],forbidden_keys=[key])

    def test_status_refresh_reuses_history_and_fitted_model(self):
        ui=ui_module();pool=classic_pool();pool['game_date']='2026-10-18'
        game=pd.DataFrame([dict(game_id='past',gameday='2026-09-01',gametime='13:00',away_team='A',home_team='B',season=2026,week=1,game_type='REG',home_score=7,away_score=7)])
        history=dict(players=pd.DataFrame({'id':[1]}),teams=pd.DataFrame({'id':[2]}),history_revision='all-rows-revision')
        status=dict(games=game,roster=pd.DataFrame())
        ui.automatic_data.clear();ui.free_history.clear();ui.free_status.clear();ui.free_model_artifacts.clear()
        with patch.object(ui,'load_history',return_value=history) as loadh,patch.object(ui,'load_status',return_value=status) as loads,patch.object(ui,'prepare_model',return_value={'sentinel':True}) as fit,patch.object(ui,'fetch_free',return_value=(pool,[])):
            ui.automatic_data(pool,'Free automatic')
            ui.automatic_data.clear(pool,'Free automatic');ui.free_status.clear(2026)
            second=pool.copy();second.loc[0,'salary']=5000
            ui.automatic_data(second,'Free automatic')
            self.assertEqual(loadh.call_count,1);self.assertEqual(loads.call_count,2);self.assertEqual(fit.call_count,1)
        ui.automatic_data.clear();ui.free_history.clear();ui.free_status.clear();ui.free_model_artifacts.clear()

    def test_fitted_model_uses_current_schedule_even_with_old_artifacts(self):
        pool,data,as_of=fixture();prepared=artifacts(data,as_of)
        data['games'].loc[data['games'].game_id.eq('game-100'),'gametime']='14:00'
        forecasts,_,missing,_=generate_forecasts(pool,data,as_of,prepared)
        self.assertTrue(forecasts.empty);self.assertTrue(all('kickoff differs' in reason for reason in missing.values()))

    def test_cached_narrow_context_preserves_forecast_values(self):
        pool,data,as_of=fixture();prepared=artifacts(data,as_of)
        first,_,_,_=generate_forecasts(pool,data,as_of,prepared)
        prepared['context']=MatchupContext(prepared['history'],prepared['games'],as_of)
        prepared['history_groups']={(pid,pos):group.sort_values('kickoff_time') for (pid,pos),group in prepared['history'].groupby(['player_id','position'])}
        second,_,_,_=generate_forecasts(pool,data,as_of,prepared)
        pd.testing.assert_frame_equal(first,second)
        self.assertLessEqual(len(prepared['history'].columns),20)

    def test_started_games_do_not_receive_retrospective_model_forecasts(self):
        pool,data,as_of=fixture();prepared=artifacts(data,as_of)
        forecasts,_,missing,_=generate_forecasts(pool,data,as_of+pd.Timedelta(days=1),prepared)
        self.assertTrue(forecasts.empty);self.assertTrue(all('Game has started' in reason for reason in missing.values()))

    def test_venue_metadata_after_snapshot_cannot_enter_forecast_features(self):
        _,data,as_of=fixture();as_of=as_of-pd.Timedelta(days=10)
        data['games']['roof']='outdoors'
        history,_,games=prepare_history(data,as_of)
        target=schedule_times(data['games']).iloc[-1]
        first=MatchupContext(history,games,as_of).team_vector(target.kickoff_time,'A','B','QB',target.game_id)
        data['games'].loc[data['games'].game_id.eq('game-99'),'roof']='closed'
        updated=schedule_times(data['games'])
        second=MatchupContext(history,updated,as_of).team_vector(target.kickoff_time,'A','B','QB',target.game_id)
        np.testing.assert_array_equal(first,second)

    def test_positive_public_forecast_cannot_override_known_inactive_roster(self):
        pool,data,as_of=fixture();prepared=artifacts(data,as_of)
        data['roster'].loc[data['roster'].gsis_id.eq('QB0'),'status']='RES'
        first=pool.iloc[0]
        public=pd.DataFrame([dict(player_name=first.player_name,position=first.position,team=first.team,opponent=first.opponent,
                                  game_date=first.game_date,projection=99,injury_status='',source='Daily Fantasy Fuel')])
        with patch('optimizer.free_model.fetch_public',return_value=(public,[])):
            forecast,_=fetch_free(pool,data,as_of,prepared)
        self.assertIn('does not report active',forecast.exclusion_reason.iloc[0])

    def test_exact_nfl_identity_is_retained_for_expert_forecasts_without_own_history(self):
        pool,data,as_of=fixture()
        data['players']=data['players'][data['players'].player_id.ne('QB0')]
        prepared=artifacts(data,as_of);first=pool.iloc[0]
        public=pd.DataFrame([dict(player_name=first.player_name,position=first.position,team=first.team,opponent=first.opponent,
                                  game_date=first.game_date,projection=20,injury_status='',source='Daily Fantasy Fuel')])
        with patch('optimizer.free_model.fetch_public',return_value=(public,[])):
            forecast,_=fetch_free(pool,data,as_of,prepared)
        self.assertEqual(forecast.historical_player_id.iloc[0],'QB0')
        self.assertTrue(pd.isna(forecast.projection_std.iloc[0]))

    def test_stale_roster_publisher_cannot_generate_own_estimates(self):
        pool,data,as_of=fixture();prepared=artifacts(data,as_of);data['roster_freshness_verified']=False
        forecasts,_,missing,_=generate_forecasts(pool,data,as_of,prepared)
        self.assertTrue(forecasts.empty);self.assertTrue(all('Roster publisher' in r for r in missing.values()))

    def test_uploaded_projections_survive_rerun_without_restamping(self):
        ui=ui_module();pool=classic_pool();upload=io.BytesIO(classic_csv().encode());upload.name='pool.csv'
        projections=io.BytesIO(pool[['player_id','projection']].to_csv(index=False).encode());projections.name='forecasts.csv'
        def uploader(label,**kwargs):
            return upload if label=='Player pool CSV' else projections if label=='Projection file (for Upload projections)' else None
        with patch('streamlit.file_uploader',side_effect=uploader),patch.object(ui,'automatic_data',return_value=(pool,[])):
            app=AppTest.from_file(str(ROOT/'optimizer/optimizer.py'),default_timeout=20).run()
            app.selectbox[2].select('Upload projections').run();app.button[-1].click().run()
            self.assertEqual(len(app.exception),0);self.assertEqual(len(app.session_state['portfolio']),3)
            stamp=app.session_state['projection_upload'][1].fetched_at.iloc[0]
            app.run()
            self.assertEqual(len(app.session_state['portfolio']),3)
            self.assertEqual(stamp,app.session_state['projection_upload'][1].fetched_at.iloc[0])

    def test_manual_forecasts_survive_lineup_count_change(self):
        ui=ui_module();pool=classic_pool();upload=io.BytesIO(classic_csv().encode());upload.name='pool.csv'
        with patch('streamlit.file_uploader',side_effect=lambda label,**kw:upload if label=='Player pool CSV' else None),patch.object(ui,'automatic_data',return_value=(pool,[])) as fetch:
            app=AppTest.from_file(str(ROOT/'optimizer/optimizer.py'),default_timeout=20).run()
            app.checkbox[0].uncheck().run()
            next(n for n in app.number_input if n.label=='Lineups').set_value(2).run()
            app.button[-1].click().run()
            self.assertEqual(len(app.error),0);self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state['portfolio']),2);self.assertEqual(fetch.call_count,1)


class SavedForecastTests(unittest.TestCase):
    def setUp(self):
        _,self.data,self.now=fixture();game=self.data['games'].iloc[90]
        kickoff=pd.Timestamp(game.gameday+' 13:00',tz='America/New_York')
        self.snapshot=pd.DataFrame([dict(player_id='100',historical_player_id='QB0',position='QB',player_name='QB Player 0',team='B',opponent='A',
                                         game_date=game.gameday,kickoff=kickoff.isoformat(),projection=15,fetched_at=(kickoff-pd.Timedelta(hours=1)).isoformat(),exclusion_reason='')])

    def test_exact_saved_forecast_scores_once_per_athlete_and_is_calibration_ready(self):
        second=self.snapshot.copy();second['player_id']='101'
        scored,metrics,skipped=score_snapshot(pd.concat([self.snapshot,second]),self.data,self.now)
        self.assertEqual(len(scored),1);self.assertTrue(skipped.empty)
        self.assertEqual(scored.actual.iloc[0],12);self.assertEqual(metrics.mae.iloc[0],3)
        self.assertLess(pd.Timestamp(scored.forecast_at.iloc[0]),pd.Timestamp(scored.game_date.iloc[0]))

    def test_postgame_snapshots_and_wrong_identity_are_not_scored(self):
        wrong=self.snapshot.copy();wrong['historical_player_id']='unknown'
        scored,_,skipped=score_snapshot(wrong,self.data,self.now)
        self.assertTrue(scored.empty);self.assertIn('not scored as zero',skipped.reason.iloc[0])
        wrong=self.snapshot.copy();wrong['fetched_at']=wrong.kickoff
        scored,_,skipped=score_snapshot(wrong,self.data,self.now)
        self.assertTrue(scored.empty);self.assertIn('at/after kickoff',skipped.reason.iloc[0])

    def test_conflicting_captain_flex_base_projections_are_rejected(self):
        second=self.snapshot.copy();second['player_id']='101';second['projection']=99
        with self.assertRaisesRegex(ValueError,'Conflicting base forecasts'):
            score_snapshot(pd.concat([self.snapshot,second]),self.data,self.now)

    def test_started_or_incomplete_games_cannot_be_scored(self):
        self.data['games'].loc[self.data['games'].game_id.eq('game-90'),'home_score']=np.nan
        scored,_,skipped=score_snapshot(self.snapshot,self.data,self.now)
        self.assertTrue(scored.empty);self.assertIn('completed schedule',skipped.reason.iloc[0])


if __name__=='__main__':unittest.main()
