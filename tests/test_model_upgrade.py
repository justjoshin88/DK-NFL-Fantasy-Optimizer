import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from optimizer.availability import depth_snapshot, player_availability
from optimizer.scenarios import sample_outcomes, learn_dependence
from optimizer.free_model import prepare_history, generate_forecasts, MODEL_AUDIT
from optimizer.forecast_validation import train_models, residual_sample
from test_free_model import fixture


class ModelUpgradeTests(unittest.TestCase):
    def test_dst_safeties_excluded_but_conversion_returns_count_as_allowed(self):
        from optimizer.free_model import defense_scores
        from test_free_model import DEF_FIELDS
        games=pd.DataFrame([dict(game_id='g',home_team='A',away_team='B',home_score=7,away_score=2)])
        rows=[]
        for team, opponent in [('A','B'),('B','A')]:
            row=dict.fromkeys(DEF_FIELDS,0);row.update(team=team,opponent_team=opponent,game_id='g')
            if team=='B':row['def_safeties']=1
            rows.append(row)
        scores=defense_scores(pd.DataFrame(rows),games).set_index('team')
        self.assertEqual(scores.loc['A','dk_score'],10)  # Safety scored while A's offense was on field.
        rows[1]['def_safeties']=0;rows[1]['def_2pt_made']=1
        scores=defense_scores(pd.DataFrame(rows),games).set_index('team')
        self.assertEqual(scores.loc['A','dk_score'],7)  # Conversion return is charged to A's DST.

    def test_depth_uses_complete_current_team_snapshot(self):
        now=pd.Timestamp('2026-10-13T12:00Z')
        frame=pd.DataFrame([dict(dt=stamp,team='LA',gsis_id=pid,pos_abb='QB',pos_rank=rank) for stamp,pid,rank in
                            [('2026-10-12T08:00Z','old',1),('2026-10-13T08:00Z','new',1),('2026-10-13T08:00Z','backup',2),('2026-10-14T08:00Z','future',1)]])
        depth,_=depth_snapshot(frame,now)
        self.assertEqual(set(depth.gsis_id),{'new','backup'})
        self.assertEqual(set(depth.team),{'LAR'})
        game=pd.Series(dict(season=2026,week=6,game_type='REG'))
        reason,_=player_availability('backup','LAR','QB',game,{},now,depth)
        self.assertIn('backup',reason)
        stale,_=depth_snapshot(frame,now+pd.Timedelta(days=4))
        self.assertTrue(stale.empty)

    def test_exact_week_fresh_injury_only(self):
        now=pd.Timestamp('2026-10-13T12:00Z');game=pd.Series(dict(season=2026,week=6,game_type='REG'))
        injuries=pd.DataFrame([dict(gsis_id='player',team='LAR',season=2026,week=5,season_type='REG',report_status='Out')])
        data=dict(injuries=injuries,injuries_updated=now-pd.Timedelta(hours=1))
        self.assertEqual(player_availability('player','LAR','WR',game,data,now)[0],'')
        data['injuries'].loc[0,'week']=6
        self.assertIn('Unavailable',player_availability('player','LAR','WR',game,data,now)[0])
        data['injuries_updated']=now-pd.Timedelta(hours=40)
        self.assertEqual(player_availability('player','LAR','WR',game,data,now)[0],'')
        data['injuries_updated']=now+pd.Timedelta(hours=1)
        self.assertEqual(player_availability('player','LAR','WR',game,data,now)[0],'')

    def test_context_cannot_read_future_opponent_results(self):
        pool,data,as_of=fixture()
        _,first,_=prepare_history(data,as_of)
        extra=data['teams'].iloc[0].copy();extra['game_id']='game-100';extra['def_tds']=99
        data['teams']=pd.concat([data['teams'],pd.DataFrame([extra])],ignore_index=True)
        data['games'].loc[data['games'].game_id.eq('game-100'),['home_score','away_score']]=[999,999]
        _,second,_=prepare_history(data,as_of)
        np.testing.assert_allclose(np.stack(first.vector),np.stack(second.vector))

    def test_closed_roof_and_weather_are_not_unarchived_target_features(self):
        _,data,as_of=fixture()
        data['games']['roof']='outdoors'
        _,a,_=prepare_history(data,as_of)
        # The final observed game's current roof change cannot enter its own pregame vector.
        last='game-99';data['games'].loc[data['games'].game_id.eq(last),'roof']='closed'
        data['games']['temp']=999; data['games']['wind']=999
        _,b,_=prepare_history(data,as_of)
        np.testing.assert_allclose(np.stack(a.vector),np.stack(b.vector))

    def test_correlation_preserves_athlete_cpt_identity_and_empirical_marginals(self):
        rows=[dict(player_id='1',player_key='a',team='A',position='QB',game_id='A@B',projection=10),
              dict(player_id='2',player_key='a',team='A',position='QB',game_id='A@B',projection=10),
              dict(player_id='3',player_key='b',team='A',position='WR',game_id='A@B',projection=10)]
        calibration={'QB':{'residuals':[-2,-1,0,1,2]},'WR':{'residuals':[-2,-1,0,1,2]},'_dependence':{('same','QB1','WR1'):.8}}
        outcomes=sample_outcomes(pd.DataFrame(rows),calibration,5000,np.random.default_rng(88))
        self.assertEqual(set(outcomes),{'a','b'})
        self.assertGreater(np.corrcoef(outcomes['a'],outcomes['b'])[0,1],.7)
        self.assertGreaterEqual(outcomes['a'].min(),8);self.assertLessEqual(outcomes['a'].max(),12)

    def test_sparse_correlations_fall_back_to_independence(self):
        rows=[]
        for i in range(10):
            for pos in ['QB','WR']:
                rows.append(dict(player_id=pos,position=pos,projection=10,team='A',game_id=str(i),residual=i))
        lookup,audit=learn_dependence(pd.DataFrame(rows))
        self.assertFalse(lookup);self.assertTrue(audit.empty)

    def test_probability_model_uses_prediction_specific_residual_bins(self):
        model=dict(residuals=[-9,9],residual_bounds=[5,10],residual_bins={0:[-1,1],2:[-5,5]})
        np.testing.assert_array_equal(residual_sample(model,3),[-1,1])
        np.testing.assert_array_equal(residual_sample(model,7),[-9,9])
        np.testing.assert_array_equal(residual_sample(model,15),[-5,5])

    def test_validation_and_calibration_precede_test_and_test_does_not_select(self):
        pool,data,as_of=fixture();_,records,_=prepare_history(data,as_of)
        models,metrics=train_models(records)
        folds=metrics.attrs['fold_metrics']
        self.assertTrue((pd.to_datetime(folds.training_last_date)<pd.to_datetime(folds.validation_start)).all())
        cal=metrics.attrs['calibration_records']
        self.assertLess(cal.game_date.max(),pd.to_datetime(metrics.test_start).min())
        altered=records.copy();cutoff=pd.to_datetime(metrics.test_start).min()
        altered.loc[altered.game_date>=cutoff,'actual']=9999
        second,other=train_models(altered)
        self.assertEqual(metrics.selected_model.tolist(),other.selected_model.tolist())
        self.assertEqual(metrics.selected_features.tolist(),other.selected_features.tolist())
        for pos in models:
            np.testing.assert_allclose(models[pos]['residuals'],second[pos]['residuals'])


if __name__=='__main__':unittest.main()
