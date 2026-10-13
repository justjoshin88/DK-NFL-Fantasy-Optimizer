import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from optimizer.free_model import (player_scores, defense_scores, feature_vector, fit_ridge, predict,
                                  train_models, generate_forecasts, prepare_history, fetch_free, name_key)
from optimizer.providers import ProviderError


STAT_FIELDS = ['passing_yards','passing_tds','passing_interceptions','rushing_yards','rushing_tds',
               'receptions','receiving_yards','receiving_tds','fumbles_lost_total','special_teams_tds',
               'fumble_recovery_tds','passing_2pt_conversions','rushing_2pt_conversions','receiving_2pt_conversions',
               'fg_made_0_19','fg_made_20_29','fg_made_30_39','fg_made_40_49','fg_made_50_59','fg_made_60_','pat_made',
               'attempts','carries','targets','fg_att']
DEF_FIELDS = ['def_sacks','def_interceptions','fumble_recovery_opp','def_tds','special_teams_tds','def_safeties',
              'def_punt_blocks','def_fg_blocks','def_pat_blocks','def_2pt_made']


def fixture():
    games, players, teams, roster = [], [], [], []
    start = pd.Timestamp('2025-04-01 13:00:00', tz='America/New_York')
    athletes = [('QB',i) for i in range(4)] + [('RB',i) for i in range(4)] + [('WR',i) for i in range(5)] + [('TE',i) for i in range(2)] + [('K',i) for i in range(2)]
    for position, i in athletes:
        roster.append(dict(full_name=f'{position} Player {i}', team='A' if i%2 else 'B', position=position, status='ACT', gsis_id=f'{position}{i}'))
    for day in range(61):
        kickoff = start + pd.Timedelta(days=day)
        game_id = f'game-{day}'
        games.append(dict(game_id=game_id, gameday=str(kickoff.date()), gametime='13:00', away_team='A', home_team='B',
                          season=2025, week=day+1, game_type='REG', away_score=24 if day<60 else np.nan,
                          home_score=20 if day<60 else np.nan))
        if day == 60:
            continue
        for position, i in athletes:
            row = dict.fromkeys(STAT_FIELDS,0)
            row.update(player_id=f'{position}{i}', player_display_name=f'{position} Player {i}', team='A' if i%2 else 'B',
                       position=position, game_id=game_id, season_type='REG', season=2025)
            if position=='QB': row.update(passing_yards=200+day%5+i, passing_tds=1, attempts=30)
            elif position=='RB': row.update(rushing_yards=60+day%7+i, rushing_tds=1, carries=15, targets=2, receptions=2, receiving_yards=10)
            elif position=='K': row.update(fg_made_30_39=2, pat_made=2, fg_att=2)
            else: row.update(receptions=5, receiving_yards=60+day%7+i, targets=8)
            players.append(row)
        for team, opponent in [('A','B'),('B','A')]:
            row = dict.fromkeys(DEF_FIELDS,0)
            row.update(team=team, opponent_team=opponent, game_id=game_id, season_type='REG', season=2025, def_sacks=2)
            teams.append(row)
    target = start + pd.Timedelta(days=60)
    pool = pd.DataFrame([dict(player_name=r['full_name'],position=r['position'],team=r['team'],opponent='B' if r['team']=='A' else 'A',
                             player_key=f'{r["gsis_id"]}:{r["team"]}',player_id=str(100+i), game_date=str(target.date()),
                             kickoff=target.isoformat(), game_id='A@B:'+str(target.date()), salary=4000,
                             eligible_slots=[r['position']]+(['FLEX'] if r['position'] in {'RB','WR','TE'} else []))
                         for i,r in enumerate(roster)])
    for team in ['A','B']:
        pool.loc[len(pool)] = dict(player_name=team,position='DST',team=team,opponent='B' if team=='A' else 'A',
                                  player_key=f'DST:{team}',player_id=str(300+len(pool)),game_date=str(target.date()),
                                  kickoff=target.isoformat(),game_id='A@B:'+str(target.date()),salary=3000,eligible_slots=['DST'])
    data = dict(games=pd.DataFrame(games),players=pd.DataFrame(players),teams=pd.DataFrame(teams),roster=pd.DataFrame(roster))
    return pool, data, target-pd.Timedelta(hours=12)


class FreeModelTests(unittest.TestCase):
    def test_dk_offense_bonus_and_kicker_scoring(self):
        qb=dict.fromkeys(STAT_FIELDS,0);qb.update(position='QB',passing_yards=300,passing_tds=2,passing_interceptions=1,rushing_yards=50,rushing_tds=1,fumbles_lost_total=1)
        wr=dict.fromkeys(STAT_FIELDS,0);wr.update(position='WR',receiving_yards=100,receptions=5,receiving_tds=1)
        k=dict.fromkeys(STAT_FIELDS,0);k.update(position='K',fg_made_0_19=1,fg_made_40_49=1,fg_made_50_59=1,pat_made=1)
        self.assertEqual(player_scores(pd.DataFrame([qb,wr,k])).tolist(),[32,24,13])

    def test_defense_points_allowed_excludes_opposing_defensive_td(self):
        rows=[]
        for team,opp in [('A','B'),('B','A')]:
            row=dict.fromkeys(DEF_FIELDS,0);row.update(team=team,opponent_team=opp,game_id='g')
            if team=='A':row.update(def_sacks=2,def_interceptions=1,fumble_recovery_opp=1,def_tds=1,def_safeties=1,def_fg_blocks=1)
            else:row.update(def_tds=1)
            rows.append(row)
        games=pd.DataFrame([dict(game_id='g',home_team='A',away_team='B',home_score=20,away_score=13)])
        scored=defense_scores(pd.DataFrame(rows),games).set_index('team')
        self.assertEqual(scored.loc['A','dk_score'],20)

    def test_schema_changes_fail_explicitly(self):
        with self.assertRaisesRegex(ProviderError,'schema changed'):
            player_scores(pd.DataFrame([dict(position='QB',passing_yards=300)]))

    def test_generated_forecasts_are_validated_and_id_stable(self):
        pool,data,as_of=fixture()
        forecasts,metrics,missing,models=generate_forecasts(pool,data,as_of)
        self.assertFalse(missing)
        self.assertEqual(len(forecasts),len(pool))
        self.assertEqual(set(models),{'QB','RB','WR','TE','K','DST'})
        self.assertTrue(np.isfinite(forecasts.projection).all())
        self.assertTrue((metrics.test_rows>=20).all())
        self.assertTrue((pd.to_datetime(metrics.test_start,utc=True)<as_of).all())

    def test_future_outcomes_do_not_enter_features_or_forecasts(self):
        pool,data,as_of=fixture()
        first,_,_,_=generate_forecasts(pool,data,as_of)
        future=data['players'].iloc[0].copy();future['game_id']='game-60';future['passing_yards']=99999
        data['players']=pd.concat([data['players'],pd.DataFrame([future])],ignore_index=True)
        data['games'].loc[data['games'].game_id.eq('game-60'),['home_score','away_score']]=[60,60]
        second,_,_,_=generate_forecasts(pool,data,as_of)
        np.testing.assert_allclose(first.projection,second.projection)

    def test_unavailable_roster_and_cold_start_are_not_filled(self):
        pool,data,as_of=fixture()
        data['roster'].loc[data['roster'].gsis_id.eq('QB0'),'status']='RES'
        data['players']=data['players'][~data['players'].player_id.eq('QB1')]
        forecasts,_,missing,_=generate_forecasts(pool,data,as_of)
        self.assertTrue(any('active status' in reason for reason in missing.values()))
        self.assertTrue(any('three observed' in reason for reason in missing.values()))
        self.assertFalse(forecasts.player_name.isin(['QB Player 0', 'QB Player 1']).any())

    def test_schedule_mismatch_blocks_estimate(self):
        pool,data,as_of=fixture();pool.loc[0,'opponent']='EVIL'
        forecasts,_,missing,_=generate_forecasts(pool,data,as_of)
        self.assertIn('schedule',missing[pool.player_key.iloc[0]])
        self.assertNotIn(pool.player_name.iloc[0],forecasts.player_name.tolist())

    def test_public_out_flag_overrides_model(self):
        pool,data,as_of=fixture()
        public=pd.DataFrame([dict(player_name=pool.player_name.iloc[0],position='QB',team=pool.team.iloc[0],
                                 opponent=pool.opponent.iloc[0],game_date=pool.game_date.iloc[0],projection=0,
                                 injury_status='O',source='Public')])
        with patch('optimizer.free_model.fetch_public',return_value=(public,[])):
            result,_=fetch_free(pool,data,as_of)
        self.assertEqual(result.injury_status.iloc[0],'O')
        self.assertIn('Unavailable',result.exclusion_reason.iloc[0])
        self.assertEqual(result.player_id.tolist(),pool.player_id.tolist())

    def test_known_suffix_normalization_is_not_fuzzy(self):
        self.assertEqual(name_key('James Cook III'),name_key('James Cook'))
        self.assertNotEqual(name_key('James Cook'),name_key('J Cook'))


if __name__=='__main__':
    unittest.main()
