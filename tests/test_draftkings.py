import csv
import io
import unittest
import itertools
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from optimizer.dk_csv import load_pool, export_rosters, roster_template, CLASSIC, SHOWDOWN
from optimizer.providers import parse_dff, match_projections, load_projection_file, contest_id, ProviderError, verify_contest_pool, fetch_sportsdata
from optimizer.lineup_engine import solve_lineup, build_portfolio, eligible_pool, validate_lineup
from optimizer.calibration import calibrate, evaluate_portfolio


HEADER = ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary', 'Game Info', 'TeamAbbrev', 'AvgPointsPerGame']


def csv_text(rows):
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return '\ufeff' + buffer.getvalue()


def classic_csv():
    rows = [HEADER]
    positions = ['QB', 'QB', 'RB', 'RB', 'RB', 'RB', 'WR', 'WR', 'WR', 'WR', 'WR', 'TE', 'TE', 'DST', 'DST']
    for i, position in enumerate(positions):
        team, game = ('A', 'A@B') if i % 2 == 0 else ('C', 'C@D')
        rows.append([position, f'Player {i} ({100+i})', f'Player {i}', str(100+i),
                     position + ('/FLEX' if position in {'RB', 'WR', 'TE'} else ''),
                     '4500', f'{game} 10/18/2099 01:00PM ET', team, '999'])
    return csv_text(rows)


def classic_pool():
    pool = load_pool(classic_csv())
    pool['projection'] = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    pool['exclusion_reason'] = ''
    return pool


def showdown_csv():
    rows = [HEADER]
    for i in range(10):
        for slot, multiplier in [('CPT', 1.5), ('FLEX', 1)]:
            rows.append(['QB' if i < 2 else 'WR', f'Player {i}', f'Player {i}', str(200+i*2+(slot=='FLEX')),
                         slot, str(int((5000+i*200)*multiplier)), 'A@B 10/18/2099 08:20PM ET', 'A' if i%2 else 'B', '10'])
    return csv_text(rows)


def source_row(name='Player 0', date='2099-10-18', projection=20, status=''):
    return (f'<tr class="projections-listing" data-name="{name}" data-team="A" data-opp="B" '
            f'data-pos="QB" data-start_date="{date}" data-ppg_proj="{projection}" data-inj="{status}" '
            'data-ou="48" data-spread="-3" data-proj_score="25.5"><td data-own="12"></td></tr>')


class DraftKingsTests(unittest.TestCase):
    def test_native_pool_and_bom(self):
        pool = classic_pool()
        self.assertEqual(pool.player_id.iloc[0], '100')
        self.assertEqual(pool.opponent.iloc[0], 'B')
        self.assertEqual(pool.game_date.iloc[0], '2099-10-18')
        self.assertEqual(pool.avg_points_per_game.iloc[0], '999')
        self.assertNotEqual(pool.projection.iloc[0], 999)

    def test_roster_is_not_player_pool(self):
        with self.assertRaisesRegex(ValueError, 'roster upload template'):
            load_pool(csv_text([CLASSIC + ['', 'Instructions'], ['']*11]))

    def test_duplicate_pool_ids_rejected(self):
        rows = list(csv.reader(io.StringIO(classic_csv().lstrip('\ufeff'))))
        rows[2][3] = rows[1][3]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_pool(csv_text(rows))

    def test_classic_lineup_salary_and_two_games(self):
        lineup = solve_lineup(classic_pool())
        validate_lineup(lineup, 'Classic')
        self.assertEqual(list(lineup.slot), CLASSIC)
        self.assertEqual(lineup.player_key.nunique(), 9)
        self.assertLessEqual(lineup.salary.sum(), 50000)
        self.assertGreaterEqual(lineup.game_id.nunique(), 2)

    def test_solver_matches_exhaustive_classic_search(self):
        pool = classic_pool()
        feasible = []
        for indices in itertools.combinations(range(len(pool)), 9):
            subset = pool.iloc[list(indices)]
            counts = subset.position.value_counts()
            if (counts.get('QB', 0) == 1 and counts.get('DST', 0) == 1 and counts.get('RB', 0) >= 2
                    and counts.get('WR', 0) >= 3 and counts.get('TE', 0) >= 1
                    and subset.salary.sum() <= 50000 and subset.game_id.nunique() >= 2):
                feasible.append(subset.projection.sum())
        self.assertEqual(solve_lineup(pool).projected_points.sum(), max(feasible))

    def test_single_game_classic_rejected(self):
        pool = classic_pool(); pool['game_id'] = 'A@B'
        with self.assertRaisesRegex(ValueError, 'two NFL games'):
            solve_lineup(pool)

    def test_showdown_captain_id_price_and_no_double_athlete(self):
        pool = load_pool(showdown_csv(), 'Showdown')
        pool['projection'] = [30 - i//2 for i in range(len(pool))]
        lineup = solve_lineup(pool, 'Showdown')
        self.assertEqual(list(lineup.slot), SHOWDOWN)
        self.assertEqual(lineup.player_key.nunique(), 6)
        self.assertEqual(lineup.team.nunique(), 2)
        self.assertEqual(lineup.iloc[0].projected_points, lineup.iloc[0].projection*1.5)
        self.assertEqual(lineup.salary.sum(), pool.set_index('player_id').loc[lineup.player_id].salary.sum())

    def test_game_type_mismatch(self):
        with self.assertRaises(ValueError):
            load_pool(classic_csv(), 'Showdown')
        with self.assertRaises(ValueError):
            load_pool(showdown_csv(), 'Classic')

    def test_exact_roster_header_and_ids(self):
        template = csv_text([CLASSIC + ['', 'Instructions'], ['']*10+['Copy ID here']])
        lineup = solve_lineup(classic_pool())
        result = export_rosters([lineup], 'Classic', template)
        rows = list(csv.reader(io.StringIO(result.decode('utf-8-sig'))))
        self.assertEqual(rows[0], CLASSIC+['', 'Instructions'])
        self.assertEqual(rows[1][:9], lineup.player_id.tolist())
        self.assertEqual(len(rows[1]), 11)
        self.assertNotIn('RB.1', rows[0])
        self.assertNotIn('Player', result.decode())

    def test_filled_template_rejected(self):
        with self.assertRaisesRegex(ValueError, 'blank'):
            roster_template(csv_text([CLASSIC, ['100']+['']*8]), 'Classic')

    def test_public_source_date_injury_and_freshness(self):
        now = datetime(2099, 10, 17, 12, tzinfo=timezone.utc)
        html = '<time itemprop="dateModified" datetime="2099-10-17T11:00:00Z"></time>' + source_row()
        forecasts = parse_dff(html, 'https://example.com', now)
        data = match_projections(classic_pool(), forecasts)
        self.assertEqual(data.projection.iloc[0], 20)
        self.assertTrue(np.isnan(data.projection.iloc[1]))
        wrong_date = forecasts.copy(); wrong_date['game_date'] = '2099-10-11'
        self.assertTrue(match_projections(classic_pool(), wrong_date).projection.isna().all())
        wrong_team = forecasts.copy(); wrong_team['team'] = 'C'
        self.assertTrue(match_projections(classic_pool(), wrong_team).projection.isna().all())
        out = forecasts.copy(); out['injury_status'] = 'O'
        self.assertIn('Unavailable', match_projections(classic_pool(), out).exclusion_reason.iloc[0])
        with self.assertRaisesRegex(ProviderError, '24 hours'):
            parse_dff(html.replace('2099-10-17T11', '2099-10-15T11'), 'url', now)

    def test_locked_games_excluded(self):
        pool = classic_pool()
        eligible, excluded = eligible_pool(pool, datetime(2099, 10, 19, tzinfo=timezone.utc))
        self.assertTrue(eligible.empty)
        self.assertEqual(len(excluded), len(pool))

    def test_portfolio_uniqueness_and_locks(self):
        pool = classic_pool()
        lock = pool.player_key.iloc[0]
        lineups, warning = build_portfolio(pool, count=3, locked_keys=[lock])
        self.assertIsNone(warning)
        self.assertEqual(len(lineups), 3)
        for lineup in lineups:
            self.assertIn(lock, set(lineup.player_key))
        for i in range(1, len(lineups)):
            self.assertGreaterEqual(len(set(lineups[i].player_key)-set(lineups[0].player_key)), 2)

    def test_exposure_is_not_rounded_above_requested_percentage(self):
        with self.assertRaisesRegex(ValueError, 'at least'):
            build_portfolio(classic_pool(), count=3, max_exposure=.1)

    def test_upload_forecast_id_and_numeric_validation(self):
        pool = classic_pool().drop(columns=['projection', 'exclusion_reason'])
        data = load_projection_file('ID,projection\n100,20\n', pool)
        self.assertEqual(data.projection.iloc[0], 20)
        with self.assertRaises(ValueError):
            load_projection_file('ID,projection\n999,20\n', pool)
        with self.assertRaises(ValueError):
            load_projection_file('ID,projection\n100,NaN\n', pool)
        safe = load_projection_file('ID,projection,salary,team\n100,20,1,EVIL\n', pool)
        self.assertEqual(safe.salary.iloc[0], 4500)
        self.assertEqual(safe.team.iloc[0], 'A')

    def test_licensed_feed_rejects_trials_and_unverified_freshness(self):
        pool = classic_pool().drop(columns=['projection', 'exclusion_reason'])
        with self.assertRaisesRegex(ProviderError, 'scramble'):
            fetch_sportsdata(pool, 'fixture', False)
        forecasts = pd.DataFrame([dict(player_name='Player 0', position='QB', team='A', opponent='B',
                                      game_date='2099-10-18', projection=20, injury_status='Not supplied',
                                      source='SportsDataIO', freshness_verified=False)])
        data = match_projections(pool, forecasts)
        self.assertIn('freshness', data.exclusion_reason.iloc[0])

    def test_contest_salary_mismatch(self):
        pool = classic_pool()
        entries = [{'draftableId': row.player_id, 'salary': 1} for _, row in pool.iterrows()]
        with patch('optimizer.providers.request', return_value={'draftables': entries}):
            with self.assertRaisesRegex(ValueError, 'salaries differ'):
                verify_contest_pool({'draft_group_id': 1}, pool)

    def test_contest_url_and_wrong_pool(self):
        self.assertEqual(contest_id('https://www.draftkings.com/draft/contest/123'), '123')
        with self.assertRaises(ValueError):
            contest_id('https://evil.example/contest/123')
        with patch('optimizer.providers.request', return_value={'draftables': [{'draftableId': '999'}]}):
            with self.assertRaisesRegex(ValueError, 'do not belong'):
                verify_contest_pool({'draft_group_id': 1}, classic_pool())

    def test_calibration_temporal_holdout(self):
        history = pd.DataFrame([{'game_date': f'2025-09-{day:02}', 'position': pos, 'projection': 10, 'actual': 10+(i%3)-1}
                                for day in range(1, 13) for pos in ['QB','RB','WR','TE','DST'] for i in range(5)])
        models, metrics = calibrate(history)
        self.assertEqual(set(models), {'QB','RB','WR','TE','DST'})
        self.assertTrue((metrics.holdout_rows == 15).all())
        lineups, _ = build_portfolio(classic_pool(), count=2, strategy='GPP', calibration=models)
        summary = evaluate_portfolio(lineups, models)
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary.model_p90 >= summary.model_p10).all())

    def test_streamlit_native_flow_and_export_persists(self):
        pool = classic_pool()
        pool['source'] = 'Test fixture'
        pool['injury_status'] = 'No injury flag reported'
        upload = io.BytesIO(classic_csv().encode('utf-8'))
        upload.name = 'pool.csv'
        path = str(Path(__file__).resolve().parents[1] / 'optimizer/optimizer.py')
        def uploader(label, **kwargs):
            return upload if label == 'Player pool CSV' else None
        module_path = str(Path(__file__).resolve().parents[1] / 'optimizer')
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        ui = importlib.import_module('streamlit_ui')
        with patch('streamlit.file_uploader', side_effect=uploader), patch.object(ui, 'automatic_data', return_value=(pool, [])):
            app = AppTest.from_file(path, default_timeout=15).run()
            app.button[-1].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.session_state['portfolio']), 3)
            app.run()
            self.assertEqual(len(app.session_state['portfolio']), 3)


if __name__ == '__main__':
    unittest.main()
