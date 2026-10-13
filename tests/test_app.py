import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from optimizer.data_loader import DataLoader
from optimizer.optimizer import NFLOptimizer
from optimizer.simulation_engine import SimulationEngine
from optimizer.strategy_manager import Strategy, StrategyManager

APP_PATH = str(Path(__file__).resolve().parents[1] / 'optimizer' / 'optimizer.py')


def player_data():
    positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'RB', 'DST', 'WR']
    return pd.DataFrame([
        dict(player_name=f'Player {i}', position=position, salary=4000,
             projection=10 + i, team='A', opponent='B', ownership=10)
        for i, position in enumerate(positions)
    ], index=range(20, 30))


class AppTests(unittest.TestCase):
    def test_optimizer_roster_and_cap(self):
        optimizer = NFLOptimizer(player_data())
        manager = StrategyManager(Strategy.GPP)
        model, variables = optimizer.build_model(strategy_manager=manager)
        lineup = optimizer.data.loc[optimizer.solve(model, variables)]
        counts = lineup.position.value_counts()
        self.assertEqual(len(lineup), 9)
        self.assertEqual(counts['QB'], 1)
        self.assertEqual(counts['DST'], 1)
        self.assertGreaterEqual(counts['RB'], 2)
        self.assertGreaterEqual(counts['WR'], 3)
        self.assertGreaterEqual(counts['TE'], 1)
        self.assertLessEqual(lineup.salary.sum(), 50000)
        self.assertLessEqual(lineup.ownership.sum(), 130)

    def test_infeasible_lineup(self):
        optimizer = NFLOptimizer(player_data().query("position != 'QB'"))
        with self.assertRaisesRegex(Exception, 'Infeasible'):
            optimizer.solve(*optimizer.build_model())

    def test_simulation_non_contiguous_index(self):
        result = SimulationEngine(player_data(), n_sims=2).run_simulations()
        self.assertAlmostEqual(result.optimal_pct.sum(), 900)

    def test_bad_data(self):
        for column, value in [('projection', np.nan), ('salary', -5),
                              ('salary', 4000.5), ('ownership', 'bad')]:
            with self.subTest(column=column, value=value):
                data = player_data().astype({'salary': float, 'ownership': object})
                data.loc[20, column] = value
                with self.assertRaises((TypeError, ValueError)):
                    DataLoader().load_data(io.StringIO(data.to_csv(index=False)))

    def test_streamlit_startup(self):
        app = AppTest.from_file(APP_PATH).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, 'DK NFL Fantasy Optimizer')

    def test_streamlit_upload_and_solve(self):
        upload = io.StringIO(player_data().to_csv(index=False))
        upload.name = 'players.csv'
        with patch('streamlit.file_uploader', side_effect=lambda label, **kw: upload if label == 'Player pool CSV' else None):
            app = AppTest.from_file(APP_PATH).run()
            upload.seek(0)
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.success[0].value, 'Lineup built successfully')

    def test_streamlit_invalid_upload(self):
        upload = io.StringIO('player_name\nSomeone\n')
        upload.name = 'invalid.csv'
        with patch('streamlit.file_uploader', side_effect=lambda label, **kw: upload if label == 'Player pool CSV' else None):
            app = AppTest.from_file(APP_PATH).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Missing required columns', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
