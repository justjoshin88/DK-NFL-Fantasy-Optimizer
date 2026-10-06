import numpy as np
import pandas as pd
from scipy.stats import gamma
from typing import List, Dict, Any
import pulp

class SimulationEngine:
    """
    Monte Carlo Simulation Engine to calculate 'Optimal %' for players.
    """

    def __init__(self, data: pd.DataFrame, n_sims: int = 10000):
        """
        Initialize simulation engine.

        :param data: DataFrame containing 'player_name', 'position', 'salary', 'projection'
        :param n_sims: Number of simulation iterations.
        """
        self.data = data
        self.n_sims = n_sims
        self.players = data.index.tolist()

    def _fit_gamma_params(self, mean: float, std: float):
        """
        Fits Gamma distribution parameters k (shape) and theta (scale) from mean and std.
        """
        # Avoid division by zero or negative values
        mean = max(mean, 0.1)
        std = max(std, 0.1)

        # mean = k * theta
        # var = k * theta^2
        # theta = var / mean
        # k = mean / theta
        theta = (std**2) / mean
        k = mean / theta
        return k, theta

    def run_simulations(self) -> pd.DataFrame:
        """
        Runs Monte Carlo simulations to find the optimal lineup for each iteration.
        Returns a DataFrame with 'optimal_pct'.
        """
        # Initialize counters for each player's appearance in the optimal lineup
        optimal_counts = np.zeros(len(self.players))

        # Pre-calculate Gamma parameters for all players
        # If projection_std isn't provided, we assume a coefficient of variation (CV)
        # based on position (e.g., WRs are more volatile than QBs)
        cv_map = {'QB': 0.2, 'RB': 0.3, 'WR': 0.4, 'TE': 0.4, 'DST': 0.3}

        params = []
        for i in self.players:
            mean = self.data.loc[i, 'projection']
            std = self.data.loc[i, 'projection_std'] if 'projection_std' in self.data.columns else mean * cv_map.get(self.data.loc[i, 'position'], 0.3)
            params.append(self._fit_gamma_params(mean, std))

        print(f"Running {self.n_sims} simulations... this may take a while.")

        for s in range(self.n_sims):
            # 1. Generate random outcomes for all players using their Gamma distributions
            simulated_scores = np.zeros(len(self.players))
            for idx, (k, theta) in enumerate(params):
                simulated_scores[idx] = np.random.gamma(k, theta)

            # 2. Solve for the optimal lineup for this specific set of scores
            # We use a lightweight version of the optimizer here
            try:
                selected = self._solve_optimal_lineup(simulated_scores)
                for p_idx in selected:
                    optimal_counts[p_idx] += 1
            except Exception:
                # Skip iterations where no valid lineup is found
                continue

            if (s + 1) % 1000 == 0:
                print(f"Simulation {s+1}/{self.n_sims} complete...")

        # Calculate percentage
        self.data['optimal_pct'] = (optimal_counts / self.n_sims) * 100
        return self.data

    def _solve_optimal_lineup(self, scores: np.ndarray) -> List[int]:
        """
        Internal helper to solve a single MILP problem for one simulation.
        """
        prob = pulp.LpProblem("Sim_Opt", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("p", self.players, cat=pulp.LpBinary)

        # Objective: Maximize simulated points
        prob += pulp.lpSum([scores[i] * player_vars[i] for i in range(len(self.players))])

        # Salary Cap
        prob += pulp.lpSum([self.data.loc[i, 'salary'] * player_vars[i] for i in self.players]) <= 50000

        # Roster Constraints (DraftKings)
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'QB']) == 1
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'RB']) >= 2
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'WR']) >= 3
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'TE']) >= 1
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'DST']) == 1
        prob += pulp.lpSum([player_vars[i] for i in self.players]) == 9

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != 'Optimal':
            raise Exception("No optimal solution")

        return [i for i in self.players if pulp.value(player_vars[i]) == 1]
