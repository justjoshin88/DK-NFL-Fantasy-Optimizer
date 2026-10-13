import pulp
from typing import List, Dict, Any, Tuple
import pandas as pd

class NFLOptimizer:
    """
    Core MILP Optimizer for DraftKings NFL Lineups.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the optimizer with player data.

        :param data: DataFrame containing 'player_name', 'position', 'salary', 'projection'
        """
        self.data = data.copy().reset_index(drop=True)
        self.players = self.data.index.tolist()
        self.salary_cap = 50000

    def build_model(self, objective_column: str = 'projection', strategy_manager=None) -> Tuple[pulp.LpProblem, Dict[int, pulp.LpVariable]]:
        """
        Builds the PuLP Mixed-Integer Linear Programming model.
        """
        prob = pulp.LpProblem("DK_NFL_Optimizer", pulp.LpMaximize)
        # Decision variables: x[i] = 1 if player i is selected, else 0
        # Decision variables: x[i] = 1 if player i is selected, else 0
        # Use pulp.LpVariable's correct signature for binary variables
        player_vars = {i: pulp.LpVariable(f"player_{i}", cat=pulp.LpBinary) for i in self.players}

        # Objective Function
        prob += pulp.lpSum([self.data.loc[i, objective_column] * player_vars[i] for i in self.players])

        # Salary Cap
        prob += pulp.lpSum([self.data.loc[i, 'salary'] * player_vars[i] for i in self.players]) <= self.salary_cap

        # Base Roster Constraints
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'QB']) == 1
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'RB']) >= 2
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'WR']) >= 3
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'TE']) >= 1
        prob += pulp.lpSum([player_vars[i] for i in self.players if self.data.loc[i, 'position'] == 'DST']) == 1
        prob += pulp.lpSum([player_vars[i] for i in self.players]) == 9
        # FLEX may only be an RB, WR, or TE.
        for i in self.players:
            if self.data.loc[i, 'position'] not in {'QB', 'RB', 'WR', 'TE', 'DST'}:
                prob += player_vars[i] == 0

        # Advanced GPP Constraints
        if strategy_manager and strategy_manager.should_apply_stacking():
            self._add_stacking_constraints(prob, player_vars)

        if strategy_manager and strategy_manager.should_apply_ownership_cap():
            self._add_ownership_constraints(prob, player_vars, strategy_manager.get_ownership_threshold())

        return prob, player_vars

    def _add_stacking_constraints(self, prob: pulp.LpProblem, player_vars: Dict[int, pulp.LpVariable]):
        """
        Implements QB + WR/TE stacking logic for GPPs.
        """
        # Example: If a QB is picked, at least one pass catcher from the same team must be picked.
        teams = self.data['team'].unique()
        for team in teams:
            qb_indices = [i for i in self.players if self.data.loc[i, 'position'] == 'QB' and self.data.loc[i, 'team'] == team]
            pass_catchers = [i for i in self.players if self.data.loc[i, 'position'] in ['WR', 'TE'] and self.data.loc[i, 'team'] == team]

            for qb in qb_indices:
                # x_qb <= sum(x_pass_catchers)
                # If qb is 1, sum of pass catchers must be >= 1
                prob += pulp.lpSum([player_vars[pc] for pc in pass_catchers]) >= player_vars[qb]

    def _add_ownership_constraints(self, prob: pulp.LpProblem, player_vars: Dict[int, pulp.LpVariable], threshold: float):
        """
        Limits total ownership to avoid 'chalk' lineups.
        """
        if 'ownership' not in self.data.columns:
            return

        prob += pulp.lpSum([self.data.loc[i, 'ownership'] * player_vars[i] for i in self.players]) <= threshold


    def solve(self, prob: pulp.LpProblem, player_vars: Dict[int, pulp.LpVariable]) -> List[int]:
        """
        Solves the model and returns the indices of the selected players.
        """
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            raise Exception(f"Optimal solution not found. Status: {pulp.LpStatus[status]}")

        selected_indices = [i for i in self.players if pulp.value(player_vars[i]) == 1]
        return selected_indices


def main():
    if __package__:
        from .streamlit_ui import main as render_app
    else:
        from streamlit_ui import main as render_app
    render_app(NFLOptimizer)


if __name__ == "__main__":
    main()
