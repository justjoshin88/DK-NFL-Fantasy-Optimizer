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
    import streamlit as st

    # Streamlit executes this file directly; support package imports as well.
    if __package__:
        from .data_loader import DataLoader
        from .strategy_manager import Strategy, StrategyManager
    else:
        from data_loader import DataLoader
        from strategy_manager import Strategy, StrategyManager

    st.set_page_config(page_title="DK NFL Fantasy Optimizer", layout="wide")
    st.title("DK NFL Fantasy Optimizer")
    st.write("Upload your NFL Classic salaries and projections to build a $50,000 lineup.")
    st.caption("Classic roster: QB, 2 RB, 3 WR, TE, FLEX, DST. Showdown is not supported.")
    with st.expander("Required file columns"):
        st.write("player_name, position, salary, projection, team, opponent")
        st.write("Optional: projection_floor, projection_ceiling, ownership (percent). "
                 "DraftKings salary files need projections added before uploading.")
    upload = st.file_uploader("Player data", type=["csv", "json"])
    strategy = st.selectbox("Strategy", ["Cash", "GPP"])
    if upload is None:
        st.info("Upload a CSV or JSON file to begin.")
        return
    try:
        loader = DataLoader()
        data = loader.normalize_positions(loader.load_data(upload)).reset_index(drop=True)
    except (ValueError, TypeError, OSError) as exc:
        st.error(str(exc))
        return
    st.dataframe(data, hide_index=True)
    if st.button("Build lineup", type="primary"):
        manager = StrategyManager(Strategy.GPP if strategy == "GPP" else Strategy.CASH)
        if strategy == "GPP" and "ownership" not in data.columns:
            st.warning("No ownership column provided. The ownership cap will not be applied.")
        try:
            with st.spinner("Optimizing lineup..."):
                optimizer = NFLOptimizer(data)
                model, variables = optimizer.build_model(manager.get_objective_column(data.columns), manager)
                lineup = optimizer.data.loc[optimizer.solve(model, variables)].copy()
        except Exception as exc:
            st.error(f"Unable to build a lineup: {exc}. Check roster coverage, salaries, and strategy constraints.")
            return
        st.success("Lineup built successfully")
        st.metric("Total salary", f"${lineup['salary'].sum():,.0f}")
        st.metric("Projected points", f"{lineup['projection'].sum():.2f}")
        st.dataframe(lineup, hide_index=True)
        st.download_button("Download lineup CSV", lineup.to_csv(index=False), "lineup.csv", "text/csv")


if __name__ == "__main__":
    main()
