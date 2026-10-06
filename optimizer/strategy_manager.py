from enum import Enum
from typing import Dict, Any
import pandas as pd

class Strategy(Enum):
    CASH = "cash"
    GPP = "gpp"

class StrategyManager:
    """
    Manages the objective functions and constraints for different DFS strategies.
    """

    def __init__(self, strategy: Strategy = Strategy.CASH):
        self.strategy = strategy

    def get_objective_column(self, data_cols: list) -> str:
        """
        Determines which column to use for optimization based on strategy.
        """
        if self.strategy == Strategy.CASH:
            # Cash games focus on the floor/median
            return 'projection_floor' if 'projection_floor' in data_cols else 'projection'

        elif self.strategy == Strategy.GPP:
            # GPP focus on optimal % or ceiling
            if 'optimal_pct' in data_cols:
                return 'optimal_pct'
            return 'projection_ceiling' if 'projection_ceiling' in data_cols else 'projection'

        return 'projection'

    def should_apply_stacking(self) -> bool:
        """GPPs typically require stacking; Cash games do not."""
        return self.strategy == Strategy.GPP

    def should_apply_ownership_cap(self) -> bool:
        """GPPs use ownership caps to avoid 'chalk' builds."""
        return self.strategy == Strategy.GPP

    def get_ownership_threshold(self) -> float:
        """Returns the recommended total ownership cap for GPPs."""
        return 130.0  # Example threshold: 130% total ownership
