import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any

class DataLoader:
    """Handles loading and validation of NFL player projections and salaries."""

    REQUIRED_COLUMNS = {
        'player_name': str,
        'position': str,
        'salary': int,
        'projection': float,
        'team': str,
        'opponent': str
    }

    def load_data(self, file_input: Any) -> pd.DataFrame:
        """
        Loads data from CSV or JSON and validates the schema.
        Supports both file paths (strings) and file-like objects (buffers).
        """
        if isinstance(file_input, str):
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"File not found: {file_input}")
            ext = os.path.splitext(file_input)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_input)
            elif ext == '.json':
                df = pd.read_json(file_input)
            else:
                raise ValueError("Unsupported file format. Please provide a .csv or .json file.")
        else:
            # Handle file-like objects (e.g., from Streamlit's file_uploader)
            # Try to determine format from filename if available
            try:
                filename = getattr(file_input, 'name', '')
                if filename.endswith('.json'):
                    df = pd.read_json(file_input)
                else:
                    # Default to CSV
                    df = pd.read_csv(file_input)
            except Exception as e:
                raise ValueError(f"Could not parse uploaded file: {e}")

        self._validate_schema(df)
        return df

    def _validate_schema(self, df: pd.DataFrame):
        """
        Ensures all required columns are present and have the correct types.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if df.empty:
            raise ValueError("The uploaded file has no players.")

        # Basic type coercion/validation
        try:
            numeric_columns = ['salary', 'projection'] + [
                col for col in ['projection_floor', 'projection_ceiling', 'ownership', 'projection_std', 'optimal_pct']
                if col in df.columns
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='raise')
                if not np.isfinite(df[col]).all():
                    raise ValueError(f"{col} must contain finite numbers for every player")
            if (df['salary'] <= 0).any() or (df['salary'] % 1 != 0).any():
                raise ValueError("salary must contain positive whole dollar amounts")
            df['salary'] = df['salary'].astype(int)
        except Exception as e:
            raise TypeError(f"Data type mismatch in salary or projection columns: {e}")
        for col in ['player_name', 'position', 'team', 'opponent']:
            if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
                raise ValueError(f"{col} must be provided for every player")
            df[col] = df[col].astype(str).str.strip()
        for col in ['ownership', 'projection_std', 'optimal_pct']:
            if col in df.columns and (df[col] < 0).any():
                raise ValueError(f"{col} cannot be negative")
        if 'ownership' in df.columns and (df['ownership'] > 100).any():
            raise ValueError("ownership must be a percentage between 0 and 100")

    def normalize_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes position strings to a standard set (QB, RB, WR, TE, DST).
        """
        # Simple mapping example; can be expanded
        mapping = {
            'Quarterback': 'QB',
            'Running Back': 'RB',
            'Wide Receiver': 'WR',
            'Tight End': 'TE',
            'Defense/Special Teams': 'DST',
            'DST': 'DST'
        }
        df['position'] = df['position'].replace(mapping).str.upper()
        df['position'] = df['position'].replace({'D/ST': 'DST', 'DEF': 'DST'})
        invalid = sorted(set(df['position']) - {'QB', 'RB', 'WR', 'TE', 'DST'})
        if invalid:
            raise ValueError(f"Unsupported positions: {', '.join(invalid)}. Use QB, RB, WR, TE, DST.")
        return df
