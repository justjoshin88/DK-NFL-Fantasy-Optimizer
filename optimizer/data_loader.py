import pandas as pd
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

        # Basic type coercion/validation
        try:
            df['salary'] = df['salary'].astype(int)
            df['projection'] = df['projection'].astype(float)
        except Exception as e:
            raise TypeError(f"Data type mismatch in salary or projection columns: {e}")

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
        df['position'] = df['position'].replace(mapping)
        return df
