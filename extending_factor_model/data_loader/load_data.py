"""Load and preprocess the data."""

import io
from pathlib import Path

import pandas as pd


def load_factor_data(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded Fama-French factor data.

    The factors are:

    - Mom - Momentum
    - Mkt-RF - Market return minus risk-free rate
    - SMB - Small minus big
    - HML - High minus low
    - RMW - Robust minus weak
    - CMA - Conservative minus aggressive
    - LT_Rev - Long-term reversal
    - ST_Rev - Short-term reversal

    The factor data has units of percent per day.

    Args:
        start_date (str, optional): Start date. Defaults to "1995-01-01".
        end_date (str, optional): End date. Defaults to "2025-01-01".

    Returns:
        pd.DataFrame: Factor data with columns for each factor and index for dates.
    """
    factor_dir = Path("data/raw/factors")
    factors = ["Mom", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "LT_Rev", "ST_Rev"]

    factor_dfs = []
    for file in factor_dir.glob("*.csv"):
        with Path(file).open("r") as f:
            content = f.read() 

        # extract csv content between empty lines
        csv_data = content.split("\n\n")[-2]
        buffer = io.StringIO(csv_data)
        factor_returns = pd.read_csv(buffer, parse_dates=[0], index_col=0) / 100.0
        factor_dfs.append(factor_returns)

    return pd.concat(factor_dfs, axis=1).loc[start_date:end_date][factors].sort_index().dropna()
