import pandas as pd
from typing import List, Tuple

def ensure_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    