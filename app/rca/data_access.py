from typing import List, Dict
import pandas as pd
from .config import CSV_PATH

REQUIRED_COLS = ["component", "fault_description", "root_cause", "corrective_action"]

def read_csv_rows(path=CSV_PATH) -> List[Dict]:
    df = pd.read_csv(path)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    for col in REQUIRED_COLS:
        df[col] = df[col].astype(str)
    return df[REQUIRED_COLS].to_dict(orient="records")
