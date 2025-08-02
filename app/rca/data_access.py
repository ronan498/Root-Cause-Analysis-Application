from typing import List, Dict
import pandas as pd
from .config import CSV_PATH

REQUIRED_COLS = ["component", "fault_description", "root_cause", "corrective_action"]
OPTIONAL_COLS = ["model"]

def read_csv_rows(path=CSV_PATH) -> List[Dict]:
    # Be tolerant of optional 'model' column
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True, engine="python")
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Ensure all required present; add optional if missing
    for col in REQUIRED_COLS + OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    return df[REQUIRED_COLS + OPTIONAL_COLS].to_dict(orient="records")

    for col in REQUIRED_COLS + OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

