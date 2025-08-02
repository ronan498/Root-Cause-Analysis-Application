# app/backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
from io import StringIO
import csv
import os
import re

from app.rca.embedder import OpenAIEmbedder
from app.rca.index import RCAIndex
from app.rca.data_access import read_csv_rows, REQUIRED_COLS
from app.rca.search import QueryEngine
from app.rca.schemas import DiagnoseRequest, DiagnoseResponseItem, ComponentsResponse
from app.rca.config import CSV_PATH
from app.rca.utils import unique_components

app = FastAPI(title="RCA Demo API", version="0.1.0")

# CORS so Streamlit can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------

_wordish = re.compile(r"^[a-z0-9][a-z0-9 _\-\/]{0,40}$")

def normalise_component(value: str | None) -> str:
    """Canonicalise component labels."""
    s = (value or "").strip().lower()
    synonyms = {
        "motors": "motor",
        "pumps": "pump",
        "compressors": "compressor",
    }
    return synonyms.get(s, s)

def is_reasonable_component(s: str) -> bool:
    """
    Heuristic guardrail:
    - short, word-like tokens
    - avoid sentences that include punctuation like ',' or '.'
    - allow 1–3 words (e.g., 'gas turbine', 'centrifugal compressor')
    """
    if not s:
        return False
    if "," in s or "." in s:
        return False
    words = s.split()
    if len(words) > 3:
        return False
    return bool(_wordish.match(s))

def read_components_from_csv() -> List[str]:
    """
    Robustly read the 'component' column from CSV using the csv module and
    keep ONLY that column, so commas in other columns can't pollute it.
    """
    comps: List[str] = []
    path = str(CSV_PATH)
    if not os.path.exists(path):
        return comps
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        comp_idx = 0
        if header:
            # Find 'component' column index if present
            try:
                comp_idx = [h.strip().lower() for h in header].index("component")
            except ValueError:
                comp_idx = 0
        for row in reader:
            if not row or comp_idx >= len(row):
                continue
            raw = row[comp_idx]
            comp = normalise_component(raw)
            if is_reasonable_component(comp):
                comps.append(comp)
    # Unique + sorted
    return sorted({c for c in comps if c})

# ---------------- Singletons ----------------

embedder = OpenAIEmbedder()
index = RCAIndex()
if not index.load():
    # Build index from seed CSV on first run
    rows = read_csv_rows(CSV_PATH)
    embs = embedder.embed_texts([r["fault_description"] for r in rows])
    index.build(embs, rows)

engine = QueryEngine(index=index, embedder=embedder)

# ---------------- Routes ----------------

@app.get("/health")
def health():
    return {"status": "ok", "rows": len(index.meta)}

# FIXED: components now come straight from CSV (first column only), normalised and validated
@app.get("/components", response_model=ComponentsResponse)
def components():
    comps = read_components_from_csv()
    if not comps:
        # Fallback to in-memory meta (already normalised by helper below)
        comps = [
            normalise_component(m.get("component"))
            for m in index.meta
            if is_reasonable_component(normalise_component(m.get("component")))
        ]
        comps = sorted({c for c in comps if c})
    return {"components": comps}

@app.post("/diagnose", response_model=list[DiagnoseResponseItem])
def diagnose(req: DiagnoseRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")
    component = normalise_component(req.component) if req.component else None
    results = engine.diagnose(req.query, top_k=req.top_k, component=component)
    return results

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a CSV with columns: component, fault_description, root_cause, corrective_action.
    New rows are embedded and added to the index and appended to data/faults.csv.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    content = (await file.read()).decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(content))

    # Validate required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise HTTPException(400, f"Missing required column: {col}")

    # Coerce to str and normalise component values
    for col in REQUIRED_COLS:
        df[col] = df[col].astype(str)
    df["component"] = df["component"].map(lambda x: normalise_component(x))

    # Light validation: drop clearly invalid 'component' values
    df = df[df["component"].map(is_reasonable_component)]
    if df.empty:
        return {"added": 0, "message": "No valid rows after validation", "components": read_components_from_csv()}

    rows = df[REQUIRED_COLS].to_dict(orient="records")

    # De-duplicate by exact fault_description already in index
    existing = set(str(r.get("fault_description")) for r in index.meta)
    new_rows = [r for r in rows if str(r.get("fault_description")) not in existing]
    if not new_rows:
        return {"added": 0, "message": "No new rows", "components": read_components_from_csv()}

    # Embed and add to index
    embs = embedder.embed_texts([r["fault_description"] for r in new_rows])
    index.add(embs, new_rows)

    # Persist to canonical CSV (append)
    try:
        csv_df = pd.read_csv(CSV_PATH)
    except Exception:
        csv_df = pd.DataFrame(columns=REQUIRED_COLS)
    csv_df = pd.concat([csv_df, pd.DataFrame(new_rows)], ignore_index=True)
    csv_df.to_csv(CSV_PATH, index=False)

    # Return fresh list for the UI
    return {"added": len(new_rows), "components": read_components_from_csv()}
