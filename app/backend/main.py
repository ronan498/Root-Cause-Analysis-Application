from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO

from app.rca.embedder import OpenAIEmbedder
from app.rca.index import RCAIndex
from app.rca.data_access import read_csv_rows, REQUIRED_COLS
from app.rca.search import QueryEngine
from app.rca.schemas import DiagnoseRequest, DiagnoseResponseItem, ComponentsResponse, ModelsResponse
from app.rca.config import CSV_PATH
from app.rca.utils import unique_components

app = FastAPI(title="RCA Demo API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _norm(x: str | None) -> str:
    return (x or "").strip().lower()

embedder = OpenAIEmbedder()
index = RCAIndex()
if not index.load():
    rows = read_csv_rows(CSV_PATH)
    embs = embedder.embed_texts([r["fault_description"] for r in rows])
    index.build(embs, rows)

engine = QueryEngine(index=index, embedder=embedder)

@app.get("/health")
def health():
    return {"status": "ok", "rows": len(index.meta)}

@app.get("/components", response_model=ComponentsResponse)
def components():
    # Prefer CSV (source of truth)
    try:
        df = pd.read_csv(CSV_PATH)
        comps = sorted({ _norm(c) for c in df.get("component", pd.Series([])).astype(str) if str(c).strip() })
    except Exception:
        comps = sorted({ _norm(m.get("component")) for m in index.meta if m.get("component") })
    return {"components": comps}

@app.get("/models", response_model=ModelsResponse)
def models(component: str = Query(..., description="Component name (e.g. 'motor')")):
    c = _norm(component)
    try:
        df = pd.read_csv(CSV_PATH)
        if "model" not in df.columns:
            return {"models": []}
        sub = df[
            df.get("component", "").astype(str).str.lower().str.strip() == c
        ]
        # Drop NaN and empty strings before normalising
        models_series = sub["model"].dropna().astype(str)
        models = sorted({_norm(m) for m in models_series.tolist() if _norm(m)})
    except Exception:
        # Fall back to meta
        models = sorted({
            _norm(m.get("model"))
            for m in index.meta
            if _norm(m.get("component")) == c and _norm(m.get("model"))
        })
    return {"models": models}


@app.post("/diagnose", response_model=list[DiagnoseResponseItem])
def diagnose(req: DiagnoseRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")
    results = engine.diagnose(
        req.query,
        top_k=req.top_k,
        component=_norm(req.component) if req.component else None,
        model=_norm(req.model) if req.model else None,
    )
    return results

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload CSV with columns:
    required: component, fault_description, root_cause, corrective_action
    optional: model
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")
    content = (await file.read()).decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(content))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required column(s): {', '.join(missing)}")

    # Coerce and normalise
    for col in ["component", "fault_description", "root_cause", "corrective_action"]:
        df[col] = df[col].astype(str)
    if "model" not in df.columns:
        df["model"] = ""
    else:
        df["model"] = df["model"].astype(str)

    # De-dup by exact fault_description already in index
    rows = df[["component", "fault_description", "root_cause", "corrective_action", "model"]].to_dict(orient="records")
    existing = set((r.get("fault_description") or "") for r in index.meta)
    new_rows = [r for r in rows if (r.get("fault_description") or "") not in existing]
    if not new_rows:
        return {"added": 0, "message": "No new rows"}

    embs = embedder.embed_texts([r["fault_description"] for r in new_rows])
    index.add(embs, new_rows)

    # Persist to CSV
    try:
        csv_df = pd.read_csv(CSV_PATH)
    except Exception:
        csv_df = pd.DataFrame(columns=["component", "fault_description", "root_cause", "corrective_action", "model"])
    csv_df = pd.concat([csv_df, pd.DataFrame(new_rows)], ignore_index=True)
    csv_df.to_csv(CSV_PATH, index=False)

    return {"added": len(new_rows)}

    if "model" not in df.columns:
        df["model"] = ""
    else:
        df["model"] = df["model"].fillna("").astype(str)
