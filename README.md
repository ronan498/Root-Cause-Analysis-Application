# Root Cause Analysis Demo

Similarity-based root cause analysis for motors, pumps, and compressors. The backend is built with FastAPI and OpenAI embeddings, and the frontend uses Streamlit.

## Features
- Describe a fault and receive likely root causes and corrective actions ranked by similarity
- Filter by component and model
- Upload new fault records to extend the knowledge base
- REST API with endpoints for health checks, querying components/models, diagnosing, and ingesting new data

## Project Structure
- `app/`
  - `backend/` – FastAPI application (`main.py`)
  - `rca/` – embedding, indexing, and search logic
  - `ingest/` – script for rebuilding or updating the index
- `frontend/` – Streamlit user interface
- `data/` – sample CSV dataset used for the initial index
- `Dockerfile`, `docker-compose.yml` – container setup

## Requirements
- Python 3.11+
- An OpenAI API key (for embeddings)

## Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY
# Start backend
uvicorn app.backend.main:app --reload --port 8000
# Start frontend in a separate shell
streamlit run frontend/streamlit_app.py
```

Once running:
- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://127.0.0.1:8501

Use the sidebar in the UI to upload CSV files (columns: `component`, `fault_description`, `root_cause`, `corrective_action`, optional `model`) to extend the knowledge base.

## Rebuild or Update the Index
```bash
python -m app.ingest.build_index            # update from data/faults.csv
python -m app.ingest.build_index --rebuild  # rebuild from scratch
```

## Run with Docker
1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Build and run:
```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- UI:  http://localhost:8501

## Contributing
Issues and pull requests are welcome. Please ensure tests run with `pytest`.
