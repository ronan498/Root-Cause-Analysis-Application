# RCA Demo (FastAPI + Streamlit + OpenAI Embeddings)

Similarity-based root cause analysis for motors, pumps, and compressors. Supports live uploads to add new components.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY
uvicorn app.backend.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

## Run with Docker

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Build and run:
   ```bash
   docker compose up --build

- API: http://localhost:8000/docs
- UI:  http://localhost:8501
