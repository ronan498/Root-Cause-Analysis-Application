import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = DATA_DIR / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / os.getenv("FAULTS_CSV", "faults.csv")
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
