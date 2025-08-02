from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import threading
import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .config import FAISS_INDEX_PATH, META_PATH

def _normalise(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10 if v.ndim == 2 else np.linalg.norm(v) + 1e-10
    return v / denom

class RCAIndex:
    """In-memory FAISS (or numpy) index + metadata.
    Cosine similarity via L2-normalised vectors and inner-product search.
    Persists FAISS index and metadata (including vectors) to disk.
    Thread-safe for add/search via a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.meta: List[Dict] = []
        self.faiss_index = None
        self.dim = None

    def save(self):
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        if self.faiss_index is not None and faiss is not None:
            import faiss as _faiss
            _faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))

    def load(self) -> bool:
        if META_PATH.exists():
            self.meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
        else:
            return False
        if FAISS_INDEX_PATH.exists() and faiss is not None:
            import faiss as _faiss
            self.faiss_index = _faiss.read_index(str(FAISS_INDEX_PATH))
            self.dim = self.faiss_index.d
        else:
            self.faiss_index = None
            self.dim = None if not self.meta else len(self.meta[0].get("embedding", []))
        return True

    def build(self, embeddings: np.ndarray, rows: List[Dict]):
        with self.lock:
            vecs = _normalise(embeddings.astype("float32"))
            self.dim = vecs.shape[1]
            if faiss is not None:
                import faiss as _faiss
                self.faiss_index = _faiss.IndexFlatIP(self.dim)
                self.faiss_index.add(vecs)
            self.meta = []
            for i, r in enumerate(rows):
                r2 = dict(r)
                r2["embedding"] = vecs[i].tolist()
                self.meta.append(r2)
        self.save()

    def add(self, embeddings: np.ndarray, rows: List[Dict]):
        with self.lock:
            vecs = _normalise(embeddings.astype("float32"))
            if self.dim is None:
                self.dim = vecs.shape[1]
            if faiss is not None and self.faiss_index is None:
                import faiss as _faiss
                self.faiss_index = _faiss.IndexFlatIP(self.dim)
            if faiss is not None:
                self.faiss_index.add(vecs)
            for i, r in enumerate(rows):
                r2 = dict(r)
                r2["embedding"] = vecs[i].tolist()
                self.meta.append(r2)
        self.save()

    def search(self, query_vec: np.ndarray, top_k: int = 3, component: Optional[str] = None) -> List[Tuple[int, float]]:
        q = _normalise(query_vec.astype("float32")).reshape(1, -1)
        with self.lock:
            ids = list(range(len(self.meta)))
            if component:
                ids = [i for i, m in enumerate(self.meta) if m.get("component", "").lower() == component.lower()]
                if not ids:
                    return []
            vecs = np.array([self.meta[i]["embedding"] for i in ids], dtype="float32")
            sims = (vecs @ q.T).ravel()
            order = np.argsort(-sims)[:top_k]
            return [(int(ids[o]), float(sims[o])) for o in order]
