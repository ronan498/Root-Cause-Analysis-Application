from typing import List, Dict, Optional
from .embedder import OpenAIEmbedder
from .index import RCAIndex

class QueryEngine:
    def __init__(self, index: RCAIndex, embedder: OpenAIEmbedder):
        self.index = index
        self.embedder = embedder

    def diagnose(self, query: str, top_k: int = 3, component: Optional[str] = None, model: Optional[str] = None) -> List[Dict]:
        qv = self.embedder.embed_text(query)
        results = self.index.search(qv, top_k=top_k, component=component, model=model)
        out = []
        for idx, score in results:
            row = self.index.meta[idx]
            out.append({
                "component": row.get("component"),
                "model": row.get("model") or None,
                "matched_fault_description": row.get("fault_description"),
                "root_cause": row.get("root_cause"),
                "corrective_action": row.get("corrective_action"),
                "similarity": float(score)
            })
        return out
