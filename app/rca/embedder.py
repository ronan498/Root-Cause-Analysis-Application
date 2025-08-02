from typing import List
import numpy as np
from openai import OpenAI
from .config import EMBED_MODEL, EMBED_BATCH

class OpenAIEmbedder:
    """Thin wrapper around OpenAI embeddings."""
    def __init__(self):
        self.client = OpenAI()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        arrs = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i+EMBED_BATCH]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            batch_vecs = [d.embedding for d in resp.data]
            arrs.append(np.array(batch_vecs, dtype=np.float32))
        if not arrs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(arrs)

    def embed_text(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)
