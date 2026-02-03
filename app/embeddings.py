from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(_MODEL_NAME)


@dataclass
class VectorItem:
    chunk_db_id: int
    paper_id: str
    chunk_id: str
    section: str = "unknown"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    lang: str = "en"


class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.items: List[VectorItem] = []
        self.texts: List[str] = []   # ✅ store chunk texts in same order

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, texts: List[str], metadatas: List[VectorItem]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")
        if not texts:
            return

        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        self.items.extend(metadatas)
        self.texts.extend(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns list of {score, text, metadata}"""
        if len(self.items) == 0:
            return []

        q = model.encode([query], show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        q = self._normalize(q)

        scores, idxs = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if 0 <= idx < len(self.items):
                results.append({
                    "score": float(score),
                    "text": self.texts[idx],
                    "metadata": self.items[idx],
                })
        return results


# Global store instance
vector_store = VectorStore(dim=384)
