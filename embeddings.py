"""Text embedder. Uses SentenceTransformers all-MiniLM-L6-v2 (384-d).

If the transformer model can't be loaded (e.g. no internet on first run), it falls back to a
384-d hashing vectorizer so semantic search still works, with lower quality. The embedder
name is stored with the vectors so queries are always encoded the same way as the documents.
"""
from functools import lru_cache

import numpy as np

import config

HASHING_NAME = "hashing-384"


class Embedder:
    def __init__(self, name: str | None = None):
        self._st = None
        self._hv = None
        if name in (None, config.EMBEDDING_MODEL):
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(config.EMBEDDING_MODEL)
                self.name = config.EMBEDDING_MODEL
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[embeddings] {config.EMBEDDING_MODEL} unavailable ({exc}); using {HASHING_NAME}")
        from sklearn.feature_extraction.text import HashingVectorizer

        self._hv = HashingVectorizer(n_features=config.EMBEDDING_DIM, alternate_sign=False,
                                     norm="l2", ngram_range=(1, 2), stop_words="english")
        self.name = HASHING_NAME

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return self._hv.transform(texts).toarray().astype(np.float32)


@lru_cache(maxsize=2)
def get_embedder(name: str | None = None) -> Embedder:
    return Embedder(name)
