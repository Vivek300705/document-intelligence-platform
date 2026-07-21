"""Embedding Module.

Generates dense vector embeddings using sentence-transformers.
"""

from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Embedding model wrapper using sentence-transformers with TF-IDF fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dim: int = 384):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self._model = None
        self._fallback_mode = False

    @property
    def model(self):
        """Lazy load SentenceTransformer model with fallback."""
        if self._model is None and not self._fallback_mode:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"SentenceTransformer not available ({e}). Using deterministic TF-IDF/Feature embedding fallback.")
                self._fallback_mode = True
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of string texts."""
        if not texts:
            return []

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed ({e}). Falling back.")

        # Deterministic lightweight vector fallback
        return [self._hash_vectorize(t) for t in texts]

    def _hash_vectorize(self, text: str) -> List[float]:
        """Generates normalized vector using hashing over target vector dimension."""
        import hashlib
        import math

        vec = [0.0] * self.vector_dim
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Deterministic hash to index
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % self.vector_dim
            val = (h % 100) / 100.0 - 0.5
            vec[idx] += val

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def embed_query(self, query: str) -> List[float]:
        """Generates vector embedding for a single search query."""
        if not query:
            return []
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []



# Model: all-MiniLM-L6-v2 chosen for balance of speed (14ms/query) and accuracy

# Fallback uses deterministic MD5 hashing when sentence-transformers unavailable