from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        ...

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        ...


class _BaseEmbedder:
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))


class HashEmbedder(_BaseEmbedder):
    """
    Deterministic embedding for stable local development.
    It is intentionally simple and replaceable.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            h = hashlib.sha1(token.encode("utf-8")).hexdigest()
            bucket = int(h[:8], 16) % self.dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-9:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_\-+/]+", text.lower())


class SentenceTransformerEmbedder(_BaseEmbedder):
    """
    Optional production-grade embedder backed by sentence-transformers.
    Loaded lazily so local hash mode works without extra dependencies.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install optional dependencies or switch EMBEDDING_PROVIDER=hash."
            ) from exc

        self.model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension() or settings.embedding_dim)

    def embed(self, text: str) -> list[float]:
        encoded = self.model.encode([text], normalize_embeddings=True)
        vector = encoded[0]
        if hasattr(vector, "tolist"):
            return [float(v) for v in vector.tolist()]
        return [float(v) for v in vector]


def create_embedder(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    dim: int | None = None,
    allow_fallback_to_hash: bool | None = None,
) -> Embedder:
    selected = (provider or settings.embedding_provider).strip().lower()
    fallback_enabled = (
        settings.embedding_fallback_to_hash if allow_fallback_to_hash is None else allow_fallback_to_hash
    )

    if selected in {"hash", "deterministic"}:
        return HashEmbedder(dim=dim)

    if selected in {"sentence-transformers", "sentence_transformers", "st"}:
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception as exc:
            if fallback_enabled:
                logger.warning("Embedder provider '%s' unavailable. Falling back to hash embedder: %s", selected, exc)
                return HashEmbedder(dim=dim)
            raise RuntimeError(f"Failed to initialize embedder provider '{selected}': {exc}") from exc

    if selected == "auto":
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception as exc:
            logger.warning("Auto embedder fallback to hash: %s", exc)
            return HashEmbedder(dim=dim)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{selected}'. Use hash, sentence-transformers, or auto.")
