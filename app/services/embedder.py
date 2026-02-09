from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import math
import re
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    provider: str
    model_name: str
    dim: int

    def embed(self, text: str) -> list[float]:
        ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_document(self, text: str) -> list[float]:
        ...

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        ...


class _BaseEmbedder:
    provider = "base"
    model_name = "base"

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

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
    provider = "hash"
    model_name = "hash-deterministic"

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
        return re.findall(r"[a-zA-Z0-9가-힣_\-+/]+", text.lower())


@dataclass(frozen=True)
class EmbeddingModelProfile:
    model_name: str
    query_prefix: str = ""
    document_prefix: str = ""


KNOWN_MODEL_PROFILES = {
    "nlpai-lab/kure-v1": EmbeddingModelProfile(
        model_name="nlpai-lab/KURE-v1",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "dragonkue/bge-m3-ko": EmbeddingModelProfile(
        model_name="dragonkue/BGE-m3-ko",
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
    ),
}


def _resolve_model_profile(
    *,
    model_name: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> EmbeddingModelProfile:
    selected_name = (model_name or settings.embedding_model_primary).strip()
    known = KNOWN_MODEL_PROFILES.get(selected_name.lower())
    canonical_name = known.model_name if known else selected_name

    configured_query_prefix = settings.embedding_query_prefix.strip()
    configured_document_prefix = settings.embedding_document_prefix.strip()

    resolved_query_prefix = (
        query_prefix
        if query_prefix is not None
        else configured_query_prefix if configured_query_prefix else (known.query_prefix if known else "")
    )
    resolved_document_prefix = (
        document_prefix
        if document_prefix is not None
        else configured_document_prefix if configured_document_prefix else (known.document_prefix if known else "")
    )

    return EmbeddingModelProfile(
        model_name=canonical_name,
        query_prefix=resolved_query_prefix,
        document_prefix=resolved_document_prefix,
    )


class SentenceTransformerEmbedder(_BaseEmbedder):
    """
    Optional production-grade embedder backed by sentence-transformers.
    Loaded lazily so local hash mode works without extra dependencies.
    """
    provider = "sentence-transformers"

    def __init__(
        self,
        model_name: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install optional dependencies or switch EMBEDDING_PROVIDER=hash."
            ) from exc

        profile = _resolve_model_profile(
            model_name=model_name,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
        self.model_name = profile.model_name
        self.query_prefix = profile.query_prefix
        self.document_prefix = profile.document_prefix
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension() or settings.embedding_dim)

    def embed(self, text: str) -> list[float]:
        encoded = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return self._coerce_matrix(encoded)[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self.query_prefix}{text}" if self.query_prefix else text for text in texts]
        encoded = self.model.encode(prepared, normalize_embeddings=True, show_progress_bar=False)
        return self._coerce_matrix(encoded)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self.document_prefix}{text}" if self.document_prefix else text for text in texts]
        encoded = self.model.encode(prepared, normalize_embeddings=True, show_progress_bar=False)
        return self._coerce_matrix(encoded)

    def embed_query(self, text: str) -> list[float]:
        return self.embed(f"{self.query_prefix}{text}" if self.query_prefix else text)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(f"{self.document_prefix}{text}" if self.document_prefix else text)

    @staticmethod
    def _coerce_matrix(matrix: object) -> list[list[float]]:
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()  # type: ignore[assignment]
        if isinstance(matrix, list):
            return [[float(v) for v in row] for row in matrix]
        # Fallback: treat as iterable of iterables
        return [[float(v) for v in row] for row in matrix]  # type: ignore[arg-type]


def create_embedder(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
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
            return SentenceTransformerEmbedder(
                model_name=model_name,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
        except Exception as exc:
            if fallback_enabled:
                logger.warning("Embedder provider '%s' unavailable. Falling back to hash embedder: %s", selected, exc)
                return HashEmbedder(dim=dim)
            raise RuntimeError(f"Failed to initialize embedder provider '{selected}': {exc}") from exc

    if selected == "auto":
        try:
            return SentenceTransformerEmbedder(
                model_name=model_name,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
        except Exception as exc:
            logger.warning("Auto embedder fallback to hash: %s", exc)
            return HashEmbedder(dim=dim)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{selected}'. Use hash, sentence-transformers, or auto.")
