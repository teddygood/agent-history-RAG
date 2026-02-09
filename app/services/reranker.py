from __future__ import annotations

import logging
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    provider: str
    model_name: str

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        ...

    def status(self) -> dict[str, object]:
        ...


class NoopReranker:
    provider = "none"
    model_name = "none"

    def __init__(self) -> None:
        self.last_error: str | None = None

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        _ = query
        return [0.0 for _ in documents]

    def status(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "loaded": False,
            "available": False,
            "last_error": self.last_error,
        }


class CrossEncoderReranker:
    provider = "sentence-transformers"

    def __init__(
        self,
        model_name: str | None = None,
        allow_fallback_to_noop: bool | None = None,
    ) -> None:
        self.model_name = (model_name or settings.reranker_model_primary).strip()
        self.allow_fallback_to_noop = (
            settings.reranker_fallback_to_base
            if allow_fallback_to_noop is None
            else allow_fallback_to_noop
        )
        self._model = None
        self._loaded = False
        self.last_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None

        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
            self._loaded = True
            self.last_error = None
            return True
        except Exception as exc:  # pragma: no cover - dependency optional
            self._loaded = True
            self._model = None
            self.last_error = str(exc)
            if self.allow_fallback_to_noop:
                logger.warning("Reranker unavailable. Falling back to base rank only: %s", exc)
                return False
            raise RuntimeError(f"Failed to initialize reranker '{self.model_name}': {exc}") from exc

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        if not self._ensure_loaded():
            return [0.0 for _ in documents]

        assert self._model is not None
        pairs = [(query, document) for document in documents]
        try:
            outputs = self._model.predict(pairs)
            if hasattr(outputs, "tolist"):
                outputs = outputs.tolist()
            return [float(value) for value in outputs]
        except Exception as exc:  # pragma: no cover - runtime safety
            self.last_error = str(exc)
            if self.allow_fallback_to_noop:
                logger.warning("Reranker inference failed. Keeping base rank only: %s", exc)
                return [0.0 for _ in documents]
            raise RuntimeError(f"Reranker inference failed: {exc}") from exc

    def status(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "loaded": self._loaded,
            "available": self._model is not None,
            "last_error": self.last_error,
        }


def create_reranker(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    allow_fallback_to_noop: bool | None = None,
) -> Reranker:
    selected = (provider or "auto").strip().lower()

    if selected in {"none", "off", "disabled"}:
        return NoopReranker()

    if selected in {"sentence-transformers", "sentence_transformers", "st", "auto"}:
        return CrossEncoderReranker(
            model_name=model_name,
            allow_fallback_to_noop=allow_fallback_to_noop,
        )

    raise ValueError(
        f"Unsupported reranker provider '{selected}'. "
        "Use auto, sentence-transformers, or none."
    )
