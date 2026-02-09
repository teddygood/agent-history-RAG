from __future__ import annotations

import hashlib
import math
import re

from app.core.config import settings


class HashEmbedder:
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
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_\-+/]+", text.lower())
