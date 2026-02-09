from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from app.core.config import settings

ChunkProfileName = Literal["auto", "default", "structured"]


@dataclass(frozen=True)
class ChunkProfile:
    name: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class ChunkPlan:
    profile_name: str
    chunk_size: int
    chunk_overlap: int
    is_structured: bool
    chunks: tuple[str, ...]


class TextChunker:
    def __init__(
        self,
        *,
        default_size: int | None = None,
        default_overlap: int | None = None,
        structured_size: int | None = None,
        structured_overlap: int | None = None,
        max_chunk_size: int | None = None,
        structure_min_lines: int | None = None,
        structure_heading_ratio: float | None = None,
    ) -> None:
        self.default_size = max(8, default_size or settings.chunk_size_default)
        self.default_overlap = max(0, default_overlap if default_overlap is not None else settings.chunk_overlap_default)
        self.structured_size = max(8, structured_size or settings.chunk_size_structured)
        self.structured_overlap = max(
            0,
            structured_overlap if structured_overlap is not None else settings.chunk_overlap_structured,
        )
        self.max_chunk_size = max(8, max_chunk_size or settings.chunk_size_max)
        self.structure_min_lines = max(1, structure_min_lines or settings.chunk_structure_min_lines)
        self.structure_heading_ratio = max(0.0, min(1.0, structure_heading_ratio or settings.chunk_structure_heading_ratio))

    def plan(self, text: str, profile: ChunkProfileName = "auto") -> ChunkPlan:
        normalized = text.strip()
        if not normalized:
            return ChunkPlan(
                profile_name="default",
                chunk_size=self.default_size,
                chunk_overlap=self.default_overlap,
                is_structured=False,
                chunks=("",),
            )

        selected = self._select_profile(normalized, profile=profile)
        words = self._tokenize_words(normalized)

        if len(words) <= selected.chunk_size:
            chunks = (normalized,)
        else:
            chunks = self._chunk_words(words, selected.chunk_size, selected.chunk_overlap)

        return ChunkPlan(
            profile_name=selected.name,
            chunk_size=selected.chunk_size,
            chunk_overlap=selected.chunk_overlap,
            is_structured=(selected.name == "structured"),
            chunks=chunks,
        )

    def _select_profile(self, text: str, profile: ChunkProfileName) -> ChunkProfile:
        if profile == "structured":
            return self._profile_structured()
        if profile == "default":
            return self._profile_default()
        if self._is_structured_text(text):
            return self._profile_structured()
        return self._profile_default()

    def _profile_default(self) -> ChunkProfile:
        return ChunkProfile(
            name="default",
            chunk_size=min(self.default_size, self.max_chunk_size),
            chunk_overlap=min(self.default_overlap, max(0, self.default_size - 1)),
        )

    def _profile_structured(self) -> ChunkProfile:
        return ChunkProfile(
            name="structured",
            chunk_size=min(self.structured_size, self.max_chunk_size),
            chunk_overlap=min(self.structured_overlap, max(0, self.structured_size - 1)),
        )

    def _is_structured_text(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < self.structure_min_lines:
            return False

        heading_like = 0
        for line in lines:
            if re.match(r"^(#{1,6}\s+|[\-\*\u2022]\s+|\d+[\.\)]\s+)", line):
                heading_like += 1
                continue
            if len(line) <= 64 and line.endswith(":"):
                heading_like += 1
                continue
            if re.match(r"^[가-힣A-Za-z0-9 _/\-]{2,48}$", line) and not re.search(r"[.!?]$", line):
                heading_like += 1

        ratio = heading_like / max(1, len(lines))
        return ratio >= self.structure_heading_ratio

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        return re.findall(r"\S+", text)

    @staticmethod
    def _chunk_words(words: list[str], chunk_size: int, chunk_overlap: int) -> tuple[str, ...]:
        step = max(1, chunk_size - chunk_overlap)
        chunks: list[str] = []
        start = 0
        n = len(words)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start += step
        return tuple(chunks)
