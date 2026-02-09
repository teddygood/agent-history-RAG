from __future__ import annotations

from app.services.chunker import TextChunker


def test_chunker_auto_uses_default_for_plain_text() -> None:
    chunker = TextChunker(
        default_size=8,
        default_overlap=2,
        structured_size=10,
        structured_overlap=2,
        max_chunk_size=32,
        structure_min_lines=3,
        structure_heading_ratio=0.3,
    )
    plan = chunker.plan("one two three four five six seven eight nine ten", profile="auto")

    assert plan.profile_name == "default"
    assert len(plan.chunks) == 2
    assert plan.chunk_size == 8
    assert plan.chunk_overlap == 2


def test_chunker_auto_uses_structured_for_heading_text() -> None:
    chunker = TextChunker(
        default_size=16,
        default_overlap=2,
        structured_size=8,
        structured_overlap=1,
        max_chunk_size=32,
        structure_min_lines=3,
        structure_heading_ratio=0.3,
    )
    text = """
# 섹션
- 항목 1
- 항목 2
요약
""".strip()
    plan = chunker.plan(text, profile="auto")

    assert plan.profile_name == "structured"
    assert plan.chunk_size == 8


def test_chunker_profile_override_works() -> None:
    chunker = TextChunker(
        default_size=12,
        default_overlap=1,
        structured_size=8,
        structured_overlap=1,
        max_chunk_size=32,
        structure_min_lines=3,
        structure_heading_ratio=0.3,
    )
    text = "# A\n- B\n- C\nD"
    plan = chunker.plan(text, profile="default")

    assert plan.profile_name == "default"
    assert plan.chunk_size == 12
