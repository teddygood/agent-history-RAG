from __future__ import annotations

from datetime import datetime, timezone

from app.models.schemas import TurnInput
from app.services.chunker import TextChunker
from app.services.embedder import HashEmbedder
from app.services.extractor import HeuristicExtractor
from app.services.ingestion import IngestionService


class FakeStore:
    def __init__(self) -> None:
        self.upsert_turn_calls: list[dict] = []

    def upsert_turn(self, **kwargs) -> None:
        self.upsert_turn_calls.append(kwargs)

    def upsert_entity(self, **kwargs) -> None:
        _ = kwargs

    def link_turn_entity(self, **kwargs) -> None:
        _ = kwargs

    def upsert_relation(self, **kwargs) -> None:
        _ = kwargs


def test_ingestion_applies_chunk_metadata() -> None:
    store = FakeStore()
    chunker = TextChunker(
        default_size=8,
        default_overlap=2,
        structured_size=10,
        structured_overlap=1,
        max_chunk_size=32,
        structure_min_lines=3,
        structure_heading_ratio=0.3,
    )
    service = IngestionService(
        store=store,
        extractor=HeuristicExtractor(),
        embedder=HashEmbedder(dim=16),
        chunker=chunker,
    )
    turns = [
        TurnInput(
            conversation_id="conv-1",
            turn_id="t1",
            speaker="user",
            text="one two three four five six seven eight nine ten",
            timestamp=datetime(2026, 2, 9, 9, 0, 0, tzinfo=timezone.utc),
        )
    ]

    stats = service.ingest_turns(turns, chunk_profile="default")

    assert stats.turns == 1
    assert stats.debug["chunk_profile_distribution"]["default"] == 1
    assert stats.debug["avg_chunks_per_turn"] > 1.0
    assert store.upsert_turn_calls
    call = store.upsert_turn_calls[0]
    assert call["chunk_profile"] == "default"
    assert call["chunk_count"] >= 2
    assert call["chunk_size"] == 8
    assert call["chunk_overlap"] == 2
