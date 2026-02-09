from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.schemas import QueryRequest
from app.retrieval.graph_retriever import GraphRetriever
from app.services.embedder import HashEmbedder
from app.services.extractor import HeuristicExtractor


class FakeStore:
    def __init__(self) -> None:
        self.embedder = HashEmbedder()
        self.marked_turns: list[str] = []
        self.entities = [
            {
                "entity_id": "ent:cb",
                "canonical_name": "continuous batching",
                "aliases": ["continuous batching"],
                "entity_type": "algorithm",
                "description": "",
                "embedding": self.embedder.embed("continuous batching"),
            },
            {
                "entity_id": "ent:pa",
                "canonical_name": "paged attention",
                "aliases": ["paged attention"],
                "entity_type": "algorithm",
                "description": "",
                "embedding": self.embedder.embed("paged attention"),
            },
            {
                "entity_id": "ent:kv",
                "canonical_name": "kv-cache",
                "aliases": ["kv cache"],
                "entity_type": "concept",
                "description": "",
                "embedding": self.embedder.embed("kv-cache"),
            },
        ]

    def search_entities(self, query_text: str, limit: int = 25):
        q = query_text.lower()
        matches = [
            item
            for item in self.entities
            if q in item["canonical_name"] or any(q in alias for alias in item["aliases"])
        ]
        return matches[:limit]

    def get_neighbors(self, entity_id: str, limit: int = 100):
        if entity_id == "ent:cb":
            return [
                {
                    "source_entity_id": "ent:cb",
                    "source_name": "continuous batching",
                    "target_entity_id": "ent:pa",
                    "target_name": "paged attention",
                    "target_embedding": self.embedder.embed("paged attention"),
                    "relation_type": "DEPENDS_ON",
                    "relation_weight": 2.0,
                    "evidence_turn_ids": ["conv-1:t4"],
                }
            ]
        if entity_id == "ent:pa":
            return [
                {
                    "source_entity_id": "ent:pa",
                    "source_name": "paged attention",
                    "target_entity_id": "ent:kv",
                    "target_name": "kv-cache",
                    "target_embedding": self.embedder.embed("kv-cache"),
                    "relation_type": "IMPLEMENTS",
                    "relation_weight": 1.0,
                    "evidence_turn_ids": ["conv-1:t4"],
                }
            ]
        return []

    def fetch_turns_for_entities(self, entity_ids, conversation_id=None, limit: int = 400):
        _ = entity_ids, conversation_id, limit
        return [
            {
                "turn_uid": "conv-1:t2",
                "conversation_id": "conv-1",
                "turn_id": "t2",
                "speaker": "assistant",
                "text": "continuous batching은 처리량을 높인다",
                "timestamp": datetime(2026, 2, 9, 9, 0, 10, tzinfo=timezone.utc),
                "embedding": self.embedder.embed("continuous batching 처리량"),
                "importance_score": 0.0,
                "last_recalled_at": None,
                "recall_count": 0,
                "matched_entity_ids": ["ent:cb"],
                "matched_entities": ["continuous batching"],
            },
            {
                "turn_uid": "conv-1:t4",
                "conversation_id": "conv-1",
                "turn_id": "t4",
                "speaker": "assistant",
                "text": "paged attention은 kv-cache를 효율적으로 다룬다",
                "timestamp": datetime(2026, 2, 9, 9, 0, 38, tzinfo=timezone.utc),
                "embedding": self.embedder.embed("paged attention kv-cache 효율"),
                "importance_score": 0.0,
                "last_recalled_at": None,
                "recall_count": 0,
                "matched_entity_ids": ["ent:pa", "ent:kv"],
                "matched_entities": ["paged attention", "kv-cache"],
            },
        ]

    def mark_turns_recalled(self, turn_uids, recalled_at):
        _ = recalled_at
        self.marked_turns = list(turn_uids)


def test_graph_retrieval_prefers_relation_supported_turn() -> None:
    store = FakeStore()
    retriever = GraphRetriever(store=store, extractor=HeuristicExtractor(), embedder=HashEmbedder())

    response = retriever.query(
        QueryRequest(query="continuous batching이 paged attention과 어떤 관계야?", top_k=2)
    )

    assert response.selected_turns
    assert response.selected_turns[0].turn_uid == "conv-1:t4"
    assert response.selected_turns[0].path_summary
    assert store.marked_turns == ["conv-1:t4", "conv-1:t2"]


def test_importance_weight_can_change_top_rank() -> None:
    class ImportanceStore(FakeStore):
        def fetch_turns_for_entities(self, entity_ids, conversation_id=None, limit: int = 400):
            rows = super().fetch_turns_for_entities(entity_ids, conversation_id, limit)
            rows[0]["importance_score"] = 0.9
            rows[1]["importance_score"] = 0.0
            return rows

    store = ImportanceStore()
    retriever = GraphRetriever(store=store, extractor=HeuristicExtractor(), embedder=HashEmbedder())

    response = retriever.query(
        QueryRequest(
            query="continuous batching이 paged attention과 어떤 관계야?",
            top_k=1,
            importance_weight=1.0,
            recency_weight=0.0,
        )
    )

    assert response.selected_turns
    assert response.selected_turns[0].turn_uid == "conv-1:t2"


def test_recency_weight_can_change_top_rank() -> None:
    class RecencyStore(FakeStore):
        def fetch_turns_for_entities(self, entity_ids, conversation_id=None, limit: int = 400):
            rows = super().fetch_turns_for_entities(entity_ids, conversation_id, limit)
            now = datetime(2026, 2, 9, 9, 2, 0, tzinfo=timezone.utc)
            rows[0]["last_recalled_at"] = now
            rows[1]["last_recalled_at"] = now - timedelta(days=30)
            return rows

    store = RecencyStore()
    retriever = GraphRetriever(store=store, extractor=HeuristicExtractor(), embedder=HashEmbedder())

    response = retriever.query(
        QueryRequest(
            query="continuous batching이 paged attention과 어떤 관계야?",
            top_k=1,
            importance_weight=0.0,
            recency_weight=1.0,
            recall_half_life_hours=24,
        )
    )

    assert response.selected_turns
    assert response.selected_turns[0].turn_uid == "conv-1:t2"
