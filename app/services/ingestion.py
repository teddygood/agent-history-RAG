from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.models.schemas import IngestStats, TurnInput
from app.services.embedder import HashEmbedder
from app.services.extractor import HeuristicExtractor
from app.services.importance import score_turn_importance
from app.services.neo4j_store import Neo4jStore
from app.services.normalize import canonicalize, to_entity_id


class IngestionService:
    def __init__(
        self,
        store: Neo4jStore,
        extractor: HeuristicExtractor | None = None,
        embedder: HashEmbedder | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or HeuristicExtractor()
        self.embedder = embedder or HashEmbedder()

    def ingest_turns(self, turns: Iterable[TurnInput]) -> IngestStats:
        stats = IngestStats()
        relation_counter: Counter[str] = Counter()

        for turn in turns:
            turn_uid = f"{turn.conversation_id}:{turn.turn_id}"
            turn_embedding = self.embedder.embed(turn.text)
            importance_score = score_turn_importance(turn.text)
            self.store.upsert_turn(
                turn_uid=turn_uid,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=turn.text,
                timestamp=turn.timestamp,
                embedding=turn_embedding,
                importance_score=importance_score,
            )
            stats.turns += 1

            extracted_entities = self.extractor.extract_entities(turn.text)
            canonical_to_entity_id: dict[str, str] = {}
            seen_entity_ids: set[str] = set()

            for entity in extracted_entities:
                canonical_name = canonicalize(entity.canonical_name)
                entity_id = to_entity_id(canonical_name)
                canonical_to_entity_id[canonical_name] = entity_id

                if entity_id in seen_entity_ids:
                    continue

                seen_entity_ids.add(entity_id)
                self.store.upsert_entity(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    alias=entity.surface,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    embedding=self.embedder.embed(canonical_name),
                )
                self.store.link_turn_entity(turn_uid=turn_uid, entity_id=entity_id)
                stats.entities += 1

            extracted_relations = self.extractor.extract_relations(turn.text, extracted_entities)
            relation_keys: set[tuple[str, str, str]] = set()
            for relation in extracted_relations:
                source_name = canonicalize(relation.source_canonical)
                target_name = canonicalize(relation.target_canonical)
                source_id = canonical_to_entity_id.get(source_name)
                target_id = canonical_to_entity_id.get(target_name)
                if not source_id or not target_id or source_id == target_id:
                    continue

                dedup_key = (source_id, target_id, relation.relation_type)
                if dedup_key in relation_keys:
                    continue
                relation_keys.add(dedup_key)

                self.store.upsert_relation(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=relation.relation_type,
                    evidence_turn_uid=turn_uid,
                    weight_increment=1.0,
                )
                stats.relations += 1
                relation_counter[relation.relation_type] += 1

        stats.debug["relation_distribution"] = dict(relation_counter)
        return stats
