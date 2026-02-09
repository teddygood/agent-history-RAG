from __future__ import annotations

from collections import Counter
import math
from typing import Iterable
from collections.abc import Callable, Sized

from app.models.schemas import IngestStats, TurnInput
from app.services.embedder import Embedder, HashEmbedder
from app.services.chunker import ChunkProfileName, TextChunker
from app.services.extractor import HeuristicExtractor
from app.services.importance import score_turn_importance
from app.services.neo4j_store import Neo4jStore
from app.services.normalize import canonicalize, to_entity_id


class IngestionService:
    def __init__(
        self,
        store: Neo4jStore,
        extractor: HeuristicExtractor | None = None,
        embedder: Embedder | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or HeuristicExtractor()
        self.embedder = embedder or HashEmbedder()
        self.chunker = chunker or TextChunker()

    def ingest_turns(
        self,
        turns: Iterable[TurnInput],
        *,
        chunk_profile: ChunkProfileName = "auto",
        progress: Callable[[IngestStats, int, int | None], None] | None = None,
        progress_every: int = 25,
        cancel: Callable[[], bool] | None = None,
    ) -> IngestStats:
        stats = IngestStats()
        relation_counter: Counter[str] = Counter()
        chunk_profile_counter: Counter[str] = Counter()
        total_chunks = 0
        max_chunks = 0

        total_turns: int | None = len(turns) if isinstance(turns, Sized) else None

        for turn in turns:
            if cancel and cancel():
                stats.debug["cancelled"] = True
                break
            turn_uid = f"{turn.conversation_id}:{turn.turn_id}"
            chunk_plan = self.chunker.plan(turn.text, profile=chunk_profile)
            turn_embedding = self._embed_chunks(chunk_plan.chunks)
            importance_score = score_turn_importance(turn.text)
            chunk_count = len(chunk_plan.chunks)
            self.store.upsert_turn(
                turn_uid=turn_uid,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=turn.text,
                timestamp=turn.timestamp,
                embedding=turn_embedding,
                importance_score=importance_score,
                chunk_profile=chunk_plan.profile_name,
                chunk_count=chunk_count,
                chunk_size=chunk_plan.chunk_size,
                chunk_overlap=chunk_plan.chunk_overlap,
            )
            stats.turns += 1
            chunk_profile_counter[chunk_plan.profile_name] += 1
            total_chunks += chunk_count
            max_chunks = max(max_chunks, chunk_count)

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
                    embedding=self.embedder.embed_document(canonical_name),
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

            if progress and progress_every > 0 and (stats.turns % progress_every == 0):
                progress(stats, stats.turns, total_turns)

        stats.debug["relation_distribution"] = dict(relation_counter)
        stats.debug["chunk_profile_distribution"] = dict(chunk_profile_counter)
        stats.debug["avg_chunks_per_turn"] = round(total_chunks / max(1, stats.turns), 3)
        stats.debug["max_chunks_per_turn"] = max_chunks
        stats.debug["chunk_profile"] = chunk_profile
        stats.debug["total_turns"] = total_turns
        if progress:
            progress(stats, stats.turns, total_turns)
        return stats

    def get_chunking_settings(self) -> dict[str, int | float]:
        return {
            "chunk_size_default": self.chunker.default_size,
            "chunk_overlap_default": self.chunker.default_overlap,
            "chunk_size_structured": self.chunker.structured_size,
            "chunk_overlap_structured": self.chunker.structured_overlap,
            "chunk_size_max": self.chunker.max_chunk_size,
            "chunk_structure_min_lines": self.chunker.structure_min_lines,
            "chunk_structure_heading_ratio": self.chunker.structure_heading_ratio,
        }

    def _embed_chunks(self, chunks: tuple[str, ...]) -> list[float]:
        materialized = [chunk for chunk in chunks if chunk]
        vectors = self.embedder.embed_documents(materialized) if materialized else []
        if not vectors:
            return self.embedder.embed_document("")
        if len(vectors) == 1:
            return vectors[0]

        dim = len(vectors[0])
        accum = [0.0] * dim
        count = 0
        for vec in vectors:
            if len(vec) != dim:
                continue
            count += 1
            for idx, value in enumerate(vec):
                accum[idx] += value
        if count <= 0:
            return vectors[0]

        averaged = [value / count for value in accum]
        norm = math.sqrt(sum(value * value for value in averaged))
        if norm <= 1e-9:
            return averaged
        return [value / norm for value in averaged]
