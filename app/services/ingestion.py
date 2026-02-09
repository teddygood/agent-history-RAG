from __future__ import annotations

from collections import Counter
import math
from typing import Iterable
from collections.abc import Callable, Sized

from app.core.config import settings
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
        batch_size: int | None = None,
        skip_existing_turns: bool = False,
    ) -> IngestStats:
        stats = IngestStats()
        relation_counter: Counter[str] = Counter()
        chunk_profile_counter: Counter[str] = Counter()
        total_chunks = 0
        max_chunks = 0

        total_turns: int | None = len(turns) if isinstance(turns, Sized) else None
        resolved_batch_size = max(1, int(batch_size or settings.ingest_batch_size))
        input_processed = 0
        skipped_turns = 0
        last_progress_at = 0
        pending: list[TurnInput] = []

        def report_progress(*, force: bool = False) -> None:
            nonlocal last_progress_at
            if not progress:
                return
            if force or progress_every <= 0 or (input_processed - last_progress_at) >= progress_every:
                last_progress_at = input_processed
                progress(stats, input_processed, total_turns)

        def flush_batch(batch: list[TurnInput]) -> bool:
            nonlocal input_processed, skipped_turns, total_chunks, max_chunks
            if not batch:
                return True

            batch_uids = [f"{turn.conversation_id}:{turn.turn_id}" for turn in batch]
            existing: set[str] = set()
            if skip_existing_turns and hasattr(self.store, "get_existing_turn_uids"):
                try:
                    existing = getattr(self.store, "get_existing_turn_uids")(batch_uids)
                except Exception:
                    existing = set()

            to_process: list[TurnInput] = []
            for turn, uid in zip(batch, batch_uids):
                input_processed += 1
                if cancel and cancel():
                    stats.debug["cancelled"] = True
                    return False
                if skip_existing_turns and uid in existing:
                    skipped_turns += 1
                    continue
                to_process.append(turn)

            if not to_process:
                report_progress()
                return True

            # 1) Chunk plans + embed all chunks in one shot.
            planned: list[tuple[TurnInput, object, int, int]] = []
            flat_chunks: list[str] = []
            for turn in to_process:
                chunk_plan = self.chunker.plan(turn.text, profile=chunk_profile)
                start = len(flat_chunks)
                flat_chunks.extend(list(chunk_plan.chunks))
                end = len(flat_chunks)
                planned.append((turn, chunk_plan, start, end))

                chunk_profile_counter[chunk_plan.profile_name] += 1
                chunk_count = len(chunk_plan.chunks)
                total_chunks += chunk_count
                max_chunks = max(max_chunks, chunk_count)

            chunk_vectors = self.embedder.embed_documents(flat_chunks) if flat_chunks else []

            turn_rows: list[dict[str, object]] = []
            entity_map: dict[str, dict[str, object]] = {}
            mention_rows: list[dict[str, str]] = []
            relation_acc: dict[tuple[str, str, str], dict[str, object]] = {}

            for turn, chunk_plan, start, end in planned:
                turn_uid = f"{turn.conversation_id}:{turn.turn_id}"
                per_turn_vectors = chunk_vectors[start:end] if chunk_vectors else []
                turn_embedding = (
                    self._average_vectors(per_turn_vectors)
                    if per_turn_vectors
                    else self.embedder.embed_document("")
                )
                importance_score = score_turn_importance(turn.text)
                chunk_count = len(chunk_plan.chunks)

                turn_rows.append(
                    {
                        "turn_uid": turn_uid,
                        "conversation_id": turn.conversation_id,
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker,
                        "text": turn.text,
                        "timestamp": turn.timestamp,
                        "embedding": turn_embedding,
                        "importance_score": importance_score,
                        "chunk_profile": chunk_plan.profile_name,
                        "chunk_count": chunk_count,
                        "chunk_size": chunk_plan.chunk_size,
                        "chunk_overlap": chunk_plan.chunk_overlap,
                    }
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

                    stats.entities += 1
                    row = entity_map.get(entity_id)
                    if row is None:
                        entity_map[entity_id] = {
                            "entity_id": entity_id,
                            "canonical_name": canonical_name,
                            "entity_type": entity.entity_type,
                            "description": entity.description,
                            # Entity embeddings are secondary signals; skip to keep ingest fast.
                            "embedding": [],
                            "aliases": [entity.surface] if entity.surface else [],
                        }
                    else:
                        aliases = row.get("aliases")
                        if isinstance(aliases, list) and entity.surface and entity.surface not in aliases:
                            aliases.append(entity.surface)
                        if not row.get("description") and entity.description:
                            row["description"] = entity.description

                    mention_rows.append({"turn_uid": turn_uid, "entity_id": entity_id})

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

                    stats.relations += 1
                    relation_counter[relation.relation_type] += 1

                    key = (source_id, target_id, relation.relation_type)
                    current = relation_acc.get(key)
                    if current is None:
                        relation_acc[key] = {
                            "source_entity_id": source_id,
                            "target_entity_id": target_id,
                            "relation_type": relation.relation_type,
                            "weight_increment": 1.0,
                            "evidence_turn_ids": {turn_uid},
                        }
                    else:
                        current["weight_increment"] = float(current.get("weight_increment", 0.0)) + 1.0
                        evidence = current.get("evidence_turn_ids")
                        if isinstance(evidence, set):
                            evidence.add(turn_uid)

            # 2) Persist as batches (fallback to per-row methods for tests).
            if hasattr(self.store, "upsert_turns_batch"):
                getattr(self.store, "upsert_turns_batch")(turn_rows)
            else:
                for row in turn_rows:
                    self.store.upsert_turn(**row)  # type: ignore[arg-type]

            entity_rows = list(entity_map.values())
            if entity_rows:
                if hasattr(self.store, "upsert_entities_batch"):
                    getattr(self.store, "upsert_entities_batch")(entity_rows)
                else:
                    for row in entity_rows:
                        aliases = row.get("aliases") if isinstance(row.get("aliases"), list) else []
                        alias = aliases[0] if aliases else str(row.get("canonical_name") or "")
                        self.store.upsert_entity(
                            entity_id=str(row["entity_id"]),
                            canonical_name=str(row["canonical_name"]),
                            alias=alias,
                            entity_type=str(row.get("entity_type") or "concept"),
                            description=str(row.get("description") or ""),
                            embedding=list(row.get("embedding") or []),
                        )

            if mention_rows:
                if hasattr(self.store, "link_turn_entities_batch"):
                    getattr(self.store, "link_turn_entities_batch")(mention_rows)
                else:
                    for row in mention_rows:
                        self.store.link_turn_entity(turn_uid=row["turn_uid"], entity_id=row["entity_id"])

            relation_rows: list[dict[str, object]] = []
            for payload in relation_acc.values():
                evidence = payload.get("evidence_turn_ids")
                payload["evidence_turn_ids"] = sorted(list(evidence)) if isinstance(evidence, set) else []
                relation_rows.append(payload)

            if relation_rows:
                if hasattr(self.store, "upsert_relations_batch"):
                    getattr(self.store, "upsert_relations_batch")(relation_rows)
                else:
                    for row in relation_rows:
                        evidence_ids = row.get("evidence_turn_ids")
                        evidence_list = evidence_ids if isinstance(evidence_ids, list) else []
                        for evidence_turn_uid in evidence_list[:1]:
                            self.store.upsert_relation(
                                source_entity_id=str(row["source_entity_id"]),
                                target_entity_id=str(row["target_entity_id"]),
                                relation_type=str(row["relation_type"]),
                                evidence_turn_uid=str(evidence_turn_uid),
                                weight_increment=float(row.get("weight_increment", 1.0)),
                            )

            report_progress()
            return True

        for turn in turns:
            if cancel and cancel():
                stats.debug["cancelled"] = True
                break
            pending.append(turn)
            if len(pending) >= resolved_batch_size:
                if not flush_batch(pending):
                    break
                pending = []

        if pending and not (cancel and cancel()):
            flush_batch(pending)

        stats.debug["relation_distribution"] = dict(relation_counter)
        stats.debug["chunk_profile_distribution"] = dict(chunk_profile_counter)
        stats.debug["avg_chunks_per_turn"] = round(total_chunks / max(1, stats.turns), 3)
        stats.debug["max_chunks_per_turn"] = max_chunks
        stats.debug["chunk_profile"] = chunk_profile
        stats.debug["total_turns"] = total_turns
        stats.debug["batch_size"] = resolved_batch_size
        stats.debug["skipped_turns"] = skipped_turns
        stats.debug["skip_existing_turns"] = bool(skip_existing_turns)
        report_progress(force=True)
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

        return self._average_vectors(vectors)

    @staticmethod
    def _average_vectors(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
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
