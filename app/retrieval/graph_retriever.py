from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse, TracePathStep, TurnResult
from app.services.embedder import Embedder, HashEmbedder
from app.services.extractor import HeuristicExtractor
from app.services.neo4j_store import Neo4jStore
from app.services.normalize import canonicalize


RELATION_PRIORS = {
    "EXPLAINS": 1.00,
    "DEPENDS_ON": 0.95,
    "IMPLEMENTS": 0.92,
    "COMPARES": 0.82,
    "RELATED_TO": 0.70,
}

DEFAULT_IMPORTANCE_WEIGHT = 0.18
DEFAULT_RECENCY_WEIGHT = 0.12
DEFAULT_RECALL_HALF_LIFE_HOURS = 72


@dataclass
class EntityPath:
    entity_id: str
    canonical_name: str
    score: float
    steps: list[TracePathStep] = field(default_factory=list)


class GraphRetriever:
    def __init__(
        self,
        store: Neo4jStore,
        extractor: HeuristicExtractor | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or HeuristicExtractor()
        self.embedder = embedder or HashEmbedder()

    def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = self.embedder.embed(request.query)
        seed_candidates = self._find_seed_entities(request.query)

        max_hops = request.max_hops or settings.graph_max_hops
        beam_width = request.beam_width or settings.graph_beam_width
        prune_threshold = request.prune_threshold if request.prune_threshold is not None else settings.graph_prune_threshold
        importance_weight = (
            request.importance_weight if request.importance_weight is not None else DEFAULT_IMPORTANCE_WEIGHT
        )
        recency_weight = request.recency_weight if request.recency_weight is not None else DEFAULT_RECENCY_WEIGHT
        recall_half_life_hours = request.recall_half_life_hours or DEFAULT_RECALL_HALF_LIFE_HOURS
        applied_params: dict[str, float | int] = {
            "max_hops": max_hops,
            "beam_width": beam_width,
            "prune_threshold": round(prune_threshold, 6),
            "importance_weight": round(importance_weight, 6),
            "recency_weight": round(recency_weight, 6),
            "recall_half_life_hours": recall_half_life_hours,
            "top_k": request.top_k,
        }

        if not seed_candidates:
            return QueryResponse(
                query=request.query,
                top_k=request.top_k,
                matched_seed_entities=[],
                selected_turns=[],
                pruned_paths=0,
                applied_params=applied_params,
            )

        reached, pruned_paths = self._traverse_graph(
            seed_candidates=seed_candidates,
            query_embedding=query_embedding,
            max_hops=max_hops,
            beam_width=beam_width,
            prune_threshold=prune_threshold,
        )

        turn_rows = self.store.fetch_turns_for_entities(
            reached.keys(),
            conversation_id=request.conversation_id,
            limit=max(400, request.top_k * 40),
        )

        turn_results = self._rank_turns(
            request=request,
            turn_rows=turn_rows,
            reached_entities=reached,
            query_embedding=query_embedding,
            importance_weight=importance_weight,
            recency_weight=recency_weight,
            recall_half_life_hours=recall_half_life_hours,
        )
        selected_turns = turn_results[: request.top_k]
        if selected_turns:
            recalled_at = datetime.now(timezone.utc)
            self.store.mark_turns_recalled([turn.turn_uid for turn in selected_turns], recalled_at=recalled_at)
            for turn in selected_turns:
                turn.last_recalled_at = recalled_at

        seed_names = [seed["canonical_name"] for seed in seed_candidates[:8]]
        return QueryResponse(
            query=request.query,
            top_k=request.top_k,
            matched_seed_entities=seed_names,
            selected_turns=selected_turns,
            pruned_paths=pruned_paths,
            applied_params=applied_params,
        )

    def _find_seed_entities(self, query_text: str) -> list[dict[str, Any]]:
        extracted = self.extractor.extract_entities(query_text)
        names = [canonicalize(item.canonical_name) for item in extracted]

        if not names:
            names = [canonicalize(query_text)]

        by_id: dict[str, dict[str, Any]] = {}
        query_embedding = self.embedder.embed(query_text)

        for name in names:
            for entity in self.store.search_entities(name, limit=20):
                embedding = entity.get("embedding", [])
                emb_score = max(0.0, self.embedder.cosine_similarity(query_embedding, embedding)) if embedding else 0.0
                lexical_score = 1.0 if name == entity["canonical_name"] else 0.8
                entity_score = lexical_score * 0.85 + emb_score * 0.15

                current = by_id.get(entity["entity_id"])
                if current is None or entity_score > current["seed_score"]:
                    entity["seed_score"] = entity_score
                    by_id[entity["entity_id"]] = entity

        ranked = sorted(by_id.values(), key=lambda item: item["seed_score"], reverse=True)
        return ranked[: settings.graph_beam_width]

    def _traverse_graph(
        self,
        *,
        seed_candidates: list[dict[str, Any]],
        query_embedding: list[float],
        max_hops: int,
        beam_width: int,
        prune_threshold: float,
    ) -> tuple[dict[str, EntityPath], int]:
        reached: dict[str, EntityPath] = {}
        frontier: list[EntityPath] = []
        pruned = 0

        for seed in seed_candidates:
            path = EntityPath(
                entity_id=seed["entity_id"],
                canonical_name=seed["canonical_name"],
                score=max(0.01, float(seed.get("seed_score", 0.7))),
                steps=[],
            )
            reached[path.entity_id] = path
            frontier.append(path)

        depth = 0
        while frontier and depth < max_hops:
            candidates: list[EntityPath] = []
            for current in frontier[:beam_width]:
                neighbors = self.store.get_neighbors(current.entity_id, limit=max(40, beam_width * 2))
                for neighbor in neighbors:
                    relation_type = neighbor["relation_type"]
                    relation_prior = RELATION_PRIORS.get(relation_type, 0.65)
                    relation_weight = float(neighbor.get("relation_weight", 1.0))
                    rel_score = min(1.2, relation_prior * (1.0 + 0.08 * (relation_weight - 1.0)))

                    similarity = max(
                        0.0,
                        self.embedder.cosine_similarity(query_embedding, neighbor.get("target_embedding", [])),
                    )
                    score = current.score * rel_score * (0.85 + 0.15 * similarity)

                    if score < prune_threshold:
                        pruned += 1
                        continue

                    step = TracePathStep(
                        from_entity_id=neighbor["source_entity_id"],
                        from_entity_name=neighbor["source_name"],
                        to_entity_id=neighbor["target_entity_id"],
                        to_entity_name=neighbor["target_name"],
                        relation_type=relation_type,
                        evidence_turn_ids=neighbor.get("evidence_turn_ids", []),
                    )
                    path = EntityPath(
                        entity_id=neighbor["target_entity_id"],
                        canonical_name=neighbor["target_name"],
                        score=score,
                        steps=[*current.steps, step],
                    )
                    candidates.append(path)

            if not candidates:
                break

            candidates.sort(key=lambda p: p.score, reverse=True)
            next_frontier: list[EntityPath] = []

            for path in candidates:
                best = reached.get(path.entity_id)
                if best is None or path.score > best.score:
                    reached[path.entity_id] = path
                    next_frontier.append(path)
                if len(next_frontier) >= beam_width:
                    break

            frontier = next_frontier
            depth += 1

        return reached, pruned

    def _rank_turns(
        self,
        *,
        request: QueryRequest,
        turn_rows: list[dict[str, Any]],
        reached_entities: dict[str, EntityPath],
        query_embedding: list[float],
        importance_weight: float,
        recency_weight: float,
        recall_half_life_hours: int,
    ) -> list[TurnResult]:
        results: list[TurnResult] = []
        now = datetime.now(timezone.utc)

        for row in turn_rows:
            matched_entity_ids = row.get("matched_entity_ids", [])
            matched_entities = row.get("matched_entities", [])

            entity_score = 0.0
            path_steps: list[TracePathStep] = []
            evidence_turn_ids: set[str] = set()

            for entity_id in matched_entity_ids:
                path = reached_entities.get(entity_id)
                if not path:
                    continue
                entity_score += path.score
                path_steps.extend(path.steps[:4])
                for step in path.steps:
                    evidence_turn_ids.update(step.evidence_turn_ids)

            text_similarity = max(0.0, self.embedder.cosine_similarity(query_embedding, row.get("embedding", [])))
            importance_score = float(row.get("importance_score", 0.0))
            last_recalled_at = self._to_optional_datetime(row.get("last_recalled_at"))

            normalized_entity_score = min(1.0, entity_score / max(1.0, len(matched_entity_ids)))
            evidence_bonus = 0.1 if row["turn_uid"] in evidence_turn_ids else 0.0
            base_score = normalized_entity_score * 0.72 + text_similarity * 0.18 + evidence_bonus

            recency_factor = self._compute_recency_factor(
                now=now,
                anchor=last_recalled_at or self._to_datetime(row.get("timestamp")),
                half_life_hours=recall_half_life_hours,
            )
            importance_component = importance_weight * importance_score
            recency_component = recency_weight * recency_factor
            final_score = base_score + importance_component + recency_component

            timestamp = self._to_datetime(row.get("timestamp"))
            turn_result = TurnResult(
                turn_uid=row["turn_uid"],
                conversation_id=row["conversation_id"],
                turn_id=row["turn_id"],
                speaker=row["speaker"],
                timestamp=timestamp,
                text=row["text"],
                score=round(final_score, 6),
                score_breakdown={
                    "base_score": round(base_score, 6),
                    "entity_score": round(normalized_entity_score, 6),
                    "text_similarity": round(text_similarity, 6),
                    "evidence_bonus": round(evidence_bonus, 6),
                    "importance_component": round(importance_component, 6),
                    "recency_component": round(recency_component, 6),
                    "final_score": round(final_score, 6),
                },
                matched_entities=matched_entities,
                path_summary=self._dedup_steps(path_steps, limit=8),
                evidence_turn_ids=sorted(evidence_turn_ids),
                importance_score=round(importance_score, 6),
                recency_factor=round(recency_factor, 6),
                last_recalled_at=last_recalled_at,
            )
            results.append(turn_result)

        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _dedup_steps(self, steps: list[TracePathStep], limit: int) -> list[TracePathStep]:
        deduped: list[TracePathStep] = []
        keys: set[tuple[str, str, str]] = set()
        for step in steps:
            key = (step.from_entity_id, step.to_entity_id, step.relation_type)
            if key in keys:
                continue
            keys.add(key)
            deduped.append(step)
            if len(deduped) >= limit:
                break
        return deduped

    def _to_datetime(self, value: Any) -> datetime:
        optional = self._to_optional_datetime(value)
        if optional is not None:
            return optional
        return datetime.now(timezone.utc)

    def _to_optional_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if hasattr(value, "to_native"):
            native = value.to_native()
            if isinstance(native, datetime):
                return native

        if hasattr(value, "iso_format"):
            value = value.iso_format()

        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        return None

    def _compute_recency_factor(self, *, now: datetime, anchor: datetime, half_life_hours: int) -> float:
        if half_life_hours <= 0:
            return 1.0
        if now.tzinfo is None and anchor.tzinfo is not None:
            now = now.replace(tzinfo=anchor.tzinfo)
        elif now.tzinfo is not None and anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=now.tzinfo)
        age_hours = max(0.0, (now - anchor).total_seconds() / 3600.0)
        decay = math.exp(-math.log(2.0) * age_hours / float(half_life_hours))
        return max(0.0, min(1.0, decay))
