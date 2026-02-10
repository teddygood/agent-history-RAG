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
from app.services.reranker import Reranker, create_reranker


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
DEFAULT_GRAPH_WEIGHT = 0.62
DEFAULT_EMBEDDING_WEIGHT = 0.23
DEFAULT_LEXICAL_WEIGHT = 0.15


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
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or HeuristicExtractor()
        self.embedder = embedder or HashEmbedder()
        self.reranker = reranker or create_reranker(provider="none")

    def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = self.embedder.embed_query(request.query)
        seed_candidates = self._find_seed_entities(request.query)

        max_hops = request.max_hops or settings.graph_max_hops
        beam_width = request.beam_width or settings.graph_beam_width
        prune_threshold = request.prune_threshold if request.prune_threshold is not None else settings.graph_prune_threshold
        hybrid_enabled = request.hybrid_enabled if request.hybrid_enabled is not None else settings.hybrid_enabled
        graph_weight = request.graph_weight if request.graph_weight is not None else settings.hybrid_graph_weight
        embedding_weight = (
            request.embedding_weight if request.embedding_weight is not None else settings.hybrid_embedding_weight
        )
        lexical_weight = request.lexical_weight if request.lexical_weight is not None else settings.hybrid_lexical_weight
        graph_weight, embedding_weight, lexical_weight = self._normalize_weights(
            graph_weight=graph_weight,
            embedding_weight=embedding_weight,
            lexical_weight=lexical_weight,
        )
        importance_weight = (
            request.importance_weight if request.importance_weight is not None else DEFAULT_IMPORTANCE_WEIGHT
        )
        recency_weight = request.recency_weight if request.recency_weight is not None else DEFAULT_RECENCY_WEIGHT
        recall_half_life_hours = request.recall_half_life_hours or DEFAULT_RECALL_HALF_LIFE_HOURS
        rerank_enabled = request.rerank_enabled if request.rerank_enabled is not None else settings.reranker_enabled
        rerank_top_n = request.rerank_top_n or settings.reranker_top_n
        rerank_weight = max(0.0, min(1.0, settings.reranker_weight))
        rerank_model = (
            request.rerank_model.strip()
            if request.rerank_model
            else str(getattr(self.reranker, "model_name", settings.reranker_model_primary))
        )
        active_reranker = self._resolve_reranker(request.rerank_model)
        reranker_status = active_reranker.status()

        applied_params: dict[str, float | int | bool | str] = {
            "max_hops": max_hops,
            "beam_width": beam_width,
            "prune_threshold": round(prune_threshold, 6),
            "hybrid_enabled": hybrid_enabled,
            "graph_weight": round(graph_weight, 6),
            "embedding_weight": round(embedding_weight, 6),
            "lexical_weight": round(lexical_weight, 6),
            "importance_weight": round(importance_weight, 6),
            "recency_weight": round(recency_weight, 6),
            "recall_half_life_hours": recall_half_life_hours,
            "rerank_enabled": rerank_enabled,
            "rerank_top_n": rerank_top_n,
            "rerank_weight": round(rerank_weight, 6),
            "rerank_model": rerank_model,
            "reranker_available": bool(reranker_status.get("available", False)),
            "top_k": request.top_k,
        }

        if not seed_candidates and not hybrid_enabled:
            return QueryResponse(
                query=request.query,
                top_k=request.top_k,
                matched_seed_entities=[],
                selected_turns=[],
                pruned_paths=0,
                applied_params=applied_params,
            )

        reached: dict[str, EntityPath] = {}
        pruned_paths = 0
        if seed_candidates:
            reached, pruned_paths = self._traverse_graph(
                seed_candidates=seed_candidates,
                query_embedding=query_embedding,
                max_hops=max_hops,
                beam_width=beam_width,
                prune_threshold=prune_threshold,
            )

        graph_turn_rows = self.store.fetch_turns_for_entities(
            reached.keys(),
            conversation_id=request.conversation_id,
            limit=max(400, request.top_k * 40),
        ) if reached else []
        lexical_turn_rows = (
            self.store.search_turns_fulltext(
                query_text=request.query,
                conversation_id=request.conversation_id,
                limit=max(200, request.top_k * 40),
            )
            if hybrid_enabled
            else []
        )

        turn_results = self._rank_turns(
            request=request,
            graph_turn_rows=graph_turn_rows,
            lexical_turn_rows=lexical_turn_rows,
            reached_entities=reached,
            query_embedding=query_embedding,
            graph_weight=graph_weight,
            embedding_weight=embedding_weight,
            lexical_weight=lexical_weight,
            importance_weight=importance_weight,
            recency_weight=recency_weight,
            recall_half_life_hours=recall_half_life_hours,
        )
        if rerank_enabled and turn_results:
            turn_results = self._apply_rerank(
                query_text=request.query,
                turn_results=turn_results,
                reranker=active_reranker,
                top_n=rerank_top_n,
                rerank_weight=rerank_weight,
            )
        selected_turns = turn_results[: request.top_k]
        if selected_turns and request.record_recall:
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
        query_embedding = self.embedder.embed_query(query_text)

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
        graph_turn_rows: list[dict[str, Any]],
        lexical_turn_rows: list[dict[str, Any]],
        reached_entities: dict[str, EntityPath],
        query_embedding: list[float],
        graph_weight: float,
        embedding_weight: float,
        lexical_weight: float,
        importance_weight: float,
        recency_weight: float,
        recall_half_life_hours: int,
    ) -> list[TurnResult]:
        results: list[TurnResult] = []
        now = datetime.now(timezone.utc)

        lexical_by_uid = {row["turn_uid"]: float(row.get("lexical_score", 0.0)) for row in lexical_turn_rows}
        lexical_normalized = self._normalize_score_map(lexical_by_uid)

        merged_rows = self._merge_turn_rows(graph_turn_rows=graph_turn_rows, lexical_turn_rows=lexical_turn_rows)

        for row in merged_rows.values():
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
            graph_signal = max(0.0, min(1.0, normalized_entity_score + evidence_bonus))
            lexical_signal = lexical_normalized.get(row["turn_uid"], 0.0)
            lexical_raw_score = lexical_by_uid.get(row["turn_uid"], 0.0)
            fusion_score = (
                graph_weight * graph_signal
                + embedding_weight * text_similarity
                + lexical_weight * lexical_signal
            )

            recency_factor = self._compute_recency_factor(
                now=now,
                anchor=last_recalled_at or self._to_datetime(row.get("timestamp")),
                half_life_hours=recall_half_life_hours,
            )
            importance_component = importance_weight * importance_score
            recency_component = recency_weight * recency_factor
            final_score = fusion_score + importance_component + recency_component

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
                    "fusion_score": round(fusion_score, 6),
                    "graph_signal": round(graph_signal, 6),
                    "embedding_signal": round(text_similarity, 6),
                    "lexical_signal": round(lexical_signal, 6),
                    "lexical_raw_score": round(lexical_raw_score, 6),
                    "entity_score": round(normalized_entity_score, 6),
                    "evidence_bonus": round(evidence_bonus, 6),
                    "importance_component": round(importance_component, 6),
                    "recency_component": round(recency_component, 6),
                    "rerank_component": 0.0,
                    "final_score": round(final_score, 6),
                },
                matched_entities=matched_entities,
                path_summary=self._dedup_steps(path_steps, limit=8),
                evidence_turn_ids=sorted(evidence_turn_ids),
                importance_score=round(importance_score, 6),
                recency_factor=round(recency_factor, 6),
                chunk_profile=row.get("chunk_profile"),
                chunk_count=int(row.get("chunk_count", 1) or 1),
                last_recalled_at=last_recalled_at,
            )
            results.append(turn_result)

        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _merge_turn_rows(
        self,
        *,
        graph_turn_rows: list[dict[str, Any]],
        lexical_turn_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for row in graph_turn_rows:
            copied = dict(row)
            copied["matched_entity_ids"] = list(dict.fromkeys(copied.get("matched_entity_ids", [])))
            copied["matched_entities"] = list(dict.fromkeys(copied.get("matched_entities", [])))
            merged[copied["turn_uid"]] = copied

        for row in lexical_turn_rows:
            turn_uid = row["turn_uid"]
            if turn_uid not in merged:
                copied = dict(row)
                copied["matched_entity_ids"] = list(dict.fromkeys(copied.get("matched_entity_ids", [])))
                copied["matched_entities"] = list(dict.fromkeys(copied.get("matched_entities", [])))
                merged[turn_uid] = copied
                continue

            existing = merged[turn_uid]
            merged_entity_ids = list(
                dict.fromkeys([*existing.get("matched_entity_ids", []), *row.get("matched_entity_ids", [])])
            )
            merged_entities = list(
                dict.fromkeys([*existing.get("matched_entities", []), *row.get("matched_entities", [])])
            )
            existing["matched_entity_ids"] = merged_entity_ids
            existing["matched_entities"] = merged_entities
            if not existing.get("embedding") and row.get("embedding"):
                existing["embedding"] = row["embedding"]
        return merged

    def _normalize_score_map(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        values = list(scores.values())
        minimum = min(values)
        maximum = max(values)

        if maximum - minimum <= 1e-9:
            return {key: 1.0 if value > 0.0 else 0.0 for key, value in scores.items()}

        return {
            key: max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
            for key, value in scores.items()
        }

    def _normalize_weights(
        self,
        *,
        graph_weight: float,
        embedding_weight: float,
        lexical_weight: float,
    ) -> tuple[float, float, float]:
        values = [max(0.0, graph_weight), max(0.0, embedding_weight), max(0.0, lexical_weight)]
        total = sum(values)
        if total <= 1e-9:
            return DEFAULT_GRAPH_WEIGHT, DEFAULT_EMBEDDING_WEIGHT, DEFAULT_LEXICAL_WEIGHT
        return (values[0] / total, values[1] / total, values[2] / total)

    def _apply_rerank(
        self,
        *,
        query_text: str,
        turn_results: list[TurnResult],
        reranker: Reranker,
        top_n: int,
        rerank_weight: float,
    ) -> list[TurnResult]:
        if not turn_results:
            return turn_results

        active_top_n = max(1, min(top_n, len(turn_results)))
        head = turn_results[:active_top_n]
        tail = turn_results[active_top_n:]

        raw_scores = reranker.score(
            query=query_text,
            documents=[item.text for item in head],
        )
        rerank_by_uid = {
            item.turn_uid: float(raw_scores[index]) if index < len(raw_scores) else 0.0
            for index, item in enumerate(head)
        }
        normalized = self._normalize_score_map(rerank_by_uid)

        for item in head:
            rerank_component = rerank_weight * normalized.get(item.turn_uid, 0.0)
            item.score = round(item.score + rerank_component, 6)
            item.score_breakdown["rerank_raw_score"] = round(rerank_by_uid.get(item.turn_uid, 0.0), 6)
            item.score_breakdown["rerank_component"] = round(rerank_component, 6)
            item.score_breakdown["final_score"] = round(item.score, 6)

        head.sort(key=lambda result: result.score, reverse=True)
        return [*head, *tail]

    def _resolve_reranker(self, requested_model: str | None) -> Reranker:
        if not requested_model:
            return self.reranker

        model_name = requested_model.strip()
        if not model_name:
            return self.reranker

        active_name = str(getattr(self.reranker, "model_name", "")).strip()
        if active_name.lower() == model_name.lower():
            return self.reranker

        return create_reranker(
            provider="sentence-transformers",
            model_name=model_name,
            allow_fallback_to_noop=settings.reranker_fallback_to_base,
        )

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
