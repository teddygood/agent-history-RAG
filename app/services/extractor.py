from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import ExtractedEntity, ExtractedRelation
from app.services.normalize import canonicalize, classify_entity_type


TECH_KEYWORDS = {
    "rag",
    "retrieval",
    "generation",
    "entity",
    "graph",
    "knowledge",
    "traceability",
    "embedding",
    "vector",
    "neo4j",
    "fastapi",
    "docker",
    "algorithm",
    "attention",
    "batching",
    "cache",
    "inference",
    "latency",
    "throughput",
    "d3",
    "query",
    "ranking",
    "rerank",
}

CANONICAL_PHRASES = [
    "continuous batching",
    "paged attention",
    "kv-cache",
    "graph rag",
    "graph-centric rag",
    "entity graph",
    "vector rag",
    "top-k",
    "beam search",
    "cosine similarity",
]

RELATION_RULES = [
    ("COMPARES", ["compare", "vs", "versus", "비교", "차이"]),
    ("DEPENDS_ON", ["depend", "depends on", "requires", "need", "의존", "필요"]),
    ("IMPLEMENTS", ["implement", "build with", "using", "구현", "사용해"]),
    ("EXPLAINS", ["explain", "means", "정의", "설명", "what is"]),
]


@dataclass
class ExtractorConfig:
    min_entity_length: int = 3
    max_ngram: int = 3


class HeuristicExtractor:
    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        lowered = text.lower()
        found: dict[str, ExtractedEntity] = {}

        for phrase in CANONICAL_PHRASES:
            if phrase in lowered:
                canonical = canonicalize(phrase)
                found[canonical] = ExtractedEntity(
                    surface=phrase,
                    canonical_name=canonical,
                    entity_type=classify_entity_type(canonical),
                    confidence=0.90,
                )

        for surface in self._extract_quoted_terms(text):
            canonical = canonicalize(surface)
            if len(canonical) < self.config.min_entity_length:
                continue
            found.setdefault(
                canonical,
                ExtractedEntity(
                    surface=surface,
                    canonical_name=canonical,
                    entity_type=classify_entity_type(canonical),
                    confidence=0.82,
                ),
            )

        for surface in self._extract_ngrams(lowered):
            canonical = canonicalize(surface)
            if len(canonical) < self.config.min_entity_length:
                continue
            found.setdefault(
                canonical,
                ExtractedEntity(
                    surface=surface,
                    canonical_name=canonical,
                    entity_type=classify_entity_type(canonical),
                    confidence=0.72,
                ),
            )

        for acronym in re.findall(r"\b[A-Z]{2,}\b", text):
            canonical = canonicalize(acronym)
            if len(canonical) < self.config.min_entity_length:
                continue
            found.setdefault(
                canonical,
                ExtractedEntity(
                    surface=acronym,
                    canonical_name=canonical,
                    entity_type=classify_entity_type(canonical),
                    confidence=0.75,
                ),
            )

        return sorted(found.values(), key=lambda item: (-item.confidence, item.canonical_name))

    def extract_relations(self, text: str, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        if len(entities) < 2:
            return []

        relation_type = self._infer_relation_type(text)
        positions = self._entity_positions(text.lower(), entities)
        ordered = sorted(entities, key=lambda e: positions.get(e.canonical_name, 10**6))

        relations: list[ExtractedRelation] = []
        for i in range(len(ordered) - 1):
            source = ordered[i].canonical_name
            target = ordered[i + 1].canonical_name
            if source == target:
                continue
            relations.append(
                ExtractedRelation(
                    source_canonical=source,
                    target_canonical=target,
                    relation_type=relation_type,
                    confidence=0.78 if relation_type != "RELATED_TO" else 0.62,
                )
            )
        return relations

    def _infer_relation_type(self, text: str) -> str:
        lowered = text.lower()
        for relation, markers in RELATION_RULES:
            if any(marker in lowered for marker in markers):
                return relation
        return "RELATED_TO"

    def _extract_quoted_terms(self, text: str) -> list[str]:
        matches = []
        matches.extend(re.findall(r"`([^`]{2,80})`", text))
        matches.extend(re.findall(r"\"([^\"]{2,80})\"", text))
        matches.extend(re.findall(r"'([^']{2,80})'", text))
        return matches

    def _extract_ngrams(self, lowered: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9\-+/]*", lowered)
        out: list[str] = []
        for n in range(1, self.config.max_ngram + 1):
            for i in range(0, len(tokens) - n + 1):
                gram_tokens = tokens[i : i + n]
                if not any(token in TECH_KEYWORDS for token in gram_tokens):
                    continue
                phrase = " ".join(gram_tokens)
                if len(phrase) >= self.config.min_entity_length:
                    out.append(phrase)
        return out

    def _entity_positions(self, lowered: str, entities: list[ExtractedEntity]) -> dict[str, int]:
        pos: dict[str, int] = {}
        for entity in entities:
            idx = lowered.find(entity.canonical_name)
            if idx >= 0:
                pos[entity.canonical_name] = idx
        return pos
