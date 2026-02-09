from __future__ import annotations

import hashlib
import re


SYNONYM_MAP = {
    "pagedattention": "paged attention",
    "continuousbatching": "continuous batching",
    "연속 배칭": "continuous batching",
    "연속배칭": "continuous batching",
    "페이지드 어텐션": "paged attention",
    "페이지드어텐션": "paged attention",
    "kv cache": "kv-cache",
    "kv 캐시": "kv-cache",
    "케이브이 캐시": "kv-cache",
    "knowledge graph": "entity graph",
    "graph rag": "graph-centric rag",
    "그래프 rag": "graph-centric rag",
    "그래프 기반 rag": "graph-centric rag",
    "graphrag": "graph-centric rag",
    "벡터 rag": "vector rag",
}


def normalize_surface(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[`'\"“”’]", "", s)
    s = re.sub(r"[^a-z0-9가-힣\-\s_/+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(text: str) -> str:
    s = normalize_surface(text)
    s_no_space = s.replace(" ", "")
    if s in SYNONYM_MAP:
        return SYNONYM_MAP[s]
    if s_no_space in SYNONYM_MAP:
        return SYNONYM_MAP[s_no_space]
    return s


def to_entity_id(canonical_name: str) -> str:
    digest = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()[:16]
    return f"ent:{digest}"


def classify_entity_type(canonical_name: str) -> str:
    if any(token in canonical_name for token in ("algorithm", "attention", "batching", "rerank", "어텐션", "배칭")):
        return "algorithm"
    if any(token in canonical_name for token in ("neo4j", "postgres", "fastapi", "d3", "python", "docker", "도커")):
        return "technology"
    if any(
        token in canonical_name
        for token in ("rag", "retrieval", "embedding", "graph", "traceability", "검색", "임베딩", "그래프")
    ):
        return "concept"
    return "concept"
