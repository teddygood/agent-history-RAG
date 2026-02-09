from __future__ import annotations

import hashlib
import re


SYNONYM_MAP = {
    "pagedattention": "paged attention",
    "continuousbatching": "continuous batching",
    "kv cache": "kv-cache",
    "knowledge graph": "entity graph",
    "graph rag": "graph-centric rag",
    "graphrag": "graph-centric rag",
}


def normalize_surface(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[`'\"“”’]", "", s)
    s = re.sub(r"[^a-z0-9\-\s_/+]", " ", s)
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
    if any(token in canonical_name for token in ("algorithm", "attention", "batching", "rerank")):
        return "algorithm"
    if any(token in canonical_name for token in ("neo4j", "postgres", "fastapi", "d3", "python", "docker")):
        return "technology"
    if any(token in canonical_name for token in ("rag", "retrieval", "embedding", "graph", "traceability")):
        return "concept"
    return "concept"
