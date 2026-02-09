from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(values)


@dataclass(frozen=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "please-change-me")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "auto")
    embedding_model_primary: str = os.getenv(
        "EMBEDDING_MODEL_PRIMARY", os.getenv("EMBEDDING_MODEL_NAME", "nlpai-lab/KURE-v1")
    )
    embedding_model_candidates: tuple[str, ...] = _env_csv("EMBEDDING_MODEL_CANDIDATES", "dragonkue/BGE-m3-ko")
    embedding_query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    embedding_document_prefix: str = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")
    embedding_fallback_to_hash: bool = _env_bool("EMBEDDING_FALLBACK_TO_HASH", True)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))
    # Keep defaults conservative to avoid OOM on CPU-only local environments.
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    embedding_cpu_threads: int = int(os.getenv("EMBEDDING_CPU_THREADS", "0"))

    # Chunk sizes are in whitespace-delimited tokens (approx. "words").
    # Defaults are conservative to avoid OOM when using long-context embedding models on CPU.
    chunk_size_default: int = int(os.getenv("CHUNK_SIZE_DEFAULT", "256"))
    chunk_overlap_default: int = int(os.getenv("CHUNK_OVERLAP_DEFAULT", "32"))
    chunk_size_structured: int = int(os.getenv("CHUNK_SIZE_STRUCTURED", "192"))
    chunk_overlap_structured: int = int(os.getenv("CHUNK_OVERLAP_STRUCTURED", "24"))
    chunk_size_max: int = int(os.getenv("CHUNK_SIZE_MAX", "512"))
    chunk_structure_min_lines: int = int(os.getenv("CHUNK_STRUCTURE_MIN_LINES", "3"))
    chunk_structure_heading_ratio: float = float(os.getenv("CHUNK_STRUCTURE_HEADING_RATIO", "0.20"))

    graph_max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "3"))
    graph_beam_width: int = int(os.getenv("GRAPH_BEAM_WIDTH", "24"))
    graph_prune_threshold: float = float(os.getenv("GRAPH_PRUNE_THRESHOLD", "0.10"))
    hybrid_enabled: bool = _env_bool("HYBRID_ENABLED", True)
    hybrid_graph_weight: float = float(os.getenv("HYBRID_GRAPH_WEIGHT", "0.62"))
    hybrid_embedding_weight: float = float(os.getenv("HYBRID_EMBEDDING_WEIGHT", "0.23"))
    hybrid_lexical_weight: float = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.15"))
    reranker_enabled: bool = _env_bool("RERANKER_ENABLED", False)
    reranker_model_primary: str = os.getenv("RERANKER_MODEL_PRIMARY", "BAAI/bge-reranker-v2-m3")
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "20"))
    reranker_weight: float = float(os.getenv("RERANKER_WEIGHT", "0.20"))
    reranker_fallback_to_base: bool = _env_bool("RERANKER_FALLBACK_TO_BASE", True)
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "5"))

    ingest_batch_size: int = int(os.getenv("INGEST_BATCH_SIZE", "16"))
    ingest_skip_existing_history: bool = _env_bool("INGEST_SKIP_EXISTING_HISTORY", True)


settings = Settings()
