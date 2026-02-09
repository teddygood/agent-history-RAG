from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "please-change-me")

    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))

    graph_max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "3"))
    graph_beam_width: int = int(os.getenv("GRAPH_BEAM_WIDTH", "24"))
    graph_prune_threshold: float = float(os.getenv("GRAPH_PRUNE_THRESHOLD", "0.10"))
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "5"))


settings = Settings()
