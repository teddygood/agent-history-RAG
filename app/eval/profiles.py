from __future__ import annotations

from typing import Any


def builtin_eval_profiles() -> dict[str, dict[str, Any]]:
    """
    Stable, explicit evaluation profiles.

    Notes:
    - We disable recall tracking to avoid mutating the graph during evaluation runs.
    - We disable importance/recency weights to avoid feedback loops during offline eval.
    - "embedding_only" is not a true vector-only retrieval. It ranks the candidate pool
      using embedding similarity only (weights), so candidate generation still depends
      on graph seeding and/or fulltext search.
    """

    base_no_bias: dict[str, Any] = {
        "record_recall": False,
        "importance_weight": 0.0,
        "recency_weight": 0.0,
    }

    return {
        "graph_only": {
            **base_no_bias,
            "hybrid_enabled": False,
            "graph_weight": 1.0,
            "embedding_weight": 0.0,
            "lexical_weight": 0.0,
            "rerank_enabled": False,
        },
        "lexical_only": {
            **base_no_bias,
            "hybrid_enabled": True,
            "graph_weight": 0.0,
            "embedding_weight": 0.0,
            "lexical_weight": 1.0,
            "rerank_enabled": False,
        },
        "embedding_only": {
            **base_no_bias,
            "hybrid_enabled": True,
            "graph_weight": 0.0,
            "embedding_weight": 1.0,
            "lexical_weight": 0.0,
            "rerank_enabled": False,
        },
        "hybrid": {
            **base_no_bias,
            "hybrid_enabled": True,
            "graph_weight": 0.62,
            "embedding_weight": 0.23,
            "lexical_weight": 0.15,
            "rerank_enabled": False,
        },
        "hybrid_rerank": {
            **base_no_bias,
            "hybrid_enabled": True,
            "graph_weight": 0.62,
            "embedding_weight": 0.23,
            "lexical_weight": 0.15,
            "rerank_enabled": True,
            "rerank_top_n": 20,
        },
    }

