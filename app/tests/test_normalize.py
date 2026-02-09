from app.services.normalize import canonicalize, to_entity_id


def test_canonicalize_synonym() -> None:
    assert canonicalize("PagedAttention") == "paged attention"
    assert canonicalize("GraphRAG") == "graph-centric rag"


def test_entity_id_stable() -> None:
    first = to_entity_id("continuous batching")
    second = to_entity_id("continuous batching")
    assert first == second
