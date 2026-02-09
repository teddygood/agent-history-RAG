from __future__ import annotations

import pytest

import app.services.embedder as embedder_module
from app.services.embedder import HashEmbedder, _resolve_model_profile, create_embedder


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dim=32)
    a = embedder.embed("continuous batching")
    b = embedder.embed("continuous batching")
    c = embedder.embed("paged attention")

    assert a == b
    assert a != c
    assert len(a) == 32


def test_hash_embedder_query_and_document_methods() -> None:
    embedder = HashEmbedder(dim=16)
    query_vec = embedder.embed_query("연속 배칭")
    doc_vec = embedder.embed_document("연속 배칭")

    assert len(query_vec) == 16
    assert query_vec == doc_vec


def test_create_embedder_hash_provider() -> None:
    embedder = create_embedder(provider="hash", dim=16)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 16


def test_create_embedder_auto_falls_back_to_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_embedder(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("missing sentence-transformers")

    monkeypatch.setattr(embedder_module, "SentenceTransformerEmbedder", failing_embedder)
    embedder = create_embedder(provider="auto", dim=24)

    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 24


def test_create_embedder_sentence_transformers_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_embedder(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("missing sentence-transformers")

    monkeypatch.setattr(embedder_module, "SentenceTransformerEmbedder", failing_embedder)
    with pytest.raises(RuntimeError, match="Failed to initialize embedder provider"):
        create_embedder(provider="sentence-transformers", allow_fallback_to_hash=False)


def test_create_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER"):
        create_embedder(provider="unknown-provider")


def test_resolve_known_model_profile_defaults() -> None:
    profile = _resolve_model_profile(model_name="nlpai-lab/KURE-v1")
    assert profile.model_name == "nlpai-lab/KURE-v1"
    assert profile.query_prefix == "query: "
    assert profile.document_prefix == "passage: "
