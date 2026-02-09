from __future__ import annotations

import json

import pytest

from app.eval.dataset import load_eval_examples
from app.eval.metrics import aggregate_metrics, ndcg_at_k


def test_load_eval_examples_supports_expected_turn_uids_alias(tmp_path) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    rows = [
        {"query": "continuous batching 설명", "expected_turn_uids": ["conv-1:t2"]},
        {"query": "paged attention 관계", "relevant_turn_uids": ["conv-1:t4"], "conversation_id": "conv-1"},
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    examples = load_eval_examples(dataset_path)
    assert len(examples) == 2
    assert examples[0].query == "continuous batching 설명"
    assert examples[0].relevant_turn_uids == frozenset({"conv-1:t2"})
    assert examples[1].conversation_id == "conv-1"


def test_load_eval_examples_requires_relevance_field(tmp_path) -> None:
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(json.dumps({"query": "missing relevance"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing relevant_turn_uids"):
        load_eval_examples(dataset_path)


def test_aggregate_metrics_smoke() -> None:
    rankings = [
        ["conv-1:t3", "conv-1:t4", "conv-1:t2"],
        ["conv-2:t1", "conv-2:t2", "conv-2:t3"],
    ]
    relevant = [
        {"conv-1:t4"},
        {"conv-2:t9"},
    ]

    summary = aggregate_metrics(rankings, relevant, ks=[1, 2, 5])
    assert summary.examples == 2
    assert summary.mrr == pytest.approx(0.25, rel=1e-6)
    assert summary.recall_at[1] == pytest.approx(0.0, rel=1e-6)
    assert summary.recall_at[2] == pytest.approx(0.5, rel=1e-6)
    assert summary.recall_at[5] == pytest.approx(0.5, rel=1e-6)
    assert summary.hit_at[2] == pytest.approx(0.5, rel=1e-6)
    assert summary.ndcg_at[2] == pytest.approx(0.3154648768, rel=1e-6)


def test_ndcg_with_multiple_relevant_docs() -> None:
    score = ndcg_at_k(
        ranked_turn_uids=["t3", "t1", "t2", "t9"],
        relevant_turn_uids={"t1", "t2"},
        k=3,
    )
    assert score == pytest.approx(0.6934264036, rel=1e-6)
