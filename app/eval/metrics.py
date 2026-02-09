from __future__ import annotations

from dataclasses import dataclass
import math


def recall_at_k(ranked_turn_uids: list[str], relevant_turn_uids: set[str], k: int) -> float:
    if k <= 0 or not relevant_turn_uids:
        return 0.0
    hits = len(set(ranked_turn_uids[:k]).intersection(relevant_turn_uids))
    return hits / len(relevant_turn_uids)


def reciprocal_rank(ranked_turn_uids: list[str], relevant_turn_uids: set[str]) -> float:
    if not relevant_turn_uids:
        return 0.0
    for index, turn_uid in enumerate(ranked_turn_uids, start=1):
        if turn_uid in relevant_turn_uids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_turn_uids: list[str], relevant_turn_uids: set[str], k: int) -> float:
    if k <= 0 or not relevant_turn_uids:
        return 0.0

    dcg = 0.0
    for index, turn_uid in enumerate(ranked_turn_uids[:k], start=1):
        gain = 1.0 if turn_uid in relevant_turn_uids else 0.0
        if gain == 0.0:
            continue
        dcg += gain / math.log2(index + 1.0)

    ideal_hits = min(k, len(relevant_turn_uids))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1.0) for index in range(1, ideal_hits + 1))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True)
class AggregateMetrics:
    examples: int
    mrr: float
    recall_at: dict[int, float]
    ndcg_at: dict[int, float]
    hit_at: dict[int, float]


def aggregate_metrics(
    per_example_rankings: list[list[str]],
    per_example_relevant: list[set[str]],
    ks: list[int],
) -> AggregateMetrics:
    if len(per_example_rankings) != len(per_example_relevant):
        raise ValueError("rankings and relevant lists must have the same length")
    if not per_example_rankings:
        return AggregateMetrics(examples=0, mrr=0.0, recall_at={}, ndcg_at={}, hit_at={})

    normalized_ks = sorted({k for k in ks if k > 0})
    if not normalized_ks:
        raise ValueError("At least one positive K is required")

    count = len(per_example_rankings)
    recall_sums = {k: 0.0 for k in normalized_ks}
    ndcg_sums = {k: 0.0 for k in normalized_ks}
    hit_sums = {k: 0.0 for k in normalized_ks}
    mrr_sum = 0.0

    for ranked, relevant in zip(per_example_rankings, per_example_relevant):
        mrr_sum += reciprocal_rank(ranked, relevant)
        for k in normalized_ks:
            recall = recall_at_k(ranked, relevant, k)
            recall_sums[k] += recall
            ndcg_sums[k] += ndcg_at_k(ranked, relevant, k)
            hit_sums[k] += 1.0 if recall > 0.0 else 0.0

    return AggregateMetrics(
        examples=count,
        mrr=mrr_sum / count,
        recall_at={k: recall_sums[k] / count for k in normalized_ks},
        ndcg_at={k: ndcg_sums[k] / count for k in normalized_ks},
        hit_at={k: hit_sums[k] / count for k in normalized_ks},
    )
