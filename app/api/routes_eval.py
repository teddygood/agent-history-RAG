from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_runtime
from app.eval import aggregate_metrics, load_eval_examples
from app.eval.profiles import builtin_eval_profiles
from app.models.schemas import QueryRequest
from app.services.runtime import AppRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["eval"])


class EvalCompareRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1, description="Path to eval JSONL (repo-relative or absolute).")
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5], description="K values for Recall@K/nDCG@K.")
    profiles: list[str] = Field(
        default_factory=lambda: ["graph_only", "lexical_only", "embedding_only", "hybrid", "hybrid_rerank"]
    )
    baseline: str = Field(default="hybrid", description="Profile name to compute deltas against.")
    conversation_id: str | None = Field(default=None, description="Optional global conversation_id override.")
    max_hops: int | None = Field(default=None, ge=1, le=6)
    beam_width: int | None = Field(default=None, ge=4, le=200)
    prune_threshold: float | None = Field(default=None, ge=0.01, le=1.0)
    include_details: bool = Field(default=False, description="If true, include per-example ranked lists.")


def _resolve_dataset_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="dataset_path must be within the project workspace")

    if not path.exists():
        raise HTTPException(status_code=400, detail=f"dataset_path not found: {path}")

    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"dataset_path must be a file: {path}")

    return path


def _parse_ks(values: list[int]) -> list[int]:
    ks = sorted({int(value) for value in values if int(value) > 0})
    if not ks:
        raise HTTPException(status_code=400, detail="ks must contain at least one positive integer")
    return ks


def _dedup_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


@router.post("/eval/compare")
def compare_eval(
    request: EvalCompareRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    dataset_path = _resolve_dataset_path(request.dataset_path)
    ks = _parse_ks(request.ks)
    max_k = max(ks)

    builtin_profiles = builtin_eval_profiles()
    requested_profiles = _dedup_preserve_order(request.profiles)
    if not requested_profiles:
        requested_profiles = ["graph_only", "lexical_only", "embedding_only", "hybrid", "hybrid_rerank"]

    unknown = [name for name in requested_profiles if name not in builtin_profiles]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown profiles: {', '.join(unknown)}")

    baseline = request.baseline.strip()
    if baseline not in builtin_profiles:
        raise HTTPException(status_code=400, detail=f"unknown baseline: {baseline}")

    profiles_to_run = requested_profiles[:]
    if baseline not in profiles_to_run:
        profiles_to_run.append(baseline)

    global_overrides: dict[str, Any] = {
        "conversation_id": request.conversation_id,
        "max_hops": request.max_hops,
        "beam_width": request.beam_width,
        "prune_threshold": request.prune_threshold,
    }
    global_overrides = {key: value for key, value in global_overrides.items() if value is not None}

    examples = load_eval_examples(dataset_path)

    results: dict[str, Any] = {}
    for profile_name in profiles_to_run:
        overrides = dict(builtin_profiles.get(profile_name, {}))
        overrides.update(global_overrides)
        overrides.setdefault("record_recall", False)

        rankings: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        details: list[dict[str, Any]] = []

        for idx, example in enumerate(examples, start=1):
            payload: dict[str, Any] = {"query": example.query, "top_k": max_k}
            if example.conversation_id:
                payload["conversation_id"] = example.conversation_id
            payload.update(overrides)
            if example.request_overrides:
                payload.update(example.request_overrides)

            response = runtime.retriever.query(QueryRequest(**payload))
            ranked_turn_uids = [turn.turn_uid for turn in response.selected_turns]

            rankings.append(ranked_turn_uids)
            relevant = set(example.relevant_turn_uids)
            relevant_sets.append(relevant)

            if request.include_details:
                details.append(
                    {
                        "index": idx,
                        "query": example.query,
                        "relevant_turn_uids": sorted(relevant),
                        "ranked_turn_uids": ranked_turn_uids,
                        "matched_seed_entities": response.matched_seed_entities,
                        "applied_params": response.applied_params,
                    }
                )

        summary = aggregate_metrics(rankings, relevant_sets, ks)
        results[profile_name] = {
            "profile": profile_name,
            "request_overrides": overrides,
            "examples": summary.examples,
            "mrr": round(summary.mrr, 6),
            "recall_at": {f"k={k}": round(v, 6) for k, v in summary.recall_at.items()},
            "ndcg_at": {f"k={k}": round(v, 6) for k, v in summary.ndcg_at.items()},
            "hit_at": {f"k={k}": round(v, 6) for k, v in summary.hit_at.items()},
            "details": details,
        }

    baseline_result = results.get(baseline, {})
    base_mrr = float(baseline_result.get("mrr", 0.0) or 0.0)
    base_recall = baseline_result.get("recall_at", {}) if isinstance(baseline_result.get("recall_at"), dict) else {}

    deltas: dict[str, Any] = {}
    for profile_name in profiles_to_run:
        item = results.get(profile_name, {})
        recall_at = item.get("recall_at", {}) if isinstance(item.get("recall_at"), dict) else {}
        deltas[profile_name] = {
            "mrr": round(float(item.get("mrr", 0.0) or 0.0) - base_mrr, 6),
            "recall_at": {key: round(float(value) - float(base_recall.get(key, 0.0)), 6) for key, value in recall_at.items()},
        }

    return {
        "dataset": str(dataset_path),
        "api_base": None,
        "observed_runtime": runtime.runtime_profile(),
        "examples": len(examples),
        "requested": {
            "ks": ks,
            "profiles": requested_profiles,
            "baseline": baseline,
            "global_overrides": global_overrides,
            "include_details": request.include_details,
        },
        "profiles": {name: results[name] for name in profiles_to_run},
        "deltas": deltas,
    }

