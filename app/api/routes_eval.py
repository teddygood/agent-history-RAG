from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_runtime
from app.eval import aggregate_metrics, load_eval_examples
from app.eval.dataset import EvalExample
from app.eval.profiles import builtin_eval_profiles
from app.models.schemas import QueryRequest
from app.services.runtime import AppRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["eval"])


class EvalCompareRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Optional path to eval JSONL (repo-relative or absolute). If omitted, uses saved eval examples.",
    )
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


class EvalExampleIn(BaseModel):
    query: str = Field(..., min_length=1)
    relevant_turn_uids: list[str] = Field(..., min_length=1)
    conversation_id: str | None = None


class EvalExampleOut(BaseModel):
    example_id: str
    query: str
    relevant_turn_uids: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def _to_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native
    if hasattr(value, "iso_format"):
        value = value.iso_format()
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    return None


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


@router.get("/eval/examples/count")
def count_eval_examples(runtime: AppRuntime = Depends(get_runtime)) -> dict[str, int]:
    return {"count": runtime.store.count_eval_examples()}


@router.get("/eval/examples", response_model=list[EvalExampleOut])
def list_eval_examples(
    limit: int = 200,
    runtime: AppRuntime = Depends(get_runtime),
) -> list[EvalExampleOut]:
    rows = runtime.store.list_eval_examples(limit=limit)
    out: list[EvalExampleOut] = []
    for row in rows:
        out.append(
            EvalExampleOut(
                example_id=str(row.get("example_id", "")),
                query=str(row.get("query", "")),
                relevant_turn_uids=[str(uid) for uid in (row.get("relevant_turn_uids") or [])],
                conversation_id=str(row.get("conversation_id") or "").strip() or None,
                created_at=_to_optional_datetime(row.get("created_at")),
                updated_at=_to_optional_datetime(row.get("updated_at")),
            )
        )
    return out


@router.post("/eval/examples", response_model=EvalExampleOut)
def create_eval_example(
    request: EvalExampleIn,
    runtime: AppRuntime = Depends(get_runtime),
) -> EvalExampleOut:
    example_id = uuid.uuid4().hex
    row = runtime.store.upsert_eval_example(
        example_id=example_id,
        query=request.query,
        relevant_turn_uids=list(request.relevant_turn_uids),
        conversation_id=request.conversation_id,
    )
    return EvalExampleOut(
        example_id=str(row.get("example_id", example_id)),
        query=str(row.get("query", request.query)),
        relevant_turn_uids=[str(uid) for uid in (row.get("relevant_turn_uids") or list(request.relevant_turn_uids))],
        conversation_id=str(row.get("conversation_id") or "").strip() or None,
        created_at=_to_optional_datetime(row.get("created_at")),
        updated_at=_to_optional_datetime(row.get("updated_at")) or datetime.now(timezone.utc),
    )


@router.delete("/eval/examples/{example_id}")
def delete_eval_example(example_id: str, runtime: AppRuntime = Depends(get_runtime)) -> dict[str, object]:
    ok = runtime.store.delete_eval_example(example_id=example_id)
    return {"deleted": ok, "example_id": example_id}


@router.post("/eval/compare")
def compare_eval(
    request: EvalCompareRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    examples: list[EvalExample]
    dataset_source = "saved"
    dataset_path: Path | None = None
    if request.dataset_path and str(request.dataset_path).strip():
        dataset_path = _resolve_dataset_path(request.dataset_path)
        dataset_source = "file"
        examples = load_eval_examples(dataset_path)
    else:
        rows = runtime.store.list_eval_examples(limit=10000)
        examples = [
            EvalExample(
                query=str(row.get("query", "")).strip(),
                relevant_turn_uids=frozenset({str(uid).strip() for uid in (row.get("relevant_turn_uids") or []) if str(uid).strip()}),
                conversation_id=str(row.get("conversation_id") or "").strip() or None,
                request_overrides=None,
            )
            for row in rows
            if str(row.get("query", "")).strip()
        ]
        if not examples:
            raise HTTPException(
                status_code=400,
                detail="no saved eval examples found; create examples via POST /eval/examples or provide dataset_path",
            )

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
        "dataset": {
            "source": dataset_source,
            "path": str(dataset_path) if dataset_path is not None else None,
        },
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
