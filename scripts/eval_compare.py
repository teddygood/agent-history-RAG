#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval import aggregate_metrics, load_eval_examples
from app.eval.profiles import builtin_eval_profiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple Graph-RAG query configurations on the same query-turn relevance dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset with query + relevant_turn_uids")
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"), help="Base URL")
    parser.add_argument("--k", default="1,3,5", help="Comma-separated K values, e.g. 1,3,5,10")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="HTTP timeout")
    parser.add_argument("--conversation-id", default=None, help="Optional global conversation_id override")
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--prune-threshold", type=float, default=None)
    parser.add_argument("--recall-half-life-hours", type=int, default=None)
    parser.add_argument(
        "--profiles",
        default="graph_only,lexical_only,embedding_only,hybrid,hybrid_rerank",
        help="Comma-separated profile names to run",
    )
    parser.add_argument("--baseline", default="hybrid", help="Profile name for delta comparison")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    ks = _parse_ks(args.k)
    max_k = max(ks)
    examples = load_eval_examples(args.dataset)

    builtin_profiles = builtin_eval_profiles()
    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    if not profile_names:
        raise ValueError("--profiles must include at least one profile name")

    unknown = [name for name in profile_names if name not in builtin_profiles]
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}")

    baseline_name = args.baseline.strip()
    if baseline_name not in builtin_profiles:
        raise ValueError(f"Unknown --baseline profile: {baseline_name}")

    global_overrides = _compact_dict(
        {
            "conversation_id": args.conversation_id,
            "max_hops": args.max_hops,
            "beam_width": args.beam_width,
            "prune_threshold": args.prune_threshold,
            "recall_half_life_hours": args.recall_half_life_hours,
        }
    )

    api_base = args.api_base.rstrip("/")
    runtime_health = _try_get_json(f"{api_base}/health", timeout_seconds=args.timeout_seconds)

    results: dict[str, dict[str, Any]] = {}

    for name in profile_names:
        profile_overrides = builtin_profiles[name]

        rankings: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        details: list[dict[str, Any]] = []

        for idx, example in enumerate(examples, start=1):
            payload: dict[str, Any] = {"query": example.query, "top_k": max_k}
            payload.update(global_overrides)
            if example.conversation_id:
                payload["conversation_id"] = example.conversation_id
            if example.request_overrides:
                payload.update(example.request_overrides)
            payload.update(profile_overrides)

            response = _post_json(f"{api_base}/query", payload, timeout_seconds=args.timeout_seconds)
            selected_turns = response.get("selected_turns", [])
            ranked_turn_uids = [
                str(item.get("turn_uid", "")).strip() for item in selected_turns if item.get("turn_uid")
            ]

            rankings.append(ranked_turn_uids)
            relevant = set(example.relevant_turn_uids)
            relevant_sets.append(relevant)
            details.append(
                {
                    "index": idx,
                    "query": example.query,
                    "relevant_turn_uids": sorted(relevant),
                    "ranked_turn_uids": ranked_turn_uids,
                    "applied_params": response.get("applied_params", {}),
                    "matched_seed_entities": response.get("matched_seed_entities", []),
                    "profile": name,
                    "request": payload,
                }
            )

        summary = aggregate_metrics(rankings, relevant_sets, ks)
        results[name] = {
            "profile": name,
            "request_overrides": profile_overrides,
            "examples": summary.examples,
            "mrr": round(summary.mrr, 6),
            "recall_at": {f"k={k}": round(v, 6) for k, v in summary.recall_at.items()},
            "ndcg_at": {f"k={k}": round(v, 6) for k, v in summary.ndcg_at.items()},
            "hit_at": {f"k={k}": round(v, 6) for k, v in summary.hit_at.items()},
            "details": details,
        }

    report = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "api_base": args.api_base,
        "observed_runtime": runtime_health,
        "requested": {
            "profiles": profile_names,
            "baseline": baseline_name,
            "ks": ks,
            "global_overrides": global_overrides,
        },
        "profiles": results,
    }

    _print_comparison(report)
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {out_path}")

    return 0


def _parse_ks(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    normalized = sorted({k for k in values if k > 0})
    if not normalized:
        raise ValueError("At least one positive K value is required")
    return normalized


def _compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = url_request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=timeout_seconds) as response:
            content = response.read().decode("utf-8")
    except url_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_text}") from exc
    except url_error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid response payload from {url}")
    return parsed


def _metric_dict_to_text(data: dict[str, Any]) -> str:
    parts = [f"{key}={float(value):.6f}" for key, value in data.items()]
    return ", ".join(parts)


def _try_get_json(url: str, timeout_seconds: float) -> dict[str, Any] | None:
    req = url_request.Request(url=url, method="GET")
    try:
        with url_request.urlopen(req, timeout=timeout_seconds) as response:
            content = response.read().decode("utf-8")
    except Exception:
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _score_lookup(report: dict[str, Any], profile: str, metric: str, key: str | None = None) -> float:
    profiles = report.get("profiles", {})
    payload = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
    value: Any = payload.get(metric)
    if key is not None and isinstance(value, dict):
        value = value.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _print_comparison(report: dict[str, Any]) -> None:
    requested = report.get("requested", {}) if isinstance(report.get("requested"), dict) else {}
    profiles = requested.get("profiles", [])
    baseline = str(requested.get("baseline", "hybrid"))
    ks = requested.get("ks", [])

    print("== Graph-RAG Evaluation Compare ==")
    print(f"dataset   : {report.get('dataset')}")
    print(f"api_base  : {report.get('api_base')}")
    observed = report.get("observed_runtime")
    if isinstance(observed, dict):
        embedding = observed.get("embedding", {})
        reranker = observed.get("reranker", {})
        if isinstance(embedding, dict):
            print(f"observed  : provider={embedding.get('provider','-')}, model={embedding.get('model_name','-')}")
        if isinstance(reranker, dict):
            print(f"reranker  : provider={reranker.get('provider','-')}, model={reranker.get('model_name','-')}, available={reranker.get('available', False)}")
    print(f"profiles  : {', '.join([str(p) for p in profiles])}")
    print(f"baseline  : {baseline}")
    print(f"ks        : {', '.join([str(k) for k in ks])}")

    header = ["profile", "mrr", "d_mrr"]
    for k in ks:
        header.extend([f"recall@{k}", f"d_r@{k}"])
    header.append("ndcg@{}".format(max(ks) if ks else 5))

    rows = []
    for name in profiles:
        mrr = _score_lookup(report, name, "mrr")
        base_mrr = _score_lookup(report, baseline, "mrr")
        row = [name, f"{mrr:.6f}", f"{(mrr-base_mrr):+.6f}"]
        for k in ks:
            key = f"k={k}"
            r = _score_lookup(report, name, "recall_at", key=key)
            base_r = _score_lookup(report, baseline, "recall_at", key=key)
            row.extend([f"{r:.6f}", f"{(r-base_r):+.6f}"])
        ndcg_key = f"k={max(ks) if ks else 5}"
        ndcg = _score_lookup(report, name, "ndcg_at", key=ndcg_key)
        row.append(f"{ndcg:.6f}")
        rows.append(row)

    _print_table(header, rows)


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [len(cell) for cell in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print("")
    print(fmt(header))
    print(" | ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Evaluation compare failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
