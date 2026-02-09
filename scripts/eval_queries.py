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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Graph-RAG query quality against query-turn relevance dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset with query + relevant_turn_uids")
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"), help="Base URL")
    parser.add_argument("--k", default="1,3,5", help="Comma-separated K values, e.g. 1,3,5,10")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="HTTP timeout")
    parser.add_argument("--conversation-id", default=None, help="Optional global conversation_id override")
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--prune-threshold", type=float, default=None)
    parser.add_argument("--importance-weight", type=float, default=None)
    parser.add_argument("--recency-weight", type=float, default=None)
    parser.add_argument("--recall-half-life-hours", type=int, default=None)
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    ks = _parse_ks(args.k)
    max_k = max(ks)
    examples = load_eval_examples(args.dataset)

    global_request_overrides = _compact_dict(
        {
            "conversation_id": args.conversation_id,
            "max_hops": args.max_hops,
            "beam_width": args.beam_width,
            "prune_threshold": args.prune_threshold,
            "importance_weight": args.importance_weight,
            "recency_weight": args.recency_weight,
            "recall_half_life_hours": args.recall_half_life_hours,
        }
    )

    rankings: list[list[str]] = []
    relevant_sets: list[set[str]] = []
    details: list[dict[str, Any]] = []

    for idx, example in enumerate(examples, start=1):
        payload = {"query": example.query, "top_k": max_k}
        payload.update(global_request_overrides)
        if example.conversation_id:
            payload["conversation_id"] = example.conversation_id
        if example.request_overrides:
            payload.update(example.request_overrides)

        response = _post_json(f"{args.api_base.rstrip('/')}/query", payload, timeout_seconds=args.timeout_seconds)
        selected_turns = response.get("selected_turns", [])
        ranked_turn_uids = [str(item.get("turn_uid", "")).strip() for item in selected_turns if item.get("turn_uid")]

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
            }
        )

    summary = aggregate_metrics(rankings, relevant_sets, ks)

    report = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "api_base": args.api_base,
        "examples": summary.examples,
        "mrr": round(summary.mrr, 6),
        "recall_at": {f"k={k}": round(v, 6) for k, v in summary.recall_at.items()},
        "ndcg_at": {f"k={k}": round(v, 6) for k, v in summary.ndcg_at.items()},
        "hit_at": {f"k={k}": round(v, 6) for k, v in summary.hit_at.items()},
        "details": details,
    }

    _print_report(report)
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


def _print_report(report: dict[str, Any]) -> None:
    print("== Graph-RAG Evaluation ==")
    print(f"dataset   : {report['dataset']}")
    print(f"api_base  : {report['api_base']}")
    print(f"examples  : {report['examples']}")
    print(f"mrr       : {report['mrr']:.6f}")
    print("recall@k  :", _metric_dict_to_text(report["recall_at"]))
    print("ndcg@k    :", _metric_dict_to_text(report["ndcg_at"]))
    print("hit@k     :", _metric_dict_to_text(report["hit_at"]))


def _metric_dict_to_text(data: dict[str, Any]) -> str:
    parts = [f"{key}={float(value):.6f}" for key, value in data.items()]
    return ", ".join(parts)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
