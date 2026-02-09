from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    query: str
    relevant_turn_uids: frozenset[str]
    conversation_id: str | None = None
    request_overrides: dict[str, Any] | None = None


def load_eval_examples(path: str | Path) -> list[EvalExample]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {source}")

    examples: list[EvalExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc.msg}") from exc

            query = str(item.get("query", "")).strip()
            if not query:
                raise ValueError(f"Missing query at {source}:{line_number}")

            relevant = _to_turn_uid_set(item.get("relevant_turn_uids"), item.get("expected_turn_uids"))
            if not relevant:
                raise ValueError(f"Missing relevant_turn_uids at {source}:{line_number}")

            conversation_id = item.get("conversation_id")
            if conversation_id is not None:
                conversation_id = str(conversation_id).strip() or None

            request_overrides = item.get("request")
            if request_overrides is not None and not isinstance(request_overrides, dict):
                raise ValueError(f"request must be an object at {source}:{line_number}")

            examples.append(
                EvalExample(
                    query=query,
                    relevant_turn_uids=frozenset(relevant),
                    conversation_id=conversation_id,
                    request_overrides=request_overrides,
                )
            )

    if not examples:
        raise ValueError(f"No usable examples in dataset: {source}")
    return examples


def _to_turn_uid_set(primary: Any, fallback: Any) -> set[str]:
    values = primary if primary is not None else fallback
    if values is None:
        return set()
    if not isinstance(values, list):
        return set()

    out: set[str] = set()
    for raw in values:
        turn_uid = str(raw).strip()
        if turn_uid:
            out.add(turn_uid)
    return out
