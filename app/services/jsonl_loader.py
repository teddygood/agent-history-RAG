from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import TurnInput


class JSONLLoader:
    REQUIRED_FIELDS = {"conversation_id", "turn_id", "speaker", "text", "timestamp"}

    def load(self, path: str) -> list[TurnInput]:
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        turns: list[TurnInput] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                missing = self.REQUIRED_FIELDS - payload.keys()
                if missing:
                    raise ValueError(f"Line {line_number}: missing fields: {sorted(missing)}")
                turns.append(TurnInput(**payload))
        return turns
