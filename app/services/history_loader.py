from __future__ import annotations

import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import TurnInput


class AgentHistoryLoader:
    def load(
        self,
        *,
        source: str,
        codex_history_path: str,
        claude_projects_root: str,
        max_files: int | None = None,
    ) -> list[TurnInput]:
        turns: list[TurnInput] = []
        if source in ("codex", "both"):
            turns.extend(self.load_codex_history(codex_history_path))
        if source in ("claude", "both"):
            turns.extend(self.load_claude_projects(claude_projects_root, max_files=max_files))
        return turns

    def load_codex_history(self, history_path: str) -> list[TurnInput]:
        path = self._resolve_codex_history_path(history_path)
        if not path.exists():
            return []

        turns: list[TurnInput] = []
        per_session_counter: defaultdict[str, int] = defaultdict(int)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                session_id = str(payload.get("session_id") or "").strip()
                text = str(payload.get("text") or "").strip()
                ts = payload.get("ts")
                if not session_id or not text or ts is None:
                    continue

                per_session_counter[session_id] += 1
                turn_index = per_session_counter[session_id]
                timestamp = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                turns.append(
                    TurnInput(
                        conversation_id=f"codex:{session_id}",
                        turn_id=f"turn-{turn_index:06d}",
                        speaker="user",
                        text=text,
                        timestamp=timestamp,
                    )
                )
        return turns

    def load_claude_projects(self, projects_root: str, max_files: int | None = None) -> list[TurnInput]:
        root = self._resolve_claude_projects_root(projects_root)
        if not root.exists():
            return []

        pattern = str(root / "**" / "*.jsonl")
        files = sorted(glob.glob(pattern, recursive=True))
        if max_files is not None:
            files = files[:max_files]

        turns: list[TurnInput] = []
        for path_str in files:
            path = Path(path_str)
            fallback_session_id = path.stem
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    event_type = payload.get("type")
                    if event_type not in ("user", "assistant"):
                        continue

                    message = payload.get("message") or {}
                    text = self._extract_message_text(message)
                    if not text:
                        continue

                    session_id = str(payload.get("sessionId") or fallback_session_id)
                    role = str(message.get("role") or event_type)
                    speaker = "assistant" if role.lower() == "assistant" else "user"
                    timestamp = self._parse_timestamp(payload.get("timestamp"))
                    turn_id = str(payload.get("uuid") or f"line-{line_number:06d}")

                    turns.append(
                        TurnInput(
                            conversation_id=f"claude:{session_id}",
                            turn_id=turn_id,
                            speaker=speaker,
                            text=text,
                            timestamp=timestamp,
                        )
                    )
        return turns

    def _extract_message_text(self, message: dict) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if not isinstance(content, list):
            return ""

        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "text":
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            elif item_type == "tool_result":
                text = item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            elif item_type == "tool_use":
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    chunks.append(f"[tool_use] {name.strip()}")

        return "\n".join(chunks).strip()

    def _parse_timestamp(self, value: object) -> datetime:
        if isinstance(value, str) and value.strip():
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.now(timezone.utc)

    def _resolve_codex_history_path(self, configured_path: str) -> Path:
        primary = Path(configured_path).expanduser()
        candidates = [primary, Path("/host-home/.codex/history.jsonl")]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return primary

    def _resolve_claude_projects_root(self, configured_root: str) -> Path:
        primary = Path(configured_root).expanduser()
        candidates = [primary, Path("/host-home/.claude/projects")]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return primary
