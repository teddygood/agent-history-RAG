from __future__ import annotations

import json
from datetime import timezone

from app.services.history_loader import AgentHistoryLoader


def test_load_codex_history(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    rows = [
        {"session_id": "s1", "ts": 1762990713, "text": "hello"},
        {"session_id": "s1", "ts": 1762990749, "text": "world"},
        {"session_id": "s2", "ts": 1762990750, "text": "another"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    loader = AgentHistoryLoader()
    turns = loader.load_codex_history(str(path))

    assert len(turns) == 3
    assert turns[0].conversation_id == "codex:s1"
    assert turns[0].turn_id == "turn-000001"
    assert turns[1].turn_id == "turn-000002"
    assert turns[0].speaker == "user"
    assert turns[0].timestamp.tzinfo == timezone.utc


def test_load_claude_projects(tmp_path) -> None:
    projects_root = tmp_path / "projects" / "workspace"
    projects_root.mkdir(parents=True, exist_ok=True)
    session_file = projects_root / "abc123.jsonl"

    events = [
        {
            "type": "user",
            "sessionId": "abc123",
            "uuid": "u1",
            "timestamp": "2026-02-09T01:00:00Z",
            "message": {"role": "user", "content": "질문"},
        },
        {
            "type": "assistant",
            "sessionId": "abc123",
            "uuid": "a1",
            "timestamp": "2026-02-09T01:00:01Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "답변"}]},
        },
        {
            "type": "assistant",
            "sessionId": "abc123",
            "uuid": "a2",
            "timestamp": "2026-02-09T01:00:02Z",
            "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hidden"}]},
        },
    ]
    with session_file.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    loader = AgentHistoryLoader()
    turns = loader.load_claude_projects(str(tmp_path / "projects"))

    assert len(turns) == 2
    assert turns[0].conversation_id == "claude:abc123"
    assert turns[0].turn_id == "u1"
    assert turns[1].speaker == "assistant"
    assert turns[1].text == "답변"
