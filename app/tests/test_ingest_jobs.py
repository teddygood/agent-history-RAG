from __future__ import annotations

import time

from app.models.schemas import HistoryIngestRequest, IngestStats, TurnInput
from app.services.ingest_jobs import IngestJobManager


class _DummyHistoryLoader:
    def __init__(self, turns: list[TurnInput]) -> None:
        self._turns = turns

    def load(self, **_kwargs) -> list[TurnInput]:
        return list(self._turns)


class _DummyIngestion:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._delay_seconds = delay_seconds

    def ingest_turns(self, turns, *, progress=None, cancel=None, **_kwargs) -> IngestStats:
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        stats = IngestStats(turns=len(list(turns)), entities=3, relations=2)
        if progress:
            progress(stats, stats.turns, stats.turns)
        return stats


class _DummyRuntime:
    def __init__(self, turns: list[TurnInput], delay_seconds: float = 0.0) -> None:
        self.history_loader = _DummyHistoryLoader(turns)
        self.ingestion = _DummyIngestion(delay_seconds=delay_seconds)


def test_ingest_job_runs_to_completion() -> None:
    turns = [
        TurnInput(
            conversation_id="codex:s1",
            turn_id="turn-000001",
            speaker="user",
            text="continuous batching improves throughput",
            timestamp="2026-02-09T00:00:00Z",
        )
    ]
    runtime = _DummyRuntime(turns)
    manager = IngestJobManager()

    job_id = manager.start_history_ingest(runtime=runtime, request=HistoryIngestRequest(source="codex", max_files=1))
    assert job_id

    deadline = time.time() + 2.0
    status = None
    while time.time() < deadline:
        job = manager.get_job(job_id)
        assert job is not None
        status = job["status"]
        if status == "succeeded":
            break
        time.sleep(0.01)

    assert status == "succeeded"
    job = manager.get_job(job_id)
    assert job is not None
    assert job["progress"]["turns_processed"] == 1
    assert job["progress"]["extracted_entities"] == 3
    assert job["progress"]["extracted_relations"] == 2


def test_ingest_job_is_idempotent_while_running() -> None:
    turns = [
        TurnInput(
            conversation_id="codex:s1",
            turn_id="turn-000001",
            speaker="user",
            text="paged attention uses paged KV cache",
            timestamp="2026-02-09T00:00:00Z",
        )
    ]
    runtime = _DummyRuntime(turns, delay_seconds=0.2)
    manager = IngestJobManager()
    request = HistoryIngestRequest(source="codex", max_files=1)

    job_id1 = manager.start_history_ingest(runtime=runtime, request=request)
    job_id2 = manager.start_history_ingest(runtime=runtime, request=request)

    assert job_id1 == job_id2

