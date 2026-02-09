from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, TYPE_CHECKING
import uuid

from app.core.config import settings
from app.models.schemas import HistoryIngestRequest, IngestStats

if TYPE_CHECKING:
    from app.services.runtime import AppRuntime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class IngestJobState:
    job_id: str
    kind: str
    status: str = "queued"  # queued|running|succeeded|failed|cancelled
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    phase: str = "queued"
    turns_total: int | None = None
    turns_processed: int = 0
    extracted_entities: int = 0
    extracted_relations: int = 0
    updated_at: datetime | None = None
    error: str | None = None
    request_key: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False

    def to_out(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "phase": self.phase,
                "turns_total": self.turns_total,
                "turns_processed": self.turns_processed,
                "extracted_entities": self.extracted_entities,
                "extracted_relations": self.extracted_relations,
                "updated_at": self.updated_at,
            },
            "error": self.error,
        }


class IngestJobManager:
    """
    In-memory job registry for long-running ingest tasks.
    Designed for local single-node usage (no persistence across restarts).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJobState] = {}

    def start_history_ingest(self, *, runtime: AppRuntime, request: HistoryIngestRequest) -> str:
        key = self._history_request_key(request)
        with self._lock:
            for job in self._jobs.values():
                if job.kind == "history" and job.request_key == key and job.status in ("queued", "running"):
                    return job.job_id

            job_id = uuid.uuid4().hex
            job = IngestJobState(
                job_id=job_id,
                kind="history",
                status="queued",
                phase="queued",
                request_key=key,
                request={
                    "source": request.source,
                    "codex_history_path": request.codex_history_path,
                    "claude_projects_root": request.claude_projects_root,
                    "max_files": request.max_files,
                    "chunk_profile": request.chunk_profile,
                },
            )
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_history_ingest, args=(job_id, runtime, request), daemon=True)
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_out() if job else None

    def list_jobs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [job.to_out() for job in jobs[: max(1, limit)]]

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.cancel_requested = True
            job.updated_at = _utcnow()
            return True

    def _run_history_ingest(self, job_id: str, runtime: AppRuntime, request: HistoryIngestRequest) -> None:
        self._update(job_id, status="running", phase="loading_history", started_at=_utcnow(), updated_at=_utcnow())
        try:
            turns = runtime.history_loader.load(
                source=request.source,
                codex_history_path=request.codex_history_path,
                claude_projects_root=request.claude_projects_root,
                max_files=request.max_files,
            )
            if not turns:
                self._update(
                    job_id,
                    status="failed",
                    phase="failed",
                    finished_at=_utcnow(),
                    error="no history turns found from given paths",
                    updated_at=_utcnow(),
                )
                return

            self._update(job_id, phase="ingesting", turns_total=len(turns), updated_at=_utcnow())
            latest_processed = 0
            latest_total: int | None = len(turns)

            def is_cancelled() -> bool:
                with self._lock:
                    job = self._jobs.get(job_id)
                    return bool(job and job.cancel_requested)

            def on_progress(stats: IngestStats, processed: int, total: int | None) -> None:
                nonlocal latest_processed, latest_total
                latest_processed = processed
                latest_total = total
                self._update(
                    job_id,
                    phase="ingesting",
                    turns_total=total,
                    turns_processed=processed,
                    extracted_entities=stats.entities,
                    extracted_relations=stats.relations,
                    updated_at=_utcnow(),
                )

            stats = runtime.ingestion.ingest_turns(
                turns,
                chunk_profile=request.chunk_profile,
                progress=on_progress,
                progress_every=25,
                cancel=is_cancelled,
                batch_size=settings.ingest_batch_size,
                skip_existing_turns=settings.ingest_skip_existing_history,
            )

            if is_cancelled():
                self._update(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    turns_processed=latest_processed,
                    turns_total=latest_total,
                    extracted_entities=stats.entities,
                    extracted_relations=stats.relations,
                    finished_at=_utcnow(),
                    updated_at=_utcnow(),
                )
                return

            self._update(
                job_id,
                status="succeeded",
                phase="done",
                turns_processed=latest_processed,
                turns_total=latest_total,
                extracted_entities=stats.entities,
                extracted_relations=stats.relations,
                finished_at=_utcnow(),
                updated_at=_utcnow(),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                phase="failed",
                finished_at=_utcnow(),
                error=str(exc),
                updated_at=_utcnow(),
            )

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)

    @staticmethod
    def _history_request_key(request: HistoryIngestRequest) -> str:
        return "|".join(
            [
                "history",
                str(request.source),
                str(request.codex_history_path),
                str(request.claude_projects_root),
                str(request.max_files) if request.max_files is not None else "none",
                str(request.chunk_profile),
            ]
        )
