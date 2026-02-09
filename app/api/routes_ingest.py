from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_runtime
from app.models.schemas import (
    ConversationIngestRequest,
    IngestResponse,
    JSONLIngestRequest,
    RebuildConversationRequest,
    TurnInput,
)
from app.services.runtime import AppRuntime

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/conversation", response_model=IngestResponse)
def ingest_conversation(
    request: ConversationIngestRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> IngestResponse:
    if not request.turns:
        raise HTTPException(status_code=400, detail="turns must not be empty")

    turns: list[TurnInput] = []
    for turn in request.turns:
        if request.conversation_id:
            turns.append(
                TurnInput(
                    conversation_id=request.conversation_id,
                    turn_id=turn.turn_id,
                    speaker=turn.speaker,
                    text=turn.text,
                    timestamp=turn.timestamp,
                )
            )
        else:
            turns.append(turn)

    stats = runtime.ingestion.ingest_turns(turns)
    conversation_id = request.conversation_id or turns[0].conversation_id

    return IngestResponse(
        conversation_id=conversation_id,
        ingested_turns=stats.turns,
        extracted_entities=stats.entities,
        extracted_relations=stats.relations,
    )


@router.post("/rebuild", response_model=IngestResponse)
def rebuild_conversation(
    request: RebuildConversationRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> IngestResponse:
    rows = runtime.store.get_turns_by_conversation(request.conversation_id)
    if not rows:
        raise HTTPException(status_code=404, detail="conversation not found")

    turns = [
        TurnInput(
            conversation_id=row["conversation_id"],
            turn_id=row["turn_id"],
            speaker=row["speaker"],
            text=row["text"],
            timestamp=_to_datetime(row["timestamp"]),
        )
        for row in rows
    ]

    stats = runtime.ingestion.ingest_turns(turns)
    return IngestResponse(
        conversation_id=request.conversation_id,
        ingested_turns=stats.turns,
        extracted_entities=stats.entities,
        extracted_relations=stats.relations,
    )


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value

    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native

    if hasattr(value, "iso_format"):
        value = value.iso_format()  # type: ignore[assignment]

    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    raise ValueError(f"Unsupported timestamp type: {type(value)}")


@router.post("/jsonl", response_model=IngestResponse)
def ingest_jsonl(
    request: JSONLIngestRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> IngestResponse:
    try:
        turns = runtime.jsonl_loader.load(request.path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    stats = runtime.ingestion.ingest_turns(turns)
    conversation_id = turns[0].conversation_id if turns else "unknown"

    return IngestResponse(
        conversation_id=conversation_id,
        ingested_turns=stats.turns,
        extracted_entities=stats.entities,
        extracted_relations=stats.relations,
    )
