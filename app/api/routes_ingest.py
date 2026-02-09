from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_runtime
from app.models.schemas import ConversationIngestRequest, IngestResponse, JSONLIngestRequest, TurnInput
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
