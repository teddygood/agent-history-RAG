from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_runtime
from app.models.schemas import EntityOut, GraphSubgraphResponse, TurnOut
from app.services.runtime import AppRuntime

router = APIRouter(tags=["graph"])


@router.get("/entities/{entity_id}", response_model=EntityOut)
def get_entity(entity_id: str, runtime: AppRuntime = Depends(get_runtime)) -> EntityOut:
    row = runtime.store.get_entity_by_id(entity_id)
    if not row:
        raise HTTPException(status_code=404, detail="entity not found")
    return EntityOut(**row)


@router.get("/turns/{turn_uid}", response_model=TurnOut)
def get_turn(turn_uid: str, runtime: AppRuntime = Depends(get_runtime)) -> TurnOut:
    row = runtime.store.get_turn_by_uid(turn_uid)
    if not row:
        raise HTTPException(status_code=404, detail="turn not found")
    return TurnOut(**row)


@router.get("/graph/subgraph", response_model=GraphSubgraphResponse)
def get_subgraph(
    seed: str = Query(..., min_length=1),
    limit: int = Query(default=120, ge=10, le=500),
    runtime: AppRuntime = Depends(get_runtime),
) -> GraphSubgraphResponse:
    payload = runtime.store.build_subgraph(seed=seed, limit=limit)
    return GraphSubgraphResponse(**payload)
