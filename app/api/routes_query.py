from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_runtime
from app.models.schemas import QueryRequest, QueryResponse
from app.services.runtime import AppRuntime

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_graph(
    request: QueryRequest,
    runtime: AppRuntime = Depends(get_runtime),
) -> QueryResponse:
    return runtime.retriever.query(request)
