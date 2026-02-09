from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_graph import router as graph_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.core.logging import configure_logging
from app.services.runtime import AppRuntime

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = AppRuntime.create()
    app.state.runtime = runtime
    yield
    runtime.close()


app = FastAPI(title="Graph-Centric Conversation RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(graph_router)

viewer_dir = Path(__file__).resolve().parents[1] / "viewer" / "public"
if viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")


@app.get("/health")
def health() -> dict[str, object]:
    runtime: AppRuntime = app.state.runtime
    payload: dict[str, object] = {"status": "ok"}
    payload.update(runtime.runtime_profile())
    return payload
