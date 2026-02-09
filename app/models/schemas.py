from __future__ import annotations

from datetime import datetime
from typing import Literal
from typing import Any

from pydantic import BaseModel, Field, field_validator

ChunkProfileLiteral = Literal["auto", "default", "structured"]


class TurnInput(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    turn_id: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    timestamp: datetime


class ConversationIngestRequest(BaseModel):
    conversation_id: str | None = None
    chunk_profile: ChunkProfileLiteral = "auto"
    turns: list[TurnInput] = Field(default_factory=list)


class JSONLIngestRequest(BaseModel):
    path: str
    chunk_profile: ChunkProfileLiteral = "auto"


class HistoryIngestRequest(BaseModel):
    source: Literal["codex", "claude", "both"] = "both"
    codex_history_path: str = "~/.codex/history.jsonl"
    claude_projects_root: str = "~/.claude/projects"
    max_files: int | None = Field(default=None, ge=1, le=10000)
    chunk_profile: ChunkProfileLiteral = "auto"


class RebuildConversationRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    chunk_profile: ChunkProfileLiteral = "auto"


class IngestResponse(BaseModel):
    conversation_id: str
    ingested_turns: int
    extracted_entities: int
    extracted_relations: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    max_hops: int | None = Field(default=None, ge=1, le=6)
    beam_width: int | None = Field(default=None, ge=4, le=200)
    prune_threshold: float | None = Field(default=None, ge=0.01, le=1.0)
    hybrid_enabled: bool | None = None
    graph_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    lexical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    importance_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_half_life_hours: int | None = Field(default=None, ge=1, le=720)
    rerank_enabled: bool | None = None
    rerank_model: str | None = None
    rerank_top_n: int | None = Field(default=None, ge=1, le=200)


class TracePathStep(BaseModel):
    from_entity_id: str
    from_entity_name: str
    to_entity_id: str
    to_entity_name: str
    relation_type: str
    evidence_turn_ids: list[str] = Field(default_factory=list)


class TurnResult(BaseModel):
    turn_uid: str
    conversation_id: str
    turn_id: str
    speaker: str
    timestamp: datetime
    text: str
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    matched_entities: list[str] = Field(default_factory=list)
    path_summary: list[TracePathStep] = Field(default_factory=list)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    importance_score: float = 0.0
    recency_factor: float = 1.0
    chunk_profile: str | None = None
    chunk_count: int = 1
    last_recalled_at: datetime | None = None


class QueryResponse(BaseModel):
    query: str
    top_k: int
    matched_seed_entities: list[str] = Field(default_factory=list)
    selected_turns: list[TurnResult] = Field(default_factory=list)
    pruned_paths: int
    applied_params: dict[str, float | int | bool | str] = Field(default_factory=dict)


class EntityOut(BaseModel):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str = "concept"
    description: str = ""


class TurnOut(BaseModel):
    turn_uid: str
    conversation_id: str
    turn_id: str
    speaker: str
    text: str
    timestamp: datetime


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str
    score: float = 0.0


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    evidence_turn_ids: list[str] = Field(default_factory=list)


class GraphSubgraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    surface: str
    canonical_name: str
    entity_type: str = "concept"
    description: str = ""
    confidence: float = 0.7

    @field_validator("canonical_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip()


class ExtractedRelation(BaseModel):
    source_canonical: str
    target_canonical: str
    relation_type: str
    confidence: float = 0.7


class IngestStats(BaseModel):
    turns: int = 0
    entities: int = 0
    relations: int = 0
    debug: dict[str, Any] = Field(default_factory=dict)
