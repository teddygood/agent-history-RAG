# Graph-Centric Conversation RAG (MVP)

This project builds a traceable RAG pipeline over conversation logs using an entity graph in Neo4j.

## Core Principles
- LLM-like extraction role only (entity/relation/intent extraction)
- Graph-first retrieval and reasoning
- Embeddings used only as a secondary signal (tie-break + pruning)
- Full traceability from query results back to source turn IDs

## What is implemented
- JSONL turn ingestion (`conversation_id`, `turn_id`, `speaker`, `text`, `timestamp`)
- Entity extraction including implicit technical concepts (heuristic extractor)
- Canonical entity normalization + alias merging
- Entity-to-entity semantic relation graph with evidence turn IDs
- Query-time entity seeding + graph traversal + Top-K turn retrieval
- Real-time query tuning sliders (max hops, beam width, prune threshold, importance/recency weights)
- Importance-aware and recall-time-aware ranking (`importance_score`, `last_recalled_at`)
- Structured reasoning trace in API output
- Minimal D3 graph viewer

## Quickstart

### 1) Start services
```bash
cp .env.example .env
docker compose up --build
```

If host port `8000` is busy, run:
```bash
API_HOST_PORT=8001 docker compose up --build
```
The API port is bound to `127.0.0.1` (localhost) by default.

### 2) Ingest sample data
```bash
curl -X POST http://localhost:8000/ingest/jsonl \
  -H 'Content-Type: application/json' \
  -d '{"path":"/workspace/data/samples/conversation.jsonl"}'
```

### 3) Query
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"continuous batching이 paged attention과 어떤 관계야?", "top_k":5, "max_hops":3, "beam_width":24, "prune_threshold":0.1, "importance_weight":0.18, "recency_weight":0.12, "recall_half_life_hours":72}'
```

### 4) Open viewer
- `http://localhost:8000/viewer`

## API Summary
- `POST /ingest/conversation`
- `POST /ingest/jsonl`
- `POST /ingest/rebuild`
- `POST /query`
- `GET /entities/{entity_id}`
- `GET /turns/{turn_uid}`
- `GET /graph/subgraph?seed=<entity or keyword>&limit=120`

## JSONL Format
```json
{"conversation_id":"conv-1","turn_id":"t1","speaker":"user","text":"continuous batching은 throughput을 높인다.","timestamp":"2026-02-09T09:00:00Z"}
```

## Notes
- Current extractor is deterministic heuristic logic for repeatability.
- You can replace `app/services/extractor.py` with an external LLM extractor while keeping retrieval unchanged.
