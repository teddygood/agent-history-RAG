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
- Structured reasoning trace in API output
- Minimal D3 graph viewer

## Quickstart

### 1) Start services
```bash
cp .env.example .env
docker compose up --build
```

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
  -d '{"query":"continuous batching이 paged attention과 어떤 관계야?", "top_k":5}'
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
