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
- Pluggable embedder (`hash` / `sentence-transformers` / `auto`) with hash fallback
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

If you want model embeddings instead of hash embeddings:
```bash
# optional (host install)
pip install ".[model]"

# set in .env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Docker Compose mode:
```bash
INSTALL_MODEL_DEPS=true EMBEDDING_PROVIDER=sentence-transformers docker compose up --build
```

### 2) Ingest sample data
```bash
curl -X POST http://localhost:8000/ingest/jsonl \
  -H 'Content-Type: application/json' \
  -d '{"path":"/workspace/data/samples/conversation.jsonl"}'
```

### 2-1) Ingest Codex + Claude history (all conversations)
```bash
curl -X POST http://localhost:8000/ingest/history \
  -H 'Content-Type: application/json' \
  -d '{"source":"both","codex_history_path":"~/.codex/history.jsonl","claude_projects_root":"~/.claude/projects"}'
```

When running in Docker Compose, host directories are mounted read-only:
- `${HOME}/.codex` -> `/host-home/.codex`
- `${HOME}/.claude` -> `/host-home/.claude`

So this also works in containerized mode:
```bash
curl -X POST http://localhost:8000/ingest/history \
  -H 'Content-Type: application/json' \
  -d '{"source":"both","codex_history_path":"/host-home/.codex/history.jsonl","claude_projects_root":"/host-home/.claude/projects"}'
```

### 3) Query
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"continuous batching이 paged attention과 어떤 관계야?", "top_k":5, "max_hops":3, "beam_width":24, "prune_threshold":0.1, "importance_weight":0.18, "recency_weight":0.12, "recall_half_life_hours":72}'
```

### 4) Open viewer
- `http://localhost:8000/viewer`

### 5) Run retrieval benchmark (Recall@K / MRR / nDCG)
Use the sample benchmark set:
```bash
python scripts/eval_queries.py \
  --dataset data/eval/query_turn_relevance.sample.jsonl \
  --api-base http://localhost:8000 \
  --k 1,3,5
```

Save a detailed JSON report:
```bash
python scripts/eval_queries.py \
  --dataset data/eval/query_turn_relevance.sample.jsonl \
  --api-base http://localhost:8000 \
  --k 1,3,5,10 \
  --out-json ./artifacts/eval-report.json
```

Dataset JSONL row format:
```json
{"query":"continuous batching과 paged attention 관계","conversation_id":"conv-1","relevant_turn_uids":["conv-1:t4"]}
```

## API Summary
- `POST /ingest/conversation`
- `POST /ingest/jsonl`
- `POST /ingest/history`
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
- Embedder provider is selected by env: `EMBEDDING_PROVIDER=hash|sentence-transformers|auto`.
- If model provider init fails and `EMBEDDING_FALLBACK_TO_HASH=true`, it falls back to hash embedder.
- You can replace `app/services/extractor.py` with an external LLM extractor while keeping retrieval unchanged.
