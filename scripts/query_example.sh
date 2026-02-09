#!/usr/bin/env bash
set -euo pipefail

curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"continuous batching과 paged attention 관계 설명", "top_k":5}'
