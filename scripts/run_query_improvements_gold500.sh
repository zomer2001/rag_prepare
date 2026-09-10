#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RUN_DATE="2026-09-08"
SEED="20260908"
SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed${SEED}"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
MAIN_OUTPUT="data/exports/evaluation/${RUN_DATE}_query_improvements_gold500"
PLANNED_OUTPUT="data/exports/evaluation/${RUN_DATE}_evoneed_planned_route20_gold500"
LOG_DIR="reports/${RUN_DATE}/logs"
RUN_LOG="${LOG_DIR}/query_improvements_gold500.log"
PYTHON=".venv/bin/python"

line_count() {
  local path="$1"
  [[ -f "${path}" ]] && awk 'NF {n++} END {print n+0}' "${path}" || printf '0\n'
}

show_status() {
  printf 'Improved controlled results: %s/4 methods\n' \
    "$(find "${MAIN_OUTPUT}" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
  printf 'High-budget planned results: %s/500 queries\n' \
    "$(line_count "${PLANNED_OUTPUT}/ours_evoneed_planned.jsonl")"
  printf 'Log: %s\n' "${RUN_LOG}"
}

if [[ "${1:---run}" == "--status" ]]; then
  show_status
  exit 0
fi
if [[ "${1:---run}" != "--run" ]]; then
  printf 'Usage: %s [--run|--status]\n' "$0" >&2
  exit 2
fi

# Query-only guard: refuse to start if any existing controlled index is
# incomplete.  The evaluation runner will therefore only load these indexes.
"${PYTHON}" - "${RUNTIME_DIR}" "${SPLIT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

runtime = Path(sys.argv[1])
split = Path(sys.argv[2])
assert sum(1 for line in (split / "queries.jsonl").open() if line.strip()) == 500
light = json.loads((runtime / "lightrag_entity" / "evaluation_index.json").read_text())
events = json.loads((runtime / "events" / "evaluation_index.json").read_text())
event_docs = json.loads((runtime / "events" / "document_index.json").read_text())
evolink_paths = list((runtime / "evolink").glob("*/document_index.json"))
assert light.get("chunk_count") == 2512, light
assert events.get("document_count") == 609, events
assert len(event_docs.get("documents") or {}) == 609, event_docs
assert len(evolink_paths) == 1, evolink_paths
evolink = json.loads(evolink_paths[0].read_text())
assert len(evolink.get("documents") or []) == 609, evolink
print("Validated fixed indexes: 2512 chunks and 609 indexed documents")
PY

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
printf 'Starting query-only improvement experiment at %s\n' "$(date -Iseconds)"

if docker compose version >/dev/null 2>&1; then
  docker compose up -d qdrant neo4j
else
  docker-compose up -d qdrant neo4j
fi

AGENTIC_MAX_RETRIEVAL_ROUNDS=1 \
EVONEED_RETRIEVAL_BUDGET=60 \
EVONEED_MIN_ROUTE_BUDGET=5 \
"${PYTHON}" scripts/evaluate_retrieval.py \
  --dataset multihop_rag --dataset-dir "${SPLIT_DIR}" \
  --methods ours,ours_evolink,ours_evoneed_equal,ours_evoneed_dynamic \
  --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir "${RUNTIME_DIR}" --output-dir "${MAIN_OUTPUT}" --fail-fast

# This reproduces the earlier planned-needs acquisition depth: every active
# planner route receives up to 20 candidates (at most 60 in total), while the
# run remains retrieval-only with one planning call and one retrieval round.
AGENTIC_MAX_RETRIEVAL_ROUNDS=1 \
EVONEED_RETRIEVAL_BUDGET=20 \
EVONEED_MIN_ROUTE_BUDGET=5 \
"${PYTHON}" scripts/evaluate_retrieval.py \
  --dataset multihop_rag --dataset-dir "${SPLIT_DIR}" \
  --methods ours_evoneed_planned \
  --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir "${RUNTIME_DIR}" --output-dir "${PLANNED_OUTPUT}" --fail-fast

show_status
printf 'Query-only improvement experiment completed at %s\n' "$(date -Iseconds)"
