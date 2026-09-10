#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed20260908"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
OUTPUT_DIR="data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500"
LOG_DIR="reports/2026-09-09/logs"
RUN_LOG="${LOG_DIR}/evoneed_deep_oneplan_d1_norerank_gold500.log"
PYTHON=".venv/bin/python"
METHOD="ours_evoneed_deep_oneplan_d1_norerank"

line_count() {
  local path="$1"
  [[ -f "${path}" ]] && awk 'NF {n++} END {print n+0}' "${path}" || printf '0\n'
}

show_status() {
  printf 'EvoNeed-Deep-OnePlan-D1-NoRerank: %s/500 queries\n' \
    "$(line_count "${OUTPUT_DIR}/${METHOD}.jsonl")"
  printf 'Output: %s\n' "${OUTPUT_DIR}"
  printf 'Log: %s\n' "${RUN_LOG}"
}

case "${1:---run}" in
  --status)
    show_status
    exit 0
    ;;
  --run)
    ;;
  *)
    printf 'Usage: %s [--run|--status]\n' "$0" >&2
    exit 2
    ;;
esac

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
print("Validated fixed Gold-500 indexes: 2512 chunks, 609 documents")
PY

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
printf 'Starting EvoNeed-Deep-OnePlan-D1-NoRerank Gold-500 at %s\n' "$(date -Iseconds)"

if docker compose version >/dev/null 2>&1; then
  docker compose up -d qdrant neo4j
else
  docker-compose up -d qdrant neo4j
fi

AGENTIC_MAX_RETRIEVAL_ROUNDS=1 \
"${PYTHON}" scripts/evaluate_retrieval.py \
  --dataset multihop_rag --dataset-dir "${SPLIT_DIR}" \
  --methods "${METHOD}" \
  --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir "${RUNTIME_DIR}" --output-dir "${OUTPUT_DIR}" --fail-fast

show_status
printf 'EvoNeed-Deep-OnePlan-D1-NoRerank completed at %s\n' "$(date -Iseconds)"
