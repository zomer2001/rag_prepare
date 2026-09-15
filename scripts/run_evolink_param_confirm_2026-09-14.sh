#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed20260908"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
OUTPUT_DIR="data/exports/evaluation/2026-09-14_evolink_param_confirm_gold500"
LOG_DIR="reports/2026-09-14/logs"
RUN_LOG="${LOG_DIR}/evolink_param_confirm_gold500.log"
PYTHON=".venv/bin/python"
METHODS="ours_evolink,ours_evoneed_deep_oneplan_shared_equal_total_norerank"

line_count() {
  local path="$1"
  [[ -f "${path}" ]] && awk 'NF {n++} END {print n+0}' "${path}" || printf '0\n'
}

show_status() {
  printf 'Ours+EvoLink (5,5): %s/500 queries\n' \
    "$(line_count "${OUTPUT_DIR}/ours_evolink.jsonl")"
  printf 'OnePlan-Shared (5,5): %s/500 queries\n' \
    "$(line_count "${OUTPUT_DIR}/ours_evoneed_deep_oneplan_shared_equal_total_norerank.jsonl")"
  printf 'Output: %s\nLog: %s\n' "${OUTPUT_DIR}" "${RUN_LOG}"
}

case "${1:---run}" in
  --status) show_status; exit 0 ;;
  --run) ;;
  *) printf 'Usage: %s [--run|--status]\n' "$0" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Missing project Python: %s\n' "${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" - "${RUNTIME_DIR}" "${SPLIT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

runtime = Path(sys.argv[1])
split = Path(sys.argv[2])
assert sum(1 for line in (split / "queries.jsonl").open() if line.strip()) == 500
light = json.loads((runtime / "lightrag_entity" / "evaluation_index.json").read_text())
events = json.loads((runtime / "events" / "evaluation_index.json").read_text())
assert light.get("chunk_count") == 2512, light
assert events.get("document_count") == 609, events
print("Validated fixed Gold500 indexes: 2512 chunks and 609 documents")
PY

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
printf 'Starting EvoLink (5,5) Gold500 confirmation at %s\n' "$(date -Iseconds)"

if docker compose version >/dev/null 2>&1; then
  docker compose up -d qdrant neo4j
else
  docker-compose up -d qdrant neo4j
fi

API_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
EVOLINK_SEED_LIMIT_PER_NAME=5 \
EVOLINK_MAX_EVENT_SEEDS=5 \
DEEPSEEK_BASE_URL="${API_BASE_URL}" \
"${PYTHON}" scripts/evaluate_retrieval.py \
  --dataset multihop_rag \
  --dataset-dir "${SPLIT_DIR}" \
  --methods "${METHODS}" \
  --cutoffs 5,10,20 \
  --chunking-strategy recursive \
  --chunk-size 700 \
  --chunk-overlap 100 \
  --runtime-dir "${RUNTIME_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fail-fast

show_status
"${PYTHON}" scripts/build_2026-09-14_experiment_report.py
printf 'EvoLink (5,5) Gold500 confirmation completed at %s\n' "$(date -Iseconds)"
