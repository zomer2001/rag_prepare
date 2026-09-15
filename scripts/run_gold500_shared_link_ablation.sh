#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RUN_DATE="2026-09-11"
SEED="20260908"
SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed${SEED}"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
OUTPUT_DIR="data/exports/evaluation/${RUN_DATE}_shared_link_ablation_gold500"
LOG_DIR="reports/${RUN_DATE}/logs"
RUN_LOG="${LOG_DIR}/shared_link_ablation_gold500.log"
PYTHON=".venv/bin/python"

line_count() {
  local path="$1"
  [[ -f "${path}" ]] && awk 'NF {n++} END {print n+0}' "${path}" || printf '0\n'
}

show_status() {
  printf 'D1-QueryRewrite-SharedLink: %s/500 queries\n' \
    "$(line_count "${OUTPUT_DIR}/ours_d1_queryrewrite_sharedlink.jsonl")"
  printf 'OnePlan-Shared-EqualTotal-NoRerank: %s/500 queries\n' \
    "$(line_count "${OUTPUT_DIR}/ours_evoneed_deep_oneplan_shared_equal_total_norerank.jsonl")"
  printf 'Output: %s\n' "${OUTPUT_DIR}"
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

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Missing project Python: %s\n' "${PYTHON}" >&2
  exit 1
fi

# Query-only guard: use the frozen September Gold500 indexes and refuse to
# continue if the expected corpus/index fingerprints are not present.
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
printf 'Starting shared EvoLink ablation at %s\n' "$(date -Iseconds)"

if docker compose version >/dev/null 2>&1; then
  docker compose up -d qdrant neo4j
else
  docker-compose up -d qdrant neo4j
fi

# Keep the existing DEEPSEEK_API_KEY from .env/shell unchanged. Use the
# official DeepSeek-compatible endpoint for both query experiments.
DEEPSEEK_BASE_URL="https://api.deepseek.com" \
"${PYTHON}" scripts/evaluate_retrieval.py \
  --dataset multihop_rag \
  --dataset-dir "${SPLIT_DIR}" \
  --methods ours_d1_queryrewrite_sharedlink,ours_evoneed_deep_oneplan_shared_equal_total_norerank \
  --cutoffs 5,10,20 \
  --chunking-strategy recursive \
  --chunk-size 700 \
  --chunk-overlap 100 \
  --runtime-dir "${RUNTIME_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fail-fast

show_status
printf 'Shared EvoLink ablation completed at %s\n' "$(date -Iseconds)"
