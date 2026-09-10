#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODE="status"
RUN_DATE="$(date +%F)"
for argument in "$@"; do
  case "${argument}" in
    --run)
      MODE="run"
      ;;
    --status)
      MODE="status"
      ;;
    --run-date=*)
      RUN_DATE="${argument#*=}"
      ;;
    *)
      printf 'Unknown option: %s\n' "${argument}" >&2
      printf 'Usage: %s [--status|--run] [--run-date=YYYY-MM-DD]\n' "$0" >&2
      exit 2
      ;;
  esac
done

DATASET_DIR="data/benchmarks/multihop_rag"
FULL_INPUT_DIR="data/benchmarks/external_multihop_full609_q2556"
FULL_NAME="multihop_full609_q2556_recursive700"
FULL_CORPUS="${FULL_INPUT_DIR}/${FULL_NAME}_corpus.json"
FULL_QUERIES="${FULL_INPUT_DIR}/${FULL_NAME}.json"
Q100_INPUT_DIR="data/benchmarks/external_multihop_full609_q100"
Q100_CORPUS="${Q100_INPUT_DIR}/multihop_full609_q100_recursive700_corpus.json"
Q100_QUERIES="${Q100_INPUT_DIR}/multihop_full609_q100_recursive700.json"
EXPECTED_CORPUS_SHA="3e3d2ff64c1723c0120267ef875414a2dc9cb33ff689c14502a7284fa0c303f0"

E2RAG_RUNTIME="data/e2rag/evaluation/2026-09-04_multihop_full609_q100"
E2RAG_OUTPUT_DIR="data/exports/evaluation/${RUN_DATE}_e2rag_full2556"
E2RAG_RAW="${E2RAG_OUTPUT_DIR}/raw.jsonl"
E2RAG_PYTHON="data/benchmarks/_venvs/e2rag/bin/python"

GRAPHRAG_RUNTIME="data/graphrag/evaluation/${RUN_DATE}_multihop_full609_q2556"
GRAPHRAG_OUTPUT_DIR="data/exports/evaluation/${RUN_DATE}_graphrag_local_full2556"
GRAPHRAG_RAW="${GRAPHRAG_OUTPUT_DIR}/raw.jsonl"

REPORT_DIR="reports/${RUN_DATE}"
LOG_DIR="${REPORT_DIR}/logs"
RUN_LOG="${LOG_DIR}/full_graph_baselines_2556.log"

line_count() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    awk 'NF {count += 1} END {print count + 0}' "${path}"
  else
    printf '0\n'
  fi
}

progress_value() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'not-started\n'
    return
  fi
  uv run python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); print("{}/{}".format(v.get("completed_count", 0), v.get("total_count", "?")))' \
    "${path}"
}

validate_summary() {
  local path="$1"
  uv run python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); assert v.get("query_count")==2556, v; assert v.get("evaluated_query_count")==2255, v; assert v.get("error_count")==0, v; print("Validated score summary:", sys.argv[1])' \
    "${path}"
}

show_status() {
  printf 'Run date: %s\n' "${RUN_DATE}"
  printf 'E2RAG entity index: %s\n' \
    "$(progress_value "${E2RAG_RUNTIME}/entity_index_progress.json")"
  printf 'E2RAG event index: %s\n' \
    "$(progress_value "${E2RAG_RUNTIME}/event_index_progress.json")"
  printf 'E2RAG index complete: %s\n' \
    "$([[ -f "${E2RAG_RUNTIME}/index_complete.json" ]] && printf yes || printf no)"
  printf 'E2RAG query results: %s/2556\n' "$(line_count "${E2RAG_RAW}")"
  printf 'GraphRAG index complete: %s\n' \
    "$([[ -f "${GRAPHRAG_RUNTIME}/index_complete.json" ]] && printf yes || printf no)"
  printf 'GraphRAG query results: %s/2556\n' "$(line_count "${GRAPHRAG_RAW}")"
  printf 'Combined log: %s\n' "${RUN_LOG}"
}

if [[ "${MODE}" == "status" ]]; then
  show_status
  printf '\nNo experiment was started. Use --run to execute the resumable pipeline.\n'
  exit 0
fi

mkdir -p "${LOG_DIR}" "${E2RAG_OUTPUT_DIR}" "${GRAPHRAG_OUTPUT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

printf 'Starting full E2RAG and GraphRAG pipeline at %s\n' "$(date -Iseconds)"
printf 'Only September full-2556 artifacts are in scope; n30 and August runs are excluded.\n'

if [[ ! -f ".env" ]]; then
  printf 'Missing .env. API credentials are required, but their values will not be printed.\n' >&2
  exit 1
fi
if [[ ! -x "${E2RAG_PYTHON}" ]]; then
  printf 'Missing E2RAG Python environment: %s\n' "${E2RAG_PYTHON}" >&2
  exit 1
fi

if [[ -f "${FULL_CORPUS}" && -f "${FULL_QUERIES}" ]]; then
  printf 'Using existing full exported inputs.\n'
elif [[ -e "${FULL_CORPUS}" || -e "${FULL_QUERIES}" ]]; then
  printf 'Full export is partial. Refusing to overwrite it; inspect %s first.\n' \
    "${FULL_INPUT_DIR}" >&2
  exit 1
else
  uv run --extra evaluation python scripts/export_hipporag_dataset.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${FULL_INPUT_DIR}" \
    --name "${FULL_NAME}" \
    --chunking-strategy recursive \
    --chunk-size 700 \
    --chunk-overlap 100
fi

uv run python - "${FULL_CORPUS}" "${FULL_QUERIES}" "${EXPECTED_CORPUS_SHA}" <<'PY'
import collections
import hashlib
import json
import sys

corpus_path, queries_path, expected_sha = sys.argv[1:]
with open(corpus_path, encoding="utf-8") as handle:
    corpus = json.load(handle)
with open(queries_path, encoding="utf-8") as handle:
    queries = json.load(handle)
with open(corpus_path, "rb") as handle:
    actual_sha = hashlib.sha256(handle.read()).hexdigest()
query_ids = [str(row["id"]) for row in queries]
types = collections.Counter(str(row.get("question_type")) for row in queries)
expected_types = {
    "comparison_query": 856,
    "inference_query": 816,
    "temporal_query": 583,
    "null_query": 301,
}
if len(corpus) != 2512:
    raise SystemExit(f"Expected 2512 chunks, found {len(corpus)}")
if len(queries) != 2556 or len(set(query_ids)) != 2556:
    raise SystemExit(
        f"Expected 2556 unique queries, found {len(queries)} rows/{len(set(query_ids))} IDs"
    )
if actual_sha != expected_sha:
    raise SystemExit(f"Corpus SHA mismatch: {actual_sha} != {expected_sha}")
if dict(types) != expected_types:
    raise SystemExit(f"Question-type counts differ: {dict(types)} != {expected_types}")
print(json.dumps({"chunks": len(corpus), "queries": len(queries), "types": types}, sort_keys=True))
PY

"${E2RAG_PYTHON}" scripts/run_e2rag_baseline.py \
  --corpus "${Q100_CORPUS}" \
  --queries "${Q100_QUERIES}" \
  --runtime-dir "${E2RAG_RUNTIME}" \
  --output "${E2RAG_RAW}" \
  --stage doctor \
  --variant vanilla \
  --limit 100 \
  --top-k 20 \
  --index-batch-size 8 \
  --llm-concurrency 4 \
  --embedding-concurrency 2

if [[ ! -f "${E2RAG_RUNTIME}/index_complete.json" ]]; then
  "${E2RAG_PYTHON}" scripts/run_e2rag_baseline.py \
    --corpus "${Q100_CORPUS}" \
    --queries "${Q100_QUERIES}" \
    --runtime-dir "${E2RAG_RUNTIME}" \
    --output "${E2RAG_RAW}" \
    --stage index \
    --variant vanilla \
    --limit 100 \
    --top-k 20 \
    --index-batch-size 8 \
    --llm-concurrency 4 \
    --embedding-concurrency 2
else
  printf 'E2RAG index marker exists; skipping index stage.\n'
fi

for marker in entity_index_complete.json event_index_complete.json index_complete.json; do
  if [[ ! -f "${E2RAG_RUNTIME}/${marker}" ]]; then
    printf 'E2RAG marker is missing after indexing: %s\n' "${marker}" >&2
    exit 1
  fi
done

"${E2RAG_PYTHON}" scripts/run_e2rag_baseline.py \
  --corpus "${FULL_CORPUS}" \
  --queries "${FULL_QUERIES}" \
  --runtime-dir "${E2RAG_RUNTIME}" \
  --output "${E2RAG_RAW}" \
  --stage retrieve \
  --variant vanilla \
  --top-k 20 \
  --allow-query-set-change

uv run python - "${FULL_QUERIES}" "${E2RAG_RAW}" <<'PY'
import json
import sys

queries_path, raw_path = sys.argv[1:]
with open(queries_path, encoding="utf-8") as handle:
    expected = {str(row["id"]) for row in json.load(handle)}
with open(raw_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
ids = [str(row["query_id"]) for row in rows]
if len(rows) != 2556 or len(set(ids)) != 2556:
    raise SystemExit(f"Incomplete/duplicate results: {len(rows)} rows, {len(set(ids))} unique")
if set(ids) != expected:
    raise SystemExit("Raw result query IDs do not exactly match the full query set")
if any(len(row.get("retrieved") or []) > 20 for row in rows):
    raise SystemExit("At least one query returned more than top-20 results")
print("Validated 2556 complete E2RAG results")
PY

uv run --extra evaluation python scripts/score_e2rag_results.py \
  --dataset-dir "${DATASET_DIR}" \
  --raw-results "${E2RAG_RAW}" \
  --output-dir "${E2RAG_OUTPUT_DIR}" \
  --dataset-name multihop_rag \
  --cutoffs 5,10,20
validate_summary "${E2RAG_OUTPUT_DIR}/e2rag_original_summary.json"

if [[ ! -f "${GRAPHRAG_RUNTIME}/settings.yaml" ]]; then
  uv run python scripts/run_graphrag_baseline.py \
    --corpus "${FULL_CORPUS}" \
    --queries "${FULL_QUERIES}" \
    --runtime-dir "${GRAPHRAG_RUNTIME}" \
    --output "${GRAPHRAG_RAW}" \
    --stage prepare \
    --top-k 20 \
    --concurrency 4
else
  printf 'GraphRAG runtime is already prepared; skipping prepare stage.\n'
fi

if [[ ! -f "${GRAPHRAG_RUNTIME}/index_complete.json" ]]; then
  uv run python scripts/run_graphrag_baseline.py \
    --corpus "${FULL_CORPUS}" \
    --queries "${FULL_QUERIES}" \
    --runtime-dir "${GRAPHRAG_RUNTIME}" \
    --output "${GRAPHRAG_RAW}" \
    --stage index \
    --top-k 20 \
    --concurrency 4
else
  printf 'GraphRAG index marker exists; skipping index stage.\n'
fi

uv run python scripts/run_graphrag_baseline.py \
  --corpus "${FULL_CORPUS}" \
  --queries "${FULL_QUERIES}" \
  --runtime-dir "${GRAPHRAG_RUNTIME}" \
  --output "${GRAPHRAG_RAW}" \
  --stage retrieve \
  --top-k 20 \
  --concurrency 4

uv run python - "${FULL_QUERIES}" "${GRAPHRAG_RAW}" <<'PY'
import json
import sys

queries_path, raw_path = sys.argv[1:]
with open(queries_path, encoding="utf-8") as handle:
    expected = {str(row["id"]) for row in json.load(handle)}
with open(raw_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
ids = [str(row["query_id"]) for row in rows]
if len(rows) != 2556 or len(set(ids)) != 2556:
    raise SystemExit(f"Incomplete/duplicate results: {len(rows)} rows, {len(set(ids))} unique")
if set(ids) != expected:
    raise SystemExit("Raw result query IDs do not exactly match the full query set")
if any(len(row.get("retrieved") or []) > 20 for row in rows):
    raise SystemExit("At least one query returned more than top-20 results")
print("Validated 2556 complete GraphRAG results")
PY

uv run --extra evaluation python scripts/score_external_baseline.py \
  --dataset-dir "${DATASET_DIR}" \
  --raw-results "${GRAPHRAG_RAW}" \
  --output-dir "${GRAPHRAG_OUTPUT_DIR}" \
  --method graphrag_local \
  --cutoffs 5,10,20
validate_summary "${GRAPHRAG_OUTPUT_DIR}/graphrag_local_summary.json"

show_status
printf 'Pipeline finished at %s\n' "$(date -Iseconds)"
