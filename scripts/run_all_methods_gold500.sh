#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RUN_DATE="2026-09-08"
SEED="20260908"
MODE="status"
FRESH_RESULTS=0
for argument in "$@"; do
  case "${argument}" in
    --run) MODE="run" ;;
    --status) MODE="status" ;;
    --fresh-results) FRESH_RESULTS=1 ;;
    --run-date=*) RUN_DATE="${argument#*=}" ;;
    *) printf 'Unknown option: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done

SOURCE_DATASET="data/benchmarks/multihop_rag"
SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed${SEED}"
EXTERNAL_DIR="data/benchmarks/external_multihop_full609_gold500_seed${SEED}"
EXTERNAL_NAME="multihop_full609_gold500_seed${SEED}_recursive700"
EXTERNAL_CORPUS="${EXTERNAL_DIR}/${EXTERNAL_NAME}_corpus.json"
EXTERNAL_QUERIES="${EXTERNAL_DIR}/${EXTERNAL_NAME}.json"
EXPECTED_CORPUS_SHA="3e3d2ff64c1723c0120267ef875414a2dc9cb33ff689c14502a7284fa0c303f0"

INTERNAL_RUNTIME="data/lightrag/evaluation/multihop_full609_elc_v1"
INTERNAL_OUTPUT="data/exports/evaluation/${RUN_DATE}_all_internal_gold500"
INTERNAL_METHODS="naive,hybrid,light_entity,ours,ours_evolink,ours_evoneed_equal,ours_evoneed_dynamic"

HIPPORAG_RUNTIME="data/hipporag/evaluation/2026-09-04_multihop_full609_q100"
HIPPORAG_OUTPUT="data/exports/evaluation/${RUN_DATE}_hipporag2_gold500"
HIPPORAG_PYTHON="data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python"

EVENTRAG_RUNTIME="data/eventrag/evaluation/2026-09-04_full609_q100_isolated_v1"
EVENTRAG_OUTPUT="data/exports/evaluation/${RUN_DATE}_eventrag_budgeted_gold500"

E2RAG_RUNTIME="data/e2rag/evaluation/2026-09-04_multihop_full609_q100"
E2RAG_OUTPUT="data/exports/evaluation/${RUN_DATE}_e2rag_gold500"
E2RAG_PYTHON="data/benchmarks/_venvs/e2rag/bin/python"

GRAPHRAG_RUNTIME="data/graphrag/evaluation/${RUN_DATE}_full609_gold500"
GRAPHRAG_OUTPUT="data/exports/evaluation/${RUN_DATE}_graphrag_local_gold500"

REPORT_DIR="reports/${RUN_DATE}"
LOG_DIR="${REPORT_DIR}/logs"
RUN_LOG="${LOG_DIR}/all_methods_gold500.log"

line_count() {
  local path="$1"
  [[ -f "${path}" ]] && awk 'NF {n++} END {print n+0}' "${path}" || printf '0\n'
}

progress_value() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then printf 'not-started\n'; return; fi
  uv run python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); print("{}/{}".format(v.get("completed_count",0),v.get("total_count","?")))' \
    "${path}"
}

show_status() {
  printf 'Gold-500 split: %s queries\n' "$(line_count "${SPLIT_DIR}/queries.jsonl")"
  printf 'E2RAG entity index: %s\n' "$(progress_value "${E2RAG_RUNTIME}/entity_index_progress.json")"
  printf 'E2RAG event index: %s\n' "$(progress_value "${E2RAG_RUNTIME}/event_index_progress.json")"
  printf 'Internal methods completed files: %s/7\n' \
    "$(find "${INTERNAL_OUTPUT}" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
  printf 'HippoRAG results: %s/500\n' "$(line_count "${HIPPORAG_OUTPUT}/raw.jsonl")"
  printf 'EventRAG results: %s/500\n' "$(line_count "${EVENTRAG_OUTPUT}/raw.jsonl")"
  printf 'E2RAG results: %s/500\n' "$(line_count "${E2RAG_OUTPUT}/raw.jsonl")"
  printf 'GraphRAG results: %s/500\n' "$(line_count "${GRAPHRAG_OUTPUT}/raw.jsonl")"
  printf 'Log: %s\n' "${RUN_LOG}"
}

archive_query_results() {
  local stamp backup target
  stamp="$(date '+%Y%m%d_%H%M%S')"
  backup="${REPORT_DIR}/previous_query_results_${stamp}"
  mkdir -p "${backup}"
  for target in \
    "${INTERNAL_OUTPUT}" \
    "${HIPPORAG_OUTPUT}" \
    "${EVENTRAG_OUTPUT}" \
    "${E2RAG_OUTPUT}" \
    "${GRAPHRAG_OUTPUT}"; do
    if [[ -e "${target}" ]]; then
      mv "${target}" "${backup}/$(basename "${target}")"
    fi
  done
  if [[ -f "${RUN_LOG}" ]]; then
    mv "${RUN_LOG}" "${backup}/$(basename "${RUN_LOG}")"
  fi
  printf 'Archived prior Gold-500 query outputs under %s\n' "${backup}"
}

archive_incompatible_evoneed_checkpoints() {
  local method checkpoint summary stamp backup
  backup=""
  for method in ours_evoneed_equal ours_evoneed_dynamic; do
    checkpoint="${INTERNAL_OUTPUT}/${method}.jsonl"
    summary="${INTERNAL_OUTPUT}/${method}_summary.json"
    if [[ ! -s "${checkpoint}" || -f "${summary}" ]]; then
      continue
    fi
    if uv run python - "${checkpoint}" <<'PY'
import json, sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
compatible = bool(rows) and all(
    row.get("trajectory", {}).get("llm_calls") == 1
    and [event.get("node") for event in row.get("trajectory", {}).get("events", [])]
    == ["plan_needs", "retrieve_needs"]
    for row in rows
)
raise SystemExit(0 if compatible else 1)
PY
    then
      continue
    fi
    if [[ -z "${backup}" ]]; then
      stamp="$(date '+%Y%m%d_%H%M%S')"
      backup="${REPORT_DIR}/incompatible_evoneed_results_${stamp}"
      mkdir -p "${backup}"
    fi
    mv "${checkpoint}" "${backup}/$(basename "${checkpoint}")"
    printf 'Archived incompatible partial EvoNeed checkpoint: %s\n' "${checkpoint}"
  done
  if [[ -n "${backup}" ]]; then
    printf 'Prior partial EvoNeed rows remain recoverable under %s\n' "${backup}"
  fi
}

validate_raw() {
  local raw="$1"
  uv run python - "${EXTERNAL_QUERIES}" "${raw}" <<'PY'
import json, sys
queries_path, raw_path = sys.argv[1:]
with open(queries_path, encoding="utf-8") as handle:
    expected = {str(row["id"]) for row in json.load(handle)}
with open(raw_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
ids = [str(row["query_id"]) for row in rows]
if len(rows) != 500 or len(set(ids)) != 500 or set(ids) != expected:
    raise SystemExit(f"Expected the same 500 unique queries, got {len(rows)}/{len(set(ids))}")
if any(len(row.get("retrieved") or []) > 20 for row in rows):
    raise SystemExit("A query returned more than 20 final results")
print("Validated 500 shared-query results:", raw_path)
PY
}

validate_summary() {
  local summary="$1"
  uv run python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); assert v.get("query_count")==500,v; assert v.get("evaluated_query_count")==500,v; assert v.get("error_count")==0,v; print("Validated summary:",sys.argv[1])' \
    "${summary}"
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-90}"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --fail --silent --show-error --max-time 2 "${url}" >/dev/null 2>&1; then
      printf '%s is ready: %s\n' "${name}" "${url}"
      return 0
    fi
    sleep 2
  done
  printf '%s did not become ready after %s seconds: %s\n' \
    "${name}" "$((attempts * 2))" "${url}" >&2
  return 1
}

ensure_main_services() {
  if ! docker info >/dev/null 2>&1; then
    if ! command -v colima >/dev/null 2>&1; then
      printf 'Docker is unavailable and Colima is not installed.\n' >&2
      return 1
    fi
    printf 'Container runtime is stopped; starting Colima.\n'
    colima start
  fi

  printf 'Starting the main Qdrant and Neo4j services.\n'
  if docker compose version >/dev/null 2>&1; then
    docker compose up -d qdrant neo4j
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose up -d qdrant neo4j
  else
    printf 'Neither docker compose nor docker-compose is available.\n' >&2
    return 1
  fi
  wait_for_http "Qdrant" "http://127.0.0.1:6333/healthz"
  wait_for_http "Neo4j" "http://127.0.0.1:7474"
}

if [[ "${MODE}" == "status" ]]; then
  show_status
  printf 'This status check did not start another process.\n'
  exit 0
fi

if [[ "${FRESH_RESULTS}" -eq 1 ]]; then
  archive_query_results
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
printf 'Starting shared Gold-500 comparison at %s\n' "$(date -Iseconds)"

if [[ ! -f "${SPLIT_DIR}/split_manifest.json" ]]; then
  uv run python scripts/prepare_gold_stratified_split.py \
    --source-dir "${SOURCE_DATASET}" \
    --output-dir "${SPLIT_DIR}" \
    --sample-size 500 \
    --seed "${SEED}"
fi

if [[ ! -f "${EXTERNAL_CORPUS}" || ! -f "${EXTERNAL_QUERIES}" ]]; then
  if [[ -e "${EXTERNAL_CORPUS}" || -e "${EXTERNAL_QUERIES}" ]]; then
    printf 'External Gold-500 export is partial; refusing to overwrite it.\n' >&2
    exit 1
  fi
  uv run --extra evaluation python scripts/export_hipporag_dataset.py \
    --dataset-dir "${SPLIT_DIR}" \
    --output-dir "${EXTERNAL_DIR}" \
    --name "${EXTERNAL_NAME}" \
    --chunking-strategy recursive \
    --chunk-size 700 \
    --chunk-overlap 100
fi

uv run python - "${SPLIT_DIR}/split_manifest.json" "${EXTERNAL_CORPUS}" \
  "${EXTERNAL_QUERIES}" "${EXPECTED_CORPUS_SHA}" <<'PY'
import hashlib, json, sys
manifest_path, corpus_path, queries_path, expected_sha = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
corpus = json.load(open(corpus_path, encoding="utf-8"))
queries = json.load(open(queries_path, encoding="utf-8"))
actual_sha = hashlib.sha256(open(corpus_path, "rb").read()).hexdigest()
assert manifest["sample_size"] == 500, manifest
assert manifest["selected_counts"] == {
    "comparison_query": 190,
    "inference_query": 181,
    "temporal_query": 129,
}, manifest
assert len(corpus) == 2512, len(corpus)
assert len(queries) == len({str(row["id"]) for row in queries}) == 500
assert actual_sha == expected_sha, (actual_sha, expected_sha)
print("Validated shared Gold-500 split and unchanged 2512-chunk corpus")
PY

if [[ ! -f "${E2RAG_RUNTIME}/index_complete.json" ]]; then
  printf 'E2RAG index is incomplete; refusing to rebuild it during a query run.\n' >&2
  exit 1
fi

ensure_main_services
archive_incompatible_evoneed_checkpoints

AGENTIC_MAX_RETRIEVAL_ROUNDS=1 EVONEED_RETRIEVAL_BUDGET=60 \
EVONEED_MIN_ROUTE_BUDGET=5 uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop_rag --dataset-dir "${SPLIT_DIR}" \
  --methods "${INTERNAL_METHODS}" --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir "${INTERNAL_RUNTIME}" --output-dir "${INTERNAL_OUTPUT}" --fail-fast

for method in naive hybrid light_entity ours ours_evolink ours_evoneed_equal ours_evoneed_dynamic; do
  validate_summary "${INTERNAL_OUTPUT}/${method}_summary.json"
done

"${HIPPORAG_PYTHON}" scripts/run_hipporag_baseline.py \
  --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
  --runtime-dir "${HIPPORAG_RUNTIME}" --output "${HIPPORAG_OUTPUT}/raw.jsonl" \
  --stage retrieve --top-k 20 --llm-concurrency 4 --allow-query-set-change
validate_raw "${HIPPORAG_OUTPUT}/raw.jsonl"
uv run --extra evaluation python scripts/score_hipporag_results.py \
  --dataset-dir "${SPLIT_DIR}" --dataset-name multihop_rag \
  --raw-results "${HIPPORAG_OUTPUT}/raw.jsonl" --output-dir "${HIPPORAG_OUTPUT}" \
  --cutoffs 5,10,20
validate_summary "${HIPPORAG_OUTPUT}/hipporag_summary.json"

EVENTRAG_NEO4J_CONTAINER_NAME=rag-graph-demo-eventrag-full609 \
EVENTRAG_NEO4J_HTTP_PORT=7476 EVENTRAG_NEO4J_BOLT_PORT=7689 \
EVENTRAG_NEO4J_DATA_DIR=./data/eventrag/neo4j_full609/data \
EVENTRAG_NEO4J_LOGS_DIR=./data/eventrag/neo4j_full609/logs \
docker-compose -f docker-compose.eventrag.yml up -d eventrag-neo4j
uv run --extra evaluation --extra eventrag python scripts/run_eventrag_baseline.py \
  --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
  --runtime-dir "${EVENTRAG_RUNTIME}" --output "${EVENTRAG_OUTPUT}/raw.jsonl" \
  --neo4j-uri bolt://127.0.0.1:7689 --stage retrieve --top-k 20 \
  --allow-query-set-change
validate_raw "${EVENTRAG_OUTPUT}/raw.jsonl"
uv run --extra evaluation python scripts/score_eventrag_results.py \
  --dataset-dir "${SPLIT_DIR}" --dataset-name multihop_rag \
  --raw-results "${EVENTRAG_OUTPUT}/raw.jsonl" --output-dir "${EVENTRAG_OUTPUT}" \
  --cutoffs 5,10,20
validate_summary "${EVENTRAG_OUTPUT}/eventrag_hybrid_budgeted_summary.json"

"${E2RAG_PYTHON}" scripts/run_e2rag_baseline.py \
  --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
  --runtime-dir "${E2RAG_RUNTIME}" --output "${E2RAG_OUTPUT}/raw.jsonl" \
  --stage retrieve --variant vanilla --top-k 20 --allow-query-set-change
validate_raw "${E2RAG_OUTPUT}/raw.jsonl"
uv run --extra evaluation python scripts/score_e2rag_results.py \
  --dataset-dir "${SPLIT_DIR}" --dataset-name multihop_rag \
  --raw-results "${E2RAG_OUTPUT}/raw.jsonl" --output-dir "${E2RAG_OUTPUT}" \
  --cutoffs 5,10,20
validate_summary "${E2RAG_OUTPUT}/e2rag_original_summary.json"

if [[ ! -f "${GRAPHRAG_RUNTIME}/settings.yaml" ]]; then
  uv run python scripts/run_graphrag_baseline.py \
    --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
    --runtime-dir "${GRAPHRAG_RUNTIME}" --output "${GRAPHRAG_OUTPUT}/raw.jsonl" \
    --stage prepare --top-k 20 --concurrency 4
fi
if [[ ! -f "${GRAPHRAG_RUNTIME}/index_complete.json" ]]; then
  uv run python scripts/run_graphrag_baseline.py \
    --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
    --runtime-dir "${GRAPHRAG_RUNTIME}" --output "${GRAPHRAG_OUTPUT}/raw.jsonl" \
    --stage index --top-k 20 --concurrency 4
fi
uv run python scripts/run_graphrag_baseline.py \
  --corpus "${EXTERNAL_CORPUS}" --queries "${EXTERNAL_QUERIES}" \
  --runtime-dir "${GRAPHRAG_RUNTIME}" --output "${GRAPHRAG_OUTPUT}/raw.jsonl" \
  --stage retrieve --top-k 20 --concurrency 4
validate_raw "${GRAPHRAG_OUTPUT}/raw.jsonl"
uv run --extra evaluation python scripts/score_external_baseline.py \
  --dataset-dir "${SPLIT_DIR}" --dataset-name multihop_rag \
  --raw-results "${GRAPHRAG_OUTPUT}/raw.jsonl" --output-dir "${GRAPHRAG_OUTPUT}" \
  --method graphrag_local --cutoffs 5,10,20
validate_summary "${GRAPHRAG_OUTPUT}/graphrag_local_summary.json"

show_status
printf 'Gold-500 comparison completed at %s\n' "$(date -Iseconds)"
