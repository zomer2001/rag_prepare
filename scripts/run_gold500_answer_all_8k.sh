#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

STAGE="all"
LIMIT="500"
for argument in "$@"; do
  case "${argument}" in
    --prepare) STAGE="prepare" ;;
    --generate) STAGE="generate" ;;
    --score) STAGE="score" ;;
    --all) STAGE="all" ;;
    --limit=*) LIMIT="${argument#*=}" ;;
    *) printf 'Unknown option: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done

DATASET_DIR="data/benchmarks/multihop_rag_gold500_seed20260908"
EXTERNAL_DIR="data/benchmarks/external_multihop_full609_gold500_seed20260908"
EXTERNAL_CORPUS="${EXTERNAL_DIR}/multihop_full609_gold500_seed20260908_recursive700_corpus.json"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
LIGHT_INDEX="${RUNTIME_DIR}/lightrag_entity/eval_entity_multihop_rag_d012350429"
LIGHT_GRAPH="${LIGHT_INDEX}/graph_chunk_entity_relation.graphml"
LIGHT_DOCS="${LIGHT_INDEX}/kv_store_full_docs.json"
LIGHT_CHUNKS="${LIGHT_INDEX}/kv_store_text_chunks.json"
EVENT_STORE="${RUNTIME_DIR}/events/event_store.json"
EVENTRAG_RUNTIME="data/eventrag/evaluation/2026-09-04_full609_q100_isolated_v1"
EVENTRAG_SNAPSHOT="${EVENTRAG_RUNTIME}/native_graph_snapshot.json"
EVENTRAG_DOCS="${EVENTRAG_RUNTIME}/state/kv_store_full_docs.json"
E2RAG_RUNTIME="data/e2rag/evaluation/2026-09-04_multihop_full609_q100"
OUTPUT_DIR="data/exports/answer_evaluation/2026-09-10_gold500_answer_all_8k_q${LIMIT}"
PYTHON=".venv/bin/python"
# API key is injected in the caller's temporary environment; it is not stored here.
# Previous default: https://api.deepseek.com
# This answer run uses the user-provided compatible endpoint.
ANSWER_BASE_URL="https://www.tokenlinks.cn"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Missing project Python: %s\n' "${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${EVENTRAG_SNAPSHOT}" ]]; then
  printf 'Missing EventRAG snapshot: %s\n' "${EVENTRAG_SNAPSHOT}" >&2
  exit 1
fi

"${PYTHON}" scripts/evaluate_answers.py \
  --stage "${STAGE}" \
  --dataset-dir "${DATASET_DIR}" \
  --external-corpus "${EXTERNAL_CORPUS}" \
  --light-chunks "${LIGHT_CHUNKS}" \
  --event-store "${EVENT_STORE}" \
  --input "naive:raw=data/exports/evaluation/2026-09-08_all_internal_gold500/naive.jsonl" \
  --input "hybrid:raw=data/exports/evaluation/2026-09-08_all_internal_gold500/hybrid.jsonl" \
  --input "light_entity:raw=data/exports/evaluation/2026-09-08_all_internal_gold500/light_entity.jsonl" \
  --input "light_entity:native=data/exports/evaluation/2026-09-08_all_internal_gold500/light_entity.jsonl" \
  --input "ours:raw=data/exports/evaluation/2026-09-08_query_improvements_gold500/ours.jsonl" \
  --input "ours:native=data/exports/evaluation/2026-09-08_query_improvements_gold500/ours.jsonl" \
  --input "evolink:raw=data/exports/evaluation/2026-09-08_query_improvements_gold500/ours_evolink.jsonl" \
  --input "evolink:native=data/exports/evaluation/2026-09-08_query_improvements_gold500/ours_evolink.jsonl" \
  --input "hipporag:raw=data/exports/evaluation/2026-09-08_hipporag2_gold500/hipporag.jsonl" \
  --input "eventrag:raw=data/exports/evaluation/2026-09-08_eventrag_budgeted_gold500/eventrag_hybrid_budgeted.jsonl" \
  --input "eventrag:native=data/exports/evaluation/2026-09-08_eventrag_budgeted_gold500/eventrag_hybrid_budgeted.jsonl" \
  --input "e2rag:raw=data/exports/evaluation/2026-09-08_e2rag_gold500/e2rag_original.jsonl" \
  --input "e2rag:native=data/exports/evaluation/2026-09-08_e2rag_gold500/e2rag_original.jsonl" \
  --input "d1_norerank:raw=data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/ours_evoneed_deep_oneplan_d1_norerank.jsonl" \
  --input "d1_norerank:native=data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/ours_evoneed_deep_oneplan_d1_norerank.jsonl" \
  --native-graph "light_entity:LightRAG entity graph=${LIGHT_GRAPH}" \
  --native-full-docs "light_entity:LightRAG entity graph=${LIGHT_DOCS}" \
  --native-graph "ours:Entity view=${LIGHT_GRAPH}" \
  --native-full-docs "ours:Entity view=${LIGHT_DOCS}" \
  --native-event-store "ours:Event view=${EVENT_STORE}" \
  --native-snapshot "eventrag:EventRAG entity-event graph=${EVENTRAG_SNAPSHOT}" \
  --native-full-docs "eventrag:EventRAG entity-event graph=${EVENTRAG_DOCS}" \
  --native-graph "e2rag:E2RAG entity graph=${E2RAG_RUNTIME}/entity_kg/graph_chunk_entity_relation.graphml" \
  --native-full-docs "e2rag:E2RAG entity graph=${E2RAG_RUNTIME}/entity_kg/kv_store_full_docs.json" \
  --native-graph "e2rag:E2RAG event graph=${E2RAG_RUNTIME}/event_kg/graph_chunk_entity_relation.graphml" \
  --native-full-docs "e2rag:E2RAG event graph=${E2RAG_RUNTIME}/event_kg/kv_store_full_docs.json" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "${ANSWER_BASE_URL}" \
  --top-k 20 \
  --max-context-tokens 8000 \
  --max-evidence-tokens 1200 \
  --structured-token-share 0.25 \
  --limit "${LIMIT}" \
  --concurrency 4

printf 'Answer experiment output: %s\n' "${OUTPUT_DIR}"
