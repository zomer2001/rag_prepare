#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

STAGE="prepare"
LIMIT="100"
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
LIGHT_CHUNKS="${RUNTIME_DIR}/lightrag_entity/eval_entity_multihop_rag_d012350429/kv_store_text_chunks.json"
EVENT_STORE="${RUNTIME_DIR}/events/event_store.json"
OUTPUT_DIR="data/exports/answer_evaluation/2026-09-10_gold500_pilot_q${LIMIT}"
PYTHON=".venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Missing project Python: %s\n' "${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" scripts/evaluate_answers.py \
  --stage "${STAGE}" \
  --dataset-dir "${DATASET_DIR}" \
  --external-corpus "${EXTERNAL_CORPUS}" \
  --light-chunks "${LIGHT_CHUNKS}" \
  --event-store "${EVENT_STORE}" \
  --input "hybrid:raw=data/exports/evaluation/2026-09-08_all_internal_gold500/hybrid.jsonl" \
  --input "evolink:raw=data/exports/evaluation/2026-09-08_query_improvements_gold500/ours_evolink.jsonl" \
  --input "d1_norerank:raw=data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/ours_evoneed_deep_oneplan_d1_norerank.jsonl" \
  --input "d1_norerank:graph=data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/ours_evoneed_deep_oneplan_d1_norerank.jsonl" \
  --input "d1_norerank:evolution=data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/ours_evoneed_deep_oneplan_d1_norerank.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k 20 \
  --max-context-tokens 16000 \
  --max-evidence-tokens 1200 \
  --structured-token-share 0.25 \
  --limit "${LIMIT}" \
  --concurrency 4
