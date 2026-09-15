#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RUN_DATE="2026-09-11"
SEED="20260908"
SPLIT_DIR="data/benchmarks/multihop_rag_gold500_seed${SEED}"
RUNTIME_DIR="data/lightrag/evaluation/multihop_full609_elc_v1"
OUTPUT_ROOT="data/exports/evaluation/${RUN_DATE}_evolink_param_probe_gold100"
LOG_DIR="reports/${RUN_DATE}/logs"
RUN_LOG="${LOG_DIR}/evolink_param_probe_gold100.log"
PYTHON=".venv/bin/python"
METHODS="ours_evolink,ours_evoneed_deep_oneplan_shared_equal_total_norerank"

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
print("Validated fixed September indexes: 2512 chunks and 609 documents")
PY

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

if docker compose version >/dev/null 2>&1; then
  docker compose up -d qdrant neo4j
else
  docker-compose up -d qdrant neo4j
fi

run_config() {
  local config_name="$1"
  local seed_limit="$2"
  local event_seed_limit="$3"
  local output_dir="${OUTPUT_ROOT}/${config_name}"

  printf '\n=== %s: seed_limit=%s, max_event_seeds=%s ===\n' \
    "${config_name}" "${seed_limit}" "${event_seed_limit}"

  EVOLINK_SEED_LIMIT_PER_NAME="${seed_limit}" \
  EVOLINK_MAX_EVENT_SEEDS="${event_seed_limit}" \
  DEEPSEEK_BASE_URL="https://api.deepseek.com" \
  "${PYTHON}" scripts/evaluate_retrieval.py \
    --dataset multihop_rag \
    --dataset-dir "${SPLIT_DIR}" \
    --methods "${METHODS}" \
    --cutoffs 5,10,20 \
    --chunking-strategy recursive \
    --chunk-size 700 \
    --chunk-overlap 100 \
    --runtime-dir "${RUNTIME_DIR}" \
    --output-dir "${output_dir}" \
    --limit 100 \
    --fail-fast
}

run_config "seed_wide_5_5" 5 5
run_config "event_wide_3_8" 3 8
run_config "seed_event_wide_5_8" 5 8

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
method_labels = {
    "ours_evolink": "Ours+EvoLink",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank": "OnePlan-Shared",
}
configs = {
    "seed_wide_5_5": {"seed_limit_per_name": 5, "max_event_seeds": 5},
    "event_wide_3_8": {"seed_limit_per_name": 3, "max_event_seeds": 8},
    "seed_event_wide_5_8": {"seed_limit_per_name": 5, "max_event_seeds": 8},
}

def load_summary(path: Path) -> dict:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return payload.get("summaries", payload)

def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)

result = {"query_count": 100, "configs": {}}
for config_name, parameters in configs.items():
    summaries = load_summary(root / config_name)
    result["configs"][config_name] = {"parameters": parameters, "methods": summaries}

(root / "summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# EvoLink 参数 100 题试跑汇总",
    "",
    "范围：2026 年 9 月 Gold500 前 100 题；固定索引不变；没有 n30、2556 题或 GraphRAG。",
    "",
    "## 参数配置",
    "",
    "| 配置 | seed_limit_per_name | max_event_seeds |",
    "|---|---:|---:|",
]
for config_name, parameters in configs.items():
    lines.append(
        f"| {config_name} | {parameters['seed_limit_per_name']} | {parameters['max_event_seeds']} |"
    )

lines.extend([
    "",
    "## 总体指标",
    "",
    "| 配置 | 方法 | 完成数 | 错误数 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) | P95(s) |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
])
for config_name in configs:
    summaries = result["configs"][config_name]["methods"]
    for method, label in method_labels.items():
        summary = summaries.get(method, {})
        metrics = summary.get("metrics", {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                config_name,
                label,
                summary.get("evaluated_query_count", 0),
                summary.get("error_count", 0),
                fmt(metrics.get("evidence_recall@20")),
                fmt(metrics.get("complete_recall@20")),
                fmt(metrics.get("ndcg@20")),
                fmt(metrics.get("mrr@20")),
                fmt((metrics.get("latency_mean_ms") or 0) / 1000),
                fmt((metrics.get("latency_p95_ms") or 0) / 1000),
            )
        )

lines.extend([
    "",
    "## 按问题类型",
    "",
    "| 配置 | 方法 | 类型 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |",
    "|---|---|---|---:|---:|---:|---:|",
])
type_labels = {
    "comparison_query": "Comparison",
    "inference_query": "Inference",
    "temporal_query": "Temporal",
}
for config_name in configs:
    summaries = result["configs"][config_name]["methods"]
    for method, label in method_labels.items():
        summary = summaries.get(method, {})
        for type_key, type_label in type_labels.items():
            metrics = (summary.get("by_question_type", {}).get(type_key) or {}).get("metrics", {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    config_name,
                    label,
                    type_label,
                    fmt(metrics.get("evidence_recall@20")),
                    fmt(metrics.get("complete_recall@20")),
                    fmt(metrics.get("ndcg@20")),
                    fmt(metrics.get("mrr@20")),
                )
            )

lines.extend([
    "",
    "## 结果位置",
    "",
    "每个配置目录都保留自己的 summary.json、逐题 JSONL 和 run_manifest.json。",
    "",
])
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {root / 'summary.md'}")
print(f"Wrote {root / 'summary.json'}")
PY

printf '\nCompleted. Results: %s\nSummary: %s/summary.md\nLog: %s\n' \
  "${OUTPUT_ROOT}" "${OUTPUT_ROOT}" "${RUN_LOG}"
