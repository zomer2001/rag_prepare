from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/2026-09-11/Gold500检索阶段总表.md"

SOURCES: list[dict[str, str]] = [
    {
        "label": "Naive",
        "method": "naive",
        "path": "data/exports/evaluation/2026-09-08_all_internal_gold500",
        "role": "基线",
        "description": "单路 Dense 原问题检索",
    },
    {
        "label": "Hybrid",
        "method": "hybrid",
        "path": "data/exports/evaluation/2026-09-08_all_internal_gold500",
        "role": "强基线",
        "description": "Dense 与 BM25 融合",
    },
    {
        "label": "LightRAG-Entity",
        "method": "light_entity",
        "path": "data/exports/evaluation/2026-09-08_all_internal_gold500",
        "role": "图基线",
        "description": "向量与实体图检索",
    },
    {
        "label": "Ours",
        "method": "ours",
        "path": "data/exports/evaluation/2026-09-08_query_improvements_gold500",
        "role": "内部主线",
        "description": "向量、关系图、事件图三路直接融合",
    },
    {
        "label": "Ours+EvoLink",
        "method": "ours_evolink",
        "path": "data/exports/evaluation/2026-09-08_query_improvements_gold500",
        "role": "内部主线",
        "description": "三路结果后的共享跨视图扩展",
    },
    {
        "label": "EvoNeed-Equal",
        "method": "ours_evoneed_equal",
        "path": "data/exports/evaluation/2026-09-08_query_improvements_gold500",
        "role": "预算附录",
        "description": "证据需求规划，平均分配总预算",
    },
    {
        "label": "EvoNeed-Dynamic",
        "method": "ours_evoneed_dynamic",
        "path": "data/exports/evaluation/2026-09-08_query_improvements_gold500",
        "role": "预算附录",
        "description": "证据需求规划，动态分配总预算",
    },
    {
        "label": "EvoNeed-Planned",
        "method": "ours_evoneed_planned",
        "path": "data/exports/evaluation/2026-09-08_evoneed_planned_route20_gold500",
        "role": "旧版附录",
        "description": "最多三路、每路最多 20 个候选",
    },
    {
        "label": "D1-Direct-NoPlan",
        "method": "ours_d1_direct_noplan",
        "path": "data/exports/evaluation/2026-09-11_d1_ablation_gold500",
        "role": "核心消融",
        "description": "固定三路、每路独立 EvoLink、无 Planner",
    },
    {
        "label": "D1-QueryRewrite",
        "method": "ours_d1_queryrewrite",
        "path": "data/exports/evaluation/2026-09-11_d1_ablation_gold500",
        "role": "核心消融",
        "description": "现有 QueryPlanner 改写 query，不生成证据需求",
    },
    {
        "label": "D1-QueryRewrite-SharedLink",
        "method": "ours_d1_queryrewrite_sharedlink",
        "path": "data/exports/evaluation/2026-09-11_shared_link_ablation_gold500",
        "role": "核心消融",
        "description": "普通 QueryPlanner 路线查询，三路候选后共享一次 EvoLink",
    },
    {
        "label": "Deep-OnePlan-D1-NoRerank",
        "method": "ours_evoneed_deep_oneplan_d1_norerank",
        "path": "data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500",
        "role": "最终方法",
        "description": "一次证据需求规划、每个需求独立扩展、RRF、无 LLM rerank",
    },
    {
        "label": "OnePlan-Shared-EqualTotal",
        "method": "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
        "path": "data/exports/evaluation/2026-09-11_shared_link_ablation_gold500",
        "role": "核心消融",
        "description": "一次证据需求规划、所有需求共享 EvoLink，扩展预算等于需求预算总和",
    },
    {
        "label": "OnePlan-NoLink",
        "method": "ours_evoneed_deep_oneplan_nolink_norerank",
        "path": "data/exports/evaluation/2026-09-11_oneplan_nolink_gold500",
        "role": "主线消融",
        "description": "一次证据需求规划，不做 EvoLink，使用相同 Need-level RRF",
    },
    {
        "label": "EvoNeed-Deep",
        "method": "ours_evoneed_deep",
        "path": "data/exports/evaluation/2026-09-09_evoneed_deep_gold500",
        "role": "历史高预算",
        "description": "更深候选池、独立扩展、含 LLM rerank；不纳入主线公平比较",
    },
    {
        "label": "HippoRAG2",
        "method": "hipporag",
        "path": "data/exports/evaluation/2026-09-08_hipporag2_gold500",
        "role": "外部方法",
        "description": "HippoRAG 2 原生图检索适配",
    },
    {
        "label": "EventRAG",
        "method": "eventrag_hybrid_budgeted",
        "path": "data/exports/evaluation/2026-09-08_eventrag_budgeted_gold500",
        "role": "外部方法",
        "description": "事件图检索适配版本",
    },
    {
        "label": "E²RAG",
        "method": "e2rag_original",
        "path": "data/exports/evaluation/2026-09-08_e2rag_gold500",
        "role": "外部方法",
        "description": "实体—事件双图检索适配版本",
    },
]


def load_summary(item: dict[str, str]) -> tuple[dict[str, Any] | None, Path]:
    summary_path = ROOT / item["path"] / "summary.json"
    if not summary_path.is_file():
        summary_path = ROOT / item["path"] / f"{item['method']}_summary.json"
    if not summary_path.is_file():
        return None, summary_path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "summaries" in payload:
        return payload.get("summaries", {}).get(item["method"]), summary_path
    return payload, summary_path


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "待运行"
    count = summary.get("evaluated_query_count", summary.get("query_count", 0))
    errors = summary.get("error_count", 0)
    if count == 500 and errors == 0:
        return "已完成"
    return f"不完整（{count}/500，错误 {errors}）"


def overall_row(item: dict[str, str], summary: dict[str, Any] | None) -> str:
    metrics = (summary or {}).get("metrics", {})
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        item["label"],
        item["role"],
        status(summary),
        fmt(metrics.get("evidence_recall@5")),
        fmt(metrics.get("evidence_recall@10")),
        fmt(metrics.get("evidence_recall@20")),
        fmt(metrics.get("complete_recall@20")),
        fmt(metrics.get("ndcg@20")),
        fmt(metrics.get("mrr@20")),
    )


def type_rows(item: dict[str, str], summary: dict[str, Any] | None) -> list[str]:
    rows: list[str] = []
    for key, label in (
        ("comparison_query", "Comparison"),
        ("inference_query", "Inference"),
        ("temporal_query", "Temporal"),
    ):
        metrics = ((summary or {}).get("by_question_type", {}).get(key) or {}).get(
            "metrics", {}
        )
        rows.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                item["label"],
                label,
                fmt(metrics.get("evidence_recall@20")),
                fmt(metrics.get("complete_recall@20")),
                fmt(metrics.get("ndcg@20")),
                fmt(metrics.get("mrr@20")),
            )
        )
    return rows


def build_report() -> str:
    loaded: list[tuple[dict[str, str], dict[str, Any] | None, Path]] = []
    for item in SOURCES:
        summary, path = load_summary(item)
        loaded.append((item, summary, path))

    lines = [
        "# Gold500 检索阶段实验结果总表",
        "",
        "日期：2026-09-11",
        "范围：2026 年 9 月 MultiHopRAG Gold500，500 题；不含 n30、2556 题和 GraphRAG 结果。",
        "固定索引：609 documents、2512 chunks；最终返回 Top-20。",
        "",
        "## 1. 总体结果",
        "",
        "主线比较应优先使用状态为“已完成”的方法，并保持相同索引、Gold500、Top-20 和最终融合规则。EvoNeed-Deep 含 LLM rerank，只作为历史高预算参考。",
        "",
        "| 方法 | 归类 | 状态 | Evidence R@5 | Evidence R@10 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item, summary, _ in loaded:
        lines.append(overall_row(item, summary))

    lines.extend(
        [
            "",
            "## 2. 按问题类型",
            "",
            "每个单元格独立列出 Evidence Recall@20、Complete Recall@20、nDCG@20 和 MRR@20。",
            "",
            "| 方法 | 问题类型 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item, summary, _ in loaded:
        lines.extend(type_rows(item, summary))

    lines.extend(
        [
            "",
            "## 3. 方法与消融关系",
            "",
            "| 方法 | 实验作用 |",
            "|---|---|",
        ]
    )
    for item in SOURCES:
        lines.append(f"| {item['label']} | {item['description']} |")

    lines.extend(
        [
            "",
            "## 4. 当前状态与下一步",
            "",
            "- D1-Direct-NoPlan、D1-QueryRewrite、D1-QueryRewrite-SharedLink、Deep-OnePlan-D1-NoRerank 和 OnePlan-Shared-EqualTotal 均已完成 500/500。",
            "- 共享/独立 EvoLink 的核心检索对照已经齐全，下一步可以直接分析 Evidence Recall、Complete Recall、nDCG、MRR 和延迟。",
            "- 若要回答问答问题：对所有主方法补 Gold500 raw 问答；只对 Ours+EvoLink 和最终 D1 增加 raw+entity_relation、raw+evolution。",
            "- 不再运行 D1 的 LLM rerank 版本；带 rerank 的 EvoNeed-Deep 不进入主线公平比较。",
            "",
            "## 5. 数据来源",
        ]
    )
    for item, _, path in loaded:
        lines.append(f"- {item['label']}：{path.relative_to(ROOT)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
