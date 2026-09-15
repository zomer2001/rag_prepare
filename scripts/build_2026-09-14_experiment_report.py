#!/usr/bin/env python3
"""Collect the 2026-09-14 Gold500 follow-up runs beside frozen baselines."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/exports/evaluation")
REPORT = Path("reports/2026-09-14/Gold500_followup_experiment_comparison.md")

METHOD_LABELS = {
    "ours": "Ours",
    "ours_association_only": "Association-only",
    "ours_evolink": "Ours+EvoLink",
    "ours_d1_queryrewrite_sharedlink_total": "QueryRewrite-Shared (strict)",
    "ours_evoneed_deep_oneplan_shared_total_norerank": "OnePlan-Shared (strict)",
    "ours_d1_queryrewrite_sharedlink": "QueryRewrite-Shared",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank": "OnePlan-Shared",
}

EXPERIMENTS = [
    (
        "Lifecycle mechanism ablation",
        ROOT / "2026-09-14_lifecycle_ablation_gold500" / "summary.json",
        ["ours", "ours_association_only", "ours_evolink"],
    ),
    (
        "EvoLink (5,5) confirmation",
        ROOT / "2026-09-14_evolink_param_confirm_gold500" / "summary.json",
        ["ours_evolink", "ours_evoneed_deep_oneplan_shared_equal_total_norerank"],
    ),
    (
        "Equal-total / strict paired budget",
        ROOT / "2026-09-14_budget_matched_gold500" / "summary.json",
        [
            "ours",
            "ours_evolink",
            "ours_d1_queryrewrite_sharedlink_total",
            "ours_evoneed_deep_oneplan_shared_total_norerank",
        ],
    ),
]

FROZEN_BASELINES = [
    ("2026-09-08 Ours baseline", ROOT / "2026-09-08_query_improvements_gold500" / "summary.json", "ours"),
    ("2026-09-08 Ours+EvoLink", ROOT / "2026-09-08_query_improvements_gold500" / "summary.json", "ours_evolink"),
    ("2026-09-11 QueryRewrite-Shared", ROOT / "2026-09-11_shared_link_ablation_gold500" / "summary.json", "ours_d1_queryrewrite_sharedlink"),
    ("2026-09-11 OnePlan-Shared", ROOT / "2026-09-11_shared_link_ablation_gold500" / "summary.json", "ours_evoneed_deep_oneplan_shared_equal_total_norerank"),
]


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summaries", payload)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def row(summary: dict, method: str) -> str:
    item = summary.get(method, {})
    metrics = item.get("metrics", {})
    latency = metrics.get("latency_mean_ms")
    return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
        METHOD_LABELS.get(method, method),
        item.get("evaluated_query_count", 0),
        item.get("error_count", 0),
        fmt(metrics.get("evidence_recall@20")),
        fmt(metrics.get("complete_recall@20")),
        fmt(metrics.get("ndcg@20")),
        fmt(metrics.get("mrr@20")),
        fmt(latency / 1000 if isinstance(latency, (float, int)) else None, 2),
    )


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2026-09-14 Gold500 后续实验对比",
        "",
        "统一数据集：`multihop_rag_gold500_seed20260908`（500 题）；固定索引和递归切分配置不变。所有本轮新增结果均位于 `2026-09-14...` 目录。",
        "",
        "指标均为检索阶段 @20；延迟为平均秒数。`—` 表示对应实验尚未完成。",
        "",
    ]
    for title, path, methods in EXPERIMENTS:
        summary = load(path)
        lines.extend(
            [
                f"## {title}",
                "",
                f"结果文件：`{path}`",
                "",
                "| 方法 | 完成数 | 错误数 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        lines.extend(row(summary, method) for method in methods)
        lines.append("")

    lines.extend(
        [
            "## 冻结旧结果（对照）",
            "",
            "| 结果 | 方法 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, path, method in FROZEN_BASELINES:
        summary = load(path)
        item = summary.get(method, {})
        metrics = item.get("metrics", {})
        latency = metrics.get("latency_mean_ms")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                METHOD_LABELS.get(method, method),
                fmt(metrics.get("evidence_recall@20")),
                fmt(metrics.get("complete_recall@20")),
                fmt(metrics.get("ndcg@20")),
                fmt(metrics.get("mrr@20")),
                fmt(latency / 1000 if isinstance(latency, (float, int)) else None, 2),
            )
        )
    lines.extend(
        [
            "",
            "## 解读口径",
            "",
            "- `Ours → Association-only → Ours+EvoLink` 固定首轮三路检索，只关闭/打开 EventEffect 生命周期桥，直接观察生命周期关系的增量。",
            "- 严格配对实验中，固定路由方法是 3 路 × 20 = 60 个首轮候选；OnePlan 使用同一个查询级 60 候选总预算（在最多 3 个 Need 间分配），二者统一使用 EvoLink 扩展上限 20。",
            "- 本报告不把 Hybrid 当作主方法结论；它属于额外增强版本。",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
