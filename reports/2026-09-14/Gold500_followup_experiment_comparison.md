# 2026-09-14 Gold500 后续实验对比

统一数据集：`multihop_rag_gold500_seed20260908`（500 题）；固定索引和递归切分配置不变。所有本轮新增结果均位于 `2026-09-14...` 目录。

指标均为检索阶段 @20；延迟为平均秒数。`—` 表示对应实验尚未完成。

## Lifecycle mechanism ablation

结果文件：`data/exports/evaluation/2026-09-14_lifecycle_ablation_gold500/summary.json`

| 方法 | 完成数 | 错误数 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | 500 | 0 | 0.838 | 0.612 | 0.629 | 0.690 | 2.13 |
| Association-only | 500 | 0 | 0.869 | 0.688 | 0.641 | 0.688 | 0.95 |
| Ours+EvoLink | 500 | 0 | 0.871 | 0.692 | 0.641 | 0.687 | 0.91 |

## EvoLink (5,5) confirmation

结果文件：`data/exports/evaluation/2026-09-14_evolink_param_confirm_gold500/summary.json`

| 方法 | 完成数 | 错误数 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours+EvoLink | 500 | 0 | 0.873 | 0.696 | 0.642 | 0.687 | 0.91 |
| OnePlan-Shared | 500 | 0 | 0.903 | 0.772 | 0.639 | 0.646 | 0.76 |

## Equal-total / strict paired budget

结果文件：`data/exports/evaluation/2026-09-14_budget_matched_gold500/summary.json`

| 方法 | 完成数 | 错误数 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | 500 | 0 | 0.838 | 0.612 | 0.630 | 0.690 | 0.77 |
| Ours+EvoLink | 500 | 0 | 0.872 | 0.694 | 0.641 | 0.687 | 0.89 |
| QueryRewrite-Shared (strict) | 500 | 0 | 0.880 | 0.712 | 0.647 | 0.693 | 0.89 |
| OnePlan-Shared (strict) | 500 | 0 | 0.904 | 0.778 | 0.637 | 0.640 | 0.72 |

## 冻结旧结果（对照）

| 结果 | 方法 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟(s) |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-08 Ours baseline | Ours | 0.836 | 0.610 | 0.633 | 0.696 | 4.83 |
| 2026-09-08 Ours+EvoLink | Ours+EvoLink | 0.879 | 0.706 | 0.646 | 0.694 | 1.37 |
| 2026-09-11 QueryRewrite-Shared | QueryRewrite-Shared | 0.890 | 0.738 | 0.650 | 0.694 | 0.89 |
| 2026-09-11 OnePlan-Shared | OnePlan-Shared | 0.900 | 0.766 | 0.637 | 0.645 | 0.92 |

## 解读口径

- `Ours → Association-only → Ours+EvoLink` 固定首轮三路检索，只关闭/打开 EventEffect 生命周期桥，直接观察生命周期关系的增量。
- 严格配对实验中，固定路由方法是 3 路 × 20 = 60 个首轮候选；OnePlan 使用同一个查询级 60 候选总预算（在最多 3 个 Need 间分配），二者统一使用 EvoLink 扩展上限 20。
- 本报告不把 Hybrid 当作主方法结论；它属于额外增强版本。
