# Gold500 检索阶段实验结果总表

日期：2026-09-11
范围：2026 年 9 月 MultiHopRAG Gold500，500 题；不含 n30、2556 题和 GraphRAG 结果。
固定索引：609 documents、2512 chunks；最终返回 Top-20。

## 1. 总体结果

主线比较应优先使用状态为“已完成”的方法，并保持相同索引、Gold500、Top-20 和最终融合规则。EvoNeed-Deep 含 LLM rerank，只作为历史高预算参考。

| 方法 | 归类 | 状态 | Evidence R@5 | Evidence R@10 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Naive | 基线 | 已完成 | 0.454 | 0.590 | 0.740 | 0.474 | 0.555 | 0.679 |
| Hybrid | 强基线 | 已完成 | 0.591 | 0.745 | 0.867 | 0.668 | 0.666 | 0.765 |
| LightRAG-Entity | 图基线 | 已完成 | 0.561 | 0.717 | 0.800 | 0.534 | 0.624 | 0.715 |
| Ours | 内部主线 | 已完成 | 0.588 | 0.725 | 0.836 | 0.610 | 0.633 | 0.696 |
| Ours+EvoLink | 内部主线 | 已完成 | 0.588 | 0.730 | 0.879 | 0.706 | 0.646 | 0.694 |
| EvoNeed-Equal | 预算附录 | 已完成 | 0.563 | 0.730 | 0.847 | 0.634 | 0.618 | 0.662 |
| EvoNeed-Dynamic | 预算附录 | 已完成 | 0.565 | 0.729 | 0.845 | 0.632 | 0.614 | 0.654 |
| EvoNeed-Planned | 旧版附录 | 已完成 | 0.558 | 0.721 | 0.839 | 0.636 | 0.609 | 0.655 |
| D1-Direct-NoPlan | 核心消融 | 已完成 | 0.575 | 0.726 | 0.859 | 0.664 | 0.630 | 0.666 |
| D1-QueryRewrite | 核心消融 | 已完成 | 0.542 | 0.717 | 0.876 | 0.694 | 0.569 | 0.555 |
| D1-QueryRewrite-SharedLink | 核心消融 | 已完成 | 0.581 | 0.740 | 0.890 | 0.738 | 0.650 | 0.694 |
| Deep-OnePlan-D1-NoRerank | 最终方法 | 已完成 | 0.571 | 0.749 | 0.902 | 0.774 | 0.632 | 0.637 |
| OnePlan-Shared-EqualTotal | 核心消融 | 已完成 | 0.580 | 0.784 | 0.900 | 0.766 | 0.637 | 0.645 |
| OnePlan-NoLink | 主线消融 | 已完成 | 0.605 | 0.775 | 0.841 | 0.640 | 0.623 | 0.649 |
| EvoNeed-Deep | 历史高预算 | 已完成 | 0.665 | 0.799 | 0.928 | 0.814 | 0.762 | 0.842 |
| HippoRAG2 | 外部方法 | 已完成 | 0.472 | 0.625 | 0.776 | 0.530 | 0.558 | 0.639 |
| EventRAG | 外部方法 | 已完成 | 0.170 | 0.420 | 0.451 | 0.180 | 0.248 | 0.247 |
| E²RAG | 外部方法 | 已完成 | 0.363 | 0.528 | 0.613 | 0.316 | 0.408 | 0.443 |

## 2. 按问题类型

每个单元格独立列出 Evidence Recall@20、Complete Recall@20、nDCG@20 和 MRR@20。

| 方法 | 问题类型 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |
|---|---|---:|---:|---:|---:|
| Naive | Comparison | 0.754 | 0.553 | 0.598 | 0.722 |
| Naive | Inference | 0.694 | 0.365 | 0.497 | 0.651 |
| Naive | Temporal | 0.784 | 0.512 | 0.571 | 0.655 |
| Hybrid | Comparison | 0.891 | 0.779 | 0.712 | 0.794 |
| Hybrid | Inference | 0.820 | 0.497 | 0.617 | 0.758 |
| Hybrid | Temporal | 0.898 | 0.744 | 0.666 | 0.732 |
| LightRAG-Entity | Comparison | 0.796 | 0.595 | 0.640 | 0.724 |
| LightRAG-Entity | Inference | 0.782 | 0.425 | 0.586 | 0.700 |
| LightRAG-Entity | Temporal | 0.829 | 0.597 | 0.653 | 0.725 |
| Ours | Comparison | 0.832 | 0.668 | 0.645 | 0.702 |
| Ours | Inference | 0.800 | 0.442 | 0.592 | 0.691 |
| Ours | Temporal | 0.894 | 0.760 | 0.672 | 0.696 |
| Ours+EvoLink | Comparison | 0.876 | 0.753 | 0.658 | 0.698 |
| Ours+EvoLink | Inference | 0.855 | 0.586 | 0.610 | 0.689 |
| Ours+EvoLink | Temporal | 0.915 | 0.806 | 0.679 | 0.696 |
| EvoNeed-Equal | Comparison | 0.859 | 0.711 | 0.643 | 0.692 |
| EvoNeed-Equal | Inference | 0.799 | 0.464 | 0.557 | 0.619 |
| EvoNeed-Equal | Temporal | 0.895 | 0.760 | 0.667 | 0.676 |
| EvoNeed-Dynamic | Comparison | 0.854 | 0.700 | 0.641 | 0.689 |
| EvoNeed-Dynamic | Inference | 0.802 | 0.475 | 0.551 | 0.602 |
| EvoNeed-Dynamic | Temporal | 0.891 | 0.752 | 0.663 | 0.677 |
| EvoNeed-Planned | Comparison | 0.859 | 0.721 | 0.634 | 0.674 |
| EvoNeed-Planned | Inference | 0.778 | 0.453 | 0.545 | 0.620 |
| EvoNeed-Planned | Temporal | 0.895 | 0.767 | 0.662 | 0.674 |
| D1-Direct-NoPlan | Comparison | 0.846 | 0.695 | 0.641 | 0.674 |
| D1-Direct-NoPlan | Inference | 0.826 | 0.514 | 0.583 | 0.642 |
| D1-Direct-NoPlan | Temporal | 0.924 | 0.829 | 0.679 | 0.689 |
| D1-QueryRewrite | Comparison | 0.889 | 0.763 | 0.546 | 0.492 |
| D1-QueryRewrite | Inference | 0.833 | 0.547 | 0.574 | 0.614 |
| D1-QueryRewrite | Temporal | 0.915 | 0.798 | 0.597 | 0.565 |
| D1-QueryRewrite-SharedLink | Comparison | 0.895 | 0.784 | 0.664 | 0.701 |
| D1-QueryRewrite-SharedLink | Inference | 0.846 | 0.591 | 0.607 | 0.690 |
| D1-QueryRewrite-SharedLink | Temporal | 0.944 | 0.876 | 0.688 | 0.690 |
| Deep-OnePlan-D1-NoRerank | Comparison | 0.926 | 0.853 | 0.661 | 0.655 |
| Deep-OnePlan-D1-NoRerank | Inference | 0.851 | 0.624 | 0.582 | 0.616 |
| Deep-OnePlan-D1-NoRerank | Temporal | 0.937 | 0.868 | 0.658 | 0.638 |
| OnePlan-Shared-EqualTotal | Comparison | 0.937 | 0.868 | 0.677 | 0.672 |
| OnePlan-Shared-EqualTotal | Inference | 0.836 | 0.591 | 0.581 | 0.629 |
| OnePlan-Shared-EqualTotal | Temporal | 0.935 | 0.860 | 0.659 | 0.629 |
| OnePlan-NoLink | Comparison | 0.878 | 0.737 | 0.656 | 0.662 |
| OnePlan-NoLink | Inference | 0.798 | 0.508 | 0.578 | 0.643 |
| OnePlan-NoLink | Temporal | 0.845 | 0.682 | 0.638 | 0.639 |
| EvoNeed-Deep | Comparison | 0.956 | 0.911 | 0.796 | 0.853 |
| EvoNeed-Deep | Inference | 0.870 | 0.630 | 0.691 | 0.815 |
| EvoNeed-Deep | Temporal | 0.968 | 0.930 | 0.810 | 0.862 |
| HippoRAG2 | Comparison | 0.774 | 0.579 | 0.572 | 0.643 |
| HippoRAG2 | Inference | 0.759 | 0.431 | 0.538 | 0.656 |
| HippoRAG2 | Temporal | 0.802 | 0.597 | 0.564 | 0.609 |
| EventRAG | Comparison | 0.500 | 0.263 | 0.243 | 0.193 |
| EventRAG | Inference | 0.331 | 0.033 | 0.217 | 0.280 |
| EventRAG | Temporal | 0.545 | 0.264 | 0.299 | 0.279 |
| E²RAG | Comparison | 0.671 | 0.405 | 0.439 | 0.457 |
| E²RAG | Inference | 0.465 | 0.144 | 0.306 | 0.360 |
| E²RAG | Temporal | 0.735 | 0.426 | 0.507 | 0.537 |

## 3. 方法与消融关系

| 方法 | 实验作用 |
|---|---|
| Naive | 单路 Dense 原问题检索 |
| Hybrid | Dense 与 BM25 融合 |
| LightRAG-Entity | 向量与实体图检索 |
| Ours | 向量、关系图、事件图三路直接融合 |
| Ours+EvoLink | 三路结果后的共享跨视图扩展 |
| EvoNeed-Equal | 证据需求规划，平均分配总预算 |
| EvoNeed-Dynamic | 证据需求规划，动态分配总预算 |
| EvoNeed-Planned | 最多三路、每路最多 20 个候选 |
| D1-Direct-NoPlan | 固定三路、每路独立 EvoLink、无 Planner |
| D1-QueryRewrite | 现有 QueryPlanner 改写 query，不生成证据需求 |
| D1-QueryRewrite-SharedLink | 普通 QueryPlanner 路线查询，三路候选后共享一次 EvoLink |
| Deep-OnePlan-D1-NoRerank | 一次证据需求规划、每个需求独立扩展、RRF、无 LLM rerank |
| OnePlan-Shared-EqualTotal | 一次证据需求规划、所有需求共享 EvoLink，扩展预算等于需求预算总和 |
| OnePlan-NoLink | 一次证据需求规划，不做 EvoLink，使用相同 Need-level RRF |
| EvoNeed-Deep | 更深候选池、独立扩展、含 LLM rerank；不纳入主线公平比较 |
| HippoRAG2 | HippoRAG 2 原生图检索适配 |
| EventRAG | 事件图检索适配版本 |
| E²RAG | 实体—事件双图检索适配版本 |

## 4. 当前状态与下一步

- D1-Direct-NoPlan、D1-QueryRewrite、D1-QueryRewrite-SharedLink、Deep-OnePlan-D1-NoRerank 和 OnePlan-Shared-EqualTotal 均已完成 500/500。
- 共享/独立 EvoLink 的核心检索对照已经齐全，下一步可以直接分析 Evidence Recall、Complete Recall、nDCG、MRR 和延迟。
- 若要回答问答问题：对所有主方法补 Gold500 raw 问答；只对 Ours+EvoLink 和最终 D1 增加 raw+entity_relation、raw+evolution。
- 不再运行 D1 的 LLM rerank 版本；带 rerank 的 EvoNeed-Deep 不进入主线公平比较。

## 5. 数据来源
- Naive：data/exports/evaluation/2026-09-08_all_internal_gold500/summary.json
- Hybrid：data/exports/evaluation/2026-09-08_all_internal_gold500/summary.json
- LightRAG-Entity：data/exports/evaluation/2026-09-08_all_internal_gold500/summary.json
- Ours：data/exports/evaluation/2026-09-08_query_improvements_gold500/summary.json
- Ours+EvoLink：data/exports/evaluation/2026-09-08_query_improvements_gold500/summary.json
- EvoNeed-Equal：data/exports/evaluation/2026-09-08_query_improvements_gold500/summary.json
- EvoNeed-Dynamic：data/exports/evaluation/2026-09-08_query_improvements_gold500/summary.json
- EvoNeed-Planned：data/exports/evaluation/2026-09-08_evoneed_planned_route20_gold500/summary.json
- D1-Direct-NoPlan：data/exports/evaluation/2026-09-11_d1_ablation_gold500/summary.json
- D1-QueryRewrite：data/exports/evaluation/2026-09-11_d1_ablation_gold500/summary.json
- D1-QueryRewrite-SharedLink：data/exports/evaluation/2026-09-11_shared_link_ablation_gold500/summary.json
- Deep-OnePlan-D1-NoRerank：data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/summary.json
- OnePlan-Shared-EqualTotal：data/exports/evaluation/2026-09-11_shared_link_ablation_gold500/summary.json
- OnePlan-NoLink：data/exports/evaluation/2026-09-11_oneplan_nolink_gold500/summary.json
- EvoNeed-Deep：data/exports/evaluation/2026-09-09_evoneed_deep_gold500/summary.json
- HippoRAG2：data/exports/evaluation/2026-09-08_hipporag2_gold500/hipporag_summary.json
- EventRAG：data/exports/evaluation/2026-09-08_eventrag_budgeted_gold500/eventrag_hybrid_budgeted_summary.json
- E²RAG：data/exports/evaluation/2026-09-08_e2rag_gold500/e2rag_original_summary.json
