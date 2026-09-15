# Gold500 检索结果汇总

日期：2026-09-11  
评测范围：2026 年 9 月的完整 Gold500 结果

## 1. 统计口径

- 数据集：MultiHopRAG Gold500
- 文档数：609
- chunk 数：2512
- 问题数：500
- 最终返回预算：Top-20
- 问题类型：Comparison 190 题、Inference 181 题、Temporal 129 题
- 已排除：n30、8 月结果、9 月 3 日 q100 小样本
- GraphRAG 尚未完成 Gold500 结果
- 2026-09-09 带 LLM 重排的 OnePlan-D1 只有 115/500，未纳入主表

主要指标含义：

- `Evidence Recall@20`：返回结果覆盖 Gold evidence 的比例
- `Complete Recall@20`：一次性找齐该题全部 Gold evidence 的比例
- `nDCG@20`、`MRR@20`：衡量 Gold evidence 排在前面的质量
- 延迟为每题平均检索时间

## 2. 总体结果

| 方法 | 方法作用 | Hit@20 | Evidence Recall@20 | Complete Recall@20 | nDCG@20 | MRR@20 | 平均延迟 |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive | 单路 Dense 原问题检索 | 0.958 | 0.740 | 0.474 | 0.555 | 0.679 | 0.37s |
| Hybrid | Dense + BM25 融合 | **0.998** | 0.867 | 0.668 | **0.666** | **0.765** | **0.32s** |
| LightRAG-Entity | 向量 + 实体图检索 | 0.984 | 0.800 | 0.534 | 0.624 | 0.715 | 2.18s |
| Ours | 向量、关系图、事件图三路直接融合 | 0.984 | 0.836 | 0.610 | 0.633 | 0.696 | 4.83s |
| Ours + EvoLink | 三路检索后进行共享跨视图扩展 | 0.992 | **0.879** | **0.706** | 0.646 | 0.694 | 1.37s |
| EvoNeed-Equal | 规划证据需求，路径预算平均分配 | 0.986 | 0.847 | 0.634 | 0.618 | 0.662 | 1.88s |
| EvoNeed-Dynamic | 规划证据需求，动态分配路径预算 | 0.986 | 0.845 | 0.632 | 0.614 | 0.654 | 2.32s |
| EvoNeed-Planned | 最多三路，每路最多 20 个候选 | 0.986 | 0.839 | 0.636 | 0.609 | 0.655 | 5.71s |
| HippoRAG2 | 外部 HippoRAG 2 原生图检索 | 0.960 | 0.776 | 0.530 | 0.558 | 0.639 | 1.54s |
| EventRAG | 外部事件图检索适配版本 | 0.764 | 0.451 | 0.180 | 0.248 | 0.247 | 2.63s |
| E²RAG | 外部实体—事件双图检索 | 0.898 | 0.613 | 0.316 | 0.408 | 0.443 | 1.54s |
| EvoNeed-Deep | 高候选深度 + 每个 need 独立 EvoLink + LLM 重排 | **0.996** | **0.928** | **0.814** | **0.762** | **0.842** | 16.14s |
| Deep-OnePlan-D1-NoRerank | 一次联合规划，不做 LLM 重排 | 0.994 | 0.902 | 0.774 | 0.632 | 0.637 | 3.94s |

### 2.1 Top-5 结果

| 方法 | Hit@5 | Evidence Recall@5 | Complete Recall@5 | nDCG@5 | MRR@5 | MAP@5 |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 0.808 | 0.454 | 0.158 | 0.445 | 0.662 | 0.362 |
| Hybrid | 0.924 | 0.591 | 0.276 | 0.558 | 0.756 | 0.470 |
| LightRAG-Entity | 0.896 | 0.561 | 0.242 | 0.528 | 0.705 | 0.444 |
| Ours | 0.906 | 0.588 | 0.272 | 0.534 | 0.687 | 0.450 |
| Ours + EvoLink | 0.902 | 0.588 | 0.276 | 0.533 | 0.684 | 0.450 |
| EvoNeed-Equal | 0.868 | 0.563 | 0.262 | 0.505 | 0.647 | 0.420 |
| EvoNeed-Dynamic | 0.866 | 0.565 | 0.270 | 0.503 | 0.640 | 0.418 |
| EvoNeed-Planned | 0.874 | 0.558 | 0.260 | 0.498 | 0.641 | 0.414 |
| HippoRAG2 | 0.816 | 0.472 | 0.164 | 0.442 | 0.622 | 0.356 |
| EventRAG | 0.382 | 0.170 | 0.020 | 0.133 | 0.195 | 0.090 |
| E²RAG | 0.660 | 0.363 | 0.112 | 0.309 | 0.415 | 0.236 |
| EvoNeed-Deep | 0.934 | 0.665 | 0.394 | 0.660 | 0.834 | 0.595 |
| Deep-OnePlan-D1-NoRerank | 0.874 | 0.571 | 0.296 | 0.501 | 0.623 | 0.417 |

### 2.2 Top-10 结果

| 方法 | Hit@10 | Evidence Recall@10 | Complete Recall@10 | nDCG@10 | MRR@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 0.898 | 0.590 | 0.278 | 0.503 | 0.675 | 0.396 |
| Hybrid | 0.988 | 0.745 | 0.454 | 0.623 | 0.764 | 0.512 |
| LightRAG-Entity | 0.956 | 0.717 | 0.424 | 0.594 | 0.713 | 0.487 |
| Ours | 0.960 | 0.725 | 0.452 | 0.593 | 0.695 | 0.489 |
| Ours + EvoLink | 0.962 | 0.730 | 0.456 | 0.595 | 0.692 | 0.490 |
| EvoNeed-Equal | 0.952 | 0.730 | 0.452 | 0.577 | 0.659 | 0.467 |
| EvoNeed-Dynamic | 0.950 | 0.729 | 0.452 | 0.574 | 0.652 | 0.463 |
| EvoNeed-Planned | 0.956 | 0.721 | 0.442 | 0.569 | 0.653 | 0.459 |
| HippoRAG2 | 0.916 | 0.625 | 0.298 | 0.506 | 0.636 | 0.395 |
| EventRAG | 0.730 | 0.420 | 0.156 | 0.237 | 0.244 | 0.137 |
| E²RAG | 0.818 | 0.528 | 0.244 | 0.379 | 0.437 | 0.274 |
| EvoNeed-Deep | 0.986 | 0.799 | 0.560 | 0.717 | 0.841 | 0.633 |
| Deep-OnePlan-D1-NoRerank | 0.950 | 0.749 | 0.510 | 0.579 | 0.633 | 0.469 |

## 3. 按问题类型的结果

每个单元格为：`Complete Recall@K / Evidence Recall@K`。

### 3.1 Top-5 分类型结果

| 方法 | Comparison（190） | Inference（181） | Temporal（129） |
|---|---:|---:|---:|
| Naive | 0.242 / 0.524 | 0.050 / 0.360 | 0.186 / 0.483 |
| Hybrid | 0.421 / 0.667 | 0.122 / 0.498 | 0.279 / 0.611 |
| LightRAG-Entity | 0.342 / 0.610 | 0.110 / 0.488 | 0.279 / 0.592 |
| Ours | 0.363 / 0.619 | 0.133 / 0.521 | 0.333 / 0.637 |
| Ours + EvoLink | 0.363 / 0.619 | 0.144 / 0.524 | 0.333 / 0.633 |
| EvoNeed-Equal | 0.342 / 0.598 | 0.127 / 0.463 | 0.333 / 0.651 |
| EvoNeed-Dynamic | 0.363 / 0.610 | 0.127 / 0.456 | 0.333 / 0.651 |
| EvoNeed-Planned | 0.342 / 0.593 | 0.127 / 0.462 | 0.326 / 0.642 |
| HippoRAG2 | 0.247 / 0.519 | 0.083 / 0.421 | 0.155 / 0.475 |
| EventRAG | 0.021 / 0.140 | 0.011 / 0.196 | 0.031 / 0.176 |
| E²RAG | 0.147 / 0.401 | 0.028 / 0.246 | 0.178 / 0.472 |
| EvoNeed-Deep | 0.495 / 0.713 | 0.221 / 0.564 | 0.488 / 0.735 |
| Deep-OnePlan-D1-NoRerank | 0.411 / 0.657 | 0.155 / 0.462 | 0.326 / 0.597 |

### 3.2 Top-10 分类型结果

| 方法 | Comparison（190） | Inference（181） | Temporal（129） |
|---|---:|---:|---:|
| Naive | 0.384 / 0.628 | 0.110 / 0.510 | 0.357 / 0.646 |
| Hybrid | 0.611 / 0.799 | 0.293 / 0.676 | 0.450 / 0.761 |
| LightRAG-Entity | 0.495 / 0.728 | 0.315 / 0.680 | 0.473 / 0.751 |
| Ours | 0.511 / 0.727 | 0.298 / 0.668 | 0.581 / 0.802 |
| Ours + EvoLink | 0.516 / 0.736 | 0.298 / 0.671 | 0.589 / 0.805 |
| EvoNeed-Equal | 0.553 / 0.756 | 0.271 / 0.660 | 0.558 / 0.791 |
| EvoNeed-Dynamic | 0.547 / 0.752 | 0.282 / 0.663 | 0.550 / 0.787 |
| EvoNeed-Planned | 0.532 / 0.742 | 0.249 / 0.639 | 0.581 / 0.805 |
| HippoRAG2 | 0.368 / 0.642 | 0.177 / 0.576 | 0.364 / 0.668 |
| EventRAG | 0.221 / 0.466 | 0.028 / 0.307 | 0.240 / 0.509 |
| E²RAG | 0.321 / 0.582 | 0.083 / 0.375 | 0.357 / 0.664 |
| EvoNeed-Deep | 0.653 / 0.831 | 0.365 / 0.715 | 0.698 / 0.870 |
| Deep-OnePlan-D1-NoRerank | 0.637 / 0.797 | 0.348 / 0.686 | 0.550 / 0.769 |

### 3.3 Top-20 分类型结果

| 方法 | Comparison（190） | Inference（181） | Temporal（129） |
|---|---:|---:|---:|
| Naive | 0.553 / 0.754 | 0.365 / 0.694 | 0.512 / 0.784 |
| Hybrid | **0.779 / 0.891** | 0.497 / 0.820 | 0.744 / 0.898 |
| LightRAG-Entity | 0.595 / 0.796 | 0.425 / 0.782 | 0.597 / 0.829 |
| Ours | 0.668 / 0.832 | 0.442 / 0.800 | 0.760 / 0.894 |
| Ours + EvoLink | 0.753 / 0.876 | **0.586 / 0.855** | **0.806 / 0.915** |
| EvoNeed-Equal | 0.711 / 0.859 | 0.464 / 0.799 | 0.760 / 0.895 |
| EvoNeed-Dynamic | 0.700 / 0.854 | 0.475 / 0.802 | 0.752 / 0.891 |
| EvoNeed-Planned | 0.721 / 0.859 | 0.453 / 0.778 | 0.767 / 0.895 |
| HippoRAG2 | 0.579 / 0.774 | 0.431 / 0.759 | 0.597 / 0.802 |
| EventRAG | 0.263 / 0.500 | 0.033 / 0.331 | 0.264 / 0.545 |
| E²RAG | 0.405 / 0.671 | 0.144 / 0.465 | 0.426 / 0.735 |
| EvoNeed-Deep | **0.911 / 0.956** | **0.630 / 0.870** | **0.930 / 0.968** |
| Deep-OnePlan-D1-NoRerank | 0.853 / 0.926 | 0.624 / 0.851 | 0.868 / 0.937 |

## 4. 方法说明与可比性

### 4.1 内部方法

- **Naive**：只用原问题进行 Dense 检索。
- **Hybrid**：Dense 与 BM25 并行检索后融合。
- **LightRAG-Entity**：使用向量和实体图两种视图检索。
- **Ours**：同时使用向量、关系图、事件图三条路径，直接融合，不做 EvoLink 扩展。
- **Ours + EvoLink**：在三路种子结果上沿共享实体、断言、事件和作用关系进行一次跨视图扩展。
- **EvoNeed-Equal**：先把问题规划为证据需求，再按活动路径平均分配总预算。
- **EvoNeed-Dynamic**：先规划证据需求，再由 planner 提议各路径预算。
- **EvoNeed-Planned**：最多启用三条路径，每条最多 20 个候选，最后保留 Top-20。

### 4.2 外部方法

- **HippoRAG2**：保留 HippoRAG 2 的原生图检索逻辑。
- **EventRAG**：保留事件图检索逻辑，使用当前项目的 budgeted 适配入口。
- **E²RAG**：保留实体图、事件图及其映射的原生检索逻辑。

### 4.3 高预算方法

- **EvoNeed-Deep**：每个 need 使用更深的候选池，每个 need 独立做 EvoLink，并增加 LLM 证据评估/重排。因此它应作为高计算量上界，不应直接与固定预算主线混为同一种方法。
- **Deep-OnePlan-D1-NoRerank**：保留一次联合规划和较深候选，但关闭 LLM 重排。它主要用于区分“候选覆盖收益”和“排序收益”。

主线内部方法使用 2026-09-08 的 Gold500 结果；Naive、Hybrid、LightRAG-Entity 沿用同一轮内部基线，Ours、EvoLink、EvoNeed Equal/Dynamic 使用查询改进轮次。EvoNeed-Planned 使用独立的 route-20 配置。高预算 Deep 方法使用不同的候选深度和重排配置，因此只能作补充对照。

## 5. 结果分析

### 5.1 不同返回深度的变化

EvoLink 的收益主要在更大的返回预算中体现：

- Top-5：Ours 与 Ours + EvoLink 的 Complete Recall 都为 0.276，Evidence Recall 都为 0.588；
- Top-10：Ours + EvoLink 的 Complete Recall 为 0.456，略高于 Ours 的 0.452；
- Top-20：Ours + EvoLink 的 Complete Recall 为 0.706，高于 Ours 的 0.610。

这说明 EvoLink 主要补回排名较后的关联证据，而不是显著改善最前面的 Top-5。它的作用更接近“扩大有效证据覆盖”，不是单纯的前排重排。

### 5.2 EvoLink 的主要价值是补全证据

相对同轮 Ours：

- Evidence Recall@20：0.836 → 0.879；
- Complete Recall@20：0.610 → 0.706；
- nDCG@20：0.633 → 0.646；
- MRR@20：0.696 → 0.694。

因此，EvoLink 明显提高了证据覆盖和完整召回，但没有提高首位排序。它更适合解释为“跨视图扩展补回遗漏证据”，而不是“让最相关证据一定排在第一位”。

### 5.3 EvoLink 对 Inference 和 Temporal 更有帮助

Ours + EvoLink 的 Complete Recall@20 在：

- Inference 上为 0.586；
- Temporal 上为 0.806；
- Comparison 上为 0.753。

这说明跨事件、跨关系和跨文档扩展，对需要多段证据或时间关联的问题更有效。Comparison 问题上，Hybrid 的排序和完整召回仍然更强。

### 5.4 EvoNeed 当前还没有超过 EvoLink

Equal、Dynamic 和 Planned 的 Complete Recall@20 分别为 0.634、0.632 和 0.636，均低于 Ours + EvoLink 的 0.706。Dynamic 还比 Equal 更慢，暂时不能证明动态预算分配本身有独立收益。

### 5.5 Deep 的高分来自更高计算预算

EvoNeed-Deep 的 Recall 和排序均最高，但平均延迟达到 16.14 秒。它同时使用更深候选池、每个 need 独立扩展和 LLM 重排，因此适合作为高预算上界，而不是当前固定预算主方法。

Deep-OnePlan-D1-NoRerank 的 Complete Recall@20 已达到 0.774，但 nDCG@20 只有 0.632、MRR@20 只有 0.637。这说明候选覆盖已经较强，LLM 重排主要改善的是候选顺序，而不是把新的 Gold evidence 找进来。

### 5.6 Inference 是当前最难的问题类型

所有方法在 Inference 上的 Complete Recall 普遍低于 Comparison 和 Temporal。EventRAG 和 E²RAG 在 Inference 上尤其低，说明当前事件图适配版本对需要多段组合证据的题目覆盖不足。

外部 EventRAG/E²RAG 的低分只能说明它们在当前 MultiHopRAG、当前索引和 Gold evidence 口径下覆盖较低，不能直接推出原论文方法本身无效。它们保留了原生检索逻辑，与当前项目的三路索引和 EvoLink 机制并不相同。

## 6. 当前结论

当前最稳妥的论文主结论是：

> 在相同 Gold500 和最终 Top-20 预算下，EvoLink 能够提高跨文档证据的完整召回，尤其适合 Inference 和 Temporal 问题；但它对证据排序的改善有限，排序优势仍主要来自 Hybrid 或额外的 LLM 重排。

EvoNeed-Deep 可以作为高计算量上界展示；Dynamic、Planned 暂时不宜单独包装成已经验证的核心创新。

## 7. 数据文件

- [内部基线汇总](../../data/exports/evaluation/2026-09-08_all_internal_gold500/summary.md)
- [查询改进汇总](../../data/exports/evaluation/2026-09-08_query_improvements_gold500/summary.md)
- [EvoNeed-Planned 汇总](../../data/exports/evaluation/2026-09-08_evoneed_planned_route20_gold500/summary.md)
- [EvoNeed-Deep 汇总](../../data/exports/evaluation/2026-09-09_evoneed_deep_gold500/summary.md)
- [Deep-OnePlan-D1-NoRerank 汇总](../../data/exports/evaluation/2026-09-09_evoneed_deep_oneplan_d1_norerank_gold500/summary.md)
