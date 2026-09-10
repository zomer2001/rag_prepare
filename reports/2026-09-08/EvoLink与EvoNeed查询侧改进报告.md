# EvoLink 与 EvoNeed 查询侧改进报告

日期：2026-09-08  
范围：只分析和修改查询、融合与排序；复用既有 609 文档、2512 chunk 的索引，不重建索引。

## 1. 当前结果与问题

| 方法 | Evidence Recall@20 | Complete Recall@20 | nDCG@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Hybrid | 0.8672 | 0.6680 | 0.6660 | 0.7649 |
| Ours | 0.8375 | 0.6080 | 0.6361 | 0.7017 |
| Ours + EvoLink | 0.8728 | 0.6940 | 0.6483 | 0.6974 |
| EvoNeed Equal | 0.8172 | 0.6140 | 0.6057 | 0.6656 |
| EvoNeed Dynamic | 0.8125 | 0.6080 | 0.6035 | 0.6621 |

EvoLink 的关联扩展是有效的：相对 Ours，Evidence Recall@20 提升 3.53 个百分点，Complete Recall@20 提升 8.60 个百分点；500 题中有 48 题增加 Gold Evidence，未出现 Recall@20 下降。但是 MRR 下降 0.43 个百分点，而且仍明显低于 Hybrid，说明问题主要在前排排序而不是召回覆盖。逐题看，关联奖励会把部分“与种子相连、但与问题不够直接”的文档推到前面，挤压原问题直接命中的 Gold。

EvoNeed 的问题主要来自查询和预算利用，而不是索引：

- 平均每题只启用 2.086 条路径；vector、graph、event 的启用率分别为 99.0%、32.4%、77.2%。对部分多跳题，planner 没有启用可提供互补证据的路径。
- Planner 的专门查询几乎全部改写了原问题。它虽然保留了主题，却可能丢失来源名、比较对象、限定词和原始措辞，弱化了本数据集非常有效的词面匹配。
- Dynamic 把平均 37.67 个预算给 vector，但实际平均只返回 12.63 个去重文档；graph 为 20.45→16.25，event 为 20.83→13.12。固定 60 候选预算没有充分转化为不同文档。
- Need coverage 融合会保证不同 need 得到展示，但当某条专门查询质量一般时，也会把弱结果提前。

## 2. 已完成的小改进

### EvoLink：降低关联奖励

将关联奖励权重从 `0.10` 降为 `0.03`。EvoLink 仍负责扩展和近似打破，但不再像一条独立检索路径那样强烈改变原始相关性排序。没有增加新模型、新索引或额外查询。

### EvoNeed：原问题锚定的专门查询

每条路径仍只执行一次 planner 生成的专门查询，但实际检索文本改为：

```text
原始问题
Retrieval focus: planner 生成的路径焦点
```

这样保留原问题的实体、来源、日期、比较关系和措辞，同时继续利用证据范式化后的 vector、graph、event 焦点。调用结构仍是“一次规划、一次并行检索、一次共享 EvoLink”，没有增加 LLM 调用或检索轮次。

## 3. 待跑的两组纯 Query 实验

1. 改进主实验：`ours`、`ours_evolink`、`ours_evoneed_equal`、`ours_evoneed_dynamic`；EvoNeed 总初始预算 60，最终 Top-20。
2. 高预算 Planned：`ours_evoneed_planned`；每条启用路径最多 20 个候选，最多三路、总量最多 60，最终 Top-20。

第二组保留“先把证据需求范式化，再按需要查询”的解释，同时明确它是 route-depth-20 版本。它只做 retrieval-only：一次 planner、一次并行检索，不做证据评估、回答生成或第二轮改写。

统一运行入口：

```bash
./scripts/run_query_improvements_gold500.sh --run
```

状态查看：

```bash
./scripts/run_query_improvements_gold500.sh --status
```

输出位置：

- `data/exports/evaluation/2026-09-08_query_improvements_gold500/`
- `data/exports/evaluation/2026-09-08_evoneed_planned_route20_gold500/`
- `reports/2026-09-08/logs/query_improvements_gold500.log`

## 4. 验证状态

- 已验证 Light Entity 索引为 2512 chunks；事件索引和 EvoLink 索引均覆盖 609 个文档。
- 已增加运行前完整性保护：任一既有索引不完整时直接退出，不自动补建。
- 全部测试通过。
- 实际 Gold-500 查询尚未启动：运行会调用外部 embedding/planner 模型并产生费用，且此前存在“只给指令、不要代跑”的明确约束，需要用户重新确认后才能启动。

最终只根据新结果判断改进是否保留：重点看 EvoLink 的 MRR/nDCG 是否改善且 Recall/Complete Recall 不明显回退，以及 EvoNeed 是否缩小与 EvoLink/Hybrid 的差距。
