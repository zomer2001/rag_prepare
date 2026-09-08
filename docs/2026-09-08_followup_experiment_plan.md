# Gold-500 后续实验与端到端评测计划

日期：2026-09-08

## 1. 研究目标

本研究面向事件持续演化、证据分散的 RAG 场景，研究如何在固定检索预算下找回更多、更完整的普通 Gold Evidence。核心改进包括：

1. 使用文本向量、知识图谱和事件索引三条路径获取互补证据；
2. 使用 EvoLink 在三路结果之间进行一次跨路径关联与回扩展；
3. 使用 EvoNeed 一次性分析问题所需的证据类型，并在固定总预算内选择路径、生成路径查询和分配预算。

主实验不依赖 lifecycle-role Gold，也不新增人工标注。事件状态、时间和跨路径连接用于检索与上下文组织，最终仍使用数据集中已有的普通 Gold Evidence 和答案进行评价。

## 2. 当前受控检索实验

### 2.1 数据与统一条件

- 查询集：Gold-500，固定随机种子 `20260908`；
- 只评估具有非空 Gold Evidence 的问题，不包含 n30 和全量 2556 题；
- 语料：相同的 2512 个递归分块，chunk size 700、overlap 100；
- 最终输出：每种方法最多 Top-20 原文证据；
- 主要指标：Evidence Recall、Complete Recall、Hit Rate、MAP、MRR、nDCG；
- 分组报告：comparison、inference、temporal 三类问题；
- 索引固定，查询实验不得重建已有索引。

### 2.2 方法与预算

| 方法 | 初始候选预算 | 最终输出 | 说明 |
|---|---:|---:|---|
| Naive | Dense 20 | 20 | 原问题向量检索 |
| Hybrid | Dense 20 + BM25 20 | 20 | RRF 融合 |
| Light Entity | Vector 20 + Graph 20 | 20 | 实体图与文本融合 |
| Ours | Vector 20 + Graph 20 + Event 20 | 20 | 三路径召回 |
| Ours + EvoLink | 三路径各 20，另做一次共享关联扩展 | 20 | 跨路径证据连接 |
| EvoNeed Equal | 总预算 60 | 20 | 对启用路径平均分配预算 |
| EvoNeed Dynamic | 总预算 60，每条启用路径至少 5 | 20 | 按一次联合规划结果动态分配 |

EvoNeed 主实验严格执行：

> 一次联合规划 → 各启用路径并行检索 → 一次共享 EvoLink → 融合为 Top-20

每题只进行一次规划模型调用。检索阶段不再调用证据评估模型或答案生成模型，也不进行第二轮查询改写。

## 3. 需要补充的检索实验

### 3.1 BM25-only 强基线

增加 `BM25-only Top-20`，形成完整的基线链条：

> BM25 → Dense → Hybrid → Graph/Event Retrieval → EvoLink → EvoNeed

该实验用于判断 Hybrid 的主要增益来自稀疏词面匹配，还是来自稀疏与稠密结果的互补融合。

### 3.2 配对显著性与逐题增益

以相同的 500 个 query 为配对单位，对下列比较进行 bootstrap 置信区间或配对显著性检验：

- EvoLink vs. Hybrid；
- EvoNeed Equal vs. EvoLink；
- EvoNeed Dynamic vs. EvoNeed Equal；
- EvoNeed Dynamic vs. Hybrid。

除总体结果外，重点报告 temporal query 的 Complete Recall 和 Evidence Recall，并统计每种方法独有命中的 Gold Evidence。

### 3.3 严格事实片段诊断指标

主指标继续使用普通 Gold Evidence。另增加不需要人工标注的辅助诊断：利用数据集中现有 `fact` 字段，检查对应事实文本是否实际出现在召回 chunk 中。

该指标只用于解释文章级 Gold 命中与具体事实片段命中的差异，不替代主指标，也不引入 lifecycle-role Gold。

## 4. 受控答案生成实验

### 4.1 目的

验证“召回更多、更完整的 Gold Evidence”是否能够在生成阶段转化为更好的答案，而不是只停留在检索指标。

### 4.2 统一生成条件

所有方法使用完全相同的生成器：

- 输入：各方法已保存的 Top-20 原文 chunks；
- 相同 reader 模型、system prompt、temperature 和最大输出长度；
- 相同总上下文 token 上限；
- 每题只调用一次答案生成模型；
- 要求答案引用输入中的 evidence ID；
- 生成任务逐题保存，支持断点续跑，不与检索进程耦合。

此实验不得向任何方法额外提供实体摘要、事件链或社区报告，以便单独验证检索质量对答案的影响。

### 4.3 评价指标

- 短答案 Exact Match / token F1；
- 答案正确性和完整性；
- 引用 Gold Evidence 的 Recall；
- Citation Precision：引用内容是否支持对应陈述；
- Groundedness：答案事实是否能够在引用原文中找到；
- 按 comparison、inference、temporal 分组报告。

LLM Judge 只能作为辅助指标；主要结果应优先使用可复现的答案指标和引用指标。

## 5. 原生端到端系统比较

该实验允许各系统保留其原生上下文结构，不强制转换成同一种内容：

| 方法 | 可提供给最终模型的原生上下文 |
|---|---|
| Hybrid / Naive | 原文 chunks |
| LightRAG | 实体、关系、原文 chunks |
| GraphRAG Local | 实体、关系、社区报告、关联原文 |
| HippoRAG 2 | 经三元组过滤和 PPR 排序后的原文 passages |
| E²RAG | 原文、线性化实体—事件子图、节点与边描述 |
| EventRAG | 事件节点、时间关系、关联事件和迭代推理上下文 |
| Ours | 原文、事件时间线、EvoLink 跨路径连接、EvoNeed 需求分组 |

虽然上下文类型不同，但必须统一或完整披露：

- 最终 reader 模型与答案提示词；
- 最大输入和输出 token；
- 最终原文证据数量；
- 在线 LLM 调用次数及输入/输出 token；
- 查询延迟和估算成本；
- 是否包含多轮查询、反思、HyDE 或 map-reduce。

原生端到端结果不能被解释为纯检索提升，应明确其同时包含检索、上下文组织和推理计算的综合影响。

## 6. 我们方法的结构化上下文消融

在完全相同的已召回 Top-20 和总 token 预算下，比较：

1. `Raw Evidence`：只提供原文；
2. `Raw + Need Groups`：按 EvoNeed 的知识需求分组原文；
3. `Raw + Event Timeline`：增加按时间排序的事件及其原文来源；
4. `Raw + Event Timeline + Cross-path Links`：再增加实体—关系—事件连接。

建议的结构化上下文格式：

```text
Question

Knowledge needs
- Need ID / evidence type / query / allocated budget

Event timeline
- time / actor / event type / object / source evidence IDs

Cross-path links
- entity -> relation/event -> source evidence IDs

Raw evidence
- [evidence_id] source metadata and original text
```

结构化信息只能用于组织和导航，不能作为脱离原文的独立事实来源。最终答案中的事实必须引用原始 evidence ID。若事件摘要或图关系无法追溯到原文，则不得作为答案依据。

该部分仍保持一次最终生成调用，不增加反思、自我纠正或第二轮检索。由此可以单独验证事件演化组织与跨路径关联是否降低了模型整合碎片证据的难度。

## 7. 推荐执行顺序

1. 完成当前 Gold-500 检索，并冻结所有 raw retrieval outputs；
2. 补跑 BM25-only；
3. 汇总检索主表、分类型结果和配对显著性；
4. 对所有方法运行统一 Raw Top-20 的受控答案生成；
5. 运行我们方法的结构化上下文消融；
6. 最后再运行原生端到端系统比较，并单独报告计算成本；
7. 根据结果确定论文主结论，避免在结果出现前宣称未经验证的生成优势。

## 8. 论文结论与实验的对应关系

| 论文主张 | 对应证据 |
|---|---|
| 三路径检索提高证据覆盖 | Ours vs. Dense / Hybrid / Light Entity |
| 跨路径关联补齐分散证据 | EvoLink vs. Ours，独有 Gold 命中分析 |
| 需求驱动预算减少无效检索 | EvoNeed Dynamic vs. Equal，并报告路径预算与延迟 |
| 对事件演化问题更有效 | temporal query 分组结果 |
| 更完整证据改善最终答案 | 统一 Raw Top-20 受控生成实验 |
| 事件链与跨路径连接帮助答案整合 | 结构化上下文消融 |

只有当对应实验得到支持时，才能写入最终结论。若结构化上下文没有显著改善答案，则论文仍可将核心贡献落在普通 Gold Evidence 的完整召回上。
