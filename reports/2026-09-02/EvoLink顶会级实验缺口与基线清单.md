# EvoLink 顶会级实验缺口与基线清单

> 日期：2026-09-02  
> 研究主线：不把本体作为论文贡献，聚焦“原文—关系事实—事件演化”的多视图组织、EventEffect 生命周期和可追溯检索。  
> 判断口径：按 ACL/EMNLP/NAACL/EACL/NeurIPS 等顶会长文常见的审稿要求审计，不把小样本 smoke test 当正式结论。

## 1. 结论先行

当前实验已经足以说明“系统能运行，并且在一个小样本上有潜力”，但离顶会级证据还差一轮完整实验。最关键的不是继续堆很多普通基线，而是补齐以下五个硬缺口：

1. **没有严格证明增益来自 EvoLink。** 主表没有同索引、同预算、只关闭 `link_evidence` 的 `ours`；因此 `ours_evolink` 的提升不能归因给跨视图链接和 EventEffect。
2. **主实验规模太小。** MultiHop-RAG 官方数据有 609 篇文档、2,556 个问题；当前只用 100 篇文档和 30 个问题，而且只有 21 个问题进入证据指标。
3. **最核心的“事实演化”没有被直接测量。** 现有 MultiHop-RAG 指标只看是否召回证据，没有测当前事实、历史事实、事实被替代的原因、过期事实错误率，也没有测顺序增量写入。
4. **最近邻方法还没有形成公平主表。** HippoRAG 2、E²RAG、EventRAG 已接入，但目前仍是小样本或不同适配条件；主表还缺 LightRAG、GraphRAG、KG²RAG 等关键方法。
5. **只有检索结果，没有完整 RAG 结果和结构质量。** 缺少统一生成器下的答案正确率、引用正确率和忠实度；EventEffect、跨视图边、事件时间/因果关系本身也没有人工标注精度。

如果只补一件事，先做 `ours` 与 `ours_evolink` 的严格对照；如果要形成可投稿论文，必须同时补“完整数据 + ChronoQA/演化测试 + 端到端答案 + 结构标注 + 成本统计”。

## 2. 当前已经有什么

### 2.1 已有检索方法

- `naive`：Dense 向量检索。
- `hybrid`：Dense + BM25，以 RRF 融合。
- `light`：官方 LightRAG 接口已经集成，但没有进入当前 n30 主表。
- `light_entity`：受控实体/关系抽取后的文本—关系双路。
- `ours`：原文、关系、事件三路独立检索与文档级 RRF，不做 EvoLink 查询期跨路扩展。
- `ours_evolink`：在 `ours` 上启用实体—Assertion—Event—Effect—Mention 协调图扩展。
- `ours_tkg`、`ours_tkg_dynamic`：另一套统一 TKG Retriever，不能和 EvoLink 混作同一个方法。
- Agentic/EvoNeed 变体：已有原型，但当前没有超过固定三路版本，不适合作为这篇论文的主贡献。

### 2.2 已有外部方法

- HippoRAG 2：使用官方 `c617143` 版本，保留 OpenIE、PPR 和原生检索，已完成 n30。
- E²RAG：官方实现已适配并完成 n30。
- EventRAG：当前是关闭知识扩张、限制抽取数量的 `EventRAG-budgeted`，不是官方全预算复现。
- LightRAG：代码已集成，已有历史实验，但当前正式主表缺失。
- GraphRAG：尚未接入。

### 2.3 当前证据强度

MultiHop-RAG n30 主结果中，`ours_evolink` 的 Evidence Recall@10 / Complete Recall@10 / nDCG@10 为 0.8651 / 0.6190 / 0.7280，高于 `hybrid` 的 0.7302 / 0.4286 / 0.6563。但这一结果只有 21 个可计分问题，不能提供稳定置信区间，也没有最关键的 `ours` 对照。

DWIE n100 显示了明确的上游瓶颈：在 Gold Mention 条件下，当前实体治理 Hard Entity F1 为 0.9219；端到端条件下 Mention F1 只有 0.4822、Hard Entity F1 只有 0.1877。这说明“给定正确 Mention 后的合并”较好，但真实抽取召回会限制整个 EvoLink 图的质量。

## 3. 还缺哪些基线

不是所有 GraphRAG 都要复现。建议分为“必须进入主表”“强烈建议”“只在扩展主张时需要”。

### 3.1 必须进入主表

| 类别 | 方法 | 为什么必须有 | 当前状态 |
|---|---|---|---|
| 文本下界 | BM25 | 判断改进是否只是关键词匹配；当前只藏在 hybrid 内部 | 缺单独结果 |
| 文本强基线 | Dense；Dense+BM25；Dense+BM25+Cross-Encoder rerank | 没有 reranker 时，图方法可能只是赢了一个偏弱文本基线 | 前两项部分已有，rerank 缺失 |
| 轻量 KG-RAG | LightRAG hybrid/mix | 与当前关系视图实现最接近，也是 EventRAG、E²RAG 的常用基线 | 已接入，缺正式同条件主表 |
| 图传播 KG-RAG | HippoRAG 2 | 强多跳图检索基线，PPR 与当前局部扩展形成直接对照 | 已有 n30，需扩大并统一协议 |
| 经典 GraphRAG | GraphRAG local | 判断增益是否来自一般的实体邻域和原文回链，而非 EventEffect | 尚未接入 |
| Chunk 图扩展 | KG²RAG | 它同样从种子 Chunk 沿事实关系扩展并组织上下文，是 EvoLink 查询扩展最关键的非事件对照 | 缺失 |
| 事件图 RAG | EventRAG | 判断“事件索引 + 迭代事件检索”是否已足够 | 只有低预算版本 |
| 实体—事件双图 | E²RAG | 与多视图、事件和实体 Mention 保留最接近；也是 ChronoQA 上必须比较的方法 | 已有 n30 适配，需在原生 ChronoQA 上正式比较 |

其中最容易被审稿人指出而当前完全缺失的是 **KG²RAG**。它的“语义种子 → KG 引导扩展 → 组织 Chunk”与 EvoLink 的查询期补证高度相似；不比较它，就难以证明 EventEffect 和三视图结构比一般 KG 扩展多解决了什么。

### 3.2 强烈建议增加一项近期强方法

在下列两项中选择一项即可，不必全部堆入主表：

- **PropRAG**：以保留上下文的 proposition 替代简单三元组，并用无在线 LLM 的 beam search 找多跳路径；适合检验 EvoLink 是否真的需要事件生命周期，而不是高质量 proposition path 就足够。
- **PathRAG**：检索并剪枝关键关系路径，重点控制图扩展噪声；如果论文强调效率和“最小充分证据”，优先选它。

### 3.3 只有扩展论文主张时才需要

- 如果保留 Agentic/EvoNeed 主张：增加 RouteRAG、IRCoT 或另一种迭代检索基线，并评估路由和每轮增益。
- 如果声称 TKGQA：增加 TimeR⁴；否则它以已有 TKG 为输入，与“从文档构图的 RAG”不是完全同任务，不必放主表。
- 如果声称长文档层级组织：增加 RAPTOR；否则不是 EvoLink 的最近邻，可以只放附录。
- 不再主张本体后，**无需为了数量加入 ontology-RAG 基线**。固定类型约束只作为实现设置说明。

## 4. 公平比较协议必须重做

### 4.1 建议同时报告两个赛道

**受控赛道**用于归因：相同原始语料、相同基础 Chunk、Embedding、生成模型、问题、检索/上下文预算和评测脚本，只允许各方法改变自己的索引与检索算法。

**原生赛道**用于复现：允许外部方法使用论文推荐的 Chunk 和参数，单独报告结果与成本。原生赛道不能和受控赛道混成一个排名后直接宣称胜负。

### 4.2 Top-K 之外还要统一 Token Budget

不同方法返回的单位不一样：有的是 Chunk，有的是完整文档、图节点或线性化子图。只统一 Top-10 不公平。至少同时固定：

- 最终进入生成器的上下文 token 数；
- 最大返回文档数；
- 最大在线 LLM 调用数或检索轮数；
- 图扩展最大节点数、边数和 hop；
- 同一文档是否允许重复占用预算。

建议画“质量—上下文 token”和“质量—延迟/费用”曲线，而不是只报一个手工挑选的参数点。

### 4.3 当前延迟表不能作为正式效率结论

当前 runner 按方法顺序在同一进程运行，并共享已加载组件；`naive` P95 为 8.33 秒，而紧接着的 `hybrid` 只有 0.26 秒，容易混入冷启动、连接预热和执行顺序效应。正式实验应：

- 每个方法独立进程测量；
- 预先规定 warm-up 次数；
- 冷启动与 warm query 分开报告；
- 随机化方法执行顺序；
- 至少重复 3 次，报告均值、标准差和 P50/P95；
- 索引时间、查询时间、生成时间分开；缓存命中与未命中分开。

## 5. 顶会级实验还要补什么

### RQ1：完整系统是否真的有效

#### 数据集

建议最低配置为三组：

1. **MultiHop-RAG 全量**：609 篇文档、2,556 个问题，其中 2,255 个非 Null、301 个 Null。不能继续只用 30 题。
2. **ChronoQA**：18 部叙事作品、1,028 个带精确证据位置的问题，专门测时间、因果和实体状态演化；这是当前方法最重要的外部测试。
3. **MuSiQue 或 HotpotQA**：证明方法不只适用于 MultiHop-RAG 新闻语料。更建议 MuSiQue，因为 2/3/4-hop 分层更适合分析跨视图扩展随 hop 增加是否仍有效。

如果论文重点写“旧事实被新事实替代”，再增加 HoH 等过期信息数据，或构造时间快照测试集；普通多跳数据无法验证生命周期价值。

#### 指标

- 检索：Recall@{5,10,20}、Complete Recall@K、MRR、MAP、nDCG、Precision@K。
- 按问题类型：inference、comparison、temporal、null；按 2/3/4-hop；按跨文档数分层。
- 统计：逐问题 paired bootstrap 95% CI；与最强基线做配对显著性检验，并报告效应量。
- 至少 3 个抽取/采样随机种子；若全流程温度为 0，也应对问题采样、索引扰动或运行重复给出稳定性。

### RQ2：提升究竟来自哪个部件

必须做一套同索引、同预算消融。建议主消融表保留以下七项：

1. `Vector only`；
2. `Vector + Semantic`；
3. `Vector + Event`；
4. `Separate 3-view + RRF`，即 `ours`；
5. `ours_evolink - EventEffect`，只通过共享实体跨路；
6. `ours_evolink - typed lifecycle`，把 creates/updates/terminates/supersedes/denies/confirms 全部退化成无类型 related-to；
7. `ours_evolink` 完整版。

再用附录补三个 leave-one-out：关闭实体治理、关闭 Mention 来源回链、关闭查询期图扩展。

这组实验分别回答：多一路是否有效、三路简单融合是否已经足够、共享实体是否足够、EventEffect 类型是否有额外价值、生命周期是否真正改变检索。

### RQ3：事实生命周期是否真的比普通事件图有用

这是当前最缺、同时最可能形成论文辨识度的实验。构造按时间顺序到达的文档流，在 `t0/t1/t2/...` 每个快照提问，问题至少覆盖：

- 当前事实是什么；
- 某个历史时点事实是什么；
- 哪个事件导致事实改变；
- 新旧报道冲突时应相信哪一条；
- 没有充分证据时是否拒答。

对照至少包括：

- Append-only Dense/Hybrid；
- Append-only 三视图 RRF（无 lifecycle）；
- 时间戳 rerank；
- last-write-wins；
- EventRAG/E²RAG；
- 完整 EvoLink。

新增核心指标：Current Fact Accuracy、Historical Fact Accuracy、Stale-Fact Error Rate、Change-Cause Accuracy、Lifecycle Transition F1、时间区间准确率，以及每次增量更新的耗时和重建比例。

还要测试四种压力情形：乱序到达、同一事实重复改写、来源互相矛盾、时间戳缺失/有噪声。否则 EventEffect 只在理想数据上成立。

### RQ4：跨视图结构是否正确

不能只用最终 Recall 反推图是对的。建议人工标注 200–500 个事件或至少 100 篇文档，评估：

- Entity/Assertion/Event 与原文 Mention 回链 Precision、Recall、F1；
- Event—Assertion 是否应该连接；
- EventEffect 六种类型的 Macro-F1 和混淆矩阵；
- Assertion 的 active/superseded/terminated 状态准确率；
- 前态—事件—后态链完整率；
- 查询期新增边/新增证据的 precision，以及 helpful / neutral / harmful 比例。

外部抽取数据方面：

- DWIE：跑更大规模或全量，并补官方 Joint BERT/公开强模型；当前只有 n100，且官方预测仍缺。
- MAVEN：事件触发词和事件类型。
- MAVEN-ERE：事件共指、时间、因果和子事件关系。数据已经在本地，但目前只有完整性校验，没有模型分数。

必须增加三组 oracle 上界：Gold Entity、Gold Event、Gold EventEffect。这样才能判断失败来自抽取、实体合并、链接还是检索算法，而不是把所有错误归到 EvoLink。

### RQ5：作为 RAG，最终答案和引用是否更好

所有方法使用同一个生成模型、同一个 Prompt 和相同上下文 token 预算，至少报告：

- 有标准答案的数据：EM、token F1、Answer Correctness；
- 引用：Citation Precision、Citation Recall、支持完整率；
- 忠实度：回答中的主张有多少能被引用证据支持；
- Null 问题：拒答 Precision、Recall、F1，而不是从检索平均分中删除；
- 时间问题：当前答案正确率、历史答案正确率、把旧事实当当前事实的比例。

建议再换一个生成模型复验，证明收益来自检索结构而不是某个生成器偏好。若使用 LLM-as-judge，应固定 rubric、打乱方法顺序、隐藏方法名，并在人评子集上校验 judge 一致性。

### RQ6：收益是否值得成本

当前 `ours_evolink` 的 P95 为 17.81 秒，而 `hybrid` 为 0.26 秒；即使该延迟对比受运行协议影响，效率仍是审稿高风险点。正式报告应包含：

- 索引阶段：LLM 调用数、输入/输出 token、费用、墙钟时间、峰值内存、磁盘和图规模；
- 增量阶段：每新增 1/10/100 篇文档的更新时间、受影响节点/边数、是否需要全量重建；
- 查询阶段：P50/P95、在线 LLM 调用数、token、图扩展节点/边数；
- 质量—成本 Pareto：在相同 0/1/2 次在线 LLM 调用、相同 2k/4k/8k context 下比较；
- 扩展深度和预算敏感性：hop={0,1,2,3}、top-k 和 RRF k。

## 6. 推荐的正式主表与实验顺序

### 6.1 主表不要混入太多内部原型

建议主表方法为：

1. BM25；
2. Dense；
3. Hybrid + reranker；
4. LightRAG；
5. GraphRAG local；
6. HippoRAG 2；
7. KG²RAG；
8. EventRAG；
9. E²RAG；
10. Ours（3-view RRF）；
11. Ours + EvoLink。

PropRAG/PathRAG 二选一放主表或附录。Agentic、EvoNeed、`ours_tkg`、`ours_tkg_dynamic` 建议移到附录或另一篇工作，避免一张表出现多个定义不清的“Ours”。

### 6.2 实施优先级

#### P0：不补就无法可靠归因

- 在同一个冻结索引上重跑 `ours` 与 `ours_evolink`。
- 实现 `-EventEffect`、`-effect type`、`-entity governance` 三个关键消融。
- 修正 Null 问题评价，不再把 301 个 Null 当作无 Gold 后直接排除。
- 给现有 n30 做逐问题差异检查，确认 EvoLink 到底新增了哪些 Gold、又引入了哪些噪声。

#### P1：形成可投稿主体

- 跑 MultiHop-RAG 全量非 Null + Null 专项。
- 接入 ChronoQA，并在其上正式跑 E²RAG。
- 补 LightRAG、GraphRAG local、KG²RAG；HippoRAG 2 和 EventRAG 按统一预算重跑。
- 用统一生成器评估答案、引用和忠实度。
- 人工标注 EventEffect/跨视图边，并补 MAVEN-ERE 结构指标。

#### P2：冲击强论文，而不是只够“有实验”

- 建立顺序增量的事实演化测试集和 stale-fact 指标。
- 增加 MuSiQue/HotpotQA 泛化实验。
- 做噪声、缺边、乱序、冲突来源鲁棒性。
- 做独立进程的成本/延迟测量、预算曲线、三次重复和统计显著性。
- 增加 PropRAG 或 PathRAG 作为近期强检索基线。

## 7. 最终判断

现阶段最危险的论文写法是：“我们的三路 + EvoLink 优于 Hybrid、EventRAG、E²RAG。”因为现有数据只能说明完整系统在 21 个可计分问题上表现较好，不能证明提升来自 EvoLink，也不能证明 EventEffect 生命周期比普通事件图或一般 KG 扩展更好。

更稳妥、也更有机会形成顶会贡献的实验逻辑是：

> 普通文本、一般 KG 扩展和事件图都能提高多跳召回，但它们不能显式回答“哪个事件让哪条事实从什么状态变成什么状态”。EvoLink 在相同检索预算下，通过有类型、可回溯的 EventEffect 把事件变化连接到事实生命周期；这不仅提高完整证据召回，还降低动态语料中的过期事实错误，并能给出变化原因及来源。

要让这段话成立，最关键的证据不是再多跑几个普通 RAG，而是：`ours` 对照、EventEffect 消融、ChronoQA、顺序增量演化测试、stale-fact 指标和结构人工标注。

## 8. 参考依据

- [MultiHop-RAG（COLM 2024）](https://openreview.net/pdf?id=t4eB3zYWBK)：完整数据和多跳检索任务。
- [HippoRAG（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)：KG + Personalized PageRank 多跳检索。
- [EventRAG（ACL 2025）](https://aclanthology.org/2025.acl-long.830/)：事件图和迭代多事件检索。
- [KG²RAG（NAACL 2025）](https://aclanthology.org/2025.naacl-long.449/)：KG 引导的 Chunk 扩展和上下文组织。
- [PropRAG（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.317/)：proposition path 与无在线 LLM 的多跳检索。
- [E²RAG 与 ChronoQA（EACL 2026）](https://aclanthology.org/2026.eacl-long.90/)：实体 Mention—事件双图及时间/因果一致性评测。
- [PathRAG（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/40268)：关系路径检索和图扩展剪枝。
- [TimeR⁴（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.394/)：已有 TKG 输入下的时间感知 retrieve–rewrite–retrieve–rerank。
- [HoH](https://arxiv.org/abs/2503.04800)：旧信息与新信息共存条件下的 RAG 动态事实评测。

