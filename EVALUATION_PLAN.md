# Anduril OSINT 多通路 RAG 量化评测与基线方案

> 版本：v1.0
> 日期：2026-08-05
> 适用系统：Collection + LightRAG 向量/语义图谱 + Event assertion pathway
> 目标：让每个可独立失败的处理阶段都可测，并把“检索到了”“回答正确”“证据支持”分开报告。

---

## 1. 结论先行

### 1.1 主基线

本项目最合适的**主基线**不是裸 LLM，也不是另一个复杂 GraphRAG，而是：

```text
相同语料 + 相同解析结果 + 相同 Chunk + 相同 Qwen3 Embedding
+ BM25/dense 召回 + RRF 融合 + 可选同一 BGE Reranker
+ 相同 DeepSeek 生成模型 + 相同上下文 token 预算
- 图谱检索 - 事件层
```

简写为 `Hybrid-RAG`。它足够强、容易复现，且与完整系统只差“图谱和事件机制”，最能回答：
这些复杂机制是否在质量、成本和延迟上产生了净收益。

至少同时保留以下对照：

1. `No-RAG`：只用同一个 DeepSeek；检验检索总体是否有效。
2. `BM25`：词法检索；作为便宜、稳健的经典下界。
3. `Dense-RAG`：当前 Qwen3 Embedding 的纯向量检索。
4. `Hybrid-RAG`：BM25 + Dense + RRF，主基线。
5. `Hybrid-RAG + Reranker`：强简单基线。
6. `LightRAG-base`：当前 `naive/local/global/hybrid/mix`，但 `use_events=false`。
7. `Full`：LightRAG + Event pathway，即本项目完整系统。
8. `Oracle-context`：人工标注的金标准证据直接给同一生成模型；作为检索上界。
9. `Long-context`：小语料可加入全文/原结构输入；作为“是否真的需要 RAG”的控制组。

可选的外部复杂基线是 RAPTOR、HippoRAG、Microsoft GraphRAG；它们适合论文扩展实验，
不应替代前面的强简单基线。

### 1.2 第一阶段必须落地的 12 个指标

如果资源有限，先做下列指标，而不是一开始构造一个不透明的总分：

| 范围 | 核心指标 | 为什么必须测 |
|---|---|---|
| 解析 | Source-marker integrity | 页码/文档来源是否在解析和切块后仍可追踪 |
| 实体 | Entity mention F1 | 是否抽对实体提及 |
| 对齐 | Entity alignment Hits@1、pairwise F1 | 别名是否合并正确，是否过合并/漏合并 |
| 事件 | Event extraction F1 | 事件类型、动作、参数是否抽对 |
| 事件合并 | CoNLL F1 + pairwise F1 | 跨文档事件是否指向同一真实事件 |
| 冲突 | Conflict macro-F1 + evidence F1 | 是否正确发现支持/否定及对应证据 |
| 检索 | Evidence Recall@10 | 金标准证据是否进入上下文 |
| 多跳检索 | Complete evidence-set recall | 回答所需的整套证据是否全部召回 |
| 排序 | nDCG@10、MRR | 有用证据是否排在前面 |
| 生成 | Answer correctness | 是否满足问题，而不仅是措辞相似 |
| 可信性 | Claim faithfulness、citation correctness/completeness | 每个事实是否有正确、完整的引用支持 |
| 工程 | p50/p95 latency、单问成本、索引吞吐 | 质量增益是否值得工程代价 |

所有结果都应按问题类型、来源类型、时间新旧、语言、单跳/多跳、是否含冲突分桶，不能只报宏观均值。

---

## 2. 评测对象与故障边界

当前系统可拆成以下可独立评测的接口：

```text
采集/版本治理
  -> 文档解析与来源保真
  -> Chunk / Entity / Relation 索引
  -> 实体规范化与对齐
  -> 事件抽取与证据绑定
  -> 跨文档事件合并
  -> 时间关系与冲突消解
  -> 查询理解与三通路召回
  -> 上下文融合/重排
  -> 答案生成与引用映射
  -> API 与运行效率
```

每个接口必须至少有：输入金标准、直接输出指标、错误类型、简单基线、oracle 上界。
否则端到端分数下降时无法判断是解析、对齐、检索还是生成造成的。

---

## 3. 全链路指标矩阵

### 3.1 Collection、数据治理与解析

| 位置 | 指标 | 定义/统计方式 | 主要对比 |
|---|---|---|---|
| 来源覆盖 | Source coverage | 实际采集来源数 / 计划来源数；按官网、合同、论文、专利、新闻、智库分层 | 手工清单/oracle |
| 采集成功 | Fetch success rate | 成功且通过 hard validation 的请求 / 应采请求 | 当前 adapter vs 手工下载 |
| 时效 | Freshness lag | 来源发布时间到入库时间的 p50/p95；无发布时间则用首次发现时间 | 不同 adapter/调度周期 |
| 完整性 | Field completeness | title、published_at、canonical_url、provider_id、正文等字段非空率 | 当前 parser vs source-specific parser |
| 内容质量 | Parse acceptance/review rate | hard validation 通过率、`needs_review` 率、警告分布 | 当前规则 vs 人工抽检 |
| 精确去重 | Exact duplicate P/R/F1 | provider ID、URL、raw hash、normalized hash 对金标准重复对 | 单键规则、当前级联规则 |
| 近似去重 | Near-duplicate P/R/F1 | SimHash 候选是否为同稿/转载/版本关系 | 无 SimHash、不同 Hamming 阈值 |
| 去重效率 | Pair completeness / reduction ratio | blocking 保留的真重复比例；候选对相对全笛卡尔积的缩减比例 | 全比较、规则 blocking |
| 版本化 | Version accuracy | 内容变化时是否建新版本；未变化时是否保持原版本 | hash-only、当前版本规则 |
| 文本解析 | CER/WER 或 normalized edit similarity | PDF/DOCX 解析文本与人工转录文本的字符/词差异 | Docling、其他 parser（可选） |
| 结构解析 | Heading/list/table F1 | 标题层级、列表、表格单元格的识别 P/R/F1；表格可加 TEDS | Markdown 原文/oracle |
| 来源保真 | Source-marker integrity | 带来源标记的 gold 单元中，切块后仍能恢复 document/page/section 的比例 | 当前 parser/chunker |
| 引用定位 | Page/section accuracy | 引用页码 exact accuracy；章节 exact/relaxed F1 | 当前 SourceMapper |

额外报告：每种格式的解析耗时、峰值内存、每页失败率、空正文率、乱码率。PDF、DOCX、Markdown 必须分开报告。

### 3.2 Chunk 与索引一致性

| 指标 | 定义 |
|---|---|
| Index completeness | `ready` 文档中在 doc registry、LightRAG KV、Qdrant、GraphML、EventRegistry 都存在预期记录的比例 |
| Index correctness | 抽样 Chunk 的正文、document_id、page/section 与 parsed 文件一致的比例 |
| Idempotence | 同一文档重建后，内容不变的实体/关系/事件稳定 ID 保持率，以及记录数量漂移 |
| Update propagation lag | 新版本入库到查询可召回的 p50/p95 时间 |
| Delete correctness | 删除后在所有存储中无孤儿记录的比例；错误删除率必须为 0 |
| Chunk boundary quality | gold 证据句被完整包含于至少一个 Chunk 的比例；跨 Chunk 断裂率 |
| Index throughput | docs/min、pages/min、tokens/s；同时报告 LLM 调用次数与成本 |
| Storage amplification | 索引字节数 / 规范化文本字节数，分 Qdrant、KV、Graph、Events |

### 3.3 实体抽取、实体对齐和关系图谱

“实体抽取”和“实体对齐”必须分开测。前者判断是否找到了提及，后者判断多个名称是否
属于同一真实实体。

| 子任务 | 指标 | 说明 |
|---|---|---|
| 实体提及识别 | Exact/relaxed span P/R/F1 | Exact 要求边界完全一致；relaxed 允许 span overlap |
| 实体类型 | Micro/macro F1 | 公司、产品、人物、机构、地点、技术等；macro-F1 防止大类掩盖小类 |
| 规范名称 | Canonical-name accuracy | gold cluster 的规范名称是否正确 |
| 候选生成 | Candidate Recall@K | 正确实体是否在 top-K 对齐候选中 |
| 排名式对齐 | Hits@1/5/10、MRR | 适合“给定提及，在候选实体中找对应实体” |
| 二分类对齐 | Pairwise precision/recall/F1 | 适合判断两个 mention/entity 是否相同 |
| 聚类式对齐 | B³、CEAF-e、CoNLL F1 | 适合整体 alias cluster；同时观察 singleton |
| 过合并/漏合并 | Over-merge、under-merge rate | 前者破坏精度，后者破坏召回；必须分别报 |
| 聚类质量 | Cluster purity、inverse purity | purity 偏向小簇，inverse purity 反映完整性，不能单独使用 |
| 长尾能力 | Hits@1/F1 by frequency/degree | 按出现频率、图节点 degree、名称歧义度分桶 |
| 关系抽取 | Triple exact/relaxed P/R/F1 | 主体、关系、客体全对才算 exact；relaxed 可使用已对齐实体 |
| 图结构 | Edge precision/recall、isolated-node rate | 评估边可靠性与实体覆盖 |
| 图检索 | Gold path recall@K | 多跳问题所需的实体/关系路径是否被图扩展召回 |
| 图一致性 | Constraint violation rate | 自环、类型不兼容、非法时间边、重复边等比例 |

实体对齐论文常报 Hits@K/MRR，但本项目还有聚类行为，因此不能只报 Hits@K。
Hits@K 会漏掉“错误地把 A、B、C 全并成一个实体”的严重过合并问题。

### 3.4 事件抽取、跨文档合并、时间和冲突

#### 事件抽取

| 指标 | 定义 |
|---|---|
| Event detection P/R/F1 | gold 事件是否被抽出；可按 trigger exact/relaxed 或结构化事件匹配 |
| Event-type micro/macro F1 | 合同、部署、合作、投资、诉讼、测试等类型分类 |
| Argument identification F1 | actors、objects、location、result、trigger 是否识别 |
| Argument-role F1 | 参数识别且角色正确才计分 |
| Action similarity / accuracy | 动作归一分类准确率；开放文本可用人工或语义评分辅助 |
| Evidence exact/overlap F1 | 证据短句 exact match、token F1、span IoU |
| Unsupported-event rate | 事件结构中无原文证据支持的比例；越低越好 |
| Event coverage | 含 gold 事件的文档中至少召回一个有效事件的比例 |
| Time exact/normalized accuracy | 日期字符串 exact；归一后按 year/month/day 精度分别计分 |
| Time precision accuracy | `unknown/year/month/day` 分类准确率 |
| Stance macro-F1 | affirmed/denied/uncertain 三分类；必须报 macro-F1 |
| Confidence calibration | Brier score、ECE、reliability diagram | 当前 `confidence` 是否可解释；不能只看平均值 |

#### 跨文档事件合并

| 指标 | 作用 |
|---|---|
| Pairwise P/R/F1 | 直接检查任意两个 event mentions 是否应合并 |
| MUC、B³、CEAF-e | 分别反映链接恢复、mention 聚类、簇对齐的不同偏差 |
| CoNLL F1 | MUC/B³/CEAF-e F1 的平均，作为主聚类指标 |
| Over-merge / under-merge rate | 明确衡量 current canonical key 和 `_equivalent` 的两类错误 |
| Canonical event accuracy | 合并后的 type、actor、object、time、summary 与 gold 是否一致 |
| Provenance retention | 合并后 gold mention/evidence/document_id 保留率，应接近 100% |
| Multi-source precision | 被标为 `multi_source` 的事件是否真由独立来源共同支持 |
| Re-index stability | 重建单文档前后无关 event cluster 不发生变化的比例 |

#### 时间关系

| 指标 | 定义 |
|---|---|
| Relation micro/macro F1 | precedes/same_time/related/unknown 分类 |
| Pair ordering accuracy | gold 可比较事件对的先后顺序正确率 |
| Timeline Kendall tau | 同一实体事件序列与 gold 排序的一致性 |
| Temporal coverage | 有可解析时间的 gold 事件被正确挂入时间线的比例 |
| Cycle/contradiction rate | `precedes` 图中的环和互相矛盾边比例，越低越好 |
| Interval overlap accuracy | 年/月/日等部分日期转换成区间后，过滤命中是否正确 |

#### 冲突发现与消解

当前实现主要处理同一 canonical event 下 affirmed/denied 的冲突，因此应同时测“冲突对”
和“事件最终状态”，并增加 `Not enough information`，避免把未知当否定。

| 指标 | 定义 |
|---|---|
| Conflict-pair P/R/F1 | 两条 mention 是否构成真实冲突 |
| Status macro-F1 | supported/denied/disputed/uncertain/NEI 的事件状态分类 |
| Evidence P/R/F1 | 支持或反驳结论所需证据是否全部找到 |
| Joint conflict score | 仅当状态正确且至少一套完整证据正确时得分，类似 FEVER score |
| False-conflict rate | 版本更新、数值差异、时间差异、口径差异被误判冲突的比例 |
| Missed-conflict rate | gold 冲突未被发现的比例 |
| Source-aware accuracy | 同源转载不被错误计作多源独立支持 |
| Temporal conflict accuracy | 同一对象不同时间状态变化不被当作同一时点矛盾 |
| Resolution calibration | resolved/disputed 置信度的 Brier/ECE |

### 3.5 查询理解与召回

检索指标应至少在 document、chunk、evidence 三个粒度计算。只看 Chunk 命中容易把一个
大段落的偶然命中误当作有效证据。

| 指标 | 定义/用途 |
|---|---|
| Precision@K | top-K 中相关项比例；控制上下文噪声 |
| Recall@K | gold 相关项进入 top-K 的比例；本项目首要召回指标 |
| Hit@K / Success@K | top-K 是否至少含一项 gold；适合单证据事实问答 |
| MRR | 第一个相关项的倒数排名均值；反映首个证据是否靠前 |
| MAP@K | 多个相关证据的平均准确率；适合多证据问题 |
| nDCG@K | 使用 0/1/2 分等级相关性时衡量整体排序 |
| R-precision | top-R 中相关比例，R 为 gold 证据数；适合 KILT 式 provenance |
| Complete evidence-set recall@K | 至少一套可完整回答问题的证据集合是否全部进入 top-K |
| Multi-hop path recall | gold 实体/关系/事件路径是否完整召回 |
| Context precision/recall | 实际送给 LLM 的上下文而非原始候选的相关性 |
| Query constraint accuracy | 年份、时间区间、实体、事件类型解析 exact/F1 |
| Filter recall | 时间/实体过滤后仍保留 gold 事件的比例 |
| Unanswerable detection | no-answer/insufficient-evidence 的 macro-F1、AUROC、拒答率 |
| Retrieval diversity | top-K 唯一文档数、唯一来源域数、独立来源数；不能代替 relevance |

必须构造以下问题桶：具体事实、实体关系、跨文档多跳、全局主题、时间线/变化、
冲突核验、来源定位、无答案。每桶分别报告，防止 global 问题被大量简单事实题稀释。

### 3.6 答案、忠实性和引用

| 维度 | 指标 | 建议实现 |
|---|---|---|
| 短答案正确性 | EM、token F1 | 人名、数值、日期、机构、产品等事实题 |
| 长答案正确性 | claim-level precision/recall/F1 | 将 gold 和答案拆成 atomic claims 后人工/判别器核对 |
| 覆盖完整性 | Gold-point coverage | gold 要点被回答的比例 |
| 答案相关性 | Answer relevance | 是否直接满足问题；ARES/RAGAS judge 可作辅助 |
| 忠实性 | Supported-claim rate | 答案中的可验证 claim 被检索上下文支持的比例 |
| 幻觉 | Hallucinated-claim rate | unsupported 或与证据矛盾的 claim 比例；与上项互补 |
| 冲突意识 | Conflict-aware answer accuracy | 是否同时表达支持/否定证据并避免武断结论 |
| 时间一致 | Temporal answer accuracy | 日期、顺序、当前/历史状态是否与 gold 一致 |
| 引用正确 | Citation entailment/correctness | 引用是否支持紧邻 claim；可分 full/partial/no support |
| 引用完整 | Citation completeness | 应有引用的 claims 中被有效引用覆盖的比例 |
| 引用定位 | Citation precision | 给出的引用中真正相关且支持 claim 的比例 |
| Provenance | Document/page/section exact accuracy | 引用是否指向正确文件、页码、章节 |
| 联合质量 | Grounded correctness / KILT-style score | 仅当答案正确且完整 gold provenance 被召回才得分 |
| 可读性 | 人工 1--5 或 blinded pairwise win rate | 只作次要指标，不覆盖事实错误 |
| 拒答 | Appropriate deflection rate | 无充分证据时正确拒答；同时报 answerable 问题误拒答率 |

ROUGE-L、BLEU、BERTScore 可作为辅助，但不应作为主指标。开放式答案措辞变化很大，
高文本相似不等于事实正确；LLM-as-a-judge 也不能独立作为唯一真值。

### 3.7 系统、成本、鲁棒性和可维护性

| 范围 | 指标 |
|---|---|
| 在线延迟 | API 端到端 p50/p95/p99；拆分 event search、retrieval、rerank、LLM、source mapping |
| 吞吐 | queries/min、concurrent request success rate |
| 成本 | 每文档索引 tokens/API 调用/人民币或美元；每次查询同样统计 |
| 资源 | 峰值 RSS、MPS 内存、CPU、Qdrant/Graph/KV 磁盘大小 |
| 稳定性 | 多次相同查询答案正确率方差、引用集合 Jaccard、事件 cluster 稳定性 |
| 可用性 | 成功率、超时率、降级到 base RAG 的比例、错误类型分布 |
| 增量更新 | 新文档索引耗时、旧文档重建影响范围、staleness |
| 规模伸缩 | 1x/5x/10x 语料下 Recall@K、p95 latency、成本和存储曲线 |
| 噪声鲁棒 | 加 10%/30%/50% distractor 后核心指标相对下降 |
| 冲突鲁棒 | 注入支持/否定、旧/新状态文档后 status 和答案变化是否正确 |
| 重复鲁棒 | 加转载、近重复后答案/置信度不应因伪多源支持虚高 |
| 改写鲁棒 | 同一问题多种中文/英文/缩写/paraphrase 的最差值与方差 |
| 故障恢复 | EventRegistry 损坏、Qdrant/LLM 暂不可用时的降级成功率和恢复时间 |

---

## 4. 各模块的基线设计

### 4.1 实体抽取与实体对齐

按由弱到强排列：

1. `Exact-normalized-string`：Unicode/case/punctuation 归一后完全相同才合并。
2. `Alias-dictionary`：人工别名表，例如公司简称、产品旧称、机构缩写。
3. `Fuzzy-string`：编辑距离/Jaccard 阈值。
4. `Embedding-NN`：实体名称+描述向量最近邻。
5. `Pairwise LLM/classifier`：对 blocking 后的困难候选做同实体二分类。
6. `Current LightRAG extraction/alignment`：当前实现。
7. `Oracle entity clusters`：人工实体簇，作为图检索上界。

公平性要求：所有方法使用同一提及集合时测“纯对齐”；再用各自预测提及测端到端对齐。

### 4.2 事件抽取与跨文档合并

1. `No-merge`：每个 mention 都是独立事件，衡量合并带来的收益与伤害。
2. `Exact canonical_key`：event type + 完全归一 actor/object/time bucket。
3. `Current rule`：当前 substring overlap + common time precision。
4. `Embedding agglomerative`：结构化事件描述向量 + 阈值聚类。
5. `Pairwise classifier/LLM + clustering`：困难候选对判断后传递闭包。
6. `Oracle event clusters`：用于检验下游时间链/冲突的上界。

当前规则的核心实验变量应包括 actor overlap、object overlap、time bucket、日期精度和
合并阈值。主要看 CoNLL F1，同时单列 over-merge/under-merge。

### 4.3 冲突识别

1. `Negation-keyword`：同一事件簇中存在否定词即冲突。
2. `Current stance rule`：affirmed + denied => disputed。
3. `NLI 3-way`：同一 claim 的 evidence pair 做 entailment/contradiction/neutral。
4. `Time-aware NLI`：先对齐实体、属性和时间，再判断矛盾。
5. `Oracle aligned events/evidence`：隔离冲突分类器自身能力。

必须把“不同时间的状态变化”“预测值与实际值”“引用另一方观点”“独立来源 vs 转载”
做成挑战集，它们是当前简单 stance 规则最容易失败的地方。

### 4.4 检索和生成

| 级别 | 方法 | 目的 |
|---|---|---|
| 下界 | No-RAG | 检查模型参数知识与数据泄漏 |
| 词法 | BM25 | 对实体名、编号、合同号、专有词通常很强 |
| 语义 | Dense-RAG | 当前向量通路的净效果 |
| 主基线 | BM25 + Dense + RRF | 强、简单、可复现 |
| 强主基线 | Hybrid + BGE reranker | 检查复杂图方法是否胜过重排 |
| 当前内部 | LightRAG naive/local/global/hybrid/mix | 检查模式路由与图谱价值 |
| 完整系统 | LightRAG mix + Event | 目标系统 |
| 长上下文 | 原文或 DOS/结构保持输入 | 检查小语料下直接阅读是否更好 |
| 检索上界 | Oracle-context | 区分 retrieval error 与 generation error |
| 生成上界 | Gold answer/人工分析员 | 只用于评分校准 |

如资源允许，再加入 RAPTOR、HippoRAG、Microsoft GraphRAG。外部系统必须固定同语料快照、
同一生成模型、同一最大上下文、相近 top-K 和相同硬件；否则比较主要反映预算差异。

---

## 5. 必做消融实验

采用 `Hybrid-RAG`、`LightRAG-base`、`Full` 三个主列，再做：

1. `Full - graph`：向量 + event，测事件层在无图时的净增益。
2. `Full - events`：即 LightRAG-base，测事件层增益。
3. `Full - time filter`：测时间解析与过滤贡献。
4. `Full - conflict context`：测冲突信息对答案的贡献。
5. `Full - reranker`：测重排的质量/延迟交换。
6. `Full - source marker`：只用于证明引用映射机制价值，不能用于生产答案。
7. `Current merge -> no merge/exact key/oracle merge`：隔离事件合并误差。
8. `Predicted entity -> oracle entity`：量化实体抽取/对齐造成的下游损失。
9. `Predicted evidence -> oracle evidence`：量化检索和生成之间的误差边界。
10. 不同 `top_k/chunk_top_k/event_limit/context budget` 的质量—延迟曲线。

对图谱增益的关键结论应来自多跳、全局、时间和冲突问题桶；在简单事实题上没有提升并不
必然说明图谱无效，但若强简单基线在所有桶都更好，则复杂层没有被实验支持。

---

## 6. 本项目金标准数据集设计

### 6.1 数据切分

不能随机拆散同一新闻转载、同一事件或同一网站的近重复文档，否则会造成泄漏。建议：

- 按 canonical event cluster/source family/time window 分组后切分。
- `dev` 用于阈值、top-K、prompt 和规则调优；`test` 只在方案冻结后运行。
- 另建 temporal holdout：测试集时间晚于开发集，模拟新资料增量进入。
- 固定 corpus snapshot、配置、模型版本、prompt hash 和随机种子。

### 6.2 MVP 标注规模

第一轮可在现有 234 篇 ready 文档上构造：

| 标注集 | 建议规模 | 内容 |
|---|---:|---|
| Parser gold | 每格式 15--20 篇/页 | 正文、标题层级、页码、章节、表格抽样 |
| Entity gold | 300--500 mentions，至少 100 clusters | span、type、canonical ID、alias cluster |
| Event gold | 300--500 mentions，至少 100 real events | type、actor、object、action、time、stance、evidence、cluster |
| Conflict challenge | 50--100 event/claim groups | 支持、反驳、未知、时间变化、转载、预测/实际 |
| QA gold | 120--200 questions | gold answer points、完整证据集、实体/事件路径、可回答性 |

QA 问题分布建议：事实 20%、实体关系 15%、多跳 15%、全局综合 15%、时间线 15%、
冲突核验 10%、来源定位 5%、无答案 5%--10%。正式报告同时给 micro 平均、macro by type 和最差桶。

### 6.3 标注协议

- 两名标注员独立标注，第三人裁决；测试集标注员不得看到系统输出。
- 类别任务报告 Cohen's kappa 或 Krippendorff alpha；span/证据报告双人 F1；排序可报 Kendall tau。
- 先做 20--30 条 pilot，修订手册，再冻结 schema。
- 允许一题有多套有效证据集；多跳题必须标出“完整证据集合”，不能只列单篇相关文档。
- 对冲突标注区分：真正矛盾、时间演化、范围/口径差异、预测与事实、转载重复、证据不足。
- LLM 可帮助预标注，但 gold 必须经人工确认；judge 模型不得与被测生成模型完全相同且未经校准。

建议的 QA 记录结构：

```json
{
  "question_id": "q_001",
  "question": "...",
  "type": "temporal_conflict",
  "answerable": true,
  "gold_answer_points": ["..."],
  "acceptable_answers": ["..."],
  "evidence_sets": [
    [{"document_id": "...", "page": 3, "quote": "..."}]
  ],
  "gold_entities": ["..."],
  "gold_events": ["..."],
  "gold_relations": ["..."],
  "notes": "..."
}
```

---

## 7. 统计与公平比较规则

1. 所有系统共用同一 corpus snapshot、问题集、生成模型版本和最大上下文 token 数。
2. 同时报告固定 `top_k` 和固定 token budget；后者更公平。
3. 固定 temperature；非确定性生成至少运行 3 次，报告均值、标准差和最差值。
4. 对每个问题保存完整 trace：候选、分数、过滤、最终 context、answer、citations、tokens、latency。
5. 主要指标给 95% paired bootstrap confidence interval；二元成败可用 McNemar test。
6. 多个系统/指标比较时报告 effect size，并进行多重比较校正，不能只报 p-value。
7. LLM pairwise judge 必须隐藏系统名、随机左右顺序、交换顺序复评，并用人工子集校准。
8. 同时报 absolute score 和相对基线增益；鲁棒性实验报相对下降率。
9. 不把 indexing cost 与 online query cost 混在一起；分别报告一次性和每次查询成本。
10. 失败样例按 taxonomy 归类并发布计数，而不是只展示成功案例。

建议失败类型：解析错误、来源丢失、实体漏抽、实体过合并、关系错误、事件幻觉、事件漏抽、
事件过合并、时间归一错误、冲突误判、召回遗漏、重排错误、context 截断、生成幻觉、引用错位、错误拒答。

---

## 8. 论文依据与可借鉴做法

以下均为论文或官方论文页面；本项目应借用其评测思想，而不是照搬不适合 OSINT 的单一总分。

1. [BEIR, NeurIPS Datasets and Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html)：跨域比较 BM25、稀疏、稠密、late-interaction 和 reranking；其结论支持把 BM25 作为稳健基线，并同时报告效果和计算代价。
2. [KILT, NAACL 2021](https://aclanthology.org/2021.naacl-main.200/)：同时评价下游答案和 provenance；KILT 分数只有在完整证据被取回时才奖励答案分，适合本项目构造 grounded correctness。
3. [ARES, NAACL 2024](https://aclanthology.org/2024.naacl-long.20/)：把 RAG 拆成 context relevance、answer faithfulness、answer relevance，并用少量人工样本校准自动 judge。
4. [RAGAS, EACL 2024](https://aclanthology.org/2024.eacl-demo.16/)：提供无参考的检索相关性、忠实性和答案质量指标；适合开发期回归，不适合替代人工 test gold。
5. [ALCE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)：从流畅度、正确性和引用质量评测带引用长答案；支持本项目单独测 citation correctness 与 completeness。
6. [RAGTruth, ACL 2024](https://aclanthology.org/2024.acl-long.585/)：对 RAG 输出做细粒度 hallucination 标注；支持 claim/span 级 hallucination rate，而非只判整段答案。
7. [FEVER, NAACL 2018](https://aclanthology.org/N18-1074/)：三分类 Supported/Refuted/NotEnoughInfo，同时要求证据；适合冲突状态、evidence F1 和联合 conflict score。
8. [SciFact, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.609/)：领域 claim 的支持/反驳、证据 abstract 和 rationale 联合评测；提醒必须在领域语料上校准。
9. [OpenEA, PVLDB 2020](https://www.vldb.org/pvldb/vol13/p2326-sun.pdf)：实体对齐的 Hits@K/MRR、数据异质性和长尾/度数分桶；适合实体候选排名，但本项目还需聚类指标补足过合并检查。
10. [MAVEN, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.129/)：将 event detection 分为 trigger identification 和 event type classification；对应本项目事件发现与类型 F1。
11. [MAVEN-ERE, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.60/)：统一事件 coreference、temporal、causal、subevent relation；支持对事件合并与时间边分任务评测。
12. [Cross-document Event Extraction / CLES, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.114/)：其管线包含事件抽取、coreference、实体/角色规范化，并用 P/R/F1 与 MUC/B³/CEAF/CoNLL F1，和本项目结构最接近。
13. [LightRAG, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.568/)：使用 comprehensiveness、diversity、empowerment、overall 的 pairwise LLM judge；这些可作为全局问题的辅助指标，但不应替代 evidence/correctness 的客观指标。
14. [HippoRAG, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)：图检索在多跳 QA 上同时比较质量、速度与成本；可作为后续复杂 graph baseline。
15. [RAPTOR, ICLR 2024](https://openreview.net/pdf?id=GN921JHCRw)：树状多层摘要检索，适合作为全局/长文综合问题的可选基线。
16. [Self-RAG, ICLR 2024](https://openreview.net/pdf?id=hSyW5go0v8)：按需检索和自我批判，适合未来增加 query routing 与检索必要性判断时参考。
17. [RAG vs Long Context, EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.66/)：提醒必须把 long-context 作为控制组，同时比较质量和成本。

---

## 9. 推荐实验表格

### 9.1 主结果

| System | Evidence R@10 | Set Recall | nDCG@10 | Answer F1/Correct | Faithfulness | Cite Correct | Cite Complete | p95 ms | Cost/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-RAG | - | - | - |  |  |  |  |  |  |
| BM25 |  |  |  |  |  |  |  |  |  |
| Dense |  |  |  |  |  |  |  |  |  |
| Hybrid-RAG |  |  |  |  |  |  |  |  |  |
| Hybrid + Reranker |  |  |  |  |  |  |  |  |  |
| LightRAG-base |  |  |  |  |  |  |  |  |  |
| Full |  |  |  |  |  |  |  |  |  |
| Oracle-context | 1.0 | 1.0 | - |  |  |  |  |  |  |

### 9.2 结构化模块

| System | Entity span F1 | Align Hits@1 | Align pair F1 | Event F1 | CoNLL F1 | Time F1 | Conflict macro-F1 | Conflict evidence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple rule baseline |  |  |  |  |  |  |  |  |
| Current |  |  |  |  |  |  |  |  |
| Strong classifier/LLM baseline |  |  |  |  |  |  |  |  |
| Oracle previous stage |  |  |  |  |  |  |  |  |

### 9.3 分桶与鲁棒性

按 question type、source type、document format、language、time recency、entity frequency、
single/multi-hop、conflict/no-conflict、answerable/unanswerable 报主指标。另报加入 distractor、
duplicate、contradictory update 后的相对下降。

---

## 10. 实施顺序

### Phase 0：冻结可复现条件

- 固定 corpus snapshot、配置、模型和 prompt 版本。
- 给查询链路增加 trace 导出，保留检索候选、最终上下文、tokens、latency、citations。
- 先实现 BM25、Dense、Hybrid-RRF 和 use_events on/off 的统一 runner。

### Phase 1：MVP gold 与客观指标

- 完成 120 个 QA、实体/事件小规模双人标注。
- 实现 evidence Recall@K、nDCG、MRR、完整证据集召回。
- 实现 answer correctness、claim faithfulness、citation correctness/completeness。
- 跑 `Hybrid-RAG`、`LightRAG-base`、`Full`、`Oracle-context`。

### Phase 2：结构化模块定位

- 实体：exact/alias/current/oracle。
- 事件：no-merge/exact/current/oracle，计算 pairwise 与 CoNLL F1。
- 冲突：keyword/current/NLI/oracle，计算 macro-F1、evidence F1、联合分。
- 把结构化误差与下游 QA 损失关联起来。

### Phase 3：压力、成本与外部系统

- distractor、duplicate、contradiction、temporal update、corpus scaling。
- 加 RAPTOR/HippoRAG/GraphRAG 等复杂基线。
- 用 paired CI 与消融结果形成论文主表和错误分析。

最终验收不建议设一个混合总分。应设多目标门槛，例如：Evidence Recall@10、citation
correctness/completeness、answer correctness 不低于主基线，至少一个目标问题桶显著提升，
且 p95 latency/单问成本在预先约定预算内。具体阈值应在 pilot 后冻结，不能看过 test 后再改。
