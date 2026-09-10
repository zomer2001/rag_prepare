# ELC-RAG 论文核心论证与实验规划报告（v2.1，方法冻结版）

> 日期：2026-09-04  
> 名称暂释：**ELC-RAG — Evidence Lifecycle Coordination for RAG**  
> 文档性质：研究问题与验证路线，不把尚未实现或尚未验证的内容写成论文成果。

## 1. 一句话版本

现实资料会同时保留一个事实过去怎么说、后来发生了什么、现在怎么说以及不同来源的意见。普通 RAG 可能搜到若干相关文本，却漏掉回答当前问题所必需的一环。ELC-RAG 利用检索阶段的事件—事实—来源连接，把局部命中的证据协调成可追溯的演化证据集。

## 2. 研究背景与真实问题

以供应商变化为例：

```text
报道 1：公司曾使用 A
公告 2：公司终止与 A 的合同
报道 3：公司随后使用 B
回应 4：公司否认“因质量问题解约”的说法
```

不同问题需要不同证据：

- “以前是谁？”只需要历史状态和来源；
- “为什么不再使用 A？”需要历史状态、变化事件或原因及来源，不一定需要 B；
- “后来换成了谁？”需要变化后的状态和来源，可能还需要连接事件；
- “是否因质量问题解约？”需要原始主张、否认/支持来源、时间和冲突状态；
- “供应商如何演变？”需要多段状态—事件序列。

因此，论文不再假设所有问题都有固定的 old-state—event—new-state 三段结构。真正的问题是：

> 给定一个关于演化事实的问题，检索器能否在有限预算内覆盖该问题所需的证据角色，并返回它们之间的时间、变化和来源连接？

## 3. 现有研究已经解决了什么

### 3.1 相似文本和普通多跳检索

Dense、BM25、Hybrid RAG 能找到文本上相关的片段；query decomposition、IRCoT、Adaptive-RAG 和 SR-RAG 能拆分问题、迭代检索或验证推理链。它们说明“Planner”“多轮检索”和“完整推理链”都不是空白。

仍需检验的问题是：对演化事实，逐片段相关性或普通子问题分解是否会系统性漏掉某种必要角色，以及专门的生命周期角色约束是否有额外收益。

### 3.2 图、事件和本体 RAG

- KG²RAG 已从语义种子沿 KG 扩展并组织 Chunk；
- EventRAG 已构建事件图并迭代检索跨来源事件关系；
- E²RAG 已连接实体图与事件图，处理时间、因果和角色一致性；
- OG-RAG 已使用本体落地的超图检索相关事实集合；
- HippoRAG/GraphRAG 等已证明图传播和图组织可帮助多跳 RAG。

所以“三路检索”“跨视图”“实体+事件”“图扩展”“ontology-grounded retrieval”都不能单独作为 ELC-RAG 的创新。

### 3.3 时间变化、过时信息和动态记忆

- TimeR⁴ 使用时间约束检索和重排已有 TKG；
- HoH 证明旧信息与当前信息共存会干扰 RAG；
- FRESCO 直接评测演化冲突下的 reranker；
- When Facts Change 研究参数知识与时间错位上下文的冲突；
- ATOM 从文本持续构建 TKG，并区分 observation time 和 valid time；
- Zep/Graphiti 在 agent memory 中保存双时态事实、历史关系和来源。

所以“保存历史事实”“双时态”“新旧冲突”“动态更新”也不是本项目独有。

### 3.4 溯源本体和状态变化表示

PROV-O 已定义 Entity 的生成、失效与 provenance；Nanopublication 区分 Assertion、Provenance 和 Publication Info；CIDOC-CRM 长期采用事件中心的历史知识表示；ProPara/OpenPI 已研究 before/after 状态变化。

这些工作否定了“Event—Assertion—Source 表示原语是新的”这一宽泛说法。ELC-RAG 只能把这些成熟思想作为表示基础，研究它们是否能支持一种不同的检索目标。

## 4. 经过收缩后的 GAP

最稳妥的 GAP 不是“没有人记录事件如何改变事实”，而是一个需要实验验证的观察：

> 现有检索通常按单条文本、节点或路径的相关性排序。对于演化事实问题，回答可能依赖一组承担不同角色的证据；每条证据都相关，不代表这组证据覆盖了该问题所需的全部角色。现有时间排序、普通图扩展、事件检索和通用问题分解是否足以获得这种角色覆盖，目前不能预设，需要在统一任务下直接比较。

这里不使用“首次”，也不把“完整”写成绝对概念。论文必须通过新标注和 killer baselines 证明这个 GAP 真实存在。

## 5. 操作性任务定义

### 5.1 问题所需角色

给定问题 `q`，定义必需证据角色集合 `R(q)`。角色随问题变化，可包括：

- historical_state；
- current_or_later_state；
- change_event；
- change_reason；
- effective_time；
- supporting_source；
- contradicting_or_retracting_source；
- intermediate_state_or_event。

角色集合需要来自人工标注规范；模型可在运行时预测，但评价不能由同一个模型自定义 Gold。

### 5.2 Role-complete evidence

检索结果 `E` 相对于问题 `q` 是 role-complete，当且仅当：

1. `R(q)` 中每个必需角色至少有一条 Gold 支持证据被检索；
2. 每条角色证据能够回到具体来源片段；
3. 若 Gold 要求角色间连接，结果中存在正确的时间、transition、succession 或 stance 连接。

一个问题可以存在多个可接受证据集合。最小性只作为预算优化目标，不强制 Gold 唯一。

### 5.3 核心指标

- Lifecycle Role Recall@K；
- Role-Complete Retrieval@K；
- Source Traceability@K；
- Connected Evidence Rate@K；
- Missing-transition Rate；
- Stale-only Retrieval Rate；
- Stance-as-truth Error；
- 标准 Evidence Recall、nDCG、MRR、答案和引用指标。

## 6. 表示模型：作为方法基础，不作为独立 novelty

当前实现可以保持 Entity、Assertion、Event、Mention、SourceDocument 和六类 EventEffect，不必为了当前论文先重构图模型。写作上把它称为“面向演化事实检索的轻量生命周期证据表示”，不声称发明了全新的本体范式。

为了解释语义边界，论文可以用下列概念分层说明现有字段，但不要写成已经完成的重构：

```text
Assertion(subject, predicate, object, claimed_valid_time)

Event --creates/modifies/terminates--> Assertion

Assertion_new --replaces/supersedes--> Assertion_old

Mention --supports/contradicts/retracts/uncertain--> Assertion

Mention --from--> SourceDocument
```

同时区分：

- valid time：来源声称事实何时有效；
- publication/observation time：来源何时发布或观察；
- recorded/transaction time：系统何时接收该记录。

当前六类边可用于连接和证据组织，但还没有通过专门实验证明每一类语义都能被稳定抽取、且对检索都有额外价值。因此：

- 可以说它们是检索的结构支撑；
- 不能说六类边本身已被证明是新的本体理论；
- 不把某来源的 `denies` 直接写成客观事实已被证伪；
- 若未来要主张状态演化语义的正确性，再单独做抽取准确率和 typed/untyped 消融。

OWL/SHACL、PROV-O、OWL-Time 等仅用于结构约束与互操作。除非规则本身通过消融带来可验证收益，论文不把“提出新本体”列为贡献。

## 7. 当前方法与可验证假设

当前主方法 EvoLink 的真实流程是：

```text
问题 q
  ↓
解析查询中的关键词、实体、关系、时间与事件线索
  ↓
从文本/关系/事件索引取得种子证据
  ↓
沿 Entity–Assertion–Event–Mention–Source 连接做有界扩展
  ↓
将直接相关证据与关联的变化证据、后续/前置事件和来源一起返回
```

核心科学假设是：

> 在演化事实问题上，查询期利用事件—事实—来源连接补充局部命中，能否比单条相关性、时间排序和普通图扩展更好地覆盖查询所需的生命周期证据角色？

这个表述不假设当前 EvoLink 已经显式计算“还缺哪个角色”。当前 EvoLink 是多路种子后的 typed graph expansion；“按缺失角色约束路径和停止”是可选的后续增强，不是当前论文成立的前置条件。

EvoNeed 不作为第三项贡献。原版 `ours_evoneed` 的第一轮仍以完整 EvoLink 取回宽证据池，再评估需求覆盖并重排；在限制为一轮时，它实质上是“检索后需求评估+证据重排”。

新增对照变体 `ours_evoneed_planned` 会在首轮直接执行 Planner 生成的最多 3 个 `route + subquery + budget` 动作，动作并行执行，仍只计 1 轮检索。它复用已有 EvoLink 索引，但尚未有实验结果。它用来回答“首轮先规划是否比先宽搜再重排更好/更省”，不单独声称 Planner novelty。

## 8. 当前实现和结果能说明什么

当前仓库已经有：

- 文本、关系和事件索引；
- Entity、Assertion、Event、Mention 和 EventEffect 结构；
- 有效区间、来源时间、系统记录时间等字段；
- EvoLink 图扩展和来源回链；
- EvoNeed 证据需求预测、检索后评估和排序原型；
- 复用同一索引的 `ours_evoneed_planned` 首轮规划检索对照版（已实现、尚未评测）；
- MultiHop-RAG 完整 609 文档上的 q100 开发实验。

q100 中，EvoLink 相比无链接版本：Evidence Recall@20 从 0.8736 到 0.8828，Complete Recall@20 从 0.7143 到 0.7253。单轮 EvoNeed 保持相同 Recall，但 nDCG 从 0.6668 到 0.7196、MRR 从 0.7055 到 0.8024，同时 P95 延迟从约 1.16 秒增至 12.34 秒。

规范解读：

- 结果证明系统能够运行，并在一般多文档检索中出现小幅链接收益；
- EvoNeed 当前更像高成本证据重排器；
- MultiHop-RAG 没有直接标注 lifecycle role，不能用这些数字证明 EventEffect 或 lifecycle closure 有效；
- 不同缓存/运行顺序下的延迟不能直接作为正式效率比较。

## 9. 必须先做的证伪实验

### 9.1 表示必要性

| 方法 | 要检验的问题 |
|---|---|
| Bitemporal/temporal adjacency only | 只按 subject/predicate 和有效区间连接是否已经足够 |
| Untyped transition | 有事件—事实边但无类型是否足够 |
| Typed transition | creates/modifies/terminates 是否增加价值 |
| Randomized transition | 任意扩边是否也能取得相似增益 |
| Gold transition | 检索算法在完美抽取下的上限 |
| Predicted transition | 当前抽取误差造成多大损失 |

若 temporal-only 与 typed transition 基本相同，则不再以事件作用为核心；论文可以收缩为双时态证据检索。

### 9.2 普通方法是否已经足够

- 强 Hybrid + reranker；
- LLM question decomposition + Hybrid；
- LLM decomposition + ordinary KG expansion；
- KG²RAG；
- EventRAG；
- E²RAG；
- Graphiti-style bitemporal invalidation；
- 当前无约束 EvoLink；
- role-constrained ELC-RAG。

若普通分解或无约束扩图达到同样 role coverage，则专门生命周期结构没有成立。

### 9.3 结构是否会误导生成

冻结完全相同的正文，对比：

1. 无结构标签；
2. 正确的时间、transition、succession、stance；
3. 打乱标签；
4. 将 source denial 与 fact invalidation 故意构造冲突。

除了答案和引用，还要测 Stance-as-truth Error 和 Unsupported Conflict Resolution。

## 10. 数据与标注计划

### 10.1 现有数据集只做分工

- MultiHop-RAG：验证普通跨文档证据检索，不证明 lifecycle；
- E²RAG ChronoQA：验证叙事时间、事件和因果，论文必须明确它是小说语料；
- 中文新闻 ChronoQA：验证真实新闻时间推理，与英文同名数据集分开引用；
- HoH：验证新旧信息共存的干扰；
- FRESCO：验证演化语义冲突下的排序；
- When Facts Change/WIKIRECENTCHANGES：主要验证生成模型面对时间知识冲突的行为。

### 10.2 直接评测集

核心评测必须来自真实新闻、公告或真实页面历史，不构造虚假业务事实。建议标注 200–500 个问题，至少包含：

- question 和 answer；
- question type；
- required roles；
- 每个角色的一条或多条 Gold source spans；
- claimed valid time、publication time；
- transition、assertion succession、source stance；
- 一个或多个可接受证据集合；
- 是否允许 unknown/no-post-state。

至少双人标注，报告角色、transition 类型和连接的 Cohen's kappa 或 Krippendorff's alpha，并公开操作性标注指南。不能由当前抽取模型生成标签后再用同一标签证明自己。

## 11. 收缩为一个创新点

论文只围绕一个方法贡献：

> **EvoLink：面向演化事实的查询期生命周期证据协调。** 它使用事件—事实—来源连接，从局部命中出发补回有关的变化上下文，目标是减少“每条都相关，但整体缺一环”的检索结果，并保留来源回链。

其他部分都降为支撑项：

- EventEffect 是 EvoLink 可扩展的表示基础，不单独宣称本体原创性；
- lifecycle-role completeness 是问题定义和评测视角，不把它写成当前算法已经严格执行的停止规则；
- EvoNeed 是检验“查询需求规划能否改善 EvoLink”的扩展实验，不是第三个创新点；
- 新标注和指标属于验证该主张的必要实验设计。

## 12. 失败时如何收缩

- typed≈untyped：不强调 transition 类型；
- temporal-only≈typed：改为双时态证据检索；
- predicted 远低于 Gold：将主问题转向可靠 transition extraction，或只使用高置信边；
- 标签不改善生成：只主张内部检索和来源组织价值；
- Planner 成本过高：换规则/小分类器，不保留 agentic 叙事；
- 只在一般 MultiHop 有效：改为跨视图证据协调，不再使用 lifecycle 主张。

## 13. 推荐写法

### Problem

> Retrieval relevance is usually scored at the level of individual passages, whereas questions about evolving facts may require a connected set of source-grounded evidence playing different temporal and transition roles. We study retrieval under query-specific lifecycle-role requirements.

### Method

> ELC-RAG uses source-grounded event–fact links at retrieval time to coordinate locally retrieved text, relational, and event evidence, expanding toward connected evidence about how a fact evolved.

### Benefit

> The intended benefit is not automatic truth adjudication, but higher coverage of the evidence roles needed for a question, fewer stale-only or missing-transition result sets, and source-traceable context for downstream generation.

### 当前不能写

> We introduce the first Event-Effect Ontology for full-lifecycle reasoning.

> Our default EvoLink planner adaptively selects first-round retrieval routes according to missing lifecycle roles.

> MultiHop-RAG results demonstrate lifecycle reasoning.

## 14. 参考工作

- [Zep/Graphiti](https://arxiv.org/abs/2501.13956)
- [PROV-O](https://www.w3.org/TR/prov-o/)
- [CIDOC-CRM](https://cidoc-crm.org/sites/default/files/Documents/cidoc_crm_version_7.1.3.html)
- [Nanopublications](https://nanopub.net/)
- [ProPara](https://aclanthology.org/N18-1144/)
- [OpenPI](https://aclanthology.org/2020.emnlp-main.520/)
- [EventRAG](https://aclanthology.org/2025.acl-long.830/)
- [E²RAG](https://aclanthology.org/2026.eacl-long.90/)
- [KG²RAG](https://aclanthology.org/2025.naacl-long.449/)
- [OG-RAG](https://aclanthology.org/2025.emnlp-main.1674/)
- [TimeR⁴](https://aclanthology.org/2024.emnlp-main.394/)
- [HoH](https://aclanthology.org/2025.acl-long.301/)
- [ATOM](https://aclanthology.org/2026.findings-eacl.49/)
- [SR-RAG](https://aclanthology.org/2026.findings-acl.1922/)
- [FRESCO（预印本）](https://arxiv.org/abs/2604.14227)
- [When Facts Change](https://aclanthology.org/2026.findings-acl.103/)

## 15. 当前判断

旧版“Event-Effect Ontology + EvoLink + EvoNeed”三贡献叙事站不住，应收缩为一个问题：对演化事实，检索到的文本可能都相关，却在变化证据上缺一环。唯一方法创新是 EvoLink 在查询期利用 EventEffect 与来源连接协调这些证据。

当前方法不需要因此重构，但论文必须补上能直接验证“相关但缺环”的数据、标注、指标和对照。若 EvoLink 在这些实验中不能提高生命周期角色覆盖，就只能收缩为一般的跨视图检索改进；若能提高，论文的核心也仍然是“查询期演化证据协调”，而不是“本体里多了几种边”。
