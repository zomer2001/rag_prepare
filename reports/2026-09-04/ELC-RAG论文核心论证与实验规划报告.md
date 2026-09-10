# ELC-RAG 论文核心论证与实验规划报告

> **版本说明：本文件是 v1，已被审稿复盘发现存在贡献边界过宽、EventEffect 语义混层和“完整证据”定义不严谨等问题，不再作为论文依据。请使用 [v2 修订版](./ELC-RAG论文核心论证与实验规划报告_v2.md)。**

> 日期：2026-09-04  
> 方法暂定名：**ELC-RAG**（Evidence Lifecycle Coordination for Retrieval-Augmented Generation）  
> 当前状态：系统主体和小规模检索实验已经完成；论文核心主张仍需消融、外部基线、演化数据集和端到端实验验证。

## 0. 结论先行

ELC-RAG 最适合研究的不是“如何替大模型判断当前真相”，也不应笼统写成“动态本体 RAG”。它真正解决的问题是：

> 当同一事实随着事件不断变化、旧报道与新报道长期共存，而且变化过程分散在不同来源中时，如何把“变化前的事实—改变它的事件—变化后的事实—每一步的来源”作为一条完整、连接且可追溯的证据链检索回来。

论文可以形成“一个问题、两个核心贡献、一个条件性贡献”：

1. **Event-Effect Ontology（事件作用本体）**：表示事件如何创建、更新、终止、替代、否定或确认一条来源可追溯的事实。
2. **EvoLink 跨视图证据检索**：连接文本、事实关系和事件三个视图，从局部命中扩展出完整演化证据。
3. **EvoNeed 生命周期需求规划器**：在检索前判断问题需要哪些证据角色，并一次并行分配检索路线和预算。该项目前只有原型，尚未完成“检索前控制路线”的实现，因此暂不能作为已完成贡献。

最关键的论文收益不是“图更复杂”，而是：

- 减少只找到旧事实、只找到新事实或漏掉变化原因的情况；
- 告诉生成模型每条证据在变化过程中扮演什么角色；
- 从已找到的局部证据判断还缺少哪种证据；
- 保留原始来源，使答案可以引用、追踪和人工审计。

当前小规模结果显示跨视图连接可能提高证据完整性，EvoNeed 可能改善证据排序，但证据强度还不足以支持顶会论文结论。

---

## 1. 研究背景

RAG 通常把文档切成片段，再根据问题与片段的语义相似度返回 Top-K。它适合回答静态事实，但真实知识库经常同时保留历史与当前内容，例如：

1. 2022 年报道“公司使用 A 作为供应商”；
2. 2023 年公告“公司终止与 A 的合同”；
3. 2024 年报道“公司改用 B 作为供应商”。

当用户询问“公司为什么不再使用 A，现在换成了谁”时，完整答案需要三类证据：旧状态、改变旧状态的事件和新状态。它们往往来自不同文档，措辞也不相同。

普通向量检索只知道文本是否与问题相似，并不知道某篇报道属于“变化前”“变化事件”还是“变化后”。结果可能是：

- 只返回与“A 供应商”最相似的旧报道；
- 返回旧状态和新状态，却漏掉说明原因的解约事件；
- 返回几篇相关报道，但没有表达它们之间的变化关系；
- 把互相冲突的新旧说法一起交给模型，让模型自行猜测顺序和关系。

这类问题常见于供应商与合同变化、公司高管更替、政策修订、产品版本、漏洞修复、诉讼进展和组织关系演变等场景。它们共有五个特点：

1. 同一关系会随时间改变；
2. 旧文档不会在新文档出现后自动消失；
3. 旧状态、变化事件和新状态可能分散在不同来源；
4. 不同来源使用不同措辞描述同一次变化；
5. 来源可能确认、否认或纠正同一说法。

因此，这不是单纯的“最新文档排序”，而是**演化事实的完整证据检索问题**。

---

## 2. 研究现状：别人已经完成了什么

### 2.1 文本检索与普通 RAG

Dense、BM25、Hybrid RAG 已经能够从大规模语料中找到关键词或语义相近的片段。问题在于它们主要优化单个片段的相关性，不显式表示多个片段在事实变化中的角色。

问题分解研究也已证明，把复杂问题拆成子问题后分别检索，可以提高多跳证据覆盖。因此，“使用 LLM Planner”本身不是新贡献；ELC-RAG 若保留 Planner，必须证明它规划的是普通方法没有处理的**生命周期证据角色**，并且真实控制检索路线，而不是只改写关键词。

### 2.2 知识图谱与图引导 RAG

GraphRAG、LightRAG、HippoRAG 等方法通过实体、关系、路径或图传播连接跨文档信息。[KG²RAG](https://aclanthology.org/2025.naacl-long.449/) 还会从语义种子片段出发，沿知识图谱扩展并组织相关片段。

这类工作已经证明“片段之间的关系”能够改善多跳检索，因此 ELC-RAG 不能只声称“我们把文本连成了图”。必须证明普通实体/关系连接还不够，而 **EventEffect 的类型和事实生命周期关系提供了额外价值**。

### 2.3 事件图 RAG

[EventRAG](https://aclanthology.org/2025.acl-long.830/) 从多来源文本构建事件知识图谱，并通过迭代检索捕捉事件之间的时间与逻辑关系。[E²RAG](https://aclanthology.org/2026.eacl-long.90/) 分开构建实体图和事件图，再通过映射连接二者，以保留叙事中的时间、因果和角色一致性。

这说明“事件图”“实体—事件双图”和“跨事件推理”都已有直接相关工作。ELC-RAG 与它们的潜在区别不能写成“我们也抽取事件”，而应落在：

- 事件不仅与实体或其他事件相连，还明确指向它所影响的事实 Assertion；
- EventEffect 区分 creates、updates、terminates、supersedes、denies、confirms；
- 每条事实和事件都能回到具体来源 Mention；
- 检索目标是找齐旧状态—事件—新状态—来源，而不只是找到一组相关事件。

这些区别目前在代码结构中存在，但还需要实验验证是否真正优于普通事件图。

### 2.4 时间、版本与过时知识 RAG

[TimeR⁴](https://aclanthology.org/2024.emnlp-main.394/) 面向已有时序知识图谱，根据问题中的时间约束进行检索、重写和重排。[HoH](https://aclanthology.org/2025.acl-long.301/) 证明了旧信息与当前信息共存时会明显干扰 RAG，并提供了动态事实及过时证据评测。文档版本类 RAG 则关注新旧版本之间的差异与当前性。

这些研究已经处理“时间”“版本”和“过时知识”，所以不能宣称 ELC-RAG 首次研究事实变化。ELC-RAG 更具体的研究对象是：**哪一个事件改变了哪一条有来源依据的事实，以及如何利用这种关系补齐演化证据。**

### 2.5 本体与 RAG

[OG-RAG](https://aclanthology.org/2025.emnlp-main.1674/) 使用领域本体对文档事实进行组织，并从本体落地的超图中检索最小相关知识集合。这证明“本体参与 RAG 检索”已有强相关工作。

本项目当前已经存在静态本体结构契约，定义 Entity、Mention、Assertion、Event、EventEffect、ConflictCase，并通过 SHACL 约束、术语映射以及 OWL-Time、PROV-O、SKOS 等词表对齐。因而将表示层称为 **Event-Effect Ontology** 是可以论证的，但需补充正式推理规则、规则正确性和本体消融；否则审稿人仍可能认为它只是带类型的图模式。

需要特别区分：

- **静态本体、动态实例**：概念和关系定义相对稳定，新的事件和事实持续进入实例图；这是当前系统最准确的定位。
- **动态本体演化**：概念、关系类型和约束本身会自动增加或修改；当前系统没有完成这一能力，也不建议在本论文中主张。

---

## 3. 研究 GAP：用人话说明现有方法哪里不够

现有方法已经能找到相似文章、连接实体、组织事件、利用时间戳或识别旧版本。但在上述持续变化、多来源共存的场景中，仍存在四个相互关联的缺口。

### GAP 1：知道发生了事件，不等于知道它改变了哪条事实

事件图可以记录“公司解约”，关系图可以记录“公司—供应商—A”，但系统未必知道前一个节点正是在终止后一条关系。缺少这条连接时，检索器很难从解约报道稳定回溯到旧供应商事实，也难继续找到替代后的新事实。

### GAP 2：检索到相关内容，不等于找齐变化过程

Top-K 相关性只要求每个结果与问题相近，不要求结果集合同时覆盖旧状态、变化事件、新状态和来源。于是分数可能很高，但回答需要的关键一环仍然缺失。

### GAP 3：知道新旧，不等于知道为什么变化

时间戳或文档版本可以说明哪篇更早、哪篇更新，却不一定表达“解约事件导致旧供应商关系终止”。对于“为什么变化”“经历了什么过程”这类问题，仅做最新优先排序不够。

### GAP 4：Planner 会拆问题，但通常不围绕演化证据角色规划

现有 Planner 可以判断问题复杂度、拆子问题或进行多轮检索。ELC-RAG 需要解决的更窄：在第一次检索前判断问题需要 state_before、change_event、effective_time、state_after、stance_status、supporting_source 中的哪些角色，再为它们分配文本、关系图和事件图路线。

因此，论文的 GAP 可以写成：

> 现有 RAG 能检索与问题相关的文本、关系或事件，但当一个事实在多个来源中持续变化时，它们通常没有把“变化前的事实、改变它的事件、变化后的事实以及各自来源”作为一个整体检索目标。结果是相关内容可能被找到，变化过程却仍然缺环。我们研究如何显式记录“事件改变了哪条事实”，并利用这层关系补齐可追溯的演化证据。

这一表述不否定已有时间 RAG、事件 RAG 或本体 RAG，只声明它们尚未共同完成的具体任务。

---

## 4. 研究动机：记录状态究竟有什么好处

ELC-RAG 记录的不是一个脱离来源的“绝对真相”，而是来源支持下的事实状态和变化关系，包括：

- Assertion 的有效时间和 active、inactive、disputed、superseded、retracted 等状态；
- Event 对 Assertion 的 creates、updates、terminates、supersedes、denies、confirms 作用；
- 来源中的 Mention、立场、时间和原文位置。

这些记录对 RAG 有五个直接好处。

### 4.1 帮助找齐证据

搜到解约事件后，可以沿 terminates 找到旧关系；搜到旧关系后，可以沿 EventEffect 找到改变它的事件和后续关系。检索从“碰巧命中几篇相似文章”变成“按缺失角色补齐证据集合”。

### 4.2 减少新旧信息被当成无关冲突

系统不需要替模型决定最终真相，但可以明确告诉模型某条证据属于变化前、变化事件或变化后。模型不再只面对几段互相矛盾、没有关系说明的正文。

### 4.3 支持不同时间视角的问题

同一知识结构可以分别支持“以前是谁”“后来是谁”“为什么更换”“变化过程是什么”“某个来源当时怎么说”等问题，而不必为每种问法单独建索引。

### 4.4 支持可解释的证据补全

若系统只有旧状态和新状态，本体规则可以发现 change_event 缺失；若已有事件但没有 supporting_source，可以回到 Mention 和文档寻找出处。这种补全不是凭空推断答案，而是判断证据链缺少哪个位置并继续检索。

### 4.5 支持引用、审计和错误定位

每个状态和变化关系都保留来源。答案出错时，可以判断问题来自抽取错误、EventEffect 连错、缺少某段证据、检索排序错误，还是生成模型误读，而不是把所有失败混在一个黑盒中。

概括起来，状态记录的价值不是“替 LLM 做结论”，而是让检索器知道：**已经找到了什么、它在变化过程中的角色是什么、还缺什么、从哪里能够继续找。**

---

## 5. ELC-RAG 的拟议创新点

### 创新 1：面向来源证据的 Event-Effect Ontology

表示层将 Entity、Assertion、Event、EventEffect 和 Mention 分开：

```text
来源 Mention ──支持/否认──> Assertion
Event ──creates/updates/terminates/...──> Assertion
Assertion ──subject/predicate/object──> Entity
```

它描述的是**事实生命周期由事件驱动**，不是“事件自身的生命周期”。本体概念和约束保持稳定，事实、事件及其状态持续写入实例图。

计划补充的形式化内容：

- 类、属性、关系和基数约束；
- EventEffect 合法性规则；
- 旧状态—变化事件—新状态链的推导规则；
- 缺失角色检测规则；
- 与无类型 effect、普通 related-to 边的对照。

### 创新 2：EvoLink 跨视图生命周期证据协调

系统保留三种检索视图：

1. 文本向量视图：原文、数字、措辞和出处；
2. 实体—Assertion 关系视图：实体和事实关系；
3. 事件视图：事件、时间、参与者和事件关系。

EvoLink 不替代三路检索，而是在查询时使用共享实体、Assertion、EventEffect 和 Mention 回链协调三路候选，将局部命中扩展成来源可追溯的证据图。最终仍返回原文片段给生成模型。

其核心区别不是“用了三路”，而是：**利用有类型的事件—事实作用关系，为结果集合补入演化链上缺失的证据。**

### 创新 3（待完成）：EvoNeed 生命周期需求规划

目标流程应为：

```text
问题
  ↓
识别所需证据角色
  ↓
为每个角色生成子查询、路线和预算
  ↓
一次并行执行文本/关系/事件检索
  ↓
EvoLink 扩展、去重、排序
```

例如“为什么不再使用 A，现在换成了谁”应一次生成：

| 证据需求 | 检索目标 | 路线 |
|---|---|---|
| state_before | A 原来与公司的关系 | graph + vector |
| change_event | 什么事件终止该关系 | event |
| state_after | 谁替代了 A | graph + vector |
| supporting_source | 每段证据的原文出处 | provenance 回链 |

为避免 Planner 漏判，可保留低预算通用向量检索作为保险。所有动作并行执行，仍属于一轮检索。

当前实现的单轮 EvoNeed 会先规划，但第一轮仍调用完整 EvoLink 三路检索，规划结果尚未决定首轮路线和预算。检索后它会评估需求覆盖、整理证据并回答。因此当前结果只能支持“需求评估可能改善排序”，不能支持“需求规划改善首轮召回”。

---

## 6. 当前具体实现

### 6.1 索引阶段

1. 文档按统一策略切分并建立文本向量索引；
2. 抽取实体和关系，构建 Entity—Assertion—Mention 结构；
3. 抽取 Event、事件时间、参与者、事件关系和 Fact Effect；
4. 通过治理与融合模块解析实体、规范化 Assertion，并生成 EventEffect；
5. 保存有效时间、状态、来源立场和原文 Mention；
6. 同步到 Neo4j 协调图，并保留向量/关系/事件各自的原生索引。

EventEffect 当前支持：`creates`、`updates`、`terminates`、`supersedes`、`denies`、`confirms`。

### 6.2 普通 ELC-RAG 查询阶段

1. Query Planner 抽取关键词、实体、关系、事件类型、显式时间和 change/latest/conflict 等意图；
2. 文本、关系和事件三路分别检索；
3. EvoLink 使用实体、Assertion 和 Event 种子扩展协调图；
4. 通过 EventEffect 找到关联事实，通过 Mention 回到来源片段；
5. 合并、去重和排序，在预算内返回原始证据。

当前普通 Planner 只提供软提示，三个通道仍固定执行，因此它不是独立创新。

### 6.3 EvoNeed 单轮原型

1. LLM 将问题分解为 text_evidence、relation_evidence、evolution_evidence；
2. evolution_evidence 可包含 state_before、change_event、effective_time、state_after、stance_status、supporting_source；
3. 执行一次完整 EvoLink 检索；
4. 评估证据是否覆盖需求，重组证据并生成带引用答案。

下一步需要把第 1 步的需求真正转换成首轮检索动作，取代当前固定三路执行。

---

## 7. 当前指标及其含义

### 7.1 MultiHop-RAG q100，完整 609 文档语料

| 方法 | Evidence Recall@20 | Complete Recall@20 | nDCG@20 | MRR@20 | P95 延迟 |
|---|---:|---:|---:|---:|---:|
| ELC-RAG w/o link（代码 `ours`） | 0.8736 | 0.7143 | 0.6608 | 0.7052 | 2596 ms |
| ELC-RAG / EvoLink-lite | **0.8828** | **0.7253** | **0.6668** | **0.7055** | 1160 ms |
| 单轮 EvoNeed + 同一 EvoLink | **0.8828** | **0.7253** | **0.7196** | **0.8024** | 12340 ms |

目前可以谨慎得出三点：

1. EvoLink 相比无链接三视图，Evidence Recall@20 提升 0.92 个百分点，Complete Recall@20 提升 1.10 个百分点，说明跨视图连接存在正向信号，但幅度不大，尚缺显著性检验和外部基线。
2. 单轮 EvoNeed 的 Recall 与 EvoLink 完全相同，但 nDCG 和 MRR 更高，说明它目前主要改善已有证据的排序/组织，没有找回更多 Gold Evidence。这与当前“先完整检索、后评估”的实现一致。
3. EvoNeed P95 延迟约为 EvoLink-lite 的 10.6 倍。即使排序收益成立，也必须通过首轮路线选择、并行执行、缓存或更轻量规划降低成本。

延迟来自不同运行阶段和缓存条件，目前不能作为严格效率排名。正式实验需要独立进程、预热、重复运行并报告均值、标准差、P50/P95、模型调用和 token。

当前结果还不能证明：

- EventEffect 类型本身带来提升；
- 本体规则优于普通图扩展；
- 状态元数据改善最终答案；
- 对 EventRAG、E²RAG、HippoRAG 等外部方法有优势；
- 效果能泛化到真正包含事实变化的数据。

---

## 8. 实验计划与需要回答的研究问题

### RQ1：跨视图连接是否比三路简单融合更好？

固定文档、Chunk、Embedding、Top-K、上下文 token 和生成模型，对比：

1. BM25；
2. Dense；
3. Dense + BM25 + reranker；
4. LightRAG；
5. GraphRAG local；
6. HippoRAG 2；
7. EventRAG；
8. E²RAG；
9. ELC-RAG w/o link；
10. ELC-RAG。

KG²RAG 是方法上很接近的图扩展基线，目前按计划暂缓，但投稿前若资源允许应补上，否则需要在论文中解释未比较原因。

指标：Recall@5/10/20、Complete Recall@K、nDCG、MRR、Precision、上下文 token、查询延迟和费用。对逐问题结果做 paired bootstrap 95% CI。

### RQ2：提升来自普通连接，还是 EventEffect 生命周期？

核心消融：

| 版本 | 目的 |
|---|---|
| Text only | 文本下界 |
| Text + Assertion | 关系视图贡献 |
| Text + Event | 事件视图贡献 |
| 三视图独立 RRF | 多一路是否已经足够 |
| Entity-link only | 仅共享实体是否足够 |
| Untyped EventEffect | 有连接但不区分作用类型 |
| Full EventEffect | 完整方法 |
| Randomized Effect | 排除任意扩图都有效 |
| w/o Mention→source | 验证来源回链价值 |

如果 Full EventEffect 不优于 Untyped Effect，就不能声称类型提高检索，应把类型价值转到答案理解、审计或后续补全实验。

### RQ3：记录的状态是否真的帮助生成模型？

冻结完全相同的检索正文，只改变交给生成模型的表示：

| 版本 | 上下文内容 |
|---|---|
| B0 | 纯正文 |
| B1 | 正文 + 来源 ID |
| B2 | B1 + 时间、有效期、状态和来源立场 |
| B3 | B2 + typed EventEffect + 前态—事件—后态连接 |
| B4 | B3，但打乱时间或 Effect 类型 |

指标：Answer EM/F1、Citation Precision/Recall、主张支持率、时间顺序准确率、变化原因准确率、旧事实误用率。B3 优于 B2 且 B4 下降，才能证明模型确实使用了状态结构。

### RQ4：生命周期需求 Planner 是否值得成为贡献？

首先完成检索前规划，然后比较：

1. 固定执行全部三路；
2. 普通问题分解后分别检索；
3. EvoNeed 按生命周期角色规划路线和预算；
4. EvoNeed + 低预算向量保险；
5. 当前“完整检索后评估”版本。

必须同时报告召回、完整率、排序、调用次数、token 和延迟。Planner 的成立条件可以是：

- 同等成本下，证据完整率更高；或
- 同等完整率下，少走无关路线、成本更低。

如果两者都不成立，Planner 应移到附录，不作为主创新。

### RQ5：在真正的事实变化数据上是否有效？

建议数据集角色如下：

| 数据集 | 用途 | 当前建议 |
|---|---|---|
| MultiHop-RAG | 跨文档完整证据检索 | q100 做开发和昂贵基线；便宜方法扩至 2,556 题 |
| ChronoQA（E²RAG 英文版） | 时间、因果、角色状态变化 | 本地 1,028 题；当前缺 2 部原文，973 题可完整运行，优先补齐后正式使用 |
| HoH | 旧信息干扰和当前/历史事实 | 不做 9 万页面的朴素全局 EventEffect 抽取；先按官方候选设置或分层抽样 1,000 题 |
| CLARK-News/自建新闻流 | 顺序写入、乱序、来源冲突 | 用于验证增量更新和真实新闻演化，作为增强实验 |

ChronoQA 能验证时间/事件顺序，但不一定直接标注 EventEffect；HoH 能验证旧知识干扰，但不一定标注变化原因。因此还需要一个小规模人工标注的生命周期评测集。

### RQ6：本体和抽取结构本身是否正确？

人工标注 100–200 篇文档或 200–500 个事件，评价：

- Event 与 Assertion 是否应该连接；
- EventEffect 六分类 Macro-F1 和混淆矩阵；
- Assertion 状态/有效期准确率；
- Mention→来源回链准确率；
- 旧状态—事件—新状态链完整率；
- 查询扩展新增证据的 helpful/neutral/harmful 比例。

增加 Gold Entity、Gold Event、Gold EventEffect 三组 Oracle，区分错误来自抽取、融合、链接还是检索。

### RQ7：收益是否值得成本？

分别报告：

- 索引时间、抽取调用、token、费用、峰值内存、图规模；
- 每增加 1/10/100 篇文档的更新时间与受影响节点；
- 查询 P50/P95、在线调用次数和 token；
- 质量—上下文 token、质量—延迟、质量—费用曲线；
- Top-K、hop、扩展节点数和路线预算敏感性。

---

## 9. 推荐的工作顺序

### P0：先验证核心故事是否成立

1. 完成 Entity-link only、Untyped Effect、Full EventEffect、Randomized Effect 消融；
2. 审计 q100 中 EvoLink 新增和丢失的 Gold Evidence；
3. 为 EventEffect 和演化链做小规模人工标注；
4. 在 ChronoQA 上跑 E²RAG 与 ELC-RAG；
5. 所有结果做置信区间和显著性检验。

### P1：证明状态对 RAG 而非只对图展示有用

1. 完成 B0–B4 冻结证据实验；
2. 统一生成器，评价答案、引用和忠实度；
3. 在 HoH 抽样集评价旧事实误用率；
4. 报告结构抽取误差如何传递到最终答案。

### P2：再决定 Planner 是否进入主论文

1. 将 EvoNeed 需求转换成首轮 route/query/budget；
2. 并行执行需求动作，保留小预算向量保险；
3. 与固定三路和普通问题分解比较；
4. 若不能提高质量—成本 Pareto，则移入附录。

### P3：完成外部基线和规模实验

1. 完成正在构建的 HippoRAG 2；
2. 跑 EventRAG、E²RAG、LightRAG、GraphRAG；
3. 资源允许时补 KG²RAG；
4. q100 完成昂贵主表，核心/便宜方法扩至全量问题；
5. 独立进程重复计时并汇总索引成本。

---

## 10. 论文是否站得住脚的判定标准

### 可以形成强论文的最低闭环

1. Full EventEffect 显著优于三视图简单融合、Entity-link only 和 Untyped Effect；
2. 对 EventRAG、E²RAG、HippoRAG 等最近邻至少在演化问题上有稳定优势；
3. 状态/Effect/来源结构在冻结正文实验中改善答案或引用，随机打乱后收益消失；
4. 收益同时出现在 MultiHop-RAG 和至少一个时间/过时信息数据集；
5. EventEffect 和来源回链本身具有可接受的人工标注准确率；
6. 额外成本可量化，并存在可接受的质量—成本工作点。

### 如果部分假设失败，如何收缩论文

- 若 typed Effect 不优于 untyped：贡献收缩为“事件—事实—来源连接”，不强调六种生命周期类型改善检索。
- 若结构不改善答案：只主张提高完整、可追溯证据检索，不声称帮助生成推理。
- 若 Planner 只提高排序但成本过高：放入附录或后续工作。
- 若只在 MultiHop-RAG 有效：不能主张事实演化，应补时间数据或改写为一般跨视图证据协调。
- 若 EventEffect 抽取精度较低：优先研究抽取/校验或使用高置信边，不能靠下游模型掩盖结构错误。

---

## 11. 推荐论文表述

### 一句话问题定义

> 当事实在不同来源中不断变化时，现有 RAG 容易把旧事实、新事实和改变它们的事件拆散，导致检索结果相关却不完整。

### 一句话方法

> ELC-RAG 使用 Event-Effect Ontology 记录事件如何改变来源可追溯的事实，并通过跨文本、事实和事件视图的协调检索，找回完整的演化证据。

### 一句话收益

> 它不替生成模型裁决真相，而是让模型同时看到变化前、变化过程、变化后及其出处，从而减少缺证、新旧混用和无法追溯的问题。

### 暂定标题

**ELC-RAG: Retrieving Complete and Traceable Evidence for Evolving Facts**  
中文：**ELC-RAG：面向演化事实的完整可追溯证据检索**

---

## 12. 参考工作

1. [EventRAG: Enhancing LLM Generation with Event Knowledge Graphs, ACL 2025](https://aclanthology.org/2025.acl-long.830/)
2. [E²RAG: Respecting Temporal-Causal Consistency, EACL 2026](https://aclanthology.org/2026.eacl-long.90/)
3. [TimeR⁴: Time-aware Retrieval-Augmented LLMs for TKGQA, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.394/)
4. [HoH: Evaluating the Impact of Outdated Information on RAG, ACL 2025](https://aclanthology.org/2025.acl-long.301/)
5. [KG²RAG: Knowledge Graph-Guided Retrieval Augmented Generation, NAACL 2025](https://aclanthology.org/2025.naacl-long.449/)
6. [OG-RAG: Ontology-grounded Retrieval-Augmented Generation, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1674/)
7. [Adaptive-RAG: Strategy Selection by Question Complexity, NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)
8. [IRCoT: Interleaving Retrieval with Chain-of-Thought, ACL 2023](https://aclanthology.org/2023.acl-long.557/)
9. [Question Decomposition for RAG, ACL SRW 2025](https://aclanthology.org/2025.acl-srw.32/)

## 13. 当前最诚实的总判断

这个方向**有成立的可能，而且问题比“再做一个 GraphRAG”更明确**：它针对的是事实变化条件下的证据缺环和新旧混用。仓库也已经具备本体契约、Assertion 生命周期、EventEffect、来源 Mention、三视图检索和 EvoLink 扩展，不是只有概念。

但现在仍处于“方法原型 + 正向小样本信号”，不是“创新已被证明”。最危险的两点是：

1. EventRAG、E²RAG、KG²RAG、OG-RAG 已经覆盖事件图、跨图连接、KG 扩展和本体 RAG，必须通过 Event→Assertion lifecycle 和完整证据目标证明差异；
2. 当前数据对生命周期的直接标注不足，typed EventEffect 和状态元数据尚未通过消融证明有用。

因此，下一步不应继续扩大概念，而应优先做 Untyped/Typed/Randomized Effect 消融、人工结构标注、ChronoQA/HoH 实验和冻结正文的状态利用实验。这四组结果将直接决定论文核心能否站住脚。
