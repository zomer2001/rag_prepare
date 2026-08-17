# Agentic RAG 与知识组织技术调研

> 报告日期：2026-08-16  
> 调研范围：Agentic RAG、GraphRAG、Knowledge Graph、Temporal Knowledge Graph、Event Graph、Ontology、Dynamic Ontology  
> 项目关联：Vector–Semantic–Event 三通路、Ours_EvoLink、Ours_EvoNeed

## 摘要

RAG 的发展包含两条相互关联但长期分开的研究路线。第一条路线改变知识组织方式：从独立文本块发展到层次摘要、实体关系图、事件图和带时间状态的知识表示。第二条路线改变检索控制方式：从一次性查询发展到检索—推理交替、证据评价、查询改写和动态路由。前者回答“系统能够访问什么知识结构”，后者回答“系统下一步应检索什么”。

当前研究正在两条路线的交汇处遇到一个具体问题：Agent 可以判断证据是否充分，事件 RAG 也可以表示事件和时间线，但系统通常没有把“缺少原文、缺少关系、缺少状态演化”定义为统一、可执行的知识需求。RouteRAG 能够学习 passage/graph/hybrid 路由，却不显式维护知识缺口；S2G-RAG 能够输出结构化 gap，但主要覆盖实体、属性、关系和证据片段；EventRAG 和 E²RAG 能够组织事件，却没有用事件演化缺口统一控制原文、语义关系和事件三个知识空间。

本报告据此组织相关研究：先说明 RAG 的知识组织如何从文本发展到图和事件，再说明 Agentic 检索如何从多轮检索发展到显式缺口规划；随后区分知识图谱、时序知识图谱、事件图、本体和动态本体，并把当前项目的语义通路和事件通路放入同一框架。报告最终提出一个较窄的研究位置：将原文、关系和演化定义为三类必要知识，使知识需求、持久表示、检索动作和证据角色保持一致，并让单轮与多轮检索共享该结构。

---

## 1. 调研范围

### 1.1 两个研究问题

本报告不把 Agentic RAG、KG、TKG、Ontology 和 Event 分别写成概念词条，而围绕两个问题组织：

1. **知识怎样组织。** 原文、实体关系、事件和状态应该以什么单位保存，怎样保持来源与时间；
2. **检索怎样控制。** 系统何时检索、检索哪个知识空间、如何根据当前证据生成下一步查询。

这两类问题相互制约。若持久知识只有文本 Chunk，Agent 即使知道缺少关系，也只能继续改写文本 Query；若系统有多种索引，但 Planner 不知道每种索引能补什么，三路检索就会退化为固定并行或末端 RRF。

### 1.2 统一分析框架

本文使用四个层次比较相关工作：

| 层次 | 要回答的问题 | 典型对象 |
|---|---|---|
| 知识需求 | 当前分析需要什么知识 | 原文、关系、演化 |
| 持久表示 | 这些知识怎样存储 | Chunk、KG、Event Graph、TKG |
| 检索动作 | 系统怎样访问它们 | Vector、Graph、Event、Hybrid |
| 证据控制 | 何时继续、停止或修正 | sufficiency、gap、reflection、route policy |

这一框架既能解释现有研究，也能检验当前三通路系统是否只是技术组合，还是形成了需求与知识结构之间的明确对应。

---

## 2. RAG 的知识组织

### 2.1 文本块和向量索引

传统 RAG 将语料切分为 Chunk，通过稠密向量或稀疏检索返回相似片段。其优点是通用、增量更新简单并保留原文；局限是检索单位相互独立，跨块依赖只能由生成模型临时恢复。多跳问题常需要先读到一个实体，再据此发现下一跳对象，单次相似度排序很难显式表示这种依赖。

文本检索解决的是“哪些材料在语义上相关”，并不直接说明不同材料中的对象是否相同、关系是否成立，也不区分旧状态和新状态。因此，后续研究沿两个方向发展：增加文本粒度层次，以及把跨文档关系显式化。

### 2.2 层次化文本

[RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) 对文本进行递归聚类和摘要，构造从局部片段到高层概括的树。它说明检索性能不仅取决于 Embedding，也取决于索引怎样组织粒度。对于需要整体概括的问题，高层摘要可以避免只返回局部细节；对于精确事实，仍需回到叶节点原文。

层次化方法主要处理局部与全局的粒度关系，不直接表示实体之间的语义关系，也不表示状态如何随事件变化。

### 2.3 实体关系图和 GraphRAG

[Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) 从文本抽取实体关系图，通过社区发现和社区摘要回答全语料问题；Local Search 又结合局部图邻域和原始文本单元。[LightRAG](https://arxiv.org/abs/2410.05779) 使用低层实体关键词和高层关系/主题关键词进行双层检索，强调轻量图索引和增量插入。[HippoRAG](https://arxiv.org/abs/2405.14831) 以知识图和 Personalized PageRank 模拟关联记忆，利用图传播恢复多跳路径。

这些方法把检索单位从相互独立的 Chunk 推进到 Entity、Relation、Community 和 Graph Path。它们特别适合：

- 同一实体跨文档出现；
- 需要桥接实体的多跳问题；
- 需要全局主题或群落信息；
- 问题中包含关系约束而非仅有主题相关性。

图组织也引入新风险。LLM 抽取的实体和关系可能遗漏或错误，错误实体合并会改变整个拓扑；若图对象没有回到原文 Mention 的链接，系统无法核验结构为什么成立。另一个限制是静态图容易把同一实体对在不同时期的关系合并到一条边上。

### 2.4 多视图组织

[SiReRAG](https://arxiv.org/abs/2412.06206) 分别构建语义相似性树和实体相关性树，再将不同视图的节点放入统一检索池。它表明复杂问题可能需要多个组织视角，而不是在向量和图之间二选一。

多视图方法的常见实现是：每个视图返回 top-k，再做分数归一化、RRF 或统一候选排序。这种 late fusion 能扩大召回面，却不知道已选候选还缺什么，也很少让一个视图的结果成为另一个视图的查询锚点。多视图是否“协同”，需要看检索过程中是否发生跨视图的证据补全，而不只是最终列表是否混合。

---

## 3. 事件与状态变化

### 3.1 为什么关系图不足以表达变化

实体关系图擅长表达“谁与谁有什么关系”，但复杂研判还会问：

- 这条关系何时成立；
- 成立之前是什么状态；
- 哪个事件创建、更新或终止了关系；
- 新关系是补充旧关系，还是替代、否认或纠正旧关系；
- 每个时期的状态分别由哪些原文支持。

仅给关系边增加文档发布日期不能回答这些问题，因为发布日期不一定等于事实有效时间。把所有事件都抽成普通关系，也会失去参与角色、时间精度、结果和事件之间的关系。

### 3.2 EventRAG

[EventRAG](https://aclanthology.org/2025.acl-long.830/) 的出发点是传统 RAG 忽略叙事背后的事件结构，单次文本拼接难以连接跨文档事件，也不容易恢复时间顺序。它从文本抽取事件、实体及关系，把结果写入向量库和 Event Knowledge Graph；使用向量相似度合并语义等价的事件/实体，并对连接不足的节点进行扩展。查询阶段，Agent 分解事件参与者、时间和逻辑关系，在事件图中迭代检索相连事件，并执行反思与纠错。

EventRAG 已经覆盖“事件图 + Agent 多轮事件检索”。因此，增加事件索引或让 Agent 继续检索事件本身不能作为新颖贡献。可以继续研究的是：事件图是否与原文和稳定关系表示保持独立又可连接，Agent 是否显式知道当前缺少的是原文、关系还是状态演化。

### 3.3 E²RAG

[E²RAG](https://aclanthology.org/2026.eacl-long.90/) 指出，实体中心 KG 通常把同一实体的所有 Mention 合并为一个节点，可能抹去实体在不同时间和上下文中的状态差异。它分别构造实体 Mention 图和事件图：同一实体在不同 Chunk 中的 Mention 不直接合并，事件可以去重；再通过与 Chunk 相关的二部边把实体 Mention 连接到参与事件，并保存 narrative position、PRECEDES、CAUSES、REVEALS 等事件边。

查询时，E²RAG 抽取实体和事件 cue，在两个向量空间寻找种子，以事件种子为前沿扩展一跳，再返回原始 passage 和线性化子图。它与当前语义—事件组织最接近，因而不能把“实体图 + 事件图 + 映射”本身写成当前项目的主要创新。可继续推进的方向是，把一次性 cue extraction 提升为跨轮 Knowledge Need State，并把事件结果进一步连接到 Assertion 的前后状态和原文 Mention。

### 3.4 事件时间线与事实状态

[DyG-RAG](https://arxiv.org/abs/2507.13396) 使用带事件语义和时间锚点的动态事件单元，并通过共享实体和时间邻近构图，重点恢复有序事件时间线。[IA-RAG](https://arxiv.org/abs/2606.06044) 进一步使用区间关系表达持续、重叠和包含。

这些工作说明时间敏感问题需要专门的时间或事件单位，但还需区分三件事：

1. **事件顺序：** A 发生在 B 之前；
2. **事件因果：** A 导致 B，需要额外因果证据；
3. **事实变化：** 事件 E 创建、更新、终止或替代某项 Assertion。

事件图可以恢复“发生了什么”，时序知识图谱可以恢复“某事实在何时有效”，EventEffect 则连接两者，解释事实为什么改变。

---

## 4. Agentic 检索

### 4.1 检索与推理交替

[IRCoT](https://aclanthology.org/2023.acl-long.557/) 将思维链与检索交替进行，使用当前推理步骤产生下一轮查询。它建立了一个重要原则：检索改变推理状态，推理状态也应改变后续检索。

[FLARE](https://arxiv.org/abs/2305.06983) 根据即将生成内容的低置信位置主动检索；[DRAGIN](https://aclanthology.org/2024.acl-long.702/) 根据实时信息需求决定检索时机并生成查询。这类方法重点解决 when to retrieve 和 what to retrieve，但主要面对文本检索空间。

### 4.2 证据评价和纠错

[Self-RAG](https://arxiv.org/abs/2310.11511) 通过反思 token 学习是否检索、证据相关性和回答质量。[CRAG](https://arxiv.org/abs/2401.15884) 使用检索评价器区分正确、错误和模糊结果，并在失败时触发外部搜索和知识精炼。[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) 根据问题复杂度选择不检索、单步或迭代策略。

这些工作推动 RAG 从固定流水线转向受反馈控制的过程，但证据不足常被表示为一个整体评分或自然语言原因。若系统有文本、关系和事件多个知识空间，仅判断“不充分”还不足以决定下一步动作。

### 4.3 RouteRAG

[RouteRAG](https://aclanthology.org/2026.findings-acl.1502/) 研究多轮推理中如何在 passage 和 graph 之间动态路由。Policy LLM 每一步可以继续推理、搜索或回答；搜索时选择 Passage、Graph 或 Hybrid，并生成 sub-query。Passage 路使用 DPR，Graph 路使用 HippoRAG 2 的 Personalized PageRank，Hybrid 使用 RRF。

RouteRAG 采用两阶段 GRPO：第一阶段以答案 Exact Match 学会正确回答，第二阶段在正确轨迹上加入检索时间奖励，以兼顾效果和成本。它解决的是“何时检索、去哪一类库检索、何时停止”，其路由状态从任务奖励中隐式学习，并不输出可审计的“当前缺原文还是关系”。动作空间也没有独立事件演化通路。

与 training-free 显式 Planner 相比，RouteRAG 的优势是能够学习复杂策略；代价是需要训练数据、策略优化和额外复现成本。显式 Planner 的可主张优势是可解释、无需任务特定训练和易于跨域迁移，但必须用实验证明规则化需求状态不会带来严重规划偏差。

### 4.4 S2G-RAG

[S2G-RAG](https://aclanthology.org/2026.acl-long.1185/) 与“判断缺什么”最接近。它在每轮读取问题和累积 Evidence Context，输出是否充分以及结构化 gap items。主要类型包括 `bridge_entity`、`attribute`、`relation`、`evidence_span` 和 `other`；系统再把 target 与 slot 拼回原问题，形成下一轮 BM25 或 E5 查询，并使用句子级 extractor 压缩上下文。

S2G-RAG 的 Judge 使用强教师轨迹和 LoRA 训练，但也报告 untrained judge，说明固定 gap schema 本身具有价值。其局限是 schema 难以表达 multi-entity joins、temporal constraints 和跨变量组合关系。这为演化需求提供了准确入口：已有多轮检索开始显式表示静态事实缺口，但尚未把前态—事件—时间—后态作为标准缺口并映射到事件知识空间。

### 4.5 从多轮改写到知识需求规划

通用 Agentic RAG 的典型循环是：

```text
retrieve
  → assess sufficient / insufficient
  → rewrite query
  → retrieve again
```

知识需求规划要求更强的状态：

```text
当前分析目标
  → 已建立结论及 Evidence ID
  → 未满足的原文/关系/演化需求
  → 通路专用 Query 与结构动作
  → 新证据是否关闭对应缺口
```

这样，检索轮次不再是目标。Agent 的价值取决于每轮是否获得了推进分析所需的新增知识，以及新增知识是否值得额外延迟和模型调用。

---

## 5. 知识图谱、时序知识图谱与本体

### 5.1 知识图谱

Knowledge Graph 保存实体、类型、属性和实体之间的关系。例如：

```text
CompanyA --ACQUIRED--> CompanyB
CompanyB --DEVELOPS--> CapabilityX
```

KG 适合实体消歧、多跳关系和结构查询。它可以为每条关系附加来源，但普通实体—关系图通常不强调事实的有效区间和状态变化。

### 5.2 时序知识图谱

Temporal Knowledge Graph 在事实层面增加时间和状态。它至少要区分事实在现实世界何时有效，以及系统何时记录该事实。常见表示包括带时间戳/区间的三元组、Assertion 节点、状态快照和事件驱动更新。

```text
Assertion:
  subject = CompanyA
  predicate = HAS_CAPABILITY
  object = CapabilityX
  valid_from = 2024-06
  valid_to = null
  status = active
```

只给文档增加发布日期不构成完整 TKG，因为文档发布时间不一定等于关系的生效时间。TKG 重点回答 state-at-time、历史状态、事实更新和冲突。

### 5.3 事件图

Event Graph 以发生为主要节点：

```text
AcquisitionEvent
  participants = CompanyA, CompanyB
  time = 2024-05
  result = CompanyA obtains CapabilityX
```

事件图擅长表达参与者、时间、事件顺序、包含、因果候选和跨文档事件共指。它回答“发生了什么”，但不一定直接回答某项关系在每个时间点是否有效。

EventEffect 可以连接 Event Graph 与 TKG：

```text
AcquisitionEvent
  --CREATES-->
CompanyA HAS_CAPABILITY CapabilityX
```

### 5.4 本体

Ontology 定义知识图谱允许使用的类别、关系和约束。它回答的不是“哪条事实是真的”，而是“在这个领域里可以怎样表达事实”。例如：

- `Organization`、`Product`、`Capability`、`AcquisitionEvent` 的类别层次；
- `ACQUIRED` 的主体和客体必须为 Organization；
- EventEffect 的类型只能为 CREATE、UPDATE、TERMINATE、SUPERSEDE、DENY 或 CONFIRM；
- 某些节点至少需要一个来源 Mention。

OWL 适合类别、层次和语义关系，SHACL 适合最低字段、值域和基数验证。本体能够减少 LLM 抽取产生的类型漂移和不可复用谓词，但不能替代原文证据，也不能凭 Schema 推断具体事实。

### 5.5 动态本体

Dynamic Ontology 或 Ontology Evolution 指 Schema 本身发生变化，例如：

- 新增一种实体或事件类型；
- 将原概念拆分、合并或重命名；
- 新增或废弃关系；
- 修改约束并迁移旧实例；
- 维护新旧本体版本的兼容性。

它与 TKG 的区别是：

- TKG 管理实例事实随时间变化；
- Dynamic Ontology 管理概念体系和约束随版本变化。

某公司更换 CEO 是实例变化，属于 TKG/Event；系统决定新增 `ModelProvider` 类型是 Schema 变化，属于本体演化。让 LLM 在索引过程中直接生成新标签不等于可靠动态本体；正式演化需要候选统计、审批、版本发布、实例迁移和检索回归测试。

### 5.6 五类组织方式的统一比较

以“公司 A 通过收购公司 B 获得能力 X”为例：

| 组织方式 | 主要表示 | 可以直接回答 | 不能单独保证 |
|---|---|---|---|
| 文本/向量 | 报道原文和 Chunk | 材料如何描述收购和能力 | 跨文档实体是否相同、关系状态 |
| KG | A 收购 B；B 开发 X | 实体关系和多跳路径 | 关系何时生效、前后状态 |
| Event Graph | 某次收购的参与者、时间和结果 | 发生了什么、何时发生 | 某事实在任意时点是否有效 |
| TKG | A 拥有 X 的有效区间和状态 | 收购前后分别是什么状态 | 哪篇原文、哪个事件导致变化 |
| Ontology | Company、Capability、Acquisition 的类型和约束 | 表示是否合法、字段是否完整 | 具体事实是否真实 |
| Dynamic Ontology | 类型和约束的版本变化 | Schema 如何演化和迁移 | 业务实例状态本身 |

这些组织方式不是互相替代关系。Ontology 约束 KG、Event Graph 和 TKG；EventEffect 将事件与事实状态连接；所有结构化对象仍需回到文本 Mention。

### 5.7 当前语义通路和事件通路

当前项目采用“分路索引、共享身份、协调图连接”的方式：

- Vector Route 保存 Chunk Embedding 和原文位置；
- Semantic Route 保存 Entity、Assertion、AssertionMention，以及 LightRAG 的实体/关系向量和图拓扑；
- Event Route 保存 Event、EventMention、EventRelation、EventEffect 和事件向量；
- Neo4j 使用 Entity 和文档身份连接语义与事件对象；
- Ontology v0.1 负责类型映射和 SHACL/OWL 约束。

这套组织具有部分声明级 TKG 特征，因为 Assertion 可以有状态和有效期，EventEffect 可以创建、终止或替代 Assertion。但更准确的称呼是“实体关系、事件和部分声明生命周期的联合表示”，不能把当前系统笼统描述为自动动态本体或完备双时态 TKG。

---

## 6. 相关方法对比

### 6.1 能力对照

| 方法 | 主要组织 | 检索控制 | 显式缺口 | 事件/演化 | 原文回链 | 主要边界 |
|---|---|---|---|---|---|---|
| RAPTOR | 多粒度摘要树 | 固定树检索 | 否 | 否 | 是 | 不显式表示关系和状态变化 |
| GraphRAG | 实体图 + 社区摘要 | local/global | 否 | 非重点 | 是 | 全局摘要可能压缩时间与细节 |
| LightRAG | 向量 + 实体关系图 | low/high-level | 否 | 无独立事件通路 | 是 | 主要是实体中心的静态关系 |
| HippoRAG | KG + PPR | 单轮图传播 | 否 | 非重点 | 是 | 依赖图质量，不诊断知识类型 |
| SiReRAG | 相似性树 + 相关性树 | 统一候选池 | 否 | 否 | 是 | 多视图最终被展平 |
| EventRAG | Event KG | Agent 迭代事件检索 | 事件要素 | 事件链 | 有 | 主要在事件图中工作 |
| E²RAG | Entity mention 图 + Event 图 | cue + 一跳传播 | 否 | 动态上下文 | 是 | 主要为单轮，不维护跨轮需求 |
| DyG-RAG | 动态事件单元 + 时间线 | 时间遍历 | 否 | 强 | 是 | 重点是事件顺序而非 Assertion 生命周期 |
| RouteRAG | Passage + HippoRAG 2 | RL 动态路由 | 隐式 | 无独立演化动作 | 是 | 需要训练，知识缺口不可审计 |
| S2G-RAG | 文本 Evidence Context | gap + 文本改写 | 静态事实 gap | 时序有限 | 是 | gap 未映射到异构持久知识 |
| 当前项目 | Chunk + Assertion + Event/Effect | 一次规划或跨轮 Need | 三类显式需求 | 前态—事件—后态 | 是 | 动态规划效果仍待验证 |

### 6.2 不能作为创新的内容

根据上述对比，以下表述不成立或过宽：

- 首次结合向量和知识图谱；
- 首次加入事件索引；
- 首次构建实体图和事件图；
- 首次根据证据不足改写 Query；
- 首次在文本和图之间动态路由；
- 事件先后关系等同于因果关系；
- 多来源等同于独立证实。

### 6.3 可以成立的研究位置

更准确的研究位置是：

> 将原文落证、实体关系和事件演化定义为三类必要知识，使这三类需求分别对应持久索引对象、通路专用查询和证据角色；再通过共享身份和 EvoLink 连接三种表示，使单轮检索和多轮 Agentic 检索能够在同一证据空间中执行一次性或持续的需求规划。

其中最有辨识度的部分不是 Event 节点本身，而是：

```text
Event
  → EventEffect
  → Assertion before/after state
  → AssertionMention / EventMention
  → source text
```

这条路径把“发生了什么”“哪项事实改变了”和“原文依据在哪里”连接起来，并能够成为 Planner 的演化缺口和 Auditor 的验证对象。

---

## 7. 三类知识需求与当前研究问题

### 7.1 两条研究路线的汇合

知识组织路线的发展可以概括为：

```text
flat chunks
  → hierarchical text
  → entity–relation graph
  → event graph
  → temporal assertion lifecycle
```

检索控制路线的发展可以概括为：

```text
single retrieval
  → retrieval–reasoning interleaving
  → relevance/sufficiency assessment
  → query rewrite
  → text–graph routing
  → structured evidence gaps
```

当前交汇问题不是再增加一种数据结构，而是让结构化知识需求控制这些表示的访问方式。

### 7.2 精确 GAP

已有 Event RAG 能够表示实体状态随事件变化，多轮 RAG 也能够判断缺少哪些静态事实；但状态演化尚未稳定成为驱动原文、关系和事件三个知识空间的标准缺口类型。

这一 GAP 可以拆成三项：

1. **需求层：** 现有 gap schema 主要覆盖实体、属性、关系和证据片段，缺少 before–event–time–after；
2. **动作层：** 缺口通常被拼回文本 Query，没有生成适合关系图和事件图的专用动作；
3. **验证层：** 多轮系统常统计是否又检索一次，缺少对具体知识槽位是否关闭的评价。

### 7.3 当前项目的三个方法层次

当前系统可以分成连续的三个方法层次：

1. **三通路协同索引。** 同一文档形成 Chunk、Assertion 和 Event 三类表示，并通过 Entity、Mention 和 Effect 连接；
2. **Ours_EvoLink。** 固定三路宽召回后，利用协调图补充关系、事件和来源证据；
3. **Ours_EvoNeed。** 根据当前证据池维护三类知识需求，选择通路、生成 Query 并验证缺口关闭。

第一层提供知识基础，第二层已有当前实验支持，第三层仍需完善和验证。这样的分层可以避免把工程系统、静态方法和 Agentic 原型混成一个不可消融的方法。

### 7.4 重点究竟是数据组织还是检索

当前研究应定位为 **both，但以知识需求为上位问题**：

- 若只有数据组织，三条索引最终仍可能只是 late fusion；
- 若只有检索路由，Agent 没有能够稳定映射的事件演化表示；
- 三类知识需求同时规定“索引中保存什么”和“查询时访问什么”。

论文主贡献可写成需求驱动的知识组织与检索方法。工程实现中，数据组织占较大工作量；算法验证中，则需要证明跨路补全和缺口规划带来检索增益。

---

## 8. 评价方法和后续方向

### 8.1 单轮检索评价

单轮实验至少应包括：

- Evidence Recall@K；
- Complete Recall@K；
- NDCG、MRR；
- 三类证据角色覆盖；
- 时间链和 Assertion lifecycle 完整度；
- 延迟、索引时间和存储成本。

关键消融包括 Vector、Vector+Semantic、Vector+Event、Separate-3Path+RRF、EvoLink、移除 EventEffect 和移除 Mention 回链。

### 8.2 多轮检索评价

Agentic 评价不能只报告最终 Recall，应增加：

- Need type 分类 Precision/Recall/F1；
- Route selection accuracy；
- 每轮新增 gold evidence；
- gap closure rate；
- evolution slot completeness；
- 重复检索率；
- 平均轮数、LLM token 和 P95 latency；
- 单位成本获得的完整证据增益。

还需要记录完整轨迹，使失败能够归因于缺口识别、Query 生成、底层召回、跨路连接或 Auditor 误判。

### 8.3 知识组织评价

若将协同索引作为论文贡献，应独立评价：

- Entity extraction 和 relation extraction；
- Entity resolution；
- Event extraction 和 event coreference；
- Assertion deduplication；
- EventEffect attachment；
- Mention backlink completeness；
- duplicate、refine、update、supersede 和 conflict 的区分。

可复用公开数据集时应使用已有实体链接、事件抽取和事件共指标注；无法覆盖 Assertion lifecycle 时，需要构造规模可控的人工标注集，而不能只以最终 QA 分数代替索引质量。

### 8.4 后续方向

短期最可行的工作是把 EvoNeed 的 NeedItem、证据绑定和缺口关闭做实，并完成静态消融和 Agentic 轨迹评测。中期可以优化固定预算下的最小充分证据集合，并让文本命中触发局部结构修复。长期再考虑完整双时态 TKG、长期增量一致性和本体版本演化。

Dynamic Ontology 不应成为当前论文的主线。它适合作为知识治理扩展：从未映射类型、频繁冲突和人工 ReviewCase 中发现 Schema 候选，经审批和迁移后发布新版本，再比较新旧本体下的索引和检索表现。

## 结论

知识组织与 Agentic 检索不是两条互不相关的技术线。向量、图谱、事件和时序结构决定 Agent 能访问什么；充分性、缺口和路由决定 Agent 怎样访问它们。当前研究已经分别解决了文本相关性、实体连接、事件时间线和多轮检索控制，但尚未稳定解决“根据缺失知识类型访问异构知识空间”。

当前项目的研究位置由此可以明确：以原文、关系和演化三类知识需求为起点，建立相应的索引对象和检索动作；用 EvoLink 连接 Entity、Assertion、Event、Effect 和 Mention；再让单轮检索执行一次性需求规划，让 Agentic 检索持续更新未满足需求。这个位置比“增加第三条事件通路”更准确，也比“构建完整动态本体”更可控。

## 参考文献

1. Lewis et al. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020.
2. Sarthi et al. [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html), ICLR 2024.
3. Edge et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), 2024.
4. Guo et al. [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779), 2024.
5. Gutiérrez et al. [HippoRAG](https://arxiv.org/abs/2405.14831), 2024.
6. [SiReRAG](https://arxiv.org/abs/2412.06206), 2024.
7. [EventRAG](https://aclanthology.org/2025.acl-long.830/), ACL 2025.
8. [DyG-RAG](https://arxiv.org/abs/2507.13396), 2025.
9. [E²RAG](https://aclanthology.org/2026.eacl-long.90/), EACL 2026.
10. [IA-RAG](https://arxiv.org/abs/2606.06044), 2026.
11. Trivedi et al. [IRCoT](https://aclanthology.org/2023.acl-long.557/), ACL 2023.
12. Jiang et al. [FLARE](https://arxiv.org/abs/2305.06983), 2023.
13. [DRAGIN](https://aclanthology.org/2024.acl-long.702/), ACL 2024.
14. Asai et al. [Self-RAG](https://arxiv.org/abs/2310.11511), 2023/2024.
15. Yan et al. [CRAG](https://arxiv.org/abs/2401.15884), 2024.
16. Jeong et al. [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/), NAACL 2024.
17. [RouteRAG](https://aclanthology.org/2026.findings-acl.1502/), ACL Findings 2026.
18. [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/), ACL 2026.
19. [TimeR⁴](https://aclanthology.org/2024.emnlp-main.394/), EMNLP 2024.
20. [OG-RAG](https://aclanthology.org/2025.emnlp-main.1674/), EMNLP 2025.
