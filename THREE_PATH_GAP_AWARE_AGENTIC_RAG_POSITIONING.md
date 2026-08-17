# 三通路知识协同与演化知识缺口驱动的 Agentic RAG：研究定位、文献脉络与方法设计

> 更新日期：2026-08-14  
> 研究对象：原文知识通路、实体关系知识通路、事件演化知识通路，以及面向持续研判的知识需求规划  
> 核心判断：该方向可以成立。最稳的入口不是“再增加一种检索器”，而是指出现有多轮 RAG 对缺失知识的建模仍以静态事实、属性和关系为主，尚未把“实体及其关系的状态如何随事件演化”作为一等知识需求，并据此协调原文、关系和事件三类知识通路。

## 一、先给结论：这个方向能不能站住

能站住，但需要收紧论文主张。

不能把核心贡献写成“首次联合文本、知识图谱和事件三条检索通路”，因为已有工作已经分别覆盖了图结构检索、实体—事件联合表示、多源事件图、文本—图谱动态路由和缺口驱动的迭代检索。尤其需要正面面对以下近邻：

- LightRAG 已通过低层实体与高层主题的双层机制联合图结构和向量表示；
- EventRAG 已构建跨文档事件知识图，并进行迭代式多事件检索与推理；
- E²RAG 已显式构造实体图、事件图及二部映射；
- RouteRAG 已学习何时从文本或图中检索，以及何时停止；
- S2G-RAG 已判断当前证据是否充分，并生成结构化缺口以指导下一轮查询。

这些近邻并没有把问题完全做完。它们分别解决“如何组织事件”“如何在文本与图之间选路”“当前还缺哪个静态事实”，但没有共同解决：在持续分析过程中，Agent 怎样识别自己当前缺的是原文事实、实体关系，还是由事件造成的状态演化；又怎样让三类知识互相提供锚点，逐步形成可核验的研判链。

因此，最稳妥的研究定位不是：

> 我们同时使用三种检索器，并由 Agent 在证据不足时改写查询。

而是：

> 我们研究持续研判过程中异构知识需求的动态变化。系统首先判断当前分析需要知道“原文说了什么”“涉及哪些实体及其关系”以及“这些实体或关系的状态如何随事件发生、改变、终止、否认或替代”；随后，根据当前知识状态中尚未满足的部分，规划原文、语义关系或事件演化通路，生成通路特定的检索动作，并把新知识重新纳入分析状态，直至形成足以支撑当前研判结论的可溯源知识链。

这一定位的研究重点是**检索控制与证据协同**，数据组织是使这种协同可以发生的必要基础。换句话说，答案是 both，但不是五五开：

| 层次 | 论文中的地位 | 建议权重 |
|---|---|---:|
| 演化知识缺口驱动的三通路规划与协同 | 主方法、主创新 | 55%—65% |
| 文本—实体关系—事件的异构数据组织 | 方法成立的表示基础 | 20%—30% |
| 来源回溯、工程实现与可审计轨迹 | 系统贡献 | 10%—20% |

如果把重点放在“新建统一 TKG 索引”，论文将承担很高的索引成本、抽取误差和全量效果风险；如果把重点放在“普通 Agentic RAG”，又会与 S2G-RAG、RouteRAG 等工作过近。最合适的位置恰好在两者之间：**利用三类异构知识组织，解决持续研判中静态事实、实体关系与事件演化知识的需求识别、通路规划和协同补全问题。**

---

## 二、论文要解决的核心问题

### 2.1 现有 Agentic RAG 的失败不只是“没找到相关文档”

复杂研判并不是把一个问题检索两轮后结束。Agent 会随着分析推进不断形成中间判断、发现新对象、修正假设，并提出新的知识需求。一个当前知识池即使包含若干高相关文档，也可能仍然不能支持下一步分析。例如：

- 已找到某项技术的介绍，但缺少“由哪个机构研发”的关系证据；
- 已找到事件结果，但缺少事件发生时间或前序状态；
- 已找到旧状态和新状态，却缺少导致变化的事件；
- 已找到一条肯定陈述，但没有检索到同时间、同条件下的否定或纠正；
- 已找到图中的规范化结论，却没有返回可以引用的原文；
- 比较问题只覆盖了对象 A，没有覆盖对象 B 的同一比较维度。

这些失败不能被一个二元的“相关/不相关”判断充分表达。更重要的是，其中一些缺的不是另一个静态事实，而是一个演化过程：原来的实体关系是什么，发生了什么事件，事件使哪个关系或状态发生了怎样的改变，新的状态从何时开始成立。

现有迭代 RAG 已经会判断“证据是否充分”，也会生成下一轮查询；现有路由 RAG 已经会选择不同检索器。但是，多数方法仍把缺口表达为自由文本中的“还需要知道什么”，或将缺口限定为桥接实体、属性、静态关系和证据片段。它们通常没有进一步回答四个问题：

1. 当前缺少的是原文事实、实体关系，还是实体/关系的事件演化知识？
2. 如果缺少演化知识，究竟缺少前态、变化事件、时间锚点、后态，还是替代/否认关系？
3. 为什么某一条知识通路比其他通路更适合满足当前分析需求？
4. 新获得的知识是否真正推进了研判，而不是只增加更多语义相似内容？

这四个问题共同构成本文可以主张的研究空白。

### 2.2 建议的研究问题

英文表述：

> During an evolving analytic workflow, how can an agent explicitly diagnose whether its current knowledge state lacks textual grounding, relational structure, or event-driven state evolution, and then coordinate heterogeneous text, semantic-graph, and event retrieval paths to acquire the knowledge needed for the next reasoning step without task-specific training?

中文表述：

> 在持续演化的研判工作流中，Agent 如何显式判断当前知识状态缺少原文依据、实体关系还是事件驱动的状态演化，并在无需任务特定训练的条件下，协调原文、语义图和事件三类异构通路，为下一步分析获取恰好需要且能够回溯来源的知识？

### 2.3 需要明确区分的三个概念

**分析目标**回答“当前正在判断什么”，它会随着工作流推进而变化，而不一定等同于用户最初的问题。

**信息缺口**回答“还不知道什么内容”，例如“还不知道合同签署时间”。

**知识类型缺口**回答“当前缺少哪种知识结构”，即原文事实、实体关系或事件演化；其中事件演化还可分为前态、事件、时间、后态、替代和否认。

**检索通路**回答“通过什么表示和检索操作最有可能补齐该缺口”，例如文本语义搜索、关系图遍历或事件时间链扩展。

四者不能混为一谈。知识类型与通路存在首选对应关系，但不是僵硬的一对一映射：时间信息有时直接写在原文中，事件也可能揭示新的实体关系。因此，Planner 应先确定需要什么知识，再综合已有锚点、索引覆盖和成本选择通路。

---

## 三、Motivation：为什么需要三种表示，又为什么必须由缺口来驱动

### 3.1 先讲清楚三条通路分别负责什么知识

三条通路的划分首先是知识功能划分，其次才是检索器划分。

| 知识通路 | 它回答的基本问题 | 主要知识对象 | 在研判中的职责 |
|---|---|---|---|
| **原文/向量通路** | 文本具体说了什么？ | chunk、句子、mention、数字、引语、限定条件 | 发现相关材料，保留完整语境，并为所有结构化结论提供可引用的原文依据 |
| **语义/实体关系通路** | 涉及哪些实体？实体之间是什么关系？ | entity、relation、assertion、relation path | 对齐跨文档对象，恢复稳定语义关系和多跳连接，回答“谁与谁有什么关系” |
| **事件演化通路** | 这些实体及其关系的状态是怎样演化的？ | event、participant、time、state_before、state_after、event relation | 表达某种关系何时产生、更新、终止、被否认或被替代，回答“发生了什么，以及它改变了什么” |

可以用三句话概括：

> 向量通路保留“文本说了什么”；语义通路组织“有哪些实体及其关系”；事件通路刻画“实体和关系的状态如何随事件演化”。

三者并不是平行重复的知识表示。原文提供最完整但结构最弱的事实依据；实体关系图提供稳定结构，但容易把变化压成一条静态边；事件表示提供发生过程和时间条件，但需要原文来验证，也需要实体关系图确定事件改变的对象。因此，三通路的核心关系应当是：

```text
原文知识负责发现与落证
        ↓ 提供 mention 和实体锚点
实体关系知识负责组织当前语义结构
        ↓ 指明哪个关系可能发生变化
事件演化知识负责解释状态迁移
        ↓ 回写新的实体/关系状态并回溯原文
```

事件演化知识最小应表示为：

```text
State_before(Entity/Relation)
  → Event(actor, action, object, time, result)
  → State_after(Entity/Relation)
  → Supporting Mentions
```

如果只有一个孤立事件摘要，而没有前态、后态和它所影响的关系，那么它仍不足以称为“演化知识”。

### 3.2 为什么固定并行检索加 RRF 还不够

固定三路并行加 RRF 能够提高覆盖率，但它本质上仍是 late fusion：

```text
question → text top-k + graph top-k + event top-k → RRF → documents
```

这一机制没有表达：

- 当前已有证据满足了哪些问题条件；
- 哪些检索结果彼此互补，哪些只是重复；
- 下一轮为什么应走某条通路；
- 新结果是否补齐了关系桥、时间前态或冲突证据；
- 何时应该停止，而不是达到固定轮数后被迫结束。

更强的协同应当形成闭环：

```text
分析目标与 Knowledge Need State → 多视图种子获取
       → TEXT / RELATION / EVOLUTION 需求判断
       → 通路选择与专用动作 → 跨通路补全
       → 更新分析状态 → 产生新需求或停止
```

### 3.3 这不是只为回答一次问题，而是面向持续研判

在一次性 QA 中，系统可以把“证据充分”理解为已经能够回答初始问题。但在研判型 Agent 工作流中，充分性是阶段性的：某批知识足以支持一个中间判断，却可能同时暴露下一项未知信息。

设第 \(t\) 步的分析状态为 \(A_t\)，已获得知识为 \(K_t\)。Agent 不是反复围绕原问题做同义改写，而是执行：

```text
分析当前材料 K_t
  → 形成阶段性判断 A_t
  → 识别下一步知识需求 N_t
  → 判断 N_t 属于原文、关系还是演化知识
  → 规划通路并获取 ΔK_t
  → 更新 K_{t+1}=K_t∪ΔK_t 与 A_{t+1}
```

因此，这里的 Agentic 不是“多检索一轮”的包装，而是一个持续的 knowledge acquisition loop。检索动作服务于正在变化的分析目标，停止条件也不是所有可能信息都被找完，而是当前研判目标已获得足够支撑、继续搜索的边际信息增益过低，或者达到预算边界。

### 3.4 为什么这个问题具有一般性

该问题并不只适用于时间问答。科研、政策、国防科技、产业分析和新闻研判都包含以下共同需求：

- 对象之间的稳定关系；
- 某一时点发生的动作与结果；
- 前后状态的改变；
- 多来源中的重复、补充、冲突和纠正；
- 可引用、可核验的原文依据。

所以三通路不是三个任意模块的拼装，而是分别覆盖**原文描述、语义结构与状态演化**三类知识。知识需求驱动的 Agent 负责把它们从“并列存在”变成“随分析状态协同工作”。

---

## 四、相关工作：从数据组织到检索控制的演化

下面的综述按研究问题的演化组织，而不是简单罗列论文。

### 4.1 第一阶段：扁平文本检索与一次性 RAG

传统 RAG 把语料切分为文本块，通过稀疏或稠密检索找到与问题相似的段落，再交给生成模型。该范式的优点是通用、更新方便且保留原文；其根本限制是检索单位彼此独立，相关性排序不能自然表达跨块依赖。对于多跳问题，一个段落的检索需求往往只有在读取另一个段落后才会出现。

这一限制推动了后续两条路线：一条改变数据组织方式，使跨块关系在索引中显式化；另一条改变查询过程，使检索能够随着推理状态迭代。

### 4.2 第二阶段：层次化组织与图结构组织

[RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) 通过递归聚类与摘要构建由细粒度文本到高层语义概括的树结构，解决长文档中局部片段难以覆盖整体语义的问题。它表明，RAG 的性能不仅取决于检索模型，也取决于索引如何组织不同粒度的信息。

[Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) 从文本中抽取实体关系图，进行社区发现并预生成社区摘要，重点解决针对整个语料的全局性问题；其 local search 又结合知识图与原始文本单元。该工作将“语料组织”从独立文本块推进到实体关系与社区层次。

[LightRAG](https://arxiv.org/abs/2410.05779) 进一步强调轻量图索引和双层检索：低层检索定位具体实体，高层检索定位抽象主题与关系，并把图结构与向量表示结合。LightRAG 的价值在于以相对清楚的实体—关系抽取和增量更新机制，降低 GraphRAG 的使用门槛；但它仍主要围绕实体中心的语义关系组织知识。

[HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) 以知识图和 Personalized PageRank 实现跨段落关联和单轮多跳检索，说明图上的关联传播能够替代部分高成本的逐轮文本检索。

[SiReRAG](https://arxiv.org/abs/2412.06206) 对这一发展作了重要扩展：它区分语义相似性与实体相关性，分别构建相似性树和相关性树，再将两类节点展平到统一检索池。其启示是，复杂问题需要多种组织视角；但它仍主要优化统一池中的候选召回，并未根据当前证据缺口动态安排不同视图之间的补全关系。

这一阶段的演化可以概括为：

```text
flat chunks
  → multi-granularity summaries
  → entity–relation graphs and communities
  → multiple organization views
```

它解决了“知识如何被组织”，但尚未充分解决“当前证据池具体缺什么，以及下一步应使用哪种组织视图”。

### 4.3 第三阶段：从实体中心图走向事件中心与实体—事件联合表示

实体关系图适合表达相对稳定的关系，但会弱化“何时发生”“如何变化”和“是否被替代”等信息。由此出现事件中心 RAG。

[EventRAG](https://aclanthology.org/2025.acl-long.830/) 的出发点有三个：传统 RAG 以段落或句子为单位，忽略叙事背后的事件结构；即使找到相关文本，也缺少明确时间结构，难以重建先后过程；单次检索或文本拼接难以连接来自多源文档的多个事件。它先用 LLM 抽取事件、实体及关系，把抽取结果同时写入向量库和 Event Knowledge Graph；再用向量相似度融合跨文档的语义等价事件/实体，并对连接不足的节点进行知识扩展。查询阶段，Agent 将问题分解为参与者、时间标记和逻辑关系等事件要素，在 EKG 内迭代检索相连事件，执行时间感知推理，并通过 reflection/self-correction 重新检查不一致之处。换言之，EventRAG 已经覆盖“事件知识图＋Agent 多轮事件检索”，所以不能再把“增加事件索引”或“Agent 迭代找事件”本身写成新颖贡献。

EventRAG 与本文的差别不在于有没有 Agent，而在于 Agent 可访问的知识空间。EventRAG 把事件图作为主要知识骨架，原文和稳定实体关系不是三个可独立规划、又能相互补全的知识通路；它的控制状态也不是显式的“当前缺原文、关系还是状态演化”。此外，EventRAG 的 knowledge expansion 可以使用 LLM 内部知识补图，本文若强调可溯源研判，应坚持所有关键关系和状态迁移回到原文 mention。

[DyG-RAG](https://arxiv.org/abs/2507.13396) 将事件语义和时间锚点组织为 Dynamic Event Units，通过共享实体和时间邻近构图，并检索有序事件时间线。它进一步证明，时间敏感问题需要专门的事件检索单位和时间遍历机制。

[E²RAG](https://aclanthology.org/2026.eacl-long.90/) 的问题动机更具体：实体中心 KG 通常把同一实体的所有 mention 合并为一个节点，这会抹去实体在不同时间和语境中的状态差异。例如人物早期与后期的行为状态可能完全不同，单一实体节点会优先返回占主导的旧状态。为此，E²RAG 分别构造实体 mention 图和事件图：同一实体在不同 chunk 中的 mention 不合并，事件可以去重；然后用 chunk-gated 二部边把具体实体 mention 连接到它参与的事件，并保留 narrative position 以及 PRECEDES、CAUSES、REVEALS 等事件边。查询时先抽取实体/事件 cue，在两个向量库中寻找种子，以事件种子为扩展前沿做一跳传播，再按传播分数返回原始 passage 和线性化子图。它还可以生成一次 hypothetical response 来增强 query cue，但整体上仍是单次 query-time retrieval，而不是随研判状态持续更新的 Agentic 控制过程。

E²RAG 与本文在数据表示上最接近，因此本文不能以“实体图＋事件图＋映射”作为主贡献。可以在它基础上向前走的地方是：把其静态的 entity/event cue extraction 与一次扩展，提升为显式的知识需求状态；原文、关系和事件不再对每个 query 固定共同检索，而是根据当前分析缺口规划；事件通路不仅返回相邻事件，还要判断它补的是前态、变化事件还是后态，并把状态变化重新连接回实体关系和原文。

[IA-RAG](https://arxiv.org/abs/2606.06044) 又把时间从单点扩展到区间，使用区间代数表示持续、重叠和包含等关系，说明事件检索还可以进一步形式化时间约束。

这些工作已经较好地回答了“怎样组织和检索事件”。仍未被充分解决的是：当原文、关系与事件知识并存时，如何根据**当前分析所缺的知识类型及演化槽位**组织三者的交互，而不是固定融合或只检索事件链。

### 4.4 第四阶段：从一次性检索走向主动、纠错与迭代检索

[IRCoT](https://aclanthology.org/2023.acl-long.557/) 将检索与思维链交替进行，用当前推理步骤产生下一轮查询，建立了“检索影响推理、推理反过来指导检索”的闭环。

[FLARE](https://arxiv.org/abs/2305.06983) 预测下一句，并在低置信词出现时主动检索，重点回答生成过程中“何时检索”和“检索什么”。[DRAGIN](https://aclanthology.org/2024.acl-long.702/) 则根据模型实时信息需求判断检索时机，并利用整个上下文中的注意力信息生成查询。

[Self-RAG](https://arxiv.org/abs/2310.11511) 通过反思 token 学习何时检索、判断证据相关性并评价生成结果；[CRAG](https://arxiv.org/abs/2401.15884) 使用检索评价器区分正确、错误和模糊结果，在检索失效时触发外部搜索和知识精炼。这些方法使 RAG 从固定流水线转向受反馈控制的过程。

[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) 根据问题复杂度在不检索、单步检索和迭代检索之间选择策略。它说明不同问题不应承担相同检索开销，但其路由主要发生在检索前，依据问题整体复杂度，而不是依据检索后的证据缺口。

这一阶段主要解决了 when to retrieve 与 what to retrieve，但多数方法仍默认单一文本检索空间，或者把证据不足表达为未结构化的自然语言反馈。

### 4.5 第五阶段：异构检索路由与结构化证据缺口

[RouteRAG](https://aclanthology.org/2026.findings-acl.1502/) 是理解本文边界最重要的路由工作。它对已有研究空白的论证分三步展开：第一，普通多轮 RAG 已经能够在推理中继续搜索，但主要使用 passage retrieval；第二，GraphRAG、LightRAG、HippoRAG 等图方法能够暴露实体关系、帮助实体消歧和多跳连接，却多为生成前的一次性图检索，且图检索成本较高；第三，已有 text–graph hybrid 往往采用预定义步骤或固定启发式，不能根据不断变化的 reasoning context 决定何时检索、检索文本还是图、何时回答。由此它提出的 GAP 是：缺少一个能够在多轮推理中学习 graph–text hybrid retrieval policy、同时控制准确率和检索成本的统一政策。

RouteRAG 的实现不是显式 Planner，而是经过训练的生成策略。Policy LLM 在每一步输出继续推理、`<search>` 或 `<answer>`；若搜索，再用特殊 token 选择 `Passage`、`Graph` 或 `Hybrid`，同时生成 sub-query。Passage 路使用 DPR，Graph 路使用 HippoRAG 2 的 Personalized PageRank，Hybrid 用 RRF 融合。训练采用两阶段 GRPO：第一阶段只奖励答案 Exact Match，先学会正确回答；第二阶段在正确轨迹上加入基于总检索时间的效率奖励。论文用 10k 条 HotpotQA 训练样本训练 3B/7B 模型。它的贡献是**学习什么时候检索、去哪一类库检索以及怎样兼顾成本**，不是显式解释“当前缺哪种知识”。

因此，与 RouteRAG 的差别必须写成：RouteRAG 的通路选择是从答案奖励中学出的隐式 policy，动作空间只有 passage/graph/hybrid，没有事件演化通路，也没有可审计的 knowledge-need state；本文采用无需任务特定训练的显式 Planner，把原文、关系和状态演化需求转成结构化动作。相应地，本文不能宣称学习到更优 policy，而应主张 plug-and-play、可解释、低训练成本和跨域迁移，并用实验检验这种显式规划能否接近或超过训练式路由。

[R³AG](https://arxiv.org/abs/2604.22849) 进一步把 retriever 能力分为检索质量和对生成的实际效用，通过对比学习捕获查询与检索器能力的动态匹配。它提示我们，通路选择不能只依据语义相关度，还应依据该通路能否给回答带来增量支持。

[SPARKLE](https://aclanthology.org/2026.acl-long.1793/) 使用独立代理模型和图结构推理控制检索策略，并通过强化学习训练代理，代表了结构化、可插拔的 Agentic 检索政策。

与“判断缺什么”最接近的是 [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/)。它在每轮读取问题和累积 Evidence Context，输出是否充分以及结构化 gap items；gap 类型为 `bridge_entity`、`attribute`、`relation`、`evidence_span` 和 `other`，再把 `target + slot` 拼回原问题形成下一轮 BM25 或 E5 查询。它还用句子级 extractor 压缩多轮上下文。其 Judge 不是完全零训练：作者用执行轨迹和强教师生成过程监督，再对 3B 模型进行 LoRA 微调；不过论文也报告了 untrained judge 版本，说明固定 schema 本身已有一定收益。

S2G-RAG 同时为本文提供了一个很好的入口。它在局限性中明确指出，当前 gap schema 偏向稳定和易用，难以表达 multi-entity joins、temporal constraints 以及跨多个变量的 compositional relations。这说明已有多轮检索已经开始显式建模“缺什么”，但主要还是桥接实体、属性、静态关系和证据片段；“实体及关系的状态如何随事件演化”尚未成为一个展开为前态—事件—后态的标准缺口类型。

因此，本文必须把差异明确限定为：

1. 在现有静态 gap 类型之外，显式增加 `EVOLUTION`，并展开为 `state_before / transition_event / temporal_anchor / state_after / supersession_or_denial`；
2. 不只把 gap 拼成下一条文本查询，而是映射到原文、实体关系和事件演化三个异构知识空间；
3. 为三类知识空间生成不同的查询、图模式和事件链扩展，而不是共享同一段自然语言改写；
4. 把这一机制嵌入持续研判工作流，使每个阶段性分析结论都能继续产生新的知识需求，而不局限于固定两跳 QA；
5. 评价的不只是“又检索了一轮”，而是是否获得了推进下一步分析所需的知识，尤其是否补齐了状态演化链。

### 4.6 文献演化后的精确 GAP

已有工作分别取得了以下进展：

- 数据组织：从 flat chunk 发展到层次摘要、实体关系图、多视图组织和事件图；
- 检索过程：从一次性检索发展到主动检索、充分性判断、查询改写和动态路由；
- 异构来源：开始在文本与图谱之间进行自适应选择；
- 缺口控制：开始用结构化 gap 指导下一跳文本检索，但类型仍主要是实体、属性、关系和证据片段。

但两条路线尚未充分汇合：**事件 RAG 已经能够表示实体状态随事件变化，多轮 RAG 也已经能够判断缺少哪些静态事实；然而，状态演化尚未成为驱动多轮 Agent 选择异构知识通路的一等缺口类型。**

这就是本文最可信的 GAP。

### 4.7 代表性工作的能力边界对照

| 工作 | 主要数据组织 | 检索控制 | 与本文最接近之处 | 尚未覆盖的本文问题 |
|---|---|---|---|---|
| RAPTOR | 多粒度摘要树 | 固定树检索 | 多粒度组织 | 不建模关系/事件知识需求和检索后缺口 |
| Microsoft GraphRAG | 实体关系图＋社区摘要 | local/global 模式 | 图与文本联合上下文 | 不依据当前缺口做关系—事件—文本协同 |
| LightRAG | 实体关系图＋向量 | low/high-level 双层检索 | 当前 KG 通路的直接基础 | 没有独立事件角色和缺口闭环 |
| HippoRAG | KG＋PPR | 图传播式单轮多跳 | 关系桥恢复 | 不诊断异构知识类型缺失 |
| SiReRAG | 相似性树＋相关性树 | 统一检索池 | 多组织视图互补 | 视图结果被展平，缺少检索后定向协同 |
| EventRAG | 跨文档事件知识图 | Agent 迭代事件检索与反思 | 事件融合和多事件推理 | 主要在 EKG 内工作，不显式规划原文—关系—演化三类知识需求 |
| DyG-RAG | 动态事件单元＋事件图 | 时间线遍历 | 时间敏感事件检索 | 重点是事件时间线，不是异构通路缺口路由 |
| E²RAG | 实体 mention 图＋事件图＋二部映射 | cue 检索＋事件前沿一跳传播 | 与 KG＋Event 表示最接近 | 主要是单轮检索，没有随分析状态更新的知识需求规划 |
| IRCoT / FLARE / DRAGIN | 主要为文本语料 | 推理中主动或动态检索 | 迭代式 what/when to retrieve | 缺少原文—关系—演化知识状态和通路专用动作 |
| Self-RAG / CRAG | 文本证据 | 自反思或纠错触发 | 证据质量反馈 | 主要判断相关性/质量，不补全角色化证据组合 |
| Adaptive-RAG | 文本语料 | 按问题复杂度选策略 | 自适应开销 | 路由发生在检索前，不依据当前缺口 |
| RouteRAG | passage＋HippoRAG 2 图 | 两阶段 GRPO 学习多轮路由 | 动态选择文本/图/混合和停止 | 隐式训练策略；没有事件演化动作和显式知识缺口状态 |
| R³AG | 多检索器 | 学习查询—检索器能力匹配 | 关注生成效用 | 未形成可解释的缺口类型与跨通路补全 |
| S2G-RAG | 文本 Evidence Context | 充分性＋静态事实 gap＋文本改写 | 与显式“判断缺什么”最接近 | schema 自承难表达 temporal constraints；没有状态演化链和异构通路规划 |

这个对照说明，本文是在两个成熟分支之间提出一个更窄的连接问题：**如何把实体/关系的状态演化变成显式知识缺口，并使其成为原文、语义关系和事件通路协同的中间控制变量。**

---

## 五、方法设计：如何判断当前分析缺少哪类知识

### 5.1 Planner 维护的不是一次性“证据清单”，而是动态知识需求状态

系统不能只在检索后问 LLM“够不够”，也不能只在任务开始时生成一个不再变化的计划。建议维护一个随研判推进更新的 Knowledge Need State：

```text
KnowledgeNeedState_t = {
  analytic_goal_t,            # 当前阶段要判断什么
  established_findings_t,     # 已有阶段性判断
  focal_entities_t,           # 当前关键实体
  known_relations_t,          # 已确认关系
  unresolved_claims_t,        # 尚未确认的命题
  required_knowledge_t,       # TEXT / RELATION / EVOLUTION
  evolution_slots_t,          # 前态、事件、时间、后态、替代/否认
  provenance_constraints_t,   # 必须能回到哪些原文和来源
  budget_t
}
```

初始 Planner 根据用户任务生成 \(KNS_0\)。每获得一批新知识并形成中间分析后，Auditor 再根据 \((A_t,K_t)\) 更新 \(KNS_{t+1}\)。这样，系统追踪的是“为了继续研判还需要什么”，而不是围绕初始 query 做机械重写。

例如，研判任务是“分析 A 公司获得某项能力的路径”。第一步可能只需要发现 A 的公开材料；随后会出现新的实体 B；接着需要确认 A 与 B 的收购关系；最后还要判断收购事件如何改变 A 的能力状态：

```text
N0 = TEXT:      A 的相关公开材料说了什么？
N1 = RELATION:  A 与 B 是什么关系？
N2 = EVOLUTION: 收购前 A 是否已有该能力？
                哪个事件改变了这一状态？
                收购后形成了什么新能力，何时生效？
N3 = TEXT:      哪些原文分别支撑前态、事件和后态？
```

这正体现三通路不是一次性融合，而是随着分析状态交替调用。

### 5.2 建议采用“三类主知识需求＋演化子结构”

主类型不宜一开始拆成大量 QA 标签。第一版只需三类，与三条知识通路直接对应：

| 主类型 | Planner 的判断问题 | 典型子需求 | 主要通路 |
|---|---|---|---|
| `TEXTUAL_GROUNDING` | 当前判断缺少哪段原文、数字、条件或直接陈述？ | direct statement、number、quote、condition、source mention | 原文/向量 |
| `RELATIONAL_SEMANTICS` | 当前缺少哪个实体、关系或多跳连接？ | entity identity、relation、bridge entity、comparison counterpart | 语义/KG |
| `EVENT_EVOLUTION` | 当前实体或关系的状态是怎样形成和变化的？ | state_before、transition_event、time、state_after、supersedes、denies、confirms | 事件演化 |

`EVENT_EVOLUTION` 是本文相对 S2G-RAG 等静态 gap schema 最重要的扩展。它不是把所有带日期的信息归为 event，而是要求检查一条状态演化链：

```text
目标对象或关系是什么？
  → 变化之前是什么状态？
  → 发生了什么事件，参与者、动作、对象和时间是什么？
  → 事件对哪个实体/关系产生了什么 effect？
  → 变化之后是什么状态？
  → 是否存在终止、替代、否认、纠正或并行冲突？
```

这组子槽位可以部分缺失。例如，当前已经知道“系统 B 取代了系统 A”，但不知道替代发生的时间和触发事件，则 Planner 输出的不是泛化的 `need more information`，而是：

```json
{
  "knowledge_type": "EVENT_EVOLUTION",
  "target_relation": "system_A REPLACED_BY system_B",
  "known_slots": ["state_before", "state_after"],
  "missing_slots": ["transition_event", "temporal_anchor"],
  "preferred_actions": ["SEARCH_EVENT", "VERIFY_TEXT"]
}
```

`PROVENANCE`、`STANCE` 和 `VALID_TIME` 更适合作为三类知识共有的属性，而不是第四、第五条通路：任何结构化实体、关系或事件都应能够回到 passage/document/source，并标明支持、否认或不确定立场以及适用时间。

### 5.3 怎样判断“缺失”

对当前阶段需求集合 \(G_t=\{g_1,\ldots,g_m\}\) 和已获得知识 \(K_t\)，建立知识单元—需求覆盖矩阵：

\[
M_{ij}=\operatorname{support}(k_j,g_i)\in[0,1]
\]

`support` 不应只使用向量相似度，而应综合：

- 证据是否包含目标实体和必要约束；
- 证据对需求命题是支持、反驳还是无关；
- 时间范围是否一致；
- 图路径是否连通要求的端点；
- 事件是否包含必要参与者、动作、对象或结果；
- 是否存在可回溯原文。

需求 \(g_i\) 的覆盖度可定义为：

\[
c_i(K_t)=\max_{k_j\in K_t}M_{ij}
\]

对于必须由多份证据联合满足的需求，则使用路径或集合覆盖函数，而不能简单取最大值。缺口集合为：

\[
\mathcal{G}_t=\{g_i\mid c_i(K_t)<\tau_i\}
\]

同时应保留三种不同状态：

- `UNCOVERED`：尚未检索到足够证据；
- `CONFLICTED`：已经找到相互冲突的证据，需要继续消歧或并列呈现；
- `UNRESOLVABLE_WITHIN_BUDGET`：在当前语料和预算内未能解决。

必须避免把“检索没有找到”直接写成“语料中不存在”。除非是封闭且可穷尽的语料，否则系统只能说未检得，不能证明不存在。

### 5.4 知识缺口判断的实现方式

最稳妥的是“结构规则＋模型判断”的混合方案：

1. 确定性检查负责实体、日期、比较对象、引用和图连通性；
2. NLI 或 LLM judge 负责语义支持、反驳和条件一致性；
3. 输出必须符合固定 schema，而不是自由文本 `missing_information`；
4. 每个缺口必须带有 `analytic goal`、`knowledge type`、`target proposition/relation`、`known slots`、`missing slots` 和 `confidence`；
5. 低置信判断可触发并行探索，但不能伪装成已确认缺口。

一个面向研判的可执行 NeedItem 可以写成：

```json
{
  "need_id": "n3",
  "knowledge_type": "EVENT_EVOLUTION",
  "analytic_goal": "determine how A's operational status changed",
  "target_relation": "A REPLACED_BY ?successor",
  "known_anchors": ["A", "2025"],
  "known_slots": ["state_before"],
  "missing_slots": ["transition_event", "state_after", "temporal_anchor"],
  "constraints": {"time_end": "2025-12-31"},
  "preferred_paths": ["EVENT", "SEMANTIC", "TEXT"],
  "confidence": 0.86
}
```

这比当前单字段的 `missing_information` 更可审计，也能直接支持通路规划和过程评测。

---

## 六、如何规划通路检索与查询改写

### 6.1 采用无需训练的显式 Planner 是否合理

合理，而且可以成为相对 RouteRAG 的重要方法取舍，但必须把“无需训练”落实为可复现的结构化规划协议，而不是一句 Prompt。

传统 content-driven retrieval 主要从 query 与文档内容的词汇或向量相似度出发：

```text
query content → similar documents → answer/reason
```

本文建议的 knowledge-need-driven workflow 先判断当前分析需要哪种知识结构：

```text
analytic state
  → required knowledge type and missing evolution slots
  → path-specific retrieval action
  → grounded knowledge increment
  → updated analytic state
```

Planner 可以直接使用通用 LLM 的结构化输出能力，不做任务特定的 SFT 或 RL。为了防止它退化成不可复现的自由推理，方法中至少需要：固定 Knowledge Need schema、受约束的动作集合、确定性字段校验、最大轮数/成本、同查询缓存和完整轨迹日志。

Training-free 的好处是：

1. **跨域可迁移。** 不依赖 HotpotQA 等特定任务的 gold answer 或轨迹标注，更适合科研、产业、政策等缺少训练数据的研判语料；
2. **可解释。** 每次检索都能说明当前缺少哪种知识、为什么选择该通路，以及新知识关闭了哪个槽位；
3. **模块可替换。** LightRAG、其他 KG retriever 或事件检索器可以替换而不重新训练统一 policy；
4. **成本可控。** 不承担 RouteRAG 的两阶段 RL 和 S2G-Judge 的轨迹蒸馏成本，适合先验证研究假设；
5. **更适合长程 Agent。** 研判目标和知识源会变化，显式状态比固定任务上学出的隐式动作 policy 更容易扩展。

相应的风险也必须承认：zero-shot Planner 可能分类不稳定，未必达到训练式 policy 的路由效率；schema 设计可能带来人工偏置；不同基础模型可能输出不同计划。因此实验必须比较 training-free、few-shot/SFT 和 RouteRAG 式 trained policy，并报告 Planner 一致性和通路决策错误。

### 6.2 通路选择不应是硬编码的一对一映射

对每个动作 \(a=(p,r)\)，其中 \(p\) 是通路，\(r\) 是该通路的查询或扩展操作，估计其效用：

\[
U(a\mid A_t,K_t,\mathcal{G}_t)=
\Delta\operatorname{Coverage}
+\lambda_p\Delta\operatorname{PathComplete}
+\lambda_c\Delta\operatorname{ConflictResolve}
-\lambda_d\operatorname{Redundancy}
-\lambda_k\operatorname{Cost}
\]

系统选择预期效用最高的一项或若干项，而不是看到 `TEMPORAL_ANCHOR` 就机械地只调用事件通路。这样可以处理事件抽取漏掉日期、但原文文本路仍能找到日期的情况。

### 6.3 三条通路的查询语言必须不同

如果第二轮只是把同一个自然语言 query 同时发送给三个 retriever，路由的研究意义会很弱。应针对每个证据空间生成不同的检索表达。

#### 文本发现通路

保留原问题中的实体、日期和限定条件，围绕未满足命题生成一条自包含查询，并加入必要的词汇扩展：

```text
原问题：A 项目后来由什么替代？
文本改写："A project" replacement successor phased out replaced by
```

文本路适合找原话、数字、引语、条件和图中漏抽的事实。查询应尽量自包含，不能只写“它后来怎样了”。

#### 关系结构通路

将缺口转成图模式或受约束的关系目标：

```text
anchors: [A]
target_relation: REPLACED_BY | SUCCESSOR_OF
unknown_endpoint: object
max_hops: 2
required_entity_types: [SYSTEM, PROJECT]
```

若缺少的是多跳桥，改写重点不是复述原问题，而是指定已知端点、未知端点、关系语义和跳数预算。

#### 事件转移通路

将缺口转为 actor—action—object—time—state 的事件查询：

```text
actors: [A]
event_types: [replacement, termination, deployment]
time_range: [2024, 2026]
state_intents: [change, latest, superseded]
expand: [predecessor, successor]
```

如果要找否认或纠正，还应显式传入 `stance=[denied, corrected]`，而不是依赖向量查询碰巧命中否定句。

### 6.4 初始获取与后续动态补全的分工

建议采用“宽入口、窄补全”的两阶段策略：

1. 初始 Planner 根据任务预估需要的三类知识，可以小预算并行建立多视图种子；对于明显简单的问题，可关闭事件路或关系路。
2. Agent 每完成一个阶段性分析后更新 Knowledge Need State。后续不再全路重复检索，而是针对新的分析目标选择一至两条预期增益最大的通路。
3. 如果结构化通路命中，必须使用原文路执行 `VERIFY_TEXT`；如果原文发现新的实体或事件，则反向送入关系或事件通路扩展。
4. 新知识进入分析状态后再次判断，而不是只在预设的第二轮结束。

这样既保留三通路初始协同的稳定收益，又使 Agentic 部分真正减少无效重复，而不是单纯增加一次 LLM 调用。

---

## 七、三条通路的结果如何真正“协同”

### 7.1 协同不是把分数相加，而是互相生成检索锚点

三条通路应通过共享 `document_id`、`mention_id`、`entity_id` 和 `event_id` 建立双向映射：

- 文本命中 → mention → 对齐 entity/assertion/event，给图路和事件路提供规范化锚点；
- KG 命中 → relation/path → 回溯 supporting mentions，并把路径端点交给文本路寻找细节；
- Event 命中 → participants/time/state → 连接相关实体关系，并扩展前序、后序或冲突事件；
- 任何规范化节点 → 原文 mention → 最终回答只引用来源文本，而不让抽取摘要自我证明。

这才是 cross-path completion。若三路只能在最后按文档 ID 做 RRF，则应诚实称为 multi-view late fusion，而不是协同推理。

### 7.2 从文档排名转向知识需求覆盖与证据组合选择

最终目标不应只是选择单篇最高分文档，而是选择能够共同覆盖当前阶段知识需求的证据集合 \(S\)：

\[
\max_{S: \operatorname{cost}(S)\le B}
\sum_{g_i\in G_q}w_i\max_{e\in S}\operatorname{support}(e,g_i)
+\lambda_l L(S)
+\lambda_t T(S)
+\lambda_s D_{source}(S)
-\lambda_r R(S)
\]

其中：

- \(L(S)\)：关系桥或事件转移路径的完整度；
- \(T(S)\)：问题要求的时间点、前态和后态覆盖；
- \(D_{source}(S)\)：来源覆盖或独立性；
- \(R(S)\)：语义重复和同源转载造成的冗余；
- \(B\)：文档数、token、延迟或调用成本预算。

RRF 可以保留为候选级基线，但主方法应当体现“已经选中的证据会改变下一份证据的边际价值”。例如，已有三份都证明事件发生，再加入第四份同义报道的价值很低；此时一份说明事件结果或否认情况的证据更有价值。

### 7.3 停止条件

Agent 应在以下任一条件满足时停止：

- 当前分析目标所要求的全部硬性知识槽位达到覆盖阈值；
- 候选动作的预期覆盖增益低于阈值；
- 连续两轮没有新增知识类型或演化槽位；
- 达到 token、检索次数或延迟预算；
- 当前证据存在不可消解冲突，系统转为带冲突回答；
- 在当前预算内不可解决，输出证据不足而不是强行回答。

“最多两轮”可以作为安全上界，但不能作为论文中的主要停止逻辑。

---

## 八、整个方法真正的难点在哪里

### 难点一：知识需求的分解必须可验证

LLM 很容易生成看起来合理、实际上超出任务要求的缺口。Knowledge Need State 必须从任务、当前分析和已获得材料中更新，而不是引入模型自己的常识假设。比较问题、复合问题和隐式时间问题尤其容易漏槽位。

### 难点二：缺失类型判断会受当前检索噪声影响

Judge 既可能把“已存在但位置靠后”的证据判为缺失，也可能被一段语义相似但不蕴含答案的文本误导。需要对 gap diagnosis 单独标注和评价，不能只用最终答案正确率间接证明。

### 难点三：图与事件抽取不是可靠真值

KG 和事件表示来自信息抽取，会漏抽、错连或错误合并。文本路必须始终作为 recall fallback；所有结构化命中必须回到原文验证。事件时间邻近只能表示相关或先后，不能直接当因果。

### 难点四：路由决策存在“能力—成本—增益”权衡

图遍历可能适合多跳关系，但并不保证比文本搜索更便宜；事件路对时间问题有优势，但事件索引可能稀疏。路由器应学习或估计预期边际覆盖，而不是使用固定问题类型规则。

### 难点五：跨通路分数不可直接比较

向量相似度、图路径分数和事件结构得分含义不同。主方法不应强行做线性归一化，而应先将候选映射为“满足了哪类知识需求和哪些演化槽位”，再进行集合级选择。

### 难点六：多来源不等于多份独立证据

多篇转载可能来自同一个原始消息。若论文声称“多源验证”，就必须至少识别同源转载或引用链；否则应保守地称为 multi-document evidence，而不是 independent corroboration。

### 难点七：效率可能抵消准确率收益

Agentic 循环增加 LLM 判断、改写和重复检索成本。论文必须报告平均轮数、通路调用率、token、p95 延迟和单位覆盖增益，而不能只报告最终准确率。

---

## 九、可以写进论文的贡献

建议将贡献压缩为四点，并按“问题—方法—协同—评测”排列。

1. **问题形式化。** 指出现有多轮检索的缺口 schema 主要覆盖静态实体、属性、关系和证据片段，进一步将实体/关系的事件驱动状态演化定义为一等知识需求，并将其分解为前态—事件—时间—后态—替代/否认槽位。
2. **Training-free 知识需求规划。** 设计可审计的 Knowledge Need State，随研判过程显式更新当前目标、已有判断和未满足知识，在无需任务特定训练的条件下选择原文、语义关系或事件演化通路。
3. **跨通路知识协同。** 利用 mention—entity/assertion—event 的共享来源映射，使一条通路的命中成为另一条通路的检索锚点，并使事件检索结果能够解释和更新实体/关系状态，最终回到原文验证。
4. **单轮与多轮统一评价。** 同时检验三类数据组织对单轮知识覆盖的贡献，以及动态 Planner 对多轮研判轨迹、演化链完整度、知识增益和成本的贡献。

不要写：

- 首次融合 KG 与 Event；
- 首次根据证据不足进行查询改写；
- 首次在文本与图之间动态路由；
- 事件时间顺序等同于因果关系；
- 检索未找到等同于语料中不存在。

---

## 十、当前实现与新方法之间还有多远

当前实现已经具备一个可靠起点：

- 文本、关系图和事件三路共享稳定文档身份；
- `ours` 相对 `light_entity` 只增加事件索引、事件检索和三路融合，消融边界清楚；
- Agentic 层具有 retrieve—assess—rewrite—answer 的有界循环；
- 输出记录检索轮数、LLM 调用、token 和 evidence trace。

但当前 Agent 仍是通用充分性判断：`EvidenceAssessment` 只包含 `relevant`、`sufficient`、`reason` 和自由文本 `missing_information`；改写后会再次调用同一个底层检索器；不同通路没有独立动作，也没有验证第二轮是否补齐某类缺口。因此，当前版本更适合作为 **generic agentic rewrite baseline**，还不是拟议方法本身。

从现有 60 问、200 文档评测看，三通路的优势主要体现在证据完整性，而不是首条命中排序：

| 对比 | Evidence Recall@10 | Complete Recall@10 | MRR@10 |
|---|---:|---:|---:|
| `light_entity` | 0.7591 | 0.4565 | 0.7707 |
| `ours` | 0.8007 | 0.5652 | 0.7230 |
| `agentic_light_entity` | 0.7482 | 0.3913 | 0.7508 |
| `agentic_ours` | 0.8025 | 0.5435 | 0.7123 |

相对 `light_entity`，静态三通路的 Evidence Recall@10 提高约 4.16 个百分点，Complete Recall@10 提高约 10.87 个百分点，但 MRR@10 下降约 4.77 个百分点。相对 `agentic_light_entity`，`agentic_ours` 的 Complete Recall@10 提高约 15.22 个百分点，但 MRR@10 仍下降约 3.85 个百分点。

更关键的是，`agentic_ours` 并未稳定超过静态 `ours`：两者 Evidence Recall@10 基本相当，Agentic 版本的 Complete Recall@10 和 MRR@10 反而略低，而平均检索轮数为 1.67、平均 LLM 调用数为 3.33。这说明当前最有力的经验结论是：

> 事件通路和三路数据组织显著改善了多证据覆盖，但通用的“判断不足—整体改写—全路重检索”尚未产生稳定增益。

这不是坏消息，反而直接构成新方法的 motivation：现有 Agent 没有区分当前缺少原文、实体关系还是状态演化，也没有做针对性的通路动作，所以增加轮次未必获得真正推动分析的新知识。

---

## 十一、最小可发表算法版本

不建议现在重做高成本统一 TKG。可以在现有三路索引上增加一个较小、可控的 Agentic 控制层：

### 11.1 数据结构改动

将 `missing_information: str` 替换为：

```text
KnowledgeNeedState
  analytic_goal: str
  sufficient: bool
  established_findings: list[Finding]
  covered_knowledge: list[CoverageItem]
  needs: list[NeedItem]
  evolution_slots: list[EvolutionSlot]
  conflicts: list[ConflictItem]
  stop_reason: enum
```

每个返回文档增加：

```text
knowledge_type
supported_need_ids
affected_entity_or_relation
state_before / state_after
stance
temporal_scope
provenance_ids
```

### 11.2 控制动作

```text
SEARCH_TEXT(need, query)
SEARCH_GRAPH(need, anchors, relation_pattern, hops)
SEARCH_EVENT(need, actors, event_types, time, state, expansion)
VERIFY_TEXT(structured_hit)
EXPAND_CROSS_PATH(seed)
UPDATE_ANALYTIC_STATE(new_knowledge)
STOP_SUFFICIENT
STOP_BUDGET
```

### 11.3 每轮流程

```text
1. Parse the task into an initial Knowledge Need State.
2. Run small-budget initial retrieval over required paths.
3. Analyze current knowledge and produce an intermediate finding.
4. Diagnose TEXT / RELATION / EVOLUTION needs and missing evolution slots.
5. Execute one or two highest-utility path-specific actions.
6. Map new hits across mention/entity/event views and verify structured hits in text.
7. Update the analytic state and the Knowledge Need State.
8. Stop if the current analytic goal is supported; otherwise continue within budget.
```

这套增强不要求先验证昂贵的全量 TKG 新索引，也能把现有三路结果和 Agentic 框架转化成一个明确的新检索算法。

---

## 十二、实验怎样设计才能证明创新点

### 12.1 核心消融

至少需要以下方法：

1. Dense/BM25 hybrid；
2. LightRAG 或 `light_entity`；
3. 固定三路＋RRF；
4. 固定三路＋统一 query rewrite；
5. query-only router：只根据原问题选通路；
6. sufficiency＋free-text gap：接近当前实现；
7. S2G 风格静态 gap：entity/attribute/relation/evidence span；
8. 静态 gap＋`EVENT_EVOLUTION`；
9. 三类知识需求＋route-specific action；
10. 三类知识需求＋cross-path completion；
11. 完整的 training-free 动态研判工作流。

这样可以分别回答：事件索引是否有用、Agent 循环是否有用、增加状态演化缺口是否有用、通路专用动作是否有用、跨路补全是否有用。

### 12.2 必须分别比较单轮与多轮

**单轮实验**回答“数据组织是否有用”。所有方法只允许一次 query planning 和一次检索预算，重点比较：

- Dense/BM25、LightRAG、EventRAG、E²RAG、固定三路和本方法的单轮模式；
- Evidence Recall、Complete Recall、MRR/nDCG；
- 关系路径覆盖和 evolution-chain coverage；
- 相同 top-k、token 和索引条件下的查询成本。

单轮实验尤其要证明：即使不依赖 Agent 多轮优势，事件演化知识和三路连接也能提高时间、变化和替代类问题的完整覆盖。

**多轮实验**回答“动态知识需求规划是否有用”。重点比较：

- 当前 generic rewrite Agent；
- IRCoT/S2G-RAG 式文本多轮；
- RouteRAG 式训练路由，若复现实验成本可以承受；
- EventRAG 的事件图内多轮；
- 本文 training-free 三通路 Planner；
- oracle knowledge-type/route 上界。

多轮指标除最终答案外，还要报告每轮新知识增益、Knowledge Need classification、Evolution Gap Closure、平均通路调用数、停止准确率、轨迹长度和成本。这样才能证明收益来自“根据研判状态找新知识”，而不是简单多给了几次检索机会。

### 12.3 不能只报告普通 Recall 和答案准确率

建议增加：

| 评测对象 | 指标 |
|---|---|
| 缺口诊断 | Gap-type Macro-F1、missing-slot F1、sufficiency accuracy、ECE/Brier |
| 通路规划 | Route accuracy、oracle-route regret、每类 gap 的 route success rate |
| 单轮收益 | ΔEvidence Recall、ΔKnowledge Coverage、new-knowledge-per-call |
| 知识链 | Complete Recall、relation-path coverage、evolution-chain coverage、redundancy、source coverage |
| 排序 | MRR、nDCG、MAP |
| 端到端回答 | EM/F1、citation precision/recall、faithfulness、insufficient-evidence accuracy |
| 效率 | 平均轮数、各通路调用率、token、p50/p95 latency、cost per solved query |

其中最重要的新指标是 `Evolution Chain Coverage` 和 `Need Closure Rate`：

\[
\operatorname{NeedClosureRate}=
\frac{|\mathcal{G}_t|-|\mathcal{G}_{t+1}|}{|\mathcal{G}_t|}
\]

如果一轮检索没有满足任何已预测知识需求，也没有补齐演化槽位，即使召回了更多相关文档，也不应被视为有效 Agentic 行为。

### 12.4 数据标注

现有 MultiHopRAG 可以继续用于端到端对比，但不足以单独验证“缺失证据类型”。建议从现有问题抽取一个小而精的过程标注集，为每题标注：

- Knowledge Need State；
- 每份 gold evidence 的知识类型和演化槽位；
- 不同部分证据池下的真实 gap；
- 对每个 gap 有效的通路动作；
- 完整回答所需的最小证据集合。

可以通过对 gold evidence 做受控删除自动构造中间状态，再由人工复核：删去原文句生成 `TEXTUAL_GROUNDING` 需求，删去桥接文档生成 `RELATIONAL_SEMANTICS` 需求，分别删去前态、事件、时间或后态生成不同的 `EVENT_EVOLUTION` 子缺口。这样既降低标注成本，又能直接评价 Planner。

### 12.5 统计要求

目前 60 问结果适合做机制验证，不足以支撑强泛化结论。正式论文至少应：

- 在多个随机种子或更大测试集上报告置信区间；
- 对 Complete Recall、MRR 和最终答案做 paired bootstrap 或随机化检验；
- 分问题类型报告结果，尤其是 temporal、comparison 和 inference；
- 单独报告事件抽取覆盖率，避免把上游漏抽误归因为检索策略失败；
- 报告索引构建成本和查询成本，但将高成本 TKG 作为扩展而非主依赖。

---

## 十三、建议的论文标题与一句话故事

### 推荐标题

**EvoNeed-RAG: Training-Free Knowledge-Need Planning over Text, Relational, and Event Memories**

中文：

**EvoNeed-RAG：面向原文、关系与事件记忆的免训练知识需求规划与协同获取**

备选：

- **TriKnow-RAG: Training-Free Cooperative Knowledge Acquisition over Text, Graph, and Event Paths**
- **Beyond Static Gaps: Evolution-Aware Planning for Agentic Text–Graph–Event Retrieval**

不建议标题中使用 `TKG` 作为中心词，因为这会把审稿焦点引向索引构建成本、时态逻辑完备性和 TKG completion，而不是当前最强的检索贡献。

### 一句话故事

> 现有多轮 RAG 已经会判断还缺哪个实体、属性或关系，但较少判断实体和关系的状态如何随事件演化；我们把原文、语义关系和事件演化作为三类知识需求，使 Agent 在持续研判中显式判断缺什么知识、去哪条通路获取，并用新知识更新下一步分析。

### 论文摘要式表述

> 持续研判要求 Agent 随分析进展不断获取新知识。现有多轮 RAG 已能判断缺少桥接实体、属性、静态关系或证据片段，事件 RAG 也能组织时间相关事件；然而，实体及关系如何随事件发生状态演化，尚未成为驱动异构知识获取的一等需求。本文提出 EvoNeed-RAG，一种无需任务特定训练的三通路知识需求规划框架。该方法维护随研判过程更新的 Knowledge Need State，将当前未知显式区分为原文落证、实体关系和事件演化，并把演化需求进一步分解为前态、变化事件、时间锚点、后态以及替代/否认槽位；Planner 据此调用原文、语义图或事件通路，利用 mention—entity—event 映射执行跨通路补全，并将新增知识重新写入分析状态。该框架将 Agentic RAG 从围绕初始 query 的重复改写，推进为由当前知识状态驱动的持续知识获取过程。

---

## 十四、最终判断

这个方向值得优先于高成本 TKG 新方法继续推进，原因不是它“简单”，而是它的风险与证据更匹配：现有实验已经表明三通路对完整证据召回有明显价值；当前 Agentic 循环没有稳定增益，又为“为何需要类型化缺口与定向路由”提供了直接问题证据；所需增强主要发生在查询时控制层，可以复用现有索引和评测框架。

但必须守住一个边界：

> 数据组织提供原文、语义关系和事件演化三种互补知识空间；Agentic Planner 根据当前研判状态规划知识获取，并让三类知识互相落证、连接和更新。前者是必要基础，后者才是主创新。

只讲数据组织，会落入 LightRAG、SiReRAG、EventRAG 和 E²RAG 的密集竞争；只讲 Agentic 改写，会落入 IRCoT、RouteRAG 和 S2G-RAG 的已有范式。把论文放在“**状态演化知识缺口 × training-free 动态规划 × 原文—关系—事件跨通路协同**”这个交叉点，才是目前最有希望站住、也最能解释现有效果的版本。
