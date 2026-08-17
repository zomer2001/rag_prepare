# 从研究空白重构 Ours_TKG：GAP、论文叙事、数据集与实验路线

更新日期：2026-08-13

## 一、先给结论

1. **不能把主创新写成“首次把 KG index 和 Event index 捏在一起”。** EventRAG 已使用事件知识图谱和跨文档事件融合；E²RAG 更直接构造实体—事件双图及其映射；DyG-RAG、IA-RAG 分别覆盖事件链和区间时间检索。因此，“统一 TKG”本身只是系统形态，不足以构成研究问题。
2. **更稳的知识表示定位是：evidence-grounded temporal assertion index（证据落地的时态断言索引）。** 它不是简单存 Entity 和 Event，而是把 `Assertion—EventEffect—Event—Mention—Source` 组织成可检索的事实状态变化路径。
3. **更稳的检索定位是：query-conditioned cooperative evidence-path retrieval（查询条件化的协同证据路径检索）。** 三通路不只是各自找结果并 RRF，而是共同补全一个在相关性、状态演化、来源证据上都足够的 evidence portfolio。
4. **“Mention 回溯证据”可以讲。** 本工作的目标限定为数据关联、融合、检索与原文证据回溯，不再把来源可信度或置信度校准作为贡献。
5. **“Agentic indexing”目前也不宜作为主创新。** 当前更准确的说法是 LLM-assisted、governed incremental indexing workflow。只有系统能自主决定新增、合并、更新、冲突、回滚、重试和人工升级，Agentic 才是可验证的机制。
6. **没有一个成熟公开数据集完整覆盖“科研＋多源＋演化＋冲突＋来源可追溯＋证据链 gold”。** 最合理的实验不是押一个数据集，而是“两个公开主 benchmark＋一个自建交叉能力子集”。

建议把论文的核心问题写成：

> Given a multi-source, temporally evolving corpus, how can a retriever recover a minimal yet sufficient provenance-grounded evidence subgraph that jointly covers relevant assertions, their event-conditioned state transitions, and independent source evidence under a fixed retrieval budget?

中文：

> 面向多源、持续演化的文档集合，如何在固定检索预算下，恢复一个最小但充分的可溯源证据子图，使其同时覆盖相关断言、事件驱动的状态变化和相互独立的来源证据？

---

## 二、不结合当前方法，文献中还能摸出的创新 GAP

这些 GAP 是先从现有研究的失败模式出发，再讨论是否与 Ours_TKG 相交。

| GAP | 已有研究做到哪里 | 仍缺什么 | 创新潜力 | 难度 | 可评价性 |
|---|---|---|---:|---:|---:|
| **G1：从“相关文档排名”转向“互补证据组合检索”** | BRIGHT 证明 reasoning-intensive retrieval 远难于语义匹配；2026 年 BRIGHT-Pro 开始为不同 reasoning aspect 标注互补 gold evidence | 检索目标仍多是逐文档相关性；缺少在预算下联合优化充分性、互补性、时间覆盖、来源独立性和冗余的 evidence portfolio | 5 | 4 | 5 |
| **G2：跨来源 Mention 的类型化知识对齐** | EventRAG/PaperQA2/SciFact 能找多源支持或冲突；常见系统以相似度或实体共现合并 | 缺少对 duplicate、refine、temporal update、supersede 与 concurrent conflict 的统一结构区分 | 5 | 4 | 5 |
| **G3：开放域双时态知识检索与可复现性** | Temporal RAG 处理事件时间；Beyond Retrieval 在纵向医疗记忆中区分 valid time 与 transaction time | 开放域文档 RAG 很少同时回答“何时为真”和“当时系统知道什么”，也很少处理迟到证据、回溯修正、撤回后重放 | 4 | 5 | 4 |
| **G4：科学主张的生命周期检索** | SciFact/SciFact-Open 做支持—反驳；PaperQA2 做文献矛盾发现；OpenScholar 做多论文综合 | 缺少 `提出→复现→限制→反驳→修正/撤回→新共识` 的 claim lineage、状态变化及证据链检索 benchmark 和方法 | 5 | 5 | 4 |
| **G5：批量建库之外的连续索引一致性** | 很多工作讨论动态知识或增量 memory | 缺少对 entity resolution、事件共指、断言状态、图边、向量索引的原子更新、回滚和历史版本一致性评估 | 4 | 5 | 3 |
| **G6：不足证据、缺失证据和否定证据的区分** | 一些 QA 数据集包含 unanswerable，claim verification 包含 refute | 系统通常无法区分“语料中不存在”“检索没找到”“存在反证”“证据暂不足”，更缺少结构化、可校准的停止条件 | 4 | 4 | 4 |

### 最值得关注的三个独立方向

#### 方向 A：Evidence Portfolio Retrieval

研究单位从一个 relevant document 改为一个**具有角色分工的证据集合**：背景、直接支持、反证、时间前态、时间后态、来源证明。BRIGHT-Pro 已经把“多方面 gold evidence”带进 benchmark，但还没有解决 temporal transition、source independence 和 graph-constrained portfolio selection。这是目前最一般、最容易形成检索论文的问题。

#### 方向 B：Typed Multi-source Knowledge Alignment

多源文档融合的关键不是评价哪个来源更可信，而是区分不同 mention 之间究竟是什么结构关系：

- 同一事实的重复表述；
- 对既有事实的条件补充或细化；
- 随时间发生的状态更新；
- 新断言对旧断言的替代；
- 同一时间和条件下的真实冲突。

这要求所有 Mention 保留并回到原文，同时在 canonical Assertion/State 层执行 typed alignment，不能把 duplicate、update、supersede 和 conflict 都压成普通合并。

#### 方向 C：Scientific Claim Evolution Retrieval

科学研判不是问“论文说了什么”，而是问：某一主张如何形成、哪些条件下成立、后来为何被限制或推翻、截至某时点应采用什么判断。它天然需要时间、事件、断言、证据和来源关系，也是现有科研 RAG 数据集最缺的交叉区。

### 独立视角下最推荐的 GAP

如果完全不考虑现有实现，我最推荐：

> **Query-conditioned evidence portfolio retrieval under temporal and cross-source alignment constraints.**

理由是它把“检索”本身重新定义为集合/子图优化，而不是增加一种索引；可以横跨科研、新闻、政策、金融与情报分析；同时能用明确的充分性、互补性、时间覆盖和来源独立指标评价。

---

## 三、你的新叙事哪些成立，哪些需要改

### 3.1 “其他是 KG index 或 Event index，我们是统一 TKG-index”

**判断：方向对，但表述过宽，容易被反例击穿。**

最直接的反例包括：

- [EventRAG](https://aclanthology.org/2025.acl-long.830/) 已把事件作为知识图中的核心对象，进行跨文档融合和多事件推理；
- [E²RAG](https://aclanthology.org/2026.eacl-long.90/) 已显式组织实体图、事件图和二部映射，目的正是防止实体合并压缩动态上下文；
- [DyG-RAG](https://arxiv.org/abs/2507.13396) 已使用动态事件单元、时间增强表示和事件时间线检索；
- [IA-RAG](https://arxiv.org/abs/2606.06044) 已形式化区间事件和时间遍历；
- [Beyond Retrieval](https://aclanthology.org/2026.knowfm-1.10/) 已在纵向医疗记忆中使用双时态状态和 supersede/conflict 仲裁。

因此，不能把差异写成：

> 别人只建 KG 或 Event，我们第一次统一二者。

应改成：

> 现有工作分别强化实体关系、事件序列、区间时间或状态仲裁，但在开放域多源文档检索中，尚未充分把**可重述的时态断言、改变断言状态的事件、以及落到原文与来源的 mention**组织成同一个可优化的 evidence path。我们构造统一的 evidence-grounded temporal assertion index，并在其上进行跨视图证据链检索。

“TKG-index”仍可作为简称，但正式定义要突出五层对象：

```text
Entity
  └─ Assertion(subject, predicate, object, valid interval, state)
       ├─ AssertionMention → passage/document/source
       └─ EventEffect(CREATE/UPDATE/TERMINATE/SUPERSEDE/DENY/CONFIRM)
            └─ Event(time/interval)
                 └─ EventMention → passage/document/source
```

真正有区分度的不是 Entity 和 Event 共存，而是：

```text
Event → changes → Assertion state → grounded by → Mentions/Sources
```

### 3.2 “KG 关联路径＋Mention 回溯”

需要拆成两句话：

1. **结构扩展：成立。** 可以从 Entity/Assertion/Event 沿类型约束的边找相关事实和演化关系。
2. **Mention 回溯：成立。** 可以从规范化对象回到原始 passage 和 document，支持 citation grounding。
“谱系”也应谨慎使用。若未实现 citation/data/method lineage，称 typed neighborhood、association path 或 assertion lifecycle path 更准确。Mention backlink 表示证据可追溯，不包含对来源质量的判断。

### 3.3 “可完成多源、有历程、有周期的复杂研判”

**可以作为目标能力，不能直接作为已有结论。** 建议将“周期”改成“纵向演化”或“生命周期”。如果确实要表示周期性，还需显式建模 recurrence、周期粒度、每次 occurrence 和 pattern，而不能依靠普通时间边。

更严谨的能力声明是：

> 面向多源纵向语料，方法检索实体关系、事件驱动的事实状态迁移和原始来源证据，支持 state-at-time、evolution、replacement、conflict 和 provenance 类研判问题。

### 3.4 三通路应该怎样重新定义

建议不要继续使用模糊的 Vector—Semantic—Event 命名，而是按**证据角色**定义：

| 通路 | 主要对象 | 任务 | 不应承担的任务 |
|---|---|---|---|
| **Text Discovery Path** | Chunk/mention | 发现表述差异大、尚未规范化或图中漏抽取的候选证据 | 不负责断言真值与时间状态 |
| **Assertion Structure Path** | Entity/Predicate/Assertion | 恢复稳定关系、约束多跳、组织同一断言的不同 mention | 不把单一图边当永久真值 |
| **Event Transition Path** | Event/EventEffect/temporal edge | 恢复某断言怎样产生、更新、终止、被否认或替代 | 不把时间相邻自动解释为因果 |

三通路共享一个 TKG 和 provenance store；五种向量入口可以保留为工程实现，但论文层面是三类 evidence role。

### 3.5 仅混合检索仍不够，必须有“协同”

如果当前过程仍是：

```text
各通路独立 top-k → RRF → 返回文档
```

Reviewer 会把它看作 multi-index late fusion。建议补成：

1. **Query Contract Parsing**：解析目标实体/断言、时间范围、问题意图和所需 evidence roles；
2. **Multi-view Seeding**：Chunk、Assertion、Event 分别提供高召回种子；
3. **Cross-path Evidence Completion**：
   - Chunk seed → Mention → Assertion/Event；
   - Assertion seed → EventEffect → 前态/后态 Event → Mentions；
   - Event seed → affected Assertions → before/after state → Sources；
4. **Budgeted Portfolio Selection**：在 token/node budget 下联合优化相关性、路径完整度、时间覆盖、多源证据覆盖，并惩罚冗余；
5. **Evidence Adequacy Verification**：检查必要角色是否齐全，不齐则定向扩展或输出 insufficient evidence。

一个可实现的目标函数是：

```text
score(S|q) =
  λr · relevance(S,q)
+ λp · lifecycle_path_completeness(S,q)
+ λt · temporal_coverage(S,q)
+ λs · multi_source_evidence_coverage(S,q)
+ λc · conflict_visibility(S,q)
- λd · redundancy(S)
- λb · budget_cost(S)
```

这比 RRF 更像方法贡献，因为各候选项的效用依赖已经选择的证据集合。

---

## 四、索引组织是否可以叫 Agent 工作流

### 当前最合适的表述

> A governed, LLM-assisted incremental indexing workflow builds and maintains a unified temporal assertion index.

它可以包含 extraction、entity/event resolution、assertion fusion、embedding refresh、graph validation 和 index update，但“用了 LLM”和“分了多个步骤”不自动等于 Agent。

### 若要升级成 Agentic Index Maintenance，至少需要

- 明确状态和工具：抽取、查重、实体解析、冲突检测、图更新、向量更新、验证、回滚；
- 自主决策：NEW / DUPLICATE / UPDATE / SUPERSEDE / CONFLICT / ABSTAIN；
- 约束验证：时间区间、一致性、悬空 mention、非法状态迁移；
- 失败恢复：重试、替代 extractor、人工升级；
- 可审计轨迹：每次图修改的依据和前后版本；
- 评估：决策准确率、错误传播、更新延迟、回滚成功率、人工介入率。

### 建议

本论文仍以**知识表示＋检索**为主。Agentic indexing 放为系统组件或扩展实验，不要与三通路、多源融合、时态建模同时争主创新，否则贡献会发散。

---

## 五、哪些更难的数据集能撞上创新点

### 5.1 公开数据集分级

| 数据集 | 规模/任务特征 | 推理 | 时间/演化 | 多源 | 科研 | 与方法的匹配 |
|---|---|---:|---:|---:|---:|---|
| [TEMPO](https://arxiv.org/abs/2601.09523) | 1,730 个复杂查询、3,976 个 step-wise retrieval plans、约 160 万文档；有 Temporal Coverage/Precision | 高 | 高 | 高 | 部分领域 | **最适合证明时间覆盖、跨期多步证据检索** |
| [ChronoQA（中文新闻）](https://www.nature.com/articles/s41597-025-06098-y) | 30 万余篇 2019–2024 新闻，5,176 问题；绝对/聚合/相对时间 | 中高 | 高 | 高 | 否 | **最适合中文、多文档、时态敏感 RAG** |
| [Temporal Wiki / Unified Clark](https://arxiv.org/abs/2506.07270) | Wikipedia 历史快照与累积新闻；研究过时、冲突事实 | 中高 | 高 | 中高 | 否 | **适合状态更新、latest/as-of 和冲突** |
| [MIRAI](https://arxiv.org/abs/2407.01231) | 991,759 条 GDELT 记录、59,161 个事件、296,630 篇新闻；国际事件预测 | 高 | 高 | 高 | 否 | **最像“复杂研判”**；但任务是未来预测，需增加 agent/tool baseline |
| [ForecastQA](https://arxiv.org/abs/2005.00792) | 10,392 个基于历史新闻的事件预测问题 | 高 | 高 | 中高 | 否 | 可测试事件链对预测证据的贡献，时间较早 |
| [STaRK-MAG](https://arxiv.org/abs/2404.13207) | 学术论文搜索；文本属性＋作者/机构/论文关系，多跳半结构检索 | 高 | 低 | 高 | 是 | **适合科研文本＋图关联**，但不能证明演化/冲突 |
| [BRIGHT](https://arxiv.org/abs/2407.12883) / [BRIGHT-Pro](https://arxiv.org/abs/2605.04018) | 1,384 个自然复杂查询；Pro 版增加多方面互补证据与 agentic protocol | 高 | 低至中 | 高 | 多个科研领域 | **适合证明 evidence portfolio 与 query decomposition** |
| [ScholarQABench](https://www.nature.com/articles/s41586-025-10072-4) | 2,967 个专家问题、208 个长答案；四个科学领域、多论文综合 | 高 | 低 | 高 | 是 | **适合端到端科研综合与 citation**；检索结构 gold 较弱 |
| [LitQA2 / PaperQA2](https://arxiv.org/abs/2409.13740) | 248 个困难科学文献问题；PaperQA2 还测试文献矛盾发现 | 高 | 低至中 | 高 | 是 | **适合科学检索和矛盾发现**，但没有规范 claim lifecycle gold |
| [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/) | 500K abstracts 上的开放域科学主张支持/反驳 | 中高 | 低 | 高 | 是 | **适合 Mention、stance、证据回溯**，但静态且多为 abstract 级 |
| [QASPER](https://aclanthology.org/2021.naacl-main.365/) | 1,585 篇 NLP 论文、5,049 问题、支持证据；55.5% 需要多段 | 高 | 低 | 低（多为单篇） | 是 | 适合长论文 evidence retrieval，不适合主创新 |
| [DeepResearch Bench](https://openreview.net/pdf?id=hQ0K2Hhq7H) / [II](https://arxiv.org/abs/2601.08536) | 100/132 个专家研究任务，跨 22 域，长报告和细粒度 rubric | 很高 | 不固定 | 很高 | 部分 | 适合最终“研判报告”，但难隔离索引/检索贡献 |

### 5.2 推荐的三层实验组合

#### 第一层：主检索能力

- **TEMPO**：检验 reasoning-intensive temporal retrieval、跨期覆盖、step-wise evidence；
- **BRIGHT-Pro**：检验互补 evidence portfolio 和 adaptive/iterative retrieval。

#### 第二层：多源演化能力

- **ChronoQA** 或 Temporal Wiki/Unified Clark：检验 state-at-time、latest、历史状态、冲突；
- 如果主打“研判”，用 **MIRAI**，但把任务拆成“历史证据检索质量”和“最终预测”两层，避免生成模型掩盖检索贡献。

#### 第三层：科研域

- **STaRK-MAG**：检验 scientific text＋relational retrieval；
- **SciFact-Open**：检验科学主张的支持/反驳 evidence；
- **ScholarQABench 或 LitQA2**：做 end-to-end 科研综合。

### 5.3 最现实的组合

资源有限时不建议同时跑十个 benchmark。最小可行组合是：

1. **TEMPO**：主检索 benchmark；
2. **SciFact-Open 或 STaRK-MAG**：科研与关联融合；
3. **自建 ScienceClaimLife 子集**：证明科学主张的演化、来源依赖和生命周期路径。

如果论文更偏中文应用，把 TEMPO 之外再增加 ChronoQA；如果更偏情报研判，用 MIRAI 替换 STaRK-MAG。

---

## 六、是否必须自建数据集，以及工作量

### 6.1 为什么仍需自建一部分

现有数据集分别覆盖：

- 科学综合：ScholarQABench/LitQA2；
- 科学支持—反驳：SciFact-Open；
- 文本＋关系：STaRK-MAG；
- 时间多步检索：TEMPO；
- 新闻演化：ChronoQA/Temporal Wiki；
- 复杂研判与预测：MIRAI。

但没有一个同时给出以下 gold：

```text
科学断言
+ 断言状态随时间变化
+ 触发变化的事件
+ 原始 evidence spans
+ 来源依赖/独立性
+ 最小充分证据集合
+ as-of / as-known-at 时间
```

因此无需“从零造一个巨大通用数据集”，但需要一个能精准打中方法贡献的中等规模子集。

### 6.2 推荐的数据单元：Evolution Dossier

一个 dossier 围绕一项可追踪主张，由 4–12 份材料构成：

- 初始研究或原始公告；
- 后续复现/未复现；
- 方法或数据限制；
- correction、expression of concern 或 retraction；
- 后续综述/meta-analysis；
- 引用、转载或二次报道，用来标注来源依赖。

可优先使用 PMC Open Access/PubMed、Crossref relations、OpenAlex/S2ORC 元数据和 [Crossref 发布的 Retraction Watch 数据](https://www.crossref.org/labs/retraction-watch/)，降低版权和采集风险。

### 6.3 Query 类型

每个 dossier 生成并人工确认 3–6 个问题：

1. **State-at-time**：截至 t，主张处于何种状态？
2. **Evolution**：结论如何从 A 变为 B？
3. **Evidence portfolio**：哪些相互独立的证据支持/反对它？
4. **Replacement**：哪个新主张替代了旧主张，依据是什么？
5. **Provenance**：某结论来自原始实验还是二次转述？
6. **Insufficient evidence**：在给定时间截面是否无法下结论？

不要让 LLM 直接生成 ground truth。可让 LLM 做候选抽取和预标，专家负责证据 span、关系、时间、来源依赖和答案裁决。

### 6.4 Gold schema

每题至少标：

- answer / answerability；
- target assertions；
- assertion state 和 valid interval；
- recorded/known-at time；
- transition event 与 effect type；
- supporting/refuting/qualifying mentions；
- source lineage cluster；
- minimal sufficient evidence set；
- optional evidence；
- reasoning/evidence path。

### 6.5 工作量估算

以下按“两位标注者＋一位裁决者”、有 LLM 预标工具估算；科学专家的阅读速度和领域决定上限。

| 规模 | Dossier / Query / 文档 | 专家标注与裁决 | 工程与管理 | 总量 | 用途 |
|---|---|---:|---:|---:|---|
| **可行性 Pilot** | 25–40 / 80–160 / 150–350 | 120–280 小时 | 1–1.5 人月 | **约 2–3 人月** | 调 schema、验证指标、做 workshop/附加实验 |
| **论文级子集** | 100–150 / 350–700 / 700–1,500 | 500–1,100 小时 | 2–3 人月 | **约 6–10 人月** | 足以支撑主论文专属实验 |
| **公开 benchmark** | 300–500 / 1,000–2,000 / 3,000–8,000 | 2,000–4,500 小时 | 5–8 人月 | **约 12–24 人月** | 独立 dataset paper 或大型联合项目 |

降低工作量的办法：

- 先只做一个领域，例如生物医学或 AI；
- 只选已有 correction/retraction/replication 元数据的 dossier；
- 把 entity/event/assertion 抽取当 weak labels，人工只验证关键路径；
- 对 hard test 全双标，对训练集抽样复核；
- 先做 100 dossier 的 frozen benchmark，不追求覆盖整套文献库。

**建议落点：100 个 dossier、约 400–500 个问题。** 这比造一个上千题的通用 QA 集更能证明方法，也更可控。

---

## 七、当前方法还差哪些关键环节

| 缺口 | 当前风险 | 最低补齐要求 | 是否主论文必需 |
|---|---|---|---:|
| **形式化问题定义** | 三通路容易被看作组件堆叠 | 定义 query、TKG、evidence path/portfolio、预算和优化目标 | 是 |
| **真正的跨通路交互** | 独立召回＋RRF 不叫协同 | seed 后跨视图补全＋集合选择＋充分性检查 | 是 |
| **多源 Mention 对齐** | duplicate、update、supersede、conflict 容易混在一起 | 保留全部 Mention，在 Assertion/State 层做类型化对齐 | 是 |
| **双时态检索** | valid time 有了，但无法回答“当时知道什么” | valid time＋recorded/transaction time；as-of/as-known-at query | 强烈建议 |
| **区间与模糊时间** | 字符串比较无法对抗 IA-RAG | 标准时间归一化、区间关系和 uncertain bound | 时间主张必需 |
| **事件—断言状态验证** | LLM 抽错一个 effect 会污染链条 | 状态迁移约束、effect validation、abstain | 是 |
| **实体/事件共指评价** | 上游错误被最终 QA 掩盖 | entity resolution、event coreference、assertion fusion 单独指标 | 是 |
| **证据充分性/停止条件** | 找到一些相关材料就停止 | 必要 evidence roles、缺口检测、定向二次检索 | 是 |
| **连续维护协议** | 目前只证明 batch 建库 | 增量加入、迟到证据、纠错、撤回、回滚及历史重放 | 若讲动态维护则必须 |
| **负证据与未知** | 系统可能把未找到解释成不存在 | no-answer/insufficient/conflict 分类与校准 | 强烈建议 |
| **可扩展性** | TKG 抽取和 Agent 代价高 | build cost、storage、update/query latency、p95 和 token cost | 是 |

### 最值得增加的单一算法机制

如果只能补一个，不是“更复杂的 embedding”，而是：

> **Budgeted cross-path evidence completion and portfolio selection.**

它直接将 unified TKG 从数据结构变成检索算法；也能通过 decoupled-index、RRF 和结构打乱等实验清楚归因。

---

## 八、必须比较什么结果

### 8.1 普通排名指标仍需报告，但不是核心证明

- Recall@k、Precision@k、nDCG@k、MRR；
- 文档级与 passage/mention 级分开；
- 所有方法相同候选数、token budget 和 reranker budget。

Recall 高只能证明“相关内容多被捞到”，不能证明状态正确、证据充分、来源独立或三路协同。

### 8.2 核心集合/路径指标

- **Complete Evidence Set Recall@k**：一个问题所需 gold evidence 是否全部覆盖；
- **Lifecycle Path Success/Completeness**：Assertion—EventEffect—Event—Mention 路径是否恢复；
- **Evidence Role Coverage**：背景、前态、转变、后态、支持、反证等角色覆盖；
- **Temporal Coverage@k / Precision@k**：直接采用 TEMPO 指标；
- **Minimality / Redundancy**：相同充分性下使用的文档、节点和 token 数；
- **Multi-source Evidence Coverage**：gold 所需的来源文档和原文 mention 是否覆盖。

### 8.3 时间与融合指标

- State-at-Time Accuracy；
- As-Known-at-Time Accuracy；
- Transition Effect Macro-F1；
- temporal ordering / interval relation F1；
- stance 与 conflict Macro-F1；
- duplicate/refine/update/supersede/conflict alignment Macro-F1。

### 8.4 最终回答指标

- answer correctness / exact or rubric score；
- citation precision、citation recall、citation completeness；
- claim-level entailment/support rate；
- conflict disclosure rate；
- unanswerable/insufficient-evidence accuracy；
- 科研长回答可增加 ScholarQABench 或 DeepResearch Bench 风格的 coverage、correctness 和 citation rubric。

### 8.5 效率与维护

- 初次建库时间、LLM token、向量/图存储放大；
- 单文档增量更新时间；
- affected node/embedding 数；
- query p50/p95 latency；
- 迟到证据和撤回后的状态修复准确率；
- Agentic 版本的工具调用数、失败率和人工升级率。

---

## 九、Baseline：最小但有说服力的组合

### 核心 7 组

| Baseline | 为什么必须 | 它代表什么 | Ours 应在哪赢 |
|---|---|---|---|
| **BM25＋Dense＋Cross-encoder Hybrid** | 防止只因弱向量基线获胜 | 强文本检索 | 普通题接近；演化/完整证据题明显胜 |
| **LightRAG** | 与现有系统和代码最接近 | 实体关系型轻量 GraphRAG | state/evolution/conflict、证据链完整度 |
| **HippoRAG 或 SiReRAG** | 覆盖强图检索或多视图知识组织 | multi-hop/structured organization | 时间状态、source-grounded path；非时间题不能大幅退化 |
| **EventRAG** | 直接的事件 KG 竞争者 | 跨文档事件融合＋多事件推理 | 断言生命周期、状态变化、可溯源证据 |
| **E²RAG** | 最危险的实体—事件协同反例 | entity mention/event dual graph | assertion lifecycle、source independence、复杂冲突 |
| **DyG-RAG 或 IA-RAG** | 覆盖事件时间线或形式区间检索 | temporal event-chain SOTA 型能力 | state transition＋evidence grounding；纯时间题至少有竞争力 |
| **Decoupled-3Path＋同款 RRF** | 证明“统一 TKG”而非更多索引带来提升 | 相同对象、相同召回预算、但索引分离 | lifecycle path、complete evidence、token efficiency |

其中最后一项是最重要的内部 baseline：

> 保留完全相同的 Chunk、Entity、Assertion、Event、embedding 和候选量，只切断 Assertion—EventEffect—Mention 的统一图连接，再做 late fusion。

只有赢过它，才能证明“捏在一起”产生了方法价值，而不是简单增加召回源。

### 任务特定 baseline

- 科研综合：OpenScholar、PaperQA2；
- STaRK：官方强 retriever/HybGRAG；
- TEMPO：官方 leaderboard 中的强 reasoning retriever；
- MIRAI：官方 agent/tool-use baseline；
- 双时态状态：Beyond Retrieval 的 state arbitration operator；
- Agentic retrieval：IRCoT/Adaptive-RAG 或 BRIGHT-Pro agentic protocol。

不建议为了“数量齐全”同时放 RAPTOR、GraphRAG、LightRAG、HippoRAG、SiReRAG。直接竞争者比历史性 baseline 更重要。

---

## 十、最关键的消融和对照

1. **Unified TKG vs Decoupled Stores＋RRF**：隔离统一连接的价值；
2. **Cross-path completion vs 仅 RRF**：隔离协同算法的价值；
3. **去掉 EventEffect**：只保留 Entity—Event 共存，证明状态迁移边必要；
4. **去掉 Mention backlinks**：证明证据落地、citation 和完整度增益；
5. **去掉 typed multi-source alignment，只按实体/事件相似度合并**：证明 duplicate/update/supersede/conflict 的结构区分有用；
6. **无时间 / point time / interval / bitemporal**：逐层验证时间表示；
7. **静态全通路 vs query-aware routing**：证明路由价值；
8. **等 token、等候选、等 reranker 预算**：排除资源优势；
9. **打乱 EventEffect 或时间边**：排除图更大带来的伪增益；
10. **注入 5%/10%/20% 抽取与共指噪声**：评价鲁棒性；
11. **时间切分和增量更新**：旧语料建库，按时间加入新证据，测试历史状态保留和当前状态修复；
12. **Oracle extraction / predicted extraction**：区分检索算法上限和上游抽取瓶颈。

---

## 十一、现有实验已经说明什么、还不能说明什么

根据当前本地结果：

- Ours 在 60 个问题上相对 LightRAG hybrid 的 evidence recall 只从约 0.821 提升到 0.833，complete recall 都约 0.648，同时 MRR 反而较低。这不能支持“第三条 Event 通路显著改变检索能力”。
- Ours_TKG dynamic 在 30 个问题上相对 fixed Ours_TKG 的 complete recall、nDCG 和 MRR 有明显提升，且 p95 更低；这是好信号，但样本太小，也没有 EventRAG/E²RAG/DyG-RAG 和 decoupled-store 对照。

所以现阶段可以声称：

> 统一结构和动态路由显示出提升复杂证据检索的潜力。

暂时不能声称：

> 统一 TKG、Mention 证据回溯和三通路协同已经分别得到验证。

---

## 十二、推荐的最终论文结构

### 主创新

**Evidence-grounded temporal assertion indexing and cooperative evidence-path retrieval.**

### 次级贡献

1. 统一表示 Assertion lifecycle、EventEffect 和 Mention/Source provenance；
2. 查询条件化的三类 evidence role 路由与跨通路补全；
3. 固定预算下的充分、互补、时间覆盖和来源独立证据组合选择；
4. 面向动态多源语料的路径、状态、来源和 citation 评价协议；
5. 可选：科学主张生命周期的小规模高质量 benchmark。

### 不应作为主创新的内容

- 多建几个 embedding collection；
- 三路结果做 RRF；
- 使用 LLM 抽 Entity/Event；
- 使用 Neo4j；
- 把简单 doc count/stance 当成数据关联融合机制；
- 多步骤 pipeline 直接称 Agent。

### 一句最精炼的创新故事

> 传统文本检索找到相似内容，实体图恢复结构关系，事件图恢复发生顺序，但复杂多源研判需要知道“哪项可验证断言在何时被什么事件改变、这些状态分别由哪些相互独立的原始证据支撑”。本方法将断言生命周期、事件效应和来源 mention 统一成可检索的时态证据索引，并通过查询条件化的跨通路证据补全与预算化组合选择，返回最小但充分的可溯源演化证据子图。

---

## 十三、建议执行顺序

1. 先冻结论文问题定义和 schema，不再扩展泛化的 Agent 叙事；
2. 实现 Decoupled-3Path baseline，确认 unified TKG 是否真的带来路径完整度增益；
3. 把 RRF 升级为 cross-path evidence completion＋portfolio selection；
4. 增加 typed multi-source Mention alignment 和双时态；不引入来源可信度贡献；
5. 在 TEMPO 做主检索实验，在 SciFact-Open/STaRK-MAG 做科研关联实验；
6. 先建 25–40 dossier pilot，验证标注 schema 和指标；
7. 结果成立后再扩展到约 100 dossier 的论文级 ScienceClaimLife 子集；
8. 最后加入 EventRAG、E²RAG、DyG-RAG/IA-RAG 和任务特定系统比较。
