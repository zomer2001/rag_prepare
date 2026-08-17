# Ours / Ours_TKG：技术机制、相关工作与 Novelty Audit

> 审阅日期：2026-08-13  
> 审阅范围：仓库当前代码、技术文档、现有评测产物，以及 2023—2026 年相关论文。  
> 判断原则：以下“已实现”只指代码和现有运行链路中真实存在的机制；设计文档中的未来方向不计入当前贡献。

## 0. 执行摘要

先给出最重要、也最可能影响投稿策略的结论：

1. **不能再把“Vector—Semantic—Event 三通路”本身作为主创新。** SiReRAG 已从“相似性＋相关性”双组织角度构造统一检索池；LightRAG 已联合向量与实体关系图；EventRAG 已有事件图、跨文档事件合并与迭代多事件推理；DyG-RAG 已有动态事件单元、时间增强向量、事件时间线检索；E²RAG 更直接提出实体图—事件图双图及二部映射。仅增加第三路并做 RRF，Reviewer 很容易判为工程组合。
2. **当前 `Ours` 的准确定位是：面向可追溯文档证据的多表示 late-fusion retrieval。** 它的事件确实是独立知识单元，也确实改变了索引和召回，但三路之间是并行召回、排名融合，并不存在“向量发现 → 语义确认 → 事件验证”的逐步交互或反向校验。因此该口号目前不符合代码事实。
3. **`Ours_TKG` 比 `Ours` 更有研究价值，但真正值得发展的不是五个向量入口。** 它把 Entity、Assertion、AssertionMention、Event、EventEffect、有效时间、立场、冲突和来源证据放入统一声明级时序图；其中 `EventEffect ∈ {CREATES, UPDATES, TERMINATES, SUPERSEDES, DENIES, CONFIRMS}` 会改变 Assertion 生命周期。这是比“三路融合”更清楚的知识表示贡献。
4. **最 solid 的主方向应改为：evidence-grounded event-conditioned assertion lifecycle retrieval（证据落地的事件条件声明生命周期检索）。** 核心问题不是“如何多召回一种内容”，而是：多源动态语料中，同一实体的关系会随事件产生、更新、终止、被否认或替代；实体中心 KG 的单一边会压缩历史状态，纯事件图又不稳定地表达长期事实。方法应显式恢复“事件 → 声明状态变化 → 来源证据”的可验证路径。必须注意，[Beyond Retrieval](https://aclanthology.org/2026.knowfm-1.10/) 已在纵向医疗 Agent memory 中提出双时态状态与 SUPPORT/REFINE/SUPERSEDE/BRANCH-CONFLICT 仲裁，因此 novelty 不能写成“首次维护事实生命周期”，而应限定在**开放域、多源、由非结构文档构建并面向 RAG 检索的可追溯生命周期路径**。
5. **当前证据尚不足以支持强 novelty 或强效果主张。** 60 题/200 文档实验中，`ours` 相比 `light_entity` 的 Evidence Recall@10 仅从 0.8241 到 0.8333，Complete Recall@10 相同，MRR 还下降；强 Hybrid 基线在 Hits、MRR 与 Complete Recall 上不弱。`ours_tkg_dynamic` 明显优于固定 `ours_tkg`，说明 query-aware route/coverage scoring 有价值，但仍只有 30 题、检索指标，没有 event/state/conflict gold 和最终回答指标。

**最终审稿判断：** 当前 `Ours` 属于“工程集成创新到一般方法创新之间”；当前 `Ours_TKG` 属于“一般方法创新”，其数据模型具有形成较 solid 方法贡献的潜力，但必须增加并验证“查询条件下的声明生命周期路径检索/验证”算法，而不是继续叠加检索通路。

---

## 1. 第一阶段：Ours / Ours_TKG 技术机制拆解

### 1.1 方法名称的实际边界

仓库中实际上存在四组容易混淆的对象：

| 名称 | 实际含义 | 是否属于本报告主要对象 |
|---|---|---|
| `light_entity` | 受控实体关系抽取；LightRAG `naive` 文本路＋`hybrid` 图路；文档级 RRF | `Ours` 的无事件消融 |
| `ours` | `light_entity` 完全复用，再增加独立事件索引、事件召回和第三路 RRF | 是，本文称 **Ours** |
| `ours_tkg` | 不建立 LightRAG 语义图；五类向量种子进入统一声明级 TKG，Neo4j 受约束扩展后回溯来源 Mention；固定全入口＋RRF | 是，本文称 **Ours_TKG** |
| `ours_tkg_dynamic` | 复用 `ours_tkg` 索引；按 query intent 启停入口、动态阈值裁剪，并按覆盖/相关性加权 | 是，但必须作为 **Ours_TKG 的增强变体** 单列，不能倒算成默认版能力 |

另有 `agentic_ours*`：最多两轮检索、证据充分性判断、query rewrite 和回答生成。它们是 Agentic 扩展，不是 Ours/Ours_TKG 基础方法的固有机制。

### 1.2 Ours / Ours_TKG 技术机制拆解表

| 审查问题 | Ours | Ours_TKG | 审稿人判断 |
|---|---|---|---|
| 原始文档预处理 | Markdown 直读；PDF/DOCX 经 Docling 规范化为 Markdown；保留 `document_id/page/section` 来源标记 | 评测中复用同一固定语料与切块；语义事实和事件分别抽取 | 两者都具备 provenance 基础 |
| 分块 | 正式评测默认 700 token、100 overlap；LightRAG 内部可能再次细分，但父 `document_id` 稳定 | 同一固定 Chunk；Chunk 本身也是五类向量入口之一 | 公平性设计合理，但需报告实际二次切块影响 |
| Vector 表示 | Chunk、LightRAG Entity、Relation 向量；Event 规范文本另行向量化 | 分别索引 Chunk、Entity、Predicate、Assertion、Event 五类向量 | Ours 是三路；TKG 是五种 seed view，不宜继续称简单三路 |
| Entity / Relation | LightRAG 受控抽取，最多 4 个关键实体、4 条关系；NetworkX 保存拓扑 | 统一 Entity ID；关系重化为 Assertion 节点，支持有效期、状态和 Mention | TKG 的 Assertion reification 是关键结构差异 |
| Event | `EventRecord` 是跨文档规范事件；`EventMention` 保留每源原文证据；有角色、时间、立场、状态 | Event 节点与 Entity、Assertion 通过 `EventEntityLink`、`EventEffect` 相连 | 都不是普通附加字段；Ours 中 Event 独立于实体图，TKG 中 Event 进入统一图 |
| Time | 事件 `start/end/time_precision`；事件间 `precedes/same_time/related`；明确不把先后当因果 | Assertion `valid_from/valid_to`＋Event `effective_time`；事件可创建、更新、终止、替代声明 | TKG 真正从“事件时间”走向“事实有效时间” |
| Source / Evidence | 所有路最终回到 `document_id`；事件交付 `EventMention.evidence` 而非事件摘要 | `EntityMention`、`AssertionMention`、EventMention 保留来源与证据；检索沿 Mention 回溯原文 | provenance 是可信实验的重要优势，但不是单独的新算法 |
| 向量通路作用 | 原文安全网、局部事实和近义匹配 | Chunk 直接召回；另为图中的四类结构节点提供 query linking | “潜在关联发现”成立，但不是专有角色；所有结构节点也由向量发现 |
| 语义/图谱通路作用 | LightRAG low-level entity＋global relation 检索，扩展后回到来源 Chunk | 以 Entity/Predicate/Assertion seed 在统一 TKG 中扩展并取 AssertionMention | Ours 的语义图相对静态；TKG 把关系提升为可带状态的声明 |
| 事件通路作用 | Event vector seed → 结构排序 → 可选一跳 event-chain → Mention evidence | Event seed → TKG 扩展 → EventMention，并可带出影响 Assertion 的 EventEffect | Ours 的事件主要是第三召回源；TKG 中事件还能改变知识状态 |
| 索引阶段协同 | 三路共享 `document_id`，但实体关系与事件独立抽取、独立存储；无跨路联合学习 | Entity resolution 统一身份；semantic relationship→Assertion；event fact effect→Assertion 生命周期 | Ours 只有身份级弱协同；TKG 有知识组织层实质交互 |
| 查询阶段协同 | 一次 Planner 服务三路；三路并行；Event 内部 RRF；外层按文档 RRF | 一次 Planner；五类向量并行 seed；Neo4j 统一扩展；最后折叠为 vector/graph/event 文档证据 | 都以 late fusion 为主；默认 TKG 并非 query-adaptive |
| 触发/路由 | Ours 默认总开三路；Planner 只决定时间、实体提示和是否一跳扩展 | `ours_tkg` 总开五入口；`dynamic` 才会按实体/关系/event intent 启停和阈值裁剪 | 动态路由不能写入默认方法贡献 |
| 召回 | 每路 `max(K,3K)`；Event 每个规范事件最多贡献两个来源文档 | 五个 Qdrant collection 各召回；图扩展考虑时间、状态和 event chain | TKG query linking 覆盖更全，但入口相关性存在冗余 |
| 融合/排序 | Event 内部：vector rank＋structured rank RRF；外部：每路按文档去重后 RRF | 默认：三类证据按文档 RRF；dynamic：route relevance/coverage 加权、跨路相关性降重 | Ours 的主要算法只是两级 RRF；dynamic 的 scoring 更值得研究 |
| Event chain | 共享实体且时间相邻的事件边；query intent 触发一跳扩展 | 可在统一图中扩事件链，并携带 EventEffect / Assertion 状态 | 当前只是一跳/有限扩展；没有完整链最优化或 completeness objective |
| 因果 | 不显式建模因果；明确禁止把 `precedes` 解释为因果 | EventEffect 表示事件对声明的作用，但不等同一般事件因果图 | 不能声称“显式因果建模” |
| 状态变化 | Event status 有 single/multi-source/disputed；未直接反写 LightRAG relation | EventEffect 真正更新 Assertion 为 active/inactive/disputed/superseded/retracted | 这是 TKG 相比 EventRAG/E²RAG 仍可能差异化之处 |
| 冲突/验证 | 同一规范事件的 affirmed＋denied 形成 conflict；多文档近似多来源支持 | AssertionMention stance＋ConflictCase；事件可 DENY/CONFIRM 声明 | 目前只是结构化暴露冲突，不是完整“验证算法”；来源独立性也未建模 |
| 最终给 LLM 的单位 | Top-K 文档 `EvidenceBundle`；每文档最多 3 个带通路标签的 Chunk/Event evidence | `RetrievedChunk` 按文档聚合，内部混合 source Chunk、AssertionMention、EventMention | 都不是直接只返回 subgraph；结构最终线性化为混合原文证据 |
| 是否改变知识组织 | 是：增加规范 Event、Mention、时间边、冲突；但不改变实体关系图本身 | 是且更强：Assertion reification＋有效期＋事件作用＋多 Mention | Ours 不能只说“多一种召回”，但查询收益目前主要仍来自新增召回；TKG 属知识表示改变 |

### 1.3 两个方法最准确的流程描述

**Ours：**

```text
Index:
Document ─┬─ fixed chunks → Chunk vectors
          ├─ controlled entity/relation extraction → LightRAG entity graph
          └─ independent event extraction → Event / Mention / temporal edges / conflicts

Query:
one planner ─┬─ naive chunk retrieval
             ├─ LightRAG local+global graph retrieval
             └─ event vector seed → structured rerank → optional 1-hop expansion
                         ↓
              per-document RRF → mixed evidence bundles
```

**Ours_TKG：**

```text
Index:
Chunks ─┬─ semantic extraction → Entity resolution → Assertion + AssertionMention
        └─ event extraction → Event + EventMention
                              └─ EventEffect → Assertion lifecycle

Five vector indexes: Chunk / Entity / Predicate / Assertion / Event
Unified graph: Entity—Assertion—Entity, Event—Entity, Event—Effect—Assertion, Mentions, conflicts

Query:
planner → five vector seed searches → time/state-constrained graph expansion
        → provenance backtracking to source Mentions → document evidence fusion
```

### 1.4 对用户提出的“三通路假设”的核验

假设是：Vector 发现潜在关联，Semantic 组织与确认实体关系，Event 建模演化并跨源验证，三者形成“发现—确认—演化/验证”。

| 命题 | 当前是否成立 | 原因 |
|---|---|---|
| Vector 负责潜在关联发现 | **基本成立** | Chunk/Event/结构节点都通过 embedding 做 query linking |
| Semantic 负责关系组织 | **成立** | Ours 有实体关系图；TKG 有 Assertion 图 |
| Semantic 对 vector 结果进行“确认” | **不成立** | 三路并行，RRF 奖励共识，但无显式 entailment/constraint confirmation |
| Event 负责时间演化 | **Ours 部分；TKG 较强** | Ours 有事件序列；TKG 还建模 Assertion 有效期和生命周期 |
| Event 做跨源证据验证 | **仅弱成立** | 目前不同文档数、多来源状态、affirmed/denied 冲突是启发式聚合；无来源独立性、NLI 或校准验证 |
| 形成发现→确认→验证的顺序机制 | **不成立** | 当前是并行召回＋late fusion，不是 cross-path interaction |

因此这套假设可以作为**下一版算法目标**，不能作为当前 Ours 的已实现贡献陈述。

---

## 2. 第二阶段：2023—当前相关研究矩阵

### 2.1 高相关方法对比

标记说明：✓ 明确具备；△ 部分/间接具备；— 未作为核心机制。方法年份按正式发表年；只有预印本的标为 arXiv。

| 方法 | 年份/会议 | 解决的问题 | 知识组织 | 检索方式 | Vector | Graph | Event | Time | Event Chain | 多通路 | 核心创新 | 与 Ours 重合 | 关键差异 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|
| Dense / Hybrid RAG | 通用基线 | 平面 Chunk 的语义/词法召回 | Chunk；Dense＋BM25 | ANN、BM25、RRF/重排 | ✓ | — | — | — | — | ✓ | 互补词法与语义信号 | Ours 的 vector 路和外层 RRF | 无显式结构；但常是最强简单基线 |
| [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) | 2024 / ICLR | 短 Chunk 难以理解长文全局 | 递归聚类、摘要树 | 检索叶/多层摘要节点 | ✓ | 树 | — | — | — | △ | 多抽象层级的树组织 | 都增加非平面组织 | RAPTOR 面向层次摘要，不建实体/事件状态 |
| [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) | 2024 / arXiv + 官方项目 | 全语料 global sensemaking | Entity-relation KG、社区、社区摘要 | local 邻域或 global map-reduce | △ | ✓ | △ claims | — | — | ✓ | 社区分层摘要回答全局问题 | 图组织、来源回溯 | 主目标是 global QFS，非事件演化 |
| [LightRAG](https://aclanthology.org/2025.findings-emnlp.568/) | 2025 / EMNLP Findings | 平面表示碎片化且图方法成本高 | Entity-relation graph＋向量 | low-level entity＋high-level relation＋mix | ✓ | ✓ | — | — | — | ✓ | 双层图检索与增量更新 | Ours 直接复用其主要图机制 | Ours 新增独立 event；故 LightRAG 能力不能算自研 |
| [HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) | 2024 / NeurIPS | 单步 Chunk retrieval 难做多跳关联 | OpenIE triples、phrase/entity KG、passage links | query seeds＋Personalized PageRank | ✓ | ✓ | — | — | — | ✓ | 图上关联记忆和单步多跳检索 | 都从语义 seed 扩图并回到 passages | HippoRAG 有全局 PPR；Ours 是 LightRAG 邻域＋RRF |
| [SiReRAG](https://openreview.net/forum?id=yp95goUAT1) | 2025 / ICLR | 只按相似性或只按相关性组织都不足 | Similarity tree＋shared-entity relatedness tree | 双树扁平到统一检索池 | ✓ | 树/实体聚组 | — | — | — | ✓ | 同时索引 similarity 与 relatedness | 强烈削弱“向量＋语义双组织即创新” | 它没有事件/时间；但双组织思想已明确存在 |
| [GFM-RAG](https://proceedings.neurips.cc/paper_files/paper/2025/file/33ca0b1102b54c191a9a45a05adafaf4-Paper-Conference.pdf) | 2025 / NeurIPS | GraphRAG 图噪声、不完整与跨域泛化 | 文档 KG＋entity-to-doc inverted index | query initialization＋GNN message passing＋doc rank | ✓ | ✓ | — | — | — | △ | 预训练 graph foundation retriever | 都用图结构改善多跳文档检索 | GFM 有训练式跨图推理；Ours/TKG 为启发式/LLM 抽取 |
| [OG-RAG](https://aclanthology.org/2025.emnlp-main.1674/) | 2025 / EMNLP | 专业领域缺少 schema grounding 与可归因事实 | Ontology-grounded factual blocks / hypergraph | 最小 hyperedge 集优化 | △ | ✓ | — | △ | — | △ | 本体约束事实块与紧凑上下文 | Ours_TKG 也有 OWL/SHACL、本体映射、Mention | Ours_TKG 不能把“使用本体”单独当 novelty；优势应在事件驱动生命周期 |
| [EventRAG](https://aclanthology.org/2025.acl-long.830/) | 2025 / ACL | 多源叙事中的事件推理、时间和逻辑一致性 | Event KG；跨源相似事件/实体融合、知识扩展 | autonomous EKG query＋迭代 multi-event inference | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 事件 KG＋agentic multi-event reasoning | 事件向量、事件合并、时间边、跨文档、多跳 | **直接威胁 Ours**；其事件推理比 Ours 一跳＋RRF 更强，但 provenance/声明生命周期不如 TKG 明确 |
| [E²RAG](https://aclanthology.org/2026.eacl-long.90/) | 2026 / EACL | KG 合并实体 mention 导致动态上下文/状态压缩 | entity-mention graph＋event graph＋二部映射 | entity/event 向量 seed＋event-frontier 一跳扩展＋raw passages/subgraph | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 分离实体与事件并用二部图保持 temporal-causal consistency | **与实体＋事件并行组织高度接近**；都回溯 raw evidence | **最直接 novelty 威胁之一**；E²RAG 保留实体 mention 上下文，Ours_TKG 的差异是 canonical Entity＋temporal Assertion lifecycle＋stance/provenance |
| [DyG-RAG](https://arxiv.org/abs/2507.13396) | 2025 / arXiv | 静态 GraphRAG 无法恢复时间序列 | Dynamic Event Unit＋带权事件图 | 时间增强 embedding＋cross-encoder＋multi-seed walk＋timeline | ✓ | ✓ | ✓ | ✓ | ✓ | △ | 时间锚定 DEU、时间衰减边和 timeline retrieval | 事件单位、时间边、chain retrieval | **直接威胁 event-chain 主张**；DyG 的时间向量/遍历更强，Ours_TKG 的机会在事实状态与证据冲突 |
| [TimeR⁴](https://aclanthology.org/2024.emnlp-main.394/) | 2024 / EMNLP | TKGQA 中复杂时间约束与 temporal hallucination | 既有 Temporal KG facts | retrieve-rewrite-retrieve-rerank；time-aware contrastive retriever | ✓ | ✓ | △ KG events/facts | ✓ | △ | ✓ | 显式时间约束重写与时间感知重排 | Ours_TKG 也按时间过滤/rerank | TimeR⁴ 假设已有结构 TKG；Ours_TKG 从非结构文本建图并保留来源 |
| [STAR-RAG](https://arxiv.org/abs/2510.16715) | 2025 / arXiv | TKG 检索时间不一致、候选/Token 过多 | time-aligned rule graph | semantic anchor＋rule-graph propagation | ✓ | ✓ | △ temporal facts | ✓ | ✓ | ✓ | 训练免费、时间一致的规则图传播 | 都从语义 seed 做时间约束图扩展 | STAR-RAG 在既有 TKG 上做规则概括；Ours_TKG 从文本建 assertion/event provenance |
| [IA-RAG](https://arxiv.org/abs/2606.06044) | 2026 / arXiv | 点时间/粗时间戳不能表达 duration、overlap、containment | Interval Event Unit＋Thematic Forest＋Allen interval relations | interval-algebra-guided directional traversal | ✓ | ✓ | ✓ | ✓（区间） | ✓ | ✓ | 形式化区间代数与模糊边界收紧 | Ours_TKG 也有 valid interval 与时间扩展 | IA-RAG 的形式时间推理更强；Ours_TKG 当前只是字符串区间比较 |
| [Beyond Retrieval](https://aclanthology.org/2026.knowfm-1.10/) | 2026 / ACL KnowFM workshop | 纵向医疗记忆中的陈旧、冲突和被替代状态 | 双时态 patient state＋evidence | persistent state query＋增量仲裁 | △ | △ | △ | ✓（valid/transaction time） | △ | 状态仲裁 | SUPPORT/REFINE/SUPERSEDE/BRANCH-CONFLICT | **直接威胁 lifecycle/state novelty** | 医疗 Agent memory、偏 current-state arbitration；Ours_TKG 面向开放域多源文档、显式 EventEffect 和检索证据路径 |
| [IRCoT](https://aclanthology.org/2023.acl-long.557/) | 2023 / ACL | 一次检索不足以回答多步问题 | passages＋生成中的 CoT state | reasoning 与 retrieval 交替 | ✓ | — | — | — | — | 迭代 | 用中间推理决定下一次检索 | agentic 变体也会二次检索 | 当前 Ours/Ours_TKG 本身不是 iterative/agentic |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) | 2024 / NAACL | 不同复杂度问题不应统一使用昂贵策略 | 无新知识结构 | complexity classifier 在 no/single/iterative retrieval 间路由 | ✓ | — | — | — | — | 路由 | 基于问题复杂度自适应策略 | `ours_tkg_dynamic` 也动态启停入口 | 当前动态版是规则/Planner intent＋阈值，不是学习式或结果监督路由 |
| [TruthfulRAG](https://ojs.aaai.org/index.php/AAAI/article/view/40489) | 2026 / AAAI | 外部上下文与模型内部知识的事实级冲突 | 从 retrieved content 抽取 triple KG | query-aware graph paths＋entropy filtering | ✓ | ✓ | — | △ | — | ✓ | 事实级 conflict localization/resolution | Ours_TKG 有 Assertion conflict 和来源 stance | 当前 TKG 只是暴露冲突，未做强 conflict resolution；不能声称领域无人研究 |

### 2.2 直接竞争关系的严厉排序

对论文 novelty 威胁从高到低为：

1. **E²RAG**：直接主张实体图＋事件图协同，并以“实体合并压缩时间上下文”为研究问题；如果 Ours 仍讲三路/实体—事件协同，会正面撞题。
2. **EventRAG**：已覆盖事件 KG、跨文档事件融合、时间边和迭代多事件推理；Ours 的 event vector＋一跳扩展本身弱于它。
3. **DyG-RAG**：已明确以事件为动态最小知识单元，并进行时间增强检索与 timeline construction；“事件链检索”不能直接作为新贡献。
4. **OG-RAG / TimeR⁴**：分别覆盖本体 grounding 和 temporal KG reranking；Ours_TKG 必须证明两者之上的独特联结。
5. **LightRAG / SiReRAG / HippoRAG**：共同说明 vector＋graph、similarity＋relatedness、多跳图扩展都已有成熟先例。
6. **IA-RAG / STAR-RAG / Beyond Retrieval**：进一步表明区间时间推理、时间一致图传播、双时态状态仲裁均已出现。Ours_TKG 不能把 valid interval、supersede 或 state tracking 单点包装为新颖；必须证明它们在开放域多源 RAG 中通过 EventEffect 和 provenance path 形成新的检索机制。

### 2.3 文献确认后的真实空白

已被充分覆盖、**不宜**再写成空白：

- “Vector RAG 无结构、GraphRAG 有结构”——过于通用；大量论文已覆盖。
- “静态实体图压缩动态状态”——E²RAG 已直接提出。
- “事件图能恢复时间链”——EventRAG、DyG-RAG 已覆盖。
- “query-aware 动态检索/多轮检索”——Adaptive-RAG、IRCoT、Self-RAG 等已覆盖。
- “本体能提高专业领域 RAG”——OG-RAG 已覆盖。

仍相对未被上述直接方法完整解决、且与当前 TKG 数据模型吻合的组合空白是：

> **从开放域多源非结构文本中同时保留 canonical entity identity、可随事件变化的声明有效期、每源立场与原文证据，并在查询时选择能解释某事实“如何产生、改变、终止、被否认或被替代”的最小 provenance-bearing lifecycle path。**

EventRAG/DyG-RAG 强在 event sequence；E²RAG 强在 entity mention—event context；TimeR⁴、STAR-RAG、IA-RAG 强在既有 TKG 或 event interval 的时间检索；TruthfulRAG 强在模型内外冲突；Beyond Retrieval 强在医疗状态的双时态仲裁。**截至本次检索，没有发现正式顶会论文把开放域多源文本中的 `EventEffect → reified Assertion lifecycle → multi-source Mention evidence` 作为一个可优化、可检索、可评价的 provenance path。** 这是较窄但可辩护的机会；它是文献检索后的限定性判断，不是“绝对首次”声明。

---

## 3. 第三阶段：Novelty Audit

### 3.1 主创新与次级创新定位

#### Ours

- **主定位：Retrieval Fusion / Multi-representation Retrieval（偏系统方法）。** 三种异质知识表示都映射回统一 `document_id`，事件内部和跨路外部分别做 rank-based fusion，最终形成可追溯 EvidenceBundle。
- 次级定位：Event-centric Knowledge Organization；Evidence/Provenance Preservation。
- 不应定位为：Temporal KG、因果推理、证据验证、动态更新或 Agentic RAG。当前对应机制都不足或属于另一个变体。

三路改变的是**检索候选空间与最终证据融合**，并没有改变生成模型，也没有形成跨路推理算法。Event 路是真正的独立检索单位，但其增益仍可能仅来自扩大候选池。

#### Ours_TKG

- **主定位：Temporal/Event-centric Knowledge Representation + Evidence-grounded Graph Retrieval。** 核心是将静态关系重化为可带有效期和状态的 Assertion，并让 EventEffect 显式改变其生命周期；所有判断保留来源 Mention。
- 次级定位：multi-view indexing（五类 vector seeds）；query-aware graph expansion；entity resolution/governance；retrieval fusion。
- 动态版的次级定位：Adaptive Retrieval / Cross-view Coverage Scoring。

最重要的层级变化是：传统 KG 的 `Entity —relation→ Entity` 被改成了：

```text
Entity —subject_of→ Assertion ←object_of— Entity
                         ↑
Event —EventEffect───────┘
  │                      │
  └─ effective_time      └─ valid_from / valid_to / status

AssertionMention → source document + stance + evidence
EventMention     → source document + stance + evidence
```

这不是增加一个字段，而是使“事实本体”和“事实变化”成为两类可链接、可查询、可追溯的知识对象。

### 3.2 候选 Research Gap 评分

评分 1—5，越高越好；实现成本一栏 5 表示成本低、容易落地。

| 候选 Gap | 创新性 | 与现有实现匹配 | 实现成本 | 实验可验证 | 叙事完整 | 总评 /25 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| A. 三路统一 RRF 同时覆盖相关性、结构性、演化性 | 2 | 5 | 5 | 4 | 3 | 19 | 易做但很难过 novelty；SiReRAG/LightRAG/EventRAG 使组合故事过弱 |
| B. Query-aware 三路/五入口动态路由与预算分配 | 3 | 4（dynamic 已有雏形） | 4 | 5 | 4 | 20 | 可形成 retrieval 论文，但须学习式或可解释优化目标，不能只靠规则阈值 |
| C. Event-chain retrieval 与 temporal reranking | 2 | 4 | 3 | 5 | 4 | 18 | DyG-RAG、EventRAG、E²RAG 已高度覆盖；单做这条 novelty 风险大 |
| D. 跨来源 event consistency / conflict verification | 4 | 3 | 2 | 4 | 4 | 17 | 有价值，但现有实现只用文档数与 stance；强验证需 source independence＋NLI＋时间对齐 |
| **E. 事件条件的声明生命周期路径检索：EventEffect→Assertion state→Mention evidence** | **5** | **5** | **3** | **5** | **5** | **23** | **最推荐；数据模型已经存在，缺的是查询算法与专属评测** |

### 3.3 唯一推荐主方向

**Research question：**

> How can RAG retrieve verifiable evidence for facts whose validity changes over time, without collapsing historical states into a static entity relation or losing stable semantic relations in an event-only graph?

中文：

> 面对多源动态文本，RAG 如何在不把历史状态压缩成静态实体边、也不把长期事实碎片化为孤立事件的前提下，检索并验证一个事实随事件产生、更新、终止、否认或替代的完整证据路径？

这个问题比“三路如何融合”更 solid，因为它定义了一个具体失败模式、一个明确结构对象（Assertion lifecycle）和一套可测能力（state-at-time、transition、evidence path completeness、conflict）。但论文必须主动讨论 Beyond Retrieval、IA-RAG 和 STAR-RAG，把贡献限定为 provenance-grounded open-domain retrieval，而不能泛称 state-centric memory 或 formal temporal reasoning。

---

## 4. 第四阶段：最好的创新故事与最小算法增强

### 4.1 当前已经具备的创新/价值

以下可如实写入“current system”或方法基础：

1. **Ours 的统一 provenance-aware fusion。** 三路候选先对齐到稳定文档身份；跨分数体系使用 RRF；最终保留各路可回溯原文，而不是用事件摘要自证。
2. **Ours_TKG 的声明重化和事件作用。** Assertion 有有效期/状态；EventEffect 显式表示 creates/updates/terminates/supersedes/denies/confirms；Mention 保留多源证据与立场。
3. **多表示 query linking。** 同一知识库同时可从 Chunk、Entity、Predicate、Assertion、Event 进入，再经统一图扩展到证据。
4. **可重建、可治理身份。** entity resolution、ambiguous review、merge rollback 和文档级 provenance transaction 对真实多源 OSINT 有工程研究价值。

### 4.2 不能声称已经具备的创新

1. **Vector→Semantic→Event 的顺序确认/验证。** 当前没有这种 cross-path pipeline。
2. **真正的因果链。** `precedes/related` 和 EventEffect 不能泛化为事件因果。
3. **跨来源独立验证。** 不同文档不等于独立来源；转载、同源稿和权威性未建模。
4. **完整 Event Chain 检索。** Ours 是可选一跳；TKG 是 bounded expansion，没有全链 completeness objective。
5. **动态知识更新的在线性能。** 有增量/重建语义，但未用 temporal holdout 证明历史状态保留与新状态生效。
6. **Agentic innovation。** 两轮 query rewrite 是常见 Agentic RAG 设计，且现有 30 题 p95 约 91.7 秒，不能成为主故事。

### 4.3 最值得补充的一个算法机制

建议加入 **Query-conditioned Lifecycle Path Retrieval（QLPR）**，而不是再加第四条通路。

#### 输入与路径单位

把候选从“文档”提升为证据路径：

```text
p = (EventMention → Event → EventEffect → Assertion_t → AssertionMention)
```

对于 replacement/change 问题，可扩为：

```text
p = Assertion_old ←SUPERSEDES— Event →CREATES/UPDATES→ Assertion_new
```

对于 conflict 问题：

```text
p = Mention_asserted → Assertion ← Mention_denied
```

#### Query-conditioned 评分

可定义：

```text
S(p|q) = α S_sem(q,p)
       + β S_time(q,p)
       + γ S_transition(intent,p)
       + δ S_evidence(p)
       + ε S_source_independence(p)
       + ζ S_coverage(p)
       - η S_redundancy(p)
```

- `S_sem`：query 与 Assertion/Event/原文 evidence 的组合语义相关性；
- `S_time`：query 时间与 Assertion validity、Event effective time 的一致性；
- `S_transition`：问题要求 change/replace/deny 时，路径是否包含对应 EventEffect；
- `S_evidence`：路径节点是否都有原文 Mention 支持；
- `S_source_independence`：按 publisher/domain/canonical source family 去转载；
- `S_coverage`：是否覆盖问题所需实体、关系、时间和状态；
- `S_redundancy`：重复文档/重复事件惩罚。

最终在 token budget 下选择路径集：

```text
P* = argmax Σ S(p|q) + λ·Coverage(P,q) − μ·Redundancy(P)
     s.t. TokenCost(P) ≤ B
```

这一步将三/五路从“结果投票”升级为**围绕同一生命周期证据路径的交互**：向量只负责 seed；图结构负责合法连接；事件作用和时间约束负责验证路径；原文 Mention 负责证据落地。

#### 为什么这是“最小但足够”的增强

- 数据结构、五类向量、Neo4j expansion 和 Mention 都已存在；不必重写索引系统。
- 主要改动集中在 `UnifiedTKGRetriever`：从文档 RRF 改为 lifecycle path construction、scoring 和 budgeted selection。
- 它直接创造可消融的算法：`-time`、`-transition`、`-source`、`-coverage`、`RRF instead of path selection`。
- 它与 EventRAG/DyG-RAG 的差异可清楚表达：对方恢复 event sequence；本方法恢复 **fact-state transition with evidence**。

### 4.4 如果只做动态路由，创新够不够？

`ours_tkg_dynamic` 已显示固定入口不是最优：同一 30 题上 nDCG@10 从 0.4983 到 0.7015、MRR@10 从 0.4442 到 0.7659，p95 从 3341 ms 降至 545 ms。但 Reviewer 仍会问：

- 阈值和权重是否在 test 上调过？
- 提升来自更少噪声，还是来自 TKG 的时间/事件结构？
- 与 Adaptive-RAG、普通 learned router 或 cross-encoder reranker 相比如何？

所以动态路由适合做次级贡献或效率机制，不宜单独承载整篇论文。最好让它服务于 lifecycle path objective。

---

## 5. 第五阶段：公平 Baseline 体系

### 5.1 最小但有说服力的 7 个 Baseline

| Baseline | 为什么必须比较 | 代表能力 | Ours_TKG 应在哪类任务超过 |
|---|---|---|---|
| 1. **Dense＋BM25＋cross-encoder reranker** | 防止复杂结构只是输给强简单检索；当前 Hybrid 已很强，但还缺统一 reranker | 强平面 hybrid retrieval | temporal transition、state-at-time、conflict、多证据完整性；普通事实不一定显著胜出 |
| 2. **LightRAG** | Ours 的图路直接源于它，也是低/高层 GraphRAG 代表 | vector＋entity/relation graph、local/global | 状态变化、历史有效性、否认/替代；全局主题未必胜 |
| 3. **HippoRAG** | 强多跳图检索，能排除提升只是来自图扩展 | PPR-based associative multi-hop retrieval | 时间约束多跳、状态演化完整路径；纯多跳事实需至少持平 |
| 4. **SiReRAG** | 最接近“多种知识组织协同”的强 indexing baseline | similarity＋relatedness 双树 | 涉及 event effect、state/history、冲突的题；普通 multi-hop 可能相近 |
| 5. **EventRAG** | 事件图、跨文档融合、多事件推理的直接竞争者 | event KG＋iterative event reasoning | 带 assertion validity、source stance 和历史状态的可验证回答 |
| 6. **E²RAG** | 实体—事件双图、状态压缩问题与本文最接近，不能回避 | entity mention/event 双图与 temporal-causal context | 多源事实更新、明确声明生命周期、跨源证据/冲突；叙事因果题不保证超过 |
| 7. **DyG-RAG 或 IA-RAG** | event timeline/interval retrieval 的直接 SOTA 型竞争者；按数据集选择，最好两者都跑 | time-enhanced DEU graph walk；或 interval-algebra traversal | 事实有效期、replace/deny/confirm、source-grounded state transitions；纯时间线/区间题需至少有竞争力 |

**任务特定可选项：** 若加入已有 TKGQA 数据集，再增加 TimeR⁴/STAR-RAG；若主打状态仲裁，必须加入 Beyond Retrieval 的 operator baseline；若主打本体领域适配，将 OG-RAG 替换 SiReRAG；若主打 Agentic 变体，再加入 IRCoT/Adaptive-RAG，但不得挤掉事件/时间直接竞争者。

### 5.2 公平比较约束

1. 相同 corpus snapshot、父文档、Chunk 划分、embedding、生成模型和时间截止点。
2. 同时固定 `top_k` 和最终 context token budget；只固定 top-k 对 event/subgraph 方法不公平。
3. 所有方法返回原文 provenance；若原方法只返回结构摘要，需同时报告其 native setting 与统一 raw-evidence setting。
4. EventRAG/E²RAG/DyG-RAG/IA-RAG 必须按官方机制实现或使用官方代码；不能用弱化版“event vector only”冒充。
5. 分开报告 indexing calls/cost、query latency/cost、LLM calls、storage amplification。
6. 对每题保存完整 trace；阈值只在 dev 调，test 冻结；至少 3 个随机种子或固定 deterministic extraction cache。
7. 对 Ours_TKG 的 entity/event extraction 质量单独评估；不能让更多 LLM 调用和更丰富抽取预算隐性形成不公平优势。

---

## 6. 第六阶段：能够证明创新的实验

### 6.1 Recall@K 高是否足够？

**不够。** Recall@K 只能说明“相关证据进入候选”，不能证明：

- 事件按正确顺序组成链；
- 某事实在指定时间有效；
- 旧状态与新状态没有被压缩；
- 事件确实造成/否认/终止某项声明；
- 多个文档是真正独立的交叉支持，而非转载；
- 最终答案使用了正确证据且没有越过证据推断因果。

而且扩大候选池通常会提高 Recall，却可能降低 precision、nDCG、context precision、faithfulness 和效率。因此至少要同时报告 Precision@K、nDCG@K、MRR、完整证据集召回和固定 token budget 下的答案/引用指标。

### 6.2 指标体系

#### A. 通用检索

- Document/Chunk/Evidence 粒度的 Recall@K、Precision@K、MAP@K、nDCG@K、MRR。
- **Complete Evidence-Set Recall@K**：至少一套可回答问题的 gold evidence set 是否完整进入 top-K。
- Context Precision/Recall：对实际发送给 LLM 的上下文计算，而不是原始候选。

#### B. Event 与生命周期指标

| 新指标 | 为什么传统 Recall 不够 | Gold 构建 | 计算 | 证明的机制 |
|---|---|---|---|---|
| Event Recall/Precision@K | Chunk 命中不表示正确事件被识别 | 人工标注 event mention、canonical event cluster | top-K event 与 gold cluster 的 P/R | Event 独立检索单位 |
| Event Chain Node/Edge Recall | 零散命中 A、C 不代表恢复 A→B→C | 标 gold event nodes 和 typed temporal/effect edges | 分别计算 node recall、edge recall | chain expansion |
| **Lifecycle Path Success@K** | 只有事实或事件一端命中仍无法解释变化 | 标 gold `old assertion—event effect—new assertion` 路径及证据 | top-K 路径中是否存在一条完整等价路径 | 主创新 QLPR |
| **Lifecycle Path Completeness** | Recall 不区分缺 EventEffect、时间或证据 | 对每条 gold path 标 required nodes/edges/mentions | `|retrieved required components| / |gold required components|`，再按题宏平均 | 结构完整性 |
| Temporal Relation F1 | 相关事件不代表顺序正确 | 标 BEFORE/AFTER/SAME/OVERLAP/UNKNOWN | relation micro/macro F1 | 时间边 |
| State-at-Time Accuracy | 文档相关不代表 `t` 时刻状态正确 | 对 `(subject,predicate,time)` 标 value/status | exact accuracy；开放值可 entity-normalized F1 | Assertion validity |
| Transition-Type F1 | “发生变化”不等于正确区分 update/terminate/deny | 标 CREATE/UPDATE/TERMINATE/SUPERSEDE/DENY/CONFIRM | macro-F1 | EventEffect |

#### C. 证据、冲突与可信性

| 指标 | Ground truth | 计算方式 | 注意事项 |
|---|---|---|---|
| Evidence P/R/F1 | 每个 event/assertion/effect 的原文 spans | span overlap/token F1 与 evidence-set F1 | 必须惩罚结构正确但无原文支持 |
| Citation Correctness | citation 是否 entail 邻近 claim | 人工/NLI 校准的 supported/partial/unsupported | 不能只看 citation 是否存在 |
| Citation Completeness | 所有可验证 claims 是否有支持引用 | supported cited claims / all verifiable claims | 与 correctness 分开 |
| Conflict Detection macro-F1 | supported/refuted/disputed/uncertain/NEI | 五分类 macro-F1 | 不同时间状态变化不能误报冲突 |
| Cross-source Verification Accuracy | gold source family、独立性、立场 | 结论＋至少两套独立证据都正确才得分 | 不同文档不自动算独立源 |
| Temporal-conflict Accuracy | 同主体同关系在重叠时间的互斥值 | 对 genuine conflict vs evolution 分类 | 直接证明 valid interval 的价值 |

#### D. 最终回答

- 短答：EM、token F1、Accuracy。
- 长答：atomic claim precision/recall/F1、Gold-point coverage。
- Temporal Answer Accuracy、Conflict-aware Answer Accuracy。
- Faithfulness / Supported-claim rate、Hallucinated-claim rate。
- Answer Relevancy 只作辅助；LLM-as-judge 必须在人评子集上校准。
- 联合主指标建议：**Grounded Lifecycle QA**——答案状态和值正确，且完整 lifecycle evidence path 和引用均正确时才得 1。

#### E. 效率

- Index：LLM calls、tokens、wall time、$/document、docs/min、storage amplification。
- Query：p50/p95 latency、LLM calls、embedding calls、graph expansion nodes、context tokens、$/query。
- 质量—成本曲线：固定 2k/4k/8k context budget 和不同 top-k。

### 6.3 数据集与问题桶

现有 MultiHop-RAG 只能做通用检索 sanity check，不能证明 event/TKG novelty。建议建立 150—250 题主测试集：

| 问题桶 | 比例 | 示例能力 |
|---|---:|---|
| 普通事实 | 15% | 不应因复杂结构退化 |
| 静态多跳关系 | 15% | Entity/Assertion 图 |
| 时间点状态 | 15% | 某日/某年关系值是什么 |
| 状态变化/替代 | 20% | old→event→new |
| 事件顺序/链 | 15% | 完整 temporal chain |
| 多源支持/冲突 | 15% | asserted/denied、转载去重、时间演化 vs 矛盾 |
| 无答案/信息不足 | 5% | evidence sufficiency 与正确拒答 |

每题标：answer points、一个或多个 evidence sets、entities、assertions、events、EventEffects、valid intervals、source family 和 citation spans。采用双人独立标注＋第三人裁决，报告一致性。

---

## 7. 第七阶段：消融实验设计

### 7.1 用户要求的完整通路组合

所有组合必须共享同一抽取产物、同一候选预算和最终 token budget：

1. Vector only
2. Semantic/Assertion only
3. Event only
4. Vector + Semantic
5. Vector + Event
6. Semantic + Event
7. Vector + Semantic + Event（Full）

报告全体和每个 question bucket；特别检查 temporal/change/conflict 桶的增益是否显著高于普通事实桶。

### 7.2 证明“事件结构”而非“更多候选”的关键控制组

| 消融/控制 | 保持不变 | 改变项 | 能回答的问题 |
|---|---|---|---|
| Event text as flat chunks | 相同 event evidence、相同数量 | 去掉 event node/edge/time/status | 提升是否只是更多文本 |
| Shuffled event edges | 相同事件与向量分数 | 随机打乱 temporal/effect edges | 链结构是否真实有用 |
| Event vector only | 相同 event nodes | 去掉结构排序和扩展 | 结构排序贡献 |
| Event union vs RRF | 相同候选 | 简单并集/轮流取 vs RRF | 融合是否超越扩大范围 |
| Equal-candidate budget | 每方法相同候选数 | 强制 event 加入时减少其他候选 | Recall 提升是否由候选膨胀造成 |
| Equal-token budget | 最终 context token 相同 | 只改变证据选择 | 对生成最公平的事件价值 |

### 7.3 Ours_TKG / 建议 QLPR 的必要消融

1. `Entity/Relation edge` vs **reified Assertion**：检验声明节点的价值。
2. `Assertion without valid interval`：检验事实有效期。
3. `Event nodes without EventEffect`：检验事件—声明作用，而不是只看事件共现。
4. `-Transition score`、`-Temporal score`、`-Evidence score`、`-Source independence`、`-Coverage objective`。
5. `Path retrieval → document RRF`：最关键，证明 lifecycle path objective 优于当前 late fusion。
6. `All five seeds` vs query-aware routes；再对比 learned router、rule router、oracle route。
7. `Predicted entity/event alignment` vs oracle alignment：量化 upstream error ceiling。
8. `No event merge / exact-key merge / current merge / oracle merge`：防止错误事件聚类伪造多源支持。
9. `No source-family dedupe` vs dedupe：检验转载是否虚增 verification。
10. `No conflict modeling` vs stance-only vs time-aware NLI：区分简单否定词和真实冲突。

### 7.4 统计检验

- 主指标提供 paired bootstrap 95% CI；Success 型指标用 McNemar。
- 多桶/多指标比较做 Holm 校正并报告 effect size。
- 阈值、RRF k、candidate multiplier、path weights 只能在 dev 选择。
- 对 extraction/generation 非确定性至少 3 次运行，或冻结抽取 cache 后只评检索算法。

---

## 8. 现有实验结果的审稿式解读

### 8.1 Ours：不能只展示最有利的 30 题版本

仓库中 60 题、200 文档的较完整对比为：

| 方法 | Hits@10 | Evidence R@10 | Complete R@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.8889 | 0.6559 | 0.4259 | 0.5353 | 0.6450 |
| Hybrid | **1.0000** | 0.8210 | 0.6111 | **0.6782** | **0.7557** |
| LightRAG | 0.8889 | 0.6281 | 0.3889 | 0.5189 | 0.6560 |
| light_entity | 0.9630 | 0.8241 | **0.6481** | 0.6727 | 0.7029 |
| Ours | 0.9630 | **0.8333** | **0.6481** | 0.6776 | 0.6952 |

这只支持一个克制结论：**事件路带来很小的 Evidence Recall 增量，但尚未改善完整证据集召回，排名首位质量也未改善。** 它不能证明 event chain、验证或时间状态建模有效。

早期 30 题实验中 Ours 相对 light_entity 的 Complete Recall@10 从 0.5714 到 0.7619，时间题提升更明显；这是很好的 hypothesis-generating result，但样本小、只有 21 道有 gold，且明显经历多轮版本调试，必须在冻结的 event-specific test 上复验。

### 8.2 Ours_TKG：动态版比固定版更有希望，但仍未证明 TKG 机制

同一 30 题结果：

| 方法 | Evidence R@10 | Complete R@10 | nDCG@10 | MRR@10 | p95 ms |
|---|---:|---:|---:|---:|---:|
| `ours_tkg` | 0.8095 | 0.4762 | 0.4983 | 0.4442 | 3340.9 |
| `ours_tkg_dynamic` | **0.8333** | **0.5714** | **0.7015** | **0.7659** | **545.3** |
| `agentic_ours_tkg` | 0.8373 | 0.5238 | 0.5300 | 0.4680 | 91742.7 |

解读：

- 五入口全开＋简单 RRF 噪声较大；动态裁剪和 relevance/coverage scoring 显著改善排序与延迟。
- Agent 二次检索只带来很小 Recall 增量，却付出约 92 秒 p95；不适合当前主贡献。
- 这些题不是 lifecycle/state/conflict gold，所以无法把 dynamic 的提升归因于 EventEffect 或 Assertion validity。
- 下一轮最重要的不是扩大 MultiHop-RAG 规模，而是建立能直接测 TKG 独特能力的 test set。

---

## 9. 第八阶段：明确结论

### A. 当前创新性等级

#### Ours：**工程集成创新 / 一般方法创新的下沿**

理由：

- Event 是独立知识单元，事件向量＋结构排序＋一跳扩展不只是普通 metadata filter；统一 provenance 和两级 RRF 也做得完整、可消融。
- 但主要组件均已有强先例：LightRAG 图路、event KG、event vector retrieval、event chain、RRF、多路检索。
- 三路只在 Planner 和最终排名交会，没有跨路交互、确认或验证目标。
- 当前正式规模结果显示事件增益有限，强 Hybrid 基线仍极具竞争力。

#### Ours_TKG：**一般方法创新；有升级为较 solid 方法创新的潜力**

理由：

- Assertion reification、valid interval、source stance、EventEffect lifecycle 和 provenance transaction 形成了一套较完整的动态事实表示，不是简单更换图数据库。
- 五类向量 seed＋统一受约束 expansion 是合理检索架构，但单独看仍是 multi-view indexing＋graph traversal。
- 默认版排序较弱；真正有效的 dynamic 版仍是启发式路由/加权，尚未围绕生命周期路径形成独立优化目标。
- 缺少 state-at-time、transition、conflict、lifecycle path 和最终 grounded answer 证据。

若完成 QLPR、直接对比 EventRAG/E²RAG/DyG-RAG/IA-RAG，并在专属 gold 上显著提升，可合理定位为**较 solid 的联合知识表示＋检索方法创新**。目前不应声称“强方法创新”。

### B. 最大风险（Reviewer 最可能提出）

1. **与 EventRAG/E²RAG/DyG-RAG/IA-RAG/Beyond Retrieval 重合过大。** 三路、实体—事件双结构、事件链、区间时间检索和状态仲裁都不是空白。
2. **三路/五入口只是 late fusion。** RRF 奖励共识但不等于 semantic confirmation 或 event verification。
3. **提升来自扩大候选池或更多 LLM 抽取预算。** Equal-candidate/equal-token 控制缺失。
4. **TKG 结构没有被 event-specific benchmark 直接验证。** MultiHop-RAG 的普通 evidence recall 不能证明 valid interval/EventEffect。
5. **跨源验证表述过强。** 当前“多源”按不同文档近似，未处理转载、来源依赖、权威性和 time-aware contradiction。
6. **默认 Ours_TKG 结果较弱且 dynamic 权重可能显得手调。** 需要 dev/test 冻结、学习/优化依据和权重敏感性。
7. **抽取误差可能破坏图。** entity over-merge、event over-merge 和错误 fact_effect 会产生错误生命周期，需要模块指标与 oracle 消融。

### C. 最推荐的论文故事：Introduction 五段式逻辑

**第一段：问题。** 多源动态语料中的知识不是一组静态事实。某一关系可能由事件创建，随后被更新、终止、否认或替代；用户常问的是“某时点什么成立”“为何后来改变”“哪些来源支持或反驳”。仅召回语义相关段落容易遗漏完整变化路径。

**第二段：现有方法为何不足。** Dense/Hybrid RAG 能发现相关文本但不显式维护事实状态；GraphRAG 组织实体关系，但 canonical entity/edge 合并容易压平历史状态；EventRAG、E²RAG、DyG-RAG 和 IA-RAG 已显著改善事件、实体—事件上下文与区间时间链检索，Beyond Retrieval 也已证明双时态状态仲裁的价值。然而这些方法分别侧重 event sequence、formal temporal traversal 或领域内 persistent state，尚未统一解决开放域多源文档中“事件如何改变一项可追溯声明、每一状态由哪些来源以何种立场支持”的证据路径检索。

**第三段：核心方法。** 我们构造 evidence-grounded temporal assertion graph：稳定实体之间的关系被重化为带有效期和状态的 Assertion；每个来源通过 Mention 提供原文证据和立场；Event 通过 typed EventEffect 显式创建、更新、终止、替代、否认或确认 Assertion，从而同时保留稳定语义事实与动态演化。

**第四段：检索协同。** 查询首先通过多视图向量索引定位 Chunk、Entity、Predicate、Assertion 和 Event seeds；随后进行 query-conditioned lifecycle path retrieval，在时间、转移类型、来源独立性、证据完整性和 token budget 约束下选择“Event→Assertion transition→Mention evidence”路径。向量负责发现，图约束负责连接，事件作用与有效期负责验证状态，原文 Mention 负责可追溯证据。

**第五段：如何证明。** 在普通事实、多跳、state-at-time、replacement、event chain、conflict 和 unanswerable 问题上，与强 Hybrid、LightRAG、HippoRAG、SiReRAG、EventRAG、E²RAG、DyG-RAG/IA-RAG 公平比较；除通用 Recall/nDCG 外，重点报告 Lifecycle Path Success/Completeness、State-at-Time Accuracy、Transition F1、跨源证据和 grounded answer/citation 指标，并通过 equal-budget 与结构打乱消融证明提升来自生命周期结构而非更多候选。

### D. 最终 Contribution（最多 3 条）

> 下面按“建议完成 QLPR 后可投稿的版本”撰写；当前代码已经支撑第一条的数据结构主体，但第二、三条尚需实现/实验。

1. **问题 → 方法 → 作用：** 针对开放域多源 RAG 中实体中心 KG 压平历史状态、事件中心图又难以表达持续事实及其来源立场的问题，我们提出 provenance-grounded temporal assertion representation，以带有效期的 Assertion、来源 Mention 和 typed EventEffect 联合表示事实及其变化，使历史状态、否认、替代和原文证据能够共存。这里的贡献是三者在开放域 RAG 中的联结，不是双时态、supersede 或事件节点本身。
2. **问题 → 方法 → 作用：** 针对相似 Chunk 或孤立 Event 无法恢复可验证状态变化的问题，我们提出 query-conditioned lifecycle path retrieval，从多视图向量 seeds 出发，在时间、转移类型、证据和来源约束下选择 Event→Assertion→Mention 路径，从而检索完整而非零散的变化证据。
3. **问题 → 方法 → 作用：** 针对传统 Recall 无法判断动态事实是否被正确恢复，我们建立覆盖 state-at-time、transition、event chain、temporal conflict 和 cross-source evidence 的评测协议，并用 equal-token/structure-shuffle/oracle 消融区分结构收益与候选扩张。

### E. 一句话定位

**English:** We propose evidence-grounded lifecycle retrieval over a temporal assertion graph, where events explicitly create, update, terminate, supersede, deny, or confirm time-scoped facts and every retrieved transition is traced back to source mentions.

**中文：** 我们提出基于时序声明图的证据落地生命周期检索，使事件能够显式创建、更新、终止、替代、否认或确认带有效期的事实，并将每条状态变化回溯到多源原文证据。

---

## 10. 建议的下一步优先级

1. **先冻结论文问题与 method boundary：** 主方法用 `ours_tkg_dynamic + QLPR`，`ours` 作为架构演进/消融，不再作为最终 strongest method。
2. **实现 lifecycle path retrieval：** 从 Neo4j expansion 返回 typed paths，不再先折叠为文档；实现 path score 与 budgeted selection。
3. **做 40—60 题 pilot gold：** 重点标 state-at-time、replacement、denial、source family 和完整路径；先验证指标与错误类型。
4. **加最危险 baseline：** EventRAG、E²RAG、DyG-RAG/IA-RAG；若主打 state arbitration 再加 Beyond Retrieval operator baseline。资源有限时宁可少跑普通 RAG，也不能缺直接竞争者。
5. **完成 equal-budget 和结构破坏消融：** flat event text、shuffled edges、no EventEffect、RRF vs path objective。
6. **pilot 成立后扩到 150—250 题 test：** 冻结权重、给 paired CI，并补最终答案/引用指标。

---

## 11. 核心参考文献

1. Sarthi et al. [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html). ICLR 2024.
2. Edge et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). arXiv 2024; Microsoft GraphRAG research work.
3. Guo et al. [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-emnlp.568/). Findings of EMNLP 2025.
4. Gutiérrez et al. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html). NeurIPS 2024.
5. Zhang et al. [SiReRAG: Indexing Similar and Related Information for Multihop Reasoning](https://openreview.net/forum?id=yp95goUAT1). ICLR 2025.
6. Luo et al. [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](https://proceedings.neurips.cc/paper_files/paper/2025/file/33ca0b1102b54c191a9a45a05adafaf4-Paper-Conference.pdf). NeurIPS 2025.
7. Yang et al. [EventRAG: Enhancing LLM Generation with Event Knowledge Graphs](https://aclanthology.org/2025.acl-long.830/). ACL 2025.
8. Zhang et al. [Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graph for Retrieval-Augmented Generation](https://aclanthology.org/2026.eacl-long.90/). EACL 2026.
9. Sun et al. [DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning](https://arxiv.org/abs/2507.13396). arXiv 2025.
10. Qian et al. [TimeR⁴: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering](https://aclanthology.org/2024.emnlp-main.394/). EMNLP 2024.
11. Sharma et al. [OG-RAG: Ontology-grounded Retrieval-Augmented Generation for Large Language Models](https://aclanthology.org/2025.emnlp-main.1674/). EMNLP 2025.
12. Trivedi et al. [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/). ACL 2023.
13. Jeong et al. [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://aclanthology.org/2024.naacl-long.389/). NAACL 2024.
14. Asai et al. [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/pdf?id=hSyW5go0v8). ICLR 2024.
15. Liu et al. [TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/40489). AAAI 2026.
16. Es et al. [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/). EACL Demo 2024.
17. Saad-Falcon et al. [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://aclanthology.org/2024.naacl-long.20/). NAACL 2024.
18. Gao et al. [Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/). EMNLP 2023.
19. Petroni et al. [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://aclanthology.org/2021.naacl-main.200/). NAACL 2021.
20. Bugert et al. / relevant cross-document event work may be complemented in implementation evaluation by [Cross-document Event Extraction via CLES](https://aclanthology.org/2024.findings-acl.114/), Findings of ACL 2024, for event coreference and cluster metrics.
21. Zhu et al. [Right Answer at the Right Time: Temporal Retrieval-Augmented Generation via Graph Summarization](https://arxiv.org/abs/2510.16715). arXiv 2025 (STAR-RAG).
22. Wang et al. [IA-RAG: Interval-Algebra-Driven Temporal Reasoning for Dynamic Knowledge Retrieval](https://arxiv.org/abs/2606.06044). arXiv 2026.
23. Zhao et al. [Beyond Retrieval: Bi-Temporal State Arbitration for Longitudinal Healthcare Agents](https://aclanthology.org/2026.knowfm-1.10/). ACL KnowFM Workshop 2026.
