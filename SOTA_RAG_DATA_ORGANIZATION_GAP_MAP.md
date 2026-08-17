# RAG 数据组织、关联与融合方向 SOTA 地图及可实现 GAP

更新日期：2026-08-13

## 0. 范围与判断标准

本报告不再从 Ours/Ours_TKG 倒推竞争者，而是先扫描 RAG 中与**数据组织、知识关联、跨文档融合、图检索、动态组织、事件与时间组织**相关的主要研究支线，再判断各支线尚未解决的问题中，哪些有条件与现有工作交汇。

本报告按用户要求不讨论来源可信度、权威度或置信度校准。“证据回溯”仅表示对象可以回到原文 passage/document，不表示对来源质量作判断。

“SOTA”不是一个统一榜单：RAPTOR、GFM-RAG、EventRAG、DyG-RAG 分别优化不同任务和 benchmark。因此下文的“SOTA/代表性前沿方法”表示该子方向中近期发表、机制有代表性或在相应 benchmark 上表现领先的方法。

### 仓库可用性评级

| 评级 | 含义 |
|---|---|
| **A** | 官方完整仓库；有安装、运行入口和评测/复现脚本；适合作为正式 baseline |
| **B** | 官方研究代码基本完整；可以运行，但需要适配数据、模型/API、依赖或手工修补 |
| **C** | 仓库存在，但文档/代码明显不完整，只适合作为参考实现 |
| **D** | 未找到官方独立仓库，或当前仓库只有占位文件，不能作为可运行 baseline |

仓库状态基于论文、官方 GitHub 页面和 2026-08-13 对仓库结构的核验；“可用”不等于已经在本机完整复现实验结果。

### 实现难度评分

| 分数 | 预计工作 |
|---:|---|
| **1** | 现有代码上数天内完成 |
| **2** | 约 1–2 周，主要是检索与排序逻辑 |
| **3** | 约 3–6 周，需要新模块、标注或较多实验 |
| **4** | 约 2–4 月，涉及数据模型、抽取、检索与评价联动 |
| **5** | 约 4–9 月以上，涉及持续建库、训练、新 benchmark 或大规模人工标注 |

---

## 1. 子领域地图

```text
数据组织与关联
├── 层次/多粒度组织
│   ├── RAPTOR
│   ├── SiReRAG
│   ├── HiRAG / LeanRAG
│   └── SAKI-RAG
├── 图结构组织与图上检索
│   ├── LightRAG / HippoRAG
│   ├── SubgraphRAG / GNN-RAG / GFM-RAG / GraphFlow
│   ├── HyperGraphRAG / OG-RAG
│   └── BYOKG-RAG / HybGRAG
├── 查询期间动态组织
│   ├── RAS
│   └── KAIR / GraphAnchor
├── 事件与时态组织
│   ├── EventRAG
│   ├── E²RAG
│   ├── DyG-RAG
│   └── IA-RAG
└── 多源融合与冲突组织
    ├── MultiRAG
    ├── HF-RAG
    ├── CARE / TruthfulRAG
    └── Astute RAG
```

最重要的观察是：

- 层次化方法主要解决**粒度和上下文范围**；
- GraphRAG 方法主要解决**实体/关系连接与多跳访问**；
- 超图/本体方法主要解决**高阶关系和领域约束**；
- 动态组织方法主要解决**当前查询需要哪些知识**；
- 事件/时态方法主要解决**事件顺序与时间关系**；
- 多源融合方法主要解决**不同检索源或不同文档的结果怎样合并**。

尚未被这些支线共同解决的是：**一个规范化断言如何被事件改变、如何保持历史状态、如何跨文本—断言—事件视图补全证据，以及这些对象如何持续写回同一个长期索引。**

---

## 2. 第一组：层次化和多粒度数据组织

### 2.1 RAPTOR — Tree-organized multilevel summaries

- 论文：[RAPTOR, ICLR 2024](https://openreview.net/forum?id=GN921JHCRw)
- 官方仓库：[parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)
- 仓库评级：**B+**。MIT；有库代码和 demo，但最后主要更新较早，依赖和模型配置需要适配。

**解决的问题**

RAPTOR 递归聚类文本块并生成多层摘要，使检索既能访问细节叶节点，也能访问全局抽象节点，解决长文档中固定 chunk 只能看到局部上下文的问题。

**仍未解决**

- 聚类边表达语义相似，不表达实体关系、事件关系或事实状态变化；
- 上层摘要可能压缩掉时间条件、否定、例外和来源差异；
- 对跨文档同一断言的合并、冲突和演化没有专门表示；
- 增量加入新文档时可能需要重聚类/重摘要；
- 检索目标仍是树节点，不是完整的关系或事件证据路径。

**可与现有工作交汇的 GAP**

把层次摘要节点与 TKG 中的 Assertion/Event/Mention 显式对齐；查询先用摘要层定位主题，再沿断言—事件路径恢复未被摘要保留的精确信息。实现难度 **3/5**，交汇强度 **3/5**。

### 2.2 SiReRAG — Similarity tree + relatedness tree

- 论文：[SiReRAG, ICLR 2025](https://openreview.net/forum?id=yp95goUAT1)
- 官方仓库：[SalesforceAIResearch/SiReRAG](https://github.com/SalesforceAIResearch/SiReRAG)
- 仓库评级：**B-**。代码、数据和评测存在，但主要是 notebook，并要求替换 LlamaIndex RAPTOR pack 中的文件；没有清晰许可证。

**解决的问题**

SiReRAG 同时建立基于语义相似性的树和基于 proposition/entity 相关性的树，并将所有节点放入统一检索池，解决只按相似性组织或只按图关系组织对多跳信息覆盖不足的问题。

**仍未解决**

- 两棵树最终进入统一池，但缺少查询时明确的跨视图交互和路径补全；
- proposition/entity relatedness 仍以静态关联为主；
- 没有事件改变 assertion state 的表示；
- 没有时间有效区间和历史状态；
- 统一池可能把不同粒度、不同语义角色的分数混在一起。

**可交汇 GAP**

从“双树统一候选池”推进到“文本发现—断言结构—事件变化的类型化协同检索”，并用必要 evidence role 而非统一相似度选择上下文。难度 **3/5**，交汇强度 **5/5**。

### 2.3 HiRAG / LeanRAG — Hierarchical KG aggregation

- 论文：[HiRAG, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.321/)
- 仓库：[hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG)，评级 **B+**，MIT，有完整代码结构。
- 论文/仓库：[LeanRAG, AAAI 2026](https://github.com/KnowledgeXLab/LeanRAG)，评级 **B**，有建图、查询、数据和评测代码，但未声明许可证。

**解决的问题**

这类方法对实体进行层次聚合，从细粒度实体向上检索语义聚合结构，减少 GraphRAG 中局部碎片和全局社区摘要之间的断裂，并降低冗余。

**仍未解决**

- 层次聚合主要按语义/实体社区，不表达事实在时间上的前后状态；
- 聚合可能把同一实体在不同时段的状态压到一个节点或摘要；
- 对跨文档事件共指、Assertion 更新/终止/替代没有专门机制；
- 层次路径是主题导航，不等于事实演化路径。

**可交汇 GAP**

构造 time/state-sensitive hierarchical aggregation：聚合时对不同有效期、不同 EventEffect 的 assertion 保持边界；查询可在主题层定位后下钻到 lifecycle path。难度 **4/5**，交汇强度 **4/5**。

### 2.4 SAKI-RAG — Sentence relation repair for long documents

- 论文：[SAKI-RAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.63/)
- 仓库评级：**D**。截至核验时间未找到论文明确链接的官方可运行仓库。

**解决的问题**

通过 SentenceAttnLinker 建立句间注意力关系，再用语义和上下文双轴检索，修复细粒度切块造成的跨段语义断裂。

**仍未解决**

- 句间 attention 是统计关联，不区分 Entity、Assertion、Event 等关系类型；
- 主要面向单篇长文档，不处理跨来源实体/事件对齐；
- 不能区分背景相关、事实支持、状态变化等 evidence role；
- 没有时间和事件演化表示。

**可交汇 GAP**

将句间软连接作为图抽取失败时的 fallback，并通过 Mention 链接到规范化 Assertion/Event；这可提高召回和抗抽取缺失能力。难度 **2/5**，交汇强度 **3/5**。

---

## 3. 第二组：图结构组织与图上检索

### 3.1 LightRAG — Dual-level entity-relation GraphRAG

- 论文：[LightRAG](https://arxiv.org/abs/2410.05779)，Findings EMNLP 2025
- 官方仓库：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- 仓库评级：**A**。MIT；持续维护；有 SDK、服务、Neo4j、多种存储、复现说明和测试，是最容易落地的正式 baseline。

**解决的问题**

从文档抽取实体和关系，结合向量索引，以 low-level/local 和 high-level/global 查询降低 GraphRAG 的复杂度并支持增量文档插入。

**仍未解决**

- canonical entity/relation 容易把同一关系在不同时段的状态压成一条边；
- relation 描述不等于可独立寻址、有有效期和状态的 Assertion；
- 增量插入不等于一致的事实状态更新；
- 对事件、EventEffect、历史状态和跨文档事件共指没有专门模型；
- local/global/hybrid 更接近模式选择，不是多类证据间的协同补全。

**可交汇 GAP**

在 LightRAG 式 Entity/Relation index 上增加 reified temporal Assertion、EventEffect 和 Mention backlink，并对比“关系描述累加”与“显式状态迁移”的检索差异。难度 **4/5**，交汇强度 **5/5**。

### 3.2 HippoRAG — KG + Personalized PageRank associative retrieval

- 论文：[HippoRAG, NeurIPS 2024](https://arxiv.org/abs/2405.14831)
- 官方仓库：[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- 仓库评级：**A**。MIT；提供代码、测试、复现目录和多个运行后端，持续维护。

**解决的问题**

把文档中的开放知识图和 Personalized PageRank 结合，实现类似关联记忆的多跳检索，提高跨文档信息整合能力。

**仍未解决**

- PageRank 扩散强调连通性，不能保证检出的节点构成有语义角色的完整证据链；
- 图边大多是静态开放关系；
- 高度节点可能带来流行度偏置和无关扩散；
- 不表达事件怎样改变 assertion；
- 没有 state-at-time 或跨时间状态隔离。

**可交汇 GAP**

用 typed temporal constraints 限制图扩散，并把目标从 node relevance 改成 Assertion—EventEffect—Mention 路径覆盖。难度 **3/5**，交汇强度 **4/5**。

### 3.3 SubgraphRAG / GNN-RAG — Efficient KG subgraph retrieval

- SubgraphRAG：[ICLR 2025 官方仓库](https://github.com/Graph-COM/SubgraphRAG)，评级 **A-/B+**，MIT，含训练和实验代码。
- GNN-RAG：[Findings ACL 2025](https://aclanthology.org/2025.findings-acl.856/)，[官方仓库](https://github.com/cmavro/GNN-RAG)，评级 **B**，代码完整但主要针对 WebQSP/CWQ 既有 KG，依赖和训练设置较旧。

**解决的问题**

使用轻量 GNN 或子图检索器定位答案节点和连接路径，降低 LLM 逐步遍历 KG 的成本，并改善多跳、多实体 KGQA。

**仍未解决**

- 通常假设 KG 已经存在且质量较好；
- 目标是 answer entity/path，不是从多源非结构文档构造和维护知识；
- 对抽取噪声、实体/事件共指和原文 evidence grounding 处理有限；
- KG 边默认静态，缺少 valid interval 和状态迁移；
- shortest path 不一定是最充分的证据路径。

**可交汇 GAP**

训练或规则化一个面向异构 TKG 的 typed subgraph retriever，节点包括 Chunk/Assertion/Event，边类型进入 scoring；但需要专属训练数据或弱监督。难度 **5/5**，交汇强度 **4/5**。

### 3.4 GFM-RAG — Graph foundation retriever

- 论文：[GFM-RAG, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/33ca0b1102b54c191a9a45a05adafaf4-Abstract-Conference.html)
- 官方仓库：[RManLuo/gfm-rag](https://github.com/RManLuo/gfm-rag)
- 仓库评级：**A**。Apache-2.0；有 pip 安装、文档、预训练模型、raw document 与 bring-your-own-graph 两种入口，仍在维护。

**解决的问题**

通过预训练 graph foundation model 在未见图上零样本进行 query-dependent 图推理和文档排序，缓解启发式图遍历、图噪声和跨数据集泛化问题。

**仍未解决**

- 输入图主要是通用 Entity/Relation 图，缺少原生的事件—断言状态语义；
- 预训练目标主要围绕相关文档/实体，未优化时间链或状态迁移完整性；
- 对持续加入新事件、retroactive update 的图分布变化未专门处理；
- 不能自动修复错误的 assertion/event alignment；
- 输出仍主要用于文档排序，不保证 evidence role 完整。

**可交汇 GAP**

把 GFM-RAG 作为统一 TKG 上的强 graph retriever，对 heterogeneous node/edge type 做适配；或者不训练模型，只将其作为对照证明 lifecycle-aware traversal 的必要性。适配难度 **4/5**；重新训练难度 **5/5**；交汇强度 **4/5**。

### 3.5 GraphFlow — Reward-proportional diverse graph retrieval

- 论文：[GraphFlow, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/89d0d5c2f720921df93bbb8fef514571-Abstract-Conference.html)
- 仓库评级：**D/B-**。论文有 supplemental，但截至核验时间未找到清晰的作者官方独立 GitHub 仓库；不适合作为首批工程 baseline。

**解决的问题**

用 transition-based flow matching 同时学习 retrieval policy 和 flow estimator，把终局奖励分解到中间检索状态，在 STaRK 的 text-rich KG 上提高准确性和多样性。

**仍未解决**

- 奖励仍围绕相关答案/文档，不表达 lifecycle completeness；
- 依赖训练和 benchmark reward，迁移到新型 TKG 需要构造监督；
- 未解决 KG 从原文构建、对象对齐和动态更新；
- 多样性不等于互补证据角色覆盖；
- 没有事件和时间状态语义。

**可交汇 GAP**

将 flow reward 改成路径完整度和 evidence role coverage，探索多条互补 assertion/event path；研究价值高，但训练与数据成本大。难度 **5/5**，交汇强度 **4/5**。

### 3.6 HyperGraphRAG — Higher-order knowledge hypergraph

- 论文：[HyperGraphRAG, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/df55ee6e59f8ac4a625219e11fe9ddba-Paper-Conference.pdf)
- 官方仓库：[LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG)
- 仓库评级：**A-/B+**。MIT；有构图、查询和 evaluation 入口，quick start 清楚，但主要依赖 OpenAI，仍是研究型代码。

**解决的问题**

把包含多个实体的高阶事实表示为 hyperedge，克服二元三元组将 n-ary 关系拆碎造成的信息损失，适合多条件、多个参与者共同构成的知识。

**仍未解决**

- hyperedge 主要表达同一事实内部的高阶共现/关系，不表达该事实的时间状态变化；
- 未区分事实本身与改变事实的事件；
- 多个文档提及同一 hyperedge 时的 mention 对齐和状态演化不足；
- 对动态更新、撤销和替代处理有限；
- 高阶结构仍可能缺少原文级证据回链。

**可交汇 GAP**

把 reified Assertion 视作有角色槽位的高阶对象，再用 EventEffect 连接 assertion 的多个状态，形成 temporal hypergraph；但需避免方法复杂度过高。难度 **4/5**，交汇强度 **4/5**。

### 3.7 OG-RAG — Ontology-grounded hypergraph

- 论文：[OG-RAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1674/)
- 官方仓库：[microsoft/ograg2](https://github.com/microsoft/ograg2)
- 仓库评级：**B+**。MIT；仓库有完整模块、配置、样例数据和评测，但 README 中仍有错误 clone 示例，Azure/OpenAI 和领域 ontology 配置成本较高。

**解决的问题**

用领域 ontology 约束事实抽取并构造 ontology-grounded hypergraph，再选择最小相关 hyperedge 集，改善专业领域中的概念歧义和事实组织。

**仍未解决**

- 依赖已有领域 ontology，开放域扩展成本高；
- ontology 描述“允许什么类型的关系”，不描述事实随事件怎样改变；
- ontology/hyperedge 多为静态；
- 新关系、新事件类型可能超出 schema；
- minimal hyperedge set 未必覆盖前态—事件—后态。

**可交汇 GAP**

将 EventEffect 和 Assertion state transition 作为轻量上层 schema，而不是依赖完整领域 ontology；比较 open schema 与 domain ontology 在演化任务上的互补性。难度 **3/5**，交汇强度 **4/5**。

### 3.8 BYOKG-RAG — Multi-strategy retrieval over custom KGs

- 论文：[BYOKG-RAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1417/)
- 官方仓库：[awslabs/graphrag-toolkit](https://github.com/awslabs/graphrag-toolkit)
- 仓库评级：**A-/B+**。Apache-2.0；项目持续维护，有 BYOKG 示例和多种图工具，但默认生态偏 Amazon Bedrock/Neptune，迁移到当前栈需适配。

**解决的问题**

LLM 生成 question entities、candidate answers、reasoning paths 和 OpenCypher，再由多个图工具链接和检索，降低单一 entity linking 或单一遍历策略对 custom KG 的脆弱性。

**仍未解决**

- 假设用户已经有 KG；
- 多策略检索是工具层融合，不解决多源文档怎样合并为一致的长期 index；
- Cypher/path 未包含事实有效期和状态变化时仍会返回静态关系；
- 不保证从规范化图对象回到完整原文 evidence set；
- 迭代 LLM 调用带来延迟和成本。

**可交汇 GAP**

把 TKG 的三类查询能力暴露为 typed tools：text discovery、assertion traversal、event transition；由 planner 根据缺失的 evidence role 调用，而不是泛化 multi-agent。难度 **3/5**，交汇强度 **4/5**。

### 3.9 HybGRAG — Textual + relational retrieval on SKB

- 论文：[HybGRAG, ACL 2025](https://aclanthology.org/2025.acl-long.43/)
- 仓库评级：**D/C**。未找到清晰的作者官方完整实现仓库；可使用开源的 [STaRK benchmark](https://github.com/snap-stanford/stark) 重建，但复现成本高。

**解决的问题**

在 semi-structured knowledge base 中联合文本属性与图关系条件检索，解决纯文本或纯关系检索无法满足混合约束的问题。

**仍未解决**

- SKB 关系预先存在且大多静态；
- 联合 textual/relational constraint 不等于事件演化或状态恢复；
- 不处理从多源非结构文档抽取、融合、历史维护；
- 目标常是 item retrieval，而非 evidence path。

**可交汇 GAP**

把 temporal/event/state 作为第三类结构约束，在相同文本＋关系条件下比较 state-sensitive retrieval。难度 **3/5**，交汇强度 **3/5**。

---

## 4. 第三组：查询期间的动态组织与关联

### 4.1 RAS — Retrieval-And-Structuring

- 论文：[RAS, ICLR 2026](https://iclr.cc/virtual/2026/poster/10008199)
- 官方仓库：[pat-jj/RAS](https://github.com/pat-jj/RAS)
- 仓库评级：**B+**。Apache-2.0；包含 framework、knowledge indexing、text-to-triples、训练数据处理和 baselines，代码可研究使用，但系统训练/配置复杂。

**解决的问题**

交替执行 query planning、targeted retrieval 和 query-specific KG construction，让当前查询逐步把散乱文本组织成可推理结构，解决单次检索和无结构上下文的局限。

**仍未解决**

- 图主要为当前查询临时构造，没有成为跨查询复用、可持续维护的长期知识索引；
- query-time triple extraction 重复且昂贵；
- 临时图未显式表达历史状态和 EventEffect；
- 多次查询可能对同一事实得到不同结构；
- 没有系统解决实体/事件规范化与跨文档长期一致性。

**可交汇 GAP**

将 persistent TKG 与 query-specific working graph 分层：长期 TKG 提供规范对象和历史状态，工作图只记录当前问题的已覆盖/缺失证据。难度 **4/5**，交汇强度 **5/5**。

### 4.2 KAIR / GraphAnchor — Evolving query-time anchoring index

- 论文：[Graph-Anchored Knowledge Indexing / KAIR, 2026](https://arxiv.org/abs/2601.16462)
- 官方仓库：论文链接现落到 [NEUIR/KAIR](https://github.com/NEUIR/KAIR)
- 仓库评级：**B**。MIT；有环境、索引、运行和评测说明，但数据依赖 DeepNote/Google Drive，规模较小，属于研究型实现。

**解决的问题**

在迭代检索过程中不断更新 anchoring index，标记关键实体/关系，帮助 LLM 判断证据是否充足并生成后续查询，解决多跳证据散落和检索方向不稳定。

**仍未解决**

- “evolving”主要指当前查询的临时 index，而不是语料随时间变化的长期 index；
- anchor graph 组织相关实体/关系，不表达 assertion validity 和事件状态迁移；
- 没有持久化处理新文档、迟到事件或历史版本；
- sufficiency 主要由 LLM 判断，缺少结构化 evidence-role contract；
- 只在通用多跳 QA 上验证。

**可交汇 GAP**

把 sufficiency 明确化为 lifecycle path/role coverage，并让临时 anchor graph 从 persistent TKG 读取、把新增 Mention/Assertion proposal 写回待审核层。难度 **4/5**，交汇强度 **5/5**。

---

## 5. 第四组：事件和时态数据组织

### 5.1 EventRAG — Cross-document event knowledge graph

- 论文：[EventRAG, ACL 2025](https://aclanthology.org/2025.acl-long.830/)
- 官方仓库：[Ryaang/EventRAG](https://github.com/Ryaang/EventRAG)
- 仓库评级：**C+/B-**。仓库确有 EventRAG、Neo4j、reproduce 和测试代码，但文档极少、无许可证、提交较早，复现和迁移风险较大。

**解决的问题**

以事件为主要知识单元，融合跨文档相似事件/实体，建立时间和逻辑关系，并通过事件知识扩展和迭代检索支持多事件推理。

**仍未解决**

- 长期稳定事实和事件之间的作用关系没有细分成 Assertion state transition；
- Event chain 能说明“发生了什么”，不一定说明“哪个关系因此从何状态变为什么状态”；
- event merge 容易把相似但不同 occurrence 合并；
- 原文 Mention 回溯和 statement-level lifecycle 不是核心优化对象；
- 事件图与 raw text/静态关系的协同目标不够形式化。

**可交汇 GAP**

从 event sequence 推进到 `EventEffect → Assertion before/after state → Mention`；这是最直接的可交汇问题之一。难度 **4/5**，交汇强度 **5/5**。

### 5.2 E²RAG — Entity-event dual graph

- 论文：[E²RAG, EACL 2026](https://aclanthology.org/2026.eacl-long.90/)
- 官方仓库：[zyzhang1130/EntityEventRAG](https://github.com/zyzhang1130/EntityEventRAG)
- 仓库评级：**B+**。MIT；有 Python package、Docker、配置、examples、experiments 和 batch evaluation，适合作为直接 baseline，但主要针对小说/叙事 ChronoQA。

**解决的问题**

把 entity-mention graph 与 event graph 分开，再用二部映射连接；每个 entity mention 保留所在事件上下文，避免 canonical entity 合并导致角色状态和时序背景被压缩。

**仍未解决**

- 主要保留 entity mention 的动态上下文，没有把 relation/assertion 本身作为有状态对象；
- 适合叙事人物演化，但多源事实更新、否认、替代等 effect 类型较弱；
- event-frontier 扩展较局部；
- 没有长期增量维护协议；
- 缺少跨文本—断言—事件的集合级检索目标。

**可交汇 GAP**

把“entity mention 不合并”推进到“canonical entity 可以合并，但 assertion mention、valid interval 和 event-conditioned state 不被压缩”。难度 **4/5**，交汇强度 **5/5**。

### 5.3 DyG-RAG — Dynamic Event Unit and timeline retrieval

- 论文：[DyG-RAG, 2025](https://arxiv.org/abs/2507.13396)
- 官方仓库：[RingBDStack/DyG-RAG](https://github.com/RingBDStack/DyG-RAG)
- 仓库评级：**B**。有安装、模型下载、OpenAI/本地模型示例、三个时态 QA 数据集和 reproduce 目录；无明确许可证、维护有限。

**解决的问题**

用带时间锚点的 Dynamic Event Unit 建事件图，结合时间增强向量、cross-encoder、multi-seed graph walk 和 Time-CoT，恢复按时间排序的事件序列。

**仍未解决**

- 时间邻近和共享实体建立的事件边不等于真实状态作用；
- timeline 适合“事件先后”，不直接表达一项 assertion 的创建、更新、终止和替代；
- 没有区分长期事实和瞬时事件；
- 多源 mention 融合和增量事件共指仍是瓶颈；
- 时间链完整不等于关系状态正确。

**可交汇 GAP**

把 timeline retrieval 与 assertion lifecycle retrieval 对照：前者恢复事件序列，后者恢复事件对特定关系状态的作用。难度 **4/5**，交汇强度 **5/5**。

### 5.4 IA-RAG — Interval algebra temporal retrieval

- 论文：[IA-RAG, 2026](https://arxiv.org/abs/2606.06044)
- 声称的官方仓库：[xiaoAugenstern/LogicalRAG_TemporalQA](https://github.com/xiaoAugenstern/LogicalRAG_TemporalQA)
- 仓库评级：**D**。核验时仓库只有一个 README，不能视为已开源可用实现。

**解决的问题**

用 Interval Event Unit、Thematic Forest 和 Allen interval algebra 表示 before/after/overlap/contain 等区间关系，并通过时间约束收紧模糊区间。

**仍未解决**

- 形式化时间关系强，但不表达事件怎样改变 assertion state；
- 主要对既有事件单元进行时间推理，不解决跨文档抽取和融合；
- 没有 Text/Assertion/Event 多视图协同；
- 不处理持续索引维护；
- 当前代码不可用，复现风险高。

**可交汇 GAP**

只借用 Allen relations/interval normalization 作为时间约束模块，不必完整复现 IA-RAG；在 TKG 中将区间逻辑用于过滤非法 lifecycle path。难度 **3/5**，交汇强度 **3/5**。

---

## 6. 第五组：多源结果与知识融合

### 6.1 MultiRAG — Multi-source line-graph fusion

- 论文：[MultiRAG, ICDE 2025](https://arxiv.org/abs/2508.03553)
- 官方仓库：[wuwenlong123/MultiRAG](https://github.com/wuwenlong123/MultiRAG)
- 仓库评级：**B**。有 src、实验、数据、评测、依赖和中英文 README；仓库规模不大且无许可证，需要先验证论文复现范围。

**解决的问题**

构造 multi-source line graphs 聚合不同知识源的逻辑关联，并在 graph/node 层过滤不一致节点，缓解多源稀疏与相互干扰。

**仍未解决**

- 多源融合主要对检索结果/知识节点进行聚合和过滤，未形成长期统一的 statement-level index；
- 不区分“同一事实在不同时间更新”与“同一时间真正矛盾”；
- 没有明确 EventEffect 和 assertion history；
- 融合后的节点难以表达前态—变化—后态；
- 对持续加入新来源的 incremental consistency 评价有限。

**可交汇 GAP**

将跨源重复、时间更新和真实冲突分为不同 fusion operator；保留所有 AssertionMention，但只在规范 Assertion/State 层合并。难度 **4/5**，交汇强度 **5/5**。

### 6.2 HF-RAG — Hierarchical rank fusion across sources

- 论文：[HF-RAG, CIKM 2025](https://arxiv.org/abs/2509.02837)
- 仓库评级：**D**。未找到清晰的官方代码仓库。

**解决的问题**

先在每个来源内融合多个 ranker，再用 z-score 统一不可比的来源分数，融合 labeled/unlabeled source 的结果。

**仍未解决**

- 解决的是 score calibration/rank fusion，不是知识对象关联；
- 不知道两个文档是否提到同一 assertion/event；
- 不恢复多跳或时间路径；
- 没有事件和历史状态；
- late fusion 无法基于已选证据判断还缺哪类信息。

**可交汇 GAP**

用 evidence-role-aware set selection 替换只按标准化分数融合；Text/Assertion/Event 通路不是同分布，应按互补角色而非强行同尺度。难度 **2/5**，交汇强度 **4/5**。

### 6.3 CARE / TruthfulRAG / Astute RAG — Conflict after retrieval

- CARE：[EMNLP 2025 官方仓库](https://github.com/eunseongc/CARE)，评级 **B+**，研究代码、训练和评测脚本完整；
- TruthfulRAG：[AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40489)，未找到清晰官方代码，评级 **D**；
- Astute RAG：[论文](https://arxiv.org/abs/2410.07176)，未找到稳定官方实现，评级 **D/C**。

**解决的问题**

这些方法主要解决检索上下文与 LLM 参数知识冲突，或在检索后识别不一致事实并调整生成。

**仍未解决**

- 冲突多在 post-retrieval/generation 层处理，而不是索引层长期维护；
- 常把冲突视为静态真假对立，难区分时间更新、限定条件变化和实体混淆；
- 不负责把多个 mention 对齐到同一 assertion；
- 不构造事件驱动的状态历史；
- 解决“答案采用哪边”但不保证“相关历史证据是否找齐”。

**可交汇 GAP**

把冲突识别前移到 statement/event organization：先依据实体、谓词、时间、条件把 mentions 对齐，再区分 concurrent conflict 与 temporal succession。这里只讨论结构区分，不讨论可信度。难度 **4/5**，交汇强度 **4/5**。

---

## 7. 仓库可用性总表

| 方法 | 子领域 | 官方仓库 | 评级 | 建议用途 |
|---|---|---|---:|---|
| LightRAG | 实体关系 GraphRAG | [GitHub](https://github.com/HKUDS/LightRAG) | **A** | 必选基础 GraphRAG baseline |
| HippoRAG | 关联图检索 | [GitHub](https://github.com/OSU-NLP-Group/HippoRAG) | **A** | 强 multi-hop graph baseline |
| GFM-RAG | 预训练图检索器 | [GitHub](https://github.com/RManLuo/gfm-rag) | **A** | 强 graph retriever；适合 STaRK/多跳 |
| SubgraphRAG | KG 子图检索 | [GitHub](https://github.com/Graph-COM/SubgraphRAG) | **A-/B+** | 已有 KG 上的高效多跳对照 |
| HyperGraphRAG | 超图组织 | [GitHub](https://github.com/LHRLAB/HyperGraphRAG) | **A-/B+** | n-ary relation/data organization 对照 |
| BYOKG-RAG | 多工具 KG 检索 | [GitHub](https://github.com/awslabs/graphrag-toolkit) | **A-/B+** | custom KG/工具化检索对照 |
| RAPTOR | 层次树组织 | [GitHub](https://github.com/parthsarthi03/raptor) | **B+** | 长文档、多粒度组织对照 |
| HiRAG | 层次 KG | [GitHub](https://github.com/hhy-huang/HiRAG) | **B+** | hierarchy + graph 对照 |
| OG-RAG | 本体超图 | [GitHub](https://github.com/microsoft/ograg2) | **B+** | 有 ontology 的领域实验 |
| E²RAG | 实体—事件双图 | [GitHub](https://github.com/zyzhang1130/EntityEventRAG) | **B+** | 最重要的实体—事件直接对照 |
| RAS | 查询时动态结构化 | [GitHub](https://github.com/pat-jj/RAS) | **B+** | persistent vs query-specific structure 对照 |
| CARE | 检索后冲突融合 | [GitHub](https://github.com/eunseongc/CARE) | **B+** | 只在冲突生成实验中使用 |
| LeanRAG | 语义聚合层次检索 | [GitHub](https://github.com/KnowledgeXLab/LeanRAG) | **B** | 冗余/层次图检索对照 |
| GNN-RAG | GNN 图检索 | [GitHub](https://github.com/cmavro/GNN-RAG) | **B** | 既有 KGQA 对照，不宜直接跨到文本建库 |
| KAIR/GraphAnchor | 动态 anchoring index | [GitHub](https://github.com/NEUIR/KAIR) | **B** | 迭代检索与充分性对照 |
| DyG-RAG | 动态事件图 | [GitHub](https://github.com/RingBDStack/DyG-RAG) | **B** | 事件时间线直接对照 |
| MultiRAG | 多源知识融合 | [GitHub](https://github.com/wuwenlong123/MultiRAG) | **B** | 多源关联融合直接对照 |
| SiReRAG | 相似＋相关双树 | [GitHub](https://github.com/SalesforceAIResearch/SiReRAG) | **B-** | 多视图组织对照；复现需手工适配 |
| EventRAG | 事件 KG | [GitHub](https://github.com/Ryaang/EventRAG) | **C+/B-** | 机制上必须比较；工程上需预留适配时间 |
| GraphFlow | 学习式多样图检索 | 未找到独立官方仓库 | **D/B-** | 暂不作为首批实现 baseline |
| HybGRAG | 文本＋关系混合检索 | 未找到独立官方仓库 | **D/C** | 可在 STaRK 上重建简化版本 |
| SAKI-RAG | 句间关联 | 未找到官方仓库 | **D** | 参考思想，不优先复现 |
| HF-RAG | 多源 rank fusion | 未找到官方仓库 | **D** | 自行实现其 fusion 很容易 |
| TruthfulRAG | 图式冲突定位 | 未找到官方仓库 | **D** | 不作为首批 baseline |
| IA-RAG | 区间代数检索 | [占位仓库](https://github.com/xiaoAugenstern/LogicalRAG_TemporalQA) | **D** | 借用 interval rules，不宣称官方复现 |

---

## 8. 从这些子领域共同暴露出的 GAP

### GAP 1：静态关系图缺少 assertion lifecycle

**来自哪些方法的共同缺口：** LightRAG、HippoRAG、GFM-RAG、HyperGraphRAG、OG-RAG、HiRAG。

它们擅长表达“哪些对象相关”，但没有统一表达：

```text
Assertion A 在 t1 成立
Event E 在 t2 发生
E 对 A 产生 UPDATE/TERMINATE/SUPERSEDE
Assertion A' 从 t2 起成立
各状态分别回到哪些 Mention
```

**与现有工作交汇：5/5**  
**实现难度：4/5**  
**创新潜力：5/5**

这是最稳的知识组织 GAP，但必须以 EventEffect、valid interval、state transition 和 Mention backtracking 形成完整机制，不能只给普通关系加 timestamp。

### GAP 2：事件图能找时间线，但不能恢复“事实如何变化”

**来自：** EventRAG、E²RAG、DyG-RAG、IA-RAG。

事件链回答 what happened；复杂研判常问 what fact changed because of it。需要区分 Event sequence 和 Assertion transition。

**交汇：5/5**  
**难度：4/5**  
**创新潜力：5/5**

最直接的算法对象是 `EventEffect → Assertion before/after` 路径。

### GAP 3：多视图通常只是统一候选池或 late fusion，不是真正协同

**来自：** SiReRAG、LightRAG hybrid、HF-RAG、HybGRAG、部分 MultiRAG。

现有方式通常是：每路 top-k → 分数归一化/RRF → top-k。它不知道已经选择的候选还缺什么，也没有让一个通路的结果约束另一个通路。

可研究：

```text
Text seed → Mention → Assertion/Event
Assertion seed → EventEffect → before/after state
Event seed → affected Assertion → supporting passages
```

然后按 evidence role completeness 选集合。

**交汇：5/5**  
**难度：3/5**  
**创新潜力：4/5**

这是最值得先做的检索创新，因为比重训 GNN/GraphFlow 成本低，且能直接做消融。

### GAP 4：persistent index 与 query-specific working graph 尚未统一

**来自：** RAS、KAIR/GraphAnchor 与静态 GraphRAG 之间的断层。

- 静态图：可跨查询复用，但可能庞大、陈旧、与问题无关；
- 查询图：针对性强，但每次重建、不能维持长期对象一致性。

可研究两层结构：

```text
Persistent TKG：规范对象、历史状态、所有 Mention
Query Working Graph：当前目标、已覆盖路径、缺失 evidence roles
```

**交汇：5/5**  
**难度：4/5**  
**创新潜力：5/5**

### GAP 5：跨文档融合没有区分重复、更新、替代和真实冲突

**来自：** MultiRAG、EventRAG、TruthfulRAG、CARE、普通 GraphRAG merge。

多文档谈到相似内容时，至少有五种关系：

- SAME_MENTION：同一事实的重复表述；
- REFINES：补充条件或细化；
- TEMPORAL_UPDATE：后来状态更新；
- SUPERSEDES：新断言替代旧断言；
- CONFLICTS：同一时间/条件下不一致。

现有方法常将这些关系压成 duplicate 或 conflict。

**交汇：5/5**  
**难度：4/5**  
**创新潜力：5/5**

该方向不需要讨论来源可信度；只需证明结构分类能改善检索的历史完整性和当前状态正确性。

### GAP 6：检索仍优化单项相关性，而不是最小充分证据子图

**来自：** 几乎所有 document/entity ranking 方法；GraphFlow 虽考虑多样性，但未优化 lifecycle role。

可定义固定预算下的集合目标：

```text
relevance
+ entity/assertion coverage
+ transition-path completeness
+ temporal coverage
+ raw-evidence coverage
- redundancy
- token cost
```

**交汇：5/5**  
**难度：3/5**  
**创新潜力：5/5**

这与用户强调的“数据关联融合和检索”最贴合。

### GAP 7：高阶关系、层次关系和时间关系各自为政

**来自：** HyperGraphRAG/OG-RAG、RAPTOR/LeanRAG、DyG-RAG/IA-RAG。

不同组织结构解决不同维度，但缺少 query-conditioned granularity selection：某题需要高阶角色关系，某题需要主题摘要，某题需要事件时间链。

可采用统一基本对象，不必把所有结构塞进一张巨图：

- canonical layer：Entity/Assertion/Event/Mention；
- derived views：hierarchy、hyperedge、timeline；
- query planner 选择视图并在 canonical IDs 上交汇。

**交汇：4/5**  
**难度：4/5**  
**创新潜力：4/5**

### GAP 8：跨文档实体、事件和断言对齐仍是隐藏瓶颈

**来自：** 所有从 raw text 建图的方法。

多数论文把 entity resolution/event coreference 当预处理，最终 QA 分数无法说明图的提升还是错误偶然抵消。需要 jointly evaluate：

- Entity resolution；
- Event coreference；
- Assertion deduplication；
- EventEffect attachment；
- Mention backlink completeness。

**交汇：5/5**  
**难度：5/5**  
**创新潜力：4/5**

研究价值高，但数据标注成本大，不建议作为第一个要补的模块。

### GAP 9：动态图论文很少评价长期增量一致性

**来自：** LightRAG incremental insert、DyG-RAG dynamic graph、RAS/KAIR evolving graph。

“dynamic/evolving”常有三种不同含义：

1. 图中有时间事件；
2. 当前查询期间图在增长；
3. 长期语料持续加入并修正索引。

真正困难的是第三种：新文档到来时，实体、断言、事件、向量和状态必须原子更新，并保留历史。

**交汇：5/5**  
**难度：5/5**  
**创新潜力：5/5**

这是强系统贡献，但实现和实验周期长，可作为后续扩展而不是首轮论文主线。

### GAP 10：结构抽取失败时缺少可恢复的 text fallback

**来自：** GFM-RAG、OG-RAG、HyperGraphRAG、EventRAG 等依赖抽取的方法。

图索引遗漏某断言或误合并实体时，纯图遍历无法召回；纯向量又无法利用结构。可以构造双向 repair loop：

```text
Text hit but no graph object → local extraction/linking proposal
Graph hit but no sufficient Mention → raw-text targeted search
```

**交汇：5/5**  
**难度：3/5**  
**创新潜力：4/5**

这是“三通路协同”很具体且容易实验化的版本。

---

## 9. GAP 按实现难度和价值排序

| 候选 GAP | 实现难度 | 交汇强度 | 创新潜力 | 推荐级别 |
|---|---:|---:|---:|---|
| Cross-path evidence completion，而非 RRF | **3** | 5 | 4 | **第一批实现** |
| 最小充分 evidence subgraph/portfolio selection | **3** | 5 | 5 | **第一批实现** |
| Text↔Graph 双向缺失修复 | **3** | 5 | 4 | **第一批实现** |
| EventEffect→Assertion before/after lifecycle path | **4** | 5 | 5 | **主知识组织贡献** |
| 区分 duplicate/refine/update/supersede/conflict | **4** | 5 | 5 | **主融合贡献** |
| Persistent TKG＋query working graph | **4** | 5 | 5 | **强扩展方向** |
| Time/state-sensitive hierarchy | **4** | 4 | 4 | 次级方向 |
| Canonical layer＋hierarchy/hypergraph/timeline 派生视图 | **4** | 4 | 4 | 次级方向，易做大 |
| Heterogeneous TKG GNN/GFM retriever | **5** | 4 | 4 | 资源充足再做 |
| Flow-based lifecycle path policy | **5** | 4 | 5 | 高风险高成本 |
| 实体/事件/断言联合对齐 benchmark | **5** | 5 | 4 | 数据集论文方向 |
| 长期增量一致性与历史回放 | **5** | 5 | 5 | 后续强系统论文 |

---

## 10. 最值得与你们“碰一碰”的三个方向

### 方向一：统一对象不是重点，统一“状态变化路径”才是重点

建议正式研究对象：

```text
AssertionMention
      ↓ grounds
Temporal Assertion State
      ↑ changed_by
EventEffect ← Event
      ↓ grounded_by
EventMention
```

对比对象：LightRAG、HyperGraphRAG、E²RAG、DyG-RAG。

要证明的是：

- 普通 KG 找到实体关系，但压缩历史状态；
- 事件图找到事件顺序，但未绑定具体关系状态变化；
- 统一 lifecycle path 同时恢复关系、变化和原文。

### 方向二：三通路从 late fusion 升级为 cross-path completion

建议三类入口：

- Text/Mention discovery；
- Assertion/Entity structure；
- Event/Transition evolution。

但最终不是三份排名，而是跨路径补全一个 evidence subgraph。直接比较：

```text
Separate stores + RRF
vs.
Unified graph + ordinary expansion
vs.
Unified graph + cross-path completion
vs.
Full model + budgeted evidence-set selection
```

这是实现难度最低、最容易产生可归因结果的主检索创新。

### 方向三：跨文档知识融合从 merge 变成 typed evolution alignment

不要只做 entity/event merge。对两个 mentions 的关系判断为：

```text
duplicate / refine / temporal_update / supersede / conflict / unrelated
```

然后：

- 所有 Mention 保留；
- canonical Assertion 可以共享；
- 状态按 valid interval 分段；
- EventEffect 决定状态边界；
- 检索按 query time 和 evidence role 返回对应历史。

这是最符合“数据关联融合”定位的方向，也不涉及可信度。

---

## 11. 建议优先下载和试跑的开源方法

### 第一批：最稳、最容易形成公平实验

1. [LightRAG](https://github.com/HKUDS/LightRAG)：静态实体关系 GraphRAG；
2. [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)：关联扩散和 multi-hop；
3. [GFM-RAG](https://github.com/RManLuo/gfm-rag)：强学习式图检索器；
4. [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG)：高阶数据组织；
5. [E²RAG](https://github.com/zyzhang1130/EntityEventRAG)：实体—事件双图；
6. [DyG-RAG](https://github.com/RingBDStack/DyG-RAG)：事件时间线；
7. 自建 `Separate-3Path + RRF`：最关键的内部 baseline。

### 第二批：根据论文主线选择

- 主打多视图组织：SiReRAG；
- 主打本体/专业领域：OG-RAG；
- 主打查询时动态结构：RAS、KAIR；
- 主打多源融合：MultiRAG；
- 主打 KGQA 检索效率：SubgraphRAG/GNN-RAG；
- 主打工具化 custom KG：BYOKG-RAG。

### 暂不优先

- IA-RAG：官方仓库当前不可用；
- GraphFlow：缺少清晰独立官方仓库且训练成本高；
- SAKI-RAG、HF-RAG、TruthfulRAG：没有清晰官方实现；
- HybGRAG：可用 STaRK 重建，但不是低成本复现。

---

## 12. 最终判断

从整个 RAG 数据组织领域看，最拥挤的是：

- vector＋graph；
- similarity＋relatedness；
- entity＋event；
- text＋relation；
- 多 ranker late fusion；
- query-aware iterative retrieval。

因此，不能再把“多一种数据结构”或“多一条检索通路”作为核心 GAP。

当前仍有条件形成方法贡献的交叉空白是：

> **多源动态文本中的规范化断言、改变断言状态的事件及其原文 Mention 尚未被统一为可检索的生命周期路径；现有多视图系统又多依赖独立召回和 late fusion，缺少面向证据角色与路径完整度的跨视图补全和集合选择。**

最可行的技术路线不是立刻训练新的图基础模型，而是：

1. 完成 Assertion lifecycle 和 typed cross-document fusion；
2. 将三通路从 RRF 改成 cross-path evidence completion；
3. 在固定预算下选择最小充分 evidence subgraph；
4. 用 LightRAG、E²RAG、DyG-RAG、HyperGraphRAG、GFM-RAG 和 Separate-3Path 公平验证；
5. 后续再扩展 persistent TKG＋query working graph 和长期增量一致性。

