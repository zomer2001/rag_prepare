# 交给其他 Agent 的论文可行性审稿提示词

请把下面整段提示词连同 `ELC-RAG论文核心论证与实验规划报告.md`、相关代码和实验结果一起交给另一个 Agent。

```text
你是一名对 RAG、Knowledge Graph RAG、Event Graph、Temporal RAG、Ontology-RAG 和 NLP 顶会审稿非常熟悉的苛刻审稿人。请审计我提供的《ELC-RAG论文核心论证与实验规划报告》，判断这项工作能否支撑 ACL/EMNLP/NAACL/EACL 长文级论文。

请不要顺着作者的叙述下结论，也不要因为系统工程量大就认为有学术创新。你必须检索并核对截至 2026-09-04 的真实相关工作，优先使用论文原文、ACL Anthology、会议官网和官方代码。重点比较 EventRAG、E²RAG、TimeR4、HoH、KG²RAG、OG-RAG、HippoRAG、GraphRAG、IRCoT、Adaptive-RAG、query decomposition，以及你找到的更接近工作。不要使用不存在的论文，也不要把预印本误写成正式会议论文。

待审方法的核心设想是：
1. 使用 Event-Effect Ontology 区分 Entity、Assertion、Event、EventEffect 和来源 Mention；
2. 用 creates/updates/terminates/supersedes/denies/confirms 表示事件如何影响事实；
3. EvoLink 跨文本、事实关系和事件视图，检索“旧状态—变化事件—新状态—来源”完整证据；
4. 拟开发 EvoNeed Planner，在第一次检索前按 state_before、change_event、effective_time、state_after、stance_status、supporting_source 规划子查询、路线和预算，并一次并行检索；
5. 系统不负责裁决绝对真相，只负责返回连接、完整且可追溯的证据。

请完成以下审计：

A. 问题是否真实
- 论文描述的业务场景是否足够普遍和重要？
- “相关但不完整”“新旧混用”“变化原因缺失”是否已经被现有数据和论文证明？
- 这是一个独立研究问题，还是普通 temporal reranking、event graph 或 graph expansion 就能解决？

B. Novelty 审计
- 分别判断 Event-Effect Ontology、EvoLink、EvoNeed Planner 的新颖性。
- 找出最可能构成 novelty collision 的 5–10 篇工作，逐项列出相同点和真正差异。
- 特别判断“Event 明确 creates/terminates 某个 Assertion 并保留来源 Mention”是否已有等价建模。
- 判断使用“ontology”一词是否成立；如果只是 typed property graph，请明确指出。
- 判断当前贡献是否只是把已有模块拼接在一起。

C. 技术正确性
- EventEffect 从文本中是否可可靠抽取？哪些类型容易混淆？
- 将 active/inactive/disputed/superseded 当作状态是否会错误地把来源说法当成客观真相？
- 跨视图扩展会不会提高 Recall 但引入大量噪声？
- 在不裁决真相的前提下，生命周期和状态还能提供哪些可验证收益？
- 一轮 Planner 是否真的比固定三路检索或普通 query decomposition 更合理？

D. 实验充分性
- 检查现有 q100 指标能支持什么、不能支持什么。
- 审计 proposed baselines、datasets、ablations、metrics 和统计检验是否足够。
- 判断 MultiHop-RAG、ChronoQA、HoH 是否真的能验证 EventEffect；指出数据标注不匹配之处。
- 给出最小人工标注集应包含哪些字段、多少样本以及一致性要求。
- 设计至少两个能直接证伪核心主张的关键实验，而不是只设计容易成功的实验。
- 检查公平性：语料、chunk、embedding、抽取模型、Top-K、context token、在线 LLM 调用和原生/受控配置。

E. 成本与可扩展性
- 评价索引阶段多次 LLM 抽取、图存储、跨视图扩展和 Planner 的成本是否可接受。
- 判断与 Dense/Hybrid/EventRAG/E²RAG/HippoRAG 相比，质量—成本收益是否可能成立。
- 指出最可能导致方法无法扩展的复杂度和工程瓶颈。

F. 最终裁决
- 按 Strong Reject / Reject / Borderline / Accept / Strong Accept 给出当前版本评分和完成计划后的潜在评分。
- 分别给出 originality、soundness、significance、clarity、reproducibility 的 1–5 分。
- 列出 3 个致命风险、5 个必须完成的实验和 3 个可删除的非核心部分。
- 如果当前故事站不住，请提出一个更窄但可发表的论文问题；如果能站住，请给出最强且不夸大的 problem statement、GAP、3 项 contribution 和标题。

输出要求：
1. 先给一页式结论，不要先复述报告；
2. 使用“作者主张—已有工作—证据—裁决”的对照表；
3. 每个文献判断附可访问链接；
4. 明确区分“代码已经实现”“报告计划实现”“实验已经证明”；
5. 不接受“据我所知首次”这种无证据表述；
6. 语气可以尖锐，但结论必须有证据且可操作。
```
