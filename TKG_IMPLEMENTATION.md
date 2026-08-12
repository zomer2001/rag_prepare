# 本体与声明级时序知识图谱 v2

## 目标

通过语义组织与事件组织的统一身份映射，将多来源文本融合为声明级时序知识图谱；事件负责
解释事实的产生、变更和终止，同时保留 Mention、来源、立场、冲突和置信信息。

## 当前可执行链路

```text
文档解析
  ├─ LightRAG 受控实体/关系抽取 ─┐
  └─ 关键事件与 fact_effects 抽取 ├─ 统一 Entity ID
                                  ├─ 关系 → Assertion + AssertionMention
                                  ├─ 事件角色 → EventEntityLink
                                  └─ 事件作用 → EventEffect → Assertion 生命周期
                                                        │
                    OWL/SHACL 静态约束 ─────────────────┤
                                                        ▼
              本地 TKG 原子存储 → 三通路中的语义图谱召回 → 文档级 RRF
```

LightRAG 建完索引后，系统从它的 NetworkX 存储读取该文档已经抽取的实体和关系，不再次调用
LLM。事件抽取新增 `fact_effects`，只有原文明示完整事实三元组和作用类型时才填写。

## 融合规则

1. 规范名/别名完全一致且本体类型一致：合并到既有 Entity。
2. 无确定候选：创建 Entity；存在多个候选：隔离创建并记录 `AMBIGUOUS`，不强合并。
3. 同一主体、谓词、客体和有效期：复用 Assertion，仅新增来源 Mention。
4. `DENIES`：增加 denied Mention，肯定与否定并存时 Assertion 为 disputed 并生成 ConflictCase。
5. `TERMINATES`：关闭 `valid_to`，状态变为 inactive；`SUPERSEDES` 标为 superseded。
6. `CREATES/CONFIRMS`：Assertion 进入 active；所有决定、证据和事件作用都保留 ID 可追踪。
7. 文档重建先移除该文档的 Mention/Effect/Decision，再重新融合；其他来源仍支持的对象不删除。

## 本体边界

- OWL：类别、层次、属性与可复用语义关系。
- SHACL：入库最低字段、取值范围和基数约束。
- `ontology/domain/research-technology.ttl`：当前科研技术领域静态术语。
- 动态本体不在本版本自动改 Schema。新增候选术语应经过统计、审查、版本迁移和回归评测，
  再发布新 ontology version；不能让 LLM 在索引时直接改本体。

## 存储选择

本地 `data/tkg/tkg_store.json` 是当前事务源，适合快速验证模式、规则和标注方案。现阶段
不双写 Neo4j；当规模和多跳时态查询负载达到阈值后，再把 Entity、Assertion 与
EventEffect 投影到独立 Neo4j 数据库。投影应是可重建的读模型，而不是新的事实源。

## 查询

- `POST /api/v1/tkg/query`：按 Entity ID/名称、predicate、时间点和状态查询。
- 统一检索中，Query Planner 识别到实体后，AssertionMention 作为语义图谱通路的增强候选，
  与原 LightRAG 图谱候选去重；向量、语义图谱、事件仍按三通路做文档级 RRF。

## 尚未自动化的部分

- Embedding 候选生成和 LLM 歧义裁决接口尚未开启；当前只自动执行高精度确定性合并，
  其余案例保留 ResolutionDecision。
- 跨客体的互斥约束依赖更细的领域本体（如一个职位同一时刻只能由一人担任）。
- Neo4j 投影尚未实施；需先定义启用阈值、独立数据库和增量同步/重建策略。
- 动态本体演化需要术语候选集、专家审批、迁移脚本和旧版/新版检索评测，不能仅靠 LLM。
