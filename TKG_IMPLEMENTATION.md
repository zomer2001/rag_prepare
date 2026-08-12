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
        JSON事务镜像 → Neo4j实例图 → 三通路中的语义图谱召回 → 文档级 RRF
```

LightRAG 建完索引后，系统从它的 NetworkX 存储读取该文档已经抽取的实体和关系，不再次调用
LLM。事件抽取新增 `fact_effects`，只有原文明示完整事实三元组和作用类型时才填写。

## 融合规则

1. LLM类型先由本体映射表标准化；未注册类型进入治理队列，不直接成为正式类型。
2. 同类型下按规范名、别名和组织后缀生成确定性候选。
3. 确定性规则不足时，复用现有 Embedding 对“名称＋类型＋描述”筛选候选。
4. 高精度、明显领先的候选自动合并；灰区由受约束LLM裁决。
5. LLM只能输出 MATCH_EXISTING、CREATE_NEW、AMBIGUOUS或REJECT；越权或格式错误降级人工治理。
6. 不确定案例创建隔离 Entity 和 ReviewCase，文档索引可继续，但不会静默硬合并。
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

Neo4j保存完整实例图：Entity、Assertion、Event、Entity/Assertion/Event Mention、
EventEffect、Conflict、ResolutionDecision、ReviewCase和MergeRecord。Assertion建成节点，
以便挂接时间、来源、事件作用和冲突。JSON保留为融合事务镜像与恢复材料，不承担在线查询。

实体合并不删除源 Entity，而是标记 `status=merged` 与 `merged_into_id`，并保存该次移动的
Mention、Assertion端点和Event链接。撤销操作只恢复该次移动的引用；存在后续依赖合并时拒绝撤销。

## 查询

- `POST /api/v1/tkg/query`：直接从Neo4j按Entity ID/名称、predicate、时间点和状态查询。
- 统一检索中，Query Planner 识别到实体后，AssertionMention 作为语义图谱通路的增强候选，
  与原 LightRAG 图谱候选去重；向量、语义图谱、事件仍按三通路做文档级 RRF。

## 尚未自动化的部分

- 跨客体的互斥约束依赖更细的领域本体（如一个职位同一时刻只能由一人担任）。
- 动态本体演化需要术语候选集、专家审批、迁移脚本和旧版/新版检索评测，不能仅靠 LLM。

## 迁移与重算

`uv run python scripts/migrate_tkg_to_neo4j.py` 只把现有JSON快照迁移到Neo4j，不调用LLM。
若要让旧文档经过新的本体映射、Embedding和LLM实体解析器，必须重新索引文档；数据搬家与
融合重算是两个独立操作。
