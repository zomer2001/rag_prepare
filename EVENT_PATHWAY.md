# 事件通路 Demo 架构

## 1. 设计结论

事件通路既是贯穿索引与查询的关系断言层，也是统一检索器中的独立第三路证据召回：

```text
索引：文档 → 解析文本 → LightRAG 向量/实体关系索引
                    └→ 事件抽取 → 归一合并 → 证据/冲突/时间关系

查询：问题 → 统一 Query Planner
           ├→ naive 文本向量证据
           ├→ hybrid 实体关系图证据
           └→ 事件向量 → 结构过滤/一跳扩展 → 事件原文证据
                         ↓ 三路文档 RRF
                    最终可回溯证据包
```

实体图回答“涉及什么对象和关系”；事件层回答“该关系在何时、什么状态、由哪些来源支持”。
事件仍是来源所作的声明，不会被系统直接提升为绝对事实。同一事件有多个来源时，这些
证据在融合中共享事件级 rank，再在最终文档结果中去重，避免来源枚举顺序改变相关度。

## 2. Demo 的技术取舍

本分支实现最小完整闭环，不增加 PostgreSQL、Neo4j 或第二套向量集合：

| 能力 | Demo 实现 | 后续生产化 |
|---|---|---|
| 原文语义召回 | 现有 LightRAG + Qdrant | 保持 |
| 实体关系图 | 现有 LightRAG + NetworkX | 可迁移图数据库 |
| 事件主表 | `data/events/event_store.json` | PostgreSQL |
| 事件提及和证据 | JSON 中的 `mentions` | PostgreSQL，证据精确到页/段 |
| 事件时间图 | JSON 中的 `relations` | Neo4j/AGE 等图数据库 |
| 事件检索 | 持久化事件向量 + 结构化过滤/排序 | Qdrant 事件集合或分布式向量服务 |
| 多源验证 | 文档数、立场冲突、置信度 | 来源等级、引用传播和校准模型 |

原因是当前目标是验证事件机制是否有用。把四套数据库同时引入，只会先验证运维复杂度。
事件模型和服务边界已独立，后续替换存储不会改变文档和查询 API 的语义。

## 3. 索引阶段

文档完成解析及 LightRAG 索引后，事件抽取器按段调用 DeepSeek。它不穷举所有动作，只选择
会改变核心对象状态、能支持时间/变化/冲突检索的关键事件；默认每段最多 5 个、每篇文档
最多 12 个，数量是上限而非目标。普通提及、背景历史、日常活动和泛泛的“讨论/指出”会
被省略。事件摘要与证据不设置字符压缩目标。输出字段包括：

- `event_type`、`summary`、`action`；
- `actors`、`objects`、`location`；
- `start_time`、`end_time`、`time_precision`；
- `result`、`stance`、`confidence`；
- 必须来自原文的 `evidence`。

事件注册表随后执行：

1. 对事件类型、主体和对象做规范化；
2. 按类型、共同实体和兼容时间窗口合并跨文档事件；
3. 保留每一份文档的 `EventMention`，不覆盖证据；
4. 根据独立文档数计算 `single_source` / `multi_source`；
5. 同一事件同时出现肯定与否定提及时标为 `disputed` 并生成冲突；
6. 对共享实体的事件生成 `precedes`、`same_time` 或 `related` 关系。
7. 为每条提及保存来源段ID、段字符区间和精确证据字符区间；事件规范文本另外生成向量并
   缓存在 `data/events/event_vectors.npz`。

重建同一文档时，会先移除该文档旧提及再写入新结果，因此不会持续累积重复事件。删除文档
也会同步删除其事件提及及无来源事件。

事件抽取失败时，已经成功的 LightRAG 索引仍保持 `ready`；文档会显示
`event_index_status=failed` 和错误原因，可单独重建事件。这样网络瞬断不会让全部索引回滚。

## 4. 查询阶段

`POST /api/v1/query` 默认启用事件通路。系统会：

1. Query Planner一次生成高低层图谱关键词、实体、时间、事件类型、状态意图和是否扩展
   事件链；计划按版本、日期和问题持久化，`light_entity`与`ours`复用同一结果；
2. 同一问题向量召回事件候选，再按实体、事件类型、时间、置信度和争议状态过滤；
3. 综合词项、实体角色、动作类型、时间意图、状态/冲突、置信度和多来源信号，输出带分项
   分数的事件级 `rankings`；事件时间线展示可显式使用 `sort_by=time` 改为时间升序，但
   评测融合始终使用默认的相关性排名；
4. 对变化、先后、纠正等问题允许从向量命中的种子事件做一跳事件链扩展；事件链只陈述
   时间先后或共享实体，不把相邻事件臆断为因果；
5. 事件证据与文本向量、语义图谱证据统一按文档RRF，保留每路的来源片段；
6. 在线回答与检索评测消费同一批最终Top-K证据，不再另走`user_prompt`旁路；
7. API 同时返回 `events`、`event_relations`、`event_conflicts`，便于界面解释。

这避免了“事件摘要自己证明自己”：最终答案仍必须引用原文，争议事件需要明确标注。

## 5. 数据结构

`data/events/event_store.json` 包含四组记录：

```text
events      规范化事件主记录及综合状态
mentions    每个来源对事件的提及、立场、置信度和证据
relations   事件间的时间/相关关系
conflicts   来源立场冲突
```

配套的 `data/events/event_vectors.npz` 保存事件规范文本的Embedding、`event_id`和索引指纹；
它不复制原文，失效时可以从事件主表重新生成。

每条事件同时保存事件时间（`start_time` / `end_time`）和系统记录时间
（`created_at` / `updated_at`）。这为以后增加 `event_version` 和“截至某日系统知道什么”查询
保留了结构空间。

## 6. API 与运行

启动方式不变。更新代码后需要重启 FastAPI 和 Streamlit 进程：

```bash
docker-compose up -d qdrant
uv run uvicorn rag_graph_demo.api.app:app --reload --host 127.0.0.1 --port 8000
uv run streamlit run src/rag_graph_demo/ui/streamlit_app.py --server.port 8501
```

已有文档不会自动补抽事件，可在 UI 的“文档管理”点击“重建事件”，或调用：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/文档ID/events/rebuild
```

为所有已有文档补建事件（默认跳过已经成功的文档，可中断后继续）：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

查询事件：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/events/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"近三年 Anduril 产品有哪些关键变化？","entity":"Anduril"}'
```

普通问答默认使用事件；可传 `"use_events": false` 做 A/B 对比。

## 7. Demo 边界

- 目前的跨文档合并是确定性规则，无法可靠区分同月、同主体、同类型的极相似事件；
- “先于”来自时间排序，“相关”来自共享实体；系统不会把时间先后自动声称为因果；
- 来源独立性暂以不同文档近似，不等同于不同出版机构；
- 事件向量目前使用单机 `.npz` 持久化矩阵，适合实验规模；大规模数据应迁移到Qdrant事件集合；
- 当前只检测肯定/否定立场冲突，参数冲突、版本冲突和状态机缺口属于下一阶段；
- 当前用生成前约束和结构化结果辅助核验，尚未实现对生成答案逐条反抽声明的二次 LLM 校验；
- JSON 注册表仅适合单用户、单 API 进程的 Demo。

建议先用 10 至 30 篇 Anduril 官网、合同和新闻材料验证抽取质量、重复率、时间线价值，再决定
是否投资生产级事件向量、版本表和图数据库。
