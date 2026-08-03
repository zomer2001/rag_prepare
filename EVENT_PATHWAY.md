# 事件通路 Demo 架构

## 1. 设计结论

事件通路不是与向量检索、图谱检索平行的“第三种搜索按钮”，而是一个贯穿索引与查询
的关系断言层：

```text
索引：文档 → 解析文本 → LightRAG 向量/实体关系索引
                    └→ 事件抽取 → 归一合并 → 证据/冲突/时间关系

查询：问题 → LightRAG 语义与图谱召回 → 带引用的原文证据
           └→ 时间约束解析 → 事件筛选/排序/冲突提示 ─┘
```

实体图回答“涉及什么对象和关系”；事件层回答“该关系在何时、什么状态、由哪些来源支持”。
事件仍是来源所作的声明，不会被系统直接提升为绝对事实。

## 2. Demo 的技术取舍

本分支实现最小完整闭环，不增加 PostgreSQL、Neo4j 或第二套向量集合：

| 能力 | Demo 实现 | 后续生产化 |
|---|---|---|
| 原文语义召回 | 现有 LightRAG + Qdrant | 保持 |
| 实体关系图 | 现有 LightRAG + NetworkX | 可迁移图数据库 |
| 事件主表 | `data/events/event_store.json` | PostgreSQL |
| 事件提及和证据 | JSON 中的 `mentions` | PostgreSQL，证据精确到页/段 |
| 事件时间图 | JSON 中的 `relations` | Neo4j/AGE 等图数据库 |
| 事件检索 | 结构化过滤 + 轻量词项相关度 | 独立事件向量 + 结构化过滤 |
| 多源验证 | 文档数、立场冲突、置信度 | 来源等级、引用传播和校准模型 |

原因是当前目标是验证事件机制是否有用。把四套数据库同时引入，只会先验证运维复杂度。
事件模型和服务边界已独立，后续替换存储不会改变文档和查询 API 的语义。

## 3. 索引阶段

文档完成解析及 LightRAG 索引后，事件抽取器按段调用 DeepSeek，输出：

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

重建同一文档时，会先移除该文档旧提及再写入新结果，因此不会持续累积重复事件。删除文档
也会同步删除其事件提及及无来源事件。

事件抽取失败时，已经成功的 LightRAG 索引仍保持 `ready`；文档会显示
`event_index_status=failed` 和错误原因，可单独重建事件。这样网络瞬断不会让全部索引回滚。

## 4. 查询阶段

`POST /api/v1/query` 默认启用事件通路。系统会：

1. 从问题中解析明确年份、年份范围或“近 N 年”；
2. 按实体、事件类型、时间、置信度和争议状态筛选事件；
3. 按词项相关度、置信度和时间完整度排序；
4. 取事件链和冲突信息，作为 `user_prompt` 约束注入 LightRAG；
5. LightRAG 仍从原文块生成答案和引用；事件只负责时间、状态和校验约束；
6. API 同时返回 `events`、`event_relations`、`event_conflicts`，便于界面解释。

这避免了“事件摘要自己证明自己”：最终答案仍必须引用原文，争议事件需要明确标注。

## 5. 数据结构

`data/events/event_store.json` 包含四组记录：

```text
events      规范化事件主记录及综合状态
mentions    每个来源对事件的提及、立场、置信度和证据
relations   事件间的时间/相关关系
conflicts   来源立场冲突
```

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
- 事件没有独立向量集合，大规模数据下应迁移为向量候选召回 + 规则/模型合并；
- 当前只检测肯定/否定立场冲突，参数冲突、版本冲突和状态机缺口属于下一阶段；
- 当前用生成前约束和结构化结果辅助核验，尚未实现对生成答案逐条反抽声明的二次 LLM 校验；
- JSON 注册表仅适合单用户、单 API 进程的 Demo。

建议先用 10 至 30 篇 Anduril 官网、合同和新闻材料验证抽取质量、重复率、时间线价值，再决定
是否投资生产级事件向量、版本表和图数据库。
