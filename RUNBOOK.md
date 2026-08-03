# `rag_prepare` 采集、三通路索引、查询与恢复 Runbook

> 项目路径：`/Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare`  
> 使用方式：按照对应章节逐行复制命令。  
> 数据流程：自动抓取 → Clean 数据 → 解析 → 向量/语义/事件索引 → 带证据查询。
> 当前分支：`feature/event-pipeline`。

---

## 1. 首次安装与配置（只需执行一次）

进入项目：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

安装 Python 依赖：

```bash
uv sync
```

如果项目中还没有 `.env`，复制示例配置：

```bash
test -f .env || cp .env.example .env
```

编辑 `.env`：

```bash
nano .env
```

确认至少包含以下配置：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-v4-flash
DEEPSEEK_EXTRACTION_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING_ENABLED=false

LLM_REQUEST_TIMEOUT_SECONDS=600
LLM_MAX_RETRIES=3
LLM_MAX_CONCURRENCY=1
GRAPH_INDEX_CONCURRENCY=1

EVENT_EXTRACTION_ENABLED=true
EVENT_EXTRACTION_CHUNK_CHARS=12000
EVENT_EXTRACTION_MAX_EVENTS=40
```

如果需要抓取 PatentsView 专利，再添加：

```dotenv
PATENTSVIEW_API_KEY=你的PatentsView密钥
```

在 `nano` 中保存并退出：

```text
Ctrl+O
回车
Ctrl+X
```

如果尚未安装 Colima、Docker CLI 和 Docker Compose：

```bash
brew install colima docker docker-compose
```

---

## 2. 日常启动基础环境（终端一）

启动 Colima：

```bash
colima start
```

进入项目：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

启动 Qdrant：

```bash
docker-compose up -d qdrant
```

检查 Qdrant 容器：

```bash
docker-compose ps
```

检查 Qdrant HTTP 服务：

```bash
curl http://127.0.0.1:6333/healthz
```

运行完整诊断：

```bash
uv run python scripts/doctor.py
```

只有看到以下结果时，才继续索引：

```json
"ready": true,
"can_index_and_query": true
```

---

## 3. 自动抓取、解析和清洗 Collection 数据

### 3.1 首次小规模试运行

每个 P0 来源最多处理 20 条：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 20 run --group p0
```

P0 当前包括：

- Anduril 官网；
- USAspending 合同；
- OpenAlex 论文；
- PatentsView 专利（需要 `PATENTSVIEW_API_KEY`）。

### 3.2 正式增量抓取

每个来源最多处理 100 条：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group p0
```

如果希望使用来源自己的分页/增量限制，不设置本轮条数：

```bash
uv run python -m rag_graph_demo.collection.cli --live run --group p0
```

### 3.3 只运行单个来源

Anduril 官网：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --source anduril_official
```

USAspending 合同：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --source usaspending
```

OpenAlex 论文：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --source openalex
```

PatentsView 专利：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --source patentsview
```

### 3.4 抓取 P1 新闻和智库

抓取 Anduril 相关新闻：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group news
```

抓取 Anduril 相关智库出版物：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group think_tank
```

一次运行全部 P1 来源：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group p1
```

新闻来源包括 Defense News、The War Zone 和 Defense One；智库来源包括 RUSI
和 CSET。RSS 只监控当前 Feed 中仍保留的近期内容，某来源本轮返回 0 条不代表运行
失败，而是当前 Feed 没有匹配的新条目。

### 3.5 不使用 API 的历史回填

从媒体公开 sitemap 直接回填 URL 明确包含 Anduril 的历史新闻：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group news_archive
```

从智库公开 sitemap 直接回填 URL 明确包含 Anduril 的历史内容：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group think_tank_archive
```

一次运行全部历史回填来源：

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group archive
```

`archive` 不需要 API Key，按站点 robots 和 sitemap 直接抓取。该组适合首次回填或
偶尔补跑，不需要每天运行。为了控制误报，它只自动接收 URL 路径明确含 `anduril`
的页面；产品同名或只有正文提及的材料需要后续人工扩展。

### 3.6 查看抓取结果

列出清洗后的 Markdown：

```bash
find data/collection/clean -name '*.md' | sort
```

统计清洗后的 Markdown 数量：

```bash
find data/collection/clean -name '*.md' | wc -l
```

查看最近四个运行报告：

```bash
find data/collection/runs -name manifest.json -print | sort | tail -n 4
```

查看待人工复核数量：

```bash
find data/collection/review_queue -type f | wc -l
```

查看永久/临时失败记录：

```bash
find data/collection/dead_letter -type f 2>/dev/null | wc -l
```

---

## 4. 将 Collection 数据建立三通路索引

确认基础诊断通过：

```bash
uv run python scripts/doctor.py
```

导入全部清洗后的 Markdown：

```bash
uv run python scripts/import_directory.py /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare/data/collection/clean
```

导入过程会自动完成：

1. 根据内容哈希跳过完全重复文件；
2. 将原件复制到 `data/raw/<document_id>/`；
3. 将解析结果写入 `data/parsed/<document_id>/`；
4. 将文档登记到 `data/registry/documents.json`；
5. 生成文本 Chunk Embedding 并写入 Qdrant；
6. 使用 DeepSeek 抽取实体和关系，写入 LightRAG/NetworkX；
7. 使用 DeepSeek 抽取时间、主体、动作、对象、结果和原文证据；
8. 归并跨文档事件，计算单源/多源/争议状态；
9. 写入 `data/events/event_store.json` 并生成相邻事件链。

查看知识库文档状态：

```bash
uv run python -c 'import json; from collections import Counter; d=json.load(open("data/registry/documents.json")); print("文档总数:",len(d)); print(Counter(v["status"] for v in d.values()))'
```

同时检查文档主状态与事件子状态：

```bash
uv run python -c 'import json; from collections import Counter; d=json.load(open("data/registry/documents.json")); print("文档:",Counter(v["status"] for v in d.values())); print("事件:",Counter(v.get("event_index_status","legacy_missing") for v in d.values()))'
```

理想结果是两组状态均全部为 `ready`。`event_count=0` 可以是成功结果，表示该文档没有
明确事件，不代表失败。

列出没有成功的文档：

```bash
uv run python -c 'import json; d=json.load(open("data/registry/documents.json")); [print(k,v["status"],v["filename"],v.get("error")) for k,v in d.items() if v["status"]!="ready"]'
```

列出事件抽取失败的文档：

```bash
uv run python -c 'import json; d=json.load(open("data/registry/documents.json")); [print(k,v.get("event_index_status"),v["filename"],v.get("event_error")) for k,v in d.items() if v.get("event_index_status")!="ready"]'
```

### 4.1 只给已有文档补建事件

升级事件分支后，旧文档通常已经有 LightRAG 索引，不应重复生成向量和实体图谱。运行：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

脚本特点：

- 默认跳过 `event_index_status=ready` 的文档；
- 中断后可直接重跑；
- 每篇文档以替换方式写入，不累积重复提及；
- 结束时重算事件状态、冲突和相邻时间关系；
- 任意文档失败时返回非零退出码。

小批量验证：

```bash
uv run python scripts/rebuild_events.py --workers 1 --limit 10
```

强制重建指定文档：

```bash
uv run python scripts/rebuild_events.py --workers 1 --force --document-id <DOCUMENT_ID>
```

---

## 5. 启动后端 API（终端二）

打开一个新终端，进入项目：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

启动 FastAPI：

```bash
uv run uvicorn rag_graph_demo.api.app:app --reload --host 127.0.0.1 --port 8000
```

保持该终端运行。

API 地址：

- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- 完整诊断：<http://127.0.0.1:8000/api/v1/diagnostics>
- 事件列表：<http://127.0.0.1:8000/api/v1/events>

命令行检查 API：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

列出已经登记的文档：

```bash
curl http://127.0.0.1:8000/api/v1/documents
```

---

## 6. 启动 Web 界面（终端三）

打开第三个终端，进入项目：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

启动 Streamlit：

```bash
uv run streamlit run src/rag_graph_demo/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

保持该终端运行，然后打开：

<http://127.0.0.1:8501>

界面中可以：

- 查看系统状态；
- 查看已经导入的文档；
- 手动上传 Markdown、PDF、DOCX；
- 查看解析结果；
- 重新索引失败文档；
- 单独重建事件索引；
- 使用 `naive`、`local`、`global`、`hybrid`、`mix` 模式查询；
- 开关事件通路并查看 `[E#]` 原文证据；
- 按实体和时间查看事件时间线；
- 查看知识图谱；
- 对比不同检索模式。

---

## 7. 最常用的完整日常流程

### 终端一：基础服务、抓取和导入

```bash
colima start
```

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

```bash
docker-compose up -d qdrant
```

```bash
uv run python scripts/doctor.py
```

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group p0
```

```bash
uv run python -m rag_graph_demo.collection.cli --live --limit 100 run --group p1
```

```bash
uv run python scripts/import_directory.py /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare/data/collection/clean
```

如果文档是从旧分支导入、事件状态尚未就绪，再运行：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

### 终端二：API

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

```bash
uv run uvicorn rag_graph_demo.api.app:app --reload --host 127.0.0.1 --port 8000
```

### 终端三：Web UI

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

```bash
uv run streamlit run src/rag_graph_demo/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

打开：<http://127.0.0.1:8501>

---

## 8. 手动导入其他文件夹的数据

系统支持：

- `.md`
- `.markdown`
- `.pdf`
- `.docx`

例如，外部资料位于：

```text
/Users/zomeryang/Documents/anduril_dataset/
```

递归解析并索引：

```bash
uv run python scripts/import_directory.py /Users/zomeryang/Documents/anduril_dataset
```

不要手动复制文件到 `data/raw`，否则不会自动生成 UUID、注册表记录、解析结果和索引。

---

## 9. 失败恢复

### 9.1 LightRAG 文档失败

先运行诊断：

```bash
uv run python scripts/doctor.py
```

列出失败文档：

```bash
uv run python -c 'import json; d=json.load(open("data/registry/documents.json")); [print(k,v["status"],v["filename"],v.get("error")) for k,v in d.items() if v["status"]=="failed"]'
```

确保 API 已经在终端二启动，然后重新索引指定文档：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_ID>/reindex
```

将 `<DOCUMENT_ID>` 替换为失败文档 ID。

查询任务状态：

```bash
curl http://127.0.0.1:8000/api/v1/jobs
```

项目现在会在 reindex 时自动清理失败插入和 duplicate 残留状态，并对 DeepSeek 瞬时连接错误进行自动重试。

### 9.2 仅事件索引失败

如果文档主状态已经是 `ready`，但 `event_index_status=failed`，不要重建 LightRAG。直接：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

也可以通过 API 单独重建：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_ID>/events/rebuild
```

事件失败不会破坏已经成功的向量和语义图谱索引；查询会自动降级到基础 LightRAG。

### 9.3 DeepSeek 返回格式不完整

事件抽取器会逐条校验 JSON。模型漏掉 `evidence` 时，系统从当前原文片段中选择真实句子
补齐，不生成伪证据；其他坏候选会被跳过。当整个片段都不合法时文档标为事件失败，
修正代码或模型配置后重新运行批处理即可。

### 9.4 验证事件库派生数据

当所有文档均为事件 `ready` 时再次运行批处理，不会调用 DeepSeek，只会重算派生数据：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

输出应类似：

```text
No documents need event rebuilding.
Refreshed derived event data: {'events': ..., 'mentions': ..., 'relations': ..., 'conflicts': ...}
```

---

## 10. 常见问题检查

### Docker/Colima 没有运行

```bash
colima start
```

```bash
docker info
```

```bash
docker-compose up -d qdrant
```

### 检查 DeepSeek 连通性

```bash
curl -v --connect-timeout 15 https://api.deepseek.com
```

未携带 API Key 时返回 `401` 说明 DNS、TCP 和 TLS 基本正常。

更可靠的项目级检查是：

```bash
uv run python scripts/doctor.py
```

它会携带本机密钥查询模型列表，但不会打印密钥。

### 检查代理变量

```bash
env | grep -i proxy
```

如果显示已经失效的代理，当前终端临时清除：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

然后重新诊断：

```bash
uv run python scripts/doctor.py
```

### 端口是否被占用

检查 API 端口：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

检查 Streamlit 端口：

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
```

检查 Qdrant 端口：

```bash
lsof -nP -iTCP:6333 -sTCP:LISTEN
```

### Docker 指向不存在的 Colima Socket

如果看到 `~/.colima/default/docker.sock: no such file or directory`：

```bash
colima start
```

```bash
docker context use colima
```

```bash
docker info
```

本机安装的是独立 `docker-compose` 命令时使用：

```bash
docker-compose up -d qdrant
```

不要改成不可用的 `docker compose` 子命令。

---

## 11. 停止系统

在 API 终端按：

```text
Ctrl+C
```

在 Streamlit 终端按：

```text
Ctrl+C
```

停止 Qdrant：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare
```

```bash
docker-compose down
```

停止 Colima：

```bash
colima stop
```

如果近期还会继续使用系统，可以只停止 API 和 Streamlit，保留 Qdrant/Colima 运行。

---

## 12. 重要数据位置

```text
data/collection/raw/       自动抓取的原始数据
data/collection/clean/     清洗后的 Markdown 和 sidecar
data/collection/runs/      自动抓取运行报告
data/collection/review_queue/ 待人工复核数据
data/raw/                  已导入知识库的原件
data/parsed/               已解析的 Markdown 和元数据
data/registry/             文档与任务注册表
data/events/               事件、来源提及、相邻关系和冲突
data/lightrag/             LightRAG 文本块、实体、关系与图谱状态
qdrant_storage/            Qdrant 向量数据
```

不要手动编辑或删除 `data/lightrag`、`data/events`、`qdrant_storage` 和 `data/registry`
中的内容。

---

## 13. 查询与证据验收

### 13.1 结构化事件查询

```bash
curl -X POST http://127.0.0.1:8000/api/v1/events/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"2024到2026年 Anduril 有哪些关键合同？","limit":20}'
```

检查：

- `inferred_start_time` 和 `inferred_end_time` 是否正确；
- `events` 是否只包含与 Anduril 直接相关的事件；
- `status` 是否区分 `single_source`、`multi_source`、`disputed`；
- `mentions.evidence` 是否为原文；
- 时间关系是否为相邻事件链，而不是所有 Anduril 事件两两相连。

### 13.2 完整三通路问答

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"按时间线梳理2024到2026年 Anduril 的关键合同和合作",
    "mode":"mix",
    "use_events":true,
    "event_limit":12,
    "enable_rerank":false
  }'
```

合格结果应同时具备：

1. `metadata.event_pathway.enabled=true`；
2. `events` 不为空且没有明显无关实体；
3. `citations` 包含 `[E1]` 等事件引用；
4. 每个 `[E#]` 都能回到文档标题、路径和 `excerpt`；
5. 回答不把 `precedes` 自动解释为因果。

如需做 A/B 对比，把 `use_events` 改为 `false`，其余参数保持不变。

---

## 14. 测试、备份与版本提交

运行代码和文档检查：

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

需要一起备份：

```text
data/raw/
data/parsed/
data/registry/
data/events/
data/lightrag/
qdrant_storage/
```

`.env` 必须单独安全保管，不能提交 Git。`data/events/` 已被 `.gitignore` 排除。

查看当前分支和未提交修改：

```bash
git status --short --branch
```

本项目的事件功能分支：

```bash
git switch feature/event-pipeline
```
