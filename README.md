# RAG Graph Demo

面向 Apple Silicon M4 16GB Mac 的 Anduril OSINT 多通路 RAG Demo。LightRAG 负责
向量检索与实体关系图谱，事件断言层补充时间、状态、演化和原文证据；Collection 管线
负责官网、合同、论文、专利、新闻和智库资料的采集与规范化。

相关文档：

- [`PROJECT_PLAN.md`](PROJECT_PLAN.md)：初始架构与选型论证。
- [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)：当前架构、三通路索引、查询协同、
  存储、验证和演进路线。
- [`THREE_PATH_FUSION_TECHNICAL_REPORT.md`](THREE_PATH_FUSION_TECHNICAL_REPORT.md)：
  文本向量、语义图谱和事件三路从索引、召回到文档级证据融合的专项技术报告。
- [`RUNBOOK.md`](RUNBOOK.md)：可逐行执行的采集、索引、事件补建、验收和恢复流程。
- [`USER_GUIDE.md`](USER_GUIDE.md)：启动、导入、索引、查询、Prompt 定制和 Agent 接入。
- [`EVENT_PATHWAY.md`](EVENT_PATHWAY.md)：事件通路的实现、数据模型、API、运行方式和边界。
- [`EVALUATION_RUNBOOK.md`](EVALUATION_RUNBOOK.md)：Naive、Hybrid、官方 LightRAG、
  `light_entity` 与 `ours` 的一键评测、数据格式、指标和断点续跑。

## 已实现能力

- 上传、去重、删除和重新索引 Markdown、DOCX、文字型 PDF。
- 文档解析任务状态、错误信息和解析结果预览。
- LightRAG `naive`、`local`、`global`、`hybrid`、`mix` 五种模式。
- 本地 `Qwen/Qwen3-Embedding-0.6B`（MPS）、智谱 `embedding-3` 或阿里云百炼
  Embedding，以及可选 BGE Reranker。
- 受控实体版 `light_entity`：高精度实体类型、去噪规则、规范命名和关系约束。
- DeepSeek OpenAI 兼容 API 的实体关系抽取与答案生成。
- Qdrant 向量存储、NetworkX 图存储和本地 JSON 状态存储。
- 文档、页码、章节和原文片段引用映射。
- 图谱可视化及同一问题的多模式横向比较。
- FastAPI、Streamlit、批量导入、全量重建、诊断和 Smoke Test。
- 统一 Query Planner，以及文本向量、语义图谱、事件向量三路可回溯证据融合。
- 事件抽取、跨文档合并、来源段绑定、时间线、立场冲突检测和事件增强问答。

统一检索器显式调用 LightRAG `naive` 文本向量和 `hybrid` 语义图谱两路，再与独立事件
向量通路做文档级 RRF；评测和在线回答使用同一批最终证据。

## 运行组件

| 组件 | 运行位置 | 默认选择 |
|---|---|---|
| FastAPI | macOS 本地进程 | `127.0.0.1:8000` |
| Streamlit | macOS 本地进程 | `127.0.0.1:8501` |
| Qdrant | Docker/Colima | `127.0.0.1:6333` |
| Embedding | macOS 本地进程、智谱或百炼 API | 默认 Qwen3 Embedding 0.6B |
| Reranker | macOS 本地进程，可关闭 | BGE Reranker v2 M3 |
| 图存储 | 本地文件 | NetworkX GraphML |
| LLM | DeepSeek API | `deepseek-v4-flash` |

第一版只需要一个数据库容器：Qdrant。无需安装 Neo4j、PostgreSQL、Redis 或
Elasticsearch。

## 首次启动

前置条件：

- Python 3.11
- `uv`
- Docker Desktop，或 Colima + Docker CLI + `docker-compose`
- DeepSeek API Key

执行：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
```

然后启动：

```bash
uv sync
docker-compose up -d qdrant
uv run python scripts/doctor.py
uv run uvicorn rag_graph_demo.api.app:app --reload --host 127.0.0.1 --port 8000
```

另开一个终端：

```bash
uv run streamlit run src/rag_graph_demo/ui/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

访问：

- Demo UI：<http://127.0.0.1:8501>
- API 文档：<http://127.0.0.1:8000/docs>
- Qdrant Dashboard：<http://127.0.0.1:6333/dashboard>

首次真正索引时会从 Hugging Face 下载 Qwen3 Embedding 模型，时间取决于网络。
模型只在第一次索引或查询时懒加载。M4 16GB 默认关闭 Reranker，先完成稳定演示后再
在 `.env` 设置 `RERANKER_ENABLED=true`。

如果已从 ModelScope 等镜像下载模型，可以把 `EMBEDDING_MODEL` 设置为本地目录，
同时保持 `EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-0.6B`，以维持稳定的 Qdrant
集合命名。

## 使用流程

1. 在“系统状态”确认 Qdrant 和 DeepSeek 已就绪。
2. 上传若干主题相近的 Markdown、DOCX 或文字型 PDF 新闻。
3. 等待文档状态从 `uploaded` 变为 `ready`。
4. 用 `naive` 测具体事实和原文定位。
5. 用 `local` 测某个公司、人物或事件的关系。
6. 用 `global` 测整批材料的主题和趋势。
7. 用 `mix` 综合原文证据与图谱关系。
8. 在“模式对比”和“知识图谱”页面展示差异。

索引阶段会调用 DeepSeek 多次进行实体关系抽取。建议先用 3 至 5 篇短新闻验证，再
逐步扩展到几十篇。

## 常用命令

检查基础环境：

```bash
uv run python scripts/doctor.py
```

不使用真实模型和 DeepSeek，验证 LightRAG 与 Qdrant 适配：

```bash
uv run python scripts/smoke_lightrag.py
```

批量导入目录并串行索引：

```bash
uv run python scripts/import_directory.py /absolute/path/to/news
```

只注册文件、不解析或建立索引：

```bash
uv run python scripts/import_directory.py /absolute/path/to/news --register-only
```

重建全部文档的 LightRAG 向量和图谱索引：

```bash
uv run python scripts/reindex_all.py
```

只为已有解析文档补建事件索引（可安全中断后重跑）：

```bash
uv run python scripts/rebuild_events.py --workers 2
```

清空本地文档、LightRAG 状态和当前 Embedding 对应的 Qdrant 集合：

```bash
uv run python scripts/reset_demo.py --yes
```

运行检查：

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pytest
```

运行不调用DeepSeek的检索评测Smoke Test：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset smoke \
  --dataset-dir demo_data/evaluation_smoke \
  --methods naive,hybrid
```

停止 Qdrant：

```bash
docker-compose down
```

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/v1/health` | 进程健康检查 |
| `GET` | `/api/v1/ready` | Qdrant、目录和密钥就绪检查 |
| `GET` | `/api/v1/diagnostics` | 完整诊断 |
| `POST` | `/api/v1/documents` | 上传文档 |
| `GET` | `/api/v1/documents` | 列出文档 |
| `POST` | `/api/v1/documents/{id}/reindex` | 重建单文档索引 |
| `DELETE` | `/api/v1/documents/{id}` | 删除文档和索引 |
| `GET` | `/api/v1/jobs` | 查看后台任务 |
| `POST` | `/api/v1/query` | 单模式问答 |
| `POST` | `/api/v1/query/compare` | 多模式比较 |
| `GET` | `/api/v1/graph` | 读取知识图谱 |
| `GET` | `/api/v1/graph/labels` | 获取图谱实体标签 |
| `GET` | `/api/v1/events` | 列出规范化事件 |
| `POST` | `/api/v1/events/query` | 按问题、实体和时间查询事件链 |
| `POST` | `/api/v1/documents/{id}/events/rebuild` | 单独重建文档事件 |

## 数据目录

```text
data/
├── raw/        # 原始上传文件
├── parsed/     # 规范化 Markdown 和来源 Sidecar
├── registry/   # 文档与任务注册表
├── events/     # 事件、提及、时间关系和冲突
├── lightrag/   # NetworkX、LightRAG KV 和状态
├── exports/
└── logs/
```

Qdrant 数据持久化到 `qdrant_storage/`。`.env`、文档、数据库、模型缓存和运行状态均
已被 `.gitignore` 排除。

## VS Code

项目已经配置 `.venv/bin/python`，可在 `Tasks: Run Task` 中运行：

- `Qdrant: Start`
- `API: Run`
- `UI: Run`
- `Tests: Run`
- `Doctor: Run`
- `LightRAG: Smoke Test`
- `Index: Rebuild All`

## 已知限制

- 第一版重点支持文字型 PDF；扫描件 OCR 默认关闭。
- PDF 页码引用依赖 Docling 分页和 LightRAG 文本块保留来源标记。
- DOCX 通常提供章节级或文档级引用，不保证原始 Word 页码。
- JSON 注册表适用于本地单用户 Demo，不适合多进程并发写入。
- 更换 Embedding 模型或维度后，应更换 `LIGHTRAG_WORKSPACE` 并重建索引。
- 删除或重建大量文档前，建议备份 `data/` 和 `qdrant_storage/`。

DeepSeek 默认模型名于 2026-07-31 按官方 API 文档验证为
`deepseek-v4-flash`；模型能力或名称变化时只需修改 `.env`。
