# Retrieval Evaluation Runbook

本项目的统一评测入口是 `scripts/evaluate_retrieval.py`。它直接调用检索接口，不经过
FastAPI、Streamlit，也不生成最终答案。

## 方法定义

| 名称 | 实现 |
|---|---|
| `naive` | 当前配置的Embedding（本地、智谱或百炼）对原文Chunk做独立稠密向量检索 |
| `hybrid` | 评测基线：同一稠密检索 + `BM25Okapi` + RRF；不使用知识图谱 |
| `light` | 官方 LightRAG `mix` 检索，调用 `aquery_data()`，不生成答案 |
| `light_entity` | 受控实体索引；统一Planner后分别运行LightRAG `naive`文本向量与LightRAG `hybrid`语义图谱，再按文档RRF |
| `ours` | 与 `light_entity` 完全相同，再加入事件向量、结构过滤和事件链扩展的第三路证据 |

所有方法使用同一批固定token切块。默认块大小700、重叠100，最终Top-K预算相同。

注意两种`hybrid`不是同一个算法：评测方法名`--methods hybrid`表示工业界常见的
Dense+BM25混合检索；LightRAG内部`mode="hybrid"`表示local实体图检索+global关系图检索，
不含BM25，必须依赖实体关系图。LightRAG `mix`才是在其图谱`hybrid`之上再加入原文Chunk
稠密向量召回。

`ours`的事件不是与实体关系同一次抽取：实体关系索引完成后，系统按独立的大事件片段调用
事件Prompt并写入Event Registry。事件规范文本的Embedding在第一次事件查询时分批延迟生成，
缓存为`events/event_vectors.npz`；事件向量召回后再做时间、实体、类型、状态和事件链处理，
最终交出的仍是来源文档中的事件证据。

`*_summary.json`中的`retrieval_diagnostics`会额外记录每条通路进入最终Top-K的问题数、比例
和平均文档数，用于区分“事件没有召回”和“事件召回但没有改变金证据指标”。Planner实体、
事件类型和无明确日期的时间信息只作为排序提示，不再要求单个事件同时包含所有实体；每个
规范事件最多向融合层贡献两个来源文档，避免一个多来源事件占满Top-K。

## 安装评测依赖

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra evaluation
```

`naive`和`hybrid`只需要Embedding模型。`light`、`light_entity`和`ours`还需要：

```bash
docker-compose up -d qdrant
uv run python scripts/doctor.py
```

并在`.env`中配置`DEEPSEEK_API_KEY`。LightRAG的图构建会调用API；`light_entity`与`ours`
每个问题只调用一次统一 Query Planner，同时生成图谱关键词和事件查询条件，
首次完整建库应先用`--limit`检查查询流程，但注意`--limit`只限制问题数，不缩小语料库。

## 零API Smoke Test

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset smoke \
  --dataset-dir demo_data/evaluation_smoke \
  --methods naive,hybrid
```

## MultiHop-RAG三基线

首次运行会从Hugging Face下载并规范化MultiHop-RAG到
`data/benchmarks/multihop_rag/`：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --prepare-only
```

随后运行三基线：

```bash
docker-compose up -d qdrant

UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods naive,hybrid,light \
  --cutoffs 4,10
```

先试10个问题：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods naive,hybrid,light \
  --limit 10 \
  --fail-fast
```

运行受控实体消融与完整系统：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods naive,hybrid,light,light_entity,ours
```

`light`与`light_entity`使用完全独立的LightRAG工作区和Qdrant workspace分区，不能共用图索引；
`ours`与`light_entity`共用同一个受控实体图索引、Query Planner和两路融合器，只额外建立
事件结构索引与持久化事件向量。因此对比
`light_entity`与`ours`可直接测量事件通路的增益。

Planner结果按版本、日期和问题持久化在运行目录的`events/query_plan_cache.json`；即使分两次
启动`light_entity`与`ours`，同一天同一问题也只调用一次Planner LLM，并复用完全相同的计划。

三路都返回可回溯证据：向量/图谱候选保留LightRAG `chunk_id + file_path`；事件候选保留
`document_id + source_segment_id + evidence`及字符区间。最终排名按文档RRF计算，但每个
入选文档最多保留三条通路各自最强的证据片段。新融合版本写入运行清单，旧版报告目录不能
与本版断点文件混用；LightRAG实体图索引本身仍可复用。

受控实体配置位于`prompts/entity_type/light_entity.yaml`，默认每个Chunk最多4个绝对关键实体、
4条关键关系（8条总记录），并关闭第二轮gleaning。模型响应在写入LightRAG前还会执行4/4
硬截断，关系两端必须属于保留下来的关键实体；实体和关系描述不限制长度。事件抽取
默认每个片段最多5个关键事件、每篇文档最多12个。Prompt或这些上限变化时，评测指纹变化并自动使用
新的`light_entity`工作区，避免新旧图谱混用。

API索引采用保守并发：DeepSeek最多4个请求、LightRAG内部2个文档、Embedding API 2个批次、
事件索引3个Worker。本地MPS/CUDA Embedding仍固定串行，避免同一模型对象并发推理。外层
LightRAG写锁仍保留，避免JSON/NetworkX状态被多个批次同时写入。

评测脚本会分别显示`dense:index`、`light/light_entity:index`、`ours:event-index`、
`event-vector:index`和`<method>:query`进度条，包括百分比、吞吐、已用时间和ETA。需要把输出重定向到纯日志或
在CI运行时，可增加`--no-progress`关闭动态进度条。

### 60题随机实验集（当前）

`data/benchmarks/multihop_rag_seed3254837894_n60_corpus200`从完整MultiHop-RAG中随机抽取
60题，并完整保留86篇金证据文档；另用独立随机种子抽取114篇干扰文档，合计200篇文档。
题目抽样种子为`3254837894`，干扰文档种子为`1621364359`；具体题号和SHA-256均记录在
该目录的`split_manifest.json`中。

构造命令（已执行，目标目录需不存在）：

```bash
uv run python scripts/prepare_evaluation_split.py \
  --source-dir data/benchmarks/multihop_rag \
  --output-dir data/benchmarks/multihop_rag_seed3254837894_n60_corpus200 \
  --sample-size 60 \
  --seed 3254837894 \
  --corpus-size 200 \
  --distractor-seed 1621364359
```

使用智谱Embedding依次评测全部五种方法：

```bash
docker-compose up -d qdrant

EMBEDDING_PROVIDER=zhipu UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed3254837894-n60-corpus200 \
  --dataset-dir data/benchmarks/multihop_rag_seed3254837894_n60_corpus200 \
  --methods naive,hybrid,light,light_entity,ours \
  --limit 60 \
  --cutoffs 4,10 \
  --index-batch-size 16 \
  --runtime-dir data/lightrag/evaluation/multihop_n60_seed3254837894_zhipu_v1 \
  --output-dir data/exports/evaluation/multihop_n60_seed3254837894_zhipu_v1 \
  --fail-fast
```

该命令按参数顺序运行五种方法。中断后用完全相同的命令重跑，会复用已完成的索引和逐题结果；
不要换runtime/output目录，也不要删除正在续跑的目录。

### 30题快速实验集

`data/benchmarks/multihop_rag_seed42_n30_corpus100` 是从完整 MultiHop-RAG 固定构造的
快速实验集：种子42抽取30题，完整保留38篇金证据文档，并固定抽取62篇干扰文档，合计
100篇文档、459个默认切块。全部金证据文档和证据文本均已校验存在。

构造命令（目标目录需不存在）：

```bash
uv run python scripts/prepare_evaluation_split.py \
  --source-dir data/benchmarks/multihop_rag \
  --output-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --sample-size 30 \
  --seed 42 \
  --corpus-size 100 \
  --distractor-seed 42
```

该集合适合验证流程和快速消融；正式结论仍应在完整语料或预先声明的正式切分上复验。

## 输入数据格式

自定义数据集目录需要两个文件，JSON数组或JSONL均可：

```text
benchmark/
├── corpus.jsonl
└── queries.jsonl
```

最小格式：

```json
{"document_id":"doc-1","title":"Title","text":"Document body"}
{"query_id":"q-1","question":"Question?","question_type":"temporal","gold_evidence":[{"document_id":"doc-1","text":"Gold fact"}]}
```

MultiHop-RAG原始字段`body/query/evidence_list/fact`也可以直接读取。

## 输出与断点续跑

默认结果写到：

```text
data/exports/evaluation/<dataset>/
├── naive.jsonl
├── hybrid.jsonl
├── light.jsonl
├── light_entity.jsonl
├── ours.jsonl
├── *_summary.json
├── summary.json
├── summary.csv
└── summary.md
```

每完成一个问题立即追加JSONL。使用相同输出目录重跑时，会跳过已有`query_id`，适合
长时间评测中断后继续。索引和向量缓存位于`data/lightrag/evaluation/<dataset>/`，
语料、Embedding提供方/模型/维度或切块参数改变后指纹会变化。稠密向量建库也会逐批落盘并显示
`[dense:index]`进度；首次冷启动中断后，用相同命令即可从已完成批次继续。

如果同一输出目录改变了问题限制、方法参数或模型配置，建议显式指定新的目录：

```bash
--runtime-dir data/lightrag/evaluation/multihop_run2 \
--output-dir data/exports/evaluation/multihop_run2
```

## 指标

脚本整体和按`question_type`分别输出：

- `hits@K`：Top-K至少命中一条金证据；
- `evidence_recall@K`：召回的不同金证据比例；
- `complete_recall@K`：Top-K是否取全问题所需全部证据；
- `map@K`、`mrr@K`、`ndcg@K`；
- 查询平均延迟、P50和P95。

无金证据的`null_query`会执行检索并记录耗时，但不会混入上述证据指标。

## 切换到阿里云百炼 Embedding

默认仍使用本地MPS。使用百炼OpenAI兼容接口时，在`.env`中设置：

```dotenv
EMBEDDING_PROVIDER=bailian
BAILIAN_API_KEY=你的百炼密钥
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
BAILIAN_EMBEDDING_DIMENSIONS_ENABLED=true
```

若使用百炼业务空间专属域名，把`BAILIAN_BASE_URL`替换为对应地域和Workspace ID的
兼容地址。切换提供方、模型或维度后必须使用新的`--runtime-dir`和`--output-dir`；
旧向量与新向量即使维度相同也不能放入同一索引。

智谱 GLM 使用同一套 Provider 接口：

```dotenv
EMBEDDING_PROVIDER=zhipu
ZHIPU_API_KEY=你的智谱密钥
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSION=1024
```

智谱与本地、百炼同样必须使用各自独立的 runtime/output 目录。
