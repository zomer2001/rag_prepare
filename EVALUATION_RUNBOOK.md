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
| `ours_evolink` | 复用`ours`的三条专业通路；将LightRAG已经抽取的语义知识和现有事件治理后同步到Neo4j，以实体和事件命中为种子做一次受限跨图扩展，返回查询级实体—断言—事件证据子图；不建立五类TKG向量索引 |
| `agentic_ours_evolink` | 通用充分性判断与自由文本改写控制同一`ours_evolink`检索器；每轮仍固定执行完整三通路，作为“多轮本身是否有效”的公平基线 |
| `ours_evoneed` | 在同一EvoLink索引上显式维护原文、关系、演化三类Knowledge Need；v2先以原问题执行一次完整三通路建立初始证据池，再按未满足类型执行专用Query，为每个需求保留完整候选深度，并以相关性主导的缺口覆盖和跨路连接融合证据 |
| `ours_tkg` | 独立统一TKG方法：语义事实与事件直接进入声明级TKG；智谱Embedding分别索引Chunk、Entity、Predicate、Assertion和Event，Neo4j受约束扩展后沿Mention回溯来源证据；不建立LightRAG语义图 |
| `ours_tkg_dynamic` | 复用同一TKG索引；按问题意图动态启停实体、关系和事件入口，依据首批相似度动态截断候选，并按实体/关系/时间覆盖做带权扩展与排序 |
| `agentic_ours_tkg` | 复用`ours_tkg`的统一TKG索引；Agent先判断证据是否充分，不充分时改写问题并再次调用同一套TKG检索，最后融合多轮证据并生成带引用的回答 |
| `agentic_ours_tkg_dynamic` | Agent控制器调用动态TKG检索；证据不足时改写问题并重新规划动态入口 |

所有方法使用同一批固定token切块。默认块大小700、重叠100，最终Top-K预算相同。

`ours_evolink`不会再次调用语义或事件抽取：语义部分直接读取`light_entity`已完成的
LightRAG图，事件部分直接复用`ours`的Event Registry；只增加实体治理、Neo4j同步和一次
查询期跨图扩展。对比命令：

```bash
docker-compose up -d qdrant neo4j

UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours,ours_evolink \
  --limit 30 --cutoffs 4,10 --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_evolink_v1 \
  --output-dir data/exports/evaluation/multihop_evolink_v1 \
  --fail-fast
```

`ours_evoneed`与`agentic_ours_evolink`同样复用上述runtime，不新增索引。前者的轨迹保存
初始Knowledge Need、每轮选路动作、更新后的缺口和最终关闭状态；Summary额外报告三类
需求数量、通路调用率、Need Closure Rate与充分证据率。核心消融命令：

```bash
AGENTIC_MAX_RETRIEVAL_ROUNDS=2 UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours_evolink,agentic_ours_evolink,ours_evoneed \
  --limit 30 --cutoffs 4,10 --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_evolink_v1 \
  --output-dir data/exports/evaluation/DATE_evoneed_core_ablation_n30 \
  --fail-fast
```

v2修改了规划Prompt和证据融合，必须写入新输出目录，但继续复用同一runtime，无需重建索引：

```bash
AGENTIC_MAX_RETRIEVAL_ROUNDS=2 UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours_evoneed \
  --limit 30 --cutoffs 4,10 --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_evolink_v1 \
  --output-dir data/exports/evaluation/2026-08-14_evoneed_v2_n30 \
  --fail-fast
```

`ours_tkg`是新增方法而不是对`ours`的原地替换。它要求`EMBEDDING_PROVIDER=zhipu`，使用
独立Neo4j workspace和独立Qdrant集合；评测索引不会覆盖主应用的`default`工作区。

30题统一TKG评测命令：

```bash
docker-compose up -d qdrant neo4j

EMBEDDING_PROVIDER=zhipu UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours_tkg \
  --limit 30 \
  --cutoffs 4,10 \
  --chunking-strategy recursive \
  --index-batch-size 16 \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_n30_zhipu_tkg_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_n30_zhipu_tkg_v1 \
  --fail-fast
```

30题Agentic TKG评测命令（复用上面的runtime索引，但使用独立输出目录）：

```bash
docker-compose up -d qdrant neo4j

EMBEDDING_PROVIDER=zhipu UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods agentic_ours_tkg \
  --limit 30 \
  --cutoffs 4,10 \
  --chunking-strategy recursive \
  --index-batch-size 16 \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_n30_zhipu_tkg_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_n30_zhipu_agentic_tkg_v1 \
  --fail-fast
```

该方法默认最多检索两轮。每轮都会先运行TKG Planner和五类向量种子检索，再做Neo4j扩展；
第一轮证据不足时，Agent会调用DeepSeek生成更聚焦的检索问题。输出的
`agentic_ours_tkg.jsonl`同时保存最终证据、生成答案、引用和完整`trajectory`。

不重建索引，直接对比原始与动态TKG检索：

```bash
EMBEDDING_PROVIDER=zhipu UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours_tkg,ours_tkg_dynamic \
  --limit 30 --cutoffs 4,10 --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_n30_zhipu_tkg_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_n30_zhipu_tkg_dynamic_v1 \
  --fail-fast
```

如果不使用固定本体，可设置`TKG_SCHEMA_MODE=open`。该模式关闭类型白名单、SHACL校验和
同类型硬匹配，抽取类型仅作为软标签；统一身份、Mention来源、Assertion、EventEffect、冲突
和回滚结构仍然保留。开放模式会使用独立指纹和Neo4j workspace，必须重建一次索引：

```bash
TKG_SCHEMA_MODE=open EMBEDDING_PROVIDER=zhipu UV_CACHE_DIR=.uv-cache \
  uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-n30-corpus100 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --methods ours_tkg_dynamic \
  --limit 30 --cutoffs 4,10 --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_n30_zhipu_tkg_open_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_n30_zhipu_tkg_open_dynamic_v1 \
  --fail-fast
```

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
