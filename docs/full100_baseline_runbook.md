# MultiHop-RAG 全量语料 / 前 100 题基线运行手册

更新日期：2026-09-04。本文只记录已准备好的入口；截至本文更新时，未启动任何新增的正式索引或 100 题检索。

## 统一口径

- 原始语料：609 篇文档。
- 统一切块：recursive 700 / overlap 100，共 2512 块。
- 问题：原始顺序前 100 题，不随机重排；其中 91 题有可用于文档检索评分的 Gold。
- Gold evidence：246 条。
- 检索深度与评分：统一保存 Top-20，报告 Hits、Evidence Recall、Complete Recall、MRR、nDCG 的 `@5/@10/@20`。
- 通用外部格式：
  - `data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json`
  - `data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json`
- 统一 Gold 评分源：`data/benchmarks/multihop_rag/`。

所有正式运行都使用独立 runtime/output 目录。重复运行同一配置会复用索引或跳过已有问题；修改语料、模型或关键参数时必须换目录，不能混写。

## 已准备的方法

| 分组 | 论文中建议名称 | 状态 | 是否需要新增索引 | 在线模型 |
|---|---|---|---:|---|
| 传统检索 | Dense (Naive) | 已可运行，full609 dense 索引已存在 | 否 | 智谱 Embedding |
| 传统检索 | Dense + BM25 (Hybrid) | 已可运行并复用 dense | 否 | 智谱 Embedding |
| 图 RAG | LightRAG | 代码已集成；当前 full609 的 stock `lightrag/` 为空 | 是 | DeepSeek + 智谱 |
| 内部对照 | Light-Entity | full609 索引已存在 | 否 | 查询规划用 DeepSeek |
| Agentic 对照 | Agentic-Hybrid / Agentic-Light-Entity | 已可运行并复用对应索引 | 否 | 每题调用 DeepSeek |
| 外部图 RAG | HippoRAG 2 | Python 3.10 环境、数据、适配与评分均已准备 | 是 | DeepSeek + 智谱 |
| 外部事件图 | E²RAG | 独立环境、full100 配置体检通过 | 是，两套图 | DeepSeek + 智谱 |
| 外部事件图 | EventRAG-budgeted | full100 配置体检通过；Neo4j 可用独立卷 | 是 | DeepSeek + 智谱 |
| 外部图 RAG | GraphRAG Local | 3.1.2 环境、输入映射、检索适配、评分已准备；官方 dry-run 通过 | 是 | DeepSeek + 智谱 |
| 树 RAG | RAPTOR | 官方源码、独立环境、来源追踪适配、评分已准备 | 是 | DeepSeek + 智谱 |

KG²RAG 按当前决定暂不进行。HippoRAG 使用 HippoRAG 2，不重复跑 HippoRAG 1。RAPTOR 与 ELC-RAG 不是最直接的事件图对手，建议排在主要图基线之后运行。

## 推荐执行顺序

1. 先跑无需新索引的 Dense、Hybrid、Light-Entity 和两个 Agentic 对照。
2. 建 stock LightRAG 索引并检索。
3. 跑 HippoRAG 2。
4. 跑 E²RAG 与 EventRAG-budgeted；二者是更直接的实体/事件图对照。
5. 预算允许后跑 GraphRAG Local。
6. 最后跑 RAPTOR。

这样安排可以最早得到论文主表所需的低成本下界和关键图对照，也能避免一开始就把预算耗在 GraphRAG 的社区摘要和 RAPTOR 的递归摘要上。

## 项目内置基线

以下各组应分开运行，便于单独统计时间和费用。它们都复用
`data/lightrag/evaluation/multihop_full609_elc_v1`，但写入不同结果目录。

```bash
uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods naive,hybrid,light_entity \
  --limit 100 --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir data/lightrag/evaluation/multihop_full609_elc_v1 \
  --output-dir data/exports/evaluation/2026-09-04_local_baselines_q100 \
  --fail-fast
```

```bash
uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods agentic_hybrid,agentic_light_entity \
  --limit 100 --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir data/lightrag/evaluation/multihop_full609_elc_v1 \
  --output-dir data/exports/evaluation/2026-09-04_agentic_controls_q100 \
  --fail-fast
```

### EvoNeed 首轮规划检索对照

`ours_evoneed_planned` 先用 EvoNeed 产生最多 3 个需求动作，然后在第一轮并行执行对应的 vector/graph/event 子查询。它复用已有 full609 EvoLink 索引，不重做文档抽取。先用 5 题检查：

```bash
AGENTIC_MAX_RETRIEVAL_ROUNDS=1 \
uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop-rag \
  --methods ours_evoneed_planned \
  --limit 5 --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --runtime-dir data/lightrag/evaluation/multihop_full609_elc_v1 \
  --output-dir data/exports/evaluation/2026-09-04_evoneed_planned_n5_r1 \
  --fail-fast
```

检查无误后，把 `--limit 5` 改为 `--limit 100`，并把输出目录改为
`data/exports/evaluation/2026-09-04_evoneed_planned_q100_r1`。不要与原版 `ours_evoneed` 共用输出目录。

stock LightRAG 会在同一 runtime 的空 `lightrag/` 子目录建立自己的索引，不会覆盖 `dense/`、`lightrag_entity/`、`events/` 或 `evolink/`：

```bash
uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --dataset multihop-rag --methods light \
  --limit 100 --cutoffs 5,10,20 \
  --chunking-strategy recursive --chunk-size 700 --chunk-overlap 100 \
  --index-batch-size 16 \
  --runtime-dir data/lightrag/evaluation/multihop_full609_elc_v1 \
  --output-dir data/exports/evaluation/2026-09-04_stock_lightrag_q100 \
  --fail-fast
```

## HippoRAG 2

详细说明见 `docs/hipporag_baseline.md`。依次把 `--stage` 设为 `doctor`、`index`、`retrieve`：

```bash
data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python \
  scripts/run_hipporag_baseline.py \
  --corpus data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json \
  --runtime-dir data/hipporag/evaluation/2026-09-04_multihop_full609_q100 \
  --output data/exports/evaluation/2026-09-04_hipporag2_q100/raw.jsonl \
  --stage doctor --limit 100 --top-k 20
```

## E²RAG

使用隔离环境，避免官方旧依赖污染主项目。依次使用 `doctor`、`index`、`retrieve`：

```bash
data/benchmarks/_venvs/e2rag/bin/python scripts/run_e2rag_baseline.py \
  --corpus data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json \
  --runtime-dir data/e2rag/evaluation/2026-09-04_multihop_full609_q100 \
  --output data/exports/evaluation/2026-09-04_e2rag_q100/raw.jsonl \
  --stage doctor --variant vanilla --limit 100 --top-k 20
```

索引、检索阶段默认只在终端显示总进度条。E²RAG 自身的详细输出保存在
runtime 目录的 `e2rag_upstream.log`；调试时可添加 `--show-upstream-output`。
实体和事件索引每完成一个 `--index-batch-size` 批次就记录断点。

## EventRAG-budgeted

正式 full609 运行使用独立 Neo4j 容器、端口和目录，不复用 n30 图：

```bash
EVENTRAG_NEO4J_CONTAINER_NAME=rag-graph-demo-eventrag-full609 \
EVENTRAG_NEO4J_HTTP_PORT=7476 \
EVENTRAG_NEO4J_BOLT_PORT=7689 \
EVENTRAG_NEO4J_DATA_DIR=./data/eventrag/neo4j_full609/data \
EVENTRAG_NEO4J_LOGS_DIR=./data/eventrag/neo4j_full609/logs \
docker-compose -f docker-compose.eventrag.yml up -d eventrag-neo4j
```

依次使用 `doctor`、`index`、`retrieve`，并在三个阶段都保留相同的 Neo4j 地址：

```bash
uv run --extra evaluation --extra eventrag python scripts/run_eventrag_baseline.py \
  --corpus data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json \
  --runtime-dir data/eventrag/evaluation/2026-09-04_multihop_full609_q100 \
  --output data/exports/evaluation/2026-09-04_eventrag_budgeted_q100/raw.jsonl \
  --neo4j-uri bolt://127.0.0.1:7689 \
  --stage doctor --limit 100 --top-k 20
```

## Microsoft GraphRAG Local

`prepare` 只生成输入、来源映射与配置，已完成；正式运行时先 `index` 后 `retrieve`：

```bash
uv run python scripts/run_graphrag_baseline.py \
  --corpus data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json \
  --runtime-dir data/graphrag/evaluation/2026-09-04_multihop_full609_q100 \
  --output data/exports/evaluation/2026-09-04_graphrag_local_q100/raw.jsonl \
  --stage index --limit 100 --top-k 20
```

这里评估的是 Local Search 找到并放入上下文的 Sources，不调用最终答案生成来冒充检索指标。来源短 ID 会映射回统一 `chunk_id/document_id`。

## RAPTOR

依次使用 `doctor`、`index`、`retrieve`：

```bash
data/benchmarks/_venvs/raptor/bin/python scripts/run_raptor_baseline.py \
  --corpus data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/external_multihop_full609_q100/multihop_full609_q100_recursive700.json \
  --runtime-dir data/raptor/evaluation/2026-09-04_multihop_full609_q100 \
  --output data/exports/evaluation/2026-09-04_raptor_q100/raw.jsonl \
  --stage doctor --limit 100 --top-k 20
```

RAPTOR 原生返回叶子与递归摘要节点，摘要节点没有文档 ID。适配器保留每个摘要节点的叶子后代，并在候选后代中用同一查询向量重排原始叶子，再折叠到文档级。这一适配必须在论文中写清楚，不能称为官方原样的 document-ranking 输出。

## 外部结果统一评分

HippoRAG/E²RAG/EventRAG 也可继续使用各自现有评分脚本。GraphRAG 与 RAPTOR 使用通用入口：

```bash
uv run --extra evaluation python scripts/score_external_baseline.py \
  --dataset-dir data/benchmarks/multihop_rag \
  --raw-results data/exports/evaluation/2026-09-04_graphrag_local_q100/raw.jsonl \
  --output-dir data/exports/evaluation/2026-09-04_graphrag_local_q100 \
  --method graphrag_local --cutoffs 5,10,20
```

将 raw 路径和 `--method` 换成 `raptor` 即可评分 RAPTOR。评分前应检查：结果 ID 无重复、只含前 100 题、Gold 题数为 91、每题最多 20 个去重文档。
