# EventRAG hybrid 基线

该适配器保留 EventRAG 的 hybrid 检索逻辑，采用低成本抽取配置：每块按文本出现顺序最多保留 6 个实体和 8 条关系，关闭 gleaning 与知识扩张，并把抽取输出限制为 1200 tokens。该版本应在实验中标记为 `EventRAG-budgeted`，不能表述为官方全预算复现。LLM 使用 DeepSeek，Embedding 使用智谱，向量存储使用 Qdrant，图存储使用独立 Neo4j。

## 启动服务

现有 Qdrant 继续由主 compose 提供；EventRAG 使用独立 Neo4j，避免和项目知识图谱混用。正式
full609 运行还应通过 compose 环境变量指定新的容器名、端口和数据目录，不能复用 n30 数据卷。

```bash
docker-compose up -d qdrant
docker-compose -f docker-compose.eventrag.yml up -d eventrag-neo4j
```

full609 隔离配置见 `docs/full100_baseline_runbook.md`。

默认地址为 Qdrant `http://127.0.0.1:6333`、EventRAG Neo4j `bolt://127.0.0.1:7688`。

## 数据准备

复用 HippoRAG 的统一导出格式，确保各方法使用相同文本块：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/export_hipporag_dataset.py \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --output-dir data/benchmarks/external_multihop_seed42_n30 \
  --name multihop_seed42_n30 \
  --chunking-strategy recursive \
  --chunk-size 700 \
  --chunk-overlap 100
```

## 检查、索引与检索

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation --extra eventrag \
  python scripts/run_eventrag_baseline.py \
  --corpus data/benchmarks/external_multihop_seed42_n30/multihop_seed42_n30_corpus.json \
  --queries data/benchmarks/external_multihop_seed42_n30/multihop_seed42_n30.json \
  --runtime-dir data/eventrag/evaluation/2026-08-18_multihop_seed42_n30 \
  --output data/exports/evaluation/2026-08-18_eventrag_hybrid_n30/raw.jsonl \
  --stage doctor
```

确认服务和配置正常后，将 `--stage doctor` 依次改成 `--stage index` 和 `--stage retrieve`。索引完成后会写入 `index_complete.json`，检索结果按 query 逐条追加，重新运行会跳过已完成 query。

## 计算指标

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/score_eventrag_results.py \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --raw-results data/exports/evaluation/2026-08-18_eventrag_hybrid_n30/raw.jsonl \
  --output-dir data/exports/evaluation/2026-08-18_eventrag_hybrid_n30 \
  --cutoffs 4,10
```

独立 Neo4j 当前对应一套 EventRAG 索引。更换语料时应使用新的独立数据卷或先明确清理旧 EventRAG 图数据，不能与旧语料混合。
