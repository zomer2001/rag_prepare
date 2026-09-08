# HippoRAG 2 基线接入

## 为什么使用 2，而不是 1

HippoRAG 1（NeurIPS 2024）使用 LLM OpenIE 构建实体—事实图，并从问题实体出发执行
Personalized PageRank。HippoRAG 2（ICML 2025）保留这条主线，但进一步：

- 把原始 passage 节点与短语/事实节点共同放入图中，融合稠密文本证据和稀疏图结构；
- 同时执行 query-to-triple 与 query-to-passage 链接，而不只依赖问题中的命名实体；
- 使用 recognition memory 的 LLM 过滤候选事实，减少错误图种子；
- 在没有可靠事实时回退到稠密 passage 检索，兼顾简单事实题与多跳题。

因此二者必须在论文中标清版本，但主实验只运行 HippoRAG 2 即可。HippoRAG 1 只在需要复现
历史结果或分析“加入 passage/recognition memory 前后的差异”时作为可选补充，不值得在当前阶段
再建立一套全量索引。

## 接入边界

HippoRAG 使用官方 `c617143` 版本，保留其 OpenIE、实体/事实/文本向量、同义边和
Personalized PageRank 检索。当前项目只增加以下兼容工作：

- 使用与主实验相同的 `recursive + 700/100` Chunk；
- 将 `document_id` 和 `chunk_id` 写入 HippoRAG source metadata；
- DeepSeek 负责 OpenIE，并显式关闭 thinking；
- 智谱 `embedding-3` 负责全部向量，固定为 1024 维；
- 将 `retrieve()` 的结果转换成当前项目的统一检索指标。

DWIE 实体抽取结果不会注入 HippoRAG。这样得到的是 HippoRAG 官方方法基线，而不是
`HippoRAG + Ours extraction` 消融。

## 环境

HippoRAG 使用独立 Python 3.10 环境：

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.10 \
  data/benchmarks/_downloads/baselines/hipporag/.venv
```

官方 `setup.py` 固定依赖 Linux/CUDA 的 vLLM，本机在线 API 模式不需要 vLLM。因此环境采用
在线模式所需依赖并使用 `--no-deps` 安装源码，避免修改当前项目的 `.venv`。

## 已完成的小规模验证

2026-08-18 的验证集合包含3个问题、19篇原文和77个统一 Chunk。结果位置：

- runtime：`data/lightrag/evaluation/2026-08-18_hipporag_stage2_q3_v2/`
- report：`data/exports/evaluation/2026-08-18_hipporag_stage2_q3_v2/`

索引成功生成1400个 phrase node、77个 passage node和1693个去重 fact。77个 Chunk 中实体
抽取全部非空，76个 Chunk 的三元组非空。索引耗时201.77秒。3个问题均完成检索，结果为：

| 指标 | 结果 |
|---|---:|
| Hits@10 | 1.0000 |
| Evidence Recall@10 | 0.8889 |
| Complete Recall@10 | 0.6667 |
| nDCG@10 | 0.7588 |
| MRR@10 | 1.0000 |
| P95 retrieval latency | 3155.27 ms |

该结果只验证方法接入和证据映射，样本量不足以作为论文结论。

## n30 阶段性运行

全量输入已生成：100篇文档、443个统一 Chunk、30个问题和59条 Gold Evidence。

建立索引并检索：

```bash
data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python \
  scripts/run_hipporag_baseline.py \
  --corpus data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_n30_recursive700_corpus.json \
  --queries data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_n30_recursive700.json \
  --runtime-dir data/lightrag/evaluation/2026-08-18_hipporag_n30_recursive700 \
  --output data/exports/evaluation/2026-08-18_hipporag_n30_recursive700/hipporag_raw.jsonl \
  --stage all \
  --top-k 10 \
  --llm-concurrency 4
```

计算统一指标：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/score_hipporag_results.py \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --dataset-name multihop-rag-seed42-n30-corpus100 \
  --raw-results data/exports/evaluation/2026-08-18_hipporag_n30_recursive700/hipporag_raw.jsonl \
  --output-dir data/exports/evaluation/2026-08-18_hipporag_n30_recursive700 \
  --cutoffs 4,10
```

相同命令可以直接重跑。HippoRAG 会保留 LLM SQLite cache、OpenIE、Embedding 和图文件；
适配脚本还会按 `query_id` 跳过已经写入 `hipporag_raw.jsonl` 的问题。

## full609/q100 正式输入（已准备，尚未运行）

2026-09-04 已生成与主实验完全相同的前 100 题及完整语料输入：

- 原始文档：609篇；
- `recursive 700/100` Chunk：2512个；
- 问题：100题，其中91题有 Gold Evidence、9题为 `null_query`；
- Gold Evidence：246条；
- 方法：HippoRAG 2，官方源码提交 `c617143`；
- LLM：`deepseek-v4-flash`，thinking disabled；
- Embedding：智谱 `embedding-3`，1024维。

输入文件：

- `data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700_corpus.json`
- `data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700.json`

下面命令截至 2026-09-04 **均未执行**。正式运行时应分开执行 doctor、index、retrieve，避免一次
`--stage all` 难以判断中断发生在哪个阶段。

仅检查环境和输入，不调用模型：

```bash
data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python \
  scripts/run_hipporag_baseline.py \
  --corpus data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700.json \
  --runtime-dir data/lightrag/evaluation/2026-09-04_hipporag2_full609_q100_recursive700 \
  --output data/exports/evaluation/2026-09-04_hipporag2_full609_q100_recursive700/hipporag_raw.jsonl \
  --stage doctor \
  --top-k 20 \
  --llm-concurrency 4
```

建立索引：

```bash
data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python \
  scripts/run_hipporag_baseline.py \
  --corpus data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700.json \
  --runtime-dir data/lightrag/evaluation/2026-09-04_hipporag2_full609_q100_recursive700 \
  --output data/exports/evaluation/2026-09-04_hipporag2_full609_q100_recursive700/hipporag_raw.jsonl \
  --stage index \
  --top-k 20 \
  --llm-concurrency 4
```

索引完成后检索100题：

```bash
data/benchmarks/_downloads/baselines/hipporag/.venv/bin/python \
  scripts/run_hipporag_baseline.py \
  --corpus data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700_corpus.json \
  --queries data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset/ours_multihop_full609_q100_recursive700.json \
  --runtime-dir data/lightrag/evaluation/2026-09-04_hipporag2_full609_q100_recursive700 \
  --output data/exports/evaluation/2026-09-04_hipporag2_full609_q100_recursive700/hipporag_raw.jsonl \
  --stage retrieve \
  --top-k 20 \
  --llm-concurrency 4
```

统一评分：

```bash
UV_CACHE_DIR=.uv-cache uv run --extra evaluation \
  python scripts/score_hipporag_results.py \
  --dataset-dir data/benchmarks/multihop_rag \
  --dataset-name multihop-rag-full609-q100 \
  --raw-results data/exports/evaluation/2026-09-04_hipporag2_full609_q100_recursive700/hipporag_raw.jsonl \
  --output-dir data/exports/evaluation/2026-09-04_hipporag2_full609_q100_recursive700 \
  --cutoffs 5,10,20
```

注意：HippoRAG 2 会对全部 phrase 节点执行分批 KNN 以构造同义边，核心计算随 phrase 数量接近
二次增长。2512个 Chunk 仍在可运行范围内，但不能简单把443-Chunk的耗时严格线性外推；首次正式
索引应预留约2–5小时，并观察内存与 `KNN for Queries` 阶段。
