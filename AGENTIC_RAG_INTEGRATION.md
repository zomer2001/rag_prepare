# Agentic RAG 集成说明

## 当前集成范围

原项目新增了三个 Agentic 评测方法，但没有改变现有检索方法、`mix`、API 或 Streamlit UI。

| 方法 | 被 Agent 控制的底层检索 |
| --- | --- |
| `agentic_hybrid` | Dense + BM25，并在每轮内部做 RRF |
| `agentic_light_entity` | Vector + Semantic Graph，不使用 Event |
| `agentic_ours` | Vector + Semantic Graph + Event 三通路 |

所有 Agentic 变体都使用 LangGraph 在对应检索器外增加有界控制循环；以
`agentic_ours` 为例：

```text
问题
  -> 原 UnifiedRetrievalOrchestrator（vector + graph + event）
  -> DeepSeek 判断证据是否充分
  -> 充分：生成带证据 ID 的回答
  -> 不充分：DeepSeek 改写查询，再运行一次原三通路
  -> 跨轮 RRF 融合
  -> 最终回答
```

默认最多检索两轮。原三通路内部仍先按文档做 RRF；Agentic 层再对不同检索轮次的文档
排序做 RRF。

## 复用与隔离

直接复用原项目的：

- DeepSeek、智谱 Embedding 和 Qdrant 配置
- LightRAG Controlled Entity 索引
- Vector、Graph、Event 三通路与 Query Planner
- 数据集、切块、指标、运行锁和逐条 JSONL Checkpoint

没有迁入独立工程的 Qdrant Collection、Embedding 适配器、数据集 Loader 或评测 CLI，
因此不会产生第二套索引和评测框架。

Agentic 核心代码位于：

```text
src/rag_graph_demo/agentic/
  graph.py      LangGraph State、节点、条件路由和跨轮 RRF
  model.py      DeepSeek 结构化调用
  models.py     Assessment、Rewrite、Answer 和 AgenticRun
  prompts.py    三套 Prompt、Evidence 模板和 Prompt 指纹
```

## 评测输出

每个 `agentic_*.jsonl` 都沿用原检索结果结构，并增加：

- `generated_answer`
- `citations`
- `insufficient_evidence`
- `trajectory.retrieval_rounds`
- `trajectory.llm_calls`
- `trajectory.input_tokens/output_tokens`
- `trajectory.events`

Summary 增加平均检索轮数、平均 LLM 调用数、总 Token 和证据不足率。

## 60 问 A/B 评测命令

使用新的 Output 目录，避免与旧评测 Manifest 冲突；Runtime 目录可以复用已有索引：

```bash
cd /Users/zomeryang/Documents/Codex/2026-07-31/rag_prepare

uv run python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-plus20260807-n60-corpus200 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_plus20260807_n60_corpus200 \
  --methods agentic_hybrid,agentic_light_entity,agentic_ours \
  --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_plus20260807_n60_zhipu_recursive_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_plus20260807_n60_agentic_variants_v1
```

该命令一次对比三个 Agentic 变体，并复用上一轮 Recursive 的 Dense、Light Entity 和
Event 索引。结果可以与既有 Baseline 输出对比。每种方法有独立 JSONL Checkpoint，任务
中断后用同一命令可以继续。

只测 Hybrid：

```bash
uv run python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-plus20260807-n60-corpus200 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_plus20260807_n60_corpus200 \
  --methods agentic_hybrid \
  --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_plus20260807_n60_zhipu_recursive_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_plus20260807_n60_agentic_hybrid_v1
```

只测 Light Entity：

```bash
uv run python scripts/evaluate_retrieval.py \
  --dataset multihop-rag-seed42-plus20260807-n60-corpus200 \
  --dataset-dir data/benchmarks/multihop_rag_seed42_plus20260807_n60_corpus200 \
  --methods agentic_light_entity \
  --chunking-strategy recursive \
  --runtime-dir data/lightrag/evaluation/multihop_seed42_plus20260807_n60_zhipu_recursive_v1 \
  --output-dir data/exports/evaluation/multihop_seed42_plus20260807_n60_agentic_light_entity_v1
```

## 配置

```dotenv
AGENTIC_MAX_RETRIEVAL_ROUNDS=2
AGENTIC_RRF_K=60
```

模型继续使用原配置：证据判断与查询改写使用 `DEEPSEEK_EXTRACTION_MODEL`，最终回答使用
`DEEPSEEK_CHAT_MODEL`。

## 当前边界

- 三个 `agentic_*` 变体目前都只作为离线评测方法。
- 没有 LangGraph Checkpointer 或跨问题对话记忆。
- 没有替换现有生产查询路径。
- 等 60 问 A/B 结果确认后，再决定是否新增 API/UI 模式 `agentic_mix`。
