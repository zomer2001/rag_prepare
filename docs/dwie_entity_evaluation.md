# DWIE 实体测评

这套测评只检查当前系统的实体部分，不运行事件抽取，也不评价最终检索效果。

## 评测范围

当前系统与 DWIE 能直接对齐的公共粗粒度类型为：

- Person
- Organization
- Location

DWIE 的时间、金额、事件等类型不进入本轮评测。当前语义通路本来就不应把事件作为实体，
直接按 DWIE 全部 311 个多标签类型计算一个总分会混淆方法边界。

评测分为两条：

1. **端到端抽取**：从原文抽取实体名，对齐回字符位置，再评价 Mention、实体类型和共指簇。
2. **Gold Mention 合并**：所有方法获得相同的 Gold Mention 与粗粒度类型，只评价实体合并。

Exact+Alias 与 `ours_entity/gold_mentions` 属于第二条；LightRAG 与
`ours_entity/end_to_end` 属于第一条，不能把两条结果放在同一列直接排名。

## 方法

| 方法 | 上游 Mention | 实体合并方式 |
| --- | --- | --- |
| DWIE Official Joint BERT | 模型预测 | 官方联合 NER、共指、关系模型；需自行训练或提供预测 |
| All Singleton | Gold | 每个 Mention 都视为独立实体 |
| Exact Match | Gold | 仅合并类型和表面字符串完全相同的 Mention |
| Exact+Alias | Gold | 大小写/标点归一、组织后缀签名、全名缩写 |
| LightRAG | DeepSeek 抽取 | LightRAG 默认同名图节点 |
| `light_entity` | DeepSeek 受控抽取 | 与 ours 共用抽取索引，不运行后续治理 |
| `ours_no_llm` | Gold | Exact/Alias、字符串候选、智谱 Embedding，不调用LLM |
| `ours_entity/end_to_end` | DeepSeek 受控抽取 | Exact/Alias、字符串候选、智谱 Embedding、DeepSeek 裁决 |
| `ours_entity/gold_mentions` | Gold | 与上一行相同，只隔离评价实体合并 |

当前代码中的 `light_entity` 是 `ours_entity` 的受控抽取上游，不单独作为主基线。它限制每个
Chunk 最多四个关键实体和四条关键关系；完整 `ours_entity` 在其后增加类型归一、候选生成、
Embedding、LLM 裁决和统一 Entity ID。

## 准备固定20篇样本

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/prepare_dwie_entity_benchmark.py \
  --sample-size 20 \
  --seed 20260817
```

样本位于：

```text
data/benchmarks/extraction/dwie_test_seed20260817_n20/
```

## 运行

先运行不调用 API 的 Exact+Alias，并登记官方方法状态：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --methods official,exact_alias
```

运行 DeepSeek + 智谱 API 方法：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --methods lightrag,ours_entity
```

要求 `.env` 中至少存在：

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_EXTRACTION_MODEL=...
EMBEDDING_PROVIDER=zhipu
ZHIPU_API_KEY=...
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSION=1024
```

同时需要 Qdrant 正常运行。索引和语义抽取有缓存，失败后可直接重跑相同命令。
DWIE 是英文语料，运行器会固定 `LIGHTRAG_SUMMARY_LANGUAGE=English`；语言、LLM、Embedding、
维度和治理阈值都写入运行指纹，避免错误复用其他配置的索引。

## 输出

默认输出到：

```text
data/benchmarks/extraction/dwie_test_seed20260817_n20/results/<run_digest>/
```

其中：

- `summary.json`：汇总指标和评测轨道说明；
- `predictions/`：逐文档 DWIE 格式预测；

索引、语义抽取与逐文档实体治理断点位于：

```text
data/benchmarks/extraction/dwie_test_seed20260817_n20/runtime/<run_digest>/
```

同一运行指纹可以直接重跑；已完成的索引、抽取和逐文档治理结果会复用。不同模型、语言、
Embedding 或治理阈值会产生新的 `run_digest`，不会混用缓存。

指标包括 Mention P/R/F1、Soft/Hard Entity P/R/F1、MUC、B³、CEAF-e 和 CoNLL F1。
Gold Mention 轨道的 Mention F1 固定为1，只看 Hard Entity 与共指指标。

## 官方模型的限制

DWIE 官方仓库提供 `13_joint_bert` 的训练配置和评测代码，但没有发布可直接下载的预训练
checkpoint。因此本工具不会伪造“官方方法已复现”的结果。取得官方预测后可运行：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --methods official \
  --official-predictions /path/to/official/predictions
```

## 当前20篇样本结果

运行指纹：`77bfbe86b387`。

端到端轨道：

| 方法 | 预测 Mention | Mention P/R/F1 | Soft Entity F1 | Hard Entity F1 | CoNLL F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| LightRAG | 631 | 0.821 / 0.719 / 0.767 | 0.760 | 0.566 | 0.717 |
| `ours_entity/end_to_end` | 284 | 0.951 / 0.375 / 0.538 | 0.366 | 0.243 | 0.479 |

Gold Mention 合并轨道：

| 方法 | Hard Entity P/R/F1 | MUC F1 | B³ F1 | CEAF-e F1 | CoNLL F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact+Alias | 0.661 / 0.800 / 0.724 | 0.921 | 0.919 | 0.856 | 0.899 |
| `ours_entity/gold_mentions` | 0.905 / 0.867 / 0.885 | 0.973 | 0.963 | 0.944 | 0.960 |

这组小样本支持两个分开的判断：

1. 当前实体治理有效。使用完全相同的 Gold Mention 时，`ours_entity` 的 Hard Entity F1 比
   Exact+Alias 高 0.161，CoNLL F1 高 0.061。
2. 当前端到端短板在抽取召回。受控抽取每个 Chunk 最多保留四个实体，使 Precision 达到
   0.951，但 Recall 只有 0.375；因此现阶段不能宣称端到端优于 LightRAG。

20篇随机样本只用于检查方法完成度和定位问题，不作为论文最终结论。正式实验应扩大样本，
并对实体配额、Chunk 长度和治理阈值做开发集选择。

## 固定100篇实验

生成测试集固定子集：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/prepare_dwie_entity_benchmark.py \
  --sample-size 100 --seed 20260817
```

推荐分阶段运行。每次结果会合并到同一 `summary.json`，相同运行指纹会复用索引和逐文档
实体治理断点。

```bash
# 不调用API的Gold Mention规则基线与官方状态
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --sample-dir data/benchmarks/extraction/dwie_test_seed20260817_n100 \
  --methods official,all_singleton,exact_match,exact_alias

# 两种端到端抽取；light_entity索引会被ours_entity复用
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --sample-dir data/benchmarks/extraction/dwie_test_seed20260817_n100 \
  --methods lightrag,light_entity

# Gold Mention：去掉LLM裁决的消融
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --sample-dir data/benchmarks/extraction/dwie_test_seed20260817_n100 \
  --methods ours_no_llm

# 完整ours：同时输出端到端和Gold Mention实体合并结果
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_dwie_entities.py \
  --sample-dir data/benchmarks/extraction/dwie_test_seed20260817_n100 \
  --methods ours_entity
```
