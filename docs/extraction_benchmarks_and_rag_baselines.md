# 实体、事件数据集与外部 RAG 基线

本文记录 2026-08-17 已准备的数据、校验方式和最小对比集合。大体量数据与第三方仓库位于
`data/benchmarks/`，受 `.gitignore` 管理，不进入项目提交。

## 本地数据

| 数据 | 本地位置 | 用途 | Gold 可用范围 |
| --- | --- | --- | --- |
| DWIE | `data/benchmarks/extraction/dwie/raw/` | 实体识别、文档级共指、关系抽取、实体链接 | 官方 train/test 标注；`annos_with_content` 含原文 |
| MAVEN | `data/benchmarks/extraction/maven/raw/` | 事件触发词识别与 168 类事件分类 | train/valid；test 不公开标签 |
| MAVEN-ERE | `data/benchmarks/extraction/maven_ere/raw/MAVEN_ERE/` | 事件共指、时间、因果和子事件关系 | train/valid；test 不公开关系标签 |

来源分别为 [DWIE 官方仓库](https://github.com/klimzaporojets/DWIE)、
[MAVEN 官方仓库](https://github.com/THU-KEG/MAVEN-dataset) 和
[MAVEN-ERE 官方仓库](https://github.com/THU-KEG/MAVEN-ERE)。使用数据前应遵循各自仓库的
许可证和引用要求。

统一校验命令：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_extraction_benchmarks.py
```

如果 DWIE 原文目录不完整，可断点续传并在下载后校验：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_extraction_benchmarks.py \
  --fetch-dwie-content
```

校验结果写入 `data/benchmarks/extraction/validation_summary.json`。校验覆盖 JSON/JSONL 读取、
DWIE 实体和关系引用、MAVEN 触发词 offset，以及 MAVEN-ERE 事件/时间节点和关系边引用。

本次全量校验结果为 `ok: true`：DWIE 802 篇原文全部可读，包含 43,373 个 Mention、
23,130 个实体和 21,749 条关系；MAVEN 的 2,913/710/857 篇 train/valid/test 均可读；
MAVEN-ERE 的相同三组文档以及 train/valid 中 1,039,406 条公开关系均通过端点引用校验。

## 实体部分采用的最小对比

实体实验不要把“抽取”和“跨文档合并”混成一个分数。建议只保留下列必要对比。

### 实体与关系抽取

1. **DWIE Joint BERT**：DWIE 官方 `13_joint_bert` 配置，作为公开任务基线。
2. **LightRAG 默认抽取**：说明未经约束的当前上游能做到什么程度。
3. **受控语义抽取**：即当前 `light_entity`/语义通路的实体—关系约束，作为本系统方法。

采用 DWIE 官方评测脚本，报告 Mention-level F1、Hard/Soft Entity-level F1；共指报告
MUC、B³、CEAF-e 与 CoNLL 平均 F1；关系报告 Mention-level 与 Hard/Soft Entity-level F1。

### 实体合并与解析

1. **Exact + Alias**：规范化名称和别名精确匹配，是必须有的规则下界。
2. **Embedding**：只按向量相似度选候选或聚类，检验语义匹配本身的贡献。
3. **Ours Full**：类型约束 + 名称/别名 + 字符串相似度 + Embedding 候选 + LLM 裁决。

再增加一个关键消融即可：将 Ours Full 中的 LLM 裁决去掉。这样总量仍然可控，同时能回答
性能来自规则、Embedding，还是最终裁决。没有必要在首轮同时复现大量通用 NER 模型。

## RAG 对比集合与可行性

外部对比可以采用 GraphRAG、LightRAG、HippoRAG 2、EventRAG；但内部仍必须保留 `ours` 和
`ours_evolink`。`ours` 与 `ours_evolink` 的差异最小，能够单独回答跨通路 EvoLink 扩展是否有效。

| 方法 | 当前状态 | 主要适配工作 | 难度 |
| --- | --- | --- | --- |
| LightRAG | 已集成并有 `light` 方法 | 固定语料、切分、模型与 Top-K | 低 |
| HippoRAG 2 | 官方源码已下载，n30 数据已成功转换 | 独立 Python 3.10 环境；绕开 macOS 不可用的 vLLM 依赖；接入同一 LLM/Embedding；导出逐题排序 | 中 |
| GraphRAG | 尚未接入 | 独立索引；把 local/global context 映射回统一文档 ID；记录索引调用与 token 成本 | 中高 |
| EventRAG | 第一作者仓库已下载 | 独立 Python 3.12 环境；修改硬编码路径；保存 source ID；将 agent context 转成统一排序结果 | 中高 |

比较时必须统一：语料版本、问题集合、文档 ID、Top-K、生成模型和评测脚本。各方法可以保留其
原生索引和检索机制，但不能只比较各自论文中的数字。

## HippoRAG 2 复现结论

[HippoRAG 官方仓库](https://github.com/OSU-NLP-Group/HippoRAG) 已下载至
`data/benchmarks/_downloads/baselines/hipporag/`。当前 30 题、100 篇语料已转换成：

- `reproduce/dataset/ours_multihop_n30_corpus.json`
- `reproduce/dataset/ours_multihop_n30.json`

转换命令：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/export_hipporag_dataset.py \
  --dataset-dir data/benchmarks/multihop_rag_seed42_n30_corpus100 \
  --output-dir data/benchmarks/_downloads/baselines/hipporag/reproduce/dataset \
  --name ours_multihop_n30
```

转换结果为 100 篇文档、30 个问题、59 条 Gold Evidence，全部按 URL 或标题映射成功。
HippoRAG 原生 `retrieve()` 会返回有序文档和分数，因此接入统一检索指标并不困难。

真正的阻力在环境，而不在数据格式。官方包固定依赖 `vllm==0.6.6.post1`，该版本在当前
Apple Silicon macOS 上没有可用 wheel；直接 `pip install hipporag` 不能得到可复现环境。
当前官方版本已经支持 OpenAI-compatible LLM 和 Embedding endpoint，但默认只读取一个
`OPENAI_API_KEY`，而且不会自动把智谱 `embedding-3` 路由到该实现。本项目因此使用独立在线
环境和一个只处理双 API key、1024 维参数及 DeepSeek thinking 开关的适配器，不改检索算法。
可选方案有两种：

1. 在 Linux/CUDA 机器或容器中按官方 Python 3.10 环境运行；最接近论文配置。
2. 在本机使用独立 Python 3.10 环境，安装源码与在线模式所需依赖但排除 vLLM，使用
   DeepSeek 和智谱服务。该方案已完成3题、77 Chunk 的小规模验证；运行方式和结果见
   `docs/hipporag_baseline.md`。

正式对比建议采用第二种方案先跑 30 题 smoke test，确认逐题文档排序和指标能进入本项目结果
格式；再决定是否在 Linux/GPU 上跑完整数据。索引会调用 LLM 做 OpenIE，不能把它当成零成本
向量基线，必须单独记录索引调用数、token 数、时间和磁盘占用。

## EventRAG 说明

[EventRAG 论文](https://aclanthology.org/2025.acl-long.830/)已有作者源码，仓库为
[Ryaang/EventRAG](https://github.com/Ryaang/EventRAG)，本地位于
`data/benchmarks/_downloads/baselines/eventrag/`。代码支持 `only_need_context`，并在实体和关系上
保留 `source_id`，所以可以做检索证据评测，而不必只比较最终答案。

不过其论文复现脚本固定使用 GPT-4o、Neo4j、Milvus 和 `../RAG-Data` 路径，输出也主要面向答案
评判。接入时需要保留原方法逻辑，只替换数据读取、路径、模型配置和逐题结果导出。它适合作为
事件图 RAG 的直接外部对比，但首轮优先级应排在 LightRAG 和 HippoRAG 之后。
