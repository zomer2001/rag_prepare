# Gold500 主线实验与 EvoLink 参数方案

日期：2026-09-11  
范围：2026 年 9 月 MultiHopRAG Gold500，500 题；不含 n30、2556 题和 GraphRAG。  
固定索引：609 documents、2512 chunks；最终返回 Top-20。

## 1. 只保留的四个主线实验

这四个实验构成一个清晰的 2×2 递进：分别控制“普通 Planner / Evidence-Need Planner”和“是否使用 EvoLink”。

| 方法 | 查询规划 | EvoLink | 状态 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 | 平均延迟 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Ours | 普通 QueryPlanner，固定三路 | 无 | 已完成 | 0.836 | 0.610 | 0.633 | 0.696 | 4.83s |
| Ours+EvoLink | 普通 QueryPlanner，固定三路 | 三路结果后共享一次 | 已完成 | 0.879 | 0.706 | 0.646 | 0.694 | 1.37s |
| OnePlan-NoLink | Evidence-Need Planner，按 Need 检索 | 无 | 已完成 | 0.841 | 0.640 | 0.623 | 0.649 | 0.71s |
| OnePlan-Shared | Evidence-Need Planner，按 Need 检索 | 所有 Need 后共享一次 | 已完成 | 0.900 | 0.766 | 0.637 | 0.645 | 0.92s |

主线叙事：

```text
普通 Planner + 三路检索
        ↓ 增加跨视图证据连接
普通 Planner + EvoLink
        ↓ 换成 Evidence-Need Planner
Evidence-Need Planner + EvoLink
```

对应的研究归因：

1. `Ours → Ours+EvoLink`：跨视图连接是否能补回遗漏的 Gold evidence。
2. `Ours+EvoLink → OnePlan-Shared`：证据需求规划是否比普通查询规划更适合多证据问题。
3. `OnePlan-NoLink → OnePlan-Shared`：OnePlan 的收益中有多少来自 EvoLink，而不是规划本身。

### 1.1 目前最重要的结果变化

- `Ours → Ours+EvoLink`：Evidence R@20 从 0.836 提升到 0.879，Complete R@20 从 0.610 提升到 0.706。
- `Ours → OnePlan-NoLink`：Complete R@20 从 0.610 提升到 0.640，但排序指标下降，说明需求规划单独带来了一些覆盖收益，还没有稳定改善排序。
- `OnePlan-NoLink → OnePlan-Shared`：Evidence R@20 从 0.841 提升到 0.900，Complete R@20 从 0.640 提升到 0.766，说明 EvoLink 对 OnePlan 的证据补全作用非常明显。
- `OnePlan-Shared` 相比 `OnePlan-NoLink` 的 MRR 基本持平（0.645 vs 0.649），主要收益在找全证据，而不是把第一条证据排得更靠前。

### 1.2 OnePlan-NoLink 按问题类型

| 问题类型 | Evidence R@20 | Complete R@20 | nDCG@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Comparison | 0.878 | 0.737 | 0.656 | 0.662 |
| Inference | 0.798 | 0.508 | 0.578 | 0.643 |
| Temporal | 0.845 | 0.682 | 0.638 | 0.639 |

## 2. 四个主线的预算口径

以 Top-20 为例：

| 方法 | 首轮检索 | EvoLink 扩展 | 最终输出 |
|---|---|---|---:|
| Ours | vector、graph、event 各请求 20 | 0 | 20 |
| Ours+EvoLink | 三路各请求 20 | 共享最多 20 | 20 |
| OnePlan-NoLink | 每个 Need 请求 20，最多 3 个 Need | 0 | 20 |
| OnePlan-Shared | 每个 Need 请求 20，最多 3 个 Need | 共享预算为 Need 预算之和，最多 60 | 20 |

因此，`OnePlan-Shared` 与 `Ours+EvoLink` 的首轮候选结构接近，但当前 EvoLink 扩展上限不同。论文中必须明确这一点，不能直接写成完全相同的总预算。

## 3. 只保留的备选消融

这些实验用于回答审稿人可能提出的具体问题，不放入主线叙事：

| 方法 | 保留原因 | 放置位置 |
|---|---|---|
| D1-QueryRewrite-SharedLink | 普通 QueryPlanner 的 route-specific query 与 OnePlan 的直接对照；共享 EvoLink 使扩展组织更接近最终版 | 主要消融/附录 |
| D1-Direct-NoPlan | 无 Planner 的最低控制，说明提升不是单纯来自原问题直接检索 | 附录 |

不再作为重点：

- `D1-QueryRewrite`：独立扩展和普通改写同时变化，解释不够干净。
- `Deep-OnePlan-D1-NoRerank`：独立扩展，不是当前最终 Shared 版本。
- `EvoNeed-Equal / Dynamic / Planned`：预算分配历史变体，和当前主线问题不直接对应。
- `EvoNeed-Deep`：含更深候选池和 LLM rerank，不能作为公平主线。

## 4. EvoLink 当前可调参数

当前默认配置位于 `Settings`：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `evolink_seed_limit_per_name` | 3 | 每个实体名从协调图取多少实体/断言 seed |
| `evolink_max_entity_names` | 8 | 一次最多使用多少个实体名作为 seed 来源 |
| `evolink_max_event_seeds` | 5 | 最多使用多少个事件候选作为扩展 seed |
| `evolink_expansion_limit` | 20 | 一次 EvoLink 返回的扩展候选上限 |
| `evolink_link_weight` | 0.1 | 扩展候选在最终融合中的权重 |

这些参数都属于查询阶段，不需要重建索引。

### 参数含义

- `seed_limit_per_name` 和 `max_entity_names` 控制实体/断言 seed 的宽度。增大可能提高召回，也可能把泛化实体和无关邻居带进来。
- `max_event_seeds` 控制事件视图参与扩展的数量，对时序题和演化题更敏感。
- `expansion_limit` 控制扩展结果规模，对 Complete Recall 和延迟影响最大。
- `link_weight` 只改变最终排序，不会增加扩展候选数；它主要影响 nDCG/MRR。

## 5. 面向时序与推理题只做两个参数实验

你的目标是提高时序题和推理题的证据指标，因此优先测“扩大可连接证据覆盖”的参数，而不是只改变排序加分：

| 参数 | 当前值 | 建议的唯一替代值 | 主要针对 |
|---|---:|---:|---|
| `evolink_seed_limit_per_name` | 3 | **5** | 推理题：每个实体名多取实体/断言 seed，增加跨实体事实连接。 |
| `evolink_max_event_seeds` | 5 | **8** | 时序题：允许更多事件候选进入 EvoLink，增加事件链和变更证据的覆盖。 |

建议每次只启用一个替代值，其他配置保持默认：

```text
Seed-Wide：seed_limit_per_name=5，max_event_seeds=5，其余默认
Event-Wide：seed_limit_per_name=3，max_event_seeds=8，其余默认
```

两个配置都必须同时用于 `Ours+EvoLink` 和 `OnePlan-Shared`。如果直接在 Gold500 全量运行，就是 2 个配置 × 2 个主线方法，共 4 个检索实验；不要做 3×3 的笛卡尔积。

### 为什么暂时不测另外三个参数

- `evolink_expansion_limit` 会直接改变扩展预算；而当前 `OnePlan-Shared` 显式使用 Need 总预算覆盖这个配置。它应当单独作为“预算曲线”实验，不能和参数调优混在一起。
- `evolink_link_weight` 只改变 document-RRF 排序，不会补回新的证据；如果后续主要优化 nDCG/MRR，再单独测 `0.05/0.20`。
- `evolink_max_entity_names` 通常受实体名数量和通用名过滤限制，不一定是当前瓶颈。

筛选标准按以下顺序：Complete Recall@20、Evidence Recall@20、nDCG@20、延迟。最终只能选一套统一配置，不能给 `Ours+EvoLink` 和 `OnePlan-Shared` 分别挑最优值，否则主线对比会失去公平性。

## 6. 一个重要的实现注意点

当前 `Ours+EvoLink` 使用 `evolink_expansion_limit` 配置；但 `OnePlan-Shared` 在代码中显式把扩展上限设为当前 Need 总预算。因此单纯修改 `EVOLINK_EXPANSION_LIMIT` 不一定会改变 `OnePlan-Shared` 的共享扩展规模。

如果要严格调节最终 OnePlan 的扩展上限，应增加一个明确的共享扩展 cap，例如：

```text
shared_expansion_limit = min(sum(need_budgets), configured_cap)
```

否则报告中应把当前 OnePlan 的扩展策略写成“按 Need 总预算共享”，而不是写成使用统一的 EvoLink 上限。

## 7. 推荐顺序

1. 四个主线已经齐全，先冻结当前默认结果。
2. 只跑 `Seed-Wide` 和 `Event-Wide` 两个配置，不做完整网格。
3. 选定统一参数后，只重跑 `Ours+EvoLink` 和 `OnePlan-Shared` 的确认结果。
4. 最后进入问答测评，不再为每个消融分别调参数。
