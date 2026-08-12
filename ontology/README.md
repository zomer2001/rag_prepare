# 静态领域本体 v0.1

本目录是声明级时序知识图谱的结构契约，不保存业务实例。

- `core/tkg-core.ttl`：Entity、Mention、Assertion、Event、EventEffect、ConflictCase 等通用概念。
- `domain/research-technology.ttl`：当前科研与技术资料场景的实体、关系和事件分类。
- `shapes/tkg-shapes.ttl`：入库前的 SHACL 最小约束。
- `mappings/*.json`：LLM/LightRAG 输出到本体术语的稳定映射。
- `competency-questions/v0.1.md`：本体必须支持回答的结构化问题。

核心边界：Mention 是某个来源中的文字证据；Assertion 是可独立管理有效期、立场和冲突的
声明；Event 是发生过的事项；EventEffect 解释事件如何创建、更新或终止 Assertion。
本体文件由 Git 版本化，实例记录保存其 `ontology_version`，升级时不隐式覆盖历史数据。

使用 OWL-Time、PROV-O、SKOS 和 Simple Event Model 的命名空间进行对齐；运行时验证不需要
联网下载外部词表。
