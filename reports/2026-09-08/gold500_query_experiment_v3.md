# Gold-500 query experiment v3

## Controlled internal methods

| Method | Initial retrieval | Initial budget | EvoLink | Final output |
|---|---|---:|---|---:|
| `naive` | one dense text query | 20 | no | 20 |
| `hybrid` | dense 20 + BM25 20 | 40 | no | 20 |
| `light_entity` | vector 20 + graph 20 | 40 | no | 20 |
| `ours` | vector 20 + graph 20 + event 20 | 60 | no | 20 |
| `ours_evolink` | same queries and route budgets as `ours` | 60 | one shared expansion, limit 20 | 20 |
| `ours_evoneed_equal` | at most three route-specific need queries | 60, equally split over active routes | one shared expansion, limit 20 | 20 |
| `ours_evoneed_dynamic` | the same route-specific need queries | 60, planner-proposed and deterministically normalized | one shared expansion, limit 20 | 20 |

`ours_evoneed_equal` and `ours_evoneed_dynamic` share the versioned joint-plan cache, so the budget allocator—not a different LLM decomposition—is the controlled difference. Every active route receives at least five candidates in the dynamic method. Main evaluation uses one retrieval round.

## External methods

HippoRAG, EventRAG-Hybrid, E2RAG-Vanilla, and GraphRAG-Local retain their native retrieval logic and return at most 20 final documents. Their native seed counts, graph expansion, LLM calls, and latency must be reported rather than described as a shared internal budget.

## Index policy

The LightRAG/entity, event, EvoLink, HippoRAG, EventRAG, and E2RAG indexes are reused unchanged. Query-result reset archives only the Gold-500 output directories and prior run log. GraphRAG has no completed Gold-500 index yet, so the run script prepares and completes that missing index before its retrieval stage.

## User-run command

```bash
bash scripts/run_all_methods_gold500.sh --fresh-results --run
```

`--fresh-results` moves existing Gold-500 query outputs into a timestamped backup under `reports/2026-09-08/`; it does not remove indexes.
