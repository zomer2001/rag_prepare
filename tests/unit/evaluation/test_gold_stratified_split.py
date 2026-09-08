from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "prepare_gold_stratified_split.py"
SPEC = importlib.util.spec_from_file_location("prepare_gold_stratified_split", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_proportional_quotas_use_largest_remainders() -> None:
    assert MODULE.proportional_quotas(
        {"comparison_query": 856, "inference_query": 816, "temporal_query": 583},
        500,
    ) == {
        "comparison_query": 190,
        "inference_query": 181,
        "temporal_query": 129,
    }


def test_selection_excludes_empty_gold_and_is_deterministic() -> None:
    rows = [
        {
            "question_type": "comparison_query" if index % 2 else "inference_query",
            "evidence_list": [{"title": f"doc-{index}"}],
        }
        for index in range(20)
    ]
    rows.append({"question_type": "null_query", "evidence_list": []})

    first, counts, quotas = MODULE.select_queries(rows, sample_size=10, seed=7)
    second, _, _ = MODULE.select_queries(rows, sample_size=10, seed=7)

    assert first == second
    assert counts == {"inference_query": 10, "comparison_query": 10}
    assert quotas == {"inference_query": 5, "comparison_query": 5}
    assert all(row["question_type"] != "null_query" for row in first)
    assert len({row["query_id"] for row in first}) == 10
