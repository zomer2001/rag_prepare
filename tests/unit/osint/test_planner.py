from __future__ import annotations

from rag_graph_demo.osint.planner import TopicPlanner, extract_terms


def test_fallback_extracts_identifiers_and_proper_names() -> None:
    terms = extract_terms(
        [
            "North Star Systems announced contract ABC-12345 with the US Navy. ",
            "North Star Systems later demonstrated C-UAS equipment.",
        ],
        limit=10,
    )
    assert "ABC-12345" in terms
    assert any("North Star Systems" in item for item in terms)


def test_new_terms_are_deduplicated() -> None:
    assert TopicPlanner._new_terms(["Alpha"], ["alpha", " Beta ", "Beta"]) == ["Beta"]
