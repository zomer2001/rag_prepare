from __future__ import annotations

import pytest

from rag_graph_demo.services.semantic_extraction_service import LLMSemanticExtractor


@pytest.mark.asyncio
async def test_semantic_extractor_returns_tkg_candidates_without_graph_store(tmp_path) -> None:
    prompt = tmp_path / "entity.yaml"
    prompt.write_text("Only stable facts.", encoding="utf-8")
    calls = []

    async def complete(user: str, system: str) -> str:
        calls.append((user, system))
        return """{
          "entities": [
            {"name":"Acme","type":"Organization","description":"A company"},
            {"name":"Lattice","type":"Model","description":"A model"}
          ],
          "relationships": [
            {"source":"Acme","target":"Lattice","predicate":"develops",
             "description":"Acme develops Lattice.","confidence":0.9}
          ]
        }"""

    result = await LLMSemanticExtractor(
        complete,
        prompt_file=prompt,
    ).extract("Report", [("chunk-1", "Acme develops Lattice.")])

    assert len(calls) == 1
    assert {item.name for item in result.entities} == {"Acme", "Lattice"}
    assert result.relationships[0].source_segment_id == "chunk-1"


@pytest.mark.asyncio
async def test_semantic_extractor_rejects_relationship_to_unretained_entity(tmp_path) -> None:
    prompt = tmp_path / "entity.yaml"
    prompt.write_text("Only stable facts.", encoding="utf-8")

    async def complete(_user: str, _system: str) -> str:
        return """{
          "entities": [{"name":"Acme","type":"Organization"}],
          "relationships": [{"source":"Acme","target":"Ghost","predicate":"uses",
            "description":"unsupported endpoint"}]
        }"""

    result = await LLMSemanticExtractor(
        complete,
        prompt_file=prompt,
    ).extract("Report", [("chunk-1", "Text")])

    assert result.relationships == []
