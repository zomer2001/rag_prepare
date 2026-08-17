import numpy as np
import pytest

from rag_graph_demo.domain.tkg import (
    EntityRecord,
    ResolutionAction,
    SemanticEntityCandidate,
    TKGSnapshot,
)
from rag_graph_demo.services.entity_resolution_service import (
    EntityResolutionService,
    OntologyTypeMapper,
)


def mapper(tmp_path):
    path = tmp_path / "types.json"
    path.write_text(
        '{"Organization":"Organization","公司":"Organization","Model":"Model"}',
        encoding="utf-8",
    )
    return OntologyTypeMapper(path)


@pytest.mark.asyncio
async def test_type_alias_and_company_suffix_match_without_llm(tmp_path) -> None:
    llm_calls = 0

    async def embed(texts, context):
        del context
        return np.ones((len(texts), 2))

    async def complete(*_):
        nonlocal llm_calls
        llm_calls += 1
        return "{}"

    service = EntityResolutionService(mapper(tmp_path), embed, complete)
    snapshot = TKGSnapshot(
        ontology_version="0.1.0",
        entities=[
            EntityRecord(
                entity_id="ent-acme",
                canonical_name="Acme",
                entity_type="Organization",
                ontology_version="0.1.0",
            )
        ],
    )

    result = await service.resolve_batch(
        snapshot,
        [SemanticEntityCandidate(name="Acme, Inc.", entity_type="公司")],
    )

    assert result[0].action == ResolutionAction.MATCH_EXISTING
    assert result[0].selected_entity_id == "ent-acme"
    assert llm_calls == 0


@pytest.mark.asyncio
async def test_unknown_ontology_type_goes_to_review_not_new_entity(tmp_path) -> None:
    async def embed(texts, context):
        raise AssertionError((texts, context))

    async def complete(*_):
        raise AssertionError

    service = EntityResolutionService(mapper(tmp_path), embed, complete)

    result = await service.resolve_batch(
        TKGSnapshot(ontology_version="0.1.0"),
        [SemanticEntityCandidate(name="Acme", entity_type="Vendor")],
    )

    assert result[0].action == ResolutionAction.AMBIGUOUS
    assert result[0].normalized_type == "Concept"
    assert result[0].decided_by == "ontology"


@pytest.mark.asyncio
async def test_embedding_gray_zone_calls_constrained_llm(tmp_path) -> None:
    prompts = []

    async def embed(texts, context):
        del context
        assert len(texts) == 2
        return np.asarray([[1.0, 0.0], [0.82, 0.572]])

    async def complete(prompt, system):
        prompts.append((prompt, system))
        return (
            '{"action":"MATCH_EXISTING","selected_entity_id":"ent-acme",'
            '"confidence":0.88,"reason":"名称与组织描述一致"}'
        )

    service = EntityResolutionService(
        mapper(tmp_path),
        embed,
        complete,
        embedding_threshold=0.7,
        auto_match_threshold=0.95,
        llm_threshold=0.78,
    )
    snapshot = TKGSnapshot(
        ontology_version="0.1.0",
        entities=[
            EntityRecord(
                entity_id="ent-acme",
                canonical_name="ACME Holdings",
                entity_type="Organization",
                ontology_version="0.1.0",
            )
        ],
    )

    result = await service.resolve_batch(
        snapshot,
        [
            SemanticEntityCandidate(
                name="Acme Group",
                entity_type="Organization",
                description="An industrial holding company.",
            )
        ],
    )

    assert result[0].action == ResolutionAction.MATCH_EXISTING
    assert result[0].decided_by == "llm"
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_open_schema_can_link_across_extracted_types(tmp_path) -> None:
    async def embed(texts, context):
        del context
        assert len(texts) == 2
        return np.asarray([[1.0, 0.0], [1.0, 0.0]])

    async def complete(*_):
        raise AssertionError("high-confidence open-schema match should not call LLM")

    service = EntityResolutionService(
        mapper(tmp_path),
        embed,
        complete,
        enforce_type=False,
        auto_match_threshold=0.94,
    )
    snapshot = TKGSnapshot(
        ontology_version="0.1.0",
        entities=[
            EntityRecord(
                entity_id="ent-google",
                canonical_name="Google",
                entity_type="Organization",
                ontology_version="0.1.0",
            )
        ],
    )

    result = await service.resolve_batch(
        snapshot,
        [SemanticEntityCandidate(name="Google", entity_type="TechnologyCompany")],
    )

    assert result[0].action == ResolutionAction.MATCH_EXISTING
    assert result[0].selected_entity_id == "ent-google"
    assert result[0].normalized_type == "TechnologyCompany"
