from fastapi import APIRouter, Depends

from rag_graph_demo.api.dependencies import get_container
from rag_graph_demo.domain.tkg import (
    EntityMergeRecord,
    EntityReviewCase,
    GovernanceStatus,
    MergeRollbackRequest,
    ReviewDecisionRequest,
    TKGQueryRequest,
    TKGQueryResponse,
    TKGSnapshot,
)
from rag_graph_demo.services.container import AppContainer

router = APIRouter()


@router.get("/tkg", response_model=TKGSnapshot)
async def get_tkg_snapshot(
    container: AppContainer = Depends(get_container),
) -> TKGSnapshot:
    return await container.tkg_registry.snapshot()


@router.post("/tkg/query", response_model=TKGQueryResponse)
async def query_tkg(
    request: TKGQueryRequest,
    container: AppContainer = Depends(get_container),
) -> TKGQueryResponse:
    if container.neo4j_store is not None:
        return await container.neo4j_store.query(request)
    return await container.tkg_registry.query(request)


@router.get("/tkg/graph")
async def query_tkg_graph(
    entity_name: str | None = None,
    limit: int = 200,
    container: AppContainer = Depends(get_container),
) -> dict[str, list[dict]]:
    if container.neo4j_store is None:
        return {"nodes": [], "edges": []}
    return await container.neo4j_store.query_graph(
        entity_name=entity_name,
        limit=max(1, min(limit, 500)),
    )


@router.get("/tkg/governance/reviews", response_model=list[EntityReviewCase])
async def list_entity_reviews(
    status: GovernanceStatus | None = GovernanceStatus.PENDING,
    container: AppContainer = Depends(get_container),
) -> list[EntityReviewCase]:
    return await container.entity_governance_service.list_reviews(status)


@router.post(
    "/tkg/governance/reviews/{review_id}/resolve",
    response_model=EntityReviewCase,
)
async def resolve_entity_review(
    review_id: str,
    request: ReviewDecisionRequest,
    container: AppContainer = Depends(get_container),
) -> EntityReviewCase:
    return await container.entity_governance_service.resolve_review(
        review_id,
        action=request.action,
        selected_entity_id=request.selected_entity_id,
        corrected_entity_type=request.corrected_entity_type,
        reviewer=request.reviewer,
        reason=request.reason,
    )


@router.post(
    "/tkg/governance/merges/{merge_id}/rollback",
    response_model=EntityMergeRecord,
)
async def rollback_entity_merge(
    merge_id: str,
    request: MergeRollbackRequest,
    container: AppContainer = Depends(get_container),
) -> EntityMergeRecord:
    return await container.entity_governance_service.rollback_merge(
        merge_id,
        actor=request.actor,
    )
