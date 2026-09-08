from __future__ import annotations

from rag_graph_demo.config import Settings
from rag_graph_demo.ingestion.parser import DocumentParser
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.lightrag_engine.local_models import embedding_model_name
from rag_graph_demo.retrieval.event_vector import EventVectorIndex
from rag_graph_demo.retrieval.orchestrator import UnifiedRetrievalOrchestrator
from rag_graph_demo.retrieval.planner import QueryPlanner
from rag_graph_demo.services.document_service import DocumentService
from rag_graph_demo.services.entity_governance_service import EntityGovernanceService
from rag_graph_demo.services.entity_resolution_service import (
    EntityResolutionService,
    OntologyTypeMapper,
)
from rag_graph_demo.services.event_service import EventService, LLMEventExtractor
from rag_graph_demo.services.ontology_service import OntologyService
from rag_graph_demo.services.query_service import QueryService
from rag_graph_demo.services.tkg_fusion_service import TKGFusionService
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.event_registry import EventRegistry
from rag_graph_demo.storage.file_store import FileStore
from rag_graph_demo.storage.job_registry import JobRegistry
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore
from rag_graph_demo.storage.tkg_registry import TKGRegistry


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_data_dirs()
        self.ontology_service = OntologyService(settings.tkg_ontology_dir)
        if settings.tkg_enabled and settings.tkg_validate_shacl:
            # Fail fast on a damaged or syntactically invalid versioned schema.
            self.ontology_service.load_schema()
        self.documents_registry = DocumentRegistry(
            settings.data_dir / "registry" / "documents.json"
        )
        self.jobs_registry = JobRegistry(settings.data_dir / "registry" / "jobs.json")
        self.file_store = FileStore(settings.data_dir)
        self.source_mapper = SourceMapper(self.documents_registry)
        # The application system is "ours": controlled LightRAG entities plus events.
        self.engine = LightRAGEngine(
            settings,
            self.source_mapper,
            extraction_profile="light_entity",
        )
        self.events_registry = EventRegistry(
            settings.data_dir / "events" / "event_store.json"
        )
        event_extractor = (
            LLMEventExtractor(
                self.engine,
                chunk_chars=settings.event_extraction_chunk_chars,
                max_events=settings.event_extraction_max_events,
                max_events_per_part=settings.event_extraction_max_events_per_part,
            )
            if settings.event_extraction_enabled
            else None
        )
        self.event_service = EventService(self.events_registry, event_extractor)
        self.neo4j_store = (
            Neo4jTKGStore(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password,
                database=settings.neo4j_database,
                workspace_id=settings.neo4j_workspace,
            )
            if settings.neo4j_enabled
            else None
        )
        self.tkg_registry = TKGRegistry(
            settings.tkg_registry_path or settings.data_dir / "tkg" / "tkg_store.json",
            ontology_version=settings.tkg_ontology_version,
            validate_snapshot=(
                self.ontology_service.validate_snapshot
                if settings.tkg_validate_shacl
                else None
            ),
        )
        self.tkg_service = TKGFusionService(
            self.tkg_registry,
            ontology_version=settings.tkg_ontology_version,
            entity_resolver=(
                EntityResolutionService(
                    OntologyTypeMapper(settings.entity_type_mapping_file),
                    self.engine.embed_texts,
                    self.engine.complete_extraction,
                    candidate_limit=settings.entity_resolution_candidate_limit,
                    embedding_threshold=(
                        settings.entity_resolution_embedding_threshold
                    ),
                    auto_match_threshold=(
                        settings.entity_resolution_auto_match_threshold
                    ),
                    llm_threshold=settings.entity_resolution_llm_threshold,
                )
                if settings.entity_resolution_enabled
                else None
            ),
            neo4j_store=self.neo4j_store,
            event_registry=self.events_registry,
        )
        self.entity_governance_service = EntityGovernanceService(
            self.tkg_registry,
            neo4j_store=self.neo4j_store,
            event_registry=self.events_registry,
        )
        self.query_planner = QueryPlanner(
            self.engine.complete_extraction,
            cache_path=settings.data_dir / "events" / "query_plan_cache.json",
        )
        self.event_vector_index = EventVectorIndex(
            settings.data_dir / "events" / "event_vectors.npz",
            self.engine.embed_texts,
            embedding_identity=(
                f"{settings.embedding_provider}/"
                f"{embedding_model_name(settings)}/{settings.embedding_dimension}d"
            ),
            batch_size=settings.embedding_batch_size,
            max_concurrency=settings.effective_embedding_concurrency,
        )
        self.retrieval_orchestrator = UnifiedRetrievalOrchestrator(
            self.engine,
            self.query_planner,
            event_service=self.event_service,
            event_vector_index=self.event_vector_index,
            tkg_registry=(
                self.neo4j_store or self.tkg_registry
                if settings.tkg_enabled
                else None
            ),
            coordination_store=(
                self.neo4j_store
                if settings.tkg_enabled and self.neo4j_store is not None
                else None
            ),
            evolink_seed_limit_per_name=settings.evolink_seed_limit_per_name,
            evolink_max_entity_names=settings.evolink_max_entity_names,
            evolink_max_event_seeds=settings.evolink_max_event_seeds,
            evolink_expansion_limit=settings.evolink_expansion_limit,
            evolink_link_weight=settings.evolink_link_weight,
        )
        self.document_service = DocumentService(
            self.file_store,
            self.documents_registry,
            self.jobs_registry,
            DocumentParser(ocr_enabled=settings.parser_ocr_enabled),
            self.engine,
            self.event_service if settings.event_extraction_enabled else None,
            self.tkg_service if settings.tkg_enabled else None,
        )
        self.query_service = QueryService(
            self.engine,
            self.event_service,
            self.documents_registry,
            self.retrieval_orchestrator,
        )

    async def shutdown(self) -> None:
        await self.engine.finalize()
        if self.neo4j_store is not None:
            await self.neo4j_store.close()
