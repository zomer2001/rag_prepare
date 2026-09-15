from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from rag_graph_demo.agentic.evoneed import EvoNeedRAGGraph
from rag_graph_demo.agentic.graph import AgenticRAGGraph
from rag_graph_demo.agentic.model import DeepSeekAgentModel
from rag_graph_demo.agentic.models import AgenticRun, NeedItem
from rag_graph_demo.agentic.prompts import (
    AGENTIC_PROMPT_PROFILE_VERSION,
    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
    EVONEED_DEEP_ONEPLAN_SYSTEM,
    EVONEED_DEEP_PLAN_SYSTEM,
    EVONEED_DEEP_PROMPT_PROFILE_VERSION,
    EVONEED_PROMPT_PROFILE_VERSION,
    evoneed_deep_oneplan_prompt_profile_digest,
    evoneed_deep_prompt_profile_digest,
    evoneed_prompt_profile_digest,
    prompt_profile_digest,
)
from rag_graph_demo.config import Settings
from rag_graph_demo.evaluation.chunking import ChunkingStrategy, build_chunks
from rag_graph_demo.evaluation.metrics import aggregate_results, query_metrics
from rag_graph_demo.evaluation.models import (
    BenchmarkChunk,
    BenchmarkDataset,
    BenchmarkDocument,
    RetrievedChunk,
)
from rag_graph_demo.evaluation.retrievers import (
    DenseRetriever,
    HybridRetriever,
    document_id_from_file_path,
    light_data_to_chunks,
)
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.lightrag_engine.local_models import (
    build_embedding_func,
    embedding_model_name,
)
from rag_graph_demo.retrieval.event_vector import EventVectorIndex
from rag_graph_demo.retrieval.fusion import fuse_evidence
from rag_graph_demo.retrieval.models import (
    QueryPlan,
    RetrievalChannel,
    UnifiedRetrievalResult,
)
from rag_graph_demo.retrieval.orchestrator import UnifiedRetrievalOrchestrator
from rag_graph_demo.retrieval.planner import (
    QUERY_PLANNER_PROFILE_VERSION,
    TKG_PLANNER_PROMPT,
    TKG_QUERY_PLANNER_PROFILE_VERSION,
    QueryPlanner,
)
from rag_graph_demo.retrieval.tkg_retriever import UnifiedTKGRetriever
from rag_graph_demo.retrieval.tkg_vector import TKGVectorIndex, tkg_vector_fingerprint
from rag_graph_demo.services.entity_resolution_service import (
    EntityResolutionService,
    OntologyTypeMapper,
)
from rag_graph_demo.services.event_service import (
    EVENT_EXTRACTION_PROFILE_VERSION,
    EventService,
    LLMEventExtractor,
)
from rag_graph_demo.services.ontology_service import OntologyService
from rag_graph_demo.services.semantic_extraction_service import (
    SEMANTIC_EXTRACTION_PROFILE_VERSION,
    LLMSemanticExtractor,
)
from rag_graph_demo.services.tkg_fusion_service import TKGFusionService
from rag_graph_demo.storage.atomic_json import read_json, write_json_atomic
from rag_graph_demo.storage.document_registry import DocumentRegistry
from rag_graph_demo.storage.event_registry import EventRegistry
from rag_graph_demo.storage.neo4j_tkg_store import Neo4jTKGStore
from rag_graph_demo.storage.tkg_registry import TKGRegistry

logger = logging.getLogger(__name__)
CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER = 1

VALID_METHODS = (
    "naive",
    "hybrid",
    "light",
    "light_entity",
    "ours",
    "ours_evolink",
    "ours_association_only",
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
    "ours_d1_direct_noplan",
    "ours_d1_queryrewrite",
    "ours_d1_queryrewrite_sharedlink",
    "ours_d1_queryrewrite_sharedlink_total",
    "ours_tkg",
    "ours_tkg_dynamic",
    "agentic_hybrid",
    "agentic_light_entity",
    "agentic_ours",
    "agentic_ours_evolink",
    "agentic_ours_tkg",
    "agentic_ours_tkg_dynamic",
)
AGENTIC_METHODS = {
    "agentic_hybrid",
    "agentic_light_entity",
    "agentic_ours",
    "agentic_ours_evolink",
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
    "agentic_ours_tkg",
    "agentic_ours_tkg_dynamic",
}
CONTROLLED_METHODS = {
    "light_entity",
    "ours",
    "ours_evolink",
    "ours_association_only",
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
    "ours_d1_direct_noplan",
    "ours_d1_queryrewrite",
    "ours_d1_queryrewrite_sharedlink",
    "ours_d1_queryrewrite_sharedlink_total",
    "agentic_light_entity",
    "agentic_ours",
    "agentic_ours_evolink",
}
EVENT_METHODS = {
    "ours",
    "ours_evolink",
    "ours_association_only",
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
    "ours_d1_direct_noplan",
    "ours_d1_queryrewrite",
    "ours_d1_queryrewrite_sharedlink",
    "ours_d1_queryrewrite_sharedlink_total",
    "agentic_ours",
    "agentic_ours_evolink",
}
EVOLINK_METHODS = {
    "ours_evolink",
    "ours_association_only",
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
    "ours_d1_direct_noplan",
    "ours_d1_queryrewrite",
    "ours_d1_queryrewrite_sharedlink",
    "ours_d1_queryrewrite_sharedlink_total",
    "agentic_ours_evolink",
}
EVONEED_METHODS = {
    "ours_evoneed",
    "ours_evoneed_planned",
    "ours_evoneed_planned_shared",
    "ours_evoneed_equal",
    "ours_evoneed_dynamic",
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
}
D1_METHODS = {
    "ours_d1_direct_noplan",
    "ours_d1_queryrewrite",
    "ours_d1_queryrewrite_sharedlink",
    "ours_d1_queryrewrite_sharedlink_total",
}
DEEP_EVONEED_METHODS = {
    "ours_evoneed_deep",
    "ours_evoneed_deep_oneplan",
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
}
DEEP_RERANK_METHODS = DEEP_EVONEED_METHODS - {
    "ours_evoneed_deep_oneplan_d1_norerank",
    "ours_evoneed_deep_oneplan_nolink_norerank",
    "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
    "ours_evoneed_deep_oneplan_shared_total_norerank",
}
STANDARD_EVONEED_METHODS = EVONEED_METHODS - DEEP_EVONEED_METHODS
TKG_METHODS = {
    "ours_tkg",
    "ours_tkg_dynamic",
    "agentic_ours_tkg",
    "agentic_ours_tkg_dynamic",
}
DENSE_METHODS = {"naive", "hybrid", "agentic_hybrid"}
HYBRID_METHODS = {"hybrid", "agentic_hybrid"}
UNIFIED_RETRIEVAL_PROFILE_VERSION = "three_path_v3_route_top20"
EVOLINK_RETRIEVAL_PROFILE_VERSION = "ours_evolink_v3_single_shared_expansion"
TKG_RETRIEVAL_PROFILE_VERSION = "unified_statement_tkg_v1"
LIGHTRAG_BENCHMARK_INPUT_VERSION = "unique_chunk_identity_v1"


class EvaluationRunner:
    def __init__(
        self,
        dataset: BenchmarkDataset,
        settings: Settings,
        *,
        runtime_dir: Path,
        output_dir: Path,
        methods: tuple[str, ...],
        cutoffs: tuple[int, ...] = (4, 10),
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        chunking_strategy: ChunkingStrategy = "fixed",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
        index_batch_size: int = 16,
        fail_fast: bool = False,
        show_progress: bool = True,
    ) -> None:
        unknown = set(methods) - set(VALID_METHODS)
        if unknown:
            raise ValueError(f"Unknown evaluation methods: {sorted(unknown)}")
        if not methods:
            raise ValueError("At least one method is required")
        self.dataset = dataset
        self.base_settings = settings
        self.runtime_dir = runtime_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.methods = methods
        self.cutoffs = tuple(sorted(set(cutoffs)))
        self.top_k = max(self.cutoffs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.query_limit = query_limit
        self.corpus_limit = corpus_limit
        self.index_batch_size = index_batch_size
        self.fail_fast = fail_fast
        self.show_progress = show_progress
        documents = dataset.documents[:corpus_limit] if corpus_limit else dataset.documents
        self.documents = tuple(documents)
        self.queries = dataset.queries[:query_limit] if query_limit else dataset.queries
        self.chunks: list[BenchmarkChunk] = build_chunks(
            self.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        self.fingerprint = benchmark_fingerprint(
            dataset.name,
            self.documents,
            self.chunks,
            embedding_identity(settings),
            chunk_size,
            chunk_overlap,
            chunking_strategy,
        )
        self.lightrag_fingerprint = lightrag_benchmark_fingerprint(self.fingerprint)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_settings = Settings(
            _env_file=None,
            **{
                **settings.model_dump(),
                "data_dir": self.runtime_dir,
                "lightrag_working_dir": self.runtime_dir / "lightrag",
                "lightrag_workspace": (
                    f"eval_{safe_name(dataset.name)}_{self.lightrag_fingerprint[:10]}"
                ),
                "chunk_target_tokens": chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
                "lightrag_chunk_top_k": self.top_k,
            },
        )
        self.eval_settings.ensure_data_dirs()
        self.light_entity_settings = Settings(
            _env_file=None,
            **{
                **self.eval_settings.model_dump(),
                "lightrag_working_dir": self.runtime_dir / "lightrag_entity",
                "lightrag_workspace": (
                    f"eval_entity_{safe_name(dataset.name)}_{self.lightrag_fingerprint[:10]}"
                ),
            },
        )
        self.light_entity_fingerprint = light_entity_fingerprint(
            self.lightrag_fingerprint,
            self.light_entity_settings,
        )
        self.light_entity_settings.lightrag_workspace = (
            f"eval_entity_{safe_name(dataset.name)}_{self.light_entity_fingerprint[:10]}"
        )
        self.event_fingerprint = hashlib.sha256(
            (
                f"{self.light_entity_fingerprint}\x1f"
                f"{EVENT_EXTRACTION_PROFILE_VERSION}\x1f"
                f"{self.eval_settings.event_extraction_chunk_chars}\x1f"
                f"{self.eval_settings.event_extraction_max_events}\x1f"
                f"{self.eval_settings.event_extraction_max_events_per_part}"
            ).encode()
        ).hexdigest()
        self.light_engine: LightRAGEngine | None = None
        self.light_entity_engine: LightRAGEngine | None = None
        self.event_service: EventService | None = None
        self.query_planner: QueryPlanner | None = None
        self.unified_orchestrator: UnifiedRetrievalOrchestrator | None = None
        self.agentic_graphs: dict[str, AgenticRAGGraph | EvoNeedRAGGraph] = {}
        self.tkg_retriever: UnifiedTKGRetriever | None = None
        self.tkg_dynamic_retriever: UnifiedTKGRetriever | None = None
        self.tkg_vector_index: TKGVectorIndex | None = None
        self.tkg_graph_store: Neo4jTKGStore | None = None
        self.evolink_graph_store: Neo4jTKGStore | None = None
        self.tkg_extraction_engine: LightRAGEngine | None = None
        self.file_path_to_document: dict[str, str] = {}

    async def run(self) -> dict[str, Any]:
        started = perf_counter()
        self._ensure_output_manifest()

        dense: DenseRetriever | None = None
        hybrid: HybridRetriever | None = None
        if DENSE_METHODS & set(self.methods):
            dense_embedding = build_embedding_func(self.eval_settings)

            async def embed(texts: list[str], context: str):
                return await dense_embedding(texts, context=context)

            dense = DenseRetriever(
                self.chunks,
                embed,
                cache_dir=self.runtime_dir / "dense",
                batch_size=self.eval_settings.embedding_batch_size,
                max_concurrency=self.eval_settings.effective_embedding_concurrency,
                show_progress=self.show_progress,
            )
            await dense.initialize(self.fingerprint)
        if HYBRID_METHODS & set(self.methods):
            assert dense is not None
            hybrid = HybridRetriever(dense)

        if {"light", *CONTROLLED_METHODS, *AGENTIC_METHODS, *TKG_METHODS} & set(
            self.methods
        ):
            self._require_light_environment()
        if "light" in self.methods:
            registry = DocumentRegistry(self.runtime_dir / "registry" / "light_documents.json")
            self.light_engine = LightRAGEngine(
                self.eval_settings,
                SourceMapper(registry),
                # LightRAG deep-copies its configuration during initialization.
                # Never reuse an EmbeddingFunc whose API client may already have
                # been initialized by another index: AsyncOpenAI owns an RLock
                # which cannot be deep-copied.
                embedding_func=build_embedding_func(self.eval_settings),
            )
            await self._ensure_light_index(
                self.light_engine,
                label="light",
                marker=self.runtime_dir / "lightrag" / "evaluation_index.json",
                fingerprint=self.lightrag_fingerprint,
            )
        if CONTROLLED_METHODS & set(self.methods):
            registry = DocumentRegistry(
                self.runtime_dir / "registry" / "light_entity_documents.json"
            )
            self.light_entity_engine = LightRAGEngine(
                self.light_entity_settings,
                SourceMapper(registry),
                embedding_func=build_embedding_func(self.light_entity_settings),
                extraction_profile="light_entity",
            )
            await self._ensure_light_index(
                self.light_entity_engine,
                label="light_entity",
                marker=(self.runtime_dir / "lightrag_entity" / "evaluation_index.json"),
                fingerprint=self.light_entity_fingerprint,
            )
        if EVENT_METHODS & set(self.methods):
            await self._ensure_event_index()
        if CONTROLLED_METHODS & set(self.methods):
            assert self.light_entity_engine is not None
            planner = QueryPlanner(
                self.light_entity_engine.complete_extraction,
                cache_path=self.runtime_dir / "events" / "query_plan_cache.json",
                strict_failures=True,
            )
            self.query_planner = planner
            event_vector = EventVectorIndex(
                self.runtime_dir / "events" / "event_vectors.npz",
                self.light_entity_engine.embed_texts,
                embedding_identity=embedding_identity(self.light_entity_settings),
                batch_size=self.eval_settings.embedding_batch_size,
                max_concurrency=self.eval_settings.effective_embedding_concurrency,
                show_progress=self.show_progress,
            )

            def resolve_document_id(_content: str, file_path: str) -> str | None:
                return document_id_from_file_path(
                    file_path,
                    self.file_path_to_document,
                )

            self.unified_orchestrator = UnifiedRetrievalOrchestrator(
                self.light_entity_engine,
                planner,
                event_service=self.event_service,
                event_vector_index=event_vector,
                document_id_resolver=resolve_document_id,
                strict_event_failures=True,
                strict_coordination_failures=True,
                evolink_seed_limit_per_name=(
                    self.eval_settings.evolink_seed_limit_per_name
                ),
                evolink_max_entity_names=self.eval_settings.evolink_max_entity_names,
                evolink_max_event_seeds=self.eval_settings.evolink_max_event_seeds,
                evolink_expansion_limit=self.eval_settings.evolink_expansion_limit,
                evolink_link_weight=self.eval_settings.evolink_link_weight,
            )
        if EVOLINK_METHODS & set(self.methods):
            await self._ensure_evolink_index()
            if (
                self.unified_orchestrator is not None
                and self.evolink_graph_store is not None
            ):
                self.unified_orchestrator.bind_coordination_store(
                    self.evolink_graph_store
                )
        if TKG_METHODS & set(self.methods):
            await self._ensure_tkg_index()
        if AGENTIC_METHODS & set(self.methods):
            agent_model = DeepSeekAgentModel(
                self.eval_settings,
                plan_cache_path=(
                    self.runtime_dir
                    / "events"
                    / f"evoneed_plan_cache_{EVONEED_PROMPT_PROFILE_VERSION}.json"
                ),
            )
            if "agentic_hybrid" in self.methods:
                assert hybrid is not None
                self.agentic_graphs["agentic_hybrid"] = AgenticRAGGraph(
                    hybrid.retrieve,
                    agent_model,
                )
            if "agentic_light_entity" in self.methods:
                self.agentic_graphs["agentic_light_entity"] = AgenticRAGGraph(
                    self._retrieve_light_entity,
                    agent_model,
                )
            if "agentic_ours" in self.methods:
                self.agentic_graphs["agentic_ours"] = AgenticRAGGraph(
                    self._retrieve_ours,
                    agent_model,
                )
            if "agentic_ours_evolink" in self.methods:
                self.agentic_graphs["agentic_ours_evolink"] = AgenticRAGGraph(
                    self._retrieve_ours_evolink,
                    agent_model,
                )
            if "ours_evoneed" in self.methods:
                self.agentic_graphs["ours_evoneed"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_actions,
                    agent_model,
                )
            if "ours_evoneed_planned" in self.methods:
                self.agentic_graphs["ours_evoneed_planned"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_actions,
                    agent_model,
                    initial_retrieval="planned_needs",
                )
            if "ours_evoneed_planned_shared" in self.methods:
                self.agentic_graphs["ours_evoneed_planned_shared"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_actions,
                    agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="shared_total",
                    minimum_route_budget=self.eval_settings.evoneed_min_route_budget,
                )
            if "ours_evoneed_equal" in self.methods:
                self.agentic_graphs["ours_evoneed_equal"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_actions,
                    agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="equal_total",
                )
            if "ours_evoneed_dynamic" in self.methods:
                self.agentic_graphs["ours_evoneed_dynamic"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_actions,
                    agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="planned_total",
                    minimum_route_budget=self.eval_settings.evoneed_min_route_budget,
                )
            if "ours_evoneed_deep" in self.methods:
                deep_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_PLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs["ours_evoneed_deep"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_actions,
                    deep_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="per_action",
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                )
            if "ours_evoneed_deep_oneplan" in self.methods:
                oneplan_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_ONEPLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs["ours_evoneed_deep_oneplan"] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_oneplan_actions,
                    oneplan_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="per_action",
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                )
            if "ours_evoneed_deep_oneplan_d1_norerank" in self.methods:
                norerank_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_ONEPLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs[
                    "ours_evoneed_deep_oneplan_d1_norerank"
                ] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_oneplan_actions,
                    norerank_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="per_action",
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                    need_fusion_mode="rrf",
                )
            if "ours_evoneed_deep_oneplan_nolink_norerank" in self.methods:
                nolink_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_ONEPLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs[
                    "ours_evoneed_deep_oneplan_nolink_norerank"
                ] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_oneplan_nolink_actions,
                    nolink_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="per_action",
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                    need_fusion_mode="rrf",
                )
            if "ours_evoneed_deep_oneplan_shared_equal_total_norerank" in self.methods:
                shared_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_ONEPLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs[
                    "ours_evoneed_deep_oneplan_shared_equal_total_norerank"
                ] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_oneplan_shared_equal_total_actions,
                    shared_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="per_action",
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                    need_fusion_mode="rrf",
                )
            if "ours_evoneed_deep_oneplan_shared_total_norerank" in self.methods:
                shared_total_agent_model = DeepSeekAgentModel(
                    self.eval_settings,
                    plan_cache_path=(
                        self.runtime_dir
                        / "events"
                        / (
                            "evoneed_plan_cache_"
                            f"{EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION}.json"
                        )
                    ),
                    plan_profile=EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                    plan_system_prompt=EVONEED_DEEP_ONEPLAN_SYSTEM,
                    plan_uses_total_budget=False,
                )
                self.agentic_graphs[
                    "ours_evoneed_deep_oneplan_shared_total_norerank"
                ] = EvoNeedRAGGraph(
                    self._retrieve_evoneed_deep_oneplan_shared_total_actions,
                    shared_total_agent_model,
                    initial_retrieval="planned_needs",
                    route_budget_mode="shared_total",
                    minimum_route_budget=1,
                    deduplicate_initial_routes=False,
                    anchor_planned_queries=False,
                    need_fusion_mode="rrf",
                )
            if "agentic_ours_tkg" in self.methods:
                if self.tkg_retriever is None:
                    raise RuntimeError("Unified TKG retriever is not initialized")
                self.agentic_graphs["agentic_ours_tkg"] = AgenticRAGGraph(
                    self.tkg_retriever.retrieve,
                    agent_model,
                )
            if "agentic_ours_tkg_dynamic" in self.methods:
                if self.tkg_dynamic_retriever is None:
                    raise RuntimeError("Dynamic TKG retriever is not initialized")
                self.agentic_graphs["agentic_ours_tkg_dynamic"] = AgenticRAGGraph(
                    self.tkg_dynamic_retriever.retrieve,
                    agent_model,
                )

        summaries: dict[str, Any] = {}
        try:
            for method in self.methods:
                retriever = self._method_retriever(method, dense, hybrid)
                per_query = await self._evaluate_method(method, retriever)
                summary = aggregate_results(per_query)
                summary["retrieval_diagnostics"] = retrieval_diagnostics(per_query)
                if method in AGENTIC_METHODS:
                    summary["agentic_diagnostics"] = agentic_diagnostics(per_query)
                if method in EVONEED_METHODS:
                    summary["evoneed_diagnostics"] = evoneed_diagnostics(per_query)
                summaries[method] = summary
                self._write_method_summary(method, summary)
            result = {
                "dataset": self.dataset.name,
                "fingerprint": self.fingerprint,
                "lightrag_fingerprint": self.lightrag_fingerprint,
                "document_count": len(self.documents),
                "chunk_count": len(self.chunks),
                "query_count": len(self.queries),
                "methods": list(self.methods),
                "cutoffs": list(self.cutoffs),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_strategy": self.chunking_strategy,
                "retrieval_profile": UNIFIED_RETRIEVAL_PROFILE_VERSION,
                "query_planner_profile": QUERY_PLANNER_PROFILE_VERSION,
                "elapsed_seconds": perf_counter() - started,
                "summaries": summaries,
            }
            if AGENTIC_METHODS & set(self.methods):
                result["agentic_profile"] = AGENTIC_PROMPT_PROFILE_VERSION
            if STANDARD_EVONEED_METHODS & set(self.methods):
                result["evoneed_profile"] = EVONEED_PROMPT_PROFILE_VERSION
            if "ours_evoneed_deep" in self.methods:
                result["evoneed_deep_profile"] = (
                    EVONEED_DEEP_PROMPT_PROFILE_VERSION
                )
            if "ours_evoneed_deep_oneplan" in self.methods:
                result["evoneed_deep_oneplan_profile"] = (
                    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION
                )
            if "ours_evoneed_deep_oneplan_d1_norerank" in self.methods:
                result["evoneed_deep_oneplan_d1_norerank_profile"] = (
                    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION
                )
            if "ours_evoneed_deep_oneplan_nolink_norerank" in self.methods:
                result["evoneed_deep_oneplan_nolink_norerank_profile"] = (
                    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION
                )
            if "ours_evoneed_deep_oneplan_shared_equal_total_norerank" in self.methods:
                result["evoneed_deep_oneplan_shared_equal_total_norerank_profile"] = (
                    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION
                )
            if "ours_evoneed_deep_oneplan_shared_total_norerank" in self.methods:
                result["evoneed_deep_oneplan_shared_total_norerank_profile"] = (
                    EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION
                )
            if TKG_METHODS & set(self.methods):
                result["tkg_retrieval_profile"] = TKG_RETRIEVAL_PROFILE_VERSION
                result["tkg_query_planner_profile"] = TKG_QUERY_PLANNER_PROFILE_VERSION
            if EVOLINK_METHODS & set(self.methods):
                result["evolink_retrieval_profile"] = (
                    EVOLINK_RETRIEVAL_PROFILE_VERSION
                )
            if D1_METHODS & set(self.methods):
                d1_methods = set(self.methods) & D1_METHODS
                if "ours_d1_queryrewrite_sharedlink" in d1_methods:
                    expansion_scope = (
                        "mixed"
                        if "ours_d1_direct_noplan" in d1_methods
                        or "ours_d1_queryrewrite" in d1_methods
                        else "one_shared_expansion_after_all_routes"
                    )
                elif "ours_d1_queryrewrite_sharedlink_total" in d1_methods:
                    expansion_scope = "one_shared_expansion_fixed_evolink_cap"
                else:
                    expansion_scope = "one_independent_expansion_per_route"
                result["d1_profile"] = {
                    "methods": sorted(d1_methods),
                    "max_routes": 3,
                    "route_budget": self.top_k,
                    "candidate_multiplier": CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                    "cross_path_expansion": expansion_scope,
                    "fusion": "plain_rrf",
                    "llm_rerank": False,
                    "query_planner_calls": (
                        1
                        if {
                            "ours_d1_queryrewrite",
                            "ours_d1_queryrewrite_sharedlink",
                            "ours_d1_queryrewrite_sharedlink_total",
                        }
                        & set(self.methods)
                        else 0
                    ),
                }
                if "ours_d1_queryrewrite_sharedlink" in d1_methods:
                    result["d1_profile"]["shared_expansion_budget"] = (
                        "route_budget_times_route_count"
                    )
                if "ours_d1_queryrewrite_sharedlink_total" in d1_methods:
                    result["d1_profile"]["shared_expansion_budget"] = (
                        "fixed_evolink_expansion_cap"
                    )
            (self.output_dir / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._write_summary_csv(summaries)
            self._write_summary_markdown(result)
            return result
        finally:
            engines = [
                self.light_engine,
                self.light_entity_engine,
                self.tkg_extraction_engine,
            ]
            for engine in engines:
                if engine is not None:
                    await engine.finalize()
            if self.tkg_vector_index is not None:
                await self.tkg_vector_index.close()
            if self.tkg_graph_store is not None:
                await self.tkg_graph_store.close()
            if self.evolink_graph_store is not None:
                await self.evolink_graph_store.close()

    def _method_retriever(
        self,
        method: str,
        dense: DenseRetriever | None,
        hybrid: HybridRetriever | None,
    ):
        if method == "naive":
            assert dense is not None
            return dense.retrieve
        if method == "hybrid":
            assert hybrid is not None
            return hybrid.retrieve
        if method == "light":
            return self._retrieve_light
        if method == "light_entity":
            return self._retrieve_light_entity
        if method == "ours":
            return self._retrieve_ours
        if method == "ours_evolink":
            return self._retrieve_ours_evolink
        if method == "ours_association_only":
            return self._retrieve_ours_association_only
        if method == "ours_evoneed":
            return self._retrieve_ours_evoneed
        if method == "ours_evoneed_planned":
            return self._retrieve_ours_evoneed_planned
        if method == "ours_evoneed_planned_shared":
            return self._retrieve_ours_evoneed_planned_shared
        if method == "ours_evoneed_equal":
            return self._retrieve_ours_evoneed_equal
        if method == "ours_evoneed_dynamic":
            return self._retrieve_ours_evoneed_dynamic
        if method == "ours_evoneed_deep":
            return self._retrieve_ours_evoneed_deep
        if method == "ours_evoneed_deep_oneplan":
            return self._retrieve_ours_evoneed_deep_oneplan
        if method == "ours_evoneed_deep_oneplan_d1_norerank":
            return self._retrieve_ours_evoneed_deep_oneplan_d1_norerank
        if method == "ours_evoneed_deep_oneplan_nolink_norerank":
            return self._retrieve_ours_evoneed_deep_oneplan_nolink_norerank
        if method == "ours_evoneed_deep_oneplan_shared_equal_total_norerank":
            return self._retrieve_ours_evoneed_deep_oneplan_shared_equal_total_norerank
        if method == "ours_evoneed_deep_oneplan_shared_total_norerank":
            return self._retrieve_ours_evoneed_deep_oneplan_shared_total_norerank
        if method == "ours_d1_direct_noplan":
            return self._retrieve_ours_d1_direct_noplan
        if method == "ours_d1_queryrewrite":
            return self._retrieve_ours_d1_queryrewrite
        if method == "ours_d1_queryrewrite_sharedlink":
            return self._retrieve_ours_d1_queryrewrite_sharedlink
        if method == "ours_d1_queryrewrite_sharedlink_total":
            return self._retrieve_ours_d1_queryrewrite_sharedlink_total
        if method == "ours_tkg":
            if self.tkg_retriever is None:
                raise RuntimeError("Unified TKG retriever is not initialized")
            return self.tkg_retriever.retrieve
        if method == "ours_tkg_dynamic":
            if self.tkg_dynamic_retriever is None:
                raise RuntimeError("Dynamic TKG retriever is not initialized")
            return self.tkg_dynamic_retriever.retrieve
        if method == "agentic_ours":
            return self._retrieve_agentic_ours
        if method == "agentic_ours_evolink":
            return self._retrieve_agentic_ours_evolink
        if method == "agentic_ours_tkg":
            return self._retrieve_agentic_ours_tkg
        if method == "agentic_ours_tkg_dynamic":
            return self._retrieve_agentic_ours_tkg_dynamic
        if method == "agentic_light_entity":
            return self._retrieve_agentic_light_entity
        if method == "agentic_hybrid":
            return self._retrieve_agentic_hybrid
        raise AssertionError(f"Unhandled evaluation method: {method}")

    async def _evaluate_method(self, method: str, retriever) -> list[dict[str, Any]]:
        path = self.output_dir / f"{method}.jsonl"
        existing = _load_jsonl_by_id(path)
        completed_ids = {query.query_id for query in self.queries} & set(existing)
        progress = tqdm(
            total=len(self.queries),
            initial=len(completed_ids),
            desc=f"{method}:query",
            unit="query",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        try:
            for query in self.queries:
                if query.query_id in existing:
                    continue
                progress.set_postfix_str(query.query_id, refresh=False)
                started = perf_counter()
                try:
                    retrieval_result = await retriever(query.question, self.top_k)
                    if isinstance(retrieval_result, AgenticRun):
                        agentic_run = retrieval_result
                        retrieved = list(agentic_run.evidence)
                    else:
                        agentic_run = None
                        retrieved = retrieval_result
                    elapsed_ms = (perf_counter() - started) * 1000
                    row = {
                        "query_id": query.query_id,
                        "question": query.question,
                        "question_type": query.question_type,
                        "answer": query.answer,
                        "gold_evidence_count": len(query.gold_evidence),
                        "elapsed_ms": elapsed_ms,
                        "metrics": query_metrics(
                            query,
                            retrieved,
                            cutoffs=self.cutoffs,
                        ),
                        "retrieved": [
                            {
                                **asdict(item),
                                "text": item.text[:1_500],
                            }
                            for item in retrieved
                        ],
                    }
                    if agentic_run is not None:
                        row.update(
                            {
                                "generated_answer": agentic_run.answer,
                                "citations": list(agentic_run.citations),
                                "insufficient_evidence": (
                                    agentic_run.insufficient_evidence
                                ),
                                "trajectory": agentic_run.trajectory(),
                            }
                        )
                except Exception as exc:
                    if self.fail_fast:
                        raise
                    logger.exception("Evaluation query failed: %s/%s", method, query.query_id)
                    row = {
                        "query_id": query.query_id,
                        "question": query.question,
                        "question_type": query.question_type,
                        "elapsed_ms": (perf_counter() - started) * 1000,
                        "metrics": {},
                        "retrieved": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _append_jsonl(path, row)
                existing[query.query_id] = row
                progress.update(1)
        finally:
            progress.close()
        return [existing[item.query_id] for item in self.queries]

    def _require_light_environment(self) -> None:
        if not self.eval_settings.deepseek_api_key:
            raise RuntimeError(
                "LightRAG evaluation requires DEEPSEEK_API_KEY. "
                "Run only local baselines with --methods naive,hybrid."
            )

    async def _ensure_light_index(
        self,
        engine: LightRAGEngine,
        *,
        label: str,
        marker: Path,
        fingerprint: str,
    ) -> None:
        if _marker_matches(marker, fingerprint):
            await engine.initialize()
            self._populate_file_path_map()
            return
        self._populate_file_path_map()
        started = perf_counter()
        progress = tqdm(
            total=len(self.chunks),
            desc=f"{label}:index",
            unit="chunk",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        try:
            for start in range(0, len(self.chunks), self.index_batch_size):
                batch = self.chunks[start : start + self.index_batch_size]
                await engine.index_documents(
                    [
                        (
                            item.chunk_id,
                            lightrag_benchmark_chunk_text(item),
                            benchmark_chunk_file_path(
                                item.document_id,
                                item.chunk_id,
                            ),
                        )
                        for item in batch
                    ]
                )
                progress.update(len(batch))
        finally:
            progress.close()
        _write_marker(
            marker,
            fingerprint,
            {
                "chunk_count": len(self.chunks),
                "elapsed_seconds": perf_counter() - started,
            },
        )

    def _populate_file_path_map(self) -> None:
        for chunk in self.chunks:
            value = benchmark_chunk_file_path(
                chunk.document_id,
                chunk.chunk_id,
            )
            self.file_path_to_document[value] = chunk.document_id
            self.file_path_to_document[Path(value).name] = chunk.document_id

    async def _ensure_event_index(self) -> None:
        assert self.light_entity_engine is not None
        marker = self.runtime_dir / "events" / "evaluation_index.json"
        registry = EventRegistry(self.runtime_dir / "events" / "event_store.json")
        extractor = LLMEventExtractor(
            self.light_entity_engine,
            chunk_chars=self.eval_settings.event_extraction_chunk_chars,
            max_events=self.eval_settings.event_extraction_max_events,
            max_events_per_part=(
                self.eval_settings.event_extraction_max_events_per_part
            ),
        )
        self.event_service = EventService(registry, extractor)
        if _marker_matches(marker, self.event_fingerprint):
            return
        checkpoint_path = self.runtime_dir / "events" / "document_index.json"
        checkpoint_exists = checkpoint_path.is_file()
        checkpoint = read_json(checkpoint_path, default={})
        if checkpoint.get("fingerprint") == self.event_fingerprint:
            completed_documents = {
                str(document_id): str(document_fingerprint)
                for document_id, document_fingerprint in (
                    checkpoint.get("documents") or {}
                ).items()
            }
        else:
            completed_documents = {}

        document_fingerprints = {
            document.document_id: event_document_fingerprint(document)
            for document in self.documents
        }
        if not checkpoint_exists:
            _, existing_mentions, _, _ = await registry.snapshot()
            recovered_ids = {mention.document_id for mention in existing_mentions}
            completed_documents.update(
                {
                    document_id: document_fingerprints[document_id]
                    for document_id in recovered_ids
                    if document_id in document_fingerprints
                }
            )
            if completed_documents:
                logger.info(
                    "Recovered %d completed event document(s) from the existing event store",
                    len(completed_documents),
                )
        completed_documents = {
            document_id: stored_fingerprint
            for document_id, stored_fingerprint in completed_documents.items()
            if document_fingerprints.get(document_id) == stored_fingerprint
        }
        write_json_atomic(
            checkpoint_path,
            {
                "fingerprint": self.event_fingerprint,
                "documents": completed_documents,
            },
        )
        started = perf_counter()
        queue: asyncio.Queue = asyncio.Queue()
        for document in self.documents:
            if document.document_id not in completed_documents:
                queue.put_nowait(document)
        worker_count = min(
            self.eval_settings.event_index_concurrency,
            max(1, len(self.documents)),
        )
        failures: list[tuple[str, Exception]] = []
        output_lock = asyncio.Lock()
        checkpoint_lock = asyncio.Lock()
        progress = tqdm(
            total=len(self.documents),
            initial=len(completed_documents),
            desc="ours:event-index",
            unit="doc",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )

        async def worker() -> None:
            while True:
                try:
                    document = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                error: Exception | None = None
                try:
                    await self.event_service.index_document(
                        document.document_id,
                        document.title,
                        f"{document.metadata_text()}\n\n{document.text}",
                    )
                except Exception as exc:  # keep successful documents resumable
                    error = exc
                    failures.append((document.document_id, exc))
                    logger.exception("Event indexing failed for %s", document.document_id)
                finally:
                    queue.task_done()
                    async with checkpoint_lock:
                        if error is None:
                            completed_documents[document.document_id] = (
                                document_fingerprints[document.document_id]
                            )
                        else:
                            completed_documents.pop(document.document_id, None)
                        write_json_atomic(
                            checkpoint_path,
                            {
                                "fingerprint": self.event_fingerprint,
                                "documents": completed_documents,
                            },
                        )
                    async with output_lock:
                        outcome = "failed" if error else "ready"
                        progress.update(1)
                        progress.set_postfix_str(
                            f"{outcome} {document.title[:36]}", refresh=False
                        )

        progress.set_postfix_str(f"workers={worker_count}", refresh=False)
        try:
            await asyncio.gather(*(worker() for _ in range(worker_count)))
        finally:
            progress.close()
        if failures:
            preview = "; ".join(
                f"{document_id}: {type(exc).__name__}: {exc}"
                for document_id, exc in failures[:5]
            )
            raise RuntimeError(
                f"Event indexing failed for {len(failures)} document(s): {preview}"
            )
        _write_marker(
            marker,
            self.event_fingerprint,
            {
                "document_count": len(self.documents),
                "elapsed_seconds": perf_counter() - started,
            },
        )

    async def _ensure_evolink_index(self) -> None:
        """Mirror existing semantic/event knowledge into one governed Neo4j graph."""
        if self.light_entity_engine is None or self.event_service is None:
            raise RuntimeError("EvoLink requires the controlled graph and event indexes")
        fingerprint = hashlib.sha256(
            (
                f"ours_evolink_v1\x1f{self.light_entity_fingerprint}\x1f"
                f"{self.event_fingerprint}\x1f"
                f"{self.eval_settings.tkg_ontology_version}\x1f"
                f"{self.eval_settings.entity_resolution_auto_match_threshold}"
            ).encode()
        ).hexdigest()
        evolink_dir = self.runtime_dir / "evolink" / fingerprint[:16]
        registry = TKGRegistry(
            evolink_dir / "graph_store.json",
            ontology_version=self.eval_settings.tkg_ontology_version,
        )
        event_registry = EventRegistry(
            self.runtime_dir / "events" / "event_store.json"
        )
        workspace_id = (
            f"eval_{safe_name(self.dataset.name)}_"
            f"{self.fingerprint[:12]}_evolink_v1"
        )
        self.evolink_graph_store = Neo4jTKGStore(
            self.eval_settings.neo4j_uri,
            self.eval_settings.neo4j_user,
            self.eval_settings.neo4j_password,
            database=self.eval_settings.neo4j_database,
            workspace_id=workspace_id,
        )
        await self.evolink_graph_store.initialize_schema()
        resolver = EntityResolutionService(
            OntologyTypeMapper(self.eval_settings.entity_type_mapping_file),
            self.light_entity_engine.embed_texts,
            self.light_entity_engine.complete_extraction,
            candidate_limit=self.eval_settings.entity_resolution_candidate_limit,
            embedding_threshold=(
                self.eval_settings.entity_resolution_embedding_threshold
            ),
            auto_match_threshold=(
                self.eval_settings.entity_resolution_auto_match_threshold
            ),
            llm_threshold=self.eval_settings.entity_resolution_llm_threshold,
        )
        fusion = TKGFusionService(
            registry,
            ontology_version=self.eval_settings.tkg_ontology_version,
            entity_resolver=resolver,
            neo4j_store=None,
            event_registry=event_registry,
        )
        checkpoint_path = evolink_dir / "document_index.json"
        checkpoint = read_json(checkpoint_path, default={})
        completed = (
            set(checkpoint.get("documents") or [])
            if checkpoint.get("fingerprint") == fingerprint
            else set()
        )
        events = await self.event_service.list_events()
        events_by_document: dict[str, list] = defaultdict(list)
        for event in events:
            for document_id in event.source_document_ids:
                events_by_document[document_id].append(event)
        pending_documents = [
            document
            for document in self.documents
            if document.document_id not in completed
        ]
        semantics_by_document = (
            await self.light_entity_engine.get_documents_semantic_extractions(
                [document.document_id for document in pending_documents]
            )
        )
        progress = tqdm(
            total=len(self.documents),
            initial=len(completed),
            desc="ours_evolink:governance",
            unit="doc",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        try:
            for document in pending_documents:
                semantic = semantics_by_document[document.document_id]
                document_events = events_by_document.get(document.document_id, [])
                await _retry_timeout(
                    lambda document=document,
                    semantic=semantic,
                    document_events=document_events: fusion.index_document(
                        document.document_id,
                        document.title,
                        semantic,
                        document_events,
                    ),
                    attempts=max(2, self.eval_settings.embedding_max_retries + 1),
                    label=f"EvoLink governance for {document.document_id}",
                )
                completed.add(document.document_id)
                write_json_atomic(
                    checkpoint_path,
                    {
                        "fingerprint": fingerprint,
                        "documents": sorted(completed),
                    },
                )
                progress.update(1)
                progress.set_postfix_str(document.title[:36], refresh=False)
        finally:
            progress.close()
        snapshot = await registry.snapshot()
        _, event_mentions, event_relations, _ = await event_registry.snapshot()
        await self.evolink_graph_store.replace_snapshot(
            snapshot,
            events=events,
            event_mentions=event_mentions,
            event_relations=event_relations,
        )

    async def _ensure_tkg_index(self) -> None:
        """Build one statement TKG without creating the legacy LightRAG graph index."""
        if self.eval_settings.embedding_provider != "zhipu":
            raise RuntimeError(
                "ours_tkg evaluation currently requires EMBEDDING_PROVIDER=zhipu "
                "to keep entity resolution and all five vector indexes comparable."
            )
        workspace_id = (
            f"eval_{safe_name(self.dataset.name)}_{self.fingerprint[:12]}_tkg_v1"
            f"{'_open' if self.eval_settings.tkg_schema_mode == 'open' else ''}"
        )
        knowledge_schema_version = (
            "open-v1"
            if self.eval_settings.tkg_schema_mode == "open"
            else self.eval_settings.tkg_ontology_version
        )
        index_fingerprint = hashlib.sha256(
            (
                f"{self.fingerprint}\x1f{SEMANTIC_EXTRACTION_PROFILE_VERSION}\x1f"
                f"{EVENT_EXTRACTION_PROFILE_VERSION}\x1f"
                f"{self.eval_settings.tkg_ontology_version}"
                + (
                    "\x1fopen_schema_v1"
                    if self.eval_settings.tkg_schema_mode == "open"
                    else ""
                )
            ).encode()
        ).hexdigest()
        tkg_dir = self.runtime_dir / "tkg" / index_fingerprint[:16]
        event_registry = EventRegistry(tkg_dir / "event_store.json")
        registry = TKGRegistry(
            tkg_dir / "tkg_store.json",
            ontology_version=knowledge_schema_version,
            validate_snapshot=(
                OntologyService(
                    self.eval_settings.tkg_ontology_dir
                ).validate_snapshot
                if self.eval_settings.tkg_schema_mode == "ontology"
                else None
            ),
        )
        # This engine is an LLM/embedding client only. We intentionally never call
        # initialize(), index_document(), or retrieve_data(), so no LightRAG graph,
        # KV, document-status, or LightRAG Qdrant collection is created for ours_tkg.
        tkg_embedding = build_embedding_func(self.eval_settings)
        self.tkg_extraction_engine = LightRAGEngine(
            self.eval_settings,
            SourceMapper(DocumentRegistry(tkg_dir / "extraction_documents.json")),
            embedding_func=tkg_embedding,
        )
        semantic_extractor = LLMSemanticExtractor(
            self.tkg_extraction_engine.complete_extraction,
            prompt_file=(
                self.eval_settings.light_entity_prompt_file
                if self.eval_settings.tkg_schema_mode == "ontology"
                else Path("prompts/entity_type/open_schema.yaml")
            ),
            max_entities=self.eval_settings.light_entity_max_entities,
            max_relationships=self.eval_settings.light_entity_max_relationships,
        )
        event_extractor = LLMEventExtractor(
            self.tkg_extraction_engine,
            chunk_chars=self.eval_settings.event_extraction_chunk_chars,
            max_events=self.eval_settings.event_extraction_max_events,
            max_events_per_part=self.eval_settings.event_extraction_max_events_per_part,
        )
        event_service = EventService(event_registry, event_extractor)
        self.tkg_graph_store = Neo4jTKGStore(
            self.eval_settings.neo4j_uri,
            self.eval_settings.neo4j_user,
            self.eval_settings.neo4j_password,
            database=self.eval_settings.neo4j_database,
            workspace_id=workspace_id,
        )
        await self.tkg_graph_store.initialize_schema()
        resolver = EntityResolutionService(
            OntologyTypeMapper(self.eval_settings.entity_type_mapping_file),
            tkg_embedding,
            self.tkg_extraction_engine.complete_extraction,
            candidate_limit=self.eval_settings.entity_resolution_candidate_limit,
            embedding_threshold=self.eval_settings.entity_resolution_embedding_threshold,
            auto_match_threshold=self.eval_settings.entity_resolution_auto_match_threshold,
            llm_threshold=self.eval_settings.entity_resolution_llm_threshold,
            enforce_type=self.eval_settings.tkg_schema_mode == "ontology",
        )
        tkg_service = TKGFusionService(
            registry,
            ontology_version=knowledge_schema_version,
            entity_resolver=resolver,
            neo4j_store=None,
            event_registry=event_registry,
        )
        checkpoint_path = tkg_dir / "document_index.json"
        checkpoint = read_json(checkpoint_path, default={})
        completed = (
            set(checkpoint.get("documents") or [])
            if checkpoint.get("fingerprint") == index_fingerprint
            else set()
        )
        progress = tqdm(
            total=len(self.documents),
            initial=len(completed),
            desc="ours_tkg:index",
            unit="doc",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        try:
            chunks_by_document: dict[str, list[BenchmarkChunk]] = defaultdict(list)
            for chunk in self.chunks:
                chunks_by_document[chunk.document_id].append(chunk)
            for document in self.documents:
                if document.document_id in completed:
                    continue
                segments = [
                    (chunk.chunk_id, chunk.text)
                    for chunk in chunks_by_document[document.document_id]
                ]
                semantic = await semantic_extractor.extract(document.title, segments)
                events = await event_service.index_document(
                    document.document_id,
                    document.title,
                    f"{document.metadata_text()}\n\n{document.text}",
                )
                await tkg_service.index_document(
                    document.document_id,
                    document.title,
                    semantic,
                    events,
                )
                completed.add(document.document_id)
                write_json_atomic(
                    checkpoint_path,
                    {
                        "fingerprint": index_fingerprint,
                        "documents": sorted(completed),
                    },
                )
                progress.update(1)
                progress.set_postfix_str(document.title[:36], refresh=False)
        finally:
            progress.close()
        snapshot = await registry.snapshot()
        events, event_mentions, event_relations, _ = await event_registry.snapshot()
        await self.tkg_graph_store.replace_snapshot(
            snapshot,
            events=events,
            event_mentions=event_mentions,
            event_relations=event_relations,
        )
        vector_fingerprint = tkg_vector_fingerprint(
            index_fingerprint,
            snapshot,
            events,
        )
        self.tkg_vector_index = TKGVectorIndex(
            qdrant_url=self.eval_settings.qdrant_url,
            qdrant_api_key=self.eval_settings.qdrant_api_key,
            workspace_id=workspace_id,
            dimension=self.eval_settings.embedding_dimension,
            embed=tkg_embedding,
            manifest_path=tkg_dir / "vector_index.json",
            batch_size=self.eval_settings.embedding_batch_size,
            max_concurrency=self.eval_settings.effective_embedding_concurrency,
            show_progress=self.show_progress,
        )
        await self.tkg_vector_index.initialize(
            snapshot,
            events,
            self.chunks,
            fingerprint=vector_fingerprint,
        )
        planner = QueryPlanner(
            self.tkg_extraction_engine.complete_extraction,
            cache_path=tkg_dir / "query_plan_cache.json",
            strict_failures=True,
            profile=TKG_QUERY_PLANNER_PROFILE_VERSION,
            prompt_template=TKG_PLANNER_PROMPT,
        )
        self.tkg_retriever = UnifiedTKGRetriever(
            planner,
            self.tkg_vector_index,
            self.tkg_graph_store,
        )
        self.tkg_dynamic_retriever = UnifiedTKGRetriever(
            planner,
            self.tkg_vector_index,
            self.tkg_graph_store,
            dynamic=True,
        )

    async def _retrieve_light(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if self.light_engine is None:
            raise RuntimeError("Stock LightRAG index is not initialized")
        return await self._retrieve_engine(self.light_engine, question, top_k)

    async def _retrieve_light_entity(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return await self._retrieve_unified(question, top_k, include_events=False)

    async def _retrieve_engine(
        self,
        engine: LightRAGEngine,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        raw = await engine.retrieve_data(
            question,
            mode="mix",
            top_k=max(top_k * 2, top_k),
            chunk_top_k=top_k,
            enable_rerank=False,
        )
        if raw.get("status") != "success":
            raise RuntimeError(str(raw.get("message") or "LightRAG retrieval failed"))
        return light_data_to_chunks(raw, self.file_path_to_document)[:top_k]

    async def _retrieve_ours(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if self.event_service is None:
            raise RuntimeError("Event index is not initialized")
        return await self._retrieve_unified(question, top_k, include_events=True)

    async def _retrieve_ours_evolink(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if self.event_service is None or self.evolink_graph_store is None:
            raise RuntimeError("EvoLink event and Neo4j indexes are not initialized")
        return await self._retrieve_unified(
            question,
            top_k,
            include_events=True,
            link_evidence=True,
        )

    async def _retrieve_ours_association_only(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Enable EvoLink associations while disabling EventEffect traversal."""
        if self.event_service is None or self.evolink_graph_store is None:
            raise RuntimeError("Event and Neo4j indexes are not initialized")
        return await self._retrieve_unified(
            question,
            top_k,
            include_events=True,
            link_evidence=True,
            include_event_effects=False,
        )

    async def _retrieve_ours_d1_direct_noplan(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Fixed three-route retrieval with independent EvoLink and no LLM planner."""
        plan = QueryPlan()
        routes = [
            ("vector", question, False),
            ("graph", question, False),
            ("event", question, True),
        ]
        return await self._retrieve_d1_independent_routes(
            question,
            top_k,
            plan=plan,
            routes=routes,
            mode="direct_noplan",
        )

    async def _retrieve_ours_d1_queryrewrite(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """One existing QueryPlanner call followed by three route-specific queries."""
        if self.query_planner is None:
            raise RuntimeError("Query planner is not initialized")
        plan = await self.query_planner.plan(question)
        routes = [
            (
                "vector",
                self._d1_route_query(
                    question,
                    plan,
                    route="vector",
                ),
                False,
            ),
            (
                "graph",
                self._d1_route_query(
                    question,
                    plan,
                    route="graph",
                ),
                False,
            ),
            (
                "event",
                self._d1_route_query(
                    question,
                    plan,
                    route="event",
                ),
                True,
            ),
        ]
        return await self._retrieve_d1_independent_routes(
            question,
            top_k,
            plan=plan,
            routes=routes,
            mode="queryrewrite",
        )

    async def _retrieve_ours_d1_queryrewrite_sharedlink(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Use the D1 route queries but run one shared EvoLink expansion."""
        if self.query_planner is None:
            raise RuntimeError("Query planner is not initialized")
        plan = await self.query_planner.plan(question)
        routes = [
            (
                "vector",
                self._d1_route_query(question, plan, route="vector"),
                False,
            ),
            (
                "graph",
                self._d1_route_query(question, plan, route="graph"),
                False,
            ),
            (
                "event",
                self._d1_route_query(question, plan, route="event"),
                True,
            ),
        ]
        return await self._retrieve_d1_shared_routes(
            question,
            top_k,
            plan=plan,
            routes=routes,
            mode="queryrewrite_sharedlink",
        )

    async def _retrieve_ours_d1_queryrewrite_sharedlink_total(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Strict D1 pair: 3x top-k first-stage candidates, EvoLink cap fixed."""
        if self.query_planner is None:
            raise RuntimeError("Query planner is not initialized")
        plan = await self.query_planner.plan(question)
        routes = [
            ("vector", self._d1_route_query(question, plan, route="vector"), False),
            ("graph", self._d1_route_query(question, plan, route="graph"), False),
            ("event", self._d1_route_query(question, plan, route="event"), True),
        ]
        return await self._retrieve_d1_shared_routes(
            question,
            top_k,
            plan=plan,
            routes=routes,
            mode="queryrewrite_sharedlink_total",
            shared_expansion_limit_override=self.eval_settings.evolink_expansion_limit,
        )

    async def _retrieve_d1_shared_routes(
        self,
        question: str,
        top_k: int,
        *,
        plan: QueryPlan,
        routes: list[tuple[RetrievalChannel, str, bool]],
        mode: str,
        shared_expansion_limit_override: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve route lists separately, then coordinate them once globally."""
        if self.unified_orchestrator is None:
            raise RuntimeError("Unified retrieval orchestrator is not initialized")
        if self.evolink_graph_store is None:
            raise RuntimeError("EvoLink coordination graph is not initialized")

        results = await asyncio.gather(
            *(
                self.unified_orchestrator.retrieve(
                    route_query,
                    top_k=top_k,
                    include_events=include_events,
                    link_evidence=False,
                    candidate_multiplier=CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                    channels={route},
                    plan_override=plan,
                )
                for route, route_query, include_events in routes
            )
        )
        shared_total_budget = max(top_k, top_k * len(routes))
        shared_expansion_limit = (
            shared_expansion_limit_override
            if shared_expansion_limit_override is not None
            else shared_total_budget
        )
        shared_evolink, evolink_error = await self.unified_orchestrator.coordinate_planned_results(
            [plan for _ in routes],
            results,
            limit=shared_total_budget,
            expansion_limit_override=shared_expansion_limit,
        )
        channel_candidates: dict[str, list[Any]] = {
            "vector": [],
            "graph": [],
            "event": [],
        }
        for result in results:
            for channel, candidates in result.candidates.items():
                if channel in channel_candidates:
                    channel_candidates[channel].extend(candidates)
        channel_candidates["graph"].extend(shared_evolink.graph_candidates)
        channel_candidates["event"].extend(shared_evolink.event_candidates)
        bundles = fuse_evidence(
            channel_candidates,
            top_k=top_k,
            linked_candidate_weight=self.eval_settings.evolink_link_weight,
        )
        event_result = next(
            (
                result.event_result
                for result, (_, _, include_events) in zip(results, routes, strict=True)
                if include_events
            ),
            None,
        )
        merged = UnifiedRetrievalResult(
            plan=plan,
            bundles=tuple(bundles),
            candidates={
                channel: tuple(channel_candidates[channel])
                for channel in ("vector", "graph", "event")
            },
            graph_data={
                "route_results": [result.graph_data for result in results],
            },
            evidence_graph=shared_evolink.evidence_graph,
            event_result=event_result,
            metadata={
                "fusion": "shared_route_rrf",
                "d1_mode": mode,
                "question": question,
                "route_queries": [
                    {
                        "route": route,
                        "query": route_query,
                        "include_events": include_events,
                    }
                    for route, route_query, include_events in routes
                ],
                "candidate_limit_per_route": top_k,
                "route_count": len(routes),
                "evolink_scope": (
                    "one_shared_expansion_fixed_evolink_cap"
                    if shared_expansion_limit_override is not None
                    else "one_shared_expansion_after_all_routes"
                ),
                "evolink_expansion_limit": shared_expansion_limit,
                "shared_evolink_total_budget": shared_total_budget,
                "evolink_error": evolink_error,
                "evolink": shared_evolink.metadata,
                "llm_rerank": False,
            },
        )
        return self._unified_result_to_chunks(merged)

    async def _retrieve_d1_independent_routes(
        self,
        question: str,
        top_k: int,
        *,
        plan: QueryPlan,
        routes: list[tuple[RetrievalChannel, str, bool]],
        mode: str,
    ) -> list[RetrievedChunk]:
        if self.unified_orchestrator is None:
            raise RuntimeError("Unified retrieval orchestrator is not initialized")
        if self.evolink_graph_store is None:
            raise RuntimeError("EvoLink coordination graph is not initialized")

        results = await asyncio.gather(
            *(
                self.unified_orchestrator.retrieve(
                    route_query,
                    top_k=top_k,
                    include_events=include_events,
                    link_evidence=True,
                    candidate_multiplier=CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                    channels={route},
                    plan_override=plan,
                )
                for route, route_query, include_events in routes
            )
        )
        channel_candidates: dict[str, list[Any]] = {
            "vector": [],
            "graph": [],
            "event": [],
        }
        for result in results:
            for channel, candidates in result.candidates.items():
                if channel in channel_candidates:
                    channel_candidates[channel].extend(candidates)
        bundles = fuse_evidence(
            channel_candidates,
            top_k=top_k,
            linked_candidate_weight=self.eval_settings.evolink_link_weight,
        )
        merged = UnifiedRetrievalResult(
            plan=plan,
            bundles=tuple(bundles),
            candidates={
                channel: tuple(channel_candidates[channel])
                for channel in ("vector", "graph", "event")
            },
            graph_data={
                "route_results": [result.graph_data for result in results],
            },
            evidence_graph={
                "route_results": [result.evidence_graph for result in results],
            },
            metadata={
                "fusion": "independent_route_rrf",
                "d1_mode": mode,
                "question": question,
                "route_queries": [
                    {
                        "route": route,
                        "query": route_query,
                        "include_events": include_events,
                    }
                    for route, route_query, include_events in routes
                ],
                "candidate_limit_per_route": top_k,
                "route_count": len(routes),
                "evolink_scope": "one_independent_expansion_per_route",
                "llm_rerank": False,
            },
        )
        return self._unified_result_to_chunks(merged)

    @staticmethod
    def _d1_route_query(question: str, plan: QueryPlan, *, route: str) -> str:
        values: list[str] = [question]
        if route == "vector":
            values.extend([*plan.low_level_keywords, *plan.entities])
        elif route == "graph":
            values.extend([*plan.entities, *plan.relations, *plan.high_level_keywords])
        else:
            values.extend(
                [
                    *plan.entities,
                    *plan.event_types,
                    *plan.state_intents,
                    *([plan.start_time] if plan.start_time else []),
                    *([plan.end_time] if plan.end_time else []),
                ]
            )
        return " ".join(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    async def _retrieve_agentic_ours(self, question: str, top_k: int) -> AgenticRun:
        return await self._retrieve_agentic("agentic_ours", question, top_k)

    async def _retrieve_agentic_ours_evolink(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "agentic_ours_evolink", question, top_k
        )

    async def _retrieve_ours_evoneed(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("ours_evoneed", question, top_k)

    async def _retrieve_ours_evoneed_planned(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_planned", question, top_k
        )

    async def _retrieve_ours_evoneed_planned_shared(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_planned_shared", question, top_k
        )

    async def _retrieve_ours_evoneed_equal(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("ours_evoneed_equal", question, top_k)

    async def _retrieve_ours_evoneed_dynamic(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("ours_evoneed_dynamic", question, top_k)

    async def _retrieve_ours_evoneed_deep(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("ours_evoneed_deep", question, top_k)

    async def _retrieve_ours_evoneed_deep_oneplan(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_deep_oneplan", question, top_k
        )

    async def _retrieve_ours_evoneed_deep_oneplan_d1_norerank(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_deep_oneplan_d1_norerank", question, top_k
        )

    async def _retrieve_ours_evoneed_deep_oneplan_nolink_norerank(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_deep_oneplan_nolink_norerank", question, top_k
        )

    async def _retrieve_ours_evoneed_deep_oneplan_shared_equal_total_norerank(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
            question,
            top_k,
        )

    async def _retrieve_ours_evoneed_deep_oneplan_shared_total_norerank(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "ours_evoneed_deep_oneplan_shared_total_norerank",
            question,
            top_k,
        )

    async def _retrieve_evoneed_deep_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]:
        """Reproduce the legacy high-depth path independently for every need."""
        if len(needs) != len(budgets):
            raise ValueError("EvoNeed needs and budgets must have the same length")
        return await asyncio.gather(
            *(
                self._retrieve_unified(
                    need.query,
                    budget,
                    include_events=need.route == "event",
                    link_evidence=True,
                    channels={need.route},
                    candidate_multiplier=3,
                )
                for need, budget in zip(needs, budgets, strict=True)
            )
        )

    async def _retrieve_evoneed_deep_oneplan_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]:
        """Execute first-stage retrieval plans without a second LLM planner."""
        if len(needs) != len(budgets):
            raise ValueError("EvoNeed needs and budgets must have the same length")
        return await asyncio.gather(
            *(
                self._retrieve_unified(
                    need.query,
                    budget,
                    include_events=need.route == "event",
                    link_evidence=True,
                    channels={need.route},
                    candidate_multiplier=1,
                    plan_override=self._query_plan_from_need(need),
                )
                for need, budget in zip(needs, budgets, strict=True)
            )
        )

    async def _retrieve_evoneed_deep_oneplan_nolink_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]:
        """Run the OnePlan first stage without any EvoLink expansion."""
        if len(needs) != len(budgets):
            raise ValueError("EvoNeed needs and budgets must have the same length")
        return await asyncio.gather(
            *(
                self._retrieve_unified(
                    need.query,
                    budget,
                    include_events=need.route == "event",
                    link_evidence=False,
                    channels={need.route},
                    candidate_multiplier=1,
                    plan_override=self._query_plan_from_need(need),
                )
                for need, budget in zip(needs, budgets, strict=True)
            )
        )

    async def _retrieve_evoneed_deep_oneplan_shared_equal_total_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
        *,
        expansion_limit_override: int | None = None,
        expansion_scope: str = "one_shared_expansion_equal_to_need_budget",
    ) -> list[list[RetrievedChunk]]:
        """Run need routes independently, then share one equal-budget EvoLink pass."""
        if self.unified_orchestrator is None:
            raise RuntimeError("Unified retrieval orchestrator is not initialized")
        if len(needs) != len(budgets):
            raise ValueError("EvoNeed needs and budgets must have the same length")
        if not needs:
            return []

        plans = [self._query_plan_from_need(need) for need in needs]
        results = await asyncio.gather(
            *(
                self.unified_orchestrator.retrieve(
                    need.query,
                    top_k=budget,
                    include_events=need.route == "event",
                    link_evidence=False,
                    channels={need.route},
                    candidate_multiplier=CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                    plan_override=plan,
                )
                for need, budget, plan in zip(needs, budgets, plans, strict=True)
            )
        )
        total_budget = max(1, sum(budgets))
        expansion_limit = max(
            1,
            min(
                total_budget,
                expansion_limit_override
                if expansion_limit_override is not None
                else total_budget,
            ),
        )
        evolink, evolink_error = await self.unified_orchestrator.coordinate_planned_results(
            plans,
            results,
            limit=expansion_limit,
            expansion_limit_override=expansion_limit,
        )

        candidates = [
            {channel: list(values) for channel, values in result.candidates.items()}
            for result in results
        ]
        graph_target = next(
            (index for index, need in enumerate(needs) if need.route == "graph"),
            None,
        )
        event_target = next(
            (index for index, need in enumerate(needs) if need.route == "event"),
            None,
        )
        fallback_target = 0
        shared_target_indices: set[int] = set()
        if evolink.graph_candidates:
            target = graph_target if graph_target is not None else (
                event_target if event_target is not None else fallback_target
            )
            candidates[target]["graph"].extend(evolink.graph_candidates)
            shared_target_indices.add(target)
        if evolink.event_candidates:
            target = event_target if event_target is not None else (
                graph_target if graph_target is not None else fallback_target
            )
            candidates[target]["event"].extend(evolink.event_candidates)
            shared_target_indices.add(target)

        batches: list[list[RetrievedChunk]] = []
        for index, (result, plan, budget, channel_candidates) in enumerate(
            zip(results, plans, budgets, candidates, strict=True)
        ):
            bundles = fuse_evidence(
                channel_candidates,
                # Keep the complete shared expansion in the final need-level
                # RRF list. Otherwise attaching a 40/60-item shared expansion
                # to one route and truncating it back to 20 would silently
                # destroy the equal-total budget comparison.
                top_k=(
                    max(budget, total_budget)
                    if index in shared_target_indices
                    else budget
                ),
                linked_candidate_weight=self.eval_settings.evolink_link_weight,
            )
            coordinated = UnifiedRetrievalResult(
                plan=plan,
                bundles=tuple(bundles),
                candidates={
                    channel: tuple(channel_candidates.get(channel, []))
                    for channel in ("vector", "graph", "event")
                },
                graph_data=result.graph_data,
                evidence_graph=evolink.evidence_graph,
                event_result=result.event_result,
                metadata={
                    **result.metadata,
                    "evolink_enabled": True,
                    "evolink_error": evolink_error,
                    "evolink": evolink.metadata,
                    "evolink_scope": expansion_scope,
                    "evolink_expansion_limit": expansion_limit,
                    "shared_evolink_total_budget": total_budget,
                },
            )
            batches.append(self._unified_result_to_chunks(coordinated))
        return batches

    async def _retrieve_evoneed_deep_oneplan_shared_total_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]:
        """OnePlan with a strict query-level total budget and a fixed EvoLink cap."""
        return await self._retrieve_evoneed_deep_oneplan_shared_equal_total_actions(
            needs,
            budgets,
            expansion_limit_override=self.eval_settings.evolink_expansion_limit,
            expansion_scope="one_shared_expansion_fixed_evolink_cap",
        )

    async def _retrieve_evoneed_actions(
        self,
        needs: list[NeedItem],
        budgets: list[int],
    ) -> list[list[RetrievedChunk]]:
        """Execute planned routes, then apply EvoLink exactly once to the batch."""
        if self.unified_orchestrator is None:
            raise RuntimeError("Unified retrieval orchestrator is not initialized")
        if len(needs) != len(budgets):
            raise ValueError("EvoNeed needs and budgets must have the same length")
        if len(needs) == 1 and needs[0].need_id == "need_question_anchor":
            return [
                await self._retrieve_unified(
                    needs[0].query,
                    budgets[0],
                    include_events=True,
                    link_evidence=True,
                    channels=None,
                    candidate_multiplier=CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                )
            ]

        plans = [self._query_plan_from_need(need) for need in needs]
        results = await asyncio.gather(
            *(
                self.unified_orchestrator.retrieve(
                    need.query,
                    top_k=budget,
                    include_events=need.route == "event",
                    link_evidence=False,
                    candidate_multiplier=CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                    channels={need.route},
                    plan_override=plan,
                )
                for need, budget, plan in zip(needs, budgets, plans, strict=True)
            )
        )
        evolink, evolink_error = await self.unified_orchestrator.coordinate_planned_results(
            plans,
            results,
            limit=self.eval_settings.evolink_expansion_limit,
        )

        candidates = [
            {channel: list(values) for channel, values in result.candidates.items()}
            for result in results
        ]
        graph_target = next(
            (index for index, need in enumerate(needs) if need.route == "graph"),
            None,
        )
        event_target = next(
            (index for index, need in enumerate(needs) if need.route == "event"),
            None,
        )
        fallback_target = 0 if needs else None
        if evolink.graph_candidates and fallback_target is not None:
            target = graph_target if graph_target is not None else (
                event_target if event_target is not None else fallback_target
            )
            candidates[target]["graph"] = [
                *candidates[target].get("graph", []),
                *evolink.graph_candidates,
            ]
        if evolink.event_candidates and fallback_target is not None:
            target = event_target if event_target is not None else (
                graph_target if graph_target is not None else fallback_target
            )
            candidates[target]["event"] = [
                *candidates[target].get("event", []),
                *evolink.event_candidates,
            ]

        batches: list[list[RetrievedChunk]] = []
        for result, plan, budget, channel_candidates in zip(
            results, plans, budgets, candidates, strict=True
        ):
            bundles = fuse_evidence(
                channel_candidates,
                top_k=budget,
                linked_candidate_weight=self.eval_settings.evolink_link_weight,
            )
            coordinated = UnifiedRetrievalResult(
                plan=plan,
                bundles=tuple(bundles),
                candidates={
                    channel: tuple(channel_candidates.get(channel, []))
                    for channel in ("vector", "graph", "event")
                },
                graph_data=result.graph_data,
                evidence_graph=evolink.evidence_graph,
                event_result=result.event_result,
                metadata={
                    **result.metadata,
                    "evolink_enabled": True,
                    "evolink_error": evolink_error,
                    "evolink": evolink.metadata,
                    "evolink_scope": "one_shared_expansion_per_need_batch",
                },
            )
            batches.append(self._unified_result_to_chunks(coordinated))
        return batches

    @staticmethod
    def _query_plan_from_need(need: NeedItem) -> QueryPlan:
        low_keywords = need.low_level_keywords or list(need.entities)
        high_keywords = need.high_level_keywords or [
            *need.relations,
            *need.event_types,
        ]
        return QueryPlan(
            low_level_keywords=list(dict.fromkeys(low_keywords))[:12],
            high_level_keywords=list(dict.fromkeys(high_keywords))[:12],
            relations=list(dict.fromkeys(need.relations))[:6],
            entities=list(dict.fromkeys(need.entities))[:6],
            event_types=list(dict.fromkeys(need.event_types))[:6],
            start_time=need.start_time,
            end_time=need.end_time,
            state_intents=list(dict.fromkeys(need.state_intents))[:6],
            expand_event_chain=need.expand_event_chain,
        )

    async def _retrieve_agentic_ours_tkg(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("agentic_ours_tkg", question, top_k)

    async def _retrieve_agentic_ours_tkg_dynamic(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic(
            "agentic_ours_tkg_dynamic", question, top_k
        )

    async def _retrieve_agentic_light_entity(
        self,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        return await self._retrieve_agentic("agentic_light_entity", question, top_k)

    async def _retrieve_agentic_hybrid(self, question: str, top_k: int) -> AgenticRun:
        return await self._retrieve_agentic("agentic_hybrid", question, top_k)

    async def _retrieve_agentic(
        self,
        method: str,
        question: str,
        top_k: int,
    ) -> AgenticRun:
        graph = self.agentic_graphs.get(method)
        if graph is None:
            raise RuntimeError(f"Agentic graph is not initialized for {method}")
        run_kwargs = {
            "top_k": top_k,
            "context_k": top_k,
            "max_retrieval_rounds": self.eval_settings.agentic_max_retrieval_rounds,
            "rrf_k": self.eval_settings.agentic_rrf_k,
        }
        if method in {
            "ours_evoneed_deep_oneplan_d1_norerank",
            "ours_evoneed_deep_oneplan_nolink_norerank",
            "ours_evoneed_deep_oneplan_shared_equal_total_norerank",
            "ours_evoneed_deep_oneplan_shared_total_norerank",
        }:
            return await graph.run(
                question,
                retrieval_budget=(top_k * 3
                    if method == "ours_evoneed_deep_oneplan_shared_total_norerank"
                    else top_k),
                retrieval_only=True,
                **run_kwargs,
            )
        if method in DEEP_RERANK_METHODS:
            return await graph.run(
                question,
                retrieval_budget=top_k,
                assessment_only=True,
                **run_kwargs,
            )
        if method in STANDARD_EVONEED_METHODS:
            return await graph.run(
                question,
                retrieval_budget=self.eval_settings.evoneed_retrieval_budget,
                retrieval_only=True,
                **run_kwargs,
            )
        return await graph.run(question, **run_kwargs)

    async def _retrieve_unified(
        self,
        question: str,
        top_k: int,
        *,
        include_events: bool,
        link_evidence: bool = False,
        include_event_effects: bool = True,
        channels: set[RetrievalChannel] | None = None,
        candidate_multiplier: int = CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
        plan_override: QueryPlan | None = None,
    ) -> list[RetrievedChunk]:
        if self.unified_orchestrator is None:
            raise RuntimeError("Unified retrieval orchestrator is not initialized")
        result = await self.unified_orchestrator.retrieve(
            question,
            top_k=top_k,
            include_events=include_events,
            link_evidence=link_evidence,
            include_event_effects=include_event_effects,
            candidate_multiplier=candidate_multiplier,
            channels=channels,
            plan_override=plan_override,
        )
        return self._unified_result_to_chunks(result)

    @staticmethod
    def _unified_result_to_chunks(
        result: UnifiedRetrievalResult,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=(
                    bundle.passages[0].chunk_id
                    if bundle.passages and bundle.passages[0].chunk_id
                    else f"bundle-{bundle.document_id}"
                ),
                document_id=bundle.document_id,
                text="\n\n".join(item.text for item in bundle.passages),
                score=bundle.score,
                metadata={
                    "fusion": "unified_document_rrf",
                    "channels": list(bundle.channels),
                    "passages": [
                        {
                            "channel": item.channel,
                            "chunk_id": item.chunk_id,
                            "rank": item.rank,
                            "file_path": item.file_path,
                            "metadata": item.metadata,
                        }
                        for item in bundle.passages
                    ],
                    "query_plan": result.plan.model_dump(mode="json"),
                    "evidence_graph": result.evidence_graph,
                    "evolink": result.metadata.get("evolink", {}),
                },
            )
            for bundle in result.bundles
        ]

    def _write_method_summary(self, method: str, summary: dict[str, Any]) -> None:
        (self.output_dir / f"{method}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_output_manifest(self) -> None:
        path = self.output_dir / "run_manifest.json"
        expected: dict[str, Any] = {
            "dataset": self.dataset.name,
            "fingerprint": self.fingerprint,
            "lightrag_fingerprint": self.lightrag_fingerprint,
            "cutoffs": list(self.cutoffs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model_id": embedding_identity(self.eval_settings),
            "retrieval_profile": UNIFIED_RETRIEVAL_PROFILE_VERSION,
            "controlled_route_candidate_multiplier": (
                CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER
            ),
            "query_planner_profile": QUERY_PLANNER_PROFILE_VERSION,
        }
        if TKG_METHODS & set(self.methods):
            expected["tkg_profile"] = {
                "retrieval": TKG_RETRIEVAL_PROFILE_VERSION,
                "planner": TKG_QUERY_PLANNER_PROFILE_VERSION,
                "semantic_extraction": SEMANTIC_EXTRACTION_PROFILE_VERSION,
                "event_extraction": EVENT_EXTRACTION_PROFILE_VERSION,
                "ontology_version": (
                    "open-v1"
                    if self.eval_settings.tkg_schema_mode == "open"
                    else self.eval_settings.tkg_ontology_version
                ),
                "workspace": (
                    f"eval_{safe_name(self.dataset.name)}_{self.fingerprint[:12]}_tkg_v1"
                    f"{'_open' if self.eval_settings.tkg_schema_mode == 'open' else ''}"
                ),
            }
            if self.eval_settings.tkg_schema_mode == "open":
                expected["tkg_profile"]["schema_mode"] = "open"
                expected["tkg_profile"]["schema_version"] = "open-v1"
        if AGENTIC_METHODS & set(self.methods):
            expected["agentic_profile"] = {
                "version": AGENTIC_PROMPT_PROFILE_VERSION,
                "prompt_digest": prompt_profile_digest(),
                "assessment_model": self.eval_settings.deepseek_extraction_model,
                "answer_model": self.eval_settings.deepseek_chat_model,
                "max_retrieval_rounds": (
                    self.eval_settings.agentic_max_retrieval_rounds
                ),
                "rrf_k": self.eval_settings.agentic_rrf_k,
                "methods": sorted(set(self.methods) & AGENTIC_METHODS),
            }
        if EVOLINK_METHODS & set(self.methods):
            expected["evolink_profile"] = {
                "version": EVOLINK_RETRIEVAL_PROFILE_VERSION,
                "seed_limit_per_name": (
                    self.eval_settings.evolink_seed_limit_per_name
                ),
                "max_entity_names": self.eval_settings.evolink_max_entity_names,
                "max_event_seeds": self.eval_settings.evolink_max_event_seeds,
                "expansion_limit": self.eval_settings.evolink_expansion_limit,
                "link_weight": self.eval_settings.evolink_link_weight,
            }
            if "ours_association_only" in self.methods:
                expected["evolink_profile"]["mechanism_ablation"] = (
                    "association_only_disables_event_effect_bridge"
                )
        if D1_METHODS & set(self.methods):
            d1_methods = set(self.methods) & D1_METHODS
            if "ours_d1_queryrewrite_sharedlink" in d1_methods:
                expansion_scope = (
                    "mixed"
                    if "ours_d1_direct_noplan" in d1_methods
                    or "ours_d1_queryrewrite" in d1_methods
                    else "one_shared_expansion_after_all_routes"
                )
            elif "ours_d1_queryrewrite_sharedlink_total" in d1_methods:
                expansion_scope = "one_shared_expansion_fixed_evolink_cap"
            else:
                expansion_scope = "one_independent_expansion_per_route"
            expected["d1_profile"] = {
                "methods": sorted(d1_methods),
                "max_routes": 3,
                "route_budget": self.top_k,
                "candidate_multiplier": CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                "cross_path_expansion": expansion_scope,
                "fusion": "plain_rrf",
                "llm_rerank": False,
                "query_planner_calls": (
                    1
                    if {
                        "ours_d1_queryrewrite",
                        "ours_d1_queryrewrite_sharedlink",
                        "ours_d1_queryrewrite_sharedlink_total",
                    }
                    & set(self.methods)
                    else 0
                ),
            }
            if "ours_d1_queryrewrite_sharedlink" in d1_methods:
                expected["d1_profile"]["shared_expansion_budget"] = (
                    "route_budget_times_route_count"
                )
            if "ours_d1_queryrewrite_sharedlink_total" in d1_methods:
                expected["d1_profile"]["shared_expansion_budget"] = (
                    "fixed_evolink_expansion_cap"
                )
        if STANDARD_EVONEED_METHODS & set(self.methods):
            expected["evoneed_profile"] = {
                "version": EVONEED_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_prompt_profile_digest(),
                "knowledge_types": [
                    "text_evidence",
                    "relation_evidence",
                    "evolution_evidence",
                ],
                "selective_routes": ["vector", "graph", "event"],
                "joint_planner_calls_per_query": 1,
                "cross_path_expansion": "one_shared_evolink_after_all_routes",
                "query_composition": "original_question_plus_need_focus",
                "initial_retrieval_budget": self.eval_settings.evoneed_retrieval_budget,
                "minimum_active_route_budget": (
                    self.eval_settings.evoneed_min_route_budget
                ),
                "candidate_multiplier": CONTROLLED_ROUTE_CANDIDATE_MULTIPLIER,
                "final_output_k": self.top_k,
            }
            if "ours_evoneed_planned" in self.methods:
                expected["evoneed_profile"]["planned_initial_retrieval"] = {
                    "mode": "planned_needs",
                    "parallel_actions": True,
                    "max_planned_actions": 3,
                }
            if "ours_evoneed_planned_shared" in self.methods:
                expected["evoneed_profile"]["planned_shared_budget"] = {
                    "mode": "planned_needs",
                    "parallel_actions": True,
                    "max_planned_actions": 3,
                    "route_budget_mode": "shared_total",
                    "total_candidate_budget": "top_k",
                    "allocation": "planner_proposed_normalized_to_top_k",
                }
            if "ours_evoneed_equal" in self.methods:
                expected["evoneed_profile"]["equal_budget"] = {
                    "mode": "planned_needs",
                    "parallel_actions": True,
                    "max_planned_actions": 3,
                    "route_budget_mode": "equal_total",
                    "total_initial_budget": self.eval_settings.evoneed_retrieval_budget,
                }
            if "ours_evoneed_dynamic" in self.methods:
                expected["evoneed_profile"]["dynamic_budget"] = {
                    "mode": "planned_needs",
                    "parallel_actions": True,
                    "max_planned_actions": 3,
                    "route_budget_mode": "planned_total",
                    "total_initial_budget": self.eval_settings.evoneed_retrieval_budget,
                    "minimum_active_route_budget": (
                        self.eval_settings.evoneed_min_route_budget
                    ),
                    "allocation": "planner_proposed_normalized_to_total",
                }
        if "ours_evoneed_deep" in self.methods:
            expected["evoneed_deep_profile"] = {
                "version": EVONEED_DEEP_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "per_action",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 3,
                "cross_path_expansion": "one_evolink_per_need",
                "llm_stages": ["plan", "assess_and_rerank"],
                "final_output_k": self.top_k,
            }
        if "ours_evoneed_deep_oneplan" in self.methods:
            expected["evoneed_deep_oneplan_profile"] = {
                "version": EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_oneplan_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "per_action",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 1,
                "cross_path_expansion": "one_evolink_per_need",
                "llm_stages": ["joint_plan", "assess_and_rerank"],
                "secondary_query_planner_calls": 0,
                "final_output_k": self.top_k,
            }
        if "ours_evoneed_deep_oneplan_d1_norerank" in self.methods:
            expected["evoneed_deep_oneplan_d1_norerank_profile"] = {
                "version": EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_oneplan_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "per_action",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 1,
                "cross_path_expansion": "one_evolink_per_need",
                "need_fusion": "plain_rrf",
                "rrf_k": self.eval_settings.agentic_rrf_k,
                "llm_stages": ["joint_plan"],
                "secondary_query_planner_calls": 0,
                "llm_rerank": False,
                "final_output_k": self.top_k,
            }
        if "ours_evoneed_deep_oneplan_nolink_norerank" in self.methods:
            expected["evoneed_deep_oneplan_nolink_norerank_profile"] = {
                "version": EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_oneplan_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "per_action",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 1,
                "cross_path_expansion": "none",
                "need_fusion": "plain_rrf",
                "rrf_k": self.eval_settings.agentic_rrf_k,
                "llm_stages": ["joint_plan"],
                "secondary_query_planner_calls": 0,
                "llm_rerank": False,
                "final_output_k": self.top_k,
            }
        if "ours_evoneed_deep_oneplan_shared_equal_total_norerank" in self.methods:
            expected["evoneed_deep_oneplan_shared_equal_total_norerank_profile"] = {
                "version": EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_oneplan_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "per_action",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 1,
                "cross_path_expansion": "one_shared_expansion_equal_to_need_budget",
                "need_fusion": "plain_rrf",
                "rrf_k": self.eval_settings.agentic_rrf_k,
                "llm_stages": ["joint_plan"],
                "secondary_query_planner_calls": 0,
                "llm_rerank": False,
                "shared_expansion_budget": "sum_of_need_budgets",
                "final_output_k": self.top_k,
            }
        if "ours_evoneed_deep_oneplan_shared_total_norerank" in self.methods:
            expected["evoneed_deep_oneplan_shared_total_norerank_profile"] = {
                "version": EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
                "prompt_digest": evoneed_deep_oneplan_prompt_profile_digest(),
                "initial_retrieval": "planned_needs",
                "max_planned_actions": 3,
                "allow_repeated_routes": True,
                "query_composition": "planner_subquery_only",
                "route_budget_mode": "shared_total",
                "query_level_total_budget": "route_count_times_top_k",
                "candidate_budget_per_need": self.top_k,
                "candidate_multiplier": 1,
                "cross_path_expansion": "one_shared_expansion_fixed_evolink_cap",
                "need_fusion": "plain_rrf",
                "rrf_k": self.eval_settings.agentic_rrf_k,
                "llm_stages": ["joint_plan"],
                "secondary_query_planner_calls": 0,
                "llm_rerank": False,
                "evolink_expansion_cap": self.eval_settings.evolink_expansion_limit,
                "final_output_k": self.top_k,
            }
        if self.chunking_strategy != "fixed":
            expected["chunking_strategy"] = self.chunking_strategy
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != expected:
                raise RuntimeError(
                    "Output directory belongs to a different evaluation configuration. "
                    "Choose a new --output-dir."
                )
            return
        if any(self.output_dir.glob("*.jsonl")):
            raise RuntimeError(
                "Output directory contains checkpoint files but no run_manifest.json. "
                "Choose a new --output-dir."
            )
        path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_summary_csv(self, summaries: dict[str, Any]) -> None:
        keys = sorted({key for value in summaries.values() for key in value.get("metrics", {})})
        with (self.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", *keys])
            writer.writeheader()
            for method, value in summaries.items():
                writer.writerow({"method": method, **value.get("metrics", {})})

    def _write_summary_markdown(self, result: dict[str, Any]) -> None:
        preferred = [
            f"hits@{self.top_k}",
            f"evidence_recall@{self.top_k}",
            f"complete_recall@{self.top_k}",
            f"ndcg@{self.top_k}",
            f"mrr@{self.top_k}",
            "latency_p95_ms",
        ]
        lines = [
            f"# Retrieval evaluation: {self.dataset.name}",
            "",
            f"- Documents: {len(self.documents)}",
            f"- Chunks: {len(self.chunks)}",
            f"- Queries: {len(self.queries)}",
            f"- Fingerprint: `{self.fingerprint}`",
            "",
            "| Method | " + " | ".join(preferred) + " |",
            "|---|" + "---:|" * len(preferred),
        ]
        for method, summary in result["summaries"].items():
            metrics = summary.get("metrics", {})
            values = [f"{metrics[key]:.4f}" if key in metrics else "-" for key in preferred]
            lines.append(f"| {method} | " + " | ".join(values) + " |")
        (self.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def embedding_identity(settings: Settings) -> str:
    if settings.embedding_provider != "local":
        return f"{embedding_model_name(settings)}/{settings.embedding_dimension}d"
    # Preserve the original local-evaluation fingerprint so completed local
    # naive/hybrid checkpoints remain resumable after adding API providers.
    return settings.embedding_model_id


async def _retry_timeout(
    operation: Callable[[], Awaitable[Any]],
    *,
    attempts: int,
    label: str,
) -> Any:
    """Retry a resumable document operation after transient worker timeouts."""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await operation()
        except TimeoutError:
            if attempt >= attempts:
                raise
            delay = min(2 ** (attempt - 1), 10)
            logger.warning(
                "%s timed out; retrying in %ss (%s/%s)",
                label,
                delay,
                attempt,
                attempts,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable timeout retry state")


def light_entity_fingerprint(base_fingerprint: str, settings: Settings) -> str:
    prompt_path = settings.light_entity_prompt_file.resolve()
    prompt_digest = (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if prompt_path.is_file()
        else f"missing:{prompt_path}"
    )
    profile = (
        f"light_entity_v1\x1f{prompt_digest}\x1f"
        f"{settings.light_entity_max_gleaning}\x1f"
        f"{settings.light_entity_max_entities}\x1f"
        f"{settings.light_entity_max_relationships}\x1f"
        f"{settings.light_entity_max_records}"
    )
    return hashlib.sha256(f"{base_fingerprint}\x1f{profile}".encode()).hexdigest()


def lightrag_benchmark_fingerprint(base_fingerprint: str) -> str:
    return hashlib.sha256(
        f"{base_fingerprint}\x1f{LIGHTRAG_BENCHMARK_INPUT_VERSION}".encode()
    ).hexdigest()


def benchmark_fingerprint(
    name: str,
    documents,
    chunks: list[BenchmarkChunk],
    embedding_model_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str = "fixed",
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{name}\x1f{embedding_model_id}\x1f{chunk_size}\x1f{chunk_overlap}".encode())
    if chunking_strategy != "fixed":
        digest.update(f"\x1f{chunking_strategy}".encode())
    for document in documents:
        digest.update(document.document_id.encode())
        digest.update(hashlib.sha256(document.text.encode()).digest())
    digest.update(str(len(chunks)).encode())
    return digest.hexdigest()


def benchmark_chunk_file_path(document_id: str, chunk_id: str) -> str:
    """Give every pre-chunk a unique source while retaining its parent document."""
    return f"{document_id}__{chunk_id}__benchmark.md"


def lightrag_benchmark_chunk_text(chunk: BenchmarkChunk) -> str:
    """Keep equal benchmark passages distinct without changing dense retrieval text."""
    identity = (
        f"<!-- benchmark-document-id: {chunk.document_id}; benchmark-chunk-id: "
        f"{chunk.chunk_id} -->"
    )
    return f"{chunk.text}\n\n{identity}"


def event_document_fingerprint(document: BenchmarkDocument) -> str:
    content = f"{document.metadata_text()}\n\n{document.text}"
    return hashlib.sha256(content.encode()).hexdigest()


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def retrieval_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose whether each retrieval path actually reaches final Top-K evidence."""
    channel_query_counts = {channel: 0 for channel in ("vector", "graph", "event")}
    channel_document_counts = {channel: 0 for channel in channel_query_counts}
    evolink_runs: list[dict[str, Any]] = []
    for row in rows:
        channels_in_query: set[str] = set()
        retrieved = row.get("retrieved") or []
        for item in retrieved:
            channels = set((item.get("metadata") or {}).get("channels") or [])
            channels_in_query.update(channels)
            for channel in channels:
                if channel in channel_document_counts:
                    channel_document_counts[channel] += 1
        if retrieved:
            evolink = (retrieved[0].get("metadata") or {}).get("evolink")
            if isinstance(evolink, dict):
                evolink_runs.append(evolink)
        for channel in channels_in_query:
            if channel in channel_query_counts:
                channel_query_counts[channel] += 1
    query_count = len(rows)
    diagnostics = {
        "channel_query_counts": channel_query_counts,
        "channel_query_rates": {
            channel: count / query_count if query_count else 0.0
            for channel, count in channel_query_counts.items()
        },
        "mean_channel_documents": {
            channel: count / query_count if query_count else 0.0
            for channel, count in channel_document_counts.items()
        },
        "tkg_seed_counts": {
            kind: sum(
                int(
                    ((item.get("metadata") or {}).get("tkg_seed_counts") or {}).get(
                        kind, 0
                    )
                )
                for row in rows
                for item in (row.get("retrieved") or [])[:1]
            )
            for kind in ("entity", "predicate", "assertion", "event")
        },
    }
    if evolink_runs:
        evolink_query_count = len(evolink_runs)
        diagnostics["evolink_coordination"] = {
            "query_count": evolink_query_count,
            **{
                field: sum(int(item.get(field) or 0) for item in evolink_runs)
                for field in (
                    "seed_entity_count",
                    "seed_assertion_count",
                    "seed_event_count",
                    "linked_assertion_count",
                    "linked_event_count",
                    "linked_effect_count",
                )
            },
            "mean_graph_nodes": sum(
                int(item.get("graph_node_count") or 0) for item in evolink_runs
            )
            / evolink_query_count,
            "mean_graph_edges": sum(
                int(item.get("graph_edge_count") or 0) for item in evolink_runs
            )
            / evolink_query_count,
        }
    return diagnostics


def agentic_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = [
        item["trajectory"]
        for item in rows
        if not item.get("error") and isinstance(item.get("trajectory"), dict)
    ]
    if not trajectories:
        return {}
    count = len(trajectories)
    return {
        "query_count": count,
        "mean_retrieval_rounds": (
            sum(int(item.get("retrieval_rounds") or 0) for item in trajectories) / count
        ),
        "mean_llm_calls": sum(int(item.get("llm_calls") or 0) for item in trajectories) / count,
        "total_input_tokens": sum(
            int(item.get("input_tokens") or 0) for item in trajectories
        ),
        "total_output_tokens": sum(
            int(item.get("output_tokens") or 0) for item in trajectories
        ),
        "insufficient_evidence_rate": (
            sum(bool(item.get("insufficient_evidence")) for item in rows if not item.get("error"))
            / count
        ),
    }


def evoneed_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate typed gaps, selected routes, and need closure from trajectories."""
    route_action_counts = {route: 0 for route in ("vector", "graph", "event")}
    route_query_counts = {route: 0 for route in route_action_counts}
    knowledge_type_counts = {
        kind: 0
        for kind in (
            "text_evidence",
            "relation_evidence",
            "evolution_evidence",
        )
    }
    query_count = 0
    initial_open_total = 0
    final_open_total = 0
    sufficient_count = 0
    for row in rows:
        trajectory = row.get("trajectory")
        if row.get("error") or not isinstance(trajectory, dict):
            continue
        events = trajectory.get("events") or []
        initial_state = next(
            (
                item.get("need_state")
                for item in events
                if item.get("node") == "plan_needs"
            ),
            {},
        )
        final_state = next(
            (
                item.get("final_need_state")
                for item in reversed(events)
                if item.get("node") == "answer"
            ),
            {},
        )
        if not isinstance(initial_state, dict) or not isinstance(final_state, dict):
            continue
        query_count += 1
        initial_needs = initial_state.get("needs") or []
        final_needs = final_state.get("needs") or []
        initial_open_total += sum(
            item.get("status") != "satisfied" for item in initial_needs
        )
        final_open_total += sum(
            item.get("status") != "satisfied" for item in final_needs
        )
        sufficient_count += bool(final_state.get("sufficient"))
        for item in initial_needs:
            kind = item.get("knowledge_type")
            if kind in knowledge_type_counts:
                knowledge_type_counts[kind] += 1
        routes_in_query: set[str] = set()
        for event in events:
            if event.get("node") != "retrieve_needs":
                continue
            for action in event.get("actions") or []:
                route = action.get("route")
                if route in route_action_counts:
                    route_action_counts[route] += 1
                    routes_in_query.add(route)
        for route in routes_in_query:
            route_query_counts[route] += 1
    if not query_count:
        return {}
    return {
        "query_count": query_count,
        "initial_knowledge_type_counts": knowledge_type_counts,
        "route_action_counts": route_action_counts,
        "route_query_rates": {
            route: count / query_count for route, count in route_query_counts.items()
        },
        "mean_route_actions": sum(route_action_counts.values()) / query_count,
        "need_closure_rate": (
            max(0, initial_open_total - final_open_total) / initial_open_total
            if initial_open_total
            else 0.0
        ),
        "final_open_need_count": final_open_total,
        "sufficient_rate": sufficient_count / query_count,
    }


def rrf_by_document(
    first: list[RetrievedChunk],
    second: list[RetrievedChunk],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    selected: dict[str, RetrievedChunk] = {}
    for source, values in (("light_entity", first), ("event", second)):
        seen: set[str] = set()
        for rank, item in enumerate(values, start=1):
            key = item.document_id or item.chunk_id
            if key in seen:
                continue
            seen.add(key)
            fusion_rank = item.metadata.get("fusion_rank", rank)
            if not isinstance(fusion_rank, int) or fusion_rank < 1:
                fusion_rank = rank
            scores[key] += 1.0 / (rrf_k + fusion_rank)
            if key not in selected or source == "event":
                selected[key] = item
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=selected[key].chunk_id,
            document_id=selected[key].document_id,
            text=selected[key].text,
            score=scores[key],
            metadata={
                **selected[key].metadata,
                "fusion": "light_entity_event_rrf",
            },
        )
        for key in ordered
    ]


def _marker_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("fingerprint") == fingerprint


def _write_marker(path: Path, fingerprint: str, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"fingerprint": fingerprint, **extra}, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            result[str(value["query_id"])] = value
    return result
