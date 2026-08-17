import pytest

from rag_graph_demo.config import Settings
from rag_graph_demo.evaluation import runner as runner_module
from rag_graph_demo.evaluation.models import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
)
from rag_graph_demo.evaluation.runner import (
    EvaluationRunner,
    embedding_identity,
    evoneed_diagnostics,
    light_entity_fingerprint,
    lightrag_benchmark_chunk_text,
    lightrag_benchmark_fingerprint,
    retrieval_diagnostics,
)


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="fixture",
        documents=(BenchmarkDocument("doc-1", "Title", "Evidence text."),),
        queries=(BenchmarkQuery("q-1", "Question?", "fact", None, ()),),
    )


def test_light_entity_and_ours_share_controlled_index_configuration(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("light_entity", "ours"),
    )

    assert runner.eval_settings.lightrag_working_dir.name == "lightrag"
    assert runner.light_entity_settings.lightrag_working_dir.name == "lightrag_entity"
    assert runner.light_entity_settings.lightrag_workspace.startswith("eval_entity_")
    assert runner._method_retriever("light_entity", None, None).__func__ is (
        runner._retrieve_light_entity.__func__
    )
    assert runner._method_retriever("ours", None, None).__func__ is (
        runner._retrieve_ours.__func__
    )


def test_ours_evolink_is_a_distinct_controlled_three_path_method(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None, embedding_provider="zhipu"),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("ours_evolink",),
    )

    assert runner._method_retriever(
        "ours_evolink", None, None
    ).__func__ is runner._retrieve_ours_evolink.__func__
    assert runner.light_entity_engine is None
    assert runner.tkg_graph_store is None
    assert runner.evolink_graph_store is None


def test_evolink_agentic_methods_are_registered_without_a_new_index(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None, embedding_provider="zhipu"),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("agentic_ours_evolink", "ours_evoneed"),
    )

    assert runner._method_retriever(
        "agentic_ours_evolink", None, None
    ).__func__ is runner._retrieve_agentic_ours_evolink.__func__
    assert runner._method_retriever(
        "ours_evoneed", None, None
    ).__func__ is runner._retrieve_ours_evoneed.__func__
    assert runner.tkg_vector_index is None


def test_ours_tkg_has_no_controlled_lightrag_method_dependency(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None, embedding_provider="zhipu"),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("ours_tkg",),
    )

    assert runner.light_entity_engine is None
    assert runner.unified_orchestrator is None
    assert runner.tkg_retriever is None


def test_agentic_ours_tkg_has_no_legacy_graph_method_dependency(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None, embedding_provider="zhipu"),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("agentic_ours_tkg",),
    )

    assert runner.light_entity_engine is None
    assert runner.unified_orchestrator is None
    assert runner.event_service is None
    assert runner.tkg_retriever is None
    assert runner._method_retriever(
        "agentic_ours_tkg", None, None
    ).__func__ is runner._retrieve_agentic_ours_tkg.__func__


def test_open_schema_mode_is_part_of_tkg_manifest(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(
            _env_file=None,
            embedding_provider="zhipu",
            tkg_schema_mode="open",
        ),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("ours_tkg_dynamic",),
    )

    runner._ensure_output_manifest()
    manifest = __import__("json").loads(
        (tmp_path / "output" / "run_manifest.json").read_text()
    )
    assert manifest["tkg_profile"]["schema_mode"] == "open"
    assert manifest["tkg_profile"]["schema_version"] == "open-v1"
    assert manifest["tkg_profile"]["workspace"].endswith("_open")


@pytest.mark.asyncio
async def test_stock_and_controlled_lightrag_get_independent_embedding_clients(
    monkeypatch, tmp_path
) -> None:
    embeddings: list[object] = []
    engines = []

    def fake_build_embedding(_settings):
        embedding = object()
        embeddings.append(embedding)
        return embedding

    class FakeEngine:
        def __init__(self, _settings, _mapper, *, embedding_func, **_kwargs):
            self.embedding_func = embedding_func
            engines.append(self)

        async def complete_extraction(self, *_args, **_kwargs):
            return "{}"

        async def embed_texts(self, *_args, **_kwargs):
            return []

        async def finalize(self):
            return None

    async def fake_ensure_index(*_args, **_kwargs):
        return None

    async def fake_evaluate(*_args, **_kwargs):
        return []

    monkeypatch.setattr(runner_module, "build_embedding_func", fake_build_embedding)
    monkeypatch.setattr(runner_module, "LightRAGEngine", FakeEngine)
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("light", "light_entity"),
    )
    monkeypatch.setattr(runner, "_require_light_environment", lambda: None)
    monkeypatch.setattr(runner, "_ensure_light_index", fake_ensure_index)
    monkeypatch.setattr(runner, "_evaluate_method", fake_evaluate)

    await runner.run()

    assert len(embeddings) == 2
    assert len(engines) == 2
    assert engines[0].embedding_func is embeddings[0]
    assert engines[1].embedding_func is embeddings[1]
    assert engines[0].embedding_func is not engines[1].embedding_func


def test_bailian_identity_cannot_collide_with_local_embedding() -> None:
    local = Settings(_env_file=None, embedding_dimension=1024)
    bailian = Settings(
        _env_file=None,
        embedding_provider="bailian",
        bailian_embedding_model="text-embedding-v4",
        embedding_dimension=1024,
    )

    assert embedding_identity(local) != embedding_identity(bailian)
    assert embedding_identity(bailian) == "bailian/text-embedding-v4/1024d"

    zhipu = Settings(
        _env_file=None,
        embedding_provider="zhipu",
        zhipu_embedding_model="embedding-3",
        embedding_dimension=1024,
    )
    assert embedding_identity(zhipu) == "zhipu/embedding-3/1024d"


def test_relationship_hard_cap_changes_controlled_index_fingerprint(tmp_path) -> None:
    prompt = tmp_path / "prompts" / "entity_type" / "entity.yaml"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("entity_types_guidance: critical only\n", encoding="utf-8")
    base = Settings(
        _env_file=None,
        light_entity_prompt_file=prompt,
        light_entity_max_relationships=4,
    )
    changed = Settings(
        _env_file=None,
        light_entity_prompt_file=prompt,
        light_entity_max_relationships=3,
    )

    assert light_entity_fingerprint("base", base) != light_entity_fingerprint(
        "base", changed
    )


def test_lightrag_benchmark_identity_keeps_equal_chunk_text_distinct(tmp_path) -> None:
    dataset = BenchmarkDataset(
        name="duplicates",
        documents=(
            BenchmarkDocument("doc-1", "Same title", "Identical evidence."),
            BenchmarkDocument("doc-2", "Same title", "Identical evidence."),
        ),
        queries=(BenchmarkQuery("q-1", "Question?", "fact", None, ()),),
    )
    runner = EvaluationRunner(
        dataset,
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("light",),
    )

    assert runner.chunks[0].text == runner.chunks[1].text
    assert lightrag_benchmark_chunk_text(runner.chunks[0]) != lightrag_benchmark_chunk_text(
        runner.chunks[1]
    )
    assert runner.lightrag_fingerprint == lightrag_benchmark_fingerprint(runner.fingerprint)


@pytest.mark.asyncio
async def test_light_index_sends_unique_identity_for_duplicate_passages(tmp_path) -> None:
    dataset = BenchmarkDataset(
        name="duplicates",
        documents=(
            BenchmarkDocument("doc-1", "Same title", "Identical evidence."),
            BenchmarkDocument("doc-2", "Same title", "Identical evidence."),
        ),
        queries=(BenchmarkQuery("q-1", "Question?", "fact", None, ()),),
    )
    runner = EvaluationRunner(
        dataset,
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("light",),
        show_progress=False,
    )
    indexed: list[tuple[str, str, str]] = []

    class FakeEngine:
        async def index_documents(self, documents):
            indexed.extend(documents)

    await runner._ensure_light_index(
        FakeEngine(),
        label="light",
        marker=tmp_path / "marker.json",
        fingerprint=runner.lightrag_fingerprint,
    )

    assert len(indexed) == 2
    assert indexed[0][1] != indexed[1][1]
    assert "benchmark-document-id: doc-1" in indexed[0][1]
    assert "benchmark-document-id: doc-2" in indexed[1][1]


@pytest.mark.asyncio
async def test_event_index_checkpoint_retries_only_failed_documents(monkeypatch, tmp_path) -> None:
    dataset = BenchmarkDataset(
        name="events",
        documents=(
            BenchmarkDocument("doc-1", "One", "First event."),
            BenchmarkDocument("doc-2", "Two", "Second event."),
        ),
        queries=(BenchmarkQuery("q-1", "Question?", "fact", None, ()),),
    )
    runner = EvaluationRunner(
        dataset,
        Settings(_env_file=None, event_index_concurrency=1),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("ours",),
        show_progress=False,
    )
    runner.light_entity_engine = object()
    calls: list[str] = []
    should_fail = {"doc-2": True}

    class FakeRegistry:
        def __init__(self, _path):
            pass

        async def snapshot(self):
            return [], [], [], []

    class FakeExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

    class FakeService:
        def __init__(self, *_args, **_kwargs):
            pass

        async def index_document(self, document_id, _title, _text):
            calls.append(document_id)
            if should_fail.pop(document_id, False):
                raise ValueError("malformed event")

    monkeypatch.setattr(runner_module, "EventRegistry", FakeRegistry)
    monkeypatch.setattr(runner_module, "LLMEventExtractor", FakeExtractor)
    monkeypatch.setattr(runner_module, "EventService", FakeService)

    with pytest.raises(RuntimeError, match="doc-2"):
        await runner._ensure_event_index()
    await runner._ensure_event_index()

    assert calls == ["doc-1", "doc-2", "doc-2"]


def test_retrieval_diagnostics_reports_final_event_path_coverage() -> None:
    rows = [
        {
            "retrieved": [
                {"metadata": {"channels": ["vector", "graph", "event"]}},
                {"metadata": {"channels": ["vector"]}},
            ]
        },
        {"retrieved": [{"metadata": {"channels": ["vector", "graph"]}}]},
    ]

    result = retrieval_diagnostics(rows)

    assert result["channel_query_counts"] == {"vector": 2, "graph": 2, "event": 1}
    assert result["channel_query_rates"]["event"] == 0.5
    assert result["mean_channel_documents"]["vector"] == 1.5


def test_retrieval_diagnostics_reports_evolink_coordination() -> None:
    rows = [
        {
            "retrieved": [
                {
                    "metadata": {
                        "channels": ["vector", "graph", "event"],
                        "evolink": {
                            "seed_entity_count": 3,
                            "seed_assertion_count": 4,
                            "seed_event_count": 2,
                            "linked_assertion_count": 5,
                            "linked_event_count": 6,
                            "linked_effect_count": 1,
                            "graph_node_count": 12,
                            "graph_edge_count": 14,
                        },
                    }
                }
            ]
        },
        {
            "retrieved": [
                {
                    "metadata": {
                        "channels": ["vector", "graph"],
                        "evolink": {
                            "seed_entity_count": 2,
                            "linked_assertion_count": 3,
                            "graph_node_count": 8,
                            "graph_edge_count": 10,
                        },
                    }
                }
            ]
        },
    ]

    result = retrieval_diagnostics(rows)["evolink_coordination"]

    assert result["query_count"] == 2
    assert result["seed_entity_count"] == 5
    assert result["seed_assertion_count"] == 4
    assert result["linked_assertion_count"] == 8
    assert result["linked_event_count"] == 6
    assert result["mean_graph_nodes"] == 10
    assert result["mean_graph_edges"] == 12


def test_evoneed_diagnostics_reports_typed_gaps_routes_and_closure() -> None:
    rows = [
        {
            "trajectory": {
                "events": [
                    {
                        "node": "plan_needs",
                        "need_state": {
                            "needs": [
                                {
                                    "knowledge_type": "text_evidence",
                                    "status": "missing",
                                },
                                {
                                    "knowledge_type": "evolution_evidence",
                                    "status": "missing",
                                },
                            ]
                        },
                    },
                    {
                        "node": "retrieve_needs",
                        "actions": [
                            {"route": "vector"},
                            {"route": "event"},
                        ],
                    },
                    {
                        "node": "answer",
                        "final_need_state": {
                            "sufficient": False,
                            "needs": [
                                {"status": "satisfied"},
                                {"status": "missing"},
                            ],
                        },
                    },
                ]
            }
        }
    ]

    result = evoneed_diagnostics(rows)

    assert result["route_action_counts"] == {
        "vector": 1,
        "graph": 0,
        "event": 1,
    }
    assert result["initial_knowledge_type_counts"]["evolution_evidence"] == 1
    assert result["need_closure_rate"] == 0.5
    assert result["sufficient_rate"] == 0
