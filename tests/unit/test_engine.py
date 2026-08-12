import asyncio

import pytest
from lightrag.base import DocStatus
from openai import APIConnectionError

from rag_graph_demo.config import Settings
from rag_graph_demo.domain.exceptions import EngineNotReadyError
from rag_graph_demo.domain.models import Citation
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine import engine as engine_module
from rag_graph_demo.lightrag_engine.engine import (
    LightRAGEngine,
    limit_controlled_extraction_result,
    rewrite_reference_titles,
    select_answer_citations,
)
from rag_graph_demo.storage.document_registry import DocumentRegistry


class InitializableFakeRag:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def initialize_storages(self) -> None:
        return None


@pytest.mark.asyncio
async def test_light_entity_profile_applies_prompt_and_precision_limits(
    monkeypatch, tmp_path
) -> None:
    prompt_file = tmp_path / "prompts" / "entity_type" / "entity.yaml"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text(
        "entity_types_guidance: core only\n"
        "entity_extraction_json_examples:\n"
        '  - \'{"entities": [], "relationships": []}\'\n',
        encoding="utf-8",
    )
    captured = {}

    def fake_lightrag(**kwargs):
        captured.update(kwargs)
        return InitializableFakeRag(**kwargs)

    async def fake_embedding(_texts, context="document"):
        del context
        return [[1.0]]

    def fake_llm_builder(_model):
        async def complete(*_args, **_kwargs):
            return "ok"

        return complete

    monkeypatch.setattr(engine_module, "LightRAG", fake_lightrag)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        lightrag_working_dir=tmp_path / "lightrag",
        deepseek_api_key=None,
        embedding_dimension=1,
        light_entity_prompt_file=prompt_file,
        light_entity_max_gleaning=0,
        light_entity_max_entities=9,
        light_entity_max_records=21,
    )
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    embedding = engine_module.EmbeddingFunc(
        embedding_dim=1,
        max_token_size=16,
        model_name="fake",
        func=fake_embedding,
    )

    await LightRAGEngine(
        settings,
        mapper,
        embedding_func=embedding,
        llm_func_builder=fake_llm_builder,
        extraction_profile="light_entity",
    ).initialize()

    assert captured["entity_extract_max_gleaning"] == 0
    assert captured["entity_extract_max_entities"] == 9
    assert captured["entity_extract_max_records"] == 21
    assert captured["embedding_func_max_async"] == settings.effective_embedding_concurrency
    assert captured["max_parallel_insert"] == settings.graph_index_concurrency
    assert captured["addon_params"]["entity_type_prompt_file"] == "entity.yaml"
    assert engine_module.os.environ["PROMPT_DIR"] == str(prompt_file.parent.parent.resolve())


def test_controlled_extraction_hard_caps_entities_and_valid_relationships() -> None:
    value = {
        "entities": [
            {"name": name, "type": "Concept", "description": name * 20}
            for name in ("A", "B", "C", "D", "E")
        ],
        "relationships": [
            {"source": "A", "target": target, "description": f"A to {target}"}
            for target in ("B", "C", "D", "E", "B")
        ],
    }

    limited = limit_controlled_extraction_result(
        value,
        max_entities=4,
        max_relationships=4,
    )

    assert [item["name"] for item in limited["entities"]] == ["A", "B", "C", "D"]
    assert len(limited["relationships"]) == 4
    assert all(item["target"] != "E" for item in limited["relationships"])
    assert len(limited["entities"][0]["description"]) == 20


@pytest.mark.asyncio
async def test_deepseek_adapter_owns_connection_arguments(monkeypatch, tmp_path) -> None:
    captured = {}

    async def fake_complete(model, prompt, **kwargs):
        captured.update({"model": model, "prompt": prompt, **kwargs})
        return "ok"

    monkeypatch.setattr(engine_module, "openai_complete_if_cache", fake_complete)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        lightrag_working_dir=tmp_path / "lightrag",
        deepseek_api_key="test-key",
        deepseek_thinking_enabled=True,
        llm_request_timeout_seconds=120,
    )
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    complete = LightRAGEngine(settings, mapper)._build_deepseek_func("deepseek-v4-flash")

    response = await complete(
        "question",
        model="ignored",
        enable_cot=False,
        base_url="https://ignored.example",
        api_key="ignored",
        timeout=1,
    )

    assert response == "ok"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["enable_cot"] is True
    assert captured["timeout"] == 120
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.asyncio
async def test_deepseek_adapter_retries_transient_connections(monkeypatch, tmp_path) -> None:
    calls = 0

    async def fake_complete(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(request=None)
        return "ok"

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(engine_module, "openai_complete_if_cache", fake_complete)
    monkeypatch.setattr(engine_module.asyncio, "sleep", no_wait)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        lightrag_working_dir=tmp_path / "lightrag",
        deepseek_api_key="test-key",
        llm_max_retries=2,
    )
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    complete = LightRAGEngine(settings, mapper)._build_deepseek_func("model")

    assert await complete("question") == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_direct_event_extraction_honors_llm_concurrency(tmp_path) -> None:
    active = 0
    maximum_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    def fake_llm_builder(_model):
        async def complete(*_args, **_kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if maximum_active == 2:
                two_started.set()
            await release.wait()
            active -= 1
            return '{"events": []}'

        return complete

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        lightrag_working_dir=tmp_path / "lightrag",
        llm_max_concurrency=2,
    )
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper, llm_func_builder=fake_llm_builder)
    tasks = [
        asyncio.create_task(engine.complete_extraction("prompt", "system"))
        for _ in range(3)
    ]
    await asyncio.wait_for(two_started.wait(), timeout=1)

    assert maximum_active == 2
    release.set()
    assert await asyncio.gather(*tasks) == [
        '{"events": []}',
        '{"events": []}',
        '{"events": []}',
    ]
    assert maximum_active == 2


class FakeRag:
    def __init__(self, status: DocStatus) -> None:
        self.status = status
        self.warmup_calls = 0

    async def embedding_func(self, texts, context):
        self.warmup_calls += 1
        return [[1.0] for _ in texts]

    async def ainsert(self, **_):
        return "track-1"

    async def aget_docs_by_track_id(self, _):
        status = type(
            "Status",
            (),
            {
                "status": self.status,
                "error_msg": "vector flush failed" if self.status == DocStatus.FAILED else None,
            },
        )()
        return {"doc-1": status}

    async def aquery_data(self, question, param):
        return {
            "status": "success",
            "data": {"chunks": [{"content": question, "chunk_id": param.mode}]},
        }


class FakeDocStatus:
    def __init__(self, values):
        self.values = values

    async def get_by_ids(self, ids):
        return [self.values.get(item) for item in ids]


class SemanticFakeGraph:
    async def get_all_nodes(self):
        return [
            {
                "id": "Acme",
                "entity_type": "Organization",
                "description": "Aggregated description from multiple sources.",
                "source_id": "chunk-other<SEP>chunk-doc-1",
                "file_path": "doc-other__other.md<SEP>doc-1__news.md",
            },
            {
                "id": "Lattice",
                "entity_type": "Model",
                "description": "Aggregated model description.",
                "source_id": "chunk-doc-1",
                "file_path": "doc-1__news.md",
            },
        ]

    async def get_all_edges(self):
        return [
            {
                "source": "Acme",
                "target": "Lattice",
                "description": "Aggregated relation description.",
                "keywords": "development",
                "source_id": "chunk-other<SEP>chunk-doc-1",
                "file_path": "doc-other__other.md<SEP>doc-1__news.md",
            }
        ]


class SemanticFakeChunks:
    async def get_by_ids(self, ids):
        values = {
            "chunk-other": {
                "content": "Other source text.",
                "file_path": "doc-other__other.md",
            },
            "chunk-doc-1": {
                "content": "Acme develops Lattice in the source chunk.",
                "file_path": "doc-1__news.md",
            },
        }
        return [values.get(item) for item in ids]


class SemanticFakeRag:
    chunk_entity_relation_graph = SemanticFakeGraph()
    text_chunks = SemanticFakeChunks()


@pytest.mark.asyncio
async def test_semantic_extraction_uses_document_chunk_as_mention_evidence(tmp_path) -> None:
    settings = Settings(_env_file=None, deepseek_api_key="test")
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper)
    engine._rag = SemanticFakeRag()

    result = await engine.get_document_semantic_extraction("doc-1")

    assert result.entities[0].description == "Acme develops Lattice in the source chunk."
    assert result.relationships[0].description == (
        "Acme develops Lattice in the source chunk."
    )
    assert result.relationships[0].source_segment_id == "chunk-doc-1"


class ResumableFakeRag(FakeRag):
    def __init__(self) -> None:
        super().__init__(DocStatus.PROCESSED)
        self.doc_status = FakeDocStatus({"doc-1": {"status": DocStatus.PROCESSED.value}})
        self.inserted_ids = []

    async def ainsert(self, **kwargs):
        self.inserted_ids.extend(kwargs["ids"])
        for item in kwargs["ids"]:
            self.doc_status.values[item] = {"status": DocStatus.PROCESSED.value}
        return "track-resumed"

    async def apipeline_process_enqueue_documents(self):
        return None


@pytest.mark.asyncio
async def test_index_document_checks_lightrag_final_status(tmp_path) -> None:
    settings = Settings(_env_file=None, deepseek_api_key="test")
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper)
    fake_rag = FakeRag(DocStatus.PROCESSED)
    engine._rag = fake_rag

    track_id = await engine.index_document(
        document_id="doc-1",
        text="content",
        file_path="doc-1__news.md",
    )

    assert track_id == "track-1"
    assert fake_rag.warmup_calls == 1


@pytest.mark.asyncio
async def test_index_document_rejects_failed_lightrag_status(tmp_path) -> None:
    settings = Settings(_env_file=None, deepseek_api_key="test")
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper)
    engine._rag = FakeRag(DocStatus.FAILED)

    with pytest.raises(EngineNotReadyError, match="vector flush failed"):
        await engine.index_document(
            document_id="doc-1",
            text="content",
            file_path="doc-1__news.md",
        )


@pytest.mark.asyncio
async def test_index_documents_skips_already_processed_chunks(tmp_path) -> None:
    settings = Settings(_env_file=None, deepseek_api_key="test")
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper)
    fake_rag = ResumableFakeRag()
    engine._rag = fake_rag

    track_id = await engine.index_documents(
        [
            ("doc-1", "existing", "doc-1.md"),
            ("doc-2", "new", "doc-2.md"),
        ]
    )

    assert track_id == "track-resumed"
    assert fake_rag.inserted_ids == ["doc-2"]


def test_answer_citations_are_filtered_and_titled() -> None:
    citations = [
        Citation(
            reference_id="1",
            title="可读标题",
            filename="news.md",
            file_path="/tmp/news.md",
        ),
        Citation(
            reference_id="2",
            title="未引用标题",
            filename="other.md",
            file_path="/tmp/other.md",
        ),
    ]
    answer = "事实 [1]\n\n### References\n\n- [1] internal-id__news.md"

    selected = select_answer_citations(answer, citations)
    rewritten = rewrite_reference_titles(answer, selected)

    assert [item.reference_id for item in selected] == ["1"]
    assert rewritten.endswith("- [1] 可读标题")


@pytest.mark.asyncio
async def test_retrieve_data_skips_answer_generation(tmp_path) -> None:
    settings = Settings(_env_file=None, deepseek_api_key="test")
    mapper = SourceMapper(DocumentRegistry(tmp_path / "documents.json"))
    engine = LightRAGEngine(settings, mapper)
    engine._rag = FakeRag(DocStatus.PROCESSED)

    result = await engine.retrieve_data("question", mode="mix", chunk_top_k=5)

    assert result["status"] == "success"
    assert result["data"]["chunks"][0]["content"] == "question"
