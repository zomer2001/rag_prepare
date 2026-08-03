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
    rewrite_reference_titles,
    select_answer_citations,
)
from rag_graph_demo.storage.document_registry import DocumentRegistry


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
    complete = LightRAGEngine(settings, mapper)._build_deepseek_func(
        "deepseek-v4-flash"
    )

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
async def test_deepseek_adapter_retries_transient_connections(
    monkeypatch, tmp_path
) -> None:
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
                "error_msg": "vector flush failed"
                if self.status == DocStatus.FAILED
                else None,
            },
        )()
        return {"doc-1": status}


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
