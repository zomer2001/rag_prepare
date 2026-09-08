from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "run_e2rag_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_e2rag_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_source_texts_handles_official_csv() -> None:
    context = '''
-----Sources-----
```csv
id,content
0,"first line\nsecond line"
1,another source
```
'''
    assert MODULE.extract_source_texts(context) == ["first line\nsecond line", "another source"]


def test_corpus_lookup_resolves_text_and_source_id() -> None:
    corpus = [
        {
            "text": "Alpha\n beta",
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
        }
    ]
    lookup = MODULE.CorpusLookup(corpus)
    item, match_type = lookup.by_text("Alpha beta")
    assert item == corpus[0]
    assert match_type == "normalized"
    source_id = "chunk-" + hashlib.md5(b"Alpha\n beta").hexdigest()  # noqa: S324
    assert lookup.by_source_id(source_id) == corpus[0]


def test_upstream_output_is_saved_outside_terminal(tmp_path, capsys) -> None:
    log_path = tmp_path / "e2rag_upstream.log"

    with MODULE.upstream_output(log_path, visible=False):
        print("ordinary upstream output")
        print("upstream diagnostic", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "ordinary upstream output",
        "upstream diagnostic",
    ]


def make_clients(cache_path: Path):
    return MODULE.ModelClients(
        deepseek_key="test-key",
        deepseek_base_url="https://example.invalid",
        deepseek_model="test-model",
        zhipu_key="test-key",
        zhipu_base_url="https://example.invalid",
        zhipu_model="test-embedding",
        dimensions=8,
        llm_concurrency=1,
        embedding_concurrency=1,
        response_cache_path=cache_path,
    )


@pytest.mark.asyncio
async def test_llm_response_is_reused_after_restart(tmp_path: Path) -> None:
    cache_path = tmp_path / "response_cache.jsonl"
    first = make_clients(cache_path)
    api_calls = 0

    async def create_response(**_kwargs):
        nonlocal api_calls
        api_calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="saved response"))]
        )

    first.deepseek.chat.completions.create = create_response
    assert await first.complete("Extract this chunk") == "saved response"

    restarted = make_clients(cache_path)

    async def unexpected_api_call(**_kwargs):
        raise AssertionError("cached request should not call the API")

    restarted.deepseek.chat.completions.create = unexpected_api_call
    assert await restarted.complete("Extract this chunk") == "saved response"
    assert api_calls == 1


def test_response_cache_ignores_partial_final_record(tmp_path: Path) -> None:
    cache_path = tmp_path / "response_cache.jsonl"
    cache_path.write_text(
        '{"key": "complete", "content": "saved"}\n{"key": "partial"',
        encoding="utf-8",
    )

    assert MODULE.load_response_cache(cache_path) == {"complete": "saved"}


def test_atomic_graph_writer_preserves_last_valid_file_on_failure(tmp_path: Path) -> None:
    class FakeStorage:
        @staticmethod
        def write_nx_graph(graph, file_name):
            Path(file_name).write_text("incomplete", encoding="utf-8")
            raise RuntimeError("interrupted")

    target = tmp_path / "graph.graphml"
    nx.write_graphml(nx.path_graph(2), target)
    original = target.read_bytes()
    MODULE.install_atomic_graphml_writer(FakeStorage)

    with pytest.raises(RuntimeError, match="interrupted"):
        FakeStorage.write_nx_graph(nx.path_graph(3), target)

    assert target.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_runtime_manifest_can_reuse_index_for_a_different_query_set(tmp_path: Path) -> None:
    manifest_path = tmp_path / "adapter_manifest.json"
    original = {
        "corpus_sha256": "same-corpus",
        "embedding_model": "same-model",
        "queries_sha256": "q100",
        "query_override_sha256": None,
        "queries": 100,
    }
    full_queries = {
        **original,
        "queries_sha256": "q2556",
        "queries": 2556,
    }
    MODULE.write_or_validate_manifest(manifest_path, original)

    with pytest.raises(RuntimeError, match="Runtime manifest differs"):
        MODULE.write_or_validate_manifest(manifest_path, full_queries)

    MODULE.write_or_validate_manifest(
        manifest_path,
        full_queries,
        allow_query_set_change=True,
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_query_set_override_still_rejects_index_changes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "adapter_manifest.json"
    original = {
        "corpus_sha256": "same-corpus",
        "embedding_model": "same-model",
        "queries_sha256": "q100",
        "query_override_sha256": None,
        "queries": 100,
    }
    MODULE.write_or_validate_manifest(manifest_path, original)

    with pytest.raises(RuntimeError, match="Runtime manifest differs"):
        MODULE.write_or_validate_manifest(
            manifest_path,
            {
                **original,
                "corpus_sha256": "different-corpus",
                "queries_sha256": "q2556",
                "queries": 2556,
            },
            allow_query_set_change=True,
        )


def test_retrieval_manifest_is_adjacent_to_raw_output(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    assert MODULE.retrieval_manifest_path(output) == tmp_path / "raw.jsonl.manifest.json"
