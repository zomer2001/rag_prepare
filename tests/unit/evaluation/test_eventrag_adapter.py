from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_adapter() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "run_eventrag_baseline.py"
    spec = importlib.util.spec_from_file_location("run_eventrag_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_source_texts_handles_multiline_csv() -> None:
    module = load_adapter()
    context = '''
-----Entities-----
```csv
id,entity
0,Example
```
-----Sources-----
```csv
id,content
0,"first line
second line"
1,"another source"
```
'''

    assert module.extract_source_texts(context) == [
        "first line\nsecond line",
        "another source",
    ]


def test_extract_source_texts_handles_eventrag_combined_format() -> None:
    module = load_adapter()
    context = '''
-----Sources-----
```csv
id,\tcontent
1,\tTitle: First, source
Body line one.
Body line two.
2,\tTitle: Second source
Second body.
```
'''

    assert module.extract_source_texts(context) == [
        "Title: First, source\nBody line one.\nBody line two.",
        "Title: Second source\nSecond body.",
    ]


def test_corpus_lookup_prefers_exact_then_normalized() -> None:
    module = load_adapter()
    lookup = module.CorpusLookup(
        [{"text": "A  source\nline", "chunk_id": "c1", "document_id": "d1"}]
    )

    exact, exact_type = lookup.resolve("A  source\nline")
    normalized, normalized_type = lookup.resolve("A source line")

    assert exact["chunk_id"] == "c1"
    assert exact_type == "exact"
    assert normalized["chunk_id"] == "c1"
    assert normalized_type == "normalized"


def test_truncate_extraction_response_keeps_first_records() -> None:
    module = load_adapter()
    response = "##".join(
        [
            '("entity"<|>"E1"<|>"Event"<|>"one")',
            '("entity"<|>"E2"<|>"Actor"<|>"two")',
            '("entity"<|>"E3"<|>"Actor"<|>"three")',
            '("relationship"<|>"E1"<|>"E2"<|>"R"<|>"d"<|>"k"<|>1)',
            '("relationship"<|>"E2"<|>"E3"<|>"R"<|>"d"<|>"k"<|>1)',
            '("content_keywords"<|>"topic")<|COMPLETE|>',
        ]
    )

    result = module.truncate_extraction_response(
        response,
        max_entities=2,
        max_relationships=1,
    )

    assert '"E1"' in result
    assert '"E2"' in result
    assert '"E3"' not in result
    assert result.count('"relationship"') == 1
    assert result.endswith("<|COMPLETE|>")


@pytest.mark.asyncio
async def test_extraction_response_is_reused_after_restart(tmp_path: Path) -> None:
    module = load_adapter()
    cache_path = tmp_path / "extraction_response_cache.jsonl"

    def make_clients():
        return module.ModelClients(
            deepseek_key="test-key",
            deepseek_base_url="https://example.invalid",
            deepseek_model="test-model",
            zhipu_key="test-key",
            zhipu_base_url="https://example.invalid",
            zhipu_model="test-embedding",
            dimensions=8,
            llm_concurrency=1,
            embedding_concurrency=1,
            extraction_max_tokens=100,
            max_entities_per_chunk=2,
            max_relationships_per_chunk=1,
            extraction_cache_path=cache_path,
        )

    first = make_clients()
    api_calls = 0

    async def create_response(**_kwargs):
        nonlocal api_calls
        api_calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='("entity"<|>"E1"<|>"Event"<|>"one")<|COMPLETE|>'
                    )
                )
            ]
        )

    first._deepseek.chat.completions.create = create_response
    prompt = "-Budgeted Baseline Rules-\nExtract this chunk."
    expected = await first.complete(prompt)

    restarted = make_clients()

    async def unexpected_api_call(**_kwargs):
        raise AssertionError("cached extraction should not call the API")

    restarted._deepseek.chat.completions.create = unexpected_api_call
    actual = await restarted.complete(prompt)

    assert api_calls == 1
    assert actual == expected
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1


def test_response_cache_ignores_partial_final_record(tmp_path: Path) -> None:
    module = load_adapter()
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        '{"key": "complete", "content": "saved"}\n{"key": "partial"',
        encoding="utf-8",
    )

    assert module.load_response_cache(cache_path) == {"complete": "saved"}


def test_completed_index_manifest_allows_only_explicit_query_change(tmp_path: Path) -> None:
    module = load_adapter()
    path = tmp_path / "adapter_manifest.json"
    original = {"corpus_sha256": "same", "query_sha256": "q100", "queries": 100}
    path.write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "index_complete.json").write_text("{}", encoding="utf-8")
    gold500 = {**original, "query_sha256": "q500", "queries": 500}

    with pytest.raises(RuntimeError):
        module.write_or_validate_manifest(path, gold500)
    module.write_or_validate_manifest(path, gold500, allow_query_set_change=True)

    with pytest.raises(RuntimeError):
        module.write_or_validate_manifest(
            path,
            {**gold500, "corpus_sha256": "different"},
            allow_query_set_change=True,
        )
