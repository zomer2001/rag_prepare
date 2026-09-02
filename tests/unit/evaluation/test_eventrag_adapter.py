from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
