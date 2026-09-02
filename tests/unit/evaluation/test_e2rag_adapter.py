from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

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
