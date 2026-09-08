from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def load_adapter() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "run_hipporag_baseline.py"
    spec = importlib.util.spec_from_file_location("run_hipporag_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exporter() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "export_hipporag_dataset.py"
    spec = importlib.util.spec_from_file_location("export_hipporag_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_limited_llm_disables_deepseek_thinking() -> None:
    module = load_adapter()

    class Delegate:
        def infer(self, **kwargs):
            return kwargs

    result = module.LimitedLLM(Delegate(), max_concurrency=1).infer(messages=[])

    assert result["extra_body"] == {"thinking": {"type": "disabled"}}


def test_zhipu_adapter_requests_and_validates_dimensions() -> None:
    module = load_adapter()
    captured = {}

    class Embeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            data = [SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index in range(2)]
            return SimpleNamespace(data=data)

    class BaseEmbedding:
        def __init__(self):
            self.embedding_model_name = "embedding-3"
            self.client = SimpleNamespace(embeddings=Embeddings())

    embedding_class = module.build_zhipu_embedding_class(BaseEmbedding, dimensions=2)
    result = embedding_class().encode(["first", "second"])

    assert captured == {
        "input": ["first", "second"],
        "model": "embedding-3",
        "dimensions": 2,
    }
    np.testing.assert_array_equal(result, np.asarray([[1.0, 0.0], [1.0, 0.0]]))


def test_exporter_uses_normalized_query_ids_when_raw_ids_are_missing(tmp_path: Path) -> None:
    exporter = load_exporter()
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_dir.mkdir()
    (dataset_dir / "corpus.jsonl").write_text(
        json.dumps({"title": "Document A", "text": "Evidence text."}) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "queries.jsonl").write_text(
        json.dumps(
            {
                "query": "What is the evidence?",
                "answer": "Evidence text.",
                "evidence_list": [{"title": "Document A"}],
                "question_type": "inference_query",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exporter.export_dataset(dataset_dir, output_dir, "sample")

    queries = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert queries[0]["id"] == "q-00001"
    assert queries[0]["question"] == "What is the evidence?"


def test_manifest_allows_only_query_change_when_explicit(tmp_path: Path) -> None:
    module = load_adapter()
    path = tmp_path / "adapter_manifest.json"
    original = {"corpus_sha256": "same", "query_sha256": "q100", "queries": 100}
    gold500 = {**original, "query_sha256": "q500", "queries": 500}
    module.write_or_validate_manifest(path, original)

    with pytest.raises(RuntimeError):
        module.write_or_validate_manifest(path, gold500)
    module.write_or_validate_manifest(path, gold500, allow_query_set_change=True)

    with pytest.raises(RuntimeError):
        module.write_or_validate_manifest(
            path,
            {**gold500, "corpus_sha256": "different"},
            allow_query_set_change=True,
        )
