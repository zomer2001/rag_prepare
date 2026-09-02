from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def load_adapter() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "run_hipporag_baseline.py"
    spec = importlib.util.spec_from_file_location("run_hipporag_baseline", path)
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
