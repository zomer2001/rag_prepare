from types import SimpleNamespace

import numpy as np
import pytest

from rag_graph_demo.config import Settings
from rag_graph_demo.lightrag_engine import local_models
from rag_graph_demo.lightrag_engine.local_models import (
    BailianEmbeddingModel,
    ZhipuEmbeddingModel,
    build_embedding_func,
)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 3.0]),
                SimpleNamespace(index=0, embedding=[4.0, 0.0]),
            ]
        )


class FakeAsyncOpenAI:
    init_kwargs = None
    embeddings_instance = FakeEmbeddings()

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = kwargs
        self.embeddings = type(self).embeddings_instance


@pytest.mark.asyncio
async def test_bailian_embedding_uses_compatible_api_and_normalizes(monkeypatch) -> None:
    monkeypatch.setattr(local_models, "AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(
        _env_file=None,
        embedding_provider="bailian",
        embedding_dimension=2,
        bailian_api_key="test-key",
        bailian_embedding_model="text-embedding-v4",
    )

    result = await BailianEmbeddingModel(settings).encode(["first", "second"], "query")

    assert np.allclose(result, [[1.0, 0.0], [0.0, 1.0]])
    assert FakeAsyncOpenAI.init_kwargs["api_key"] == "test-key"
    assert FakeAsyncOpenAI.init_kwargs["base_url"].endswith("/compatible-mode/v1")
    assert FakeAsyncOpenAI.embeddings_instance.request == {
        "model": "text-embedding-v4",
        "input": ["first", "second"],
        "encoding_format": "float",
        "dimensions": 2,
    }


@pytest.mark.asyncio
async def test_bailian_embedding_requires_key() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="bailian",
        embedding_dimension=2,
        bailian_api_key=None,
    )

    with pytest.raises(ValueError, match="BAILIAN_API_KEY"):
        await BailianEmbeddingModel(settings).encode(["text"])


@pytest.mark.asyncio
async def test_zhipu_embedding_uses_glm_compatible_api(monkeypatch) -> None:
    FakeAsyncOpenAI.embeddings_instance = FakeEmbeddings()
    monkeypatch.setattr(local_models, "AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(
        _env_file=None,
        embedding_provider="zhipu",
        embedding_dimension=2,
        zhipu_api_key="test-key",
        zhipu_embedding_model="embedding-3",
    )

    result = await ZhipuEmbeddingModel(settings).encode(["first", "second"])
    embedding_func = build_embedding_func(settings)

    assert np.allclose(result, [[1.0, 0.0], [0.0, 1.0]])
    assert FakeAsyncOpenAI.init_kwargs["base_url"] == (
        "https://open.bigmodel.cn/api/paas/v4"
    )
    assert FakeAsyncOpenAI.embeddings_instance.request == {
        "model": "embedding-3",
        "input": ["first", "second"],
        "dimensions": 2,
    }
    assert embedding_func.model_name == "zhipu/embedding-3"
    assert embedding_func.max_token_size == 3072
