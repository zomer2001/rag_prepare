from lightrag.utils import TiktokenTokenizer

from rag_graph_demo.evaluation.chunking import build_chunks
from rag_graph_demo.evaluation.models import BenchmarkDocument


def test_recursive_chunking_prefers_sentence_boundaries() -> None:
    document = BenchmarkDocument(
        "doc-1",
        "Boundary test",
        "Alpha has a complete first sentence. Beta has a complete second sentence. "
        "Gamma has a complete third sentence.",
    )

    chunks = build_chunks(
        (document,),
        chunk_size=10,
        chunk_overlap=0,
        strategy="recursive",
    )

    bodies = [chunk.text.rsplit("\n\n", 1)[-1] for chunk in chunks]
    assert len(chunks) > 1
    assert all(body.endswith(".") for body in bodies)


def test_recursive_chunks_respect_token_budget_before_metadata() -> None:
    document = BenchmarkDocument(
        "doc-1",
        "Budget test",
        "A very long uninterrupted value " + "token " * 100,
    )
    tokenizer = TiktokenTokenizer()

    chunks = build_chunks(
        (document,),
        chunk_size=20,
        chunk_overlap=4,
        strategy="recursive",
    )

    bodies = [chunk.text.rsplit("\n\n", 1)[-1] for chunk in chunks]
    assert len(chunks) > 1
    assert all(len(tokenizer.encode(body)) <= 20 for body in bodies)
