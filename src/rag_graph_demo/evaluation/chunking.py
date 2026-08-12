from __future__ import annotations

import hashlib
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from lightrag.chunker import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer

from rag_graph_demo.evaluation.models import BenchmarkChunk, BenchmarkDocument

ChunkingStrategy = Literal["fixed", "recursive"]
RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]


def build_chunks(
    documents: tuple[BenchmarkDocument, ...],
    *,
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkingStrategy = "fixed",
) -> list[BenchmarkChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if strategy not in {"fixed", "recursive"}:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    tokenizer = TiktokenTokenizer()
    result: list[BenchmarkChunk] = []
    for document in documents:
        if strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=lambda value: len(tokenizer.encode(value)),
                separators=RECURSIVE_SEPARATORS,
                keep_separator="end",
                strip_whitespace=True,
            )
            pieces = [{"content": value} for value in splitter.split_text(document.text)]
        else:
            pieces = chunking_by_token_size(
                tokenizer,
                document.text,
                chunk_overlap_token_size=chunk_overlap,
                chunk_token_size=chunk_size,
            )
        metadata = document.metadata_text()
        for order, piece in enumerate(pieces):
            text = f"{metadata}\n\n{piece['content']}".strip()
            digest = hashlib.sha256(
                f"{document.document_id}\x1f{order}\x1f{text}".encode()
            ).hexdigest()[:20]
            result.append(
                BenchmarkChunk(
                    chunk_id=f"chunk-{digest}",
                    document_id=document.document_id,
                    text=text,
                    order=order,
                )
            )
    return result
