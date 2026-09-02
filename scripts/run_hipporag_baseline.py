#!/usr/bin/env python3
"""Run HippoRAG as an isolated retrieval baseline with DeepSeek and Zhipu APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported environment variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        try:
            parsed = shlex.split(value, comments=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip("'\"")
        os.environ.setdefault(key, value)


@contextmanager
def temporary_openai_key(api_key: str) -> Iterator[None]:
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


class LimitedLLM:
    """Apply DeepSeek API controls while preserving HippoRAG's cached LLM behavior."""

    def __init__(self, delegate: Any, max_concurrency: int) -> None:
        self._delegate = delegate
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra_body
        with self._semaphore:
            return self._delegate.infer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def import_hipporag(hipporag_root: Path) -> dict[str, Any]:
    source_dir = hipporag_root.resolve() / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"HippoRAG source directory does not exist: {source_dir}")
    sys.path.insert(0, str(source_dir))

    from hipporag import Chunk, HippoRAG
    from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
    from hipporag.llm.openai_gpt import CacheOpenAI
    from hipporag.utils.config_utils import BaseConfig

    return {
        "BaseConfig": BaseConfig,
        "CacheOpenAI": CacheOpenAI,
        "Chunk": Chunk,
        "HippoRAG": HippoRAG,
        "OpenAIEmbeddingModel": OpenAIEmbeddingModel,
    }


def build_zhipu_embedding_class(base_class: type, *, dimensions: int) -> type:
    class ZhipuEmbeddingModel(base_class):
        def encode(self, texts: list[str]) -> np.ndarray:
            prepared = [text.replace("\n", " ") or " " for text in texts]
            response = self.client.embeddings.create(
                input=prepared,
                model=self.embedding_model_name,
                dimensions=dimensions,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            result = np.asarray([item.embedding for item in ordered], dtype=np.float32)
            if result.shape != (len(prepared), dimensions):
                raise ValueError(
                    "Zhipu embedding shape mismatch: "
                    f"expected {(len(prepared), dimensions)}, got {result.shape}"
                )
            return result

    ZhipuEmbeddingModel.__name__ = "ZhipuEmbeddingModel"
    return ZhipuEmbeddingModel


def load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_revision(hipporag_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=hipporag_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_or_validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"Runtime manifest differs from this run: {path}. "
                "Use a new --runtime-dir for a different corpus or model configuration."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["query_id"])] = row
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=json_default) + "\n")
        handle.flush()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set | tuple):
        return list(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hipporag-root",
        type=Path,
        default=Path("data/benchmarks/_downloads/baselines/hipporag"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--stage", choices=("doctor", "index", "retrieve", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file.resolve())

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    llm_model = args.llm_model or os.getenv("DEEPSEEK_EXTRACTION_MODEL", "deepseek-v4-flash")
    llm_base_url = args.llm_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    embedding_model = args.embedding_model or os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
    embedding_base_url = args.embedding_base_url or os.getenv(
        "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )
    embedding_dimensions = args.embedding_dimensions or int(
        os.getenv("EMBEDDING_DIMENSION", "1024")
    )

    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    if not zhipu_key:
        raise RuntimeError("ZHIPU_API_KEY is not configured")
    if args.llm_concurrency <= 0:
        raise ValueError("--llm-concurrency must be positive")

    corpus_path = args.corpus.resolve()
    query_path = args.queries.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_path = args.output.resolve()
    hipporag_root = args.hipporag_root.resolve()
    corpus = load_json_array(corpus_path)
    queries = load_json_array(query_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    if not corpus or not queries:
        raise ValueError("Corpus and query files must both be non-empty")

    api = import_hipporag(hipporag_root)
    manifest = {
        "method": "hipporag2",
        "hipporag_revision": upstream_revision(hipporag_root),
        "corpus_sha256": file_digest(corpus_path),
        "query_sha256": file_digest(query_path),
        "corpus_chunks": len(corpus),
        "queries": len(queries),
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_dimensions": embedding_dimensions,
        "deepseek_thinking": "disabled",
        "top_k": args.top_k,
    }
    write_or_validate_manifest(runtime_dir / "adapter_manifest.json", manifest)

    if args.stage == "doctor":
        print(json.dumps({**manifest, "deepseek_key": "set", "zhipu_key": "set"}, indent=2))
        return 0

    config = api["BaseConfig"](
        save_dir=str(runtime_dir / "hipporag"),
        llm_name=llm_model,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        retrieval_top_k=args.top_k,
        openie_mode="online",
        force_openie_from_scratch=False,
        force_index_from_scratch=False,
    )
    with temporary_openai_key(deepseek_key):
        native_llm = api["CacheOpenAI"].from_experiment_config(config)
    llm = LimitedLLM(native_llm, args.llm_concurrency)
    embedding_class = build_zhipu_embedding_class(
        api["OpenAIEmbeddingModel"], dimensions=embedding_dimensions
    )
    with temporary_openai_key(zhipu_key):
        embedding = embedding_class(global_config=config, embedding_model_name=embedding_model)

    rag = api["HippoRAG"](
        global_config=config,
        extraction_llm=llm,
        qa_llm=llm,
        embedding_model=embedding,
    )
    chunks = [
        api["Chunk"](
            content=str(item["text"]),
            source_id=str(item.get("document_id") or item["idx"]),
            metadata={
                "chunk_id": str(item.get("chunk_id") or item["idx"]),
                "title": str(item.get("title") or ""),
                "order": int(item.get("order") or 0),
            },
        )
        for item in corpus
    ]

    index_marker = runtime_dir / "index_complete.json"
    if args.stage in {"index", "all"}:
        if index_marker.is_file():
            print(f"HippoRAG index already complete: {index_marker}")
        else:
            started = perf_counter()
            rag.index(docs=chunks)
            index_marker.write_text(
                json.dumps(
                    {"elapsed_seconds": perf_counter() - started, **manifest},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"HippoRAG index completed: {index_marker}")

    if args.stage == "index":
        return 0
    if not index_marker.is_file():
        raise RuntimeError(
            "HippoRAG index is incomplete; run with --stage index or --stage all first"
        )

    completed = load_completed(output_path)
    for query in queries:
        query_id = str(query.get("id") or query.get("query_id"))
        if query_id in completed:
            print(f"Skipping completed query: {query_id}")
            continue
        question = str(query.get("question") or query.get("query") or "")
        started = perf_counter()
        solution = rag.retrieve([question], num_to_retrieve=args.top_k)[0]
        elapsed_ms = (perf_counter() - started) * 1000
        retrieved = []
        metadata_rows = solution.doc_metadata or [{} for _ in solution.docs]
        scores = solution.doc_scores.tolist() if solution.doc_scores is not None else []
        for rank, document_text in enumerate(solution.docs):
            metadata = dict(metadata_rows[rank]) if rank < len(metadata_rows) else {}
            retrieved.append(
                {
                    "chunk_id": metadata.get("chunk_id") or f"hipporag-{rank + 1}",
                    "document_id": metadata.get("source_id"),
                    "text": document_text,
                    "score": scores[rank] if rank < len(scores) else None,
                    "metadata": {
                        **metadata,
                        "rank": rank + 1,
                        "graph_seeds": solution.graph_seeds or [],
                    },
                }
            )
        append_jsonl(
            output_path,
            {
                "query_id": query_id,
                "question": question,
                "elapsed_ms": elapsed_ms,
                "retrieved": retrieved,
            },
        )
        print(f"Retrieved {query_id}: {len(retrieved)} chunks in {elapsed_ms:.1f} ms")

    print(f"Raw HippoRAG results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
