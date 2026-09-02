#!/usr/bin/env python3
"""Run the unmodified E2RAG implementation with DeepSeek and Zhipu APIs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from openai import AsyncOpenAI


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        try:
            parsed = shlex.split(value.strip(), comments=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


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


def upstream_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
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
            raise RuntimeError(f"Runtime manifest differs: {path}. Use a new --runtime-dir.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def completed_query_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(row["query_id"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_source_texts(context: str) -> list[str]:
    match = re.search(r"-----Sources-----\s*```csv\s*(.*?)```", context, re.DOTALL)
    if not match:
        return []
    rows = list(csv.reader(io.StringIO(match.group(1).strip())))
    if not rows:
        return []
    header = [value.strip().casefold() for value in rows[0]]
    try:
        content_index = header.index("content")
    except ValueError:
        return []
    return [row[content_index] for row in rows[1:] if len(row) > content_index]


class CorpusLookup:
    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self._exact: dict[str, dict[str, Any]] = {}
        self._normalized: dict[str, dict[str, Any]] = {}
        self._source_ids: dict[str, dict[str, Any]] = {}
        for item in corpus:
            text = str(item["text"]).strip()
            self._exact.setdefault(text, item)
            self._normalized.setdefault(normalize_text(text), item)
            source_id = "chunk-" + hashlib.md5(text.encode()).hexdigest()  # noqa: S324
            self._source_ids[source_id] = item

    def by_text(self, text: str) -> tuple[dict[str, Any] | None, str]:
        item = self._exact.get(text.strip())
        if item is not None:
            return item, "exact"
        item = self._normalized.get(normalize_text(text))
        return (item, "normalized") if item is not None else (None, "unmatched")

    def by_source_id(self, source_id: str) -> dict[str, Any] | None:
        return self._source_ids.get(source_id)


class ModelClients:
    def __init__(
        self,
        *,
        deepseek_key: str,
        deepseek_base_url: str,
        deepseek_model: str,
        zhipu_key: str,
        zhipu_base_url: str,
        zhipu_model: str,
        dimensions: int,
        llm_concurrency: int,
        embedding_concurrency: int,
    ) -> None:
        self.deepseek_model = deepseek_model
        self.zhipu_model = zhipu_model
        self.dimensions = dimensions
        self.deepseek = AsyncOpenAI(
            api_key=deepseek_key, base_url=deepseek_base_url, timeout=120, max_retries=3
        )
        self.zhipu = AsyncOpenAI(
            api_key=zhipu_key, base_url=zhipu_base_url, timeout=60, max_retries=3
        )
        self.llm_limit = asyncio.Semaphore(llm_concurrency)
        self.embedding_limit = asyncio.Semaphore(embedding_concurrency)

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("stream", None)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        request: dict[str, Any] = {
            "model": self.deepseek_model,
            "messages": messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        for key in ("max_tokens", "temperature", "top_p"):
            if kwargs.get(key) is not None:
                request[key] = kwargs[key]
        async with self.llm_limit:
            response = await self.deepseek.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        async with self.embedding_limit:
            response = await self.zhipu.embeddings.create(
                model=self.zhipu_model,
                input=[text.replace("\n", " ") or " " for text in texts],
                dimensions=self.dimensions,
            )
        ordered = sorted(response.data, key=lambda item: item.index)
        result = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if result.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"Embedding shape mismatch: expected {(len(texts), self.dimensions)}, "
                f"got {result.shape}"
            )
        return result

    async def close(self) -> None:
        await self.deepseek.close()
        await self.zhipu.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--stage", choices=("doctor", "index", "retrieve", "all"), default="all")
    parser.add_argument("--corpus-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query", help="Override the query file with one smoke-test query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--event-hop-depth", type=int, default=2)
    parser.add_argument(
        "--variant",
        choices=(
            "vanilla",
            "combined_extraction",
            "hypothetical_extraction",
            "combined_embedding",
            "hypothetical_embedding",
        ),
        default="vanilla",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--index-batch-size", type=int, default=8)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    parser.add_argument("--embedding-concurrency", type=int, default=2)
    parser.add_argument("--chunk-token-size", type=int, default=32768)
    parser.add_argument(
        "--e2rag-root",
        type=Path,
        default=Path("data/benchmarks/_downloads/baselines/entityeventrag"),
    )
    return parser.parse_args()


def query_param(api: dict[str, Any], args: argparse.Namespace) -> Any:
    values = {
        "mode": "entity_event",
        "only_need_context": True,
        "top_k": args.top_k,
        "event_hop_depth": args.event_hop_depth,
    }
    if args.variant != "vanilla":
        values["enable_hypothetical_document"] = True
    if args.variant == "hypothetical_extraction":
        values["extract_from_original_query"] = False
    elif args.variant == "combined_embedding":
        values["use_combined_embedding"] = True
    elif args.variant == "hypothetical_embedding":
        values["use_direct_embedding"] = True
    return api["QueryParam"](**values)


def retrieved_rows(
    response: Any,
    lookup: CorpusLookup,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    context = str(response[0]) if isinstance(response, tuple) else str(response)
    nodes = response[1] if isinstance(response, tuple) and len(response) > 1 else []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in extract_source_texts(context):
        item, match_type = lookup.by_text(text)
        chunk_id = (
            str(item.get("chunk_id") or item.get("idx"))
            if item is not None
            else "e2rag-" + hashlib.sha256(text.encode()).hexdigest()[:16]
        )
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(item.get("document_id")) if item else None,
                "text": text,
                "score": 1.0 / (len(rows) + 1),
                "metadata": {"match_type": match_type, "rank": len(rows) + 1},
            }
        )
        if len(rows) >= top_k:
            break
    if not rows:
        for node in nodes or []:
            for source_id in str(node.get("source_id") or "").split("<SEP>"):
                item = lookup.by_source_id(source_id)
                if item is None:
                    continue
                chunk_id = str(item.get("chunk_id") or item.get("idx"))
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": str(item.get("document_id")),
                        "text": str(item["text"]),
                        "score": float(node.get("score") or 0),
                        "metadata": {"match_type": "source_id", "rank": len(rows) + 1},
                    }
                )
                if len(rows) >= top_k:
                    break
            if len(rows) >= top_k:
                break
    stats = {
        "nodes": len(nodes or []),
        "event_nodes": sum(
            1
            for node in nodes or []
            if str(node.get("entity_type") or node.get("node_type") or "").strip('"').upper()
            == "EVENT"
        ),
    }
    return rows, stats


async def run(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file.resolve())
    if min(
        args.top_k,
        args.index_batch_size,
        args.embedding_batch_size,
        args.llm_concurrency,
        args.embedding_concurrency,
    ) <= 0:
        raise ValueError("Batch sizes, top-k, and concurrency values must be positive")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if not deepseek_key or not zhipu_key:
        raise RuntimeError("DEEPSEEK_API_KEY and ZHIPU_API_KEY are required")
    llm_model = args.llm_model or os.getenv("DEEPSEEK_EXTRACTION_MODEL", "deepseek-v4-flash")
    llm_base_url = args.llm_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    embedding_model = args.embedding_model or os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
    embedding_base_url = args.embedding_base_url or os.getenv(
        "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )
    dimensions = args.embedding_dimensions or int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    corpus_path = args.corpus.resolve()
    queries_path = args.queries.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_path = args.output.resolve()
    e2rag_root = args.e2rag_root.resolve()
    corpus = load_json_array(corpus_path)
    if args.corpus_limit is not None:
        corpus = corpus[: args.corpus_limit]
    queries = load_json_array(queries_path)
    if args.query:
        queries = [{"id": "smoke-1", "question": args.query}]
    elif args.limit is not None:
        queries = queries[: args.limit]
    if not corpus or not queries:
        raise ValueError("Corpus and queries must both be non-empty")

    from entity_event_rag import EntityEventRAG
    from entity_event_rag.base import QueryParam
    from entity_event_rag.utils import EmbeddingFunc

    api = {
        "EntityEventRAG": EntityEventRAG,
        "EmbeddingFunc": EmbeddingFunc,
        "QueryParam": QueryParam,
    }
    manifest = {
        "method": "e2rag_official_unmodified",
        "e2rag_revision": upstream_revision(e2rag_root),
        "corpus_sha256": file_digest(corpus_path),
        "queries_sha256": file_digest(queries_path),
        "query_override_sha256": (
            hashlib.sha256(args.query.encode()).hexdigest() if args.query else None
        ),
        "corpus_chunks": len(corpus),
        "queries": len(queries),
        "variant": args.variant,
        "top_k": args.top_k,
        "event_hop_depth": args.event_hop_depth,
        "chunk_token_size": args.chunk_token_size,
        "entity_extract_max_gleaning": 1,
        "index_batch_size": args.index_batch_size,
        "graph_storage": "NetworkXStorage",
        "vector_storage": "NanoVectorDBStorage",
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_dimensions": dimensions,
        "source_modified": False,
    }
    write_or_validate_manifest(runtime_dir / "adapter_manifest.json", manifest)
    if args.stage == "doctor":
        print(json.dumps({**manifest, "deepseek_key": "set", "zhipu_key": "set"}, indent=2))
        return 0

    clients = ModelClients(
        deepseek_key=deepseek_key,
        deepseek_base_url=llm_base_url,
        deepseek_model=llm_model,
        zhipu_key=zhipu_key,
        zhipu_base_url=embedding_base_url,
        zhipu_model=embedding_model,
        dimensions=dimensions,
        llm_concurrency=args.llm_concurrency,
        embedding_concurrency=args.embedding_concurrency,
    )

    async def complete(*model_args: Any, **model_kwargs: Any) -> str:
        return await clients.complete(*model_args, **model_kwargs)

    async def embed(texts: list[str]) -> np.ndarray:
        return await clients.embed(texts)

    embedding_func = api["EmbeddingFunc"](
        embedding_dim=dimensions,
        max_token_size=8192,
        func=embed,
    )
    common = {
        "llm_model_func": complete,
        "embedding_func": embedding_func,
        "graph_storage": "NetworkXStorage",
        "vector_storage": "NanoVectorDBStorage",
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": 0,
        "embedding_batch_num": args.embedding_batch_size,
        "llm_model_max_async": args.llm_concurrency,
        "embedding_func_max_async": args.embedding_concurrency,
        "entity_extract_max_gleaning": 1,
    }
    entity_dir = runtime_dir / "entity_kg"
    event_dir = runtime_dir / "event_kg"
    entity_rag = api["EntityEventRAG"](
        working_dir=str(entity_dir), extraction_type="entity", **common
    )
    event_rag = api["EntityEventRAG"](
        working_dir=str(event_dir), extraction_type="event", **common
    )
    try:
        texts = [str(item["text"]) for item in corpus]
        if args.stage in {"index", "all"}:
            for label, rag in (("entity", entity_rag), ("event", event_rag)):
                marker = runtime_dir / f"{label}_index_complete.json"
                if marker.is_file():
                    print(f"E2RAG {label} index already complete: {marker}")
                    continue
                progress_path = runtime_dir / f"{label}_index_progress.json"
                completed_count = 0
                elapsed_before = 0.0
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    completed_count = int(progress.get("completed_count") or 0)
                    elapsed_before = float(progress.get("elapsed_seconds") or 0)
                started = perf_counter()
                for start in range(completed_count, len(texts), args.index_batch_size):
                    stop = min(start + args.index_batch_size, len(texts))
                    await rag.ainsert(texts[start:stop])
                    progress_path.write_text(
                        json.dumps(
                            {
                                "completed_count": stop,
                                "total_count": len(texts),
                                "elapsed_seconds": elapsed_before + perf_counter() - started,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"E2RAG {label} index progress: {stop}/{len(texts)}")
                marker.write_text(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed_before + perf_counter() - started,
                            **manifest,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"E2RAG {label} index completed: {marker}")
            (runtime_dir / "index_complete.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if args.stage == "index":
            return 0
        if not (runtime_dir / "index_complete.json").is_file():
            raise RuntimeError("E2RAG index is incomplete; run --stage index first")

        combined_rag = api["EntityEventRAG"](
            working_dir=str(entity_dir),
            extraction_type="entity",
            entity_storage_dir=str(entity_dir),
            event_storage_dir=str(event_dir),
            **common,
        )
        lookup = CorpusLookup(corpus)
        completed = completed_query_ids(output_path)
        for query in queries:
            query_id = str(query.get("id") or query.get("query_id"))
            if query_id in completed:
                print(f"Skipping completed query: {query_id}")
                continue
            question = str(query.get("question") or query.get("query") or "")
            started = perf_counter()
            response = await combined_rag.aquery(question, query_param(api, args))
            elapsed_ms = (perf_counter() - started) * 1000
            retrieved, stats = retrieved_rows(response, lookup, top_k=args.top_k)
            append_jsonl(
                output_path,
                {
                    "query_id": query_id,
                    "question": question,
                    "elapsed_ms": elapsed_ms,
                    "retrieved": retrieved,
                    "official_response_stats": stats,
                },
            )
            print(
                f"Retrieved {query_id}: {len(retrieved)} chunks, "
                f"{stats['event_nodes']} event nodes in {elapsed_ms:.1f} ms"
            )
    finally:
        await clients.close()
    print(f"Raw E2RAG results written to: {output_path}")
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
