#!/usr/bin/env python3
"""Run EventRAG hybrid retrieval with DeepSeek, Zhipu, Qdrant, and isolated Neo4j."""

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
import sys
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import numpy as np
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
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
        try:
            parsed = shlex.split(value.strip(), comments=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


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
            if not (path.parent / "index_complete.json").is_file():
                path.write_text(
                    json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return
            raise RuntimeError(
                f"Runtime manifest differs from this run: {path}. "
                "Use a new --runtime-dir for different data or settings."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(row["query_id"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def import_eventrag(eventrag_root: Path) -> dict[str, Any]:
    """Import only the EventRAG hybrid path, avoiding unused agent-only dependencies."""
    if not (eventrag_root / "eventrag" / "eventrag.py").is_file():
        raise FileNotFoundError(f"EventRAG source does not exist: {eventrag_root}")
    sys.path.insert(0, str(eventrag_root))

    async def unavailable_default(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("The adapter must provide its own model functions")

    llm_stub = types.ModuleType("eventrag.llm")
    llm_stub.gpt_4o_mini_complete = unavailable_default
    llm_stub.openai_embedding = unavailable_default
    agent_stub = types.ModuleType("eventrag.graph_agent")
    agent_stub.agent_query = unavailable_default
    sys.modules.setdefault("eventrag.llm", llm_stub)
    sys.modules.setdefault("eventrag.graph_agent", agent_stub)

    from eventrag import EventRAG, QueryParam
    from eventrag.base import BaseVectorStorage
    from eventrag.prompt import PROMPTS
    from eventrag.utils import EmbeddingFunc

    return {
        "BaseVectorStorage": BaseVectorStorage,
        "EmbeddingFunc": EmbeddingFunc,
        "EventRAG": EventRAG,
        "PROMPTS": PROMPTS,
        "QueryParam": QueryParam,
    }


def configure_budgeted_extraction_prompt(
    prompts: dict[str, Any],
    *,
    max_entities: int,
    max_relationships: int,
) -> None:
    marker = "-Budgeted Baseline Rules-"
    template = prompts["entity_extraction"]
    if marker in template:
        return
    budget = f"""
{marker}
1. Read the text in its original order.
2. Return at most {max_entities} entity records and at most
   {max_relationships} relationship records.
3. If more records are available, keep the first ones encountered in the text.
4. Do not add background knowledge or infer concepts not directly stated in the text.

"""
    prompts["entity_extraction"] = template.replace(
        "#############################\n-Real Data-",
        f"{budget}#############################\n-Real Data-",
    )


def truncate_extraction_response(
    response: str,
    *,
    max_entities: int,
    max_relationships: int,
) -> str:
    completion = "<|COMPLETE|>"
    records = response.replace(completion, "").split("##")
    kept: list[str] = []
    entity_count = 0
    relationship_count = 0
    for record in records:
        value = record.strip()
        lowered = value.casefold()
        if lowered.startswith('(\"entity\"<|>'):
            if entity_count >= max_entities:
                continue
            entity_count += 1
        elif lowered.startswith('(\"relationship\"<|>'):
            if relationship_count >= max_relationships:
                continue
            relationship_count += 1
        kept.append(value)
    return "##".join(value for value in kept if value) + completion


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
        extraction_max_tokens: int,
        max_entities_per_chunk: int,
        max_relationships_per_chunk: int,
    ) -> None:
        self.deepseek_model = deepseek_model
        self.zhipu_model = zhipu_model
        self.dimensions = dimensions
        self.extraction_max_tokens = extraction_max_tokens
        self.max_entities_per_chunk = max_entities_per_chunk
        self.max_relationships_per_chunk = max_relationships_per_chunk
        self._deepseek = AsyncOpenAI(
            api_key=deepseek_key,
            base_url=deepseek_base_url,
            timeout=120,
            max_retries=3,
        )
        self._zhipu = AsyncOpenAI(
            api_key=zhipu_key,
            base_url=zhipu_base_url,
            timeout=60,
            max_retries=3,
        )
        self._llm_limit = asyncio.Semaphore(llm_concurrency)
        self._embedding_limit = asyncio.Semaphore(embedding_concurrency)

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
        if "expand the knowledge graph by identifying additional related concepts" in prompt:
            return "<|COMPLETE|>"
        is_extraction = "-Budgeted Baseline Rules-" in prompt
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
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        if is_extraction:
            request["max_tokens"] = min(
                int(request.get("max_tokens", self.extraction_max_tokens)),
                self.extraction_max_tokens,
            )
        async with self._llm_limit:
            response = await self._deepseek.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        if is_extraction:
            return truncate_extraction_response(
                content,
                max_entities=self.max_entities_per_chunk,
                max_relationships=self.max_relationships_per_chunk,
            )
        return content

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        async with self._embedding_limit:
            response = await self._zhipu.embeddings.create(
                model=self.zhipu_model,
                input=[text.replace("\n", " ") or " " for text in texts],
                dimensions=self.dimensions,
            )
        ordered = sorted(response.data, key=lambda item: item.index)
        result = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if result.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"Zhipu embedding shape mismatch: expected {(len(texts), self.dimensions)}, "
                f"got {result.shape}"
            )
        return result

    async def close(self) -> None:
        await self._deepseek.close()
        await self._zhipu.close()


def build_qdrant_storage_class(
    base_class: type,
    *,
    qdrant_url: str,
    qdrant_api_key: str | None,
    workspace: str,
    dimensions: int,
) -> type:
    bypass_proxy_for_local_services(qdrant_url)

    @dataclass
    class QdrantVectorDBStorage(base_class):
        def __post_init__(self) -> None:
            suffix = hashlib.sha256(workspace.encode()).hexdigest()[:12]
            namespace = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.namespace)
            self.collection_name = f"eventrag_{suffix}_{namespace}"
            self._client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            self._ready = False
            self._ready_lock = asyncio.Lock()
            self._max_batch_size = max(1, int(self.global_config["embedding_batch_num"]))

        async def _ensure_collection(self) -> None:
            if self._ready:
                return
            async with self._ready_lock:
                if not await self._client.collection_exists(self.collection_name):
                    await self._client.create_collection(
                        self.collection_name,
                        vectors_config=models.VectorParams(
                            size=dimensions,
                            distance=models.Distance.COSINE,
                        ),
                    )
                self._ready = True

        async def upsert(self, data: dict[str, dict[str, Any]]) -> list[str]:
            if not data:
                return []
            await self._ensure_collection()
            rows = [(key, value) for key, value in data.items() if value.get("content")]
            point_ids = {
                key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{key}"))
                for key, _ in rows
            }
            existing_ids: set[str] = set()
            all_ids = list(point_ids.values())
            for start in range(0, len(all_ids), 256):
                existing = await self._client.retrieve(
                    self.collection_name,
                    ids=all_ids[start : start + 256],
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(str(point.id) for point in existing)
            rows = [(key, value) for key, value in rows if point_ids[key] not in existing_ids]
            for start in range(0, len(rows), self._max_batch_size):
                batch = rows[start : start + self._max_batch_size]
                vectors = np.asarray(
                    await self.embedding_func([value["content"] for _, value in batch]),
                    dtype=np.float32,
                )
                await self._client.upsert(
                    self.collection_name,
                    points=[
                        models.PointStruct(
                            id=point_ids[key],
                            vector=vector.tolist(),
                            payload={
                                "eventrag_id": key,
                                **{
                                    field: value[field]
                                    for field in self.meta_fields
                                    if field in value
                                },
                            },
                        )
                        for (key, value), vector in zip(batch, vectors, strict=True)
                    ],
                    wait=True,
                )
            return [key for key, _ in rows]

        async def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            await self._ensure_collection()
            vector = np.asarray(await self.embedding_func([query]), dtype=np.float32)[0]
            response = await self._client.query_points(
                self.collection_name,
                query=vector.tolist(),
                limit=top_k,
                score_threshold=0.2,
                with_payload=True,
            )
            return [
                {
                    **dict(point.payload or {}),
                    "id": (point.payload or {}).get("eventrag_id", str(point.id)),
                    "distance": float(point.score),
                }
                for point in response.points
            ]

        async def find_similar_entity(
            self,
            _entity_name: str,
            entity_description: str,
            similarity_threshold: float = 0.9,
        ) -> str | None:
            matches = await self.query(entity_description, top_k=1)
            if matches and matches[0]["distance"] > similarity_threshold:
                return matches[0].get("entity_name")
            return None

        async def index_done_callback(self) -> None:
            return None

        async def close(self) -> None:
            await self._client.close()

    QdrantVectorDBStorage.__name__ = "QdrantVectorDBStorage"
    return QdrantVectorDBStorage


def build_eventrag_class(eventrag_class: type, qdrant_class: type) -> type:
    class QdrantEventRAG(eventrag_class):
        def _get_storage_class(self) -> dict[str, type]:
            classes = super()._get_storage_class()
            classes["QdrantVectorDBStorage"] = qdrant_class
            return classes

    return QdrantEventRAG


def extract_source_texts(context: str) -> list[str]:
    match = re.search(r"-----Sources-----\s*```csv\s*(.*?)```", context, re.DOTALL)
    if not match:
        return []
    body = match.group(1).strip()
    combined_rows = list(re.finditer(r"(?m)^\d+,\t", body))
    if combined_rows:
        values = []
        for index, current in enumerate(combined_rows):
            end = combined_rows[index + 1].start() if index + 1 < len(combined_rows) else len(body)
            values.append(body[current.end() : end].strip())
        return values
    rows = list(csv.reader(io.StringIO(body)))
    if not rows:
        return []
    header = [value.strip().casefold() for value in rows[0]]
    try:
        content_index = header.index("content")
    except ValueError:
        return []
    return [row[content_index] for row in rows[1:] if len(row) > content_index]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def bypass_proxy_for_local_services(*urls: str) -> None:
    hosts = {"localhost", "127.0.0.1", "::1"}
    hosts.update(host for value in urls if (host := urlparse(value).hostname))
    for variable in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(variable, "")
        os.environ[variable] = ",".join(
            dict.fromkeys(item for item in [*existing.split(","), *sorted(hosts)] if item)
        )


class CorpusLookup:
    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self._exact: dict[str, list[dict[str, Any]]] = {}
        self._normalized: dict[str, list[dict[str, Any]]] = {}
        for item in corpus:
            text = str(item["text"]).strip()
            self._exact.setdefault(text, []).append(item)
            self._normalized.setdefault(normalize_text(text), []).append(item)

    def resolve(self, text: str) -> tuple[dict[str, Any] | None, str]:
        candidates = self._exact.get(text.strip())
        match_type = "exact"
        if not candidates:
            candidates = self._normalized.get(normalize_text(text))
            match_type = "normalized"
        return (candidates[0], match_type) if candidates else (None, "unmatched")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


async def check_services(args: argparse.Namespace) -> None:
    qdrant = AsyncQdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    try:
        await qdrant.get_collections()
    finally:
        await qdrant.close()
    neo4j = AsyncGraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    try:
        await neo4j.verify_connectivity()
    finally:
        await neo4j.close()


async def close_rag(rag: Any) -> None:
    for storage in (
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.chunk_entity_relation_graph,
    ):
        close = getattr(storage, "close", None)
        if close is not None:
            await close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eventrag-root",
        type=Path,
        default=Path("data/benchmarks/_downloads/baselines/eventrag"),
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
    parser.add_argument("--embedding-concurrency", type=int, default=2)
    parser.add_argument("--extraction-max-tokens", type=int, default=1200)
    parser.add_argument("--max-entities-per-chunk", type=int, default=6)
    parser.add_argument("--max-relationships-per-chunk", type=int, default=8)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-user")
    parser.add_argument("--neo4j-password")
    parser.add_argument("--entity-similarity-threshold", type=float, default=0.8)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file.resolve())
    args.llm_model = args.llm_model or os.getenv("DEEPSEEK_EXTRACTION_MODEL", "deepseek-v4-flash")
    args.llm_base_url = args.llm_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    args.embedding_model = args.embedding_model or os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
    args.embedding_base_url = args.embedding_base_url or os.getenv(
        "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )
    args.embedding_dimensions = args.embedding_dimensions or int(
        os.getenv("EMBEDDING_DIMENSION", "1024")
    )
    args.qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    args.qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")
    args.neo4j_uri = args.neo4j_uri or os.getenv("EVENTRAG_NEO4J_URI", "bolt://127.0.0.1:7688")
    args.neo4j_user = args.neo4j_user or os.getenv("EVENTRAG_NEO4J_USER", "neo4j")
    args.neo4j_password = args.neo4j_password or os.getenv(
        "EVENTRAG_NEO4J_PASSWORD", "eventrag-change-me"
    )
    bypass_proxy_for_local_services(args.qdrant_url, args.neo4j_uri)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if not deepseek_key or not zhipu_key:
        raise RuntimeError("DEEPSEEK_API_KEY and ZHIPU_API_KEY are required")
    positive_values = (
        args.top_k,
        args.embedding_batch_size,
        args.llm_concurrency,
        args.embedding_concurrency,
        args.extraction_max_tokens,
        args.max_entities_per_chunk,
        args.max_relationships_per_chunk,
    )
    if min(positive_values) <= 0:
        raise ValueError("top-k, batch size, and concurrency values must be positive")

    corpus_path = args.corpus.resolve()
    queries_path = args.queries.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_path = args.output.resolve()
    eventrag_root = args.eventrag_root.resolve()
    corpus = load_json_array(corpus_path)
    queries = load_json_array(queries_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    runtime_dir.mkdir(parents=True, exist_ok=True)

    workspace = str(runtime_dir)
    manifest = {
        "method": "eventrag_hybrid_budgeted",
        "eventrag_revision": upstream_revision(eventrag_root),
        "corpus_sha256": file_digest(corpus_path),
        "query_sha256": file_digest(queries_path),
        "corpus_chunks": len(corpus),
        "queries": len(queries),
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_dimensions": args.embedding_dimensions,
        "qdrant_url": args.qdrant_url,
        "neo4j_uri": args.neo4j_uri,
        "entity_similarity_threshold": args.entity_similarity_threshold,
        "extraction_profile": "budgeted_first_records_v1",
        "extraction_max_tokens": args.extraction_max_tokens,
        "max_entities_per_chunk": args.max_entities_per_chunk,
        "max_relationships_per_chunk": args.max_relationships_per_chunk,
        "top_k": args.top_k,
    }
    write_or_validate_manifest(runtime_dir / "adapter_manifest.json", manifest)
    await check_services(args)

    api = import_eventrag(eventrag_root)
    configure_budgeted_extraction_prompt(
        api["PROMPTS"],
        max_entities=args.max_entities_per_chunk,
        max_relationships=args.max_relationships_per_chunk,
    )
    if args.stage == "doctor":
        print(json.dumps({**manifest, "deepseek_key": "set", "zhipu_key": "set"}, indent=2))
        return 0

    os.environ["NEO4J_URI"] = args.neo4j_uri
    os.environ["NEO4J_USERNAME"] = args.neo4j_user
    os.environ["NEO4J_PASSWORD"] = args.neo4j_password
    clients = ModelClients(
        deepseek_key=deepseek_key,
        deepseek_base_url=args.llm_base_url,
        deepseek_model=args.llm_model,
        zhipu_key=zhipu_key,
        zhipu_base_url=args.embedding_base_url,
        zhipu_model=args.embedding_model,
        dimensions=args.embedding_dimensions,
        llm_concurrency=args.llm_concurrency,
        embedding_concurrency=args.embedding_concurrency,
        extraction_max_tokens=args.extraction_max_tokens,
        max_entities_per_chunk=args.max_entities_per_chunk,
        max_relationships_per_chunk=args.max_relationships_per_chunk,
    )

    async def eventrag_complete(*model_args: Any, **model_kwargs: Any) -> str:
        return await clients.complete(*model_args, **model_kwargs)

    async def eventrag_embed(texts: list[str]) -> np.ndarray:
        return await clients.embed(texts)

    embedding_func = api["EmbeddingFunc"](
        embedding_dim=args.embedding_dimensions,
        max_token_size=8192,
        func=eventrag_embed,
        concurrent_limit=args.embedding_concurrency,
    )
    qdrant_class = build_qdrant_storage_class(
        api["BaseVectorStorage"],
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        workspace=workspace,
        dimensions=args.embedding_dimensions,
    )
    eventrag_class = build_eventrag_class(api["EventRAG"], qdrant_class)

    rag = None
    try:
        with working_directory(runtime_dir):
            rag = eventrag_class(
                working_dir=str(runtime_dir / "state"),
                llm_model_func=eventrag_complete,
                embedding_func=embedding_func,
                graph_storage="Neo4JStorage",
                vector_storage="QdrantVectorDBStorage",
                chunk_token_size=8192,
                chunk_overlap_token_size=0,
                embedding_batch_num=args.embedding_batch_size,
                llm_model_max_async=args.llm_concurrency,
                embedding_func_max_async=args.embedding_concurrency,
                entity_extract_max_gleaning=0,
                entity_similarity_threshold=args.entity_similarity_threshold,
                addon_params={"example_number": 1},
                log_level="INFO",
            )
            index_marker = runtime_dir / "index_complete.json"
            if args.stage in {"index", "all"}:
                if index_marker.is_file():
                    print(f"EventRAG index already complete: {index_marker}")
                else:
                    started = perf_counter()
                    await rag.ainsert([str(item["text"]) for item in corpus])
                    index_marker.write_text(
                        json.dumps(
                            {"elapsed_seconds": perf_counter() - started, **manifest},
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"EventRAG index completed: {index_marker}")
            if args.stage == "index":
                return 0
            if not index_marker.is_file():
                raise RuntimeError("EventRAG index is incomplete; run --stage index first")

            lookup = CorpusLookup(corpus)
            completed = load_completed(output_path)
            for query in queries:
                query_id = str(query.get("id") or query.get("query_id"))
                if query_id in completed:
                    print(f"Skipping completed query: {query_id}")
                    continue
                question = str(query.get("question") or query.get("query") or "")
                started = perf_counter()
                context = await rag.aquery(
                    question,
                    api["QueryParam"](
                        mode="hybrid",
                        only_need_context=True,
                        top_k=args.top_k,
                    ),
                )
                elapsed_ms = (perf_counter() - started) * 1000
                retrieved = []
                seen: set[str] = set()
                for source_text in extract_source_texts(str(context)):
                    item, match_type = lookup.resolve(source_text)
                    chunk_id = (
                        str(item.get("chunk_id") or item.get("idx"))
                        if item
                        else f"eventrag-{hashlib.sha256(source_text.encode()).hexdigest()[:16]}"
                    )
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    retrieved.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": str(item.get("document_id")) if item else None,
                            "text": source_text,
                            "score": None,
                            "metadata": {
                                "rank": len(retrieved) + 1,
                                "mode": "hybrid",
                                "content_match": match_type,
                            },
                        }
                    )
                    if len(retrieved) >= args.top_k:
                        break
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
            print(f"Raw EventRAG results written to: {output_path}")
            return 0
    finally:
        if rag is not None:
            await close_rag(rag)
        await clients.close()


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
