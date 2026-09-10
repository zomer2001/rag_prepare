#!/usr/bin/env python3
"""Run the official RAPTOR tree with provenance-aware document retrieval."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_or_validate(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Runtime manifest differs: {path}. Use a new --runtime-dir.")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cosine_distance(a: list[float], b: list[float]) -> float:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 1.0
    return 1.0 - float(np.dot(left, right) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--raptor-root",
        type=Path,
        default=Path("data/benchmarks/_downloads/baselines/raptor"),
    )
    parser.add_argument("--stage", choices=("doctor", "index", "retrieve", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-nodes", type=int, default=20)
    parser.add_argument("--max-candidate-tokens", type=int, default=12000)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--summarization-length", type=int, default=100)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--embedding-model", default="embedding-3")
    parser.add_argument("--embedding-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.raptor_root.resolve()
    sys.path.insert(0, str(root))
    # The official module imports SentenceTransformer eagerly even when a custom
    # remote embedding model is supplied. Avoid importing PyTorch in this adapter.
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers
    try:
        from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
        from raptor.EmbeddingModels import BaseEmbeddingModel
        from raptor.SummarizationModels import BaseSummarizationModel
        from raptor.tree_structures import Tree
    except ImportError as exc:
        raise RuntimeError(
            "RAPTOR dependencies are missing; run scripts/setup_missing_graph_baselines.sh"
        ) from exc

    class CompatibleEmbeddingModel(BaseEmbeddingModel):
        def __init__(self) -> None:
            self.client = OpenAI(
                api_key=os.environ.get("ZHIPU_API_KEY", "missing"),
                base_url=args.embedding_base_url,
            )

        def create_embedding(self, text: str) -> list[float]:
            response = self.client.embeddings.create(
                input=[text.replace("\n", " ")],
                model=args.embedding_model,
                dimensions=args.embedding_dimensions,
            )
            return list(response.data[0].embedding)

    class CompatibleSummarizationModel(BaseSummarizationModel):
        def __init__(self) -> None:
            self.client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "missing"),
                base_url=args.llm_base_url,
            )

        def summarize(
            self,
            context: str,
            max_tokens: int = 150,
            stop_sequence: str | None = None,
        ) -> str:
            response = self.client.chat.completions.create(
                model=args.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize the following evidence, retaining names, dates, "
                            f"relations, and changes.\n\n{context}"
                        ),
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
                stop=stop_sequence,
            )
            return response.choices[0].message.content or ""

    corpus_path = args.corpus.resolve()
    queries_path = args.queries.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_path = args.output.resolve()
    corpus = _load_json(corpus_path)
    queries = _load_json(queries_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    positive = (
        args.top_k,
        args.candidate_nodes,
        args.max_candidate_tokens,
        args.num_layers,
        args.summarization_length,
        args.embedding_dimensions,
        args.workers,
    )
    if min(positive) <= 0:
        raise ValueError("RAPTOR numeric options must be positive")
    revision = _git_revision(root)
    print(json.dumps({"raptor_revision": revision, "chunks": len(corpus), "queries": len(queries)}))
    if args.stage == "doctor":
        return 0

    _load_env(args.env_file.resolve())
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("ZHIPU_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY and ZHIPU_API_KEY are required")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "adapter": "raptor_provenance_leaf_rerank_v1",
        "raptor_revision": revision,
        "corpus_sha256": _sha256(corpus_path),
        "queries_sha256": _sha256(queries_path),
        "chunks": len(corpus),
        "queries": len(queries),
        "num_layers": args.num_layers,
        "summarization_length": args.summarization_length,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dimensions,
    }
    _write_or_validate(runtime_dir / "adapter_manifest.json", manifest)
    embedding = CompatibleEmbeddingModel()

    if args.stage in {"index", "all"}:
        summarizer = CompatibleSummarizationModel()
        config = ClusterTreeConfig(
            num_layers=args.num_layers,
            summarization_length=args.summarization_length,
            summarization_model=summarizer,
            embedding_models={"ELC": embedding},
            cluster_embedding_model="ELC",
        )
        builder = ClusterTreeBuilder(config)
        texts = [str(item["text"]) for item in corpus]
        leaf_nodes: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(builder.create_node, index, text): index
                for index, text in enumerate(texts)
            }
            for future in as_completed(futures):
                index, node = future.result()
                leaf_nodes[index] = node
        layer_to_nodes = {0: list(leaf_nodes.values())}
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = builder.construct_tree(all_nodes, all_nodes, layer_to_nodes)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, builder.num_layers, layer_to_nodes)
        with (runtime_dir / "tree.pkl").open("wb") as handle:
            pickle.dump(tree, handle)
        provenance = {
            str(index): {
                "chunk_id": str(item["chunk_id"]),
                "document_id": str(item["document_id"]),
                "title": str(item.get("title") or ""),
                "text": str(item["text"]),
            }
            for index, item in enumerate(corpus)
        }
        (runtime_dir / "leaf_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (runtime_dir / "index_complete.json").write_text(
            json.dumps({"complete": True, "nodes": len(tree.all_nodes)}) + "\n",
            encoding="utf-8",
        )
        if args.stage == "index":
            return 0

    if args.stage in {"retrieve", "all"}:
        tree_path = runtime_dir / "tree.pkl"
        provenance_path = runtime_dir / "leaf_provenance.json"
        if not tree_path.is_file() or not provenance_path.is_file():
            raise RuntimeError("RAPTOR index is incomplete; run --stage index first")
        with tree_path.open("rb") as handle:
            tree = pickle.load(handle)  # noqa: S301 - reads this adapter's own index
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

        def descendant_leaves(node_index: int) -> set[int]:
            node = tree.all_nodes[node_index]
            if not node.children:
                return {node_index} if str(node_index) in provenance else set()
            result: set[int] = set()
            for child in node.children:
                result.update(descendant_leaves(child))
            return result

        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed: set[str] = set()
        if output_path.is_file():
            with output_path.open(encoding="utf-8") as handle:
                completed = {str(json.loads(line)["query_id"]) for line in handle if line.strip()}
        node_list = [tree.all_nodes[index] for index in sorted(tree.all_nodes)]
        with output_path.open("a", encoding="utf-8") as handle:
            for query in queries:
                query_id = str(query["id"])
                if query_id in completed:
                    continue
                started = time.perf_counter()
                query_embedding = embedding.create_embedding(str(query["question"]))
                node_distances = [
                    _cosine_distance(query_embedding, node.embeddings["ELC"]) for node in node_list
                ]
                candidate_leaves: set[int] = set()
                token_budget = 0
                for position in np.argsort(node_distances):
                    node = node_list[int(position)]
                    token_budget += len(node.text.split())
                    if token_budget > args.max_candidate_tokens:
                        break
                    candidate_leaves.update(descendant_leaves(node.index))
                    if len(candidate_leaves) >= args.candidate_nodes:
                        break
                ranked_leaves = sorted(
                    candidate_leaves,
                    key=lambda index: _cosine_distance(
                        query_embedding, tree.all_nodes[index].embeddings["ELC"]
                    ),
                )
                retrieved = []
                seen_documents: set[str] = set()
                for leaf_index in ranked_leaves:
                    mapped = provenance[str(leaf_index)]
                    document_id = str(mapped["document_id"])
                    if document_id in seen_documents:
                        continue
                    seen_documents.add(document_id)
                    distance = _cosine_distance(
                        query_embedding, tree.all_nodes[leaf_index].embeddings["ELC"]
                    )
                    retrieved.append(
                        {
                            "chunk_id": str(mapped["chunk_id"]),
                            "document_id": document_id,
                            "text": str(mapped["text"]),
                            "score": 1.0 - distance,
                            "metadata": {
                                "title": mapped.get("title"),
                                "leaf_index": leaf_index,
                                "source": "raptor_collapsed_candidate_leaf_rerank",
                            },
                        }
                    )
                    if len(retrieved) >= args.top_k:
                        break
                handle.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "question": query["question"],
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "retrieved": retrieved,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
        print(f"Raw RAPTOR results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
