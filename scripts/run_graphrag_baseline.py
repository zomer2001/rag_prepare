#!/usr/bin/env python3
"""Prepare and run Microsoft GraphRAG Local Search on an exported benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

SETTINGS_TEMPLATE = """\
completion_models:
  default_completion_model:
    model_provider: openai
    model: {llm_model}
    api_base: {llm_base_url}
    auth_method: api_key
    api_key: ${{DEEPSEEK_API_KEY}}
    retry:
      type: exponential_backoff
embedding_models:
  default_embedding_model:
    model_provider: openai
    model: {embedding_model}
    api_base: {embedding_base_url}
    auth_method: api_key
    api_key: ${{ZHIPU_API_KEY}}
    call_args:
      dimensions: {embedding_dimensions}
    retry:
      type: exponential_backoff
concurrent_requests: {concurrency}
input:
  type: jsonl
  file_pattern: '.*\\.jsonl$$'
  id_column: id
  title_column: title
  text_column: text
chunking:
  type: tokens
  size: 4096
  overlap: 0
  encoding_model: o200k_base
input_storage:
  type: file
  base_dir: input
output_storage:
  type: file
  base_dir: output
reporting:
  type: file
  base_dir: logs
cache:
  type: json
  storage:
    type: file
    base_dir: cache
vector_store:
  type: lancedb
  db_uri: output/lancedb
embed_text:
  embedding_model_id: default_embedding_model
extract_graph:
  completion_model_id: default_completion_model
  prompt: prompts/extract_graph.txt
  entity_types: [organization, person, geo, event]
  max_gleanings: 1
summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: prompts/summarize_descriptions.txt
  max_length: 500
extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english
cluster_graph:
  max_cluster_size: 10
extract_claims:
  enabled: false
  completion_model_id: default_completion_model
  prompt: prompts/extract_claims.txt
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1
community_reports:
  completion_model_id: default_completion_model
  graph_prompt: prompts/community_report_graph.txt
  text_prompt: prompts/community_report_text.txt
  max_length: 2000
  max_input_length: 8000
snapshots:
  graphml: false
  embeddings: false
local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: prompts/local_search_system_prompt.txt
  text_unit_prop: 0.8
  community_prop: 0.1
  top_k_entities: 10
  top_k_relationships: 10
  max_context_tokens: 12000
global_search:
  completion_model_id: default_completion_model
  map_prompt: prompts/global_search_map_system_prompt.txt
  reduce_prompt: prompts/global_search_reduce_system_prompt.txt
  knowledge_prompt: prompts/global_search_knowledge_system_prompt.txt
drift_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: prompts/drift_search_system_prompt.txt
  reduce_prompt: prompts/drift_search_reduce_prompt.txt
basic_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: prompts/basic_search_system_prompt.txt
"""


def _load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def _load_env(path: Path) -> dict[str, str]:
    result = dict(os.environ)
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result.setdefault(key.strip(), value.strip().strip("\"'"))
    return result


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


def _graphrag_version(python: Path) -> str:
    command = [
        str(python),
        "-c",
        "from importlib.metadata import version; print(version('graphrag'))",
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--graphrag-venv",
        type=Path,
        default=Path("data/benchmarks/_venvs/graphrag"),
    )
    parser.add_argument(
        "--stage", choices=("doctor", "prepare", "index", "retrieve", "all"), default="all"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--community-level", type=int, default=2)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--embedding-model", default="embedding-3")
    parser.add_argument("--embedding-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    queries_path = args.queries.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_path = args.output.resolve()
    venv = args.graphrag_venv.resolve()
    python = venv / "bin" / "python"
    executable = venv / "bin" / "graphrag"
    if not python.is_file() or not executable.is_file():
        raise RuntimeError(f"GraphRAG environment is missing: {venv}")
    corpus = _load_json(corpus_path)
    queries = _load_json(queries_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    if args.top_k <= 0 or args.concurrency <= 0 or args.embedding_dimensions <= 0:
        raise ValueError("top-k, concurrency, and embedding dimensions must be positive")

    version = _graphrag_version(python)
    print(json.dumps({"graphrag_version": version, "chunks": len(corpus), "queries": len(queries)}))
    if args.stage == "doctor":
        return 0

    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "adapter": "graphrag_local_context_v1",
        "graphrag_version": version,
        "corpus_sha256": _sha256(corpus_path),
        "queries_sha256": _sha256(queries_path),
        "chunks": len(corpus),
        "queries": len(queries),
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dimensions,
    }
    _write_or_validate(runtime_dir / "adapter_manifest.json", manifest)

    if args.stage in {"prepare", "all"}:
        prompts_dir = runtime_dir / "prompts"
        if not prompts_dir.is_dir():
            subprocess.run(
                [
                    str(executable),
                    "init",
                    "--root",
                    str(runtime_dir),
                    "--model",
                    args.llm_model,
                    "--embedding",
                    args.embedding_model,
                ],
                check=True,
            )
        input_dir = runtime_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "corpus.jsonl"
        input_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": str(item["chunk_id"]),
                        "title": str(item.get("title") or item["chunk_id"]),
                        "text": str(item["text"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for item in corpus
            ),
            encoding="utf-8",
        )
        shutil.copyfile(queries_path, runtime_dir / "queries.json")
        source_map = {
            str(item["chunk_id"]): {
                "chunk_id": str(item["chunk_id"]),
                "document_id": str(item["document_id"]),
                "title": str(item.get("title") or ""),
                "text": str(item["text"]),
            }
            for item in corpus
        }
        (runtime_dir / "source_map.json").write_text(
            json.dumps(source_map, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (runtime_dir / "settings.yaml").write_text(
            SETTINGS_TEMPLATE.format(
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                embedding_model=args.embedding_model,
                embedding_base_url=args.embedding_base_url,
                embedding_dimensions=args.embedding_dimensions,
                concurrency=args.concurrency,
            ),
            encoding="utf-8",
        )
        print(f"Prepared GraphRAG runtime: {runtime_dir}")
        if args.stage == "prepare":
            return 0

    env = _load_env(args.env_file.resolve())
    if args.stage in {"index", "all"}:
        if not (runtime_dir / "settings.yaml").is_file():
            raise RuntimeError("Run --stage prepare before indexing")
        subprocess.run(
            [str(executable), "index", "--root", str(runtime_dir), "--method", "standard"],
            check=True,
            env=env,
        )
        (runtime_dir / "index_complete.json").write_text(
            json.dumps({"complete": True, "method": "standard"}) + "\n", encoding="utf-8"
        )
        if args.stage == "index":
            return 0

    if args.stage in {"retrieve", "all"}:
        if not (runtime_dir / "index_complete.json").is_file():
            raise RuntimeError("GraphRAG index is incomplete; run --stage index first")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        retrieve_script = Path(__file__).with_name("graphrag_retrieve.py").resolve()
        subprocess.run(
            [
                str(python),
                str(retrieve_script),
                "--runtime-dir",
                str(runtime_dir),
                "--output",
                str(output_path),
                "--top-k",
                str(args.top_k),
                "--community-level",
                str(args.community_level),
                "--limit",
                str(len(queries)),
            ],
            check=True,
            env=env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
