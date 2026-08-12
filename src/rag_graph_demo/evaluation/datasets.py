from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_graph_demo.evaluation.models import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
    GoldEvidence,
)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def _find_file(directory: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of {', '.join(names)} exists in benchmark directory {directory}")


def stable_document_id(value: dict[str, Any]) -> str:
    explicit = value.get("document_id") or value.get("doc_id") or value.get("id")
    if explicit not in (None, ""):
        return str(explicit)
    identity = "\x1f".join(
        str(value.get(key) or "").strip() for key in ("title", "source", "published_at")
    )
    if not identity.strip("\x1f"):
        identity = str(value.get("body") or value.get("text") or value.get("content") or "")
    return "doc-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def load_bundle(directory: Path, *, name: str | None = None) -> BenchmarkDataset:
    directory = directory.resolve()
    corpus_path = _find_file(directory, ("corpus.jsonl", "corpus.json"))
    query_path = _find_file(
        directory,
        ("queries.jsonl", "queries.json", "MultiHopRAG.json"),
    )
    documents = tuple(_document(value) for value in _records(corpus_path))
    queries = tuple(
        _query(value, index) for index, value in enumerate(_records(query_path), start=1)
    )
    if not documents:
        raise ValueError(f"Benchmark corpus is empty: {corpus_path}")
    if not queries:
        raise ValueError(f"Benchmark query set is empty: {query_path}")
    return BenchmarkDataset(
        name=name or directory.name,
        documents=documents,
        queries=queries,
    )


def _document(value: dict[str, Any]) -> BenchmarkDocument:
    text = value.get("text") or value.get("body") or value.get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each corpus record requires a non-empty text/body/content field")
    title = str(value.get("title") or stable_document_id(value))
    known = {"document_id", "doc_id", "id", "text", "body", "content", "title"}
    return BenchmarkDocument(
        document_id=stable_document_id(value),
        title=title,
        text=text,
        metadata={key: item for key, item in value.items() if key not in known},
    )


def _query(value: dict[str, Any], index: int) -> BenchmarkQuery:
    question = value.get("question") or value.get("query")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each query record requires a non-empty question/query field")
    evidence_values = value.get("gold_evidence")
    if evidence_values is None:
        evidence_values = value.get("evidence_list") or []
    gold: list[GoldEvidence] = []
    for evidence in evidence_values:
        if isinstance(evidence, str):
            gold.append(GoldEvidence(text=evidence))
            continue
        if not isinstance(evidence, dict):
            continue
        text = evidence.get("text") or evidence.get("fact") or evidence.get("content")
        document_id = evidence.get("document_id") or evidence.get("doc_id")
        if document_id is None and any(
            evidence.get(key) not in (None, "") for key in ("title", "source", "published_at")
        ):
            document_id = stable_document_id(evidence)
        gold.append(
            GoldEvidence(
                text=str(text) if text not in (None, "") else None,
                document_id=str(document_id) if document_id not in (None, "") else None,
            )
        )
    for document_id in value.get("gold_doc_ids") or []:
        gold.append(GoldEvidence(document_id=str(document_id)))
    return BenchmarkQuery(
        query_id=str(value.get("query_id") or value.get("id") or f"q-{index:05d}"),
        question=question,
        question_type=str(value.get("question_type") or value.get("type") or "unknown"),
        answer=str(value["answer"]) if value.get("answer") is not None else None,
        gold_evidence=tuple(_deduplicate_evidence(gold)),
    )


def _deduplicate_evidence(values: list[GoldEvidence]) -> list[GoldEvidence]:
    result: list[GoldEvidence] = []
    seen: set[tuple[str | None, str | None]] = set()
    for value in values:
        key = (value.document_id, value.text)
        if key not in seen and any(key):
            result.append(value)
            seen.add(key)
    return result


def prepare_multihop_rag(cache_dir: Path) -> BenchmarkDataset:
    """Download and canonicalize MultiHop-RAG through Hugging Face Datasets."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return load_bundle(cache_dir, name="multihop_rag")
    except FileNotFoundError:
        pass
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MultiHop-RAG download needs the evaluation extra: uv sync --extra evaluation"
        ) from exc

    repository = "yixuantt/MultiHopRAG"
    configs = get_dataset_config_names(repository)
    loaded: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        split = load_dataset(repository, config, split="train")
        loaded[config] = [dict(item) for item in split]
    corpus = next(
        (items for items in loaded.values() if items and "body" in items[0]),
        None,
    )
    queries = next(
        (items for items in loaded.values() if items and "evidence_list" in items[0]),
        None,
    )
    if corpus is None or queries is None:
        raise RuntimeError(f"Unexpected MultiHop-RAG configs/columns: {list(loaded)}")
    _write_jsonl(cache_dir / "corpus.jsonl", corpus)
    _write_jsonl(cache_dir / "queries.jsonl", queries)
    return load_bundle(cache_dir, name="multihop_rag")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
