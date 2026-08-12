from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark bundles while preserving dataset-specific metric slices."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="ALIAS=DIR",
        help="Repeat for each source bundle",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def explicit_id(record: dict[str, Any], *, kind: str) -> str:
    keys = ("document_id", "doc_id", "id") if kind == "document" else ("query_id", "id")
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Source {kind} record has no explicit ID")


def parse_source(value: str) -> tuple[str, Path]:
    alias, separator, path = value.partition("=")
    if not separator or not alias.strip() or not path.strip():
        raise ValueError(f"Expected --source ALIAS=DIR, got {value!r}")
    alias = alias.strip()
    if not all(character.isalnum() or character in "-_" for character in alias):
        raise ValueError(f"Invalid source alias: {alias}")
    return alias, Path(path).resolve()


def rewrite_query(
    query: dict[str, Any],
    *,
    alias: str,
    document_ids: set[str],
) -> dict[str, Any]:
    result = dict(query)
    result["query_id"] = f"{alias}::{explicit_id(query, kind='query')}"
    original_type = str(query.get("question_type") or query.get("type") or "unknown")
    result["question_type"] = f"{alias}:{original_type}"
    result["source_dataset"] = alias

    evidence_values = query.get("gold_evidence")
    if evidence_values is None:
        evidence_values = query.get("evidence_list") or []
    rewritten_evidence: list[Any] = []
    for evidence in evidence_values:
        if not isinstance(evidence, dict):
            rewritten_evidence.append(evidence)
            continue
        item = dict(evidence)
        document_id = item.get("document_id") or item.get("doc_id")
        if document_id not in (None, ""):
            document_id = str(document_id)
            if document_id not in document_ids:
                raise ValueError(
                    f"Query {result['query_id']} references missing document {document_id}"
                )
            item["document_id"] = f"{alias}::{document_id}"
            item.pop("doc_id", None)
        rewritten_evidence.append(item)
    result["gold_evidence"] = rewritten_evidence
    result.pop("evidence_list", None)

    rewritten_gold_ids = []
    for document_id in query.get("gold_doc_ids") or []:
        value = str(document_id)
        if value not in document_ids:
            raise ValueError(f"Query {result['query_id']} references missing document {value}")
        rewritten_gold_ids.append(f"{alias}::{value}")
    if rewritten_gold_ids:
        result["gold_doc_ids"] = rewritten_gold_ids
    return result


def merge_bundles(
    sources: list[tuple[str, Path]], output_dir: Path, *, name: str
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for alias, source_dir in sources:
        source_corpus = read_jsonl(source_dir / "corpus.jsonl")
        source_queries = read_jsonl(source_dir / "queries.jsonl")
        document_ids = {explicit_id(item, kind="document") for item in source_corpus}
        if len(document_ids) != len(source_corpus):
            raise ValueError(f"Duplicate document IDs in {source_dir}")
        query_ids = {explicit_id(item, kind="query") for item in source_queries}
        if len(query_ids) != len(source_queries):
            raise ValueError(f"Duplicate query IDs in {source_dir}")

        for document in source_corpus:
            item = dict(document)
            item["document_id"] = f"{alias}::{explicit_id(document, kind='document')}"
            item.pop("doc_id", None)
            item["source_dataset"] = alias
            corpus.append(item)
        queries.extend(
            rewrite_query(query, alias=alias, document_ids=document_ids) for query in source_queries
        )
        source_summaries.append(
            {
                "alias": alias,
                "directory": str(source_dir),
                "documents": len(source_corpus),
                "queries": len(source_queries),
            }
        )

    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "queries.jsonl", queries)
    query_ids = [str(item["query_id"]) for item in queries]
    document_ids = [str(item["document_id"]) for item in corpus]
    manifest = {
        "dataset": name,
        "sources": source_summaries,
        "query_count": len(queries),
        "corpus_size": len(corpus),
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "corpus_document_ids_sha256": hashlib.sha256("\n".join(document_ids).encode()).hexdigest(),
        "id_policy": "source alias prefix added to every document and query ID",
        "question_type_policy": "source alias prefix retained for per-dataset slices",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    sources = [parse_source(value) for value in args.source]
    aliases = [alias for alias, _ in sources]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Source aliases must be unique")
    manifest = merge_bundles(sources, args.output_dir.resolve(), name=args.name)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Merged benchmark: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
