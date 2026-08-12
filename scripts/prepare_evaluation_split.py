from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic query sample with either the full corpus or a "
            "gold-complete corpus plus fixed distractors."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-split-dir",
        type=Path,
        help=(
            "Preserve queries.jsonl from this existing split, then sample the "
            "remaining queries up to --sample-size"
        ),
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        help="Keep all gold documents and sample distractors up to this corpus size",
    )
    parser.add_argument("--distractor-seed", type=int, default=42)
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


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        if source.samefile(destination):
            return "existing-hardlink"
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def stable_document_id(value: dict[str, Any]) -> str:
    explicit = value.get("document_id") or value.get("doc_id") or value.get("id")
    if explicit not in (None, ""):
        return str(explicit)
    identity = "\x1f".join(
        str(value.get(key) or "").strip()
        for key in ("title", "source", "published_at")
    )
    if not identity.strip("\x1f"):
        identity = str(
            value.get("body") or value.get("text") or value.get("content") or ""
        )
    return "doc-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def gold_document_ids(records: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for record in records:
        evidence_values = record.get("gold_evidence")
        if evidence_values is None:
            evidence_values = record.get("evidence_list") or []
        for evidence in evidence_values:
            if not isinstance(evidence, dict):
                continue
            explicit = evidence.get("document_id") or evidence.get("doc_id")
            if explicit not in (None, ""):
                result.add(str(explicit))
            elif any(
                evidence.get(key) not in (None, "")
                for key in ("title", "source", "published_at")
            ):
                result.add(stable_document_id(evidence))
        result.update(str(item) for item in record.get("gold_doc_ids") or [])
    return result


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    corpus_path = source_dir / "corpus.jsonl"
    queries_path = source_dir / "queries.jsonl"
    if not corpus_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError("source-dir requires corpus.jsonl and queries.jsonl")

    records = read_jsonl(queries_path)
    corpus_records = read_jsonl(corpus_path)
    if args.sample_size <= 0 or args.sample_size > len(records):
        raise ValueError(
            f"sample-size must be between 1 and {len(records)}, got {args.sample_size}"
        )
    indexed_records: list[tuple[int, dict[str, Any]]] = []
    for index, value in enumerate(records):
        record = dict(value)
        record.setdefault("query_id", f"q-{index + 1:05d}")
        indexed_records.append((index, record))

    base_split_dir: Path | None = None
    base_records: list[dict[str, Any]] = []
    if args.base_split_dir is not None:
        base_split_dir = args.base_split_dir.resolve()
        base_queries_path = base_split_dir / "queries.jsonl"
        if not base_queries_path.is_file():
            raise FileNotFoundError(
                "base-split-dir requires an existing queries.jsonl"
            )
        base_records = read_jsonl(base_queries_path)
        if args.sample_size < len(base_records):
            raise ValueError(
                f"sample-size {args.sample_size} is smaller than the "
                f"{len(base_records)} preserved base queries"
            )
        source_by_query_id = {
            str(record["query_id"]): record for _, record in indexed_records
        }
        missing_base_ids = sorted(
            str(record.get("query_id") or "")
            for record in base_records
            if str(record.get("query_id") or "") not in source_by_query_id
        )
        if missing_base_ids:
            raise ValueError(
                f"Base split contains {len(missing_base_ids)} query ID(s) "
                "missing from source"
            )
        base_ids = {str(record["query_id"]) for record in base_records}
        candidates = [
            (index, record)
            for index, record in indexed_records
            if str(record["query_id"]) not in base_ids
        ]
        additional_count = args.sample_size - len(base_records)
        selected_candidate_positions = sorted(
            random.Random(args.seed).sample(range(len(candidates)), additional_count)
        )
        selected = [dict(record) for record in base_records]
        selected.extend(
            dict(candidates[position][1]) for position in selected_candidate_positions
        )
    else:
        selected_indices = sorted(
            random.Random(args.seed).sample(range(len(records)), args.sample_size)
        )
        selected = [dict(indexed_records[index][1]) for index in selected_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_gold_ids = gold_document_ids(selected)
    corpus_by_id = {stable_document_id(record): record for record in corpus_records}
    missing_gold_ids = sorted(selected_gold_ids - set(corpus_by_id))
    if missing_gold_ids:
        raise ValueError(
            f"Selected queries reference {len(missing_gold_ids)} missing gold document(s)"
        )

    if args.corpus_size is None:
        selected_corpus = corpus_records
        distractor_ids = set(corpus_by_id) - selected_gold_ids
        corpus_mode = link_or_copy(corpus_path, output_dir / "corpus.jsonl")
    else:
        if args.corpus_size < len(selected_gold_ids):
            raise ValueError(
                f"corpus-size {args.corpus_size} is smaller than the "
                f"{len(selected_gold_ids)} required gold documents"
            )
        if args.corpus_size > len(corpus_records):
            raise ValueError(
                f"corpus-size cannot exceed source corpus size {len(corpus_records)}"
            )
        distractor_candidates = sorted(set(corpus_by_id) - selected_gold_ids)
        distractor_count = args.corpus_size - len(selected_gold_ids)
        distractor_ids = set(
            random.Random(args.distractor_seed).sample(
                distractor_candidates,
                distractor_count,
            )
        )
        selected_ids = selected_gold_ids | distractor_ids
        selected_corpus = [
            record
            for record in corpus_records
            if stable_document_id(record) in selected_ids
        ]
        write_jsonl(output_dir / "corpus.jsonl", selected_corpus)
        corpus_mode = "gold-complete-plus-fixed-distractors"
    output_queries = output_dir / "queries.jsonl"
    if output_queries.exists():
        raise FileExistsError(output_queries)
    write_jsonl(output_queries, selected)
    query_ids = [str(record["query_id"]) for record in selected]
    manifest = {
        "source_dir": str(source_dir),
        "source_query_count": len(records),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sampling": (
            "base queries preserved in their existing order; additional queries "
            "sampled with python random.Random(seed).sample and restored to source order"
            if base_records
            else "python random.Random(seed).sample; selected records restored to source order"
        ),
        "base_split_dir": str(base_split_dir) if base_split_dir else None,
        "base_query_count": len(base_records),
        "additional_query_count": len(selected) - len(base_records),
        "query_ids": query_ids,
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "corpus_mode": corpus_mode,
        "source_corpus_count": len(corpus_records),
        "corpus_size": len(selected_corpus),
        "gold_document_count": len(selected_gold_ids),
        "distractor_count": len(distractor_ids),
        "distractor_seed": args.distractor_seed,
        "corpus_document_ids_sha256": hashlib.sha256(
            "\n".join(stable_document_id(record) for record in selected_corpus).encode()
        ).hexdigest(),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {args.sample_size} queries from {len(records)} with seed "
        f"{args.seed}: {output_dir} ({len(selected_gold_ids)} gold + "
        f"{len(distractor_ids)} distractor documents)"
    )


if __name__ == "__main__":
    main()
