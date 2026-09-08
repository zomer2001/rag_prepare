#!/usr/bin/env python3
"""Create a deterministic, question-type-stratified split with non-empty Gold evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def proportional_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if sample_size <= 0 or sample_size > total:
        raise ValueError(f"sample-size must be between 1 and {total}")
    raw = {key: sample_size * value / total for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_queries(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, source in enumerate(rows, start=1):
        evidence = source.get("evidence_list") or []
        if not evidence:
            continue
        row = dict(source)
        row.setdefault("query_id", f"q-{index:05d}")
        groups[str(row.get("question_type") or "unknown")].append((index, row))
    counts = {key: len(values) for key, values in groups.items()}
    quotas = proportional_quotas(counts, sample_size)
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    for key in sorted(groups):
        values = groups[key]
        positions = rng.sample(range(len(values)), quotas[key])
        selected.extend(values[position] for position in positions)
    selected.sort(key=lambda pair: pair[0])
    return [row for _, row in selected], counts, quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_corpus = source_dir / "corpus.jsonl"
    source_queries = source_dir / "queries.jsonl"
    if not source_corpus.is_file() or not source_queries.is_file():
        raise FileNotFoundError("source-dir requires corpus.jsonl and queries.jsonl")
    selected, eligible_counts, quotas = select_queries(
        read_jsonl(source_queries),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_corpus = output_dir / "corpus.jsonl"
    output_queries = output_dir / "queries.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    if any(path.exists() for path in (output_corpus, output_queries, manifest_path)):
        raise FileExistsError(f"Refusing to overwrite an existing split: {output_dir}")
    try:
        os.link(source_corpus, output_corpus)
        corpus_mode = "hardlink"
    except OSError:
        output_corpus.write_bytes(source_corpus.read_bytes())
        corpus_mode = "copy"
    write_jsonl_atomic(output_queries, selected)
    query_ids = [str(row["query_id"]) for row in selected]
    manifest = {
        "source_dir": str(source_dir),
        "sample_size": len(selected),
        "seed": args.seed,
        "eligibility": "non-empty evidence_list",
        "sampling": "proportional by question_type; deterministic random sample; source order",
        "eligible_counts": eligible_counts,
        "selected_counts": quotas,
        "query_ids": query_ids,
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "corpus_mode": corpus_mode,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
