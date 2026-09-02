#!/usr/bin/env python3
"""Create a deterministic DWIE test subset for entity evaluation."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "benchmarks" / "extraction" / "dwie" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE / "annos_with_content")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    candidates = []
    for path in sorted(source_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "test" in document.get("tags", []):
            candidates.append(path)
    if not 0 < args.sample_size <= len(candidates):
        raise ValueError(f"sample-size must be between 1 and {len(candidates)}")

    selected_set = set(random.Random(args.seed).sample(candidates, args.sample_size))
    selected = [path for path in candidates if path in selected_set]
    output_dir = args.output_dir or (
        ROOT
        / "data"
        / "benchmarks"
        / "extraction"
        / f"dwie_test_seed{args.seed}_n{args.sample_size}"
    )
    output_dir = output_dir.resolve()
    gold_dir = output_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    for stale in gold_dir.glob("*.json"):
        stale.unlink()
    for path in selected:
        shutil.copy2(path, gold_dir / path.name)

    manifest = {
        "dataset": "DWIE",
        "source_dir": str(source_dir),
        "split": "test",
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sampling": "random.Random(seed).sample; restored to sorted source order",
        "document_ids": [path.stem for path in selected],
        "evaluation_scope": ["Person", "Organization", "Location"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
