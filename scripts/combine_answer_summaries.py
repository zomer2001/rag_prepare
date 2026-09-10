#!/usr/bin/env python3
"""Combine answer-evaluation summaries and write comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_answers import write_summary_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined: dict[str, dict[str, Any]] = {}
    for path in args.input:
        values = json.loads(path.resolve().read_text(encoding="utf-8"))
        for run_id, summary in values.items():
            if run_id in combined and combined[run_id] != summary:
                raise ValueError(f"Conflicting summary for {run_id}: {path}")
            combined[run_id] = summary
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answer_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_tables(output_dir, combined)
    print(f"Combined {len(combined)} answer runs in: {output_dir}")


if __name__ == "__main__":
    main()
