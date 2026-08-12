from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_graph_demo.evaluation.public_benchmarks import (
    download_and_prepare_beir,
    download_hotpotqa,
    prepare_hotpotqa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and create deterministic small HotpotQA or BEIR retrieval bundles."
    )
    parser.add_argument(
        "dataset",
        help="hotpotqa or beir-<dataset>, for example beir-scifact or beir-nfcorpus",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/benchmarks/_downloads"),
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--corpus-size", type=int, default=100)
    parser.add_argument("--distractor-seed", type=int, default=20260807)
    parser.add_argument("--split", default="test", help="BEIR qrels split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.casefold().replace("_", "-")
    if dataset == "hotpotqa":
        raw_path = download_hotpotqa(args.download_dir)
        manifest = prepare_hotpotqa(
            raw_path,
            args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    elif dataset.startswith("beir-") and len(dataset) > len("beir-"):
        name = dataset.removeprefix("beir-")
        if not all(character.isalnum() or character in "-_" for character in name):
            raise ValueError(f"Invalid BEIR dataset name: {name}")
        manifest = download_and_prepare_beir(
            args.download_dir / "beir",
            args.output_dir,
            name=name,
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
            corpus_size=args.corpus_size,
            distractor_seed=args.distractor_seed,
        )
    else:
        raise ValueError("dataset must be hotpotqa or beir-<dataset>")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Prepared benchmark: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
