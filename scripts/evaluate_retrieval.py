from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from rag_graph_demo.config import get_settings
from rag_graph_demo.evaluation.datasets import load_bundle, prepare_multihop_rag
from rag_graph_demo.evaluation.runner import VALID_METHODS, EvaluationRunner
from rag_graph_demo.evaluation.runtime_lock import EvaluationRuntimeLock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Naive, Hybrid, stock LightRAG, controlled-entity LightRAG, and "
            "event-enhanced retrieval plus Agentic variants with one dataset and emit "
            "comparable metrics."
        )
    )
    parser.add_argument(
        "--dataset",
        default="multihop-rag",
        help="multihop-rag or a local benchmark bundle name",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directory with corpus.json[l] and queries.json[l]",
    )
    parser.add_argument(
        "--methods",
        default="naive,hybrid,light",
        help=f"Comma-separated methods: {','.join(VALID_METHODS)}",
    )
    parser.add_argument("--cutoffs", default="4,10")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N queries")
    parser.add_argument(
        "--corpus-limit",
        type=int,
        help="Smoke-test only: index the first N corpus documents",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--chunking-strategy",
        choices=("fixed", "recursive"),
        default="fixed",
        help="fixed token windows or boundary-aware recursive splitting",
    )
    parser.add_argument("--index-batch-size", type=int, default=16)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Persistent index/cache directory; defaults under data/lightrag/evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory; defaults under data/exports/evaluation",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars for CI or redirected logs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download/validate the dataset without building indexes or running queries",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.dataset_dir:
        dataset = load_bundle(args.dataset_dir, name=args.dataset)
    elif args.dataset.casefold() in {"multihop-rag", "multihop_rag"}:
        dataset = prepare_multihop_rag(Path("data/benchmarks/multihop_rag"))
    else:
        raise ValueError("--dataset-dir is required for non-MultiHop-RAG datasets")

    if args.prepare_only:
        print(
            f"Prepared {dataset.name}: {len(dataset.documents)} documents, "
            f"{len(dataset.queries)} queries"
        )
        return 0

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    cutoffs = tuple(int(item) for item in args.cutoffs.split(",") if item.strip())
    dataset_name = dataset.name.replace("/", "_")
    runtime_dir = args.runtime_dir or Path("data/lightrag/evaluation") / dataset_name
    output_dir = args.output_dir or Path("data/exports/evaluation") / dataset_name
    runner = EvaluationRunner(
        dataset,
        settings,
        runtime_dir=runtime_dir,
        output_dir=output_dir,
        methods=methods,
        cutoffs=cutoffs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        query_limit=args.limit,
        corpus_limit=args.corpus_limit,
        index_batch_size=args.index_batch_size,
        fail_fast=args.fail_fast,
        show_progress=not args.no_progress,
    )
    with EvaluationRuntimeLock(runtime_dir):
        result = await runner.run()
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))
    print(f"Reports written to: {output_dir.resolve()}")
    return 0


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
