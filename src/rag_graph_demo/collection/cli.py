"""Collection pipeline CLI (brief §7.1).

Examples:
    uv run python -m rag_graph_demo.collection.cli discover --source anduril_official
    uv run python -m rag_graph_demo.collection.cli run --source usaspending --since 2026-07-01
    uv run python -m rag_graph_demo.collection.cli run --group p0 --no-index
    uv run python -m rag_graph_demo.collection.cli retry --run-id <id>
    uv run python -m rag_graph_demo.collection.cli validate --record-id <id>

``--no-index`` is the default and the only allowed mode in this phase: the
pipeline never calls LightRAG, embeddings, Qdrant, or any reindex API.
Network access is OFF by default (offline fixtures); pass ``--live``
explicitly for real requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import SourceRegistry
from .runner import CollectionRunner
from .scheduler import load_manifest

DEFAULT_COLLECTION_ROOT = "data/collection"
DEFAULT_FIXTURES_DIR = "tests/fixtures/collection"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_graph_demo.collection.cli",
        description="Anduril OSINT collection pipeline (no indexing in this phase).",
    )
    parser.add_argument(
        "--config-dir", default="config", help="config directory (sources.yaml etc.)"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_COLLECTION_ROOT, help="collection output root"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow real network access (default: offline fixtures)",
    )
    parser.add_argument("--fixtures-dir", default=DEFAULT_FIXTURES_DIR, help="offline fixture root")
    parser.add_argument(
        "--no-index",
        action="store_true",
        default=True,
        help="default and only allowed mode; indexing is not implemented in this phase",
    )
    parser.add_argument("--limit", type=int, default=None, help="max items per source")
    parser.add_argument("--since", default=None, help="ISO date lower bound for incremental runs")

    sub = parser.add_subparsers(dest="command", required=True)

    # --no-index is accepted both as a global flag and after a subcommand
    # (the brief's examples use `run --group p0 --no-index`).
    no_index = argparse.ArgumentParser(add_help=False)
    no_index.add_argument(
        "--no-index",
        action="store_true",
        default=True,
        help="default and only allowed mode; indexing is not implemented in this phase",
    )

    discover = sub.add_parser(
        "discover", parents=[no_index], help="discover candidates only (no body download)"
    )
    discover.add_argument("--source", action="append", required=True, help="source_id (repeatable)")
    discover.add_argument("--group", default=None, help="source group, e.g. p0")

    run = sub.add_parser("run", parents=[no_index], help="full pipeline for selected sources")
    run.add_argument("--source", action="append", default=None, help="source_id (repeatable)")
    run.add_argument("--group", default=None, help="source group, e.g. p0")

    retry = sub.add_parser(
        "retry", parents=[no_index], help="re-run retryable failures from a previous run"
    )
    retry.add_argument("--run-id", required=True)

    validate = sub.add_parser("validate", parents=[no_index], help="re-validate one stored record")
    validate.add_argument("--record-id", required=True)

    return parser


def _select_sources(
    registry: SourceRegistry, source_ids: list[str] | None, group: str | None
) -> list[str]:
    if source_ids:
        return source_ids
    if group:
        return [s.source_id for s in registry.by_group(group)]
    return [s.source_id for s in registry.enabled()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = SourceRegistry(args.config_dir)
    runner = CollectionRunner(
        registry,
        collection_root=args.data_dir,
        live=args.live,
        fixtures_dir=args.fixtures_dir,
        limit=args.limit,
        since=args.since,
    )

    if args.command == "validate":
        result = runner.validate_record(args.record_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1

    if args.command == "retry":
        manifest_path = Path(args.data_dir) / "runs" / args.run_id / "manifest.json"
        if not manifest_path.exists():
            print(f"run manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        previous = load_manifest(manifest_path)
        retryable_urls = [e.url for e in previous.errors if e.retryable and e.url]
        if not retryable_urls:
            print(
                json.dumps({"run_id": args.run_id, "retried": 0, "note": "no retryable failures"})
            )
            return 0
        manifests = [
            runner.run_source(sid, candidate_urls=retryable_urls)
            for sid in previous.source_ids
        ]
        _print_reports(manifests)
        return _exit_code(manifests)

    source_ids = _select_sources(
        registry, getattr(args, "source", None), getattr(args, "group", None)
    )
    if args.command == "discover":
        manifests = [runner.discover_source(sid) for sid in source_ids]
    else:  # run
        manifests = [runner.run_source(sid) for sid in source_ids]
    _print_reports(manifests)
    return _exit_code(manifests)


def _print_reports(manifests) -> None:
    reports = [m.model_dump(mode="json") for m in manifests]
    print(json.dumps(reports[0] if len(reports) == 1 else reports, ensure_ascii=False, indent=2))


def _exit_code(manifests) -> int:
    # "no new data" is not a failure (brief §7.3); permanent failures are.
    for manifest in manifests:
        if manifest.metrics.permanent_failures > 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
