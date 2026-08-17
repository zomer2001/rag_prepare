from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import TopicProfile
from .runner import TopicRunner, state_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_graph_demo.osint.cli",
        description="Single-process topic-driven OSINT crawler with bounded expansion.",
    )
    parser.add_argument("--topic", required=True, help="path to a TopicProfile YAML file")
    parser.add_argument("--data-dir", default="data", help="project data directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="create or show the topic query plan without crawling")
    run = sub.add_parser("run", help="discover, fetch, store, and expand the topic")
    run.add_argument("--live", action="store_true", help="allow real network access")
    run.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="maximum additional rounds for this invocation",
    )
    sub.add_parser("status", help="show persisted topic progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = TopicProfile.load(Path(args.topic))
    runner = TopicRunner(profile, data_dir=args.data_dir, live=getattr(args, "live", False))
    if args.command == "plan":
        state = runner.plan()
    elif args.command == "run":
        state = runner.run(rounds=args.rounds)
    else:
        state = runner.state_store.load(profile)
    print(json.dumps(state_summary(state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
