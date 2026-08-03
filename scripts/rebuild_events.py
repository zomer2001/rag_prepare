#!/usr/bin/env python3
"""Resume-safe batch event extraction for already parsed documents."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from rag_graph_demo.config import get_settings
from rag_graph_demo.domain.models import JobStatus
from rag_graph_demo.services.container import AppContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the event assertion layer without rebuilding LightRAG."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        choices=range(1, 5),
        metavar="1-4",
        help="Concurrent DeepSeek extraction calls (default: 2)",
    )
    parser.add_argument("--limit", type=int, help="Process at most N documents")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild documents whose event index is already ready",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Only process this document id; may be repeated",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    if not settings.event_extraction_enabled:
        print("EVENT_EXTRACTION_ENABLED is false; nothing was changed.")
        return 2
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is not configured; nothing was changed.")
        return 2

    container = AppContainer(settings)
    try:
        documents = await container.document_service.list_documents()
        selected_ids = set(args.document_id)
        candidates = [
            item
            for item in documents
            if item.parsed_path
            and Path(item.parsed_path).exists()
            and (not selected_ids or item.document_id in selected_ids)
            and (args.force or item.event_index_status != "ready")
        ]
        candidates.sort(key=lambda item: Path(item.parsed_path or "").stat().st_size)
        if args.limit is not None:
            candidates = candidates[: max(0, args.limit)]
        if not candidates:
            derived = await container.events_registry.rebuild_derived()
            print("No documents need event rebuilding.")
            print("Refreshed derived event data:", derived)
            return 0

        print(
            f"Rebuilding events for {len(candidates)} document(s) "
            f"with {args.workers} worker(s)."
        )
        queue: asyncio.Queue = asyncio.Queue()
        for document in candidates:
            queue.put_nowait(document)

        counts: Counter[str] = Counter()
        completed = 0
        output_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal completed
            while True:
                try:
                    document = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                job = await container.document_service.create_job(
                    document.document_id, "events"
                )
                await container.document_service.rebuild_events(
                    document.document_id, job.job_id
                )
                finished = await container.jobs_registry.get(job.job_id)
                outcome = finished.status.value if finished else "missing"
                counts[outcome] += 1
                completed += 1
                async with output_lock:
                    detail = finished.message if finished else "job record missing"
                    print(
                        f"[{completed}/{len(candidates)}] {outcome.upper()} "
                        f"{document.title}: {detail}",
                        flush=True,
                    )
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(args.workers)))
        derived = await container.events_registry.rebuild_derived()
        print("Summary:", dict(counts))
        print("Derived event data:", derived)
        return 0 if counts[JobStatus.FAILED.value] == 0 else 1
    finally:
        await container.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
