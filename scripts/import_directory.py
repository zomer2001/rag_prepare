from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rag_graph_demo.config import get_settings
from rag_graph_demo.domain.models import JobStatus
from rag_graph_demo.services.container import AppContainer
from rag_graph_demo.storage.file_store import ALLOWED_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import supported documents from a local directory."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Register files without building the LightRAG index.",
    )
    return parser.parse_args()


async def run(directory: Path, *, register_only: bool) -> int:
    settings = get_settings()
    if not directory.is_dir():
        print(f"Directory not found: {directory}")
        return 2
    if not register_only and not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is required unless --register-only is used.")
        return 2

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    if not files:
        print("No supported Markdown, DOCX, or PDF files found.")
        return 0

    container = AppContainer(settings)
    failures = 0
    try:
        if not register_only:
            print("Checking local embedding model...")
            await container.engine.warmup_embedding()
        for path in files:
            document, duplicate = await container.document_service.create_upload(
                filename=path.name,
                mime_type=None,
                content=path.read_bytes(),
            )
            if duplicate:
                print(f"SKIP  {path.name} (duplicate of {document.document_id})")
                continue
            if register_only:
                print(f"ADD   {path.name} ({document.document_id})")
                continue

            job = await container.document_service.create_job(
                document.document_id,
                "index",
            )
            print(f"INDEX {path.name} ({document.document_id})")
            await container.document_service.index_document(
                document.document_id,
                job.job_id,
            )
            finished = await container.jobs_registry.get(job.job_id)
            if finished is None or finished.status != JobStatus.SUCCEEDED:
                failures += 1
                print(f"FAIL  {path.name}: {finished.message if finished else 'unknown'}")
            else:
                print(f"READY {path.name}")
    finally:
        await container.shutdown()

    print(f"Imported {len(files) - failures}/{len(files)} discovered files.")
    return 1 if failures else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(
        asyncio.run(run(args.directory, register_only=args.register_only))
    )


if __name__ == "__main__":
    main()
