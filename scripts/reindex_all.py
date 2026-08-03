from __future__ import annotations

import asyncio

from rag_graph_demo.config import get_settings
from rag_graph_demo.domain.models import JobStatus
from rag_graph_demo.services.container import AppContainer


async def run() -> int:
    settings = get_settings()
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is required to rebuild the LightRAG index.")
        return 2

    container = AppContainer(settings)
    failures = 0
    try:
        documents = await container.document_service.list_documents()
        if not documents:
            print("No registered documents to reindex.")
            return 0

        for document in reversed(documents):
            job = await container.document_service.create_job(
                document.document_id,
                "reindex",
            )
            print(f"REINDEX {document.filename} ({document.document_id})")
            await container.document_service.reindex_document(
                document.document_id,
                job.job_id,
            )
            finished = await container.jobs_registry.get(job.job_id)
            if finished is None or finished.status != JobStatus.SUCCEEDED:
                failures += 1
                print(
                    f"FAIL    {document.filename}: "
                    f"{finished.message if finished else 'unknown'}"
                )
            else:
                print(f"READY   {document.filename}")
    finally:
        await container.shutdown()

    print(f"Reindexed {len(documents) - failures}/{len(documents)} documents.")
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
