from __future__ import annotations

import asyncio
import json

from rag_graph_demo.config import get_settings
from rag_graph_demo.services.container import AppContainer


async def migrate() -> dict[str, int | str]:
    container = AppContainer(get_settings())
    if container.neo4j_store is None:
        raise RuntimeError("NEO4J_ENABLED must be true")
    try:
        await container.neo4j_store.verify()
        await container.neo4j_store.initialize_schema()
        await container.tkg_service.sync_neo4j()
        snapshot = await container.tkg_registry.snapshot()
        return {
            "status": "success",
            "entities": len(snapshot.entities),
            "assertions": len(snapshot.assertions),
            "reviews": len(snapshot.review_cases),
            "merges": len(snapshot.merge_records),
        }
    finally:
        await container.shutdown()


def main() -> None:
    print(json.dumps(asyncio.run(migrate()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
