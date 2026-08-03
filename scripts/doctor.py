from __future__ import annotations

import asyncio
import json

from rag_graph_demo.services.diagnostics import build_diagnostics


def main() -> None:
    result = asyncio.run(build_diagnostics(check_qdrant=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["summary"]["ready"] else 1)


if __name__ == "__main__":
    main()

