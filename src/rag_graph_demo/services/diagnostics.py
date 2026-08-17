from __future__ import annotations

import asyncio
import platform
import sys
from pathlib import Path
from typing import Any

import httpx
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient

from rag_graph_demo.config import get_settings


def _check_directory(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"ok": True, "path": str(path)}
    except OSError as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}


def _check_qdrant() -> dict[str, Any]:
    settings = get_settings()
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=3,
            check_compatibility=False,
            trust_env=False,
        )
        collections = client.get_collections().collections
        return {
            "ok": True,
            "url": settings.qdrant_url,
            "collections": len(collections),
        }
    except Exception as exc:
        return {"ok": False, "url": settings.qdrant_url, "error": str(exc)}


def _check_deepseek() -> dict[str, Any]:
    settings = get_settings()
    result: dict[str, Any] = {
        "configured": bool(settings.deepseek_api_key),
        "base_url": settings.deepseek_base_url,
        "chat_model": settings.deepseek_chat_model,
        "extraction_model": settings.deepseek_extraction_model,
    }
    if not settings.deepseek_api_key:
        return {**result, "ok": False, "error": "DEEPSEEK_API_KEY is not configured"}

    try:
        with httpx.Client(
            base_url=settings.deepseek_base_url,
            timeout=15,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = client.get(
                "/models",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            )
        response.raise_for_status()
        model_ids = {
            str(item.get("id"))
            for item in response.json().get("data", [])
            if item.get("id")
        }
        missing = sorted(
            {settings.deepseek_chat_model, settings.deepseek_extraction_model}
            - model_ids
        )
        if missing:
            return {
                **result,
                "ok": False,
                "error": f"Configured model(s) unavailable: {', '.join(missing)}",
                "available_models": sorted(model_ids),
            }
        return {**result, "ok": True, "available_models": sorted(model_ids)}
    except Exception as exc:
        return {**result, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _check_neo4j() -> dict[str, Any]:
    settings = get_settings()
    if not settings.neo4j_enabled:
        return {"ok": None, "enabled": False}
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        await driver.verify_connectivity()
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (n:RagTKG {workspace_id: $workspace_id}) "
                "RETURN count(n) AS nodes",
                workspace_id=settings.neo4j_workspace,
            )
            row = await result.single()
        return {
            "ok": True,
            "enabled": True,
            "uri": settings.neo4j_uri,
            "database": settings.neo4j_database,
            "workspace_id": settings.neo4j_workspace,
            "nodes": row["nodes"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "enabled": True,
            "uri": settings.neo4j_uri,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        await driver.close()


async def build_diagnostics(*, check_qdrant: bool) -> dict[str, Any]:
    settings = get_settings()
    directories = [_check_directory(path) for path in settings.required_data_dirs]
    qdrant = await asyncio.to_thread(_check_qdrant) if check_qdrant else {"ok": None}
    neo4j = await _check_neo4j()
    deepseek = await asyncio.to_thread(_check_deepseek)
    embedding_key_configured = {
        "local": True,
        "bailian": bool(settings.bailian_api_key),
        "zhipu": bool(settings.zhipu_api_key),
    }[settings.embedding_provider]
    effective_embedding_model = {
        "local": settings.embedding_model_id,
        "bailian": settings.bailian_embedding_model,
        "zhipu": settings.zhipu_embedding_model,
    }[settings.embedding_provider]
    embedding = {
        "ok": embedding_key_configured,
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "model_id": settings.embedding_model_id,
        "effective_model": effective_embedding_model,
        "dimension": settings.embedding_dimension,
        "device": settings.embedding_device,
    }
    if not embedding["ok"]:
        embedding["error"] = f"{settings.embedding_provider.upper()}_API_KEY is not configured"
    neo4j_ready = not settings.neo4j_required or bool(neo4j["ok"])
    infrastructure_ready = (
        all(item["ok"] for item in directories)
        and bool(qdrant["ok"])
        and neo4j_ready
    )
    ready = infrastructure_ready and bool(deepseek["ok"]) and bool(embedding["ok"])

    return {
        "summary": {
            "ready": ready,
            "infrastructure_ready": infrastructure_ready,
            "can_index_and_query": ready,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "directories": directories,
        "qdrant": qdrant,
        "neo4j": neo4j,
        "deepseek": deepseek,
        "embedding": embedding,
        "lightrag": {
            "workspace": settings.lightrag_workspace,
            "working_dir": str(settings.lightrag_working_dir),
            "default_query_mode": settings.lightrag_query_mode,
        },
    }
