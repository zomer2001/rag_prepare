from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_graph_demo.collection.registry import SourceRegistry
from rag_graph_demo.collection.runner import CollectionRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "collection"


@pytest.fixture()
def registry() -> SourceRegistry:
    return SourceRegistry(CONFIG_DIR)


@pytest.fixture()
def fixture_copy(tmp_path: Path) -> Path:
    """Writable copy of the offline fixtures (tests may mutate content)."""
    dest = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, dest)
    return dest


@pytest.fixture()
def runner(registry: SourceRegistry, fixture_copy: Path, tmp_path: Path) -> CollectionRunner:
    return CollectionRunner(
        registry,
        collection_root=tmp_path / "collection",
        live=False,
        fixtures_dir=fixture_copy,
    )
