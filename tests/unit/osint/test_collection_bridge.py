from __future__ import annotations

import json
from pathlib import Path

from rag_graph_demo.collection.models import (
    AuthorityTier,
    DiscoveryCandidate,
    RecordType,
    SourceConfig,
    SourcePolicy,
)
from rag_graph_demo.collection.runner import CollectionRunner


class Registry:
    def __init__(self, source: SourceConfig) -> None:
        self.source = source

    def get(self, source_id: str) -> SourceConfig:
        assert source_id == self.source.source_id
        return self.source

    def config_hash(self) -> str:
        return "fixture-config"


def test_external_candidate_reuses_collection_evidence_pipeline(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures" / "generic"
    fixture_root.mkdir(parents=True)
    (fixture_root / "page.html").write_text(
        """<html><head><title>Topic Evidence</title></head><body>
        <main><h1>Topic Evidence</h1><p>This is a sufficiently detailed canonical body
        about a public research topic and its supporting evidence. It is long enough to
        pass the normal collection quality path without becoming a search summary.</p></main>
        </body></html>""",
        encoding="utf-8",
    )
    (fixture_root / "manifest.json").write_text(
        json.dumps({"GET https://example.com/topic": "page.html"}),
        encoding="utf-8",
    )
    source = SourceConfig(
        source_id="topic_fixture",
        name="fixture topic",
        publisher="Example",
        authority_tier=AuthorityTier.P2,
        record_type=RecordType.OFFICIAL_PAGE,
        adapter="generic_web",
        fixture_dir="generic",
        policy=SourcePolicy(rate_limit_seconds=0),
    )
    runner = CollectionRunner(
        Registry(source),  # type: ignore[arg-type]
        collection_root=tmp_path / "collection",
        live=False,
        fixtures_dir=tmp_path / "fixtures",
    )
    manifest = runner.run_candidates(
        source.source_id,
        [
            DiscoveryCandidate(
                candidate_id="candidate-1",
                source_id=source.source_id,
                url="https://example.com/topic",
                title="Search title",
                summary="Discovery summary must not be the body",
                discovered_by="search",
                extra={"topic_id": "fixture"},
            )
        ],
    )
    assert manifest.metrics.published_clean == 1
    clean = next((tmp_path / "collection" / "clean").rglob("*.md")).read_text()
    assert "sufficiently detailed canonical body" in clean
    assert "Discovery summary must not be the body" not in clean
