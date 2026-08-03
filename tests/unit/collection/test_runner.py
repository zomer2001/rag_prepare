"""End-to-end runner tests (offline fixtures): idempotence, versioning,
review queue, run report, crash-safe state, and the no-index invariant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rag_graph_demo.collection.runner import CollectionRunner
from rag_graph_demo.collection.scheduler import RunLock, RunLockError, load_manifest


class TestEndToEnd:
    def test_usaspending_full_run(self, runner: CollectionRunner, tmp_path):
        manifest = runner.run_source("usaspending")
        assert manifest.metrics.discovered == 2
        assert manifest.metrics.published_clean == 2
        assert manifest.metrics.new_records == 2
        assert manifest.indexed is False

        clean_root = tmp_path / "collection" / "clean"
        clean_files = list(clean_root.rglob("*.md"))
        assert len(clean_files) == 2
        text = clean_files[0].read_text(encoding="utf-8")
        assert "rag-source" not in text
        assert "award_ceiling" in text or "current_award_amount" in text

        # sidecar carries the four money kinds distinctly
        sidecar = json.loads(next(clean_root.rglob("*.sidecar.json")).read_text())
        kinds = {a["kind"] for a in sidecar["contract"]["amounts"]}
        assert {"current_award_amount", "award_ceiling", "obligation", "outlay"} <= kinds

        # manifest persisted and auditable
        stored = load_manifest(tmp_path / "collection" / "runs" / manifest.run_id / "manifest.json")
        assert stored.config_hash
        assert stored.adapter_versions.get("usaspending") == "1.0.0"
        assert stored.indexed is False

    def test_repeat_run_creates_no_new_records(self, runner: CollectionRunner, tmp_path):
        first = runner.run_source("usaspending")
        second = runner.run_source("usaspending")
        assert first.metrics.new_records == 2
        assert second.metrics.new_records == 0
        assert second.metrics.exact_duplicates == 2
        assert second.metrics.published_clean == 0
        clean_files = list((tmp_path / "collection" / "clean").rglob("*.md"))
        assert len(clean_files) == 2

    def test_updated_content_creates_new_version(
        self, runner: CollectionRunner, fixture_copy, tmp_path
    ):
        runner.run_source("usaspending")
        detail = fixture_copy / "usaspending" / "detail_359204966.json"
        payload = json.loads(detail.read_text())
        payload["total_obligation"] = 90000000.0  # page update
        detail.write_text(json.dumps(payload))

        second = runner.run_source("usaspending")
        assert second.metrics.new_records == 0
        assert second.metrics.new_versions == 1
        record_dirs = [p for p in (tmp_path / "collection" / "normalized").iterdir() if p.is_dir()]
        versioned = [p for p in record_dirs if (p / "v2.json").exists()]
        assert len(versioned) == 1
        latest = json.loads((versioned[0] / "latest.json").read_text())
        assert latest["version_index"] == 2
        kinds = {a["kind"]: a["value"] for a in latest["contract"]["amounts"]}
        assert kinds["obligation"] == "90000000"

    def test_anduril_run_and_review_queue_for_thin_page(self, runner: CollectionRunner, tmp_path):
        manifest = runner.run_source("anduril_official")
        assert manifest.metrics.discovered == 3
        assert manifest.metrics.published_clean >= 2
        clean_root = tmp_path / "collection" / "clean"
        texts = [p.read_text(encoding="utf-8") for p in clean_root.rglob("*.md")]
        joined = "\n".join(texts)
        assert "Roadrunner-M is a high-explosive variant" in joined
        assert "cookie" not in joined.lower()
        # thin old_post page (<200 chars body) should be queued for review, not dropped
        review = list((tmp_path / "collection" / "review_queue").glob("*.json"))
        assert review, "expected thin page in review queue"

    def test_openalex_run_writes_paper_records(self, runner: CollectionRunner, tmp_path):
        manifest = runner.run_source("openalex")
        assert manifest.metrics.published_clean == 2
        sidecar = json.loads(
            next((tmp_path / "collection" / "clean").rglob("*.sidecar.json")).read_text()
        )
        assert sidecar["record_type"] == "paper"

    def test_patents_policy_skip_without_key(self, runner: CollectionRunner, monkeypatch):
        monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)
        manifest = runner.run_source("patentsview")
        assert manifest.metrics.policy_skipped == 1
        assert manifest.metrics.published_clean == 0
        assert manifest.metrics.permanent_failures == 0  # skip is not a failure

    def test_patents_run_with_key(self, runner: CollectionRunner, monkeypatch, tmp_path):
        monkeypatch.setenv("PATENTSVIEW_API_KEY", "fixture-key")
        manifest = runner.run_source("patentsview")
        assert manifest.metrics.published_clean == 1
        sidecar = json.loads(
            next((tmp_path / "collection" / "clean").rglob("*.sidecar.json")).read_text()
        )
        assert sidecar["record_type"] == "patent"
        assert sidecar["patent"]["publication_number"] == "US12345678"

    def test_cursor_not_advanced_when_failures(
        self, runner: CollectionRunner, fixture_copy, tmp_path
    ):
        # break one page: remove its fixture -> 404 -> permanent failure
        (fixture_copy / "anduril_official" / "pages" / "old_post.html").unlink()
        manifest = runner.run_source("anduril_official")
        assert manifest.metrics.permanent_failures >= 1
        state = runner.state_store.load("anduril_official")
        assert "last_successful_run" not in state.cursor  # cursor held for replay

    def test_validate_command_roundtrip(self, runner: CollectionRunner):
        runner.run_source("usaspending")
        latest = next((runner.collection_root / "normalized").iterdir())
        record_id = json.loads((latest / "latest.json").read_text())["record_id"]
        result = runner.validate_record(record_id)
        assert result["valid"], result["errors"]
        missing = runner.validate_record("nope:missing")
        assert not missing["valid"]


class TestLocking:
    def test_second_lock_rejected(self, tmp_path):
        with RunLock(tmp_path, "src"), pytest.raises(RunLockError):
            RunLock(tmp_path, "src").acquire()
        # released: can lock again
        with RunLock(tmp_path, "src"):
            pass


class TestNoIndexInvariant:
    INDEXING_MODULES = ("lightrag", "qdrant_client", "sentence_transformers")

    def test_collection_run_imports_no_indexing_modules(self, runner: CollectionRunner):
        """Order-independent: the collection run itself must not pull in any
        indexing stack, regardless of what other test modules imported."""
        before = set(sys.modules)
        runner.run_source("usaspending")
        new_modules = set(sys.modules) - before
        loaded = {m.split(".")[0] for m in new_modules}
        assert not loaded.intersection(self.INDEXING_MODULES), loaded

    def test_collection_package_has_no_indexing_imports(self):
        import rag_graph_demo.collection as collection_pkg

        pkg_root = Path(collection_pkg.__file__).parent
        for py_file in pkg_root.rglob("*.py"):
            for line in py_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    assert "lightrag" not in stripped.lower(), (py_file, line)
                    assert "qdrant" not in stripped.lower(), (py_file, line)

    def test_run_does_not_touch_existing_data_dirs(self, runner: CollectionRunner, tmp_path):
        runner.run_source("usaspending")
        assert not (tmp_path / "collection" / "lightrag").exists()
        assert not (tmp_path / "data" / "lightrag").exists()
