"""Dedupe/versioning: stable IDs, version bump on change, exact duplicates,
SimHash near-duplicate relations (relations only, never deletion)."""

from __future__ import annotations

from rag_graph_demo.collection.dedupe import (
    RecordStore,
    SimHashIndex,
    hamming_distance,
    make_record_id,
    normalized_hash,
    simhash64,
)


class TestRecordId:
    def test_provider_id_preferred_and_stable(self):
        a = make_record_id("usaspending", "CONT_AWD_X", None)
        b = make_record_id("usaspending", "CONT_AWD_X", "https://ignored.example/")
        assert a == b and a.startswith("usaspending:")

    def test_url_fallback(self):
        assert make_record_id("anduril_official", None, "https://x.com/a") != make_record_id(
            "anduril_official", None, "https://x.com/b"
        )


class TestVersioning:
    def _record(self, version_id: str, body: str = "body"):
        return {
            "record_id": "src:r1",
            "version_id": version_id,
            "version_index": 1,
            "title": "t",
            "body_markdown": body,
            "first_seen_at": "2026-08-01T00:00:00Z",
            "last_seen_at": "2026-08-03T00:00:00Z",
        }

    def test_same_content_no_new_version(self, tmp_path):
        store = RecordStore(tmp_path)
        assert store.save_version(self._record("v1")) == (True, True)
        assert store.save_version(self._record("v1")) == (False, False)
        latest = store.load_latest("src:r1")
        assert latest["version_index"] == 1

    def test_changed_content_new_version_same_record(self, tmp_path):
        store = RecordStore(tmp_path)
        store.save_version(self._record("v1"))
        is_new_record, is_new_version = store.save_version(self._record("v2"))
        assert (is_new_record, is_new_version) == (False, True)
        latest = store.load_latest("src:r1")
        assert latest["version_index"] == 2
        assert latest["first_seen_at"] == "2026-08-01T00:00:00Z"  # preserved
        assert (tmp_path / "src:r1" / "v1.json").exists()
        assert (tmp_path / "src:r1" / "v2.json").exists()


class TestSimHash:
    def test_identical_text_zero_distance(self):
        assert hamming_distance(simhash64("hello world"), simhash64("hello world")) == 0

    def test_near_duplicate_detected_not_deleted(self, tmp_path):
        index = SimHashIndex(tmp_path / "sim.jsonl")
        text_a = "Roadrunner-M intercepts hostile unmanned aircraft at range. " * 5
        text_b = "Roadrunner-M intercepts hostile unmanned aircraft at range! " * 5
        index.add("src:r1", "v1", text_a)
        found = index.candidates(text_b)
        assert found and found[0][0] == "src:r1"

    def test_unrelated_text_not_flagged(self, tmp_path):
        index = SimHashIndex(tmp_path / "sim.jsonl")
        index.add("src:r1", "v1", "quantum chromodynamics lattice gauge theory " * 10)
        assert index.candidates("chocolate cake recipe with butter and sugar " * 10) == []

    def test_normalized_hash_is_unicode_nfc(self):
        assert normalized_hash("café") == normalized_hash("café")
