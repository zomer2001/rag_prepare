"""CLI smoke tests: --help, discover/run with fixtures, retry with no
retryable failures, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_graph_demo.collection.cli import main

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "collection"


def run_cli(argv):
    return main(argv)


class TestCli:
    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "discover" in capsys.readouterr().out

    def test_discover_source(self, tmp_path, capsys):
        code = run_cli(
            [
                "--config-dir", str(CONFIG_DIR),
                "--data-dir", str(tmp_path / "collection"),
                "--fixtures-dir", str(FIXTURES_DIR),
                "discover", "--source", "anduril_official",
            ]
        )
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["metrics"]["discovered"] == 3
        assert report["indexed"] is False
        discovery_log = tmp_path / "collection" / "discovery" / "anduril_official.jsonl"
        assert discovery_log.exists()

    def test_run_group_p0(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)
        code = run_cli(
            [
                "--config-dir", str(CONFIG_DIR),
                "--data-dir", str(tmp_path / "collection"),
                "--fixtures-dir", str(FIXTURES_DIR),
                "run", "--group", "p0", "--no-index",
            ]
        )
        assert code == 0
        reports = json.loads(capsys.readouterr().out)
        assert isinstance(reports, list) and len(reports) == 4
        by_source = {r["source_ids"][0]: r for r in reports}
        assert by_source["usaspending"]["metrics"]["published_clean"] == 2
        assert by_source["patentsview"]["metrics"]["policy_skipped"] == 1
        assert all(r["indexed"] is False for r in reports)

    def test_run_unknown_source_errors(self, tmp_path):
        from rag_graph_demo.collection.registry import RegistryError

        with pytest.raises(RegistryError):
            run_cli(
                [
                    "--config-dir", str(CONFIG_DIR),
                    "--data-dir", str(tmp_path / "collection"),
                    "--fixtures-dir", str(FIXTURES_DIR),
                    "run", "--source", "does_not_exist",
                ]
            )

    def test_retry_without_retryable_failures(self, tmp_path, capsys):
        run_cli(
            [
                "--config-dir", str(CONFIG_DIR),
                "--data-dir", str(tmp_path / "collection"),
                "--fixtures-dir", str(FIXTURES_DIR),
                "run", "--source", "usaspending",
            ]
        )
        report = json.loads(capsys.readouterr().out)
        code = run_cli(
            [
                "--config-dir", str(CONFIG_DIR),
                "--data-dir", str(tmp_path / "collection"),
                "--fixtures-dir", str(FIXTURES_DIR),
                "retry", "--run-id", report["run_id"],
            ]
        )
        assert code == 0
        retry_report = json.loads(capsys.readouterr().out)
        assert retry_report["retried"] == 0
