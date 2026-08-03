"""Run scheduling primitives: atomic locks, per-source state, manifests.

Brief §7.2: one active run per source+cursor window; raw objects append-only;
cursor advances only after the batch succeeds past its threshold; manifests
record code version, config hash and adapter versions.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import RunManifest, SourceState

COLLECTION_SUBDIRS = (
    "state",
    "discovery",
    "raw",
    "normalized",
    "clean",
    "observations",
    "review_queue",
    "dead_letter",
    "runs",
)


def ensure_collection_dirs(root: Path | str) -> Path:
    root = Path(root)
    for name in COLLECTION_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


class RunLockError(RuntimeError):
    pass


class RunLock:
    """Atomic lock via O_EXCL file creation with a lease timestamp."""

    def __init__(self, state_root: Path | str, source_id: str, lease_seconds: int = 3600) -> None:
        self.path = Path(state_root) / f"{source_id}.lock"
        self.lease_seconds = lease_seconds
        self._acquired = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                taken_at = datetime.fromisoformat(payload["taken_at"])
                age = (datetime.now(UTC) - taken_at).total_seconds()
            except (ValueError, KeyError, json.JSONDecodeError):
                age = self.lease_seconds + 1
            if age < self.lease_seconds:
                raise RunLockError(f"active run lock held for {self.path.stem}")
            self.path.unlink(missing_ok=True)  # expired lease
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "taken_at": datetime.now(UTC).isoformat(),
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunLockError(f"active run lock held for {self.path.stem}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        self._acquired = True

    def release(self) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class StateStore:
    """Per-source cursor/validator/health persistence."""

    def __init__(self, state_root: Path | str) -> None:
        self.root = Path(state_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_id: str) -> Path:
        return self.root / f"{source_id}.json"

    def load(self, source_id: str) -> SourceState:
        path = self.path_for(source_id)
        if not path.exists():
            return SourceState(source_id=source_id)
        return SourceState.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, state: SourceState) -> None:
        state.updated_at = datetime.now(UTC)
        path = self.path_for(state.source_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def write_manifest(runs_root: Path | str, manifest: RunManifest) -> Path:
    folder = Path(runs_root) / manifest.run_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    errors_path = folder / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as fh:
        for error in manifest.errors:
            fh.write(json.dumps(error.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def load_manifest(path: Path | str) -> RunManifest:
    return RunManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def discovery_log_path(discovery_root: Path | str, source_id: str) -> Path:
    return Path(discovery_root) / f"{source_id}.jsonl"


def append_discovery(discovery_root: Path | str, source_id: str, candidates: list[dict]) -> Path:
    path = discovery_log_path(discovery_root, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for candidate in candidates:
            fh.write(json.dumps(candidate, ensure_ascii=False, default=str) + "\n")
    return path


def write_dead_letter(dead_root: Path | str, entry: dict[str, Any]) -> Path:
    folder = Path(dead_root)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"{stamp}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
