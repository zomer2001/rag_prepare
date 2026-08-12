from __future__ import annotations

import fcntl
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import IO


class EvaluationAlreadyRunningError(RuntimeError):
    """Raised when another process is using the same evaluation runtime."""


class EvaluationRuntimeLock:
    """Process-scoped advisory lock for one evaluation runtime directory."""

    def __init__(self, runtime_dir: Path | str) -> None:
        self.path = Path(runtime_dir) / ".evaluation.lock"
        self._handle: IO[str] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise EvaluationAlreadyRunningError(
                f"Evaluation runtime is already in use: {self.path.parent} "
                f"({owner}). Stop that process or choose another --runtime-dir."
            ) from exc

        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> EvaluationRuntimeLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
