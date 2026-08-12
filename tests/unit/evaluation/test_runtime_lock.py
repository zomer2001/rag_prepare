from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_graph_demo.evaluation.runtime_lock import (
    EvaluationAlreadyRunningError,
    EvaluationRuntimeLock,
)


def test_runtime_lock_rejects_second_process_owner(tmp_path: Path) -> None:
    first = EvaluationRuntimeLock(tmp_path)
    second = EvaluationRuntimeLock(tmp_path)

    with first:
        owner = json.loads((tmp_path / ".evaluation.lock").read_text(encoding="utf-8"))
        assert isinstance(owner["pid"], int)
        with pytest.raises(EvaluationAlreadyRunningError, match="already in use"):
            second.acquire()

    with second:
        assert second.path == tmp_path / ".evaluation.lock"
