from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)

