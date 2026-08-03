"""Source registry: loads config/*.yaml into typed SourceConfig objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .models import SourceConfig

DEFAULT_CONFIG_DIR = Path("config")


class RegistryError(RuntimeError):
    pass


class SourceRegistry:
    def __init__(self, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir)
        self.sources: dict[str, SourceConfig] = {}
        self.entities: dict[str, Any] = {}
        self.queries: dict[str, Any] = {}
        self.retention: dict[str, Any] = {}
        self._load()

    def _load_yaml(self, name: str, required: bool = True) -> dict[str, Any]:
        path = self.config_dir / name
        if not path.exists():
            if required:
                raise RegistryError(f"missing config file: {path}")
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise RegistryError(f"config file {path} must contain a mapping")
        return data

    def _load(self) -> None:
        raw_sources = self._load_yaml("sources.yaml")
        for entry in raw_sources.get("sources", []):
            cfg = SourceConfig.model_validate(entry)
            if cfg.source_id in self.sources:
                raise RegistryError(f"duplicate source_id: {cfg.source_id}")
            self.sources[cfg.source_id] = cfg
        self.entities = self._load_yaml("entities.yaml", required=False)
        self.queries = self._load_yaml("queries.yaml", required=False)
        self.retention = self._load_yaml("retention.yaml", required=False)

    def get(self, source_id: str) -> SourceConfig:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise RegistryError(f"unknown source_id: {source_id}") from exc

    def enabled(self) -> list[SourceConfig]:
        return [s for s in self.sources.values() if s.enabled]

    def by_group(self, group: str) -> list[SourceConfig]:
        return [s for s in self.enabled() if group in s.groups]

    def config_hash(self) -> str:
        """Stable hash over all config files, recorded in every run manifest."""
        digest = hashlib.sha256()
        for name in ("sources.yaml", "entities.yaml", "queries.yaml", "retention.yaml"):
            path = self.config_dir / name
            if path.exists():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def entity_aliases(self, kind: str) -> list[str]:
        """e.g. entity_aliases('company_aliases') from entities.yaml."""
        value = self.entities.get(kind, [])
        return [str(v) for v in value] if isinstance(value, list) else []

    def product_seeds(self) -> list[str]:
        return self.entity_aliases("product_seeds")

    def queries_for(self, source_id: str) -> list[str]:
        for block in self.queries.get("matrix", []):
            if isinstance(block, dict) and block.get("source") == source_id:
                return [str(q) for q in block.get("queries", [])]
        return []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SourceRegistry(sources={sorted(self.sources)})"


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)
