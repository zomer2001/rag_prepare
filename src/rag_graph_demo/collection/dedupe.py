"""Deduplication, stable record IDs, content versioning (brief §4.6).

Order of identity: provider ID → canonical URL → raw SHA-256 → normalized
text hash → SimHash near-duplicate candidates. Near-duplicates only create
relations; nothing is deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Reversible-by-construction normalization for hashing only: Unicode NFC,
    whitespace collapse, strip. The stored body itself is NOT rewritten here."""
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text).strip()


def normalized_hash(text: str) -> str:
    return sha256_text(normalize_text(text))


def make_record_id(
    source_id: str, provider_record_id: str | None, canonical_url: str | None
) -> str:
    """Stable record ID: prefer provider ID, fall back to canonical URL hash."""
    if provider_record_id:
        key = f"provider:{provider_record_id}"
    elif canonical_url:
        key = f"url:{canonical_url}"
    else:
        raise ValueError("record identity requires provider_record_id or canonical_url")
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{source_id}:{digest}"


# ---------------------------------------------------------------------------
# SimHash near-duplicate detection
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text).lower())


def simhash64(text: str) -> int:
    bits = [0] * 64
    for token in _tokens(text):
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")  # noqa: S324
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    value = 0
    for i, score in enumerate(bits):
        if score > 0:
            value |= 1 << i
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimHashIndex:
    """Append-only JSONL index of version_id → simhash for near-dup linking."""

    def __init__(self, path: Path | str, threshold: int = 6) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self._entries: list[tuple[str, str, int]] = []  # record_id, version_id, hash
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._entries.append((row["record_id"], row["version_id"], int(row["simhash"])))

    def candidates(
        self, text: str, exclude_record_id: str | None = None
    ) -> list[tuple[str, str, int]]:
        h = simhash64(text)
        found = []
        for record_id, version_id, existing in self._entries:
            if exclude_record_id and record_id == exclude_record_id:
                continue
            dist = hamming_distance(h, existing)
            if dist <= self.threshold:
                found.append((record_id, version_id, dist))
        found.sort(key=lambda item: item[2])
        return found

    def add(self, record_id: str, version_id: str, text: str) -> int:
        h = simhash64(text)
        self._entries.append((record_id, version_id, h))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"record_id": record_id, "version_id": version_id, "simhash": h},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return h


# ---------------------------------------------------------------------------
# Version store
# ---------------------------------------------------------------------------


class RecordStore:
    """Tracks known records and their versions under data/collection/normalized.

    Layout: normalized/<record_id>/v<index>.json plus normalized/<record_id>/latest.json
    Version rule (brief §4.6): identical normalized hash → same version (no new
    record); changed hash → new version, record_id unchanged, first_seen kept.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def record_dir(self, record_id: str) -> Path:
        safe = record_id.replace("/", "_")
        return self.root / safe

    def latest_path(self, record_id: str) -> Path:
        return self.record_dir(record_id) / "latest.json"

    def load_latest(self, record_id: str) -> dict | None:
        path = self.latest_path(record_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_version(self, record: dict) -> tuple[bool, bool]:
        """Persist one normalized record dict.
        Returns (is_new_record, is_new_version)."""
        record_id = record["record_id"]
        folder = self.record_dir(record_id)
        folder.mkdir(parents=True, exist_ok=True)
        latest = self.load_latest(record_id)
        if latest is None:
            record["version_index"] = 1
            is_new_record, is_new_version = True, True
        elif latest["version_id"] == record["version_id"]:
            # Same content observed again: refresh last_seen only, no new version.
            latest["last_seen_at"] = record["last_seen_at"]
            self._atomic_write(self.latest_path(record_id), latest)
            return False, False
        else:
            record["version_index"] = int(latest.get("version_index", 1)) + 1
            record["first_seen_at"] = latest.get("first_seen_at", record["first_seen_at"])
            is_new_record, is_new_version = False, True

        version_path = folder / f"v{record['version_index']}.json"
        self._atomic_write(version_path, record)
        self._atomic_write(self.latest_path(record_id), record)
        return is_new_record, is_new_version

    def update_latest(self, record: dict) -> None:
        """Rewrite the stored latest/vN files in place (same version_id), used
        to attach derived paths (e.g. clean_path) after initial save."""
        folder = self.record_dir(record["record_id"])
        folder.mkdir(parents=True, exist_ok=True)
        self._atomic_write(folder / f"v{record['version_index']}.json", record)
        self._atomic_write(self.latest_path(record["record_id"]), record)

    @staticmethod
    def _atomic_write(path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        tmp.replace(path)
