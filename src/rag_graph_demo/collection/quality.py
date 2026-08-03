"""Hard validation and quality scoring (brief §4.7).

Validation failures are never masked by quality scores; low-quality but valid
records go to the review queue instead of being dropped.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, NormalizedRecord

MIN_BODY_CHARS = 200
REVIEW_COMPLETENESS_THRESHOLD = 0.7


def hard_validate(record: dict[str, Any], *, check_raw_file: bool = True) -> list[str]:
    """Return a list of hard validation errors (empty list = valid)."""
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: {record.get('schema_version')!r}")
    if not record.get("record_id"):
        errors.append("missing record_id")
    if not record.get("version_id"):
        errors.append("missing version_id")
    title = (record.get("title") or "").strip()
    if not title:
        errors.append("empty title")
    url = record.get("canonical_url")
    if url is not None:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            errors.append(f"invalid canonical_url: {url!r}")
    content = record.get("content") or {}
    raw_path = content.get("raw_path")
    raw_sha = content.get("raw_sha256")
    if not raw_path:
        errors.append("missing content.raw_path")
    elif check_raw_file:
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"raw file missing: {raw_path}")
        elif raw_sha:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != raw_sha:
                errors.append("raw file hash mismatch")
    try:
        body = record.get("body_markdown") or ""
        body.encode("utf-8")
    except UnicodeError:
        errors.append("body is not valid UTF-8")
    # pydantic re-validation of the full model
    try:
        NormalizedRecord.model_validate(record)
    except Exception as exc:
        errors.append(f"model validation failed: {exc}")
    return errors


def score_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Compute parse_confidence/completeness/needs_review/warnings."""
    warnings: list[str] = list((record.get("quality") or {}).get("warnings", []))
    body = record.get("body_markdown") or ""
    completeness = 1.0
    if len(body.strip()) < MIN_BODY_CHARS:
        completeness -= 0.4
        warnings.append(f"short body: {len(body.strip())} chars")
    if not record.get("published_at"):
        completeness -= 0.1
        warnings.append("missing published_at")
    if not record.get("canonical_url") and not (record.get("source") or {}).get(
        "provider_record_id"
    ):
        completeness -= 0.2
        warnings.append("no canonical_url and no provider_record_id")
    completeness = max(0.0, round(completeness, 2))

    parse_confidence = float((record.get("quality") or {}).get("parse_confidence", 1.0))
    needs_review = bool((record.get("quality") or {}).get("needs_review", False))
    if completeness < REVIEW_COMPLETENESS_THRESHOLD or parse_confidence < 0.5:
        needs_review = True
    return {
        "parse_confidence": parse_confidence,
        "completeness": completeness,
        "needs_review": needs_review,
        "warnings": warnings,
    }


def write_review_entry(
    review_root: Path | str,
    record: dict[str, Any],
    reasons: list[str],
) -> Path:
    """Append a review-queue entry; records are queued, never discarded."""
    import json
    from datetime import UTC, datetime

    folder = Path(review_root)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"{stamp}_{record.get('record_id', 'unknown').replace('/', '_')}.json"
    payload = {"record_id": record.get("record_id"), "reasons": reasons, "record": record}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path
