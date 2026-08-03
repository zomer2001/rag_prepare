"""Normalization helpers: dates, money, clean Markdown + JSON sidecar.

Rules (brief §4.5):
- Dates are ISO 8601 with an explicit precision; a year-only source string
  must NOT be inflated into a fake month/day.
- Money is a decimal string + currency + exact source field name.
- Body Markdown carries prose only; business metadata lives in the sidecar.
- Clean files must NOT contain the existing project's `<!-- rag-source ... -->`
  marker — that marker is reserved for a future explicit promotion step.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

RAG_SOURCE_MARKER = "<!-- rag-source"

_DATE_FULL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_DATE_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DATE_YEAR = re.compile(r"^(\d{4})$")
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_LONG = re.compile(
    r"(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    r"|(?P<day2>\d{1,2})\s+(?P<month2>[A-Za-z]{3,9})\.?,?\s+(?P<year2>\d{4})"
)


def parse_date(raw: str | None) -> tuple[str | None, str]:
    """Parse a date string. Returns (iso_value, confidence) where confidence
    is one of full/month/year/none. Never fabricates missing components."""
    if not raw:
        return None, "none"
    text = raw.strip()
    if not text:
        return None, "none"
    match = _DATE_FULL.match(text)
    if match:
        year, month, day = match.groups()
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{month}-{day}", "full"
    match = _DATE_MONTH.match(text)
    if match and 1 <= int(match.group(2)) <= 12:
        return f"{match.group(1)}-{match.group(2)}", "month"
    match = _DATE_YEAR.match(text)
    if match:
        return match.group(1), "year"
    match = _DATE_LONG.search(text)
    if match:
        month_name = (match.group("month") or match.group("month2") or "")[:3].lower()
        month = _MONTHS.get(month_name)
        day = match.group("day") or match.group("day2")
        year = match.group("year") or match.group("year2")
        if month and day and year and 1 <= int(day) <= 31:
            return f"{year}-{month:02d}-{int(day):02d}", "full"
    return None, "none"


def money_string(value: Any) -> str | None:
    """Render a provider amount as an exact decimal string, or None."""
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
        if not decimal.is_finite():
            return None
        # normalize() drops insignificant trailing zeros; value stays exact.
        return format(decimal.normalize(), "f")
    except (InvalidOperation, ValueError):
        return None


def build_clean_markdown(record: dict[str, Any]) -> str:
    """RAG-ready Markdown with a small human-readable header. Deliberately no
    rag-source marker and no LightRAG metadata (brief §5)."""
    lines = [
        f"# {record['title']}",
        "",
        f"- source: {record['source']['publisher']} ({record['source']['source_id']})",
        f"- record_id: {record['record_id']}",
        f"- version_id: {record['version_id']}",
    ]
    if record.get("canonical_url"):
        lines.append(f"- canonical_url: {record['canonical_url']}")
    if record.get("published_at"):
        lines.append(f"- published_at: {record['published_at']}")
    lines.append(f"- fetched_at: {record['fetched_at']}")
    lines.append("")
    body = record.get("body_markdown") or ""
    lines.append(body.strip())
    text = "\n".join(lines).rstrip() + "\n"
    if RAG_SOURCE_MARKER in text:
        raise ValueError("clean markdown must not contain a rag-source marker")
    return text


def write_clean_and_sidecar(
    clean_root: Path | str,
    record: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write clean/<record_id>/v<index>.md + .sidecar.json.
    Returns (clean_path, sidecar_path)."""
    folder = Path(clean_root) / record["record_id"].replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    version = record["version_index"]
    clean_path = folder / f"v{version}.md"
    sidecar_path = folder / f"v{version}.sidecar.json"

    markdown = build_clean_markdown(record)
    _atomic_write_text(clean_path, markdown)

    sidecar = {k: v for k, v in record.items() if k != "body_markdown"}
    sidecar["clean_body_sha256"] = _sha(markdown)
    _atomic_write_text(
        sidecar_path, json.dumps(sidecar, ensure_ascii=False, indent=2, default=str)
    )
    return clean_path, sidecar_path


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
