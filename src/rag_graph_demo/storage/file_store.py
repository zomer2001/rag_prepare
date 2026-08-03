from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from rag_graph_demo.domain.exceptions import UnsupportedDocumentError

ALLOWED_SUFFIXES = {".md", ".markdown", ".docx", ".pdf"}


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^\w.\- ()\u4e00-\u9fff]", "_", name, flags=re.UNICODE)
    return cleaned[:180] or "document"


class FileStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def validate_suffix(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UnsupportedDocumentError(
                f"Unsupported document type: {suffix or 'missing extension'}"
            )
        return suffix

    def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        suffix = self.validate_suffix(filename)
        directory = self.data_dir / "raw" / document_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"original{suffix}"
        path.write_bytes(content)
        return path.resolve()

    def parsed_paths(self, document_id: str) -> tuple[Path, Path]:
        directory = self.data_dir / "parsed" / document_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "document.md", directory / "metadata.json"

    def delete_document_files(self, document_id: str) -> None:
        for group in ("raw", "parsed"):
            path = self.data_dir / group / document_id
            if path.exists():
                shutil.rmtree(path)

    @staticmethod
    def checksum(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

