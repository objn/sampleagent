"""Persist original uploads before scheduling RAG processing."""

import hashlib
import os
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from Models.postgresql import create_upload
from RAG.categories import normalize_category
from RAG.knowledge import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, SUPPORTED_FORMATS_ERROR


def upload_dir() -> Path:
    directory = Path(os.getenv("RAG_UPLOAD_DIR", "uploads")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def queue_upload(filename: str, content_type: str | None, data: bytes, category: str = "general") -> dict:
    category = normalize_category(category)
    original_name = PureWindowsPath(filename).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(SUPPORTED_FORMATS_ERROR)
    if not data:
        raise ValueError("The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds 5 MB")

    upload_id = uuid4()
    stored_name = f"{uuid4().hex}{suffix}"
    path = upload_dir() / stored_name
    with path.open("xb") as handle:
        handle.write(data)
    try:
        record = create_upload(
            upload_id=upload_id,
            stored_name=stored_name,
            original_name=original_name,
            content_type=content_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            category=category,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {**record, "status_url": f"/rag/uploads/{upload_id}"}


def read_queued_upload(stored_name: str, sha256: str) -> bytes:
    if Path(stored_name).name != stored_name or "\\" in stored_name:
        raise ValueError("Invalid stored filename")
    data = (upload_dir() / stored_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Stored file checksum does not match")
    return data
