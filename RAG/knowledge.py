"""Extract, chunk, embed, and retrieve uploaded knowledge."""

import hashlib
import os
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from Models.postgresql import existing_document, save_document, search_chunks
from Models.redis import invalidate_tool_results
from RAG.categories import normalize_category


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_CHUNKS = 200
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_FORMATS_ERROR = "Only PDF, DOCX, TXT, and Markdown files are supported"
EMBEDDING_MODEL = "text-embedding-3-small"


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=1536)


def _docx_text(data: bytes) -> str:
    """Keep paragraphs and table rows in their original document order."""
    try:
        document = Document(BytesIO(data))
        lines = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    lines.append(text)
            else:
                for row in block.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
        return "\n".join(lines)
    except Exception as exc:
        raise ValueError("Could not read the DOCX") from exc


def extract_chunks(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(SUPPORTED_FORMATS_ERROR)
    if not data:
        raise ValueError("The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds 5 MB")

    if suffix == ".pdf":
        try:
            pages = [
                (number, page.extract_text() or "")
                for number, page in enumerate(PdfReader(BytesIO(data)).pages, start=1)
            ]
        except Exception as exc:
            raise ValueError("Could not read the PDF") from exc
    elif suffix == ".docx":
        pages = [(None, _docx_text(data))]
    else:
        try:
            pages = [(None, data.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must be UTF-8") from exc

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [
        (page_number, chunk)
        for page_number, text in pages
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]
    if not chunks:
        raise ValueError("No extractable text found; scanned PDFs need OCR")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError("The file contains too much text (maximum 200 chunks)")
    return chunks


def ingest_file(filename: str, data: bytes, category: str = "general") -> dict:
    category = normalize_category(category)
    safe_name = Path(filename).name or "upload"
    chunks = extract_chunks(safe_name, data)
    sha256 = hashlib.sha256(data).hexdigest()
    if previous := existing_document(sha256, category):
        return {
            "document_id": str(previous["id"]),
            "filename": previous["filename"],
            "chunks": previous["chunks"],
            "duplicate": True,
        }

    vectors = _embeddings().embed_documents([content for _, content in chunks])
    saved = save_document(sha256, category, safe_name, chunks, vectors)
    if not saved["duplicate"]:
        invalidate_tool_results(os.getenv("REDIS_URL"), "search_uploaded_knowledge")
    return {**saved, "chunks": len(chunks)}


def search_knowledge(query: str, limit: int = 4, category: str | None = None) -> list[dict]:
    """Search uploaded knowledge by meaning and return matching source passages."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10")
    if category is not None:
        category = normalize_category(category)
    vector = _embeddings().embed_query(query)
    return search_chunks(vector, limit, category)
