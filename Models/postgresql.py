"""PostgreSQL storage for uploaded RAG documents and vector search."""

import os
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from uuid import UUID, uuid4

import psycopg
from dotenv import dotenv_values
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


EMBEDDING_DIMENSIONS = 1536
_upload_schema_ready = False
_upload_schema_lock = Lock()


def _file_database_settings() -> dict:
    """Read local development credentials even if the dev server masks env vars."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    return dotenv_values(env_file) if env_file.is_file() else {}


def database_uri() -> str:
    # langgraph dev can set DATABASE_URI=memory: for its own checkpointer.
    # That value is not a PostgreSQL connection string for application data.
    for name in ("APP_DATABASE_URI", "DATABASE_URI"):
        uri = os.getenv(name)
        if uri and uri.startswith(("postgresql://", "postgres://")):
            return uri
    file_settings = _file_database_settings()
    for name in ("APP_DATABASE_URI", "DATABASE_URI"):
        uri = file_settings.get(name)
        if uri and uri.startswith(("postgresql://", "postgres://")):
            return uri
    password = os.getenv("POSTGRES_PASSWORD") or file_settings.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "Set APP_DATABASE_URI to a PostgreSQL URL or set POSTGRES_PASSWORD "
            "in the process or project .env for the local PostgreSQL service"
        )
    return (
        "postgresql://sampleagent:"
        f"{quote(password, safe='')}@127.0.0.1:2125/sampleagent"
    )


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(database_uri(), row_factory=dict_row)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY,
            sha256 TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            filename TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute("ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'general'")
    conn.execute("ALTER TABLE rag_documents DROP CONSTRAINT IF EXISTS rag_documents_sha256_key")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS rag_documents_sha256_category_idx ON rag_documents (sha256, category)")
    conn.execute("CREATE INDEX IF NOT EXISTS rag_documents_category_idx ON rag_documents (category)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id BIGSERIAL PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            content TEXT NOT NULL,
            embedding vector(1536) NOT NULL,
            UNIQUE (document_id, chunk_index)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
        ON rag_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    conn.commit()
    return conn


def _upload_connect() -> psycopg.Connection:
    global _upload_schema_ready
    conn = psycopg.connect(database_uri(), row_factory=dict_row)
    try:
        if not _upload_schema_ready:
            with _upload_schema_lock:
                if not _upload_schema_ready:
                    with _connect():
                        pass
                    conn.execute("""
                        DO $$ BEGIN
                            IF to_regclass('public.rag_files') IS NULL
                               AND to_regclass('public.rag_uploads') IS NOT NULL THEN
                                ALTER TABLE rag_uploads RENAME TO rag_files;
                            END IF;
                        END $$
                    """)
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS rag_files (
                            id UUID PRIMARY KEY,
                            stored_name TEXT NOT NULL UNIQUE,
                            original_name TEXT NOT NULL,
                            content_type TEXT,
                            size_bytes BIGINT NOT NULL,
                            sha256 TEXT NOT NULL,
                            category TEXT NOT NULL DEFAULT 'general',
                            status TEXT NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                            document_id UUID,
                            chunk_count INTEGER,
                            duplicate BOOLEAN,
                            error TEXT,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            started_at TIMESTAMPTZ,
                            finished_at TIMESTAMPTZ
                        )
                        """
                    )
                    conn.execute("ALTER TABLE rag_files ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'general'")
                    conn.execute("""
                        DO $$ BEGIN
                            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'rag_files_document_id_fkey') THEN
                                ALTER TABLE rag_files ADD CONSTRAINT rag_files_document_id_fkey
                                FOREIGN KEY (document_id) REFERENCES rag_documents(id);
                            END IF;
                        END $$
                    """)
                    conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS rag_files_queue_idx
                        ON rag_files (status, created_at)
                        """
                    )
                    conn.commit()
                    _upload_schema_ready = True
    except Exception:
        conn.close()
        raise
    return conn


def create_upload(
    upload_id: UUID,
    stored_name: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    sha256: str,
    category: str,
) -> dict:
    with _upload_connect() as conn:
        row = conn.execute(
            """
            INSERT INTO rag_files
                (id, stored_name, original_name, content_type, size_bytes, sha256, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (upload_id, stored_name, original_name, content_type, size_bytes, sha256, category),
        ).fetchone()
    return _public_upload(row)


def _public_upload(row: dict) -> dict:
    return {
        "upload_id": str(row["id"]),
        "stored_name": row["stored_name"],
        "original_name": row["original_name"],
        "content_type": row["content_type"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "category": row["category"],
        "status": row["status"],
        "document_id": str(row["document_id"]) if row["document_id"] else None,
        "chunks": row["chunk_count"],
        "duplicate": row["duplicate"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat(),
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


def get_upload(upload_id: UUID) -> dict | None:
    with _upload_connect() as conn:
        row = conn.execute(
            "SELECT * FROM rag_files WHERE id = %s", (upload_id,)
        ).fetchone()
    return _public_upload(row) if row else None


def claim_upload() -> dict | None:
    with _upload_connect() as conn:
        conn.execute(
            """
            UPDATE rag_files
            SET status = 'failed', error = 'Worker stopped repeatedly', finished_at = now()
            WHERE status = 'processing'
              AND started_at < now() - interval '1 hour'
              AND attempts >= 3
            """
        )
        row = conn.execute(
            """
            WITH next_upload AS (
                SELECT id FROM rag_files
                WHERE status = 'pending'
                   OR (status = 'processing'
                       AND started_at < now() - interval '1 hour'
                       AND attempts < 3)
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE rag_files AS u
            SET status = 'processing', started_at = now(),
                finished_at = NULL, error = NULL, attempts = attempts + 1
            FROM next_upload
            WHERE u.id = next_upload.id
            RETURNING u.id, u.stored_name, u.original_name, u.sha256, u.category
            """
        ).fetchone()
    return dict(row) if row else None


def complete_upload(upload_id: UUID, result: dict) -> None:
    with _upload_connect() as conn:
        conn.execute(
            """
            UPDATE rag_files
            SET status = 'completed', document_id = %s, chunk_count = %s,
                duplicate = %s, error = NULL, finished_at = now()
            WHERE id = %s AND status = 'processing'
            """,
            (UUID(result["document_id"]), result["chunks"], result["duplicate"], upload_id),
        )


def fail_upload(upload_id: UUID, message: str) -> None:
    with _upload_connect() as conn:
        conn.execute(
            """
            UPDATE rag_files
            SET status = 'failed', error = %s, finished_at = now()
            WHERE id = %s AND status = 'processing'
            """,
            (message[:500], upload_id),
        )


def existing_document(sha256: str, category: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT d.id, d.filename, count(c.id)::integer AS chunks
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.sha256 = %s AND d.category = %s
            GROUP BY d.id
            """,
            (sha256, category),
        ).fetchone()
    return dict(row) if row else None


def save_document(
    sha256: str,
    category: str,
    filename: str,
    chunks: list[tuple[int | None, str]],
    vectors: list[list[float]],
) -> dict:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have an embedding")
    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
        raise ValueError("Embedding dimension must be 1536")

    document_id = uuid4()
    with _connect() as conn:
        inserted = conn.execute(
            """
            INSERT INTO rag_documents (id, sha256, category, filename)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (sha256, category) DO NOTHING
            RETURNING id
            """,
            (document_id, sha256, category, filename),
        ).fetchone()
        if inserted is None:
            row = conn.execute(
                "SELECT id, filename FROM rag_documents WHERE sha256 = %s AND category = %s", (sha256, category)
            ).fetchone()
            return {"document_id": str(row["id"]), "filename": row["filename"], "duplicate": True}

        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO rag_chunks
                    (document_id, chunk_index, page_number, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (document_id, index, page, content, Vector(vector))
                    for index, ((page, content), vector) in enumerate(zip(chunks, vectors))
                ],
            )
    return {"document_id": str(document_id), "filename": filename, "duplicate": False}


def search_chunks(vector: list[float], limit: int = 4, category: str | None = None) -> list[dict]:
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError("Embedding dimension must be 1536")
    embedding = Vector(vector)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id AS document_id, d.filename, d.category, c.page_number,
                   c.content, 1 - (c.embedding <=> %s) AS score
            FROM rag_chunks c
            JOIN rag_documents d ON d.id = c.document_id
            WHERE (%s::text IS NULL OR d.category = %s)
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (embedding, category, category, embedding, limit),
        ).fetchall()
    return [
        {**row, "document_id": str(row["document_id"]), "score": float(row["score"])}
        for row in rows
    ]
