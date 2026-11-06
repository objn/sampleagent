"""Persist actual user and agent chat messages for customer history lookup."""

from threading import Lock
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from Models.postgresql import database_uri


_schema_ready = False
_schema_lock = Lock()


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with psycopg.connect(database_uri()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'agent')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (user_id, session_id, message_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS chat_history_user_recent_idx
                ON chat_history (user_id, created_at DESC, id DESC)
            """)
        _schema_ready = True


def _public_message(row: dict) -> dict:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "message_id": row["message_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"].isoformat(),
    }


def save_chat_message(
    *, user_id: str, session_id: str, role: str, content: str,
    message_id: str | None = None,
) -> dict:
    """Store a real chat event; a repeated message ID is idempotent."""
    if not user_id.strip() or not session_id.strip() or not content.strip():
        raise ValueError("user_id, session_id, and content must be nonempty")
    if role not in {"user", "agent"}:
        raise ValueError("role must be 'user' or 'agent'")
    if message_id is not None and not message_id.strip():
        raise ValueError("message_id must be nonempty when supplied")
    message_id = message_id or str(uuid4())
    _ensure_schema()
    with psycopg.connect(database_uri(), row_factory=dict_row) as conn:
        row = conn.execute("""
            INSERT INTO chat_history (user_id, session_id, message_id, role, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, session_id, message_id) DO NOTHING
            RETURNING *
        """, (user_id, session_id, message_id, role, content)).fetchone()
        if row is None:
            row = conn.execute("""
                SELECT * FROM chat_history
                WHERE user_id = %s AND session_id = %s AND message_id = %s
            """, (user_id, session_id, message_id)).fetchone()
    return _public_message(row)


def recent_customer_messages(
    user_id: str, *, limit: int = 10,
    exclude_session_id: str | None = None,
    exclude_message_id: str | None = None,
) -> list[dict]:
    """Return the latest messages across sessions in chronological order."""
    if not user_id.strip():
        raise ValueError("user_id must be nonempty")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    if (exclude_session_id is None) != (exclude_message_id is None):
        raise ValueError("Both exclusion identifiers must be supplied together")
    _ensure_schema()
    with psycopg.connect(database_uri(), row_factory=dict_row) as conn:
        rows = conn.execute("""
            SELECT id, user_id, session_id, message_id, role, content, created_at
            FROM chat_history
            WHERE user_id = %s
              AND (%s::text IS NULL OR session_id <> %s OR message_id <> %s)
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """, (
            user_id, exclude_message_id, exclude_session_id,
            exclude_message_id, limit,
        )).fetchall()
    return [_public_message(row) for row in reversed(rows)]
