CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, session_id, message_id)
);

CREATE INDEX IF NOT EXISTS chat_history_user_recent_idx
ON chat_history (user_id, created_at DESC, id DESC);
