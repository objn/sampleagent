import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from Models.chat_history import recent_customer_messages, save_chat_message


class FakeCursor:
    def __init__(self, *, row=None, rows=None):
        self.row = row
        self.rows = rows or []

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursors):
        self.cursors = iter(cursors)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.calls.append((query, params))
        return next(self.cursors)


def row(number, content):
    return {
        "id": number, "user_id": "user-1", "session_id": "session-old",
        "message_id": f"message-{number}", "role": "agent" if number % 2 else "user",
        "content": content, "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


class ChatHistoryTests(unittest.TestCase):
    def test_recent_messages_are_limited_and_returned_oldest_first(self):
        connection = FakeConnection([FakeCursor(rows=[row(3, "new"), row(2, "old")])])
        with patch("Models.chat_history._ensure_schema"), \
             patch("Models.chat_history.database_uri", return_value="database-url"), \
             patch("Models.chat_history.psycopg.connect", return_value=connection):
            messages = recent_customer_messages(
                "user-1", exclude_session_id="session-current",
                exclude_message_id="current-1",
            )
        self.assertEqual([m["content"] for m in messages], ["old", "new"])
        self.assertEqual(
            connection.calls[0][1],
            ("user-1", "current-1", "session-current", "current-1", 10),
        )
        self.assertIn("ORDER BY created_at DESC, id DESC", connection.calls[0][0])

    def test_repeated_message_id_returns_existing_row(self):
        existing = row(7, "Original text")
        connection = FakeConnection([FakeCursor(row=None), FakeCursor(row=existing)])
        with patch("Models.chat_history._ensure_schema"), \
             patch("Models.chat_history.database_uri", return_value="database-url"), \
             patch("Models.chat_history.psycopg.connect", return_value=connection):
            saved = save_chat_message(
                user_id="user-1", session_id="session-old", role="agent",
                content="Original text", message_id="message-7",
            )
        self.assertEqual(saved["id"], 7)
        self.assertIn("ON CONFLICT", connection.calls[0][0])
        self.assertEqual(len(connection.calls), 2)


if __name__ == "__main__":
    unittest.main()
