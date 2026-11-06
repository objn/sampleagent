import os
import unittest
from unittest.mock import patch

from Models.postgresql import database_uri


class DatabaseUriTests(unittest.TestCase):
    def test_langgraph_memory_uri_uses_local_postgres(self):
        with patch.dict(os.environ, {
            "DATABASE_URI": "memory:",
            "POSTGRES_PASSWORD": "test-password",
        }, clear=True):
            uri = database_uri()
        self.assertEqual(
            uri,
            "postgresql://sampleagent:test-password@127.0.0.1:2125/sampleagent",
        )

    def test_explicit_app_uri_has_priority(self):
        with patch.dict(os.environ, {
            "APP_DATABASE_URI": "postgresql://app:secret@db:5432/app",
            "DATABASE_URI": "memory:",
            "POSTGRES_PASSWORD": "test-password",
        }, clear=True):
            self.assertEqual(
                database_uri(), "postgresql://app:secret@db:5432/app"
            )

    def test_memory_uri_reads_project_dotenv_password(self):
        with patch.dict(os.environ, {"DATABASE_URI": "memory:"}, clear=True), \
             patch("Models.postgresql._file_database_settings", return_value={
                 "POSTGRES_PASSWORD": "file-password",
             }):
            uri = database_uri()
        self.assertEqual(
            uri,
            "postgresql://sampleagent:file-password@127.0.0.1:2125/sampleagent",
        )

    def test_missing_postgres_config_has_clear_error(self):
        with patch.dict(os.environ, {"DATABASE_URI": "memory:"}, clear=True), \
             patch("Models.postgresql._file_database_settings", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "APP_DATABASE_URI"):
                database_uri()


if __name__ == "__main__":
    unittest.main()
