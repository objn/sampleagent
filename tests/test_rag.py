import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import UUID

from docx import Document
from fastapi.testclient import TestClient

from Api.app import app
from RAG.knowledge import MAX_UPLOAD_BYTES, extract_chunks, ingest_file, search_knowledge
from RAG.uploads import queue_upload
from RAG.worker import process_one


class RagExtractionTests(unittest.TestCase):
    def test_docx_paragraphs_and_faq_table_are_extracted_in_order(self):
        document = Document()
        document.add_paragraph("OOCA frequently asked questions")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "How do I book a session?"
        table.cell(0, 1).text = "Open the booking page."
        document.add_paragraph("Contact support for more help.")
        stream = BytesIO()
        document.save(stream)

        chunks = extract_chunks("ooca_faq.DOCX", stream.getvalue())
        text = "\n".join(chunk for _, chunk in chunks)
        self.assertLess(text.index("OOCA frequently"), text.index("How do I book"))
        self.assertLess(text.index("How do I book"), text.index("Contact support"))
        self.assertIn("How do I book a session? | Open the booking page.", text)
        self.assertTrue(all(page is None for page, _ in chunks))

    def test_invalid_docx_is_rejected_during_extraction(self):
        with self.assertRaisesRegex(ValueError, "Could not read the DOCX"):
            extract_chunks("broken.docx", b"not a Word document")

    def test_utf8_text_is_split_into_searchable_chunks(self):
        chunks = extract_chunks("notes.txt", "ข้อมูลสำหรับค้นหา".encode("utf-8"))
        self.assertEqual(chunks, [(None, "ข้อมูลสำหรับค้นหา")])

    def test_unsupported_and_oversized_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only PDF, DOCX, TXT, and Markdown"):
            extract_chunks("image.png", b"data")
        with self.assertRaisesRegex(ValueError, "exceeds 5 MB"):
            extract_chunks("notes.txt", b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_ingestion_embeds_saves_and_invalidates_search_cache(self):
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.0] * 1536]
        saved = {
            "document_id": "example-id",
            "filename": "notes.txt",
            "duplicate": False,
        }
        with (
            patch("RAG.knowledge.existing_document", return_value=None),
            patch("RAG.knowledge._embeddings", return_value=embedder),
            patch("RAG.knowledge.save_document", return_value=saved) as save,
            patch("RAG.knowledge.invalidate_tool_results") as invalidate,
            patch.dict("os.environ", {"REDIS_URL": "redis://localhost:2126/1"}),
        ):
            result = ingest_file("notes.txt", b"hello")
        self.assertEqual(result["chunks"], 1)
        embedder.embed_documents.assert_called_once_with(["hello"])
        self.assertEqual(save.call_args.args[1:4], ("general", "notes.txt", [(None, "hello")]))
        invalidate.assert_called_once_with(
            "redis://localhost:2126/1", "search_uploaded_knowledge"
        )

    def test_search_embeds_query_and_returns_database_matches(self):
        embedder = Mock()
        embedder.embed_query.return_value = [0.0] * 1536
        with (
            patch("RAG.knowledge._embeddings", return_value=embedder),
            patch("RAG.knowledge.search_chunks", return_value=[{"content": "hello"}]) as search,
        ):
            result = search_knowledge("hello", 2)
        self.assertEqual(result, [{"content": "hello"}])
        search.assert_called_once_with([0.0] * 1536, 2, None)


class RagApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_upload_route_queues_file_and_returns_201(self):
        expected = {
            "upload_id": "00000000-0000-0000-0000-000000000001",
            "status": "pending",
            "status_url": "/rag/uploads/00000000-0000-0000-0000-000000000001",
        }
        with patch("Api.app.queue_upload", return_value=expected) as enqueue:
            response = self.client.post(
                "/rag/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), expected)
        self.assertEqual(response.headers["location"], expected["status_url"])
        enqueue.assert_called_once_with("notes.txt", "text/plain", b"hello", "general")

    def test_docx_upload_is_queued(self):
        expected = {
            "upload_id": "00000000-0000-0000-0000-000000000001",
            "status": "pending",
            "status_url": "/rag/uploads/00000000-0000-0000-0000-000000000001",
        }
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        with patch("Api.app.queue_upload", return_value=expected) as enqueue:
            response = self.client.post(
                "/rag/upload", data={"category": "ooca"},
                files={"file": ("ooca_faq.docx", b"docx bytes", content_type)},
            )
        self.assertEqual(response.status_code, 201)
        enqueue.assert_called_once_with("ooca_faq.docx", content_type, b"docx bytes", "ooca")

    def test_status_route_returns_404_for_missing_upload(self):
        with patch("Api.app.get_upload", return_value=None):
            response = self.client.get(
                "/rag/uploads/00000000-0000-0000-0000-000000000001"
            )
        self.assertEqual(response.status_code, 404)

    def test_status_route_returns_processing_state(self):
        upload_id = "00000000-0000-0000-0000-000000000001"
        with patch("Api.app.get_upload", return_value={"upload_id": upload_id, "status": "processing"}):
            response = self.client.get(f"/rag/uploads/{upload_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "processing")

    def test_search_route_returns_matches(self):
        matches = [{"filename": "notes.txt", "content": "hello", "score": 0.9}]
        with patch("Api.app.search_knowledge", return_value=matches) as search:
            response = self.client.post(
                "/rag/search", json={"query": "hello", "limit": 2}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": matches})
        search.assert_called_once_with("hello", 2, None)


class UploadQueueTests(unittest.TestCase):
    def test_docx_file_is_accepted_into_upload_queue(self):
        with TemporaryDirectory() as directory:
            with (
                patch("RAG.uploads.upload_dir", return_value=Path(directory)),
                patch("RAG.uploads.create_upload", return_value={"status": "pending"}) as create,
            ):
                result = queue_upload("ooca_faq.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"docx", "ooca")
        self.assertEqual(result["status"], "pending")
        self.assertTrue(create.call_args.kwargs["stored_name"].endswith(".docx"))
        self.assertEqual(create.call_args.kwargs["category"], "ooca")

    def test_original_file_is_saved_under_random_name_before_record_creation(self):
        with TemporaryDirectory() as directory:
            def create_record(**fields):
                stored = Path(directory) / fields["stored_name"]
                self.assertTrue(stored.exists())
                self.assertEqual(stored.read_bytes(), b"hello")
                self.assertEqual(fields["original_name"], "notes.txt")
                self.assertEqual(fields["content_type"], "text/plain")
                self.assertEqual(fields["size_bytes"], 5)
                self.assertEqual(fields["category"], "general")
                return {"upload_id": str(fields["upload_id"]), "status": "pending"}

            with (
                patch("RAG.uploads.upload_dir", return_value=Path(directory)),
                patch("RAG.uploads.create_upload", side_effect=create_record),
            ):
                record = queue_upload("notes.txt", "text/plain", b"hello")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertIn(record["upload_id"], record["status_url"])

    def test_worker_updates_completed_status_after_processing(self):
        upload_id = UUID("00000000-0000-0000-0000-000000000001")
        job = {
            "id": upload_id,
            "stored_name": "random.txt",
            "original_name": "notes.txt",
            "sha256": "hash",
            "category": "manuals",
        }
        result = {"document_id": str(upload_id), "chunks": 1, "duplicate": False}
        with (
            patch("RAG.worker.claim_upload", return_value=job),
            patch("RAG.worker.read_queued_upload", return_value=b"hello"),
            patch("RAG.worker.ingest_file", return_value=result) as ingest,
            patch("RAG.worker.complete_upload") as complete,
        ):
            self.assertTrue(process_one())
        ingest.assert_called_once_with("notes.txt", b"hello", "manuals")
        complete.assert_called_once_with(upload_id, result)

    def test_category_is_passed_from_upload_to_search(self):
        with patch("Api.app.queue_upload", return_value={"status_url": "/rag/uploads/id"}) as enqueue:
            response = TestClient(app).post(
                "/rag/upload", data={"category": "Manuals"},
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
        self.assertEqual(response.status_code, 201)
        enqueue.assert_called_once_with("notes.txt", "text/plain", b"hello", "Manuals")
        with patch("Api.app.search_knowledge", return_value=[]) as search:
            response = TestClient(app).post(
                "/rag/search", json={"query": "hello", "category": "manuals"}
            )
        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("hello", 4, "manuals")
