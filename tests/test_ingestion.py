import sys
from types import SimpleNamespace

from qdrant_client import QdrantClient

from services import document_extraction
from services.document_extraction import (
    DocumentBlock,
    ExtractionResult,
    IndexedChunk,
    extract_document,
)
from services.drive_ingestion import DriveIngestionWorker
from services.drive_vectorstore import DriveDocumentVectorStore
from services.ingestion import IngestionStore


def _extraction(text="Shared Drive knowledge"):
    block = DocumentBlock(text=text, locator_type="section", section="Overview")
    chunk = IndexedChunk(
        text=text,
        chunk_index=0,
        locator_type="section",
        page_number=None,
        section="Overview",
        start_char=0,
        end_char=len(text),
    )
    return ExtractionResult(
        blocks=[block],
        chunks=[chunk],
        total_chars=len(text),
        page_count=0,
    )


def test_ingestion_store_deduplicates_and_recovers_jobs(tmp_path):
    store = IngestionStore(str(tmp_path / "ingestion.db"))

    first = store.enqueue_job(requested_by="admin")
    duplicate = store.enqueue_job(requested_by="admin")
    assert duplicate["job_id"] == first["job_id"]
    assert duplicate["deduplicated"] is True

    claimed = store.claim_next_job()
    assert claimed["status"] == "running"
    assert store.recover_incomplete_jobs() == 1
    assert store.claim_next_job()["job_id"] == first["job_id"]


def test_ingestion_store_lists_and_marks_documents_removed(tmp_path):
    store = IngestionStore(str(tmp_path / "ingestion.db"))
    store.record_document(
        corpus_id="shared-drive",
        file_id="file-1",
        source_name="Guide.pdf",
        mime_type="application/pdf",
        drive_path="Docs/Guide.pdf",
        web_view_link="https://drive.google.com/file/d/file-1/view",
        modified_time="2026-01-01T00:00:00Z",
        source_fingerprint="fingerprint",
        status="indexed",
        last_seen_job_id="job-1",
        active_revision_id="revision",
        chunk_count=2,
        page_count=1,
    )

    listed = store.list_documents(query="Guide")
    assert listed["total"] == 1
    assert listed["documents"][0]["active_revision_id"] == "revision"
    assert store.documents_missing_from_job("shared-drive", "job-2")[0]["file_id"] == "file-1"

    store.mark_removed("shared-drive", "file-1", "job-2")
    assert store.get_document("shared-drive", "file-1")["status"] == "removed"


def test_pdf_extraction_preserves_one_based_page_numbers(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    fake_reader = SimpleNamespace(
        pages=[
            FakePage("First page contains enough readable text for the extraction threshold."),
            FakePage("Second page contains another independently citable statement."),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda _path: fake_reader),
    )
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-fake")

    result = extract_document(str(source), mime_type="application/pdf")

    assert result.page_count == 2
    assert [chunk.page_number for chunk in result.chunks] == [1, 2]
    assert all(chunk.locator_type == "page" for chunk in result.chunks)


def test_structured_text_extraction_keeps_markdown_sections(monkeypatch, tmp_path):
    source = tmp_path / "guide.docx"
    source.write_bytes(b"fake")
    monkeypatch.setattr(
        document_extraction,
        "read_file",
        lambda _path, max_chars=None: {
            "content": "# Overview\nOverview body.\n# Details\nDetailed body.",
        },
    )

    result = extract_document(str(source))

    assert [block.section for block in result.blocks] == ["Overview", "Details"]
    assert [chunk.section for chunk in result.chunks] == ["Overview", "Details"]


def test_drive_document_vector_revision_search_and_removal(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    store = DriveDocumentVectorStore(client, "drive_documents_test", 3)
    extraction = _extraction("Python knowledge in the shared guide")
    metadata = {
        "id": "file-1",
        "name": "Guide.txt",
        "mimeType": "text/plain",
        "drive_path": "Guides/Guide.txt",
        "webViewLink": "https://drive.google.com/file/d/file-1/view",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }
    store.stage_revision(
        corpus_id="shared-drive",
        file_metadata=metadata,
        revision_id="revision-1",
        source_fingerprint="fingerprint",
        pipeline_version="v1",
        extraction=extraction,
        vectors=[[1.0, 0.0, 0.0]],
    )
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []

    store.activate_revision("shared-drive", "file-1", "revision-1")
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert results[0]["metadata"]["page_number"] is None
    assert results[0]["metadata"]["section"] == "Overview"

    store.remove_file("shared-drive", "file-1")
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []
    client.close()


def test_worker_incremental_sync_skips_unchanged_and_removes_missing(
    monkeypatch,
    tmp_path,
):
    store = IngestionStore(str(tmp_path / "ingestion.db"))

    class FakeVectorStore:
        def __init__(self):
            self.staged = []
            self.activated = []
            self.removed = []

        def stage_revision(self, **kwargs):
            self.staged.append(kwargs)

        def activate_revision(self, corpus_id, file_id, revision_id):
            self.activated.append((corpus_id, file_id, revision_id))

        def remove_file(self, corpus_id, file_id):
            self.removed.append((corpus_id, file_id))

    vector_store = FakeVectorStore()
    current_files = [
        {
            "id": "file-1",
            "name": "Guide.txt",
            "mimeType": "text/plain",
            "size": "20",
            "modifiedTime": "2026-01-01T00:00:00Z",
            "md5Checksum": "checksum-1",
            "webViewLink": "https://drive.google.com/file/d/file-1/view",
            "drive_path": "Guide.txt",
        }
    ]
    download_calls = []

    def downloader(file_id):
        download_calls.append(file_id)
        path = tmp_path / f"{len(download_calls)}.txt"
        path.write_text("content", encoding="utf-8")
        return {"temp_path": str(path)}

    monkeypatch.setattr(
        "services.drive_ingestion.extract_document",
        lambda _path, mime_type="": _extraction(),
    )
    monkeypatch.setattr(
        "services.drive_ingestion.embedding.embed_texts",
        lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    worker = DriveIngestionWorker(
        store=store,
        vector_store=vector_store,
        folder_id="folder",
        discover=lambda _folder: list(current_files),
        downloader=downloader,
        max_attempts=1,
    )

    first = store.enqueue_job(requested_by="admin")
    assert worker.run_once() is True
    assert store.get_job(first["job_id"])["status"] == "succeeded"
    assert len(download_calls) == 1

    second = store.enqueue_job(requested_by="admin")
    worker.run_once()
    assert store.get_job(second["job_id"])["skipped_count"] == 1
    assert len(download_calls) == 1

    first_revision = vector_store.activated[-1][2]
    full = store.enqueue_job(requested_by="admin", mode="full")
    worker.run_once()
    assert store.get_job(full["job_id"])["indexed_count"] == 1
    assert len(download_calls) == 2
    assert vector_store.activated[-1][2] != first_revision

    current_files.clear()
    removed = store.enqueue_job(requested_by="admin")
    worker.run_once()
    assert store.get_job(removed["job_id"])["removed_count"] == 1
    assert vector_store.removed == [("shared-drive", "file-1")]
    assert store.get_document("shared-drive", "file-1")["status"] == "removed"
