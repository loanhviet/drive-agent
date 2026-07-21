import pytest

import server
from services.drive_ingestion import DriveIngestionWorker
from services.ingestion import IngestionStore
from tools import drive_knowledge


class FakeDriveVectorStore:
    def __init__(self):
        self.removed = []

    def stage_revision(self, **_kwargs):
        return None

    def activate_revision(self, _corpus_id, _file_id, _revision_id):
        return None

    def remove_file(self, corpus_id, file_id):
        self.removed.append((corpus_id, file_id))


def test_search_drive_knowledge_returns_structured_citations(monkeypatch):
    class FakeSearchStore:
        def search(self, query_vector, **kwargs):
            assert query_vector == [1.0, 0.0]
            assert kwargs["top_k"] == 2
            return [
                {
                    "text": "Grounded page evidence",
                    "score": 0.9,
                    "metadata": {
                        "file_id": "file-1",
                        "source_name": "Guide.pdf",
                        "locator_type": "page",
                        "page_number": 3,
                        "section": None,
                        "web_view_link": "https://drive.google.com/file/d/file-1/view",
                    },
                }
            ]

    monkeypatch.setattr(drive_knowledge.embedding, "embed_query", lambda _query: [1.0, 0.0])
    monkeypatch.setattr(
        drive_knowledge,
        "get_drive_document_store",
        lambda: FakeSearchStore(),
    )

    result = drive_knowledge.search_drive_knowledge("Question", top_k=2)

    assert result["status"] == "found"
    assert result["results"][0]["citation_id"] == "S1"
    assert result["results"][0]["citation"]["page_number"] == 3


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("", 5, "must not be empty"),
        ("valid", 0, "between 1 and 10"),
        ("valid", True, "between 1 and 10"),
    ],
)
def test_search_drive_knowledge_validates_input(query, top_k, message):
    with pytest.raises(ValueError, match=message):
        drive_knowledge.search_drive_knowledge(query, top_k=top_k)


def test_worker_handles_missing_folder_and_discovery_failure(tmp_path):
    store = IngestionStore(str(tmp_path / "ingestion.db"))
    worker = DriveIngestionWorker(
        store=store,
        vector_store=FakeDriveVectorStore(),
        folder_id="",
        max_attempts=1,
    )
    missing = store.enqueue_job(requested_by="admin")
    worker.run_once()
    assert store.get_job(missing["job_id"])["error"] == "drive_folder_not_configured"

    failing = DriveIngestionWorker(
        store=store,
        vector_store=FakeDriveVectorStore(),
        folder_id="folder",
        discover=lambda _folder: (_ for _ in ()).throw(RuntimeError("Drive offline")),
        max_attempts=1,
    )
    failed = store.enqueue_job(requested_by="admin")
    failing.run_once()
    assert store.get_job(failed["job_id"])["status"] == "failed"
    assert "Drive offline" in store.get_job(failed["job_id"])["error"]


def test_worker_records_unsupported_and_stale_documents(tmp_path):
    store = IngestionStore(str(tmp_path / "ingestion.db"))
    vector_store = FakeDriveVectorStore()
    files = [
        {
            "id": "unsupported",
            "name": "Sheet.xlsx",
            "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "modifiedTime": "2026-01-01T00:00:00Z",
        }
    ]
    worker = DriveIngestionWorker(
        store=store,
        vector_store=vector_store,
        folder_id="folder",
        discover=lambda _folder: files,
        max_attempts=1,
    )
    job = store.enqueue_job(requested_by="admin")
    worker.run_once()
    assert store.get_document("shared-drive", "unsupported")["status"] == "unsupported"
    assert store.get_job(job["job_id"])["status"] == "succeeded"

    store.record_document(
        corpus_id="shared-drive",
        file_id="file-1",
        source_name="Guide.txt",
        mime_type="text/plain",
        drive_path="Guide.txt",
        web_view_link="",
        modified_time="2026-01-01T00:00:00Z",
        source_fingerprint="old",
        status="indexed",
        last_seen_job_id="old-job",
        active_revision_id="old-revision",
        chunk_count=1,
    )
    files[:] = [
        {
            "id": "file-1",
            "name": "Guide.txt",
            "mimeType": "text/plain",
            "modifiedTime": "2026-02-01T00:00:00Z",
            "md5Checksum": "new",
        }
    ]
    worker.downloader = lambda _file_id: (_ for _ in ()).throw(RuntimeError("download failed"))
    stale_job = store.enqueue_job(requested_by="admin")
    worker.run_once()

    document = store.get_document("shared-drive", "file-1")
    assert store.get_job(stale_job["job_id"])["status"] == "partial"
    assert document["status"] == "stale"
    assert document["active_revision_id"] == "old-revision"


def test_worker_lifecycle_starts_and_stops(tmp_path):
    worker = DriveIngestionWorker(
        store=IngestionStore(str(tmp_path / "ingestion.db")),
        vector_store=FakeDriveVectorStore(),
        folder_id="",
        poll_seconds=0.01,
    )

    worker.start()
    assert worker.is_alive
    worker.stop(timeout=1)
    assert not worker.is_alive


@pytest.mark.anyio
async def test_readiness_reports_all_local_checks(
    client,
    monkeypatch,
):
    class FakeWorker:
        is_alive = True

    class FakeIngestionStore:
        def sync_status(self):
            return {}

    class FakeVectorStore:
        def ensure_collection(self):
            return None

    monkeypatch.setattr(server, "GOOGLE_DRIVE_FOLDER_ID", "folder")
    monkeypatch.setattr(server, "GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    monkeypatch.setattr(server.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(server, "get_drive_ingestion_worker", lambda: FakeWorker())
    monkeypatch.setattr(server, "get_ingestion_store", lambda: FakeIngestionStore())
    monkeypatch.setattr(server, "get_drive_document_store", lambda: FakeVectorStore())

    response = await client.get("/api/ready")

    assert response.status_code == 200
    assert all(response.json()["checks"].values())
