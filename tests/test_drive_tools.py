import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from registry.context import execution_context
from registry.registry import ToolRegistry
from services.artifacts import ArtifactError, ArtifactStore
from services.audit import AuditStore
from services.documents import DocumentCache
from services import drive_service
from tools import google_drive, read_file as read_file_module


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeFiles:
    def __init__(self, pages=None, metadata=None):
        self.pages = pages or []
        self.metadata = metadata or {}
        self.list_calls = []
        self.media_calls = []
        self.export_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        index = 1 if kwargs.get("pageToken") else 0
        return FakeRequest(self.pages[index])

    def get(self, **kwargs):
        if kwargs.get("fields") == "name, mimeType":
            return FakeRequest(self.metadata)
        raise AssertionError(f"Unexpected get call: {kwargs}")

    def get_media(self, **kwargs):
        self.media_calls.append(kwargs)
        return b"media-request"

    def export_media(self, **kwargs):
        self.export_calls.append(kwargs)
        return b"export-request"


class FakeDriveService:
    def __init__(self, files):
        self._files = files

    def files(self):
        return self._files


class FakeDownloader:
    def __init__(self, buffer, request):
        self.buffer = buffer
        self.request = request
        self.done = False

    def next_chunk(self):
        if not self.done:
            self.buffer.write(b"downloaded content")
            self.done = True
        return None, self.done


def actor(user_id="user-1"):
    return {"user_id": user_id, "role": "admin", "scopes": ["drive:read"]}


def test_list_files_follows_pagination(monkeypatch):
    files = FakeFiles(
        pages=[
            {
                "files": [
                    {
                        "id": "first",
                        "name": "first.txt",
                        "mimeType": "text/plain",
                        "size": "12",
                        "modifiedTime": "2026-01-01T00:00:00Z",
                    }
                ],
                "nextPageToken": "next-page",
            },
            {
                "files": [
                    {
                        "id": "second",
                        "name": "second.pdf",
                        "mimeType": "application/pdf",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(drive_service, "_get_service", lambda: FakeDriveService(files))

    result = drive_service.list_files(folder_id="folder-1", page_size=25)

    assert [item["id"] for item in result] == ["first", "second"]
    assert files.list_calls[0]["q"] == "trashed = false and 'folder-1' in parents"
    assert files.list_calls[0]["pageToken"] is None
    assert files.list_calls[1]["pageToken"] == "next-page"


@pytest.mark.parametrize("page_size", [0, 1001])
def test_list_files_validates_page_size(monkeypatch, page_size):
    monkeypatch.setattr(drive_service, "_get_service", lambda: object())

    with pytest.raises(ValueError, match="between 1 and 1000"):
        drive_service.list_files(page_size=page_size)


def test_download_regular_drive_file(monkeypatch):
    files = FakeFiles(metadata={"name": "notes.txt", "mimeType": "text/plain"})
    monkeypatch.setattr(drive_service, "_get_service", lambda: FakeDriveService(files))
    monkeypatch.setattr(drive_service, "MediaIoBaseDownload", FakeDownloader)

    download = drive_service.download_file("file-1")
    try:
        assert download["file_id"] == "file-1"
        assert download["file_name"] == "notes.txt"
        assert Path(download["temp_path"]).suffix == ".txt"
        assert Path(download["temp_path"]).read_bytes() == b"downloaded content"
        assert files.media_calls == [{"fileId": "file-1"}]
    finally:
        os.unlink(download["temp_path"])


def test_download_google_doc_uses_export(monkeypatch):
    files = FakeFiles(
        metadata={
            "name": "Document",
            "mimeType": "application/vnd.google-apps.document",
        }
    )
    monkeypatch.setattr(drive_service, "_get_service", lambda: FakeDriveService(files))
    monkeypatch.setattr(drive_service, "MediaIoBaseDownload", FakeDownloader)

    download = drive_service.download_file("doc-1")
    try:
        assert Path(download["temp_path"]).suffix == ".docx"
        assert files.export_calls == [
            {
                "fileId": "doc-1",
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }
        ]
    finally:
        os.unlink(download["temp_path"])


def test_get_and_read_drive_artifact_without_exposing_temp_path(monkeypatch, tmp_path):
    temporary_file = tmp_path / "drive-notes.txt"
    temporary_file.write_text("Python is remembered from Drive.", encoding="utf-8")
    artifact_store = ArtifactStore()
    document_cache = DocumentCache()
    monkeypatch.setattr(google_drive, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)
    monkeypatch.setattr(
        google_drive.drive_service,
        "download_file",
        lambda file_id: {
            "file_id": file_id,
            "file_name": "drive-notes.txt",
            "mime_type": "text/plain",
            "temp_path": str(temporary_file),
        },
    )

    with execution_context(actor()):
        downloaded = google_drive.get_drive_file("drive-file")
        assert "temp_path" not in downloaded
        read = read_file_module.read_file_tool.handler(downloaded["artifact_id"])

    assert read["content"] == "Python is remembered from Drive."
    assert read["file_id"] == "drive-file"
    assert "document_ref" in read
    assert not temporary_file.exists()
    assert document_cache.get(read["document_ref"], "user-1").content == read["content"]


def test_reader_failure_also_deletes_artifact_file(monkeypatch, tmp_path):
    temporary_file = tmp_path / "broken.txt"
    temporary_file.write_text("content", encoding="utf-8")
    artifact_store = ArtifactStore()
    artifact = artifact_store.register(
        "user-1",
        {
            "file_id": "broken-file",
            "file_name": "broken.txt",
            "mime_type": "text/plain",
            "temp_path": str(temporary_file),
        },
    )
    monkeypatch.setattr(read_file_module, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(
        read_file_module,
        "read_file",
        lambda _path, max_chars=None: (_ for _ in ()).throw(ValueError("reader failed")),
    )

    with execution_context(actor()):
        with pytest.raises(ValueError, match="reader failed"):
            read_file_module.read_file_tool.handler(artifact.artifact_id)

    assert not temporary_file.exists()


def test_reader_limits_llm_preview_but_caches_full_document(monkeypatch, tmp_path):
    temporary_file = tmp_path / "long.txt"
    temporary_file.write_text("0123456789", encoding="utf-8")
    artifact_store = ArtifactStore()
    document_cache = DocumentCache()
    artifact = artifact_store.register(
        "user-1",
        {
            "file_id": "long-file",
            "file_name": "long.txt",
            "mime_type": "text/plain",
            "temp_path": str(temporary_file),
        },
    )
    monkeypatch.setattr(read_file_module, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)
    monkeypatch.setattr(read_file_module, "FILE_PREVIEW_CHARS", 5)

    with execution_context(actor()):
        result = read_file_module.read_file_tool.handler(artifact.artifact_id)

    assert result["content"] == "01234"
    assert result["total_chars"] == 10
    assert result["is_truncated"] is True
    assert result["next_offset"] == 5
    assert document_cache.get(result["document_ref"], "user-1").content == "0123456789"
    assert not temporary_file.exists()


def test_artifact_is_user_scoped_and_expires(tmp_path):
    clock = [0.0]
    store = ArtifactStore(ttl_seconds=10, clock=lambda: clock[0])
    temporary_file = tmp_path / "private.txt"
    temporary_file.write_text("private", encoding="utf-8")
    artifact = store.register(
        "owner",
        {
            "file_id": "private",
            "file_name": "private.txt",
            "mime_type": "text/plain",
            "temp_path": str(temporary_file),
        },
    )

    with pytest.raises(ArtifactError, match="does not belong"):
        store.consume(artifact.artifact_id, "another-user")
    assert temporary_file.exists()

    clock[0] = 11.0
    with pytest.raises(ArtifactError, match="not found or has expired"):
        store.consume(artifact.artifact_id, "owner")
    assert not temporary_file.exists()


def test_registry_runs_staged_drive_tools_in_authenticated_context(monkeypatch, tmp_path):
    temporary_file = tmp_path / "registry-drive.txt"
    temporary_file.write_text("Registry reads Drive files.", encoding="utf-8")
    artifact_store = ArtifactStore()
    document_cache = DocumentCache()
    monkeypatch.setattr(google_drive, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)
    monkeypatch.setattr(
        google_drive.drive_service,
        "download_file",
        lambda file_id: {
            "file_id": file_id,
            "file_name": "registry-drive.txt",
            "mime_type": "text/plain",
            "temp_path": str(temporary_file),
        },
    )
    registry = ToolRegistry(
        authenticator=lambda _token: actor(),
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )
    registry.register(google_drive.get_drive_file_tool)
    registry.register(read_file_module.read_file_tool)

    downloaded = registry.call("get_drive_file", {"file_id": "drive-id"}, "token")
    read = registry.call(
        "read_file_tool",
        {"artifact_id": downloaded["result"]["artifact_id"]},
        "token",
    )

    assert downloaded["ok"] is True
    assert read["ok"] is True
    assert read["result"]["content"] == "Registry reads Drive files."
    assert all(log["status"] == "success" for log in registry.get_audit_log())

def test_walk_files_recurses_and_tracks_drive_path(monkeypatch):
    tree = {
        "root": [
            {
                "id": "folder-a",
                "name": "Folder A",
                "mimeType": drive_service.FOLDER_MIME_TYPE,
            },
            {
                "id": "root-file",
                "name": "Root.txt",
                "mimeType": "text/plain",
            },
        ],
        "folder-a": [
            {
                "id": "nested-file",
                "name": "Nested.pdf",
                "mimeType": "application/pdf",
            }
        ],
    }
    monkeypatch.setattr(
        drive_service,
        "list_files",
        lambda folder_id, page_size=1000: tree[folder_id],
    )

    files = drive_service.walk_files("root")

    assert [(item["id"], item["drive_path"]) for item in files] == [
        ("root-file", "Root.txt"),
        ("nested-file", "Folder A/Nested.pdf"),
    ]


def test_search_files_matches_normalized_name_tokens_in_nested_folder(monkeypatch):
    candidates = [
        {
            "id": "report",
            "name": "Báo-Cáo Tuyển Dụng 2026.pdf",
            "mimeType": "application/pdf",
            "drive_path": "HR/Báo-Cáo Tuyển Dụng 2026.pdf",
        },
        {
            "id": "notes",
            "name": "Meeting notes.txt",
            "mimeType": "text/plain",
            "drive_path": "Meeting notes.txt",
        },
    ]
    seen_roots = []

    def walk_files(folder_id):
        seen_roots.append(folder_id)
        return candidates

    monkeypatch.setattr(drive_service, "walk_files", walk_files)

    result = drive_service.search_files(
        "bao cao 2026",
        folder_id="root-folder",
    )

    assert seen_roots == ["root-folder"]
    assert [item["id"] for item in result] == ["report"]


def test_search_files_uses_configured_root_and_applies_limit(monkeypatch):
    monkeypatch.setattr(drive_service, "GOOGLE_DRIVE_FOLDER_ID", "configured-root")
    monkeypatch.setattr(
        drive_service,
        "walk_files",
        lambda folder_id: [
            {"id": "first", "name": "Plan alpha.md"},
            {"id": "second", "name": "Plan beta.md"},
        ]
        if folder_id == "configured-root"
        else [],
    )

    result = drive_service.search_files("plan", limit=1)

    assert [item["id"] for item in result] == ["first"]


@pytest.mark.parametrize(
    ("query", "limit"),
    [("", 20), ("---", 20), ("report", 0), ("report", 51), ("report", True)],
)
def test_search_files_validates_input(monkeypatch, query, limit):
    monkeypatch.setattr(drive_service, "GOOGLE_DRIVE_FOLDER_ID", "")

    with pytest.raises(ValueError):
        drive_service.search_files(query, limit=limit)


def test_search_drive_files_tool_returns_bounded_metadata(monkeypatch):
    monkeypatch.setattr(
        google_drive.drive_service,
        "search_files",
        lambda query, folder_id, limit: [
            {"id": "match", "name": "Matched.pdf", "drive_path": "Docs/Matched.pdf"}
        ],
    )

    result = google_drive.search_drive_files("matched", folder_id="root", limit=3)

    assert result == {
        "status": "found",
        "query": "matched",
        "results_count": 1,
        "files": [
            {"id": "match", "name": "Matched.pdf", "drive_path": "Docs/Matched.pdf"}
        ],
    }


def test_search_drive_files_tool_requires_drive_read_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(google_drive.drive_service, "search_files", lambda *args, **kwargs: [])
    registry = ToolRegistry(
        authenticator=lambda _token: {
            "user_id": "user-1",
            "role": "user",
            "scopes": [],
        },
        audit_store=AuditStore(str(tmp_path / "audit-search.db")),
    )
    registry.register(google_drive.search_files_tool)

    result = registry.call("search_drive_files", {"query": "report"}, "token")

    assert result["ok"] is False
    assert result["error"]["code"] == "missing_scope"


def test_read_document_segment_paginates_without_consuming_cache(monkeypatch):
    document_cache = DocumentCache()
    document = document_cache.put(
        "user-1",
        "0123456789",
        {
            "file_id": "file-1",
            "file_name": "numbers.txt",
            "mime_type": "text/plain",
        },
    )
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)

    with execution_context(actor()):
        first = read_file_module.read_document_segment(
            document.document_ref,
            offset=0,
            max_chars=4,
        )
        final = read_file_module.read_document_segment(
            document.document_ref,
            offset=8,
            max_chars=4,
        )

    assert first == {
        "document_ref": document.document_ref,
        "file_id": "file-1",
        "file_name": "numbers.txt",
        "mime_type": "text/plain",
        "content": "0123",
        "offset": 0,
        "end_offset": 4,
        "total_chars": 10,
        "is_truncated": True,
        "next_offset": 4,
    }
    assert final["content"] == "89"
    assert final["end_offset"] == 10
    assert final["is_truncated"] is False
    assert final["next_offset"] is None
    assert document_cache.get(document.document_ref, "user-1") == document


@pytest.mark.parametrize(
    ("offset", "max_chars", "message"),
    [
        (-1, 4, "offset"),
        (True, 4, "offset"),
        (10, 4, "offset"),
        (0, 0, "max_chars"),
        (0, read_file_module.FILE_PREVIEW_CHARS + 1, "max_chars"),
        (0, True, "max_chars"),
    ],
)
def test_read_document_segment_validates_bounds(
    monkeypatch,
    offset,
    max_chars,
    message,
):
    document_cache = DocumentCache()
    document = document_cache.put("user-1", "0123456789", {})
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)

    with execution_context(actor()):
        with pytest.raises(ValueError, match=message):
            read_file_module.read_document_segment(
                document.document_ref,
                offset=offset,
                max_chars=max_chars,
            )


def test_read_document_segment_rejects_another_user(monkeypatch):
    document_cache = DocumentCache()
    document = document_cache.put("owner", "private document", {})
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)

    with execution_context(actor("other-user")):
        with pytest.raises(ValueError, match="does not belong"):
            read_file_module.read_document_segment(
                document.document_ref,
                offset=0,
                max_chars=4,
            )


def test_read_document_segment_rejects_expired_reference(monkeypatch):
    clock = [0.0]
    document_cache = DocumentCache(ttl_seconds=10, clock=lambda: clock[0])
    document = document_cache.put("user-1", "temporary document", {})
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)
    clock[0] = 11.0

    with execution_context(actor()):
        with pytest.raises(ValueError, match="not found or has expired"):
            read_file_module.read_document_segment(
                document.document_ref,
                offset=0,
                max_chars=4,
            )
