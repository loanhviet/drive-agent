"""
Google Drive Service - Wraps Google Drive API v3.
Supports listing, recursively discovering, and downloading file content.
"""

import os
import tempfile
from collections import deque

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import DRIVE_MAX_FILE_BYTES, GOOGLE_DRIVE_FOLDER_ID, GOOGLE_SERVICE_ACCOUNT_FILE

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"

_service = None


def _get_service():
    global _service
    if _service is None:
        creds_path = GOOGLE_SERVICE_ACCOUNT_FILE
        if not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Google Service Account file not found: '{creds_path}'. "
                "Download it from Google Cloud Console and place it in the project root."
            )
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=SCOPES,
        )
        _service = build("drive", "v3", credentials=creds)
    return _service


def list_files(folder_id: str = None, page_size: int = 50) -> list[dict]:
    """List files, including folders, under one Drive parent."""
    service = _get_service()
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")

    query_parts = ["trashed = false"]
    if folder_id:
        safe_folder_id = folder_id.replace("'", "\\'")
        query_parts.append(f"'{safe_folder_id}' in parents")
    elif GOOGLE_DRIVE_FOLDER_ID:
        safe_folder_id = GOOGLE_DRIVE_FOLDER_ID.replace("'", "\\'")
        query_parts.append(f"'{safe_folder_id}' in parents")

    query = " and ".join(query_parts)
    files: list[dict] = []
    page_token = None
    while True:
        results = (
            service.files()
            .list(
                q=query,
                pageSize=page_size,
                pageToken=page_token,
                fields=(
                    "nextPageToken, files(id, name, mimeType, size, modifiedTime, "
                    "md5Checksum, webViewLink, parents)"
                ),
                orderBy="modifiedTime desc",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return [
        {
            "id": item["id"],
            "name": item["name"],
            "mimeType": item.get("mimeType", ""),
            "size": item.get("size", "unknown"),
            "modifiedTime": item.get("modifiedTime", ""),
            "md5Checksum": item.get("md5Checksum", ""),
            "webViewLink": item.get("webViewLink", ""),
            "parents": item.get("parents", []),
        }
        for item in files
    ]


def walk_files(folder_id: str, page_size: int = 1000) -> list[dict]:
    """Recursively discover non-folder files and attach a stable display path."""
    if not folder_id or not folder_id.strip():
        raise ValueError("folder_id is required for recursive Drive discovery")

    queue = deque([(folder_id, "")])
    visited: set[str] = set()
    discovered: list[dict] = []
    while queue:
        parent_id, parent_path = queue.popleft()
        if parent_id in visited:
            continue
        visited.add(parent_id)
        for item in list_files(folder_id=parent_id, page_size=page_size):
            name = item.get("name", "")
            drive_path = f"{parent_path}/{name}".strip("/")
            if item.get("mimeType") == FOLDER_MIME_TYPE:
                if item["id"] not in visited:
                    queue.append((item["id"], drive_path))
                continue
            discovered.append({**item, "drive_path": drive_path})
    return discovered


def download_file(file_id: str, max_bytes: int = DRIVE_MAX_FILE_BYTES) -> dict:
    """Download a Drive file to a bounded temporary file."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    service = _get_service()

    file_meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    mime_type = file_meta.get("mimeType", "")
    file_name = file_meta.get("name", "")

    export_map = {
        "application/vnd.google-apps.document": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".docx",
        ),
        "application/vnd.google-apps.spreadsheet": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xlsx",
        ),
        "application/vnd.google-apps.presentation": (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".pptx",
        ),
    }

    if mime_type in export_map:
        export_mime, extension = export_map[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)
        extension = os.path.splitext(file_name)[1] or ".bin"

    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    try:
        downloader = MediaIoBaseDownload(temporary, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
            temporary.flush()
            if temporary.tell() > max_bytes:
                raise ValueError(
                    f"Drive file exceeds the configured limit of {max_bytes} bytes"
                )
        temporary.close()
    except Exception:
        temporary.close()
        try:
            os.unlink(temporary.name)
        except FileNotFoundError:
            pass
        raise

    return {
        "file_id": file_id,
        "file_name": file_name,
        "mime_type": mime_type,
        "temp_path": temporary.name,
    }
