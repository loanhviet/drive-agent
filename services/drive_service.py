"""
Google Drive Service - Wraps Google Drive API v3.
Supports listing files and downloading file content.
"""

import io
import os
import tempfile
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_DRIVE_FOLDER_ID

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_service = None


def _get_service():
    global _service
    if _service is None:
        creds_path = GOOGLE_SERVICE_ACCOUNT_FILE
        if not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Google Service Account file not found: '{creds_path}'. "
                f"Download it from Google Cloud Console and place it in the project root."
            )
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        _service = build("drive", "v3", credentials=creds)
    return _service


def list_files(folder_id: str = None, page_size: int = 50) -> list[dict]:
    """List files in Google Drive (optionally within a specific folder)."""
    service = _get_service()

    query_parts = ["trashed = false"]
    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")
    elif GOOGLE_DRIVE_FOLDER_ID:
        query_parts.append(f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents")

    query = " and ".join(query_parts)

    results = service.files().list(
        q=query,
        pageSize=page_size,
        fields="files(id, name, mimeType, size, modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute()

    files = results.get("files", [])
    return [
        {
            "id": f["id"],
            "name": f["name"],
            "mimeType": f.get("mimeType", ""),
            "size": f.get("size", "unknown"),
            "modifiedTime": f.get("modifiedTime", ""),
        }
        for f in files
    ]


def download_file(file_id: str) -> dict:
    """Download a file from Google Drive to a temp file. Returns metadata and temp path."""
    service = _get_service()

    file_meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    mime_type = file_meta.get("mimeType", "")
    file_name = file_meta.get("name", "")

    # Google Docs/Sheets/Slides → export to a compatible format
    export_map = {
        "application/vnd.google-apps.document": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        "application/vnd.google-apps.spreadsheet": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
        "application/vnd.google-apps.presentation": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    }

    if mime_type in export_map:
        export_mime, ext = export_map[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)
        ext = os.path.splitext(file_name)[1] or ".bin"

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(buffer.getvalue())
        tmp_path = tmp.name

    return {
        "file_id": file_id,
        "file_name": file_name,
        "mime_type": mime_type,
        "temp_path": tmp_path,
    }
