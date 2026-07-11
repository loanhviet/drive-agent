"""
Google Drive Tools - List files and read file content.
"""

from registry.models import ToolDefinition
from services import drive_service
from services.file_reader import read_file


# ============================================================
# Tool 1: LIST DRIVE FILES
# ============================================================

def list_drive_files(folder_id: str = "") -> dict:
    """List all files in Google Drive."""
    fid = folder_id if folder_id else None
    files = drive_service.list_files(folder_id=fid)
    return {
        "total_files": len(files),
        "files": files,
    }


list_files_tool = ToolDefinition(
    name="list_drive_files",
    description=(
        "List all files in Google Drive. "
        "Returns file names, IDs, types, sizes, and modification times. "
        "Optionally provide a folder_id to list files in a specific folder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "folder_id": {
                "type": "string",
                "description": "Optional Google Drive folder ID to list files from. Leave empty for default folder.",
            },
        },
        "required": [],
    },
    required_scopes=["drive:read"],
    handler=list_drive_files,
)


# ============================================================
# Tool 2: READ DRIVE FILE
# ============================================================

def read_drive_file(file_id: str) -> dict:
    """Download a file from Google Drive and read its content."""
    import os
    download = drive_service.download_file(file_id=file_id)
    temp_path = download["temp_path"]
    try:
        result = read_file(temp_path)
    finally:
        os.unlink(temp_path)
    return {
        "file_id": download["file_id"],
        "file_name": download["file_name"],
        "mime_type": download["mime_type"],
        "content": result["content"],
    }


read_file_tool = ToolDefinition(
    name="read_drive_file",
    description=(
        "Read the content of a file from Google Drive by its file ID. "
        "Supports many formats: PDF, DOCX, XLSX, PPTX, images, Google Docs/Sheets/Slides, text files, and more. "
        "Content is converted to Markdown using MarkItDown. "
        "Use list_drive_files first to get file IDs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The Google Drive file ID to read.",
            },
        },
        "required": ["file_id"],
    },
    required_scopes=["drive:read"],
    handler=read_drive_file,
)


ALL_DRIVE_TOOLS = [list_files_tool, read_file_tool]
