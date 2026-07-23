"""
Google Drive Tools - List files and read file content.
"""

from registry.models import ToolDefinition
from registry.context import get_current_actor
from services import drive_service
from services.artifacts import get_artifact_store


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
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=list_drive_files,
)


# ============================================================
# Tool 2: SEARCH DRIVE FILES
# ============================================================

def search_drive_files(
    query: str,
    folder_id: str = "",
    limit: int = 20,
) -> dict:
    """Search for Drive files by normalized file-name tokens."""
    files = drive_service.search_files(
        query,
        folder_id=folder_id or None,
        limit=limit,
    )
    return {
        "status": "found" if files else "not_found",
        "query": query.strip(),
        "results_count": len(files),
        "files": files,
    }


search_files_tool = ToolDefinition(
    name="search_drive_files",
    description=(
        "Search Google Drive files by file name without reading their content. "
        "Matching is case-insensitive and accent-insensitive. "
        "Optionally provide a folder_id to search recursively within that folder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "File-name words to search for.",
            },
            "folder_id": {
                "type": "string",
                "description": "Optional Drive folder ID. Defaults to the configured folder.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum number of matching files to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=search_drive_files,
)


# ============================================================
# Tool 3: DOWNLOAD DRIVE FILE
# ============================================================

def get_drive_file(file_id: str) -> dict:
    """Download a Drive file and return a short-lived artifact ID."""
    actor = get_current_actor()
    download = drive_service.download_file(file_id=file_id)
    try:
        artifact = get_artifact_store().register(actor["user_id"], download)
    except Exception:
        get_artifact_store().delete_file(download["temp_path"])
        raise
    return {
        "artifact_id": artifact.artifact_id,
        "file_id": artifact.metadata["file_id"],
        "file_name": artifact.metadata["file_name"],
        "mime_type": artifact.metadata["mime_type"],
    }


get_drive_file_tool = ToolDefinition(
    name="get_drive_file",
    description=(
        "Download a Google Drive file by its file ID and return an artifact_id. "
        "Use list_drive_files first to get a file ID, then pass this artifact_id to read_file_tool."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The Google Drive file ID to download.",
            },
        },
        "required": ["file_id"],
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=get_drive_file,
)


ALL_DRIVE_TOOLS = [list_files_tool, search_files_tool, get_drive_file_tool]
