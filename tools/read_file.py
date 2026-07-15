"""
Read File Tool - Read and convert files to Markdown using MarkItDown.
"""

from registry.models import ToolDefinition
from registry.context import get_current_actor
from config import FILE_PREVIEW_CHARS
from services.artifacts import get_artifact_store
from services.documents import get_document_cache
from services.file_reader import read_file


def read_file_tool(artifact_id: str) -> dict:
    """Read one user-owned Drive artifact and remove its temporary file."""
    actor = get_current_actor()
    artifact = get_artifact_store().consume(artifact_id, actor["user_id"])
    try:
        full_result = read_file(artifact.path, max_chars=None)
    finally:
        get_artifact_store().delete_file(artifact.path)

    document = get_document_cache().put(
        actor["user_id"],
        full_result["content"],
        {
            **artifact.metadata,
            "source_type": "drive_file",
        },
    )
    if FILE_PREVIEW_CHARS < 1:
        raise ValueError("FILE_PREVIEW_CHARS must be positive")
    preview = full_result["content"][:FILE_PREVIEW_CHARS]
    return {
        **artifact.metadata,
        "document_ref": document.document_ref,
        "content": preview,
        "total_chars": full_result["total_chars"],
        "is_truncated": len(preview) < full_result["total_chars"],
    }


read_file_tool = ToolDefinition(
    name="read_file_tool",
    description=(
        "Read the content of a previously downloaded Google Drive artifact and convert it to Markdown. "
        "Pass the artifact_id returned by get_drive_file. "
        "The result includes document_ref for save_memory when the user asks to save the file content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The artifact ID returned by get_drive_file.",
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=read_file_tool,
)


ALL_READ_FILE_TOOLS = [read_file_tool]
