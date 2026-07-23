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
    is_truncated = len(preview) < full_result["total_chars"]
    return {
        **artifact.metadata,
        "document_ref": document.document_ref,
        "content": preview,
        "total_chars": full_result["total_chars"],
        "is_truncated": is_truncated,
        "next_offset": len(preview) if is_truncated else None,
    }


def read_document_segment(
    document_ref: str,
    offset: int,
    max_chars: int = FILE_PREVIEW_CHARS,
) -> dict:
    """Read one bounded segment from a cached, user-owned document."""
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or not 1 <= max_chars <= FILE_PREVIEW_CHARS
    ):
        raise ValueError(
            f"max_chars must be an integer between 1 and {FILE_PREVIEW_CHARS}"
        )

    actor = get_current_actor()
    document = get_document_cache().get(document_ref, actor["user_id"])
    total_chars = len(document.content)
    if offset >= total_chars:
        raise ValueError("offset must be less than total_chars")

    end_offset = min(total_chars, offset + max_chars)
    is_truncated = end_offset < total_chars
    return {
        "document_ref": document.document_ref,
        "file_id": document.metadata.get("file_id", ""),
        "file_name": document.metadata.get("file_name", ""),
        "mime_type": document.metadata.get("mime_type", ""),
        "content": document.content[offset:end_offset],
        "offset": offset,
        "end_offset": end_offset,
        "total_chars": total_chars,
        "is_truncated": is_truncated,
        "next_offset": end_offset if is_truncated else None,
    }


read_file_tool = ToolDefinition(
    name="read_file_tool",
    description=(
        "Read the content of a previously downloaded Google Drive artifact and convert it to Markdown. "
        "Pass the artifact_id returned by get_drive_file. "
        "The result includes document_ref for save_memory and next_offset for "
        "read_document_segment when the preview is truncated."
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


read_document_segment_tool = ToolDefinition(
    name="read_document_segment",
    description=(
        "Read the next bounded segment of a document cached by read_file_tool. "
        "Use document_ref and next_offset from the previous result."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_ref": {
                "type": "string",
                "minLength": 1,
                "description": "Temporary document reference returned by read_file_tool.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Character offset to start reading from.",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": FILE_PREVIEW_CHARS,
                "description": "Maximum characters to return in this segment.",
            },
        },
        "required": ["document_ref", "offset"],
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=read_document_segment,
)


ALL_READ_FILE_TOOLS = [read_file_tool, read_document_segment_tool]
