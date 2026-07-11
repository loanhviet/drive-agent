"""
Read File Tool - Read and convert files to Markdown using MarkItDown.
"""

from registry.models import ToolDefinition
from services.file_reader import read_file


def read_local_file(file_path: str) -> dict:
    """Read a local file and convert to Markdown."""
    return read_file(file_path)


read_file_tool = ToolDefinition(
    name="read_file",
    description=(
        "Read a local file and convert its content to Markdown. "
        "Supports many formats: PDF, DOCX, XLSX, PPTX, images, text files, and more. "
        "Provide the full file path."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The full path to the local file to read.",
            },
        },
        "required": ["file_path"],
    },
    required_scopes=["drive:read"],
    handler=read_local_file,
)


ALL_READ_FILE_TOOLS = [read_file_tool]
