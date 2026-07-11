"""
File Reader Service - Convert files to Markdown using MarkItDown.
"""

import os
import re
from pathlib import Path
from threading import Lock

from markitdown import MarkItDown

MAX_CHARS = 15000
SUPPORTED_EXTENSIONS = {
    ".csv",
    ".docx",
    ".html",
    ".htm",
    ".json",
    ".md",
    ".pdf",
    ".pptx",
    ".txt",
    ".xlsx",
    ".xml",
}

_converter: MarkItDown | None = None
_converter_lock = Lock()


def _get_converter() -> MarkItDown:
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                _converter = MarkItDown()
    return _converter


def _clean_content(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", normalized).strip()


def read_file(file_path: str) -> dict:
    """Read a local file and convert its content to Markdown using MarkItDown."""
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: '{file_path}'")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension or '[no extension]'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"File is empty: '{file_path}'")

    try:
        result = _get_converter().convert(str(path))
    except Exception as error:
        raise ValueError(f"Could not read '{path.name}': {error}") from error

    content = _clean_content(getattr(result, "text_content", "") or "")
    if not content:
        raise ValueError(f"No readable content found in '{path.name}'")

    total_chars = len(content)
    is_truncated = total_chars > MAX_CHARS
    return {
        "file_name": path.name,
        "content": content[:MAX_CHARS] if is_truncated else content,
        "total_chars": total_chars,
        "is_truncated": is_truncated,
    }
