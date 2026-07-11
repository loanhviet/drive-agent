"""
File Reader Service - Convert files to Markdown using MarkItDown.
"""

import os

MAX_CHARS = 15000


def read_file(file_path: str) -> dict:
    """Read a local file and convert its content to Markdown using MarkItDown."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # TODO: Implement file reading using MarkItDown
    # - Initialize MarkItDown
    # - Convert the file at file_path
    # - Get text_content from the result
    # - Truncate if longer than MAX_CHARS
    # - Return {"file_name": ..., "content": ...}
    pass
