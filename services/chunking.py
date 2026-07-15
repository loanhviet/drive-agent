"""Structure-aware, dependency-free chunking for document RAG."""

import re
from dataclasses import dataclass

from config import CHUNK_OVERLAP, CHUNK_SIZE


_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    section: str


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.isdigit() or len(stripped) > 180:
        return False
    if _NUMBERED_HEADING.match(stripped):
        return True
    letters = [character for character in stripped if character.isalpha()]
    words = stripped.split()
    return bool(letters) and 1 <= len(words) <= 16 and stripped == stripped.upper()


def _section_markers(text: str) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if _is_heading(line):
            markers.append((offset, line.strip()))
        offset += len(line)
    return markers


def _section_for_range(markers: list[tuple[int, str]], start: int, end: int) -> str:
    section = ""
    cursor = start
    spans: list[tuple[int, str]] = []
    for marker_offset, marker_name in markers:
        if marker_offset <= start:
            section = marker_name
            continue
        if marker_offset >= end:
            break
        spans.append((marker_offset - cursor, section))
        cursor = marker_offset
        section = marker_name
    spans.append((end - cursor, section))
    return max(spans, key=lambda item: item[0])[1]


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split text near natural boundaries and retain source offsets/sections."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    normalized = text.strip()
    if not normalized:
        return []

    markers = _section_markers(normalized)
    chunks: list[DocumentChunk] = []
    start = 0
    separators = ("\n\n", "\n", ". ", " ")
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            boundary = max(
                (
                    normalized.rfind(separator, start + chunk_size // 2, end)
                    for separator in separators
                ),
                default=-1,
            )
            if boundary > start:
                end = boundary + 1

        raw_chunk = normalized[start:end]
        leading_space = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing_space = len(raw_chunk) - len(raw_chunk.rstrip())
        chunk_start = start + leading_space
        chunk_end = end - trailing_space
        chunk_text = normalized[chunk_start:chunk_end]
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_char=chunk_start,
                    end_char=chunk_end,
                    section=_section_for_range(markers, chunk_start, chunk_end),
                )
            )
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Backward-compatible text-only view of :func:`chunk_document`."""
    return [
        chunk.text
        for chunk in chunk_document(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]
