"""Page/section-aware extraction for the shared Drive knowledge base."""

import re
from dataclasses import dataclass
from pathlib import Path

from services.chunking import chunk_document
from services.file_reader import read_file


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")


class UnsupportedDocumentError(ValueError):
    """Raised when a file cannot produce useful, citable text."""


@dataclass(frozen=True)
class DocumentBlock:
    text: str
    locator_type: str
    page_number: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class IndexedChunk:
    text: str
    chunk_index: int
    locator_type: str
    page_number: int | None
    section: str | None
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ExtractionResult:
    blocks: list[DocumentBlock]
    chunks: list[IndexedChunk]
    total_chars: int
    page_count: int


def is_supported_document(file: dict) -> bool:
    mime_type = str(file.get("mimeType", "")).casefold()
    name = str(file.get("name", "")).casefold()
    if mime_type in {
        "application/pdf",
        "application/vnd.google-apps.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
    }:
        return True
    return Path(name).suffix in {".pdf", ".docx", ".txt", ".md"}


def _clean_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    return re.sub(r"\n{4,}", "\n\n\n", normalized).strip()


def _extract_pdf(path: Path) -> tuple[list[DocumentBlock], int]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception as error:
        raise UnsupportedDocumentError(f"Could not open PDF: {error}") from error

    blocks: list[DocumentBlock] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception as error:
            raise UnsupportedDocumentError(
                f"Could not extract PDF page {page_number}: {error}"
            ) from error
        if text:
            blocks.append(
                DocumentBlock(
                    text=text,
                    locator_type="page",
                    page_number=page_number,
                )
            )

    readable_chars = sum(len(re.sub(r"\s+", "", block.text)) for block in blocks)
    if readable_chars < 50:
        raise UnsupportedDocumentError(
            "no_extractable_text: PDF appears scanned or contains too little readable text"
        )
    return blocks, len(reader.pages)


def _looks_like_heading(line: str) -> str | None:
    stripped = line.strip()
    match = _MARKDOWN_HEADING.match(stripped)
    if match:
        return match.group(1).strip()
    if _NUMBERED_HEADING.match(stripped):
        return stripped
    letters = [character for character in stripped if character.isalpha()]
    words = stripped.split()
    if letters and 1 <= len(words) <= 16 and len(stripped) <= 180 and stripped == stripped.upper():
        return stripped
    return None


def _section_blocks(text: str) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    section = "Document start"
    buffer: list[str] = []
    for line in text.splitlines():
        heading = _looks_like_heading(line)
        if heading:
            body = _clean_text("\n".join(buffer))
            if body:
                blocks.append(
                    DocumentBlock(text=body, locator_type="section", section=section)
                )
            section = heading
            buffer = []
            continue
        buffer.append(line)
    body = _clean_text("\n".join(buffer))
    if body:
        blocks.append(DocumentBlock(text=body, locator_type="section", section=section))
    return blocks


def _extract_structured_text(path: Path) -> tuple[list[DocumentBlock], int]:
    result = read_file(str(path), max_chars=None)
    content = _clean_text(result["content"])
    if not content:
        raise UnsupportedDocumentError("no_extractable_text: document is empty")
    return _section_blocks(content), 0


def extract_document(path: str, mime_type: str = "") -> ExtractionResult:
    source = Path(path)
    if mime_type == "application/pdf" or source.suffix.casefold() == ".pdf":
        blocks, page_count = _extract_pdf(source)
    else:
        blocks, page_count = _extract_structured_text(source)

    chunks: list[IndexedChunk] = []
    block_offset = 0
    for block in blocks:
        for chunk in chunk_document(block.text):
            chunks.append(
                IndexedChunk(
                    text=chunk.text,
                    chunk_index=len(chunks),
                    locator_type=block.locator_type,
                    page_number=block.page_number,
                    section=block.section,
                    start_char=block_offset + chunk.start_char,
                    end_char=block_offset + chunk.end_char,
                )
            )
        block_offset += len(block.text) + 1

    if not chunks:
        raise UnsupportedDocumentError("no_extractable_text: document has no indexable chunks")
    return ExtractionResult(
        blocks=blocks,
        chunks=chunks,
        total_chars=sum(len(block.text) for block in blocks),
        page_count=page_count,
    )
