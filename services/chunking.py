"""Small, dependency-free character chunker for document RAG."""

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text near natural boundaries while preserving overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    separators = ("\n\n", "\n", ". ", " ")
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(
                (text.rfind(separator, start + chunk_size // 2, end) for separator in separators),
                default=-1,
            )
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
