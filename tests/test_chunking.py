import pytest

from services.chunking import chunk_document, split_text


def test_empty_text_has_no_chunks():
    assert split_text("   ") == []


def test_chunking_prefers_natural_boundaries_and_overlap():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."

    chunks = split_text(text, chunk_size=32, chunk_overlap=8)

    assert len(chunks) >= 2
    assert all(chunk.strip() == chunk for chunk in chunks)
    assert all(len(chunk) <= 32 for chunk in chunks)
    assert "First sentence" in chunks[0]
    assert "Fourth sentence" in chunks[-1]


def test_document_chunks_include_sections_and_offsets():
    text = "1. INTRODUCTION\nFirst paragraph about context.\n\n2.1. RESULTS\nAccuracy is 90 percent."

    chunks = chunk_document(text, chunk_size=50, chunk_overlap=10)

    assert chunks[0].chunk_index == 0
    assert chunks[0].section == "1. INTRODUCTION"
    assert chunks[-1].section == "2.1. RESULTS"
    assert all(text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunking_validates_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, chunk_overlap=overlap)
