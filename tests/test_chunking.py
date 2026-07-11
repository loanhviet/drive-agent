import pytest

from services.chunking import split_text


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


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunking_validates_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, chunk_overlap=overlap)
