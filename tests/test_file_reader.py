from types import SimpleNamespace

import pytest

import services.file_reader as file_reader


class StubConverter:
    def __init__(self, content="converted content", error=None):
        self.content = content
        self.error = error

    def convert(self, _path):
        if self.error:
            raise self.error
        return SimpleNamespace(text_content=self.content)


def test_reads_and_cleans_supported_file(monkeypatch, tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        file_reader,
        "_converter",
        StubConverter("Xin chào Việt Nam  \r\n\r\n\r\n\r\nPython  "),
    )

    result = file_reader.read_file(str(source))

    assert result == {
        "file_name": "notes.txt",
        "content": "Xin chào Việt Nam\n\n\nPython",
        "total_chars": 26,
        "is_truncated": False,
    }


def test_real_markitdown_reads_text_file(monkeypatch, tmp_path):
    source = tmp_path / "real.txt"
    source.write_text("Drive Agent remembers Python.", encoding="utf-8")
    monkeypatch.setattr(file_reader, "_converter", None)

    result = file_reader.read_file(str(source))

    assert "Drive Agent remembers Python" in result["content"]


def test_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        file_reader.read_file(str(tmp_path / "missing.txt"))


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a regular file"):
        file_reader.read_file(str(tmp_path))


def test_unsupported_file_is_rejected(tmp_path):
    source = tmp_path / "archive.bin"
    source.write_bytes(b"binary")

    with pytest.raises(ValueError, match="Unsupported file type '.bin'"):
        file_reader.read_file(str(source))


def test_empty_file_is_rejected(tmp_path):
    source = tmp_path / "empty.txt"
    source.touch()

    with pytest.raises(ValueError, match="File is empty"):
        file_reader.read_file(str(source))


def test_empty_conversion_result_is_rejected(monkeypatch, tmp_path):
    source = tmp_path / "empty-result.md"
    source.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(file_reader, "_converter", StubConverter("  \n\n  "))

    with pytest.raises(ValueError, match="No readable content"):
        file_reader.read_file(str(source))


def test_converter_error_keeps_file_context(monkeypatch, tmp_path):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not-a-real-pdf")
    monkeypatch.setattr(file_reader, "_converter", StubConverter(error=RuntimeError("bad PDF")))

    with pytest.raises(ValueError, match="Could not read 'broken.pdf': bad PDF"):
        file_reader.read_file(str(source))


def test_long_content_is_truncated_with_metadata(monkeypatch, tmp_path):
    source = tmp_path / "long.txt"
    source.write_text("placeholder", encoding="utf-8")
    content = "x" * (file_reader.MAX_CHARS + 25)
    monkeypatch.setattr(file_reader, "_converter", StubConverter(content))

    result = file_reader.read_file(str(source))

    assert len(result["content"]) == file_reader.MAX_CHARS
    assert result["total_chars"] == file_reader.MAX_CHARS + 25
    assert result["is_truncated"] is True
