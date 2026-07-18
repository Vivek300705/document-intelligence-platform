"""Tests for Document Parser Module."""

import pytest
from pathlib import Path
from ingestion.parser import DocumentParser


def test_parse_txt_file(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a sample document for testing parser functionality.", encoding="utf-8")

    pages = DocumentParser.parse_file(txt_file)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "sample document" in pages[0]["text"]
    assert pages[0]["source"] == "sample.txt"


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        DocumentParser.parse_file("non_existent_file.pdf")


def test_unsupported_format(tmp_path):
    invalid_file = tmp_path / "sample.xyz"
    invalid_file.write_text("test", encoding="utf-8")
    with pytest.raises(ValueError):
        DocumentParser.parse_file(invalid_file)


# Additional edge case tests added after manual testing with varied PDFs