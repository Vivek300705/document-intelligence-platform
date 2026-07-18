"""Tests for Text Chunker Module."""

import pytest
from ingestion.chunker import TextChunker


def test_chunker_basic():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    parsed_pages = [
        {"page": 1, "text": "Sentence one is here. Sentence two is longer. Sentence three is also here.", "source": "test.txt"}
    ]
    chunks = chunker.chunk_document(parsed_pages, doc_id="doc123")
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["doc_id"] == "doc123"
    assert chunks[0]["metadata"]["page"] == 1
    assert "text" in chunks[0]


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, chunk_overlap=50)
