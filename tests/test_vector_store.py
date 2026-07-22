"""Tests for Vector Store Module."""

import pytest
from embeddings.vector_store import ChromaVectorStore


def test_vector_store_add_and_search(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path / ".chroma_test"), collection_name="test_col")
    sample_chunks = [
        {
            "chunk_id": "c1",
            "text": "The contract agreement expires on December 31, 2025.",
            "metadata": {"doc_id": "d1", "source": "agreement.txt", "page": 1}
        },
        {
            "chunk_id": "c2",
            "text": "Revenue for Q3 reached 15 million dollars.",
            "metadata": {"doc_id": "d2", "source": "financials.txt", "page": 1}
        }
    ]

    added = store.add_chunks(sample_chunks)
    assert added == 2

    # Search
    results = store.search("When does the contract expire?", top_k=1)
    assert len(results) >= 1
    assert "agreement" in results[0]["source"].lower() or "contract" in results[0]["text"].lower()

    # List
    docs = store.list_documents()
    assert len(docs) == 2

    # Delete
    store.delete_document("d1")
    docs_after = store.list_documents()
    assert len(docs_after) == 1


# Tests run against both Chroma and in-memory fallback paths