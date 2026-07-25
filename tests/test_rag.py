"""Tests for RAG Pipeline Module."""

import pytest
from rag.prompt_templates import RAGPromptTemplate
from rag.pipeline import RAGPipeline
from embeddings.vector_store import ChromaVectorStore


def test_prompt_template_builder():
    chunks = [
        {"source": "doc1.txt", "page": 2, "score": 0.95, "text": "Sample context snippet."}
    ]
    prompt = RAGPromptTemplate.build_prompt("What is in doc1?", chunks)
    assert "Sample context snippet" in prompt
    assert "QUESTION: What is in doc1?" in prompt
    assert "[Chunk 1 | Source: doc1.txt | Page: 2" in prompt


def test_rag_pipeline_end_to_end(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path / ".chroma_rag_test"), collection_name="test_rag")
    store.add_chunks([
        {
            "chunk_id": "c100",
            "text": "The project budget is capped at $75,000.",
            "metadata": {"doc_id": "proj1", "source": "budget.txt", "page": 1}
        }
    ])

    pipeline = RAGPipeline(vector_store=store)
    res = pipeline.answer_question("What is the project budget?", doc_id="proj1")
    assert "answer" in res
    assert len(res["citations"]) > 0
    assert res["citations"][0]["source"] == "budget.txt"


# End-to-end test uses in-memory store to avoid requiring Chroma install in CI