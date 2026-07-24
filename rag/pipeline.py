"""RAG Pipeline Orchestrator.

Combines retrieval, prompt building, LLM synthesis, and citation extraction into a single query pipeline.
"""

from typing import List, Dict, Any, Optional
import logging

from embeddings.vector_store import ChromaVectorStore
from rag.prompt_templates import RAGPromptTemplate
from rag.generator import LLMGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Full RAG Pipeline for question answering over indexed documents."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        generator: Optional[LLMGenerator] = None
    ):
        """Initializes RAG Pipeline components."""
        self.vector_store = vector_store or ChromaVectorStore()
        self.generator = generator or LLMGenerator()

    def answer_question(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Executes retrieval and generation pipeline for a user question.

        Args:
            query: User natural language question.
            doc_id: Optional document ID scope filter.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            Dict containing answer, formatted citations, and retrieved chunk details.
        """
        logger.info(f"RAG query: '{query}' (doc_id: {doc_id}, top_k: {top_k})")

        # Step 1: Retrieval
        retrieved_chunks = self.vector_store.search(query=query, top_k=top_k, doc_id=doc_id)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "Based on the provided document, I cannot find sufficient information to answer your question.",
                "citations": [],
                "retrieved_chunks": [],
                "chunk_count": 0
            }

        # Step 2: Prompt Formatting
        system_prompt = RAGPromptTemplate.SYSTEM_PROMPT
        user_prompt = RAGPromptTemplate.build_prompt(query, retrieved_chunks)

        # Step 3: Generation
        raw_answer = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_chunks=retrieved_chunks
        )

        # Step 4: Extract and format citations
        citations = []
        seen_citations = set()
        for chunk in retrieved_chunks:
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", 1)
            score = chunk.get("score", 0.0)
            cit_key = (source, page)

            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append({
                    "source": source,
                    "page": page,
                    "similarity_score": score,
                    "snippet": chunk.get("text", "")[:150] + "..."
                })

        return {
            "query": query,
            "answer": raw_answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "chunk_count": len(retrieved_chunks)
        }


# Citations de-duplicated by (source, page) to avoid repeating same reference