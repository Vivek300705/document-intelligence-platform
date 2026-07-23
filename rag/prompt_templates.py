"""RAG Prompt Templates.

Constructs grounded prompt context and formatting for LLM generation.
"""

from typing import List, Dict, Any


class RAGPromptTemplate:
    """Prompt builder for grounded document Question-Answering."""

    SYSTEM_PROMPT = (
        "You are an expert Document Intelligence AI assistant. "
        "Answer the user's question accurately and concisely using strictly the provided context chunks below.\n"
        "If the information needed to answer the question is not present in the context, explicitly state: "
        "'Based on the provided document, I cannot find sufficient information to answer your question.'\n"
        "Always cite the source document and page number for facts you reference."
    )

    @staticmethod
    def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats query and retrieved chunks into a clean prompt context string."""
        context_str = RAGPromptTemplate.format_chunks_context(retrieved_chunks)

        prompt = (
            f"--- CONTEXT START ---\n"
            f"{context_str}\n"
            f"--- CONTEXT END ---\n\n"
            f"QUESTION: {query}\n\n"
            f"Please provide a grounded answer with page citations based solely on the context above:"
        )
        return prompt

    @staticmethod
    def format_chunks_context(chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunk dictionaries into numbered context blocks."""
        if not chunks:
            return "No relevant context chunks found."

        formatted_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page", 1)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            block = (
                f"[Chunk {i} | Source: {source} | Page: {page} | Similarity Score: {score}]\n"
                f"{text}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)


# Prompt design: explicit 'CONTEXT START/END' delimiters reduce hallucination rate