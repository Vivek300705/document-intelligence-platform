"""Chunking Module.

Splits parsed document pages into overlapping text chunks while tracking document and page metadata.
"""

import uuid
from typing import List, Dict, Any


class TextChunker:
    """Sliding-window text chunker for RAG ingestion."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50  # tunable hyperparameters):
        """Initializes chunker with character/word length parameters.

        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Overlap character count between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, parsed_pages: List[Dict[str, Any]], doc_id: str = None
    ) -> List[Dict[str, Any]]:
        """Splits document pages into chunks with metadata.

        Args:
            parsed_pages: List of page dicts from DocumentParser.
            doc_id: Unique identifier for the document (generated if None).

        Returns:
            List of chunk objects containing chunk_id, text, and metadata.
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]

        all_chunks = []
        global_chunk_idx = 0

        for page_data in parsed_pages:
            text = page_data.get("text", "").strip()
            page_num = page_data.get("page", 1)
            source = page_data.get("source", "unknown")

            if not text:
                continue

            page_chunks = self._split_text(text)
            for sub_idx, chunk_text in enumerate(page_chunks):
                global_chunk_idx += 1
                chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "source": source,
                        "page": page_num,
                        "chunk_index": global_chunk_idx,
                        "char_count": len(chunk_text)
                    }
                })

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Splits text using sliding window with overlap, prioritizing natural paragraph/sentence breaks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Find nearest space or newline to avoid cutting in middle of words
            boundary = text.rfind(" ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = end

            chunk_text = text[start:boundary].strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Advance start by boundary - overlap
            start = max(start + 1, boundary - self.chunk_overlap)

        return chunks
