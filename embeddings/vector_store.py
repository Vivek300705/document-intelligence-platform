"""Vector Store Module.

Manages persistent Chroma vector storage, document indexing, and similarity retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from embeddings.embedder import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Wrapper for Chroma DB vector store with pure Python cosine vector store fallback."""

    def __init__(
        self,
        persist_dir: str = ".chroma_db",
        collection_name: str = "document_intelligence",
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedding_generator or EmbeddingGenerator()

        self._client = None
        self._collection = None
        self._fallback_mode = False
        self._memory_store = []  # List of {"id": ..., "embedding": ..., "text": ..., "metadata": ...}

    @property
    def client(self):
        """Lazy load Chroma Client with fallback."""
        if self._client is None and not self._fallback_mode:
            try:
                import chromadb
                logger.info(f"Initializing Persistent Chroma client at {self.persist_dir}")
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            except Exception as e:
                logger.warning(f"ChromaDB not installed ({e}). Using in-memory cosine vector store.")
                self._fallback_mode = True
        return self._client

    @property
    def collection(self):
        """Lazy load or create Chroma collection."""
        if self.client is not None:
            if self._collection is None:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            return self._collection
        return None


    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds and indexes document chunks into Chroma DB or fallback store."""
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        if self.collection is not None:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        else:
            for c_id, emb, txt, meta in zip(ids, embeddings, texts, metadatas):
                self._memory_store.append({
                    "id": c_id,
                    "embedding": emb,
                    "text": txt,
                    "metadata": meta
                })

        logger.info(f"Successfully indexed {len(chunks)} chunks into vector store.")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs top-k cosine similarity search over chunk embeddings."""
        if not query.strip():
            return []

        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            return []

        if self.collection is not None:
            where_clause = {"doc_id": doc_id} if doc_id else None
            total_count = self.collection.count()
            if total_count == 0:
                return []

            n_results = min(top_k, total_count)  # prevent n_results > collection size error
            kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
            if where_clause:
                kwargs["where"] = where_clause

            try:
                results = self.collection.query(**kwargs)
            except Exception as e:
                logger.error(f"Chroma query failed ({e}). Returning empty results.")
                return []

            formatted_results = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

                for i in range(len(docs)):
                    dist = distances[i]
                    sim_score = max(0.0, min(1.0, 1.0 - dist))
                    formatted_results.append({
                        "chunk_id": ids[i],
                        "text": docs[i],
                        "score": round(sim_score, 4),
                        "distance": round(dist, 4),
                        "source": metas[i].get("source", "Unknown"),
                        "page": metas[i].get("page", 1),
                        "doc_id": metas[i].get("doc_id", ""),
                        "metadata": metas[i]
                    })
            return formatted_results

        # In-Memory Cosine Similarity Fallback Calculation
        import math
        filtered_items = self._memory_store
        if doc_id:
            filtered_items = [item for item in filtered_items if item["metadata"].get("doc_id") == doc_id]

        if not filtered_items:
            return []

        scored_items = []
        for item in filtered_items:
            emb = item["embedding"]
            # Cosine similarity calculation dot(a, b) / (||a|| * ||b||)
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_q = math.sqrt(sum(a * a for a in query_embedding))
            norm_e = math.sqrt(sum(b * b for b in emb))
            sim = (dot / (norm_q * norm_e)) if (norm_q * norm_e) > 0 else 0.0
            sim_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))

            scored_items.append({
                "chunk_id": item["id"],
                "text": item["text"],
                "score": round(sim_score, 4),
                "distance": round(1.0 - sim_score, 4),
                "source": item["metadata"].get("source", "Unknown"),
                "page": item["metadata"].get("page", 1),
                "doc_id": item["metadata"].get("doc_id", ""),
                "metadata": item["metadata"]
            })

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns unique documents stored in vector database."""
        docs_summary = {}
        if self.collection is not None:
            all_data = self.collection.get(include=["metadatas"])
            metas = all_data.get("metadatas", [])
            for m in metas:
                if not m:
                    continue
                d_id = m.get("doc_id")
                if d_id and d_id not in docs_summary:
                    docs_summary[d_id] = {
                        "doc_id": d_id,
                        "source": m.get("source", "Unknown"),
                        "chunk_count": 0
                    }
                if d_id in docs_summary:
                    docs_summary[d_id]["chunk_count"] += 1
        else:
            for item in self._memory_store:
                m = item["metadata"]
                d_id = m.get("doc_id")
                if d_id and d_id not in docs_summary:
                    docs_summary[d_id] = {
                        "doc_id": d_id,
                        "source": m.get("source", "Unknown"),
                        "chunk_count": 0
                    }
                if d_id in docs_summary:
                    docs_summary[d_id]["chunk_count"] += 1

        return list(docs_summary.values())

    def delete_document(self, doc_id: str) -> bool:
        """Deletes all chunks associated with a document ID."""
        try:
            if self.collection is not None:
                self.collection.delete(where={"doc_id": doc_id})
            else:
                self._memory_store = [item for item in self._memory_store if item["metadata"].get("doc_id") != doc_id]
            logger.info(f"Deleted document ID '{doc_id}' from vector store.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{doc_id}': {e}")
            return False



# Chroma PersistentClient stores embeddings on disk — survives app restarts

# In-memory cosine store activates when chromadb package is not installed