"""FastAPI Serving Layer.

Exposes REST APIs for document upload, grounded RAG question answering, and structured extraction.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker
from embeddings.vector_store import ChromaVectorStore
from rag.pipeline import RAGPipeline
from extraction.pipeline import ExtractionPipeline

app = FastAPI(
    title="Document Intelligence & RAG API",
    description="REST API for document ingestion, grounded RAG Q&A, and structured entity extraction.",
    version="1.0.0"
)

# Enable CORS for Streamlit / Frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]  # open CORS for Streamlit frontend communication,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
vector_store = ChromaVectorStore()
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
rag_pipeline = RAGPipeline(vector_store=vector_store)
extraction_pipeline = ExtractionPipeline()

# Temp directory for uploads
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Pydantic Request/Response Models ---

class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is the total contract value and expiration date?")
    doc_id: Optional[str] = Field(None, example="a1b2c3d4")
    top_k: int = Field(default=3, ge=1, le=10)


class ExtractRequest(BaseModel):
    doc_id: Optional[str] = Field(None, example="a1b2c3d4")
    text: Optional[str] = Field(None, example="Sample document text for extraction...")


# --- API Endpoints ---

@app.get("/health", tags=["Health"])
def health_check():
    """System health check endpoint."""
    docs = vector_store.list_documents()
    return {
        "status": "healthy",
        "indexed_documents_count": len(docs),
        "version": "1.0.0"
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    """Lists all indexed documents in the vector store."""
    return {"documents": vector_store.list_documents()}


@app.post("/upload", tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document (PDF, DOCX, TXT), runs text extraction, chunking, and Chroma indexing."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Allowed: PDF, DOCX, TXT."
        )

    # Save temp file
    temp_path = UPLOAD_DIR / file.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Parse document text
        parsed_pages = DocumentParser.parse_file(temp_path)
        if not parsed_pages:
            raise HTTPException(status_code=400, detail="Failed to extract any text from uploaded file.")

        # 2. Chunk document
        chunks = chunker.chunk_document(parsed_pages)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to generate text chunks from document.")

        doc_id = chunks[0]["metadata"]["doc_id"]

        # 3. Add to Vector Store
        added_count = vector_store.add_chunks(chunks)

        return {
            "status": "success",
            "message": f"Successfully processed and indexed {file.filename}",
            "doc_id": doc_id,
            "filename": file.filename,
            "page_count": len(parsed_pages),
            "chunk_count": added_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/ask", tags=["RAG Query"])
def ask_question(request: QuestionRequest):
    """Executes grounded RAG question answering over indexed document chunks."""
    try:
        response = rag_pipeline.answer_question(
            query=request.question,
            doc_id=request.doc_id,
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution error: {str(e)}")


@app.post("/extract", tags=["Extraction"])
def extract_fields(request: ExtractRequest):
    """Extracts structured entities and document classification."""
    target_text = ""
    doc_name = "custom_document"

    if request.text and request.text.strip():
        target_text = request.text.strip()
    elif request.doc_id:
        # Search all chunks for doc_id
        doc_chunks = vector_store.search(query="document summary", top_k=10, doc_id=request.doc_id)
        if not doc_chunks:
            raise HTTPException(status_code=44, detail=f"No document chunks found for doc_id '{request.doc_id}'.")
        
        doc_name = doc_chunks[0].get("source", "document")
        target_text = "\n\n".join([c["text"] for c in doc_chunks])
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'doc_id' or 'text' payload.")

    try:
        extraction_result = extraction_pipeline.extract_from_text(target_text, doc_name=doc_name)
        return extraction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction execution error: {str(e)}")


# FastAPI chosen for auto-generated OpenAPI docs at /docs — great for demos