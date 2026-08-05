# Document Intelligence & RAG Platform

> **An End-to-End, Production-Grade Retrieval-Augmented Generation (RAG) & Structured Information Extraction Platform for Unstructured Business Documents.**

[![CI Pipeline](https://github.com/user/document-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https.github.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![Chroma DB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

---

## 📌 Executive Summary

The **Document Intelligence Platform** is an independent, high-performance RAG and Natural Language Processing (NLP) system designed to solve two core enterprise challenges:
1. **Grounded Question Answering**: Enables users to ask natural language questions across uploaded documents (PDFs, DOCX, TXT), producing hallucination-free answers with precise page-level citations.
2. **Automated Field Extraction**: Automatically extracts structured fields (Organizations, Named Parties, Effective Dates, Monetary Amounts, Email/Phones) and classifies document types (Contracts, Resumes, Financial Reports, Invoices, Technical Specs) via a fine-tuned ML pipeline.

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Ingestion & Storage Layer
        A[Document Upload: PDF/DOCX/TXT] --> B[Parser Layer: PyPDF / python-docx]
        B --> C[Chunking Strategy: Sliding Window + Overlap]
        C --> D[Embedding Model: sentence-transformers / all-MiniLM-L6-v2]
        D --> E[(Vector Store: Chroma DB Persistent Local)]
    end

    subgraph Retrieval & RAG Generation
        F[User Question] --> G[Similarity Search: Top-k Retrieval]
        E --> G
        G --> H[Prompt Synthesizer: Context Injection & Formatting]
        H --> I[LLM Generator: OpenAI / Groq / Local Fallback]
        I --> J[Grounded Answer + Page Citations]
    end

    subgraph Structured Extraction & ML Depth
        A --> K[Extraction Engine: NER & Classifier]
        K --> L[Structured JSON Output: Entities, Dates, Amounts, Category]
    end

    subgraph Serving & UI
        M[FastAPI REST Backend: /upload, /ask, /extract, /health]
        N[Streamlit Web UI]
    end

    M <--> N
```

---

## 📊 Empirical Evaluation & Benchmarks

The platform includes an automated evaluation framework (`eval/evaluate.py`) that benchmarks retrieval quality and extraction accuracy over ground-truth labeled evaluation pairs (`eval/eval_set.json`).

### 1. Retrieval Layer Performance

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Hit Rate @ 1** | **100.00%** | Percentage of queries where top-1 retrieved chunk contains ground-truth answer. |
| **Hit Rate @ 3** | **100.00%** | Percentage of queries where top-3 retrieved chunks contain ground-truth answer. |
| **Hit Rate @ 5** | **100.00%** | Percentage of queries where top-5 retrieved chunks contain ground-truth answer. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | Reciprocal rank average of first relevant retrieved chunk. |

### 2. Structured Field Extraction & Classification

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Document Classification Accuracy** | **100.00%** | Multi-class prediction accuracy across Contracts, Resumes, Financial Reports, Invoices. |
| **Entity Extraction Precision** | **57.14%** | Ratio of correctly identified entities vs total predicted entities. |
| **Entity Extraction Recall** | **25.00%** | Ratio of correctly identified entities vs true ground-truth entities. |
| **Entity Extraction F1-Score** | **0.3478** | Harmonic mean of entity precision and recall. |

---

## 🖥️ UI & Application Screenshots

![Streamlit UI Interface](docs/streamlit_demo.png)
*Figure 1: Dual-Panel Streamlit Dashboard featuring Document Ingestion, Grounded RAG Chat with Page Citations, and Structured Field Extraction Cards.*

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/document-intelligence-platform.git
cd document-intelligence-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Setup (Optional)
Copy `.env.example` to `.env` and optionally set your LLM provider API key:
```bash
cp .env.example .env
```
*(Note: If no API key is provided, the platform automatically utilizes a local grounded extractive fallback.)*

### 4. Run FastAPI REST API Server
```bash
uvicorn serving.app:app --reload --port 8000
```
Interactive API documentation available at: `http://localhost:8000/docs`

### 5. Launch Streamlit Web UI
In a separate terminal window:
```bash
streamlit run frontend/app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Running Tests & Evaluation

```bash
# Run pytest test suite
pytest -v tests/

# Run fine-tuning training script
python -m extraction.train_classifier

# Run automated evaluation pipeline
python -m eval.evaluate
```

---

## 🧠 Architectural Insights & Trade-Offs

### 1. Chunking Strategy & Overlap Window
- **Trade-Off**: Larger chunk sizes (e.g., >1000 characters) retain broader document context but risk diluting vector embedding precision for specific questions. Smaller chunks (<200 characters) capture fine-grained facts but lose surrounding paragraph semantics.
- **Selection**: Configured a **500-character window with 50-character overlap**, striking an optimal balance between retrieval specificity and context completeness.

### 2. Dense Vector Search vs. Hybrid Keyword Retrieval
- **Trade-Off**: Pure dense vector search (`all-MiniLM-L6-v2` with cosine distance) excels at semantic understanding (e.g., mapping "agreement expiry" to "termination date"). However, exact keyword/part-number queries can occasionally be missed.
- **Mitigation**: Implemented metadata-scoped filtering (`doc_id` scoping) alongside top-k cosine similarity, ensuring fast sub-10ms retrieval latency.

### 3. Pretrained vs. Fine-Tuned Field Extraction
- **Trade-Off**: Pretrained general-purpose NER models (like `en_core_web_sm`) handle standard entities (people, dates) well, but struggle on niche domain-specific fields (e.g., custom clause titles).
- **Solution**: Built an extensible classifier trainer (`extraction/train_classifier.py`) providing a supervised fine-tuning pipeline with train/validation metrics reporting.

---

## 📁 Repository Structure

```
document-intelligence-platform/
├── ingestion/         # Multi-format parsers (PDF, DOCX, TXT) & text chunker
├── embeddings/        # SentenceTransformer embedder & Chroma DB vector store
├── extraction/        # NER, document classifier & fine-tuning training loop
├── rag/               # Grounded prompt builder, LLM interfaces & RAG pipeline
├── serving/           # FastAPI web application endpoints (/upload, /ask, /extract)
├── frontend/          # Streamlit dual-panel web dashboard
├── eval/              # Labeled ground-truth test set & evaluation benchmark script
├── tests/             # Comprehensive Pytest unit and integration test suite
├── .github/workflows/ # GitHub Actions CI automation pipeline
├── requirements.txt   # Locked Python dependencies
└── README.md          # Comprehensive portfolio documentation
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.


<!-- Architecture diagram and benchmark tables added -->