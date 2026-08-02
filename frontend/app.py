"""Streamlit Frontend Application.

Interactive Dashboard for Document Ingestion, Grounded RAG Chat, and Structured Entity Extraction.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load .env keys (OPENAI_API_KEY, GROQ_API_KEY) from project root
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=root_dir / ".env", override=True)
except ImportError:
    pass

import streamlit as st
import os
import tempfile
import json
import pandas as pd

# Inject Streamlit Cloud secrets into os.environ (no-op locally if secrets not set)
try:
    for key in ["GROQ_API_KEY", "OPENAI_API_KEY"]:
        val = st.secrets.get(key) or st.secrets.get("llm", {}).get(key)
        if val:
            os.environ[key] = val
except Exception:
    pass

from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker
from embeddings.vector_store import ChromaVectorStore
from rag.pipeline import RAGPipeline
from extraction.pipeline import ExtractionPipeline


# Page setup
st.set_page_config(
    page_title="Document Intelligence & RAG Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .citation-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .badge-contract { background-color: #DBEAFE; color: #1E40AF; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-resume { background-color: #DCFCE7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-financial { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-invoice { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services(version=3):
    """Initializes and caches singleton services (embedding + extraction only)."""
    vector_store = ChromaVectorStore()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    extraction_pipeline = ExtractionPipeline()
    return vector_store, chunker, extraction_pipeline


vector_store, chunker, extraction_pipeline = get_services(version=3)

# --- Sidebar ---
st.sidebar.title("📄 Document Hub")
st.sidebar.markdown("Upload documents & manage indexed database.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Document (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt", "md"],
    help="Upload contracts, resumes, reports, or invoices for analysis."
)

if uploaded_file is not None:
    if st.sidebar.button("⚡ Process & Index Document", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                parsed_pages = DocumentParser.parse_file(tmp_path)
                # Ensure original source name is preserved
                for page in parsed_pages:
                    page["source"] = uploaded_file.name

                chunks = chunker.chunk_document(parsed_pages)
                doc_id = chunks[0]["metadata"]["doc_id"]
                added_count = vector_store.add_chunks(chunks)

                st.sidebar.success(f"Indexed **{added_count} chunks** (Doc ID: `{doc_id}`)!")
                st.session_state["selected_doc_id"] = doc_id
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

# List indexed documents
indexed_docs = vector_store.list_documents()
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Indexed Documents")

if indexed_docs:
    doc_options = {"All Documents": None}
    for d in indexed_docs:
        label = f"{d['source']} ({d['chunk_count']} chunks)"
        doc_options[label] = d["doc_id"]

    selected_label = st.sidebar.selectbox("Filter Scope:", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_label]
else:
    st.sidebar.info("No documents indexed yet. Upload a file above!")
    selected_doc_id = None

# --- LLM API Key Input ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 LLM API Key (Optional)")
st.sidebar.caption("Enter a Groq or OpenAI key for full AI-generated answers. Leave blank to use the built-in local summarizer.")

llm_key_input = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="gsk_... (Groq) or sk-... (OpenAI)",
    help="Get a free Groq key at console.groq.com"
)

# Dynamically build RAG pipeline — UI key box overrides .env; if empty, .env keys are used as-is.
def get_rag_pipeline(api_key: str = "") -> RAGPipeline:
    """Builds a fresh RAG pipeline using the provided API key or falls back to .env."""
    import os
    from rag.generator import LLMGenerator
    if api_key.startswith("gsk_"):
        # UI Groq key takes priority
        os.environ["GROQ_API_KEY"] = api_key
        os.environ.pop("OPENAI_API_KEY", None)
    elif api_key.startswith("sk-"):
        # UI OpenAI key takes priority
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.pop("GROQ_API_KEY", None)
    # else: leave os.environ untouched — .env keys already loaded at startup will be used
    generator = LLMGenerator()
    return RAGPipeline(vector_store=vector_store, generator=generator)

rag_pipeline = get_rag_pipeline(llm_key_input)

# Detect which provider is actually active (from UI override or .env)
import os as _os
_active_groq = _os.getenv("GROQ_API_KEY", "").startswith("gsk_")
_active_openai = _os.getenv("OPENAI_API_KEY", "").startswith("sk-")
provider_label = "🟢 Groq LLaMA" if _active_groq else \
                 "🟢 OpenAI GPT" if _active_openai else \
                 "🟡 Local Summarizer (no key)"
st.sidebar.caption(f"Mode: **{provider_label}**")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Chroma Database**: Persistent (`.chroma_db`)  \n**Embedding Model**: `all-MiniLM-L6-v2`")


# --- Main Dashboard Header ---
st.markdown('<div class="main-header">Document Intelligence & RAG Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded Question Answering & Automatic Structured Field Extraction</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_rag, tab_extract, tab_store = st.tabs(["💬 Grounded RAG Chat", "🔍 Structured Extraction", "📊 Vector Database Explorer"])

# --- TAB 1: Grounded RAG Chat ---
with tab_rag:
    st.subheader("Natural Language Question Answering")
    st.caption("Answers are grounded exclusively in retrieved document chunks with source citations.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display past chat
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("citations"):
                with st.expander("📌 Source Citations & Context Chunks"):
                    for cit in chat["citations"]:
                        st.markdown(
                            f"**Document**: `{cit['source']}` | **Page**: `{cit['page']}` | "
                            f"**Similarity**: `{cit['similarity_score']:.2%}`"
                        )
                        st.caption(f"\"{cit['snippet']}\"")

    # Chat Input Box
    query = st.chat_input("Ask a question about your uploaded documents...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching vectors & generating grounded response..."):
            res = rag_pipeline.answer_question(query=query, doc_id=selected_doc_id, top_k=8)
            answer = res["answer"]
            citations = res["citations"]

        with st.chat_message("assistant"):
            st.write(answer)
            if citations:
                with st.expander("📌 Source Citations & Context Chunks"):
                    for cit in citations:
                        st.markdown(
                            f"**Document**: `{cit['source']}` | **Page**: `{cit['page']}` | "
                            f"**Similarity**: `{cit['similarity_score']:.2%}`"
                        )
                        st.caption(f"\"{cit['snippet']}\"")

        st.session_state["chat_history"].append({
            "question": query,
            "answer": answer,
            "citations": citations
        })

# --- TAB 2: Structured Entity Extraction ---
with tab_extract:
    st.subheader("Automated Document Classification & NER")
    st.caption("Pulls key entities (Organizations, People, Dates, Amounts, Emails) and document category.")

    if selected_doc_id:
        if st.button("🚀 Extract Structured Fields for Selected Document"):
            with st.spinner("Running NLP Extraction Pipeline..."):
                doc_chunks = vector_store.search(query="document overview summary", top_k=10, doc_id=selected_doc_id)
                full_text = "\n\n".join([c["text"] for c in doc_chunks])
                doc_name = doc_chunks[0]["source"] if doc_chunks else "document"

                result = extraction_pipeline.extract_from_text(full_text, doc_name=doc_name)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### Document Classification")
                doc_type = result["document_type"]
                conf = result["classification_confidence"]

                st.metric("Predicted Type", doc_type)
                st.metric("Model Confidence", f"{conf:.1%}")

                st.markdown("**Probabilities:**")
                for cat, p in result.get("probabilities", {}).items():
                    st.progress(float(p), text=f"{cat}: {p:.1%}")

            with col2:
                st.markdown("### Extracted Named Entities")
                entities = result["entities"]

                e1, e2, e3 = st.columns(3)
                e1.metric("Organizations", len(entities.get("organizations", [])))
                e2.metric("People / Parties", len(entities.get("persons", [])))
                e3.metric("Dates Found", len(entities.get("dates", [])))

                st.markdown("---")
                t1, t2, t3, t4 = st.tabs(["🏢 Organizations", "👤 People", "📅 Dates & Amounts", "📋 Raw JSON"])

                with t1:
                    orgs = [x["text"] for x in entities.get("organizations", [])]
                    if orgs:
                        st.write(orgs)
                    else:
                        st.info("No organization entities detected.")

                with t2:
                    persons = [x["text"] for x in entities.get("persons", [])]
                    if persons:
                        st.write(persons)
                    else:
                        st.info("No person entities detected.")

                with t3:
                    st.write("**Dates:**", [x["text"] for x in entities.get("dates", [])])
                    st.write("**Monetary Amounts:**", [x["text"] for x in entities.get("amounts", [])])
                    st.write("**Emails:**", entities.get("emails", []))

                with t4:
                    st.json(result)
    else:
        st.info("Please select or upload a document in the sidebar to run structured field extraction.")


# --- TAB 3: Vector Database Explorer ---
with tab_store:
    st.subheader("Chroma Vector Store Index")
    docs = vector_store.list_documents()

    if docs:
        df = pd.DataFrame(docs)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Clear Vector Database"):
            import chromadb
            client = vector_store.client
            client.delete_collection(vector_store.collection_name)
            st.success("Vector store collection reset!")
            st.rerun()
    else:
        st.info("Vector database is empty.")


# st.cache_resource caches vector store and extraction pipeline across reruns

# UI API key input overrides .env keys — useful for live demos without redeployment

# st.secrets injected into os.environ at startup for Streamlit Cloud compatibility