"""Tests for FastAPI Endpoints."""

import pytest
from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_documents_list_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    assert "documents" in response.json()


def test_upload_and_ask_flow(tmp_path):
    # Create sample text file
    sample_file = tmp_path / "test_doc.txt"
    sample_file.write_text(
        "Master Service Agreement. Effective date is June 1, 2025. "
        "The vendor promises 99.9% uptime SLA.",
        encoding="utf-8"
    )

    with open(sample_file, "rb") as f:
        upload_resp = client.post("/upload", files={"file": ("test_doc.txt", f, "text/plain")})

    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["status"] == "success"
    doc_id = upload_data["doc_id"]

    # Ask endpoint
    ask_resp = client.post("/ask", json={"question": "What is the uptime SLA?", "doc_id": doc_id})
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert "answer" in ask_data
    assert len(ask_data["citations"]) > 0

    # Extract endpoint
    ext_resp = client.post("/extract", json={"doc_id": doc_id})
    assert ext_resp.status_code == 200
    ext_data = ext_resp.json()
    assert "document_type" in ext_data


# TestClient runs full upload→ask→extract flow in a single test case