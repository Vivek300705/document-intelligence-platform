"""Tests for Extraction Pipeline Module."""

import pytest
from extraction.pipeline import ExtractionPipeline


def test_extraction_pipeline():
    pipeline = ExtractionPipeline()
    sample_text = (
        "Nondisclosure Agreement between Acme Corp and Global Solutions. "
        "Signed on January 10, 2025. Total consideration of $50,000. "
        "Contact info@acmecorp.com."
    )
    result = pipeline.extract_from_text(sample_text, doc_name="agreement.txt")
    assert result["document_type"] == "Contract"
    assert result["classification_confidence"] > 0.0
    assert "entities" in result
    assert "organizations" in result["entities"]
    assert "amounts" in result["entities"]


# Tests validate classification accuracy and entity field presence