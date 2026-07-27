"""Structured Extraction Pipeline.

Integrates Named Entity Recognition (NER) and Document Type Classification into a unified extraction engine.
"""

from typing import Dict, Any, Optional
import logging

from extraction.ner import EntityExtractor
from extraction.classifier import DocumentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Unified engine for structured information extraction from documents."""

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        classifier: Optional[DocumentClassifier] = None
    ):
        """Initializes NER and Classification modules."""
        self.ner = entity_extractor or EntityExtractor()
        self.classifier = classifier or DocumentClassifier()

    def extract_from_text(self, text: str, doc_name: str = "document") -> Dict[str, Any]:
        """Extracts structured entities and document classification from text.

        Args:
            text: Input document text.
            doc_name: Filename or document identifier.

        Returns:
            Structured JSON dict containing classification and entity details.
        """
        logger.info(f"Running structured extraction for '{doc_name}' ({len(text)} chars)")

        # 1. Document Type Classification
        cls_result = self.classifier.predict(text)

        # 2. Entity Extraction
        entities = self.ner.extract_entities(text)

        # 3. Summary Statistics
        total_entities = sum(len(v) for v in entities.values())

        return {
            "document_name": doc_name,
            "document_type": cls_result.get("predicted_label", "General Document"),
            "classification_confidence": cls_result.get("confidence", 0.0),
            "probabilities": cls_result.get("probabilities", {}),
            "entities": entities,
            "summary_stats": {
                "total_entities_found": total_entities,
                "character_count": len(text)
            }
        }


# ExtractionPipeline is the public API — combines NER + classifier in one call