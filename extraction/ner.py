"""Named Entity Recognition (NER) Module.

Extracts key entities (Organizations, People, Dates, Monetary Amounts, Email/Phones) from text.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Named Entity Recognizer using spaCy and hybrid pattern matchers."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        """Lazy loads spaCy model with fallback pattern matching."""
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load(self.model_name)
                    logger.info(f"Loaded spaCy model: {self.model_name}")
                except OSError:
                    logger.warning(f"spaCy model {self.model_name} not found. Using blank English model.")
                    self._nlp = spacy.blank("en")
            except ImportError:
                logger.warning("spaCy not installed. Using rule-based fallback NER.")
                self._nlp = None
        return self._nlp

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extracts structured entities from text.

        Returns:
            Dict of entity categories: organizations, persons, dates, amounts, emails, phones.
        """
        if not text.strip():
            return {
                "organizations": [],
                "persons": [],
                "dates": [],
                "amounts": [],
                "emails": [],
                "phones": []
            }

        entities = {
            "organizations": [],
            "persons": [],
            "dates": [],
            "amounts": [],
            "emails": self._extract_regex(text, r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phones": self._extract_regex(text, r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
        }

        # spaCy extraction
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                val = ent.text.strip()
                if not val or len(val) < 2:
                    continue

                if ent.label_ in ["ORG"]:
                    if val not in [x["text"] for x in entities["organizations"]]:
                        entities["organizations"].append({"text": val, "label": "ORG"})
                elif ent.label_ in ["PERSON"]:
                    if val not in [x["text"] for x in entities["persons"]]:
                        entities["persons"].append({"text": val, "label": "PERSON"})
                elif ent.label_ in ["DATE", "TIME"]:
                    if val not in [x["text"] for x in entities["dates"]]:
                        entities["dates"].append({"text": val, "label": "DATE"})
                elif ent.label_ in ["MONEY", "CARDINAL", "QUANTITY"]:
                    if val not in [x["text"] for x in entities["amounts"]]:
                        entities["amounts"].append({"text": val, "label": "AMOUNT"})

        # Regex fallback enrichers if lists are empty
        if not entities["dates"]:
            date_matches = self._extract_regex(
                text,
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b"
            )
            entities["dates"] = [{"text": d, "label": "DATE"} for d in date_matches]

        if not entities["amounts"]:
            amount_matches = self._extract_regex(text, r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?")
            entities["amounts"] = [{"text": a, "label": "AMOUNT"} for a in amount_matches]

        return entities

    def _extract_regex(self, text: str, pattern: str) -> List[str]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        seen = set()
        result = []
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result


# Hybrid approach: spaCy model for context-aware NER, regex for amounts/emails