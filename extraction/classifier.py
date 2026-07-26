"""Document Classifier Module.

Classifies input text into document types (Contract, Resume, Financial Report, Invoice, Technical Document).
"""

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENT_TYPES = ["Contract", "Resume", "Financial Report", "Invoice", "Technical Document"]


class DocumentClassifier:
    """Document classification model using ML pipeline & keyword heuristics."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.vectorizer = None
        self.model = None
        self.categories = DOCUMENT_TYPES

        self._initialize_heuristics()

    def _initialize_heuristics(self):
        """Keyword heuristic patterns for robust fallback classification."""
        self.keywords = {
            "Contract": ["agreement", "party", "shall", "covenant", "terms", "termination", "liability", "indemnify", "jurisdiction", "contract"],
            "Resume": ["experience", "education", "skills", "summary", "projects", "work history", "bachelor", "master", "university", "resume"],
            "Financial Report": ["revenue", "ebitda", "net income", "fiscal year", "balance sheet", "quarter", "margin", "cash flow", "assets", "financial"],
            "Invoice": ["invoice", "bill to", "subtotal", "tax", "due date", "amount due", "payment terms", "description", "unit price", "po number"],
            "Technical Document": ["architecture", "api", "function", "endpoint", "database", "python", "system", "component", "deployment", "implementation"]
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts document class and confidence scores.

        Args:
            text: Input document text string.

        Returns:
            Dict containing predicted_label, confidence, and score breakdown.
        """
        if not text or not text.strip():
            return {
                "predicted_label": "General Document",
                "confidence": 1.0,
                "probabilities": {cat: 0.2 for cat in self.categories}
            }

        # Check if trained ML model is available
        if self.model and self.vectorizer:
            try:
                X = self.vectorizer.transform([text])
                probs = self.model.predict_proba(X)[0]
                best_idx = probs.argmax()
                best_cat = self.model.classes_[best_idx]
                prob_dict = {str(c): round(float(p), 4) for c, p in zip(self.model.classes_, probs)}

                return {
                    "predicted_label": str(best_cat),
                    "confidence": round(float(probs[best_idx]), 4),
                    "probabilities": prob_dict
                }
            except Exception as e:
                logger.warning(f"ML model prediction failed ({e}). Using keyword scoring.")

        # Heuristic scoring fallback
        text_lower = text.lower()
        scores = {}
        for cat, kw_list in self.keywords.items():
            cat_score = sum(text_lower.count(kw) for kw in kw_list)
            scores[cat] = cat_score

        total_score = sum(scores.values())
        if total_score == 0:
            return {
                "predicted_label": "Technical Document",
                "confidence": 0.5,
                "probabilities": {cat: round(1.0 / len(self.categories), 4) for cat in self.categories}
            }

        probabilities = {cat: round(score / total_score, 4) for cat, score in scores.items()}
        best_cat = max(scores, key=scores.get)

        return {
            "predicted_label": best_cat,
            "confidence": probabilities[best_cat],
            "probabilities": probabilities
        }


# Keyword heuristic covers zero-shot case when ML model artifacts not present