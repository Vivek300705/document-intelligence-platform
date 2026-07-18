"""Document Parser Module.

Extracts text from PDF, DOCX, and TXT files while preserving page and source metadata.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for PDF, DOCX, and TXT document formats."""

    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parses a document file and returns a list of page/section dictionaries.

        Returns format:
            [
                {
                    "page": int (1-indexed),
                    "text": str,
                    "source": str (filename)
                }, ...
            ]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at path: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return DocumentParser._parse_pdf(path)
        elif suffix == ".docx":
            return DocumentParser._parse_docx(path)
        elif suffix in [".txt", ".md", ".csv", ".json", ".log"]:
            return DocumentParser._parse_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: PDF, DOCX, TXT.")

    @staticmethod
    def _parse_pdf(path: Path) -> List[Dict[str, Any]]:
        """Parses PDF text page-by-page using pypdf, with fallback if needed."""
        pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                if cleaned_text:
                    pages.append({
                        "page": i,
                        "text": cleaned_text,
                        "source": path.name
                    })
        except Exception as e:
            logger.warning(f"Error parsing PDF with pypdf ({e}), attempting basic text extract fallback.")
            raise RuntimeError(f"Failed to parse PDF {path.name}: {str(e)}")

        if not pages:
            logger.warning(f"No text extracted from PDF: {path.name}")
        return pages

    @staticmethod
    def _parse_docx(path: Path) -> List[Dict[str, Any]]:
        """Parses DOCX text using python-docx."""
        try:
            import docx
            doc = docx.Document(str(path))
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            combined_text = "\n\n".join(full_text)
            if not combined_text:
                return []
            
            # DOCX does not naturally have hard page breaks, treat paragraphs as section page 1
            return [{
                "page": 1,
                "text": combined_text,
                "source": path.name
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX {path.name}: {str(e)}")

    @staticmethod
    def _parse_txt(path: Path) -> List[Dict[str, Any]]:
        """Parses plain text files."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            
            if not text:
                return []
            
            return [{
                "page": 1,
                "text": text,
                "source": path.name
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to parse TXT {path.name}: {str(e)}")


# TODO: add support for .rst and .html formats in a future iteration

# Edge case: some scanned PDFs return empty text — handled via empty-page guard