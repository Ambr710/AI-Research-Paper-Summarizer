"""
pdf_utils.py
------------
Handles PDF upload -> raw text extraction.
"""

from pypdf import PdfReader
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF file given as bytes.

    Args:
        file_bytes: the raw bytes of the uploaded PDF file.

    Returns:
        A single string containing all extracted text, pages joined
        by double newlines.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n\n".join(pages_text)


def get_pdf_metadata(file_bytes: bytes) -> dict:
    """Return basic metadata (title, author, number of pages) if available."""
    reader = PdfReader(io.BytesIO(file_bytes))
    meta = reader.metadata or {}
    return {
        "title": meta.get("/Title", "Unknown"),
        "author": meta.get("/Author", "Unknown"),
        "num_pages": len(reader.pages),
    }
