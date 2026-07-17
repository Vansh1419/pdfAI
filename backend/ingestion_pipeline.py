"""Orchestrates a single uploaded PDF through extraction -> indexing.
Kept UI-agnostic: pass an optional on_progress(str) callback instead of
importing Streamlit here."""

from typing import Callable, Optional

from backend.ingestion import extract_text_chunks
from backend.vectorstore import build_vector_store


def process_pdf(
    pdf_path: str,
    session_id: str,
    source_name: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Extract text (with OCR fallback), chunk it, and index it into the
    session's slice of the main vector store. Returns the chunk count."""
    if on_progress:
        on_progress(f"📄 Extracting text (with OCR fallback) from **{source_name}** …")
    text_chunks = extract_text_chunks(pdf_path)

    if on_progress:
        on_progress(f"💾 Indexing **{source_name}** into vector store ({len(text_chunks)} chunks) …")
    build_vector_store(text_chunks, session_id, source_name)

    return len(text_chunks)
