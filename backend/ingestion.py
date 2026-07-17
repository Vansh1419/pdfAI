"""PDF -> text chunks, with an OCR fallback for scanned (image-only) pages."""

import io

import fitz  # pymupdf, used only to rasterize pages for OCR
import pdfplumber
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, OCR_DPI


def ocr_page(pdf_path: str, page_num: int) -> str:
    """Render a single page to an image and OCR it. Only called when normal
    text extraction returns nothing (i.e. the page is a scanned image)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_text_chunks(pdf_path: str) -> list[dict]:
    """Extract and chunk text from every page of a PDF, falling back to OCR
    on pages with no extractable text layer."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                text = ocr_page(pdf_path, page_num)
            if not text.strip():
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                chunks.append({
                    "type": "text",
                    "content": chunk,
                    "page": page_num + 1,
                    "chunk_index": i,
                    "source": pdf_path,
                })
    return chunks
