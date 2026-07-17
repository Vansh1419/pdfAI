# Document Chat — PDF Documentation Assistant

## Overview

Answering questions from technical PDFs (manuals, specs, reports) usually means manually skimming dozens of pages, re-reading the same sections across multiple documents, and losing track of which PDF had the answer.

To solve this, I built **Document Chat** — an internal chatbot that indexes uploaded PDFs per session and answers questions directly from their content, including scanned pages via OCR. It turns a manual "search every PDF" process into a single conversational query, with each chat session scoped to its own set of documents.

## Tech Stack

**Backend**

- Language: Python 3.11+
- Package Manager: pip
- Frameworks/Libraries: LangChain, LangChain-Groq, LangChain-HuggingFace, LangChain-Chroma, PyMuPDF, pdfplumber, pytesseract
- LLM: `qwen/qwen3-32b` (via Groq)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace)
- OCR: Tesseract (via pytesseract), used as a fallback for scanned/image-only pages
- Database: JSON file (session + chat metadata), ChromaDB (vector store — main documents + chat history, in separate collections)

**Frontend**

- Streamlit

## Setup Instructions

1. Clone the repository

```bash
git clone <repo-url>
cd pdfai
```

2. Set up environment

```bash
uv venv --python=3.12
source .venv/bin/activate
```

3. Install dependencies

```bash
uv add -r requirements.txt
sudo apt install tesseract-ocr   # macOS: brew install tesseract
```

4. Configure environment variables

```bash
cp .env.example .env
# then set GROQ_API_KEY
```

5. Run the app

```bash
streamlit run app.py
```

⚠️ Note: Do not delete `data/sessions.json` or `data/chroma_db/` while using the application — they store chat history and indexed document content required for context continuity.

## Architecture

The system follows a session-scoped retrieval pipeline to keep each chat's answers grounded only in the documents that chat owns:

1. **Session Creation** — user starts a new chat from the sidebar; a unique `session_id` is generated and persisted.
2. **PDF Upload** — user uploads one or more PDFs to that session before chatting is unlocked.
3. **Text Extraction** — text is pulled per page via `pdfplumber`; pages with no extractable text (scanned images) fall back to Tesseract OCR.
4. **Chunking & Indexing** — extracted text is split into overlapping chunks and upserted into a single shared ChromaDB collection, tagged with `session_id` in the metadata.
5. **Query Input** — user asks a question in the chat.
6. **Retrieval** — a single app-wide `main_retriever` fetches the top-k relevant chunks, filtered at query time to the active `session_id`.
7. **History Retrieval** — a separate, isolated chat-history retriever fetches relevant prior turns for the same session.
8. **Response Generation** — retrieved context, relevant history, and the question are combined into a prompt and passed to the Groq LLM to generate the answer.
9. **Persistence** — the question and answer are saved to the history vector store and the session's message log (JSON) for continuity across refreshes and restarts.

## Known Limitations

- Retrieval quality on scanned PDFs depends on OCR accuracy — poor scans produce poor chunks.
- `main_retriever`'s session filter is set on a shared instance right before each query, so it's not safe under truly concurrent requests to the same process.
- `sessions.json` is a single flat file — fine for single-user local use, not safe for concurrent multi-user writes.
- No automatic re-indexing if the same PDF is re-uploaded after being edited.
- Single LLM provider (Groq) — no fallback if rate-limited or down.
- No authentication — anyone with app access can see and open all sessions.

## Roadmap

- SQLite/Postgres-backed session store for multi-user, concurrency-safe deployment
- Per-request retriever filtering instead of mutating a shared `main_retriever` instance
- Source citation in answers (link back to exact page in the source PDF)
- Support additional file formats (docx, txt, images)
- Feedback loop (thumbs up/down) to improve retrieval ranking
- Fallback LLM provider for reliability
- Authentication and per-user session isolation