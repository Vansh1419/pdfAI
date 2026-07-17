"""Central configuration. No Streamlit or LangChain imports here on purpose —
this module must stay importable by scripts, tests, and the app alike."""

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ── Storage paths ────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(DATA_DIR, "chroma_db"))
SESSIONS_FILE = os.getenv("SESSIONS_FILE", os.path.join(DATA_DIR, "sessions.json"))

# ── Vector store collections ────────────────────────────────────────────
COLLECTION_NAME = "rag_docs"          # main document store, filtered by session_id
HISTORY_COLLECTION = "chat_history"   # separate store, conversation memory only

# ── Models ───────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Retrieval / chunking tunables ───────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
HISTORY_K = int(os.getenv("HISTORY_K", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
