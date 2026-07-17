# ╔══════════════════════════════════════════════════════════════════════╗
# ║   TEXT RAG CHAT APP — Streamlit frontend + LangChain backend          ║
# ║   Text-only pipeline: PDF text extraction + OCR fallback → Chroma →   ║
# ║   Groq LLM. Sessions, sidebar history, single `main_retriever`,       ║
# ║   separate history store.                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

import os, uuid, tempfile
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]     = os.getenv("HF_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import streamlit as st
import fitz                      # pymupdf (used only for OCR page rendering)
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

# OCR fallback for scanned pages
import pytesseract
from PIL import Image
import io

# ── CONFIG ────────────────────────────────────────────────────────────────
CHROMA_DIR          = "./chroma_db"
COLLECTION_NAME     = "rag_docs"          # main document store (all sessions, filtered by session_id)
HISTORY_COLLECTION  = "chat_history"      # separate store, only for conversation memory
GROQ_MODEL          = "qwen/qwen3-32b"
TOP_K               = 5
HISTORY_K           = 4

st.set_page_config(page_title="Document Chat", layout="wide")

# ════════════════════════════════════════════════════════════════════════
# 1. EXTRACT TEXT CHUNKS  (+ OCR fallback for scanned pages)
# ════════════════════════════════════════════════════════════════════════
def ocr_page(pdf_path: str, page_num: int) -> str:
    """Render page to image and OCR it — used only when normal extraction is empty."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_text_chunks(pdf_path: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                text = ocr_page(pdf_path, page_num)      # OCR fallback (scanned page)
            if not text.strip():
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                chunks.append({
                    "type": "text", "content": chunk, "page": page_num + 1,
                    "chunk_index": i, "source": pdf_path,
                })
    return chunks


# ════════════════════════════════════════════════════════════════════════
# 4. EMBEDDINGS + VECTOR STORE  (unchanged, session_id added to metadata)
# ════════════════════════════════════════════════════════════════════════
class LangChainEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, lc_embedding):
        self.lc_embedding = lc_embedding
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.lc_embedding.embed_documents(input)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

lc_embeddings = get_embeddings()
chroma_embedding_fn = LangChainEmbeddingAdapter(lc_embeddings)


def build_vector_store(text_chunks, session_id, source_name,
                        collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=chroma_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(text_chunks):
        documents.append(chunk["content"])
        metadatas.append({"type": "text", "page": chunk["page"], "source": source_name,
                           "chunk_index": chunk["chunk_index"], "session_id": session_id})
        ids.append(f"{session_id}_text_{source_name}_{i}")

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return collection


@st.cache_resource
def get_main_retriever(k=TOP_K, score_threshold=0.3):
    """Single retriever instance reused for the whole app. Filter is set
    per-query via search_kwargs before invoking (see ask())."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=lc_embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )

main_retriever = get_main_retriever()   # single retriever for whole web page


# ════════════════════════════════════════════════════════════════════════
# 5. SEPARATE HISTORY STORE / RETRIEVER  (chat memory — kept isolated
#    from main_retriever and from the original RAG chain logic)
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_history_store():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=HISTORY_COLLECTION, embedding_function=chroma_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

history_store = get_history_store()


def save_turn_to_history(session_id, role, content):
    history_store.upsert(
        documents=[content],
        metadatas=[{"session_id": session_id, "role": role}],
        ids=[f"{session_id}_{uuid.uuid4().hex}"],
    )


def get_history_retriever(session_id, k=HISTORY_K):
    vectorstore = Chroma(
        collection_name=HISTORY_COLLECTION,
        embedding_function=lc_embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"session_id": {"$eq": session_id}}}
    )


def retrieve_relevant_history(session_id, query):
    try:
        retriever = get_history_retriever(session_id)
        docs = retriever.invoke(query)
        return "\n".join(d.page_content for d in docs)
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════
# 6. RAG CHAIN  (unchanged prompt / chain structure)
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, max_tokens=1024)

llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions from a document.
Use ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Relevant prior conversation (may be empty):
{history}

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[TEXT — page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain():
    return (
        RunnableParallel({
            "context": (lambda x: x["question"]) | main_retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        })
        | prompt | llm | StrOutputParser()
    )

rag_chain = build_rag_chain()


def ask(session_id: str, question: str) -> str:
    """Filter the single main_retriever to this session, then run the chain."""
    main_retriever.search_kwargs["filter"] = {"session_id": {"$eq": session_id}}
    history_text = retrieve_relevant_history(session_id, question)
    answer = rag_chain.invoke({"question": question, "history": history_text})
    save_turn_to_history(session_id, "user", question)
    save_turn_to_history(session_id, "assistant", answer)
    return answer


# ════════════════════════════════════════════════════════════════════════
# 7. PDF INGESTION PIPELINE (per uploaded file)
# ════════════════════════════════════════════════════════════════════════
def process_pdf(pdf_path: str, session_id: str, source_name: str, status):
    status.write(f"📄 Extracting text (with OCR fallback) from **{source_name}** …")
    text_chunks = extract_text_chunks(pdf_path)
    status.write(f"💾 Indexing **{source_name}** into vector store ({len(text_chunks)} chunks) …")
    build_vector_store(text_chunks, session_id, source_name)
    return len(text_chunks)


# ════════════════════════════════════════════════════════════════════════
# 8. STREAMLIT APP STATE  (persisted to disk so refresh/restart doesn't lose it)
# ════════════════════════════════════════════════════════════════════════
import json

SESSIONS_FILE = "sessions.json"


def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()   # session_id -> {name, messages, pdfs_ready, pdf_names}
if "current_session" not in st.session_state:
    # default to most recently created session, if any exist
    st.session_state.current_session = (
        next(reversed(st.session_state.sessions), None) if st.session_state.sessions else None
    )


def new_session():
    sid = uuid.uuid4().hex[:8]
    st.session_state.sessions[sid] = {
        "name": f"Chat {len(st.session_state.sessions) + 1}",
        "messages": [], "pdfs_ready": False, "pdf_names": [],
    }
    st.session_state.current_session = sid
    save_sessions(st.session_state.sessions)


def delete_session(sid: str):
    st.session_state.sessions.pop(sid, None)
    if st.session_state.current_session == sid:
        st.session_state.current_session = (
            next(reversed(st.session_state.sessions), None) if st.session_state.sessions else None
        )
    save_sessions(st.session_state.sessions)


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        new_session()
        st.rerun()
    st.divider()
    for sid_i, sess_i in st.session_state.sessions.items():
        label = sess_i["name"] + (" ✅" if sess_i["pdfs_ready"] else " ⏳")
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(label, key=f"sel_{sid_i}", use_container_width=True):
                st.session_state.current_session = sid_i
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sid_i}"):
                delete_session(sid_i)
                st.rerun()

if st.session_state.current_session is None:
    st.info("Create a new chat from the sidebar to get started.")
    st.stop()

sid = st.session_state.current_session
sess = st.session_state.sessions[sid]

st.title(sess["name"])

# ── PDF upload gate (required before chatting in this session) ─────────────
if not sess["pdfs_ready"]:
    st.subheader("Upload PDF(s) to start this chat")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"],
                              accept_multiple_files=True, key=f"uploader_{sid}")
    if files and st.button("Process PDFs"):
        status = st.status("Processing PDFs…", expanded=True)
        total_text = 0
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            t = process_pdf(tmp_path, sid, f.name, status)
            total_text += t
            sess["pdf_names"].append(f.name)
            os.unlink(tmp_path)
        status.update(label=f"✅ Processed {len(files)} PDF(s) — "
                             f"{total_text} text chunks indexed.",
                       state="complete")
        sess["pdfs_ready"] = True
        save_sessions(st.session_state.sessions)
        st.rerun()
    st.stop()

st.caption("📎 " + ", ".join(sess["pdf_names"]))

# ── Chat interface ───────────────────────────────────────────────────────
for msg in sess["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about your document(s)…")
if query:
    sess["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = ask(sid, query)
        st.markdown(answer)
    sess["messages"].append({"role": "assistant", "content": answer})
    save_sessions(st.session_state.sessions)