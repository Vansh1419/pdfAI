"""Chat-history vector store — kept isolated from the main document
retriever/collection. Used only to give the LLM relevant prior turns."""

import uuid
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from backend.config import CHROMA_DIR, HISTORY_COLLECTION, HISTORY_K
from backend.embeddings import get_lc_embeddings, get_chroma_embedding_fn


@lru_cache(maxsize=1)
def get_history_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=HISTORY_COLLECTION,
        embedding_function=get_chroma_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def save_turn(session_id: str, role: str, content: str) -> None:
    get_history_collection().upsert(
        documents=[content],
        metadatas=[{"session_id": session_id, "role": role}],
        ids=[f"{session_id}_{uuid.uuid4().hex}"],
    )


def get_history_retriever(session_id: str, k: int = HISTORY_K):
    vectorstore = Chroma(
        collection_name=HISTORY_COLLECTION,
        embedding_function=get_lc_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"session_id": {"$eq": session_id}}}
    )


def retrieve_relevant_history(session_id: str, query: str) -> str:
    try:
        docs = get_history_retriever(session_id).invoke(query)
        return "\n".join(d.page_content for d in docs)
    except Exception:
        return ""
