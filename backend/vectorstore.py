"""Main document vector store (all sessions, one collection, filtered by
session_id) and the single `main_retriever` instance used across the app."""

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from backend.config import CHROMA_DIR, COLLECTION_NAME, TOP_K, SCORE_THRESHOLD
from backend.embeddings import get_lc_embeddings, get_chroma_embedding_fn


def build_vector_store(
    text_chunks: list[dict],
    session_id: str,
    source_name: str,
    collection_name: str = COLLECTION_NAME,
    persist_dir: str = CHROMA_DIR,
):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_chroma_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    for i, chunk in enumerate(text_chunks):
        documents.append(chunk["content"])
        metadatas.append({
            "type": "text",
            "page": chunk["page"],
            "source": source_name,
            "chunk_index": chunk["chunk_index"],
            "session_id": session_id,
        })
        ids.append(f"{session_id}_text_{source_name}_{i}")

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return collection


@lru_cache(maxsize=1)
def get_main_retriever(k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD):
    """Single retriever instance reused for the whole app. Its session filter
    is set per-query (see backend.chain.ask) rather than at construction time,
    so this stays one object instead of one-per-session."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_lc_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
