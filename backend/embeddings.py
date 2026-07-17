"""Embedding model singletons, shared by the main vector store and the
chat-history vector store."""

from functools import lru_cache
from typing import List

from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from backend.config import EMBEDDING_MODEL


class LangChainEmbeddingAdapter(EmbeddingFunction):
    """Adapts a LangChain embedding object to Chroma's EmbeddingFunction interface."""

    def __init__(self, lc_embedding):
        self.lc_embedding = lc_embedding

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.lc_embedding.embed_documents(input)


@lru_cache(maxsize=1)
def get_lc_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_chroma_embedding_fn() -> LangChainEmbeddingAdapter:
    return LangChainEmbeddingAdapter(get_lc_embeddings())
