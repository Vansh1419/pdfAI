"""The RAG chain itself: prompt template, Groq LLM, and the ask() function
the frontend calls for every user message."""

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

from backend.config import GROQ_MODEL
from backend.history_store import retrieve_relevant_history, save_turn
from backend.vectorstore import get_main_retriever

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions from a document.
try to answer from the context below.

Relevant prior conversation (may be empty):
{history}

Context:
{context}

Question: {question}

Answer:""")


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL,reasoning_format="hidden", temperature=0.2, max_tokens=1024)


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[TEXT — page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@lru_cache(maxsize=1)
def get_rag_chain():
    retriever = get_main_retriever()
    return (
        RunnableParallel({
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        })
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


def ask(session_id: str, question: str) -> str:
    """Filter the single main_retriever to this session, run the chain,
    and persist both sides of the exchange to the history store."""
    retriever = get_main_retriever()
    retriever.search_kwargs["filter"] = {"session_id": {"$eq": session_id}}

    history_text = retrieve_relevant_history(session_id, question)
    answer = get_rag_chain().invoke({"question": question, "history": history_text})

    save_turn(session_id, "user", question)
    save_turn(session_id, "assistant", answer)
    return answer
