# ╔══════════════════════════════════════════════════════════════════════╗
# ║        MULTIMODAL RAG — SINGLE CELL  (paste into one cell)          ║
# ║  PDF → Extract text + images → Summarize images (LLaVA) →           ║
# ║  ChromaDB → Groq LLM → Answer with inline image display             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ─── CONFIG — change these ───────────────────────────────────────────────────
PDF_PATH        = "../pdfs/Tower_1.pdf"
QUERY           = "What is my red jumper wire health?"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_docs"
GROQ_MODEL      = "qwen/qwen3-32b"
LLAVA_MODEL     = "llava:13b"
TOP_K           = 5
# ─────────────────────────────────────────────────────────────────────────────

import os, base64
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]      = os.getenv("HF_API_KEY", "")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY", "")

# ── Imports ──────────────────────────────────────────────────────────────────
import fitz                                                      # pymupdf
import pdfplumber
import ollama
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
from IPython.display import display, Image as IPImage, HTML, Markdown
import ipywidgets as widgets


# ════════════════════════════════════════════════════════════════════════════
# 1. EXTRACT IMAGES  (with surrounding text context)
# ════════════════════════════════════════════════════════════════════════════
def extract_images_with_context(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    image_records = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        blocks     = page.get_text("blocks")

        for img_index, img in enumerate(page.get_images(full=True)):
            xref      = img[0]
            img_rects = page.get_image_rects(xref)
            if not img_rects:
                continue
            img_rect = img_rects[0]

            context_above, context_below = [], []
            for block in blocks:
                bx0, by0, bx1, by1, block_text = block[0], block[1], block[2], block[3], block[4]
                if not block_text.strip():
                    continue
                if by1 <= img_rect.y0 and (img_rect.y0 - by1) < 200:
                    context_above.append((by1, block_text.strip()))
                if by0 >= img_rect.y1 and (by0 - img_rect.y1) < 200:
                    context_below.append((by0, block_text.strip()))

            context_above = [t for _, t in sorted(context_above, key=lambda x: -x[0])]
            context_below = [t for _, t in sorted(context_below, key=lambda x:  x[0])]

            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            img_b64   = base64.b64encode(img_bytes).decode()

            image_records.append({
                "page":           page_num + 1,
                "img_index":      img_index,
                "xref":           xref,
                "image_b64":      img_b64,
                "image_bytes":    img_bytes,          # ← kept for inline display
                "context_above":  " ".join(context_above[-3:]),
                "context_below":  " ".join(context_below[:3]),
                "full_page_text": page_text,
            })

    print(f"✅  Extracted {len(image_records)} images from PDF")
    return image_records


# ════════════════════════════════════════════════════════════════════════════
# 2. DISPLAY IMAGES INLINE  (call anytime after extraction)
# ════════════════════════════════════════════════════════════════════════════
def show_extracted_images(image_records: list[dict], max_width: int = 500):
    """Render all extracted images inline in the notebook."""
    if not image_records:
        print("No images found.")
        return

    display(HTML(f"<h3>📸 Extracted Images ({len(image_records)} total)</h3>"))
    for rec in image_records:
        display(HTML(
            f"<b>Page {rec['page']} — Image #{rec['img_index']}</b><br>"
            f"<small>Context above: {rec['context_above'][:120] or '—'}</small>"
        ))
        display(IPImage(data=rec["image_bytes"], format="png", width=max_width))
        display(HTML("<hr>"))


# ════════════════════════════════════════════════════════════════════════════
# 3. SUMMARISE IMAGES with LLaVA (local Ollama)
# ════════════════════════════════════════════════════════════════════════════
def summarize_image_llava(record: dict) -> str:
    context_prompt = ""
    if record["context_above"]:
        context_prompt += f"Text BEFORE this image:\n{record['context_above']}\n\n"
    if record["context_below"]:
        context_prompt += f"Text AFTER this image:\n{record['context_below']}\n\n"

    response = ollama.chat(
        model=LLAVA_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"{context_prompt}"
                "Summarize this image for a RAG system. Include:\n"
                "- Type of visual (chart, diagram, photo, table, etc.)\n"
                "- Key data, labels, entities shown\n"
                "- How it relates to the surrounding text\n"
                "Be specific and information-dense."
            ),
            "images": [record["image_b64"]],
        }]
    )
    return response["message"]["content"]


# ════════════════════════════════════════════════════════════════════════════
# 4. EXTRACT TEXT CHUNKS
# ════════════════════════════════════════════════════════════════════════════
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
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                chunks.append({
                    "type":        "text",
                    "content":     chunk,
                    "page":        page_num + 1,
                    "chunk_index": i,
                    "source":      pdf_path,
                })
    print(f"✅  Extracted {len(chunks)} text chunks from PDF")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# 5. EMBEDDINGS + VECTOR STORE
# ════════════════════════════════════════════════════════════════════════════
class LangChainEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, lc_embedding):
        self.lc_embedding = lc_embedding
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.lc_embedding.embed_documents(input)

lc_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
chroma_embedding_fn = LangChainEmbeddingAdapter(lc_embeddings)


def build_vector_store(text_chunks, image_records,
                       collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR):
    client     = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=chroma_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(text_chunks):
        documents.append(chunk["content"])
        metadatas.append({"type": "text", "page": chunk["page"],
                          "source": chunk["source"], "chunk_index": chunk["chunk_index"]})
        ids.append(f"text_{i}")

    for j, img in enumerate(image_records):
        embeddable = f"[IMAGE SUMMARY - Page {img['page']}]\n{img['summary']}"
        if img.get("context_above"):
            embeddable += f"\n\nSurrounding context: {img['context_above'][:300]}"
        documents.append(embeddable)
        metadatas.append({
            "type":          "image",
            "page":          img["page"],
            "source":        pdf_path,
            "img_index":     img["img_index"],
            "context_above": img.get("context_above", "")[:500],
            "context_below": img.get("context_below", "")[:500],
        })
        ids.append(f"image_{j}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅  Stored {len(text_chunks)} text + {len(image_records)} image chunks in ChromaDB")
    return collection


def get_retriever(collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR,
                  k=TOP_K, filter_type=None, score_threshold=0.3):
    vectorstore  = Chroma(
        collection_name=collection_name,
        embedding_function=lc_embeddings,
        persist_directory=persist_dir,
    )
    search_kwargs = {"k": k, "score_threshold": score_threshold}
    if filter_type:
        search_kwargs["filter"] = {"type": {"$eq": filter_type}}
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. RAG CHAIN (Groq LLM)
# ════════════════════════════════════════════════════════════════════════════
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, max_tokens=1024)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions from a document.
Use ONLY the context below. If an [IMAGE] chunk is relevant, reference it naturally.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        tag  = "IMAGE" if doc.metadata.get("type") == "image" else "TEXT"
        page = doc.metadata.get("page", "?")
        parts.append(f"[{tag} — page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(filter_type=None, k=TOP_K):
    retriever = get_retriever(k=k, filter_type=filter_type)
    return (
        RunnableParallel({
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt | llm | StrOutputParser()
    )


def build_rag_chain_with_sources(filter_type=None, k=TOP_K):
    retriever  = get_retriever(k=k, filter_type=filter_type)
    base_chain = build_rag_chain(filter_type=filter_type, k=k)
    return RunnableParallel({"answer": base_chain, "source_documents": retriever})


# ════════════════════════════════════════════════════════════════════════════
# 7. DISPLAY ANSWER + RELEVANT IMAGES SIDE BY SIDE
# ════════════════════════════════════════════════════════════════════════════
def show_answer_with_images(query: str, image_records: list[dict],
                            filter_type=None, k=TOP_K, max_img_width=400):
    """Run RAG query and display the LLM answer + any relevant images inline."""

    display(HTML(f"<h3>🔍 Query: <em>{query}</em></h3>"))

    # ── Run chain ────────────────────────────────────────────────────────────
    chain  = build_rag_chain_with_sources(filter_type=filter_type, k=k)
    result = chain.invoke(query)

    answer      = result["answer"]
    source_docs = result["source_documents"]

    # ── LLM Answer ───────────────────────────────────────────────────────────
    display(HTML("<h4>💬 LLM Answer</h4>"))
    display(Markdown(answer))

    # ── Source chunks ────────────────────────────────────────────────────────
    display(HTML("<h4>📄 Retrieved Chunks</h4>"))
    for doc in source_docs:
        t    = doc.metadata.get("type", "text")
        page = doc.metadata.get("page", "?")
        display(HTML(f"<b>[{t.upper()}] Page {page}</b>"))
        display(Markdown(f"> {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}"))

    # ── Show images from retrieved image chunks ───────────────────────────────
    image_pages = {
        doc.metadata["page"]
        for doc in source_docs
        if doc.metadata.get("type") == "image"
    }

    if image_pages:
        display(HTML("<h4>🖼️ Referenced Images</h4>"))
        # Build a lookup: page → list of image records
        page_to_imgs = {}
        for rec in image_records:
            page_to_imgs.setdefault(rec["page"], []).append(rec)

        for page in sorted(image_pages):
            for rec in page_to_imgs.get(page, []):
                display(HTML(
                    f"<b>Page {rec['page']} — Image #{rec['img_index']}</b><br>"
                    f"<small><i>Summary: {rec.get('summary', '')[:200]}</i></small>"
                ))
                display(IPImage(data=rec["image_bytes"], format="png", width=max_img_width))
    else:
        display(HTML("<p><i>No image chunks were retrieved for this query.</i></p>"))


# ════════════════════════════════════════════════════════════════════════════
# 8. RUN EVERYTHING
# ════════════════════════════════════════════════════════════════════════════

# ── Step 1: Extract ──────────────────────────────────────────────────────────
print("📄 Step 1/4 — Extracting text and images …")
text_chunks   = extract_text_chunks(PDF_PATH)
image_records = extract_images_with_context(PDF_PATH)

# ── Step 2: Show all extracted images (optional preview) ─────────────────────
print("\n🖼️  Step 2/4 — Displaying extracted images …")
show_extracted_images(image_records)

# ── Step 3: Summarise images with LLaVA + build vector store ─────────────────
print("\n🤖 Step 3/4 — Summarising images with LLaVA & building ChromaDB …")
for i, rec in enumerate(image_records):
    print(f"  Summarising image {i+1}/{len(image_records)} (page {rec['page']}) …")
    rec["summary"] = summarize_image_llava(rec)

build_vector_store(text_chunks, image_records)

# ── Step 4: Answer query with image display ───────────────────────────────────
print(f"\n💬 Step 4/4 — Running RAG query …\n")
show_answer_with_images(QUERY, image_records)

# ── Optional: try image-only retrieval ───────────────────────────────────────
print("\n\n── Image-only retrieval ──")
show_answer_with_images(
    "Describe all diagrams in the document",
    image_records,
    filter_type="image",
    k=3,
)
