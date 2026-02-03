import os
import uuid
from typing import List, Dict, Any, Optional

import streamlit as st

# PDF text extraction
import fitz  # PyMuPDF

# Your vector store (FAISS + SentenceTransformer)
from app.embeddings import vector_store, VectorItem


# -----------------------------
# OpenAI client (supports v1+ and old SDK)
# -----------------------------
def _get_openai_mode():
    """
    Returns:
      ("new", client) if OpenAI() exists
      ("old", openai_module) if openai.ChatCompletion exists
      (None, None) if not available
    """
    try:
        from openai import OpenAI  # new SDK
        return "new", OpenAI()
    except Exception:
        pass

    try:
        import openai  # old SDK
        return "old", openai
    except Exception:
        return None, None


def _ensure_openai_key():
    """
    Streamlit Cloud: set OPENAI_API_KEY in Secrets.
    Local: set env var OPENAI_API_KEY.
    """
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not found. Add it in Streamlit Secrets or set it as an environment variable.")
        return False
    return True


# -----------------------------
# Chunking + PDF extraction
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pages_text.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking (good enough for a clean demo).
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks


# -----------------------------
# RAG Answering
# -----------------------------
def build_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        [f"[score={s['score']:.3f}] {s['text']}" for s in sources]
    )

    return (
        "You are a research paper assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: 'Not found in the provided papers.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer:"
    )


def llm_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    mode, client_or_mod = _get_openai_mode()
    prompt = build_prompt(question, sources)

    if mode == "new":
        # New OpenAI SDK (v1+)
        client = client_or_mod
        # Choose a model you have access to; gpt-4o-mini is a safe default for many accounts.
        # If you want, change to "gpt-5" or another model you use.
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return (resp.output_text or "").strip()

    if mode == "old":
        # Old OpenAI SDK (0.27.10) – ChatCompletion
        openai = client_or_mod
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research paper assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return resp["choices"][0]["message"]["content"].strip()

    return "OpenAI library not installed. Please install `openai` and set OPENAI_API_KEY."


def answer_question_local(question: str, top_k: int = 5) -> Dict[str, Any]:
    sources = vector_store.search(question, top_k=top_k)
    if not sources:
        return {"answer": "No indexed content found yet. Upload a PDF first.", "sources": []}

    answer = llm_answer(question, sources)
    return {"answer": answer, "sources": sources}


# -----------------------------
# Session helpers
# -----------------------------
def init_session():
    if "indexed_papers" not in st.session_state:
        st.session_state.indexed_papers = []  # list of dicts: {paper_id, name, chunks}
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None


def add_pdf_to_index(file_name: str, file_bytes: bytes, chunk_size: int, overlap: int):
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        st.error(f"No text extracted from {file_name}.")
        return

    paper_id = str(uuid.uuid4())[:8]

    metadatas: List[VectorItem] = []
    for i in range(len(chunks)):
        metadatas.append(
            VectorItem(
                chunk_db_id=i,          # for demo
                paper_id=paper_id,
                chunk_id=f"{paper_id}_{i}",
                section="pdf",
                page_start=None,
                page_end=None,
                lang="en",
            )
        )

    vector_store.add(chunks, metadatas)

    st.session_state.indexed_papers.append(
        {"paper_id": paper_id, "name": file_name, "chunks": len(chunks)}
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Scientific Literature Explorer", layout="wide")
init_session()

st.title(" Scientific Literature Explorer (RAG)")
st.caption("Upload PDFs → Index → Ask questions. Shows sources used.")

# API key check
_ensure_openai_key()

with st.sidebar:
    st.header(" Settings")
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 50)
    overlap = st.slider("Overlap (characters)", 0, 400, 150, 10)
    top_k = st.slider("Top-K chunks", 1, 10, 5)

    st.divider()
    st.subheader(" Indexed Papers")
    if not st.session_state.indexed_papers:
        st.write("No papers indexed yet.")
    else:
        for p in st.session_state.indexed_papers:
            st.write(f"• **{p['name']}**  \n  id: `{p['paper_id']}`  | chunks: {p['chunks']}")

    if st.button(" Clear Session Index"):
        # Note: this clears the session list, but FAISS in memory stays in current process.
        st.session_state.indexed_papers = []
        st.session_state.last_answer = None
        st.success("Cleared session list. (For full reset, restart the app.)")


col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("1) Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(" Index Uploaded PDFs", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Indexing PDFs (extracting text + embedding)..."):
                for f in uploaded_files:
                    add_pdf_to_index(f.name, f.read(), chunk_size, overlap)
            st.success("Indexing complete. Now ask questions!")

with col2:
    st.subheader("2) Ask Questions (RAG)")
    question = st.text_area(
        "Ask a question about the indexed papers",
        placeholder="e.g., What is the main objective of the paper?",
        height=120
    )

    if st.button(" Ask", type="primary"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching + generating answer..."):
                result = answer_question_local(question, top_k=top_k)
                st.session_state.last_answer = result

# Show answer + sources
if st.session_state.last_answer:
    st.divider()
    st.subheader(" Answer")
    st.write(st.session_state.last_answer["answer"])

    st.subheader(" Sources used")
    sources = st.session_state.last_answer.get("sources", [])
    if not sources:
        st.info("No sources returned.")
    else:
        for i, s in enumerate(sources, 1):
            score = s.get("score", 0.0)
            text = s.get("text", "")
            meta = s.get("metadata", None)

            with st.expander(f"Source #{i} | score={score:.3f}"):
                st.write(text)
                if meta is not None:
                    # VectorItem dataclass is not JSON serializable by default; convert safely
                    try:
                        st.json(meta.__dict__)
                    except Exception:
                        st.write(meta)
