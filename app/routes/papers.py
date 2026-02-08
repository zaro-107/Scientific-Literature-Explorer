# app/routes/papers.py

import uuid
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Paper, PaperChunk

from app.language_utils import detect_language
from app.translator import TranslatorToEnglish

from app.embeddings import vector_store, VectorItem  # your vector store wrapper

print("✅ RUNNING papers.py from commit: de0e21a - NO filename kwarg")

router = APIRouter(prefix="", tags=["papers"])
translator = TranslatorToEnglish()


# -----------------------------
# Helpers
# -----------------------------
def extract_text_by_page(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[str] = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append(text.strip())
    return pages


def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple robust chunker (character-based).
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _vector_search(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Calls your vector_store with multiple possible method names.
    Normalizes results to: [{"text":..., "meta": {...}}, ...]
    """
    hits = None

    # Try common method names
    for name in ("search", "similarity_search", "query"):
        fn = getattr(vector_store, name, None)
        if callable(fn):
            try:
                hits = fn(question, top_k=top_k)
                break
            except TypeError:
                try:
                    hits = fn(question, k=top_k)
                    break
                except Exception:
                    pass
            except Exception:
                pass

    if hits is None:
        raise RuntimeError("vector_store has no supported search method (search/query/similarity_search).")

    normalized: List[Dict[str, Any]] = []

    if isinstance(hits, list):
        for h in hits:
            # dict format
            if isinstance(h, dict):
                text = h.get("text") or h.get("content") or ""
                meta = h.get("meta") or {}
                if "score" not in meta and "score" in h:
                    meta["score"] = h["score"]
                normalized.append({"text": text, "meta": meta})
                continue

            # object format (e.g., VectorItem-like)
            text = getattr(h, "text", None) or getattr(h, "content", None) or ""
            meta = getattr(h, "meta", None) or {}
            score = getattr(h, "score", None)
            if score is not None:
                meta = dict(meta)
                meta["score"] = score
            normalized.append({"text": text, "meta": meta})

    return normalized


def _build_grounded_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    No-LLM grounded answer: returns strongest evidence (minimal hallucination).
    You can later plug an LLM here.
    """
    if not sources:
        return (
            "I couldn’t find strong evidence for that question in the uploaded PDF. "
            "Try rephrasing or asking something more specific (method, dataset, section name)."
        )

    evidence = "\n\n".join([f"• {s['text'][:900]}" for s in sources[:3] if s.get("text")])
    return (
        f"Here is the most relevant grounded evidence from the paper:\n\n{evidence}\n\n"
        f"If you want a sharper answer, ask something more specific (e.g., 'What dataset name is used?')."
    )


# -----------------------------
# Request/Response Models
# -----------------------------
class UploadResponse(BaseModel):
    paper_id: str          # public string id (Paper.paper_id)
    filename: str
    pages: int
    chunks_indexed: int
    language: Optional[str] = None


class AskRequest(BaseModel):
    paper_id: Optional[str] = None  # public string id (Paper.paper_id)
    question: str
    top_k: int = 4
    min_score: float = 0.25
    strict: bool = True


class Source(BaseModel):
    text: str
    meta: Dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# -----------------------------
# Routes
# -----------------------------
@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Extract pages
    pages_original = extract_text_by_page(pdf_bytes)
    if not pages_original or all(not p.strip() for p in pages_original):
        raise HTTPException(status_code=400, detail="Could not extract text from this PDF.")

    # Detect language (best-effort)
    sample_text = next((p for p in pages_original if p.strip()), "")
    lang: Optional[str] = None
    try:
        lang = detect_language(sample_text) if sample_text else None
    except Exception:
        lang = None

    # Translate pages to English if needed (best-effort)
    pages_en: List[str] = []
    if lang and lang.lower() not in ("en", "english"):
        for p in pages_original:
            if not p.strip():
                pages_en.append("")
                continue
            try:
                pages_en.append(translator.translate(p))
            except Exception:
                pages_en.append(p)  # fallback
    else:
        pages_en = pages_original

    # ✅ Create Paper record (matches your Paper model)
    public_paper_id = uuid.uuid4().hex[:12]  # shorter like "a1b2c3d4e5f6"
    paper = Paper(
        owner_id=None,                 # set later if auth is added
        paper_id=public_paper_id,      # ✅ Paper.paper_id (string)
        title=file.filename,           # ✅ Paper.title (filename/title)
        source="upload",
    )

    try:
        db.add(paper)
        db.commit()
        db.refresh(paper)  # ✅ now Paper.id (int PK) exists
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error creating paper: {e}")

    chunks_indexed = 0

    # Chunk -> store in DB + vector store
    try:
        for page_num, page_text_en in enumerate(pages_en, start=1):
            chunk_texts = simple_chunk(page_text_en)

            for idx, chunk_text in enumerate(chunk_texts):
                # ✅ chunk_id matches your model expectation
                chunk_id = f"{public_paper_id}_{idx:04d}"

                db_chunk = PaperChunk(
                    paper_id_fk=paper.id,              # ✅ FK to papers.id (int)
                    chunk_id=chunk_id,
                    section="unknown",
                    page_start=page_num,
                    page_end=page_num,
                    lang=(lang or "en"),
                    text_original=pages_original[page_num - 1],
                    text_en=chunk_text,                # translated or original
                    embedding_id=chunk_id,             # optional mapping
                )
                db.add(db_chunk)

                meta = {
                    "paper_id": public_paper_id,  # ✅ public id for filtering in /ask
                    "page_start": page_num,
                    "page_end": page_num,
                    "chunk_id": chunk_id,
                    "title": file.filename,
                }
                vector_store.add(VectorItem(id=chunk_id, text=chunk_text, meta=meta))
                chunks_indexed += 1

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed while chunking/indexing: {e}")

    return UploadResponse(
        paper_id=public_paper_id,
        filename=file.filename,
        pages=len(pages_original),
        chunks_indexed=chunks_indexed,
        language=lang,
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Search vector store
    try:
        hits = _vector_search(question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Filter by paper_id if provided (public string id)
    if req.paper_id:
        filtered: List[Dict[str, Any]] = []
        for h in hits:
            meta = h.get("meta") or {}
            if str(meta.get("paper_id")) == str(req.paper_id):
                filtered.append(h)
        hits = filtered

    # Apply min_score if score exists
    final_hits: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("meta") or {}
        sc = meta.get("score", None)
        if sc is None:
            final_hits.append(h)
        else:
            if _safe_float(sc, 0.0) >= float(req.min_score):
                final_hits.append(h)

    if req.strict and not final_hits:
        return AskResponse(
            answer=(
                "I couldn’t find strong evidence for that question in the uploaded PDF. "
                "Try rephrasing or ask something more specific (method, dataset, section)."
            ),
            sources=[],
        )

    answer = _build_grounded_answer(question, final_hits)
    sources = [Source(text=h.get("text", ""), meta=h.get("meta") or {}) for h in final_hits[: req.top_k]]
    return AskResponse(answer=answer, sources=sources)
