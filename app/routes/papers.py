import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Paper, PaperChunk
from app.language_utils import detect_language
from app.translator import TranslatorToEnglish
from app.embeddings import vector_store, VectorItem

import fitz  # PyMuPDF


router = APIRouter(prefix="", tags=["papers"])
translator = TranslatorToEnglish()


# ---------- helpers ----------
def extract_text_by_page(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append(text.strip())
    return pages


def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character chunker (robust & fast). Later we can upgrade to token-based.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks


# ---------- API schemas ----------
class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    paper_id: str
    chunk_id: str
    section: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    lang: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    query: str
    top_k: int
    matches: List[Citation]


# ---------- endpoints ----------
@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    pages = extract_text_by_page(pdf_bytes)
    full_text = "\n".join([p for p in pages if p])

    if len(full_text) < 200:
        raise HTTPException(status_code=400, detail="Could not extract enough text from this PDF.")

    # Create paper record
    paper_id = str(uuid.uuid4())[:12]
    paper = Paper(paper_id=paper_id, title=file.filename, source="upload")
    db.add(paper)
    db.commit()
    db.refresh(paper)

    # Ingest chunks page-wise so citations can carry page numbers
    all_texts_for_embedding: List[str] = []
    all_meta: List[VectorItem] = []

    chunk_count = 0

    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text:
            continue

        # detect language on the page text (good enough + faster)
        lang = detect_language(page_text, default="en")

        chunks = simple_chunk(page_text)
        for ch in chunks:
            chunk_count += 1
            chunk_id = f"{paper_id}_{chunk_count:04d}"

            text_original = ch
            text_en = translator.translate(ch, lang) if lang != "en" else ch

            # store in DB
            chunk_row = PaperChunk(
                paper_id_fk=paper.id,
                chunk_id=chunk_id,
                section="unknown",
                page_start=page_idx,
                page_end=page_idx,
                lang=lang,
                text_original=text_original,
                text_en=text_en,
                embedding_id=chunk_id,  # we use chunk_id as embedding id
            )
            db.add(chunk_row)
            db.commit()
            db.refresh(chunk_row)

            # collect for vector store
            all_texts_for_embedding.append(text_en)
            all_meta.append(
                VectorItem(
                    chunk_db_id=chunk_row.id,
                    paper_id=paper.paper_id,
                    chunk_id=chunk_id,
                    section="unknown",
                    page_start=page_idx,
                    page_end=page_idx,
                    lang=lang,
                )
            )

    # Add all embeddings in one go (fast)
    vector_store.add(all_texts_for_embedding, all_meta)

    return {
        "message": "Paper uploaded & indexed successfully",
        "paper_id": paper.paper_id,
        "title": paper.title,
        "pages": len(pages),
        "chunks_indexed": chunk_count,
    }


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    results = vector_store.search(req.query, top_k=req.top_k)

    citations: List[Citation] = []
    for score, meta in results:
        # fetch DB chunk for snippet (english + original available)
        chunk = db.query(PaperChunk).filter(PaperChunk.id == meta.chunk_db_id).first()
        if not chunk:
            continue

        snippet = (chunk.text_en or chunk.text_original or "")[:350]

        citations.append(
            Citation(
                paper_id=meta.paper_id,
                chunk_id=meta.chunk_id,
                section=meta.section,
                page_start=meta.page_start,
                page_end=meta.page_end,
                lang=meta.lang,
                snippet=snippet,
                score=score,
            )
        )

    return AskResponse(query=req.query, top_k=req.top_k, matches=citations)
