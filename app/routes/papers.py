from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import HTTPException

# You already have these imports in your project (based on your earlier code)
from app.embeddings import vector_store  # must exist in your project

# Optional: if you already have an embedding function, import it.
# If you don't, this endpoint can still work if your vector_store supports text-query search.
try:
    from app.embeddings import embed_text  # OPTIONAL: define if you have it
except Exception:
    embed_text = None


class AskRequest(BaseModel):
    paper_id: Optional[str] = None
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


def _safe_score(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _search_vector_store(question: str, top_k: int):
    """
    Compatibility layer: supports common vector store APIs.
    Returns: list of dicts like {"text": "...", "meta": {...}}
    """
    # 1) If store supports text search directly
    for method_name in ("search", "similarity_search", "query"):
        fn = getattr(vector_store, method_name, None)
        if callable(fn):
            try:
                # Try calling with text
                res = fn(question, top_k=top_k)
                if res is not None:
                    return res
            except TypeError:
                # Some stores use different param names
                try:
                    res = fn(question, k=top_k)
                    if res is not None:
                        return res
                except Exception:
                    pass
            except Exception:
                pass

    # 2) If store needs embeddings
    if embed_text is not None:
        emb = embed_text(question)
        for method_name in ("search_by_vector", "query_by_vector", "similarity_search_by_vector"):
            fn = getattr(vector_store, method_name, None)
            if callable(fn):
                try:
                    return fn(emb, top_k=top_k)
                except TypeError:
                    return fn(emb, k=top_k)

    raise RuntimeError("vector_store search method not found (add a search/query method or embed_text).")


def _normalize_hits(hits: Any) -> List[Dict[str, Any]]:
    """
    Normalize different hit formats into:
      [{"text": str, "meta": {"score": float, "page":..., "paper_id":...}}]
    """
    norm: List[Dict[str, Any]] = []

    if hits is None:
        return norm

    # If vector_store returns list of your VectorItem objects
    if isinstance(hits, list):
        for h in hits:
            # dict format
            if isinstance(h, dict):
                text = h.get("text") or h.get("content") or ""
                meta = h.get("meta") or {}
                if "score" not in meta and "score" in h:
                    meta["score"] = h["score"]
                norm.append({"text": text, "meta": meta})
                continue

            # object format
            text = getattr(h, "text", None) or getattr(h, "content", None) or ""
            meta = getattr(h, "meta", None) or {}
            score = getattr(h, "score", None)
            if score is not None:
                meta = dict(meta)
                meta["score"] = score
            norm.append({"text": text, "meta": meta})

    return norm


def _build_grounded_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Simple grounded answer without an LLM:
    - Returns the most relevant chunk(s) as 'evidence'
    - Works reliably and avoids hallucination.
    """
    if not sources:
        return (
            "I couldn’t find strong evidence for that question in the uploaded PDF(s). "
            "Try rephrasing or ask something more specific (method, dataset, section title, etc.)."
        )

    # Take top sources and stitch a short grounded response
    evidence = "\n\n".join([f"- {s['text'][:800]}" for s in sources[:3] if s.get("text")])
    return (
        f"Based on the most relevant parts of the paper, here is the grounded evidence:\n\n{evidence}\n\n"
        f"If you want, ask a more specific question (e.g., 'What dataset name is used?' or 'What is the main contribution?')."
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        raw_hits = _search_vector_store(req.question.strip(), top_k=req.top_k)
        hits = _normalize_hits(raw_hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Optional filtering by paper_id if your meta contains it
    if req.paper_id:
        filtered = []
        for h in hits:
            pid = (h.get("meta") or {}).get("paper_id") or (h.get("meta") or {}).get("doc_id")
            if pid is None or str(pid) == str(req.paper_id):
                filtered.append(h)
        hits = filtered

    # Enforce min_score if score exists
    def keep(h):
        sc = (h.get("meta") or {}).get("score", None)
        if sc is None:
            return True
        return _safe_score(sc) >= float(req.min_score)

    hits = [h for h in hits if keep(h)]

    if req.strict and not hits:
        answer = (
            "I couldn’t find strong evidence for that question in the uploaded PDF(s). "
            "Try rephrasing or ask something more specific (method name, dataset, section, etc.)."
        )
        return AskResponse(answer=answer, sources=[])

    # ✅ Grounded answer (no hallucination)
    answer = _build_grounded_answer(req.question, hits)

    # Return only top_k sources
    out_sources = [Source(text=h.get("text", ""), meta=h.get("meta") or {}) for h in hits[: req.top_k]]

    return AskResponse(answer=answer, sources=out_sources)
