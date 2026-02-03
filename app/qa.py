import os
from typing import List, Dict, Any

from openai import OpenAI

from app.embeddings import vector_store

client = OpenAI()  # reads OPENAI_API_KEY from environment


def generate_answer(question: str, retrieved: List[Dict[str, Any]]) -> str:
    # Build context from retrieved chunks
    context = "\n\n".join(
        [f"[score={c['score']:.3f}] {c['text']}" for c in retrieved]
    )

    prompt = (
        "You are a research paper assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: 'Not found in the provided papers.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    response = client.responses.create(
        model="gpt-5",  # you can change later
        input=prompt,
    )

    return response.output_text.strip()


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    retrieved = vector_store.search(question, top_k=top_k)
    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": retrieved}
