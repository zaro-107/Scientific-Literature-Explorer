import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root

from openai import OpenAI
from app.embeddings import vector_store

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"[score={c['score']:.3f}] {c['text']}" for c in retrieved])

    prompt = (
        "You are a research paper assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: 'Not found in the provided papers.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    response = client.responses.create(
        model="gpt-5",
        input=prompt,
    )

    return (response.output_text or "").strip()


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    retrieved = vector_store.search(question, top_k=top_k)
    if not retrieved:
        return {"answer": "Not found in the provided papers.", "sources": []}

    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": retrieved}
