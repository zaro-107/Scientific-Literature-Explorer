from typing import Dict, List
from app.embeddings import vector_store

from app.qa import llm  # your existing LLM instance
from app.qa import answer_question
print(answer_question("What is the paper about?"))


SUMMARY_PROMPT = """
You are an AI research assistant.

Generate a structured summary of the given academic paper content.

Content:
{context}

Return the summary in the following structure:
- Problem
- Approach / Methodology
- Key Findings
- Limitations
- Keywords
"""


def generate_structured_summary(
    paper_id: str,
    max_chunks: int = 10
) -> Dict:
    """
    Generates a structured summary for a paper using top semantic chunks.
    """

    # Retrieve top chunks for overall paper meaning
    docs = vector_store.similarity_search(
        query="Summarize the entire paper",
        k=max_chunks,
        filter={"paper_id": paper_id}
    )

    if not docs:
        return {"error": "No content found for this paper."}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = SUMMARY_PROMPT.format(context=context)

    response = llm.invoke(prompt)

    return {
        "paper_id": paper_id,
        "summary_text": response,
        "chunks_used": len(docs)
    }
