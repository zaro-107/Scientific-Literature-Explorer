from qa import answer_question

def compare_papers(question, paper_chunks_list):
    results = []
    for i, chunks in enumerate(paper_chunks_list):
        context = "\n".join(chunks)
        answer = answer_question(question)
        results.append({"paper_id": i+1, "answer": answer})
    return results
