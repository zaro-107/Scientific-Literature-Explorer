#  Scientific Literature Explorer  
### Deep Search & Question Answering Platform using Generative AI (RAG)

Scientific Literature Explorer is an AI-powered application that helps researchers, students, and professionals quickly understand the **core ideas of scientific papers** without reading them end-to-end.

It uses **Retrieval-Augmented Generation (RAG)** to perform semantic search over uploaded research papers and answer user questions with **source-backed responses**.

---

##  Features

-  Upload scientific research papers (PDF)
-  Automatic text extraction and intelligent chunking
- Semantic search using **Sentence Transformers + FAISS**
-  Ask natural language questions about papers
-  Answers generated using **Generative AI**
-  Shows the most relevant source chunks used for answers
-  Fast, interactive UI built with **Streamlit**
-  Deployable directly on **Streamlit Cloud**

---

##  How It Works (RAG Pipeline)

1. **PDF Upload**  
   User uploads one or more research papers.

2. **Text Extraction & Chunking**  
   PDFs are parsed and split into overlapping text chunks.

3. **Embedding Generation**  
   Each chunk is converted into vector embeddings using a transformer model.

4. **Vector Search (FAISS)**  
   Relevant chunks are retrieved based on semantic similarity to the user’s query.

5. **Answer Generation (LLM)**  
   A language model generates answers using only the retrieved context.

---

## 🏗️ Project Structure

