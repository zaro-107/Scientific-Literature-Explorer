import streamlit as st
import requests

API_URL = "http://127.0.0.1:YOUR_PORT"


st.set_page_config(page_title="Scientific Explorer", layout="wide")
st.title("🔎 Scientific Explorer (Multilingual RAG)")

st.markdown("Ask questions from your stored papers. The app will show answers + the source chunks used.")

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("Ask a question", placeholder="e.g., What is the main objective of the paper?", height=120)
    top_k = st.slider("How many chunks to retrieve (top_k)", 1, 10, 5)

    ask_btn = st.button("🚀 Ask", type="primary")

with col2:
    st.subheader("API Status")
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        st.success("Backend is running ✅")
        st.json(r.json())
    except Exception:
        st.error("Backend not reachable ❌\nRun: python -m uvicorn app.main:app --reload")

if ask_btn:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/qa/ask",
                    json={"question": question, "top_k": int(top_k)},
                    timeout=60
                )
                if resp.status_code != 200:
                    st.error(resp.text)
                else:
                    data = resp.json()

                    st.subheader("✅ Answer")
                    st.write(data.get("answer", ""))

                    st.subheader("📌 Sources used")
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No sources returned.")
                    else:
                        for i, s in enumerate(sources, 1):
                            score = s.get("score", 0)
                            text = s.get("text", "")
                            meta = s.get("metadata", {})

                            with st.expander(f"Source #{i}  |  score={score:.3f}"):
                                st.write(text)

                                # Show metadata nicely
                                if meta:
                                    st.caption("Metadata")
                                    st.json(meta)

            except Exception as e:
                st.error(f"Request failed: {e}")
