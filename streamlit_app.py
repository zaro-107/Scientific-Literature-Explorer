# streamlit_app.py
import time
import json
import streamlit as st

# TODO 1: import your backend pipeline function here
# Example:
# from app.rag import answer_question  # <- you will replace this with your actual import


st.set_page_config(page_title="Scientific Literature Explorer", page_icon="📄", layout="wide")

# --- Minimal CSS polish (still Streamlit, but looks premium) ---
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
small { opacity: 0.75; }
.source-card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.badge {
  display:inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 12px;
  margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📄 Scientific Literature Explorer")
st.caption("Ask anything about your research paper — answers are grounded in the uploaded PDF.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    strict_mode = st.toggle("Strict grounding (recommended)", value=True,
                            help="If ON, the app refuses to answer when evidence is weak.")
    top_k = st.slider("Top-K sources", 2, 8, 4)
    min_score = st.slider("Min similarity threshold", 0.0, 1.0, 0.25, 0.01,
                          help="Higher = stricter. Tune depending on your scoring.")
    st.divider()
    st.subheader(" Suggested questions")
    st.write("- What is the main contribution?\n- What methodology is used?\n- What dataset?\n- What are limitations?\n- Summarize the results table.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Layout ---
left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Upload PDF(s)")
    files = st.file_uploader("Upload one or more research papers (PDF)", type=["pdf"], accept_multiple_files=True)

    if files:
        st.session_state.uploaded_files = files
        st.session_state.doc_ready = True
        st.success(f"Loaded {len(files)} PDF(s). You can start asking questions.")
    else:
        st.info("Upload at least one PDF to begin.")

    st.divider()
    st.subheader("Session")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()

    # Optional: show file list
    if st.session_state.uploaded_files:
        st.write("**Current PDFs:**")
        for f in st.session_state.uploaded_files:
            st.write(f"- {f.name}")

with right:
    st.subheader("2) Ask questions (unlimited)")
    st.caption("You can ask anything — the answer is limited only by what exists in the PDF.")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources used"):
                    for s in m["sources"]:
                        meta = s.get("meta", {})
                        page = meta.get("page", "—")
                        score = meta.get("score", "—")
                        title = meta.get("title", "")

                        st.markdown(
                            f"""<div class="source-card">
                            <span class="badge">page: {page}</span>
                            <span class="badge">score: {score}</span>
                            <span class="badge">{title}</span>
                            <div style="margin-top:8px; white-space:pre-wrap;">{s.get("text","")}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )

    prompt = st.chat_input("Ask a question about the uploaded paper(s)...")

    if prompt:
        if not st.session_state.doc_ready:
            st.warning("Upload at least one PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- Call backend ---
            with st.chat_message("assistant"):
                thinking = st.empty()
                thinking.markdown(" Searching the paper and preparing grounded answer...")

                # TODO 2: call your real function here.
                # It should return:
                #   answer: str
                #   sources: list[ { "text": str, "meta": { "page": int|str, "score": float|str, "title": str } } ]
                #
                # Example expected signature:
                # answer, sources = answer_question(files=st.session_state.uploaded_files,
                #                                 question=prompt, top_k=top_k)
                #
                # For now, placeholder:
                answer = "⚠️ Hook up your backend function in TODO 2."
                sources = []

                # --- Strict grounding behavior ---
                # If your retrieval scores are higher-is-better and normalized (0..1), this works well.
                # Adjust if your scoring uses distance (lower-is-better).
                best_score = None
                if sources:
                    try:
                        best_score = max(float(s.get("meta", {}).get("score", 0.0)) for s in sources)
                    except Exception:
                        best_score = None

                if strict_mode and (not sources or (best_score is not None and best_score < min_score)):
                    answer = ("I couldn’t find strong evidence for that question in the uploaded PDF(s). "
                              "Try rephrasing, or ask something more specific (section name, method, dataset, etc.).")
                    sources = []

                # --- Streaming effect (simple but premium) ---
                thinking.empty()
                out = st.empty()
                acc = ""
                for ch in answer:
                    acc += ch
                    out.markdown(acc)
                    time.sleep(0.004)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                # Export as Markdown
                export_md = f"## Q: {prompt}\n\n## Answer\n{answer}\n\n## Sources\n"
                for i, s in enumerate(sources, 1):
                    meta = s.get("meta", {})
                    export_md += f"\n### Source {i} (page {meta.get('page','—')}, score {meta.get('score','—')})\n{s.get('text','')}\n"

                st.download_button("⬇️ Download this answer (Markdown)",
                                   data=export_md.encode("utf-8"),
                                   file_name="answer_with_sources.md",
                                   mime="text/markdown",
                                   use_container_width=True)
