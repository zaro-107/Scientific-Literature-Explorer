# streamlit_app.py
import os
import time
import requests
import streamlit as st

# =========================
# Config
# =========================
# Set this in Render (UI service) -> Environment:
# BACKEND_URL = https://scientific-literature-explorer-1.onrender.com
BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", "https://scientific-literature-explorer-1.onrender.com")

# If your backend routes are different, change these:
UPLOAD_ENDPOINT = "/upload"   # e.g. "/papers/upload"
ASK_ENDPOINT = "/ask"  # e.g. "/papers/ask"


# =========================
# Page + CSS
# =========================
st.set_page_config(page_title="Scientific Literature Explorer", page_icon="📄", layout="wide")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.title("📄 Scientific Literature Explorer")
st.caption("Upload a research paper → get grounded answers with sources.")


# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False      # whether backend has processed something
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []    # local selected PDFs
if "paper_id" not in st.session_state:
    st.session_state.paper_id = None        # backend identifier (returned by /upload)


# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    st.write("**Backend:**")
    st.code(BACKEND_URL, language="text")

    strict_mode = st.toggle(
        "Strict grounding (recommended)",
        value=True,
        help="If ON, the UI will refuse to answer when evidence looks weak.",
    )
    top_k = st.slider("Top-K sources", 2, 8, 4)
    min_score = st.slider(
        "Min similarity threshold",
        0.0,
        1.0,
        0.25,
        0.01,
        help="Higher = stricter. Adjust depending on your backend scoring.",
    )

    st.divider()
    st.subheader("Session")
    st.write("**Paper ID:**", st.session_state.paper_id or "—")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()

    st.divider()
    st.subheader("Suggested questions")
    st.write(
        "- What is the main contribution?\n"
        "- What methodology is used?\n"
        "- What dataset?\n"
        "- What are limitations?\n"
        "- Summarize the results."
    )


# =========================
# Helpers
# =========================
def backend_url(path: str) -> str:
    return f"{BACKEND_URL.rstrip('/')}{path}"

def upload_pdf_to_backend(uploaded_file) -> str:
    """
    Upload ONE PDF to backend. Expects backend JSON to contain paper_id (or id).
    """
    uploaded_file.seek(0)
    resp = requests.post(
        backend_url(UPLOAD_ENDPOINT),
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    pid = data.get("paper_id") or data.get("id")
    if not pid:
        raise RuntimeError(f"Backend did not return paper_id. Got keys: {list(data.keys())}")
    return pid

def ask_backend(paper_id: str, question: str, top_k: int, min_score: float, strict: bool):
    """
    Calls backend /ask. Expects JSON: { answer: str, sources: list }.
    """
    payload = {
        "paper_id": paper_id,
        "question": question,
        "top_k": top_k,
        "min_score": min_score,
        "strict": strict,
    }
    r = requests.post(backend_url(ASK_ENDPOINT), json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("answer", ""), data.get("sources", []) or []


# =========================
# Layout
# =========================
left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Upload PDF(s)")

    files = st.file_uploader(
        "Upload one or more research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:
        st.session_state.uploaded_files = files

        st.info(
            "Now click **Send PDF(s) to backend**. "
            "Your backend will extract text + build the index."
        )

        if st.button("⬆️ Send PDF(s) to backend", type="primary", use_container_width=True):
            try:
                with st.spinner("Uploading & processing... (first run can take longer)"):
                    # If your backend supports multiple PDFs in ONE combined index,
                    # you should implement a /upload-multi endpoint.
                    # For now, we upload each and keep the last paper_id.
                    last_id = None
                    for f in files:
                        last_id = upload_pdf_to_backend(f)

                st.session_state.paper_id = last_id
                st.session_state.doc_ready = True
                st.success(f"Processed {len(files)} PDF(s). Paper ID: {st.session_state.paper_id}")

            except Exception as e:
                st.session_state.doc_ready = False
                st.session_state.paper_id = None
                st.error(f"Upload failed: {e}")
    else:
        st.session_state.uploaded_files = []
        st.info("Upload at least one PDF to begin.")

    st.divider()
    st.subheader("Current PDFs")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.write(f"- {f.name}")
    else:
        st.write("—")


with right:
    st.subheader("2) Ask questions")
    st.caption("Ask anything — the answer should be grounded in the PDF with sources.")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources used"):
                    for s in m["sources"]:
                        meta = s.get("meta", {}) or {}
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
                            unsafe_allow_html=True,
                        )

    prompt = st.chat_input("Ask a question about the uploaded paper(s)...")

    if prompt:
        # Validate state
        if not st.session_state.doc_ready or not st.session_state.paper_id:
            st.warning("Upload PDF(s) and click **Send PDF(s) to backend** first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant response
            with st.chat_message("assistant"):
                thinking = st.empty()
                thinking.markdown("🔎 Searching the paper and preparing a grounded answer...")

                try:
                    answer, sources = ask_backend(
                        paper_id=st.session_state.paper_id,
                        question=prompt,
                        top_k=top_k,
                        min_score=min_score,
                        strict=strict_mode,
                    )
                except Exception as e:
                    thinking.empty()
                    st.error(f"Backend error: {e}")
                    answer, sources = "", []

                # Optional strict mode (UI-side safety net)
                best_score = None
                if sources:
                    try:
                        best_score = max(float(s.get("meta", {}).get("score", 0.0)) for s in sources)
                    except Exception:
                        best_score = None

                if strict_mode and (not sources or (best_score is not None and best_score < min_score)):
                    answer = (
                        "I couldn’t find strong evidence for that question in the uploaded PDF(s). "
                        "Try rephrasing or ask something more specific (method name, dataset, section title, etc.)."
                    )
                    sources = []

                # Stream effect
                thinking.empty()
                out = st.empty()
                acc = ""
                for ch in (answer or ""):
                    acc += ch
                    out.markdown(acc)
                    time.sleep(0.004)

                # Save assistant message
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer or "(no answer returned)", "sources": sources}
                )

                # Export as Markdown
                export_md = f"## Q: {prompt}\n\n## Answer\n{answer}\n\n## Sources\n"
                for i, s in enumerate(sources, 1):
                    meta = s.get("meta", {}) or {}
                    export_md += (
                        f"\n### Source {i} (page {meta.get('page','—')}, score {meta.get('score','—')})\n"
                        f"{s.get('text','')}\n"
                    )

                st.download_button(
                    "⬇️ Download this answer (Markdown)",
                    data=export_md.encode("utf-8"),
                    file_name="answer_with_sources.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
