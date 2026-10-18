"""PDF Chat Dashboard — Streamlit UI for the simple RAG pipeline in rag.py."""

import html
import os
from datetime import datetime
from typing import List

import streamlit as st

try:  # optional: read .env before rag/providers read their settings
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from providers import (  # noqa: E402
    ALL_PROVIDERS,
    NOT_FOUND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROVIDER_ENV_VARS,
    PROVIDER_OLLAMA,
    PROVIDER_SIGNUP_URLS,
    build_context,
    cite_label,
    ollama_models,
    resolve_api_key,
    simple_local_answer,
    sources_footer,
    stream_answer,
)
from rag import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    TESSDATA,
    RAGStore,
    build_indexes,
    chunk_blocks,
    delete_store,
    ensure_compatible,
    extract_pdf,
    file_hash,
    list_saved_indexes,
    load_embedding_model,
    load_reranker,
    load_store,
    save_store,
    search,
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PDF Chat & Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CSS ====================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    :root {
        --primary-color: #1e40af;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.75rem; font-weight: 700; }
    .stat-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid var(--secondary-color);
        transition: transform 0.3s ease;
    }
    .stat-card:hover { transform: translateY(-4px); }
    .stat-card h3 { margin: 0; font-size: 2.25rem; color: var(--primary-color); font-weight: 700; }
    .reference-summary {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid var(--secondary-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid var(--secondary-color);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1.25rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid var(--success-color);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1.25rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        white-space: nowrap;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4);
    }
    .stButton button:hover, .stButton button:focus, .stButton button:active { color: white; }
    .stButton button:disabled { color: rgba(255, 255, 255, 0.85); opacity: 0.45; }
    [data-testid="stSidebar"] .stButton button { padding: 0.5rem 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== SESSION STATE ====================
if "store" not in st.session_state:
    st.session_state.store = RAGStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "seen_uploads" not in st.session_state:
    st.session_state.seen_uploads = set()  # uploader file_ids already processed for the current store


# ==================== CACHED RESOURCES ====================
@st.cache_resource(show_spinner=f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Loading reranker...")
def get_reranker():
    return load_reranker()


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models():
    return ollama_models()


# ==================== HELPERS ====================
def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def ingest_pdf(file_name: str, file_bytes: bytes, store: RAGStore, notes: list) -> int:
    """Extract (in memory, OCR fallback) → structure-aware chunks → store. Returns # chunks added."""
    blocks, info = extract_pdf(file_bytes)
    chunks, chunk_meta = chunk_blocks(blocks)
    if not chunks:
        hint = " (scanned PDF — install Tesseract for OCR)" if info["ocr_unavailable"] else ""
        notes.append(("warning", f"⚠️ No text extracted from {file_name}{hint}"))
        return 0
    text = "\n".join(chunks)
    store.add_document(
        file_name,
        chunks,
        {
            "total_text_length": len(text),
            "num_chunks": len(chunks),
            "sample_text": text[:500],
            "pages": info["pages"],
            "ocr_pages": info["ocr_pages"],
            "tables": info["tables"],
            "sha1": file_hash(file_bytes),
        },
        chunk_meta,
    )
    if info["ocr_unavailable"]:
        notes.append(("warning", f"⚠️ {file_name}: {len(info['ocr_unavailable'])} scanned page(s) skipped — Tesseract not found."))
    return len(chunks)


def render_sources(sources: List[dict]) -> None:
    with st.expander(f"📌 Sources ({len(sources)} found)"):
        for i, s in enumerate(sources, 1):
            where = html.escape(cite_label(s["doc"], s.get("page")))
            section = f" · {html.escape(s['section'][:80])}" if s.get("section") else ""
            kind = f" · {s['type']}" if s.get("type", "text") != "text" else ""
            st.markdown(
                f"""<div class="reference-summary">
                    <h4>[{i}] {where}{section}{kind} — Relevance: {s['score']:.0%}</h4>
                    <p dir="auto" style="white-space: pre-wrap;">{html.escape(s['text'][:400])}{'...' if len(s['text']) > 400 else ''}</p>
                </div>""",
                unsafe_allow_html=True,
            )


def chat_to_markdown(history: list) -> str:
    lines = [f"# Chat Transcript\n\n_Generated {datetime.now().isoformat(timespec='seconds')}_\n"]
    for msg in history:
        lines.append(f"## {msg['role'].capitalize()}\n\n{msg['content']}\n")
    return "\n".join(lines)


def chat_to_pdf_bytes(history: list):
    """Return PDF bytes if fpdf2 is installed, else None."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Chat Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.now().isoformat(timespec='seconds')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for msg in history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, msg["role"].capitalize(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        # Default core fonts only support latin-1; replace anything else gracefully.
        safe = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(2)
    return bytes(pdf.output(dest="S"))


# ==================== HEADER ====================
st.markdown(
    """
<div class="main-header">
    <h1>📄 PDF Chat Dashboard</h1>
    <p>Simple RAG — chat with your PDFs using local embeddings, hybrid retrieval, and free or local LLMs</p>
</div>
""",
    unsafe_allow_html=True,
)

store: RAGStore = st.session_state.store

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ AI Settings")

    ai_provider = st.selectbox("🤖 AI Provider", ALL_PROVIDERS, help="Choose how answers are generated")

    api_key_input = ""
    if ai_provider in PROVIDER_ENV_VARS:
        env_var = PROVIDER_ENV_VARS[ai_provider]
        env_loaded = bool(os.environ.get(env_var))
        api_key_input = st.text_input(
            f"API Key for {ai_provider.split('(')[0].strip()}",
            type="password",
            help=(
                f"Leave empty to use the {env_var} environment variable"
                if env_loaded
                else "Get a free API key from the provider's website"
            ),
        )
        if env_loaded and not api_key_input:
            st.success(f"🔐 Using {env_var} from environment")
        elif ai_provider in PROVIDER_SIGNUP_URLS:
            st.info(f"🔗 Get a free key: {PROVIDER_SIGNUP_URLS[ai_provider]}")
    ollama_model = OLLAMA_MODEL
    if ai_provider == PROVIDER_OLLAMA:
        models = get_ollama_models()
        if models is None:
            st.error(f"🦙 Ollama not reachable at {OLLAMA_BASE_URL} — run `ollama serve`.")
        elif not models:
            st.warning(f"🦙 Ollama is running but has no models — run `ollama pull {OLLAMA_MODEL}`.")
        else:
            default = next((i for i, m in enumerate(models) if m.split(":")[0] == OLLAMA_MODEL.split(":")[0]), 0)
            ollama_model = st.selectbox("🦙 Ollama model", models, index=default)
            st.success(f"🦙 Ollama connected ({len(models)} model{'s' if len(models) != 1 else ''})")

    st.markdown("---")
    st.markdown("### 🎯 Retrieval")
    use_hybrid = st.checkbox(
        "Hybrid search (BM25 + dense, RRF)",
        value=True,
        help="Fuse keyword (BM25) and semantic rankings with Reciprocal Rank Fusion — better for names, codes and numbers.",
    )
    use_reranker = st.checkbox(
        "Rerank with cross-encoder",
        value=False,
        help="Re-scores the top candidates with a small multilingual cross-encoder. More precise, a bit slower.",
    )
    similarity_threshold = st.slider(
        "Minimum Relevance", 0.0, 1.0, 0.15 if use_reranker else 0.35, 0.05,
        help="Evidence floor. Passages below it are ignored; if none pass, the app says the answer wasn't found.",
    )
    num_results = st.slider("Number of Results", 1, 20, 6)

    doc_filter: List[str] = []
    if store.documents:
        doc_filter = st.multiselect(
            "Limit to documents",
            options=store.documents,
            default=store.documents,
            help="Restrict retrieval to a subset of loaded PDFs.",
        )

    st.markdown("---")
    st.markdown("### 💾 Index Library")
    saved = list_saved_indexes()
    selected_saved = st.selectbox("Saved indexes", ["—"] + saved) if saved else "—"
    col_load, col_del = st.columns(2)
    with col_load:
        if st.button("Load", disabled=(selected_saved == "—")):
            loaded = load_store(selected_saved)
            with st.spinner("Checking index compatibility..."):
                if ensure_compatible(loaded, get_embedding_model()):
                    st.toast(f"Re-embedded '{selected_saved}' with {EMBEDDING_MODEL_NAME}")
            st.session_state.store = loaded
            st.session_state.seen_uploads = set()
            st.success(f"Loaded '{selected_saved}'")
            st.rerun()
    with col_del:
        if st.button("Delete", disabled=(selected_saved == "—")):
            delete_store(selected_saved)
            st.success(f"Deleted '{selected_saved}'")
            st.rerun()

    save_name = st.text_input("Save current as", placeholder="my-index")
    if st.button("💾 Save Index", disabled=(not save_name or store.index is None)):
        try:
            folder = save_store(store, save_name.strip())
            st.success(f"Saved to {folder}")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    st.markdown("---")
    st.markdown("### 📊 Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks", len(store.chunks))
    with c2:
        st.metric("PDFs", len(store.documents))
    st.metric("Searches", len(st.session_state.search_history))

    if store.pdf_metadata:
        with st.expander("📄 PDF Details"):
            for name, meta in store.pdf_metadata.items():
                details = [plural(meta["num_chunks"], "chunk"), f"{meta['total_text_length']:,} chars"]
                if meta.get("pages"):
                    details.insert(0, plural(meta["pages"], "page"))
                if meta.get("tables"):
                    details.append(plural(meta["tables"], "table"))
                if meta.get("ocr_pages"):
                    details.append(f"OCR on {plural(len(meta['ocr_pages']), 'page')}")
                st.write(f"**{name}** — " + ", ".join(details))
    st.caption(f"Embeddings: `{EMBEDDING_MODEL_NAME}` · OCR: {'Tesseract ✓' if TESSDATA else 'not available'}")

# ==================== STATS CARDS ====================
stat_cards = [
    (f"{len(store.chunks):,}", "Text Chunks"),
    (str(len(store.documents)), "PDFs Loaded"),
    (str(len(st.session_state.chat_history)), "Messages"),
    (str(len(st.session_state.search_history)), "Searches"),
]
for col, (value, label) in zip(st.columns(4), stat_cards):
    with col:
        st.markdown(
            f"""<div class="stat-card"><h3>{value}</h3><p>{label}</p></div>""",
            unsafe_allow_html=True,
        )
st.markdown("<br>", unsafe_allow_html=True)

# ==================== PDF UPLOAD ====================
st.markdown("### 📤 Upload PDF Documents")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    added_any = False
    notes: list = []
    for uploaded_file in uploaded_files:
        if uploaded_file.file_id in st.session_state.seen_uploads:
            continue
        st.session_state.seen_uploads.add(uploaded_file.file_id)
        if store.has_document(uploaded_file.name):
            notes.append(("info", f"⏭️ Already loaded: {uploaded_file.name}"))
            continue
        data = uploaded_file.getvalue()
        duplicate = store.find_by_hash(file_hash(data))
        if duplicate:
            notes.append(("info", f"⏭️ {uploaded_file.name} is identical to already-loaded {duplicate}"))
            continue
        try:
            with st.spinner(f"Extracting {uploaded_file.name}..."):
                n = ingest_pdf(uploaded_file.name, data, store, notes)
        except Exception as exc:
            notes.append(("error", f"❌ Could not read {uploaded_file.name}: {exc}"))
            continue
        if n:
            meta = store.pdf_metadata[uploaded_file.name]
            extra = f", OCR on {plural(len(meta['ocr_pages']), 'page')}" if meta["ocr_pages"] else ""
            extra += f", {plural(meta['tables'], 'table')}" if meta["tables"] else ""
            notes.append(("success", f"✅ Extracted {plural(n, 'chunk')} from {uploaded_file.name} ({plural(meta['pages'], 'page')}{extra})"))
            added_any = True

    if added_any:
        new_chunks = len(store.chunks) - (store.index.ntotal if store.index is not None else 0)
        with st.spinner(f"Embedding {new_chunks} new chunks and rebuilding BM25..."):
            build_indexes(store, get_embedding_model())
        notes.append(("box", f"✅ {plural(len(store.chunks), 'chunk')} indexed across {plural(len(store.documents), 'PDF')}."))
    if notes:
        # Rerun so the sidebar and stats (drawn above this section) reflect the new documents.
        st.session_state.upload_notes = notes
        st.rerun()

notes_slot = st.empty()  # always present, so last run's notes clear immediately instead of lingering faded
upload_notes = st.session_state.pop("upload_notes", [])
if upload_notes:
    with notes_slot.container():
        for kind, text in upload_notes:
            if kind == "box":
                st.markdown(f"<div class='success-box'><strong>{text}</strong></div>", unsafe_allow_html=True)
            else:
                getattr(st, kind)(text)

# ==================== CHAT ====================
if store.index is not None and store.chunks:
    st.markdown("### 💬 Chat with Your Documents")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    if prompt := st.chat_input("Ask anything about your PDFs..."):
        if prompt not in st.session_state.search_history:
            st.session_state.search_history.append(prompt)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving..."):
                try:
                    results = search(
                        store,
                        get_embedding_model(),
                        prompt,
                        k=num_results,
                        hybrid=use_hybrid,
                        doc_filter=doc_filter or None,
                        min_score=similarity_threshold,
                        reranker=get_reranker() if use_reranker else None,
                    )
                except Exception as exc:
                    st.error(f"❌ Retrieval error: {exc}")
                    results = []

            if not results:
                # Grounding: never let the LLM answer without evidence.
                msg = (
                    f"{NOT_FOUND}\n\n_No passage reached the {similarity_threshold:.0%} relevance floor"
                    f"{' in the selected documents' if doc_filter and len(doc_filter) < len(store.documents) else ''}. "
                    "Try rephrasing, lowering Minimum Relevance, or widening the document filter._"
                )
                st.markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            else:
                context, n_used = build_context(results)
                results = results[:n_used]
                api_key = resolve_api_key(ai_provider, api_key_input)
                if ai_provider in PROVIDER_ENV_VARS and not api_key:
                    st.warning("⚠️ No API key provided, using simple extraction")

                try:
                    answer = st.write_stream(stream_answer(ai_provider, prompt, context, api_key, ollama_model))
                except Exception as exc:
                    st.error(f"❌ AI Error: {exc} — falling back to extractive answer.")
                    answer = simple_local_answer(prompt, context)
                    st.markdown(answer)

                answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                footer = sources_footer(answer, results)
                if footer:
                    st.markdown(footer)
                sources = [
                    {k: r[k] for k in ("doc", "page", "section", "type", "score", "text")} for r in results
                ]
                render_sources(sources)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer + footer, "sources": sources}
                )

    # ----- Chat actions -----
    st.markdown("---")
    action_cols = st.columns([1, 1, 1, 3])
    with action_cols[0]:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with action_cols[1]:
        st.download_button(
            "⬇️ Markdown",
            data=chat_to_markdown(st.session_state.chat_history),
            file_name=f"chat-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            disabled=not st.session_state.chat_history,
        )
    with action_cols[2]:
        pdf_bytes = chat_to_pdf_bytes(st.session_state.chat_history) if st.session_state.chat_history else None
        st.download_button(
            "⬇️ PDF",
            data=pdf_bytes if pdf_bytes is not None else b"",
            file_name=f"chat-{datetime.now():%Y%m%d-%H%M%S}.pdf",
            mime="application/pdf",
            disabled=(pdf_bytes is None),
            help="Install fpdf2 to enable PDF export" if pdf_bytes is None else "Download chat as PDF",
        )

else:
    st.markdown(
        """
    <div class="info-box">
        <strong>👆 Upload PDFs or load a saved index to start.</strong><br>
        The pipeline extracts (with OCR for scanned pages), chunks, embeds, indexes (FAISS + BM25) and lets you chat with page-level citations.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ==================== FOOTER ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 12px; padding: 2rem; text-align: center;">
    <p style='font-size: 1.1rem; font-weight: 600;'>📄 PDF Chat Dashboard</p>
    <p style='font-size: 0.9rem; color: #6b7280;'>Open-source · MIT Licensed</p>
</div>
""",
    unsafe_allow_html=True,
)
