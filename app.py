"""
DocuMind AI: Modern Document Intelligence & RAG Studio.
Enterprise-grade interface for local document analysis and conversational intelligence.
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import List

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from providers import (
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
from rag import (
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

# ==============================================================================
# 1. PAGE & APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="DocuMind AI — Document Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. DESIGN SYSTEM & PROFESSIONAL STYLESHEET
# ==============================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --bg-main: #090d16;
        --bg-surface: #111827;
        --bg-surface-elevated: #1e293b;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-active: rgba(99, 102, 241, 0.45);
        --brand-primary: #6366f1;
        --brand-secondary: #3b82f6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-emerald: #10b981;
    }

    * { font-family: var(--font-sans); }
    code, pre { font-family: var(--font-mono) !important; }

    /* Canvas glow */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 5%, rgba(99, 102, 241, 0.09) 0%, transparent 45%),
                    radial-gradient(circle at 90% 90%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                    var(--bg-main) !important;
        color: var(--text-primary);
    }

    /* Top Workspace Header */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.8rem;
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.25rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }
    .brand-text h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .brand-text p {
        margin: 0;
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .nav-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Metric & Insight Cards */
    .metric-panel {
        background: linear-gradient(145deg, #131b2e 0%, #0d1322 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-panel:hover {
        transform: translateY(-3px);
        border-color: var(--border-active);
    }
    .metric-val {
        font-size: 1.95rem;
        font-weight: 800;
        color: #818cf8;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-top: 0.2rem;
    }

    /* Studio Card Containers */
    .studio-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .studio-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .studio-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }

    /* Source Citation Previewer */
    .citation-box {
        background: #0f172a;
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.8rem;
    }
    .citation-head {
        color: #a5b4fc;
        font-weight: 700;
        font-size: 0.92rem;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    .citation-body {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0;
    }

    /* Empty state canvas */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        background: rgba(17, 24, 39, 0.3);
    }
    .empty-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        opacity: 0.8;
    }
    .empty-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }
    .empty-subtitle {
        color: var(--text-secondary);
        font-size: 0.92rem;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Global Buttons */
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        padding: 0.65rem 1.3rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45) !important;
        color: #ffffff !important;
    }
    .stButton button:disabled {
        opacity: 0.35 !important;
        box-shadow: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. SESSION INITIALIZATION & LOGIC
# ==============================================================================
if "store" not in st.session_state:
    st.session_state.store = RAGStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "seen_uploads" not in st.session_state:
    st.session_state.seen_uploads = set()
if "current_view" not in st.session_state:
    st.session_state.current_view = "studio"

# ==============================================================================
# 4. CACHED ENGINE RESOURCES
# ==============================================================================
@st.cache_resource(show_spinner=f"Connecting to embedding service ({EMBEDDING_MODEL_NAME})...")
def get_embedding_model():
    return load_embedding_model()

@st.cache_resource(show_spinner="Preparing reranker...")
def get_reranker():
    return load_reranker()

@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models():
    return ollama_models()

def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"

def ingest_pdf(file_name: str, file_bytes: bytes, store: RAGStore, notes: list) -> int:
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
    with st.expander(f"📌 Grounded Sources ({len(sources)} verified passages)"):
        for i, s in enumerate(sources, 1):
            where = html.escape(cite_label(s["doc"], s.get("page")))
            section = f" · {html.escape(s['section'][:80])}" if s.get("section") else ""
            kind = f" · {s['type']}" if s.get("type", "text") != "text" else ""
            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-head">
                        <span>[{i}] {where}{section}{kind}</span>
                        <span style="color: #34d399;">Score: {s['score']:.0%}</span>
                    </div>
                    <p class="citation-body">{html.escape(s['text'][:450])}{'...' if len(s['text']) > 450 else ''}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

def chat_to_markdown(history: list) -> str:
    lines = [f"# PDF Intelligence Studio Transcript\n\n_Generated {datetime.now().isoformat(timespec='seconds')}_\n"]
    for msg in history:
        lines.append(f"## {msg['role'].capitalize()}\n\n{msg['content']}\n")
    return "\n".join(lines)

def chat_to_pdf_bytes(history: list):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PDF Studio Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.now().isoformat(timespec='seconds')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for msg in history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, msg["role"].capitalize(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        safe = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(2)
    return bytes(pdf.output(dest="S"))

store: RAGStore = st.session_state.store

# ==============================================================================
# 5. TOP NAVBAR & GLOBAL STATUS BANNER
# ==============================================================================
st.markdown(
    """
    <div class="top-navbar">
        <div class="brand-group">
            <div class="brand-icon">⚡</div>
            <div class="brand-text">
                <h2>DocuMind AI</h2>
                <p>Enterprise RAG &middot; Hybrid Dense/Sparse Search &middot; Grounded Citations</p>
            </div>
        </div>
        <div class="nav-status-badge">
            <span class="pulse-dot"></span>
            <span>Local Ollama Online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 6. SIDEBAR: CONTROL CENTER & REPOSITORY LIBRARY
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Engine Configuration")

    ai_provider = st.selectbox(
        "LLM Provider",
        ALL_PROVIDERS,
        index=ALL_PROVIDERS.index(PROVIDER_OLLAMA) if PROVIDER_OLLAMA in ALL_PROVIDERS else 0,
        help="Choose the model engine generating responses."
    )

    api_key_input = ""
    if ai_provider in PROVIDER_ENV_VARS:
        env_var = PROVIDER_ENV_VARS[ai_provider]
        env_loaded = bool(os.environ.get(env_var))
        api_key_input = st.text_input(
            f"{ai_provider.split('(')[0].strip()} Key",
            type="password",
            help=f"Loaded from {env_var} if empty." if env_loaded else "Enter free API key."
        )
        if env_loaded and not api_key_input:
            st.caption(f"✓ Using {env_var} from environment")
        elif ai_provider in PROVIDER_SIGNUP_URLS:
            st.caption(f"Get free key: [{ai_provider}]({PROVIDER_SIGNUP_URLS[ai_provider]})")

    ollama_model = OLLAMA_MODEL
    if ai_provider == PROVIDER_OLLAMA:
        models = get_ollama_models()
        if models is None:
            st.error(f"Ollama offline at {OLLAMA_BASE_URL}")
        elif not models:
            st.warning(f"No models installed in Ollama.")
        else:
            default_idx = next((i for i, m in enumerate(models) if "llama3.2" in m), 0)
            ollama_model = st.selectbox("Active Local Model", models, index=default_idx)
            st.caption(f"✓ Accelerated via Ollama ({len(models)} local models)")

    st.markdown("---")
    st.markdown("### 🔍 Precision Tuning")
    use_hybrid = st.checkbox("Hybrid Fusion (FAISS + BM25)", value=True, help="Fuses keyword and semantic vectors.")
    use_reranker = st.checkbox("Cross-Encoder Reranker", value=False, help="Re-scores candidates for ultra-high precision.")
    similarity_threshold = st.slider(
        "Evidence Threshold", 0.0, 1.0, 0.15 if use_reranker else 0.35, 0.05,
        help="Passages scoring below this floor will be rejected to prevent hallucinations."
    )
    num_results = st.slider("Max Retrieved Passages", 1, 15, 6)

    doc_filter: List[str] = []
    if store.documents:
        doc_filter = st.multiselect(
            "Filter Ingested Documents",
            options=store.documents,
            default=store.documents,
            help="Limit retrieval to specific documents."
        )

    st.markdown("---")
    st.markdown("### 🗄️ Index Library")
    saved = list_saved_indexes()
    selected_saved = st.selectbox("Saved Indexes", ["—"] + saved) if saved else "—"
    c_load, c_del = st.columns(2)
    with c_load:
        if st.button("Load", disabled=(selected_saved == "—"), use_container_width=True):
            loaded = load_store(selected_saved)
            with st.spinner("Verifying index vectors..."):
                if ensure_compatible(loaded, get_embedding_model()):
                    st.toast(f"Re-indexed '{selected_saved}'")
            st.session_state.store = loaded
            st.session_state.seen_uploads = set()
            st.success(f"Loaded '{selected_saved}'")
            st.rerun()
    with c_del:
        if st.button("Delete", disabled=(selected_saved == "—"), use_container_width=True):
            delete_store(selected_saved)
            st.success(f"Deleted '{selected_saved}'")
            st.rerun()

    save_name = st.text_input("Export active index as", placeholder="e.g. quantum-notes")
    if st.button("💾 Save to Library", disabled=(not save_name or store.index is None), use_container_width=True):
        try:
            folder = save_store(store, save_name.strip())
            st.success(f"Persisted to {folder}")
            st.rerun()
        except Exception as exc:
            st.error(f"Save error: {exc}")

# ==============================================================================
# 7. MAIN WORKSPACE: REIMAGINED TWO-COLUMN STUDIO LAYOUT
# ==============================================================================
col_left, col_right = st.columns([1, 1.6], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: INGESTION, REPOSITORY STATS & INSPECTOR
# ------------------------------------------------------------------------------
with col_left:
    st.markdown(
        """
        <div class="studio-card">
            <div class="studio-card-header">
                <h3 class="studio-card-title">📥 Document Intake</h3>
                <span style="font-size: 0.8rem; color: #94a3b8;">Multi-page &middot; Tables</span>
            </div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin-top: 0;">Upload technical PDFs, manuals, or research papers. Files are chunked and indexed automatically without cloud exposure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        added_any = False
        notes: list = []
        for uf in uploaded_files:
            if uf.file_id in st.session_state.seen_uploads:
                continue
            st.session_state.seen_uploads.add(uf.file_id)
            if store.has_document(uf.name):
                notes.append(("info", f"Already loaded: {uf.name}"))
                continue
            data = uf.getvalue()
            duplicate = store.find_by_hash(file_hash(data))
            if duplicate:
                notes.append(("info", f"{uf.name} matches existing {duplicate}"))
                continue
            try:
                with st.spinner(f"Extracting {uf.name}..."):
                    n = ingest_pdf(uf.name, data, store, notes)
            except Exception as exc:
                notes.append(("error", f"Error on {uf.name}: {exc}"))
                continue
            if n:
                meta = store.pdf_metadata[uf.name]
                notes.append(("success", f"Indexed {uf.name} ({meta['pages']} pages, {n} chunks)"))
                added_any = True

        if added_any:
            new_chunks = len(store.chunks) - (store.index.ntotal if store.index is not None else 0)
            with st.spinner(f"Embedding {new_chunks} passages locally..."):
                build_indexes(store, get_embedding_model())
            notes.append(("box", f"All passages embedded & indexed successfully."))
        if notes:
            st.session_state.upload_notes = notes
            st.rerun()

    upload_notes = st.session_state.pop("upload_notes", [])
    if upload_notes:
        for kind, text in upload_notes:
            if kind == "box":
                st.markdown(f"<div class='success-box'><strong>{text}</strong></div>", unsafe_allow_html=True)
            else:
                getattr(st, kind)(text)

    # Document Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
            <div class="metric-panel">
                <p class="metric-val">{len(store.documents)}</p>
                <p class="metric-label">Active PDFs</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-panel">
                <p class="metric-val">{len(store.chunks):,}</p>
                <p class="metric-label">Indexed Chunks</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if store.pdf_metadata:
        with st.expander("📑 Detailed Document Manifest", expanded=False):
            for name, meta in store.pdf_metadata.items():
                st.markdown(f"**{name}**")
                st.caption(f"Pages: {meta.get('pages', 'N/A')} · Chunks: {meta['num_chunks']} · Characters: {meta['total_text_length']:,}")
                st.markdown("---")

# ------------------------------------------------------------------------------
# RIGHT COLUMN: INTELLIGENT CHAT & EVIDENCE SYNTHESIZER
# ------------------------------------------------------------------------------
with col_right:
    if store.index is not None and store.chunks:
        st.markdown(
            """
            <div class="studio-card-header">
                <h3 class="studio-card-title">💬 Document Dialogue</h3>
                <span style="font-size: 0.8rem; color: #10b981; font-weight: 600;">● Grounded Verification Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Action Pills
        st.markdown("<p style='color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.3rem;'>Quick prompt templates:</p>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        selected_prompt = None
        with p1:
            if st.button("📌 Summarize Key Points", use_container_width=True):
                selected_prompt = "Summarize the key points and conclusions of this document."
        with p2:
            if st.button("📊 Extract Data & Tables", use_container_width=True):
                selected_prompt = "Extract all key numbers, metrics, and tabular data mentioned."
        with p3:
            if st.button("⚠️ Identify Risks & Limits", use_container_width=True):
                selected_prompt = "What limitations, risks, or caveats are highlighted?"

        # Chat history rendering
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"])

        user_input = st.chat_input("Ask any question regarding the uploaded documents...")
        prompt = selected_prompt or user_input

        if prompt:
            if prompt not in st.session_state.search_history:
                st.session_state.search_history.append(prompt)

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing grounded response..."):
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
                        st.error(f"Search failure: {exc}")
                        results = []

                if not results:
                    msg = (
                        f"{NOT_FOUND}\n\n_No passage cleared the {similarity_threshold:.0%} confidence floor. "
                        "Try lowering the evidence threshold in the sidebar or rephrasing your question._"
                    )
                    st.markdown(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                else:
                    context, n_used = build_context(results)
                    results = results[:n_used]
                    api_key = resolve_api_key(ai_provider, api_key_input)

                    try:
                        answer = st.write_stream(stream_answer(ai_provider, prompt, context, api_key, ollama_model))
                    except Exception as exc:
                        st.error(f"Inference error: {exc} — using fallback.")
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

        # Bottom Utilities
        st.markdown("<br>", unsafe_allow_html=True)
        act1, act2, act3 = st.columns([1, 1, 1])
        with act1:
            if st.button("🗑️ Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with act2:
            st.download_button(
                "⬇️ Export Markdown",
                data=chat_to_markdown(st.session_state.chat_history),
                file_name=f"studio-session-{datetime.now():%Y%m%d-%H%M%S}.md",
                mime="text/markdown",
                disabled=not st.session_state.chat_history,
                use_container_width=True,
            )
        with act3:
            pdf_bytes = chat_to_pdf_bytes(st.session_state.chat_history) if st.session_state.chat_history else None
            st.download_button(
                "⬇️ Export PDF",
                data=pdf_bytes if pdf_bytes is not None else b"",
                file_name=f"studio-session-{datetime.now():%Y%m%d-%H%M%S}.pdf",
                mime="application/pdf",
                disabled=(pdf_bytes is None),
                use_container_width=True,
            )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📂</div>
                <div class="empty-title">Document Intelligence Canvas Ready</div>
                <p class="empty-subtitle">
                    Upload one or more PDF files in the intake panel on the left to begin.
                    Your documents will be parsed, vectorized with local embeddings, and made available for instant, cited question answering.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
