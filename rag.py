"""Simple RAG core: PDF extraction, chunking, embedding, hybrid retrieval, and on-disk persistence.

Framework-agnostic — no Streamlit imports here so the pipeline stays portable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 220))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 40))
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "eng+ara")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
STORE_VERSION = 2
RRF_K = 60


# ==================== TEXT NORMALIZATION ====================
_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")  # incl. tatweel
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Function words only add BM25 noise (they match everything); already in normalized form.
_STOPWORDS = set(
    "a an the of to in on at by for from with and or but not no is are was were be been being do does did "
    "has have had it its this that these those there their they them he she his her we our you your i me my "
    "what which who whom whose when where why how many much can could will would shall should may might must "
    "as if than then so such into about over under per also any all each some more most other only very "
    "في من الي علي عن مع هذا هذه ذلك تلك التي الذي الذين هو هي هم ان او ما ماذا لا لم لن كل قد كان كانت "
    "ثم بين عند حتي كم هل اين متي كيف لماذا اي بعد قبل".split()
)
_PAGE_NUMBER = re.compile(r"^\s*(page|صفحة)?\s*\d{1,4}(\s*(/|of|من)\s*\d{1,4})?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """NFKC folds Arabic presentation forms, ligatures and NBSPs into plain characters."""
    text = unicodedata.normalize("NFKC", text).replace("\xad", "-")
    text = _INVISIBLE.sub("", text)
    text = re.sub(r"\s+([ً-ٰٟ])", r"\1", text)  # re-attach detached Arabic marks
    return re.sub(r"[ \t]+", " ", text).strip()


def detect_lang(text: str) -> str:
    ar, la = len(_ARABIC_CHARS.findall(text)), len(_LATIN_CHARS.findall(text))
    if ar + la == 0:
        return "other"
    ratio = ar / (ar + la)
    return "ar" if ratio > 0.8 else "en" if ratio < 0.2 else "mixed"


def tokenize(text: str) -> List[str]:
    """BM25 tokens: lowercase words, Arabic letter variants unified, definite article stripped."""
    text = _ARABIC_DIACRITICS.sub("", text.lower())
    text = re.sub("[إأآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه").translate(_DIGITS)
    tokens = []
    for tok in re.findall(r"\w+", text):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.startswith(("ال", "وال")):
            tok = tok[2:] if tok.startswith("ال") else tok[3:]
        tokens.append(tok)
    return tokens


# ==================== PDF EXTRACTION ====================
def _find_tessdata() -> Optional[str]:
    candidates = [
        os.environ.get("TESSDATA_PREFIX", ""),
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
    ]
    for path in candidates:
        if path and Path(path, "eng.traineddata").exists():
            return path
    try:
        return pymupdf.get_tessdata()
    except Exception:
        return None


TESSDATA = _find_tessdata()


def _needs_ocr(page: pymupdf.Page, text: str) -> bool:
    """Scanned page: (almost) no extractable text, or text that is mostly unmapped glyphs."""
    visible = re.sub(r"\s", "", text)
    if len(visible) < 25:
        return bool(page.get_images()) or len(visible) == 0
    return visible.count("\ufffd") / len(visible) > 0.3


def _table_markdown(rows: List[List[Optional[str]]]) -> str:
    rows = [[clean_text(c or "").replace("\n", " ").replace("|", "/") for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _join_lines(lines: List[str], ocr: bool) -> str:
    """Join a block's lines into one paragraph; keep line-end hyphens attached; drop OCR speckle lines."""
    out = ""
    for line in lines:
        line = clean_text(line)
        if not line or (ocr and sum(ch.isalnum() for ch in line) < 3):
            continue
        out = out + line if out.endswith("-") else f"{out} {line}" if out else line
    return out


def _reading_order(blocks: List[dict], width: float, rtl: bool) -> List[dict]:
    """Order blocks column by column. Full-width blocks split the page into horizontal bands."""
    mid, tol = width / 2, width * 0.05

    def column(b: dict) -> int:
        x0, _, x1, _ = b["bbox"]
        if x1 <= mid + tol:
            return 0
        if x0 >= mid - tol:
            return 1
        return -1

    cols = [column(b) for b in blocks]
    if cols.count(0) < 2 or cols.count(1) < 2:  # single-column page
        return sorted(blocks, key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))

    ordered, band = [], []
    first, second = (1, 0) if rtl else (0, 1)

    def flush():
        for c in (first, second):
            ordered.extend(sorted((b for b in band if column(b) == c), key=lambda b: b["bbox"][1]))
        band.clear()

    for b in sorted(blocks, key=lambda b: b["bbox"][1]):
        if column(b) == -1:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def extract_pdf(data: bytes) -> Tuple[List[dict], dict]:
    """Return (blocks, info). Each block: {page, text, kind: text|heading|table, ocr, size, bold}.

    PyMuPDF text layer first; per-page OCR fallback for scanned pages; tables → Markdown.
    """
    doc = pymupdf.open(stream=data, filetype="pdf")
    if doc.needs_pass and not doc.authenticate(""):
        raise ValueError("PDF is password-protected")

    blocks: List[dict] = []
    info = {"pages": doc.page_count, "ocr_pages": [], "ocr_unavailable": [], "tables": 0}
    flags = (pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES) | pymupdf.TEXT_DEHYPHENATE

    for page in doc:
        pno = page.number + 1
        textpage, ocr = None, False
        if _needs_ocr(page, page.get_text()):
            if TESSDATA:
                try:
                    textpage = page.get_textpage_ocr(flags=flags, language=OCR_LANGUAGES, dpi=300, full=True, tessdata=TESSDATA)
                    ocr = True
                    info["ocr_pages"].append(pno)
                except Exception:
                    info["ocr_unavailable"].append(pno)
            else:
                info["ocr_unavailable"].append(pno)

        page_blocks: List[dict] = []
        table_boxes: List[pymupdf.Rect] = []
        if not ocr and len(page.get_cdrawings()) >= 4:  # ruled tables need vector lines; skips text-only pages
            try:
                for table in page.find_tables().tables:
                    if table.row_count < 2 or table.col_count < 2:
                        continue
                    md = _table_markdown(table.extract())
                    if md:
                        table_boxes.append(pymupdf.Rect(table.bbox))
                        page_blocks.append({"bbox": tuple(table.bbox), "text": md, "kind": "table", "size": 0, "bold": False})
            except Exception:
                pass  # table detection is best-effort

        d = page.get_text("dict", flags=flags, textpage=textpage)
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            rect = pymupdf.Rect(b["bbox"])
            center = (rect.tl + rect.br) / 2
            if any(center in tb for tb in table_boxes):
                continue
            spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            text = _join_lines(["".join(s["text"] for s in line["spans"]) for line in b["lines"]], ocr)
            if not spans or not text or _PAGE_NUMBER.match(text):
                continue
            page_blocks.append({
                "bbox": tuple(b["bbox"]),
                "text": text,
                "kind": "text",
                "size": round(max(s["size"] for s in spans), 1),
                "bold": all(s["flags"] & 16 for s in spans),
                "lines": len(b["lines"]),
            })

        rtl = detect_lang(" ".join(b["text"] for b in page_blocks)) == "ar"
        height = page.rect.height
        for b in _reading_order(page_blocks, page.rect.width, rtl):
            b.update(page=pno, ocr=ocr, margin=b["bbox"][3] < height * 0.12 or b["bbox"][1] > height * 0.88)
            blocks.append(b)

    blocks = _drop_running_headers(blocks, doc.page_count)
    _mark_headings(blocks)
    info["tables"] = sum(b["kind"] == "table" for b in blocks)
    return blocks, info


def _drop_running_headers(blocks: List[dict], pages: int) -> List[dict]:
    """Remove headers/footers: text repeated on many pages (digits ignored), or repeated in the page margins."""
    if pages < 3:
        return blocks
    key = lambda b: re.sub(r"\d+", "#", b["text"].lower())  # noqa: E731
    seen: dict = {}
    for b in blocks:
        seen.setdefault(key(b), set()).add(b["page"])
    return [
        b for b in blocks
        if not (len(seen[key(b)]) >= max(3, pages * 0.2) or (b["margin"] and len(seen[key(b)]) >= 3))
    ]


def _mark_headings(blocks: List[dict]) -> None:
    """Short blocks set in a larger (or bold) font than the body text become section headings."""
    sized = [(b["size"], len(b["text"])) for b in blocks if b["kind"] == "text" and not b["ocr"]]
    if not sized:
        return
    # Body size = the font size carrying the most characters.
    weights: dict = {}
    for size, n in sized:
        weights[size] = weights.get(size, 0) + n
    body = max(weights, key=weights.get)
    body_bold = statistics.mean(b["bold"] for b in blocks if b["kind"] == "text" and not b["ocr"]) > 0.5
    for b in blocks:
        if b["kind"] != "text" or b["ocr"]:
            continue
        short = len(b["text"]) <= 150 and b.get("lines", 1) <= 3 and not b["text"].endswith((".", ":", "،"))
        bigger = b["size"] >= body * 1.15
        if short and (bigger or (b["bold"] and not body_bold)) and re.search(r"\w", b["text"]):
            b["kind"] = "heading"


# ==================== CHUNKING ====================
def _pack(paragraphs: List[str], size: int, overlap: int) -> List[str]:
    """Greedily pack paragraphs into ~size-word chunks; long paragraphs are windowed with overlap."""
    chunks: List[str] = []
    buf: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(buf) + len(words) > size and buf:
            chunks.append(" ".join(buf))
            buf = buf[-overlap:] if overlap else []
        buf.extend(words)
        while len(buf) > size:
            chunks.append(" ".join(buf[:size]))
            buf = buf[max(1, size - overlap):]
    if buf and (not chunks or len(buf) > overlap):
        chunks.append(" ".join(buf))
    return chunks


def _split_table(md: str, size: int) -> List[str]:
    """Split long tables by rows, repeating the header row in every piece."""
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]
    pieces, cur = [], []
    for row in rows:
        if cur and len(" ".join(header + cur + [row]).split()) > size:
            pieces.append("\n".join(header + cur))
            cur = []
        cur.append(row)
    if cur or not pieces:
        pieces.append("\n".join(header + cur))
    return pieces


def chunk_blocks(
    blocks: List[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> Tuple[List[str], List[dict]]:
    """Structure-aware chunking. Chunks never span pages; a new section starts a new chunk unless the
    current one is still tiny; tables stay whole (split by rows with the header repeated)."""
    chunks: List[str] = []
    metas: List[dict] = []
    section, chunk_section = "", ""
    paragraphs: List[str] = []
    cur_page, cur_ocr = None, False
    min_words = max(chunk_size // 5, 8)

    def emit(text: str, kind: str, sec: str) -> None:
        if kind != "table" and (len(text.split()) < 4 or text == sec):
            return  # stray fragment or a heading with no body on this page
        chunks.append(text)
        metas.append({"page": cur_page, "section": sec, "type": kind, "lang": detect_lang(text)})

    def flush() -> None:
        for text in _pack(paragraphs, chunk_size, chunk_overlap):
            emit(text, "ocr" if cur_ocr else "text", chunk_section)
        paragraphs.clear()

    def add(text: str) -> None:
        nonlocal chunk_section
        if not paragraphs:
            chunk_section = section
        paragraphs.append(text)

    for b in blocks:
        if b["page"] != cur_page:
            flush()
            cur_page, cur_ocr = b["page"], b["ocr"]
        if b["kind"] == "heading":
            if sum(len(p.split()) for p in paragraphs) >= min_words:
                flush()
            section = b["text"][:150]
            add(b["text"])  # heading stays in the text so merged small sections keep their titles
        elif b["kind"] == "table":
            flush()
            for piece in _split_table(b["text"], chunk_size):
                emit(piece, "table", section)
        else:
            add(b["text"])
    flush()
    return chunks, metas


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split plain text into overlapping word-based chunks (kept for non-PDF callers)."""
    return _pack([text], chunk_size, chunk_overlap) if text.split() else []


# ==================== STORE ====================
@dataclass
class RAGStore:
    """In-memory store for chunks, their source document/page metadata, and parallel indexes."""

    chunks: List[str] = field(default_factory=list)
    doc_names: List[str] = field(default_factory=list)
    chunk_meta: List[dict] = field(default_factory=list)
    pdf_metadata: dict = field(default_factory=dict)
    embedding_model: str = ""
    index: Optional[faiss.Index] = None
    bm25: Optional[BM25Okapi] = None

    @property
    def documents(self) -> List[str]:
        return sorted(set(self.doc_names))

    def has_document(self, name: str) -> bool:
        return name in self.pdf_metadata

    def find_by_hash(self, sha1: str) -> Optional[str]:
        return next((n for n, m in self.pdf_metadata.items() if m.get("sha1") == sha1), None)

    def add_document(self, doc_name: str, chunks: List[str], meta: dict, chunk_meta: Optional[List[dict]] = None) -> None:
        self.chunks.extend(chunks)
        self.doc_names.extend([doc_name] * len(chunks))
        self.chunk_meta.extend(chunk_meta or [{"page": None, "section": "", "type": "text"} for _ in chunks])
        self.pdf_metadata[doc_name] = meta

    def index_text(self, i: int) -> str:
        """Text that gets embedded / BM25-indexed: section heading gives each chunk its context."""
        section = self.chunk_meta[i].get("section") if i < len(self.chunk_meta) else ""
        text = self.chunks[i]
        return f"{section}\n{text}" if section and not text.startswith(section) else text


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = min(model.max_seq_length or 512, 512)  # chunks are ~220 words; caps CPU cost
    return model


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME, max_length=512)


def _embed(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return np.asarray(
        model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )


def build_indexes(store: RAGStore, model: SentenceTransformer) -> int:
    """Incrementally embed chunks not yet in FAISS; rebuild BM25 (cheap). Returns # newly embedded."""
    if not store.chunks:
        store.index, store.bm25 = None, None
        return 0

    start = store.index.ntotal if store.index is not None else 0
    if start > len(store.chunks):
        store.index, start = None, 0
    new = [store.index_text(i) for i in range(start, len(store.chunks))]
    if new:
        embeddings = _embed(model, new)
        if store.index is None:
            store.index = faiss.IndexFlatIP(embeddings.shape[1])
        store.index.add(embeddings)
    store.embedding_model = EMBEDDING_MODEL_NAME
    store.bm25 = BM25Okapi([tokenize(store.index_text(i)) or ["\u2205"] for i in range(len(store.chunks))])
    return len(new)


def ensure_compatible(store: RAGStore, model: SentenceTransformer) -> bool:
    """Re-embed a loaded store if it was built with another model or is out of sync. Returns True if rebuilt."""
    dim = model.get_sentence_embedding_dimension()
    ok = (
        store.index is not None
        and store.embedding_model == EMBEDDING_MODEL_NAME
        and store.index.d == dim
        and store.index.ntotal == len(store.chunks)
    )
    if ok:
        return False
    store.index = None
    build_indexes(store, model)
    return True


# ==================== RETRIEVAL ====================
def search(
    store: RAGStore,
    model: SentenceTransformer,
    query: str,
    k: int,
    *,
    hybrid: bool = True,
    doc_filter: Optional[List[str]] = None,
    min_score: float = 0.0,
    reranker: Optional[CrossEncoder] = None,
) -> List[dict]:
    """Dense + BM25 fused with Reciprocal Rank Fusion, optionally cross-encoder reranked.

    Returns dicts {idx, text, doc, page, section, type, score, dense} sorted by relevance.
    ``min_score`` is an evidence floor on dense cosine similarity (or on the reranker score when used).
    """
    if not store.chunks or store.index is None:
        return []

    n = len(store.chunks)
    allowed = np.ones(n, dtype=bool)
    if doc_filter:
        allowed = np.array([d in doc_filter for d in store.doc_names])
        if not allowed.any():
            return []

    query_emb = _embed(model, [query])
    dense_scores, dense_idx = store.index.search(query_emb, n)
    dense = {int(i): float(s) for i, s in zip(dense_idx[0], dense_scores[0]) if i >= 0}
    dense_rank = [i for i in map(int, dense_idx[0]) if i >= 0 and allowed[i]]

    fused = {i: 1.0 / (RRF_K + r + 1) for r, i in enumerate(dense_rank)}
    query_tokens = tokenize(query)
    if hybrid and store.bm25 is not None and query_tokens:
        bm25 = np.asarray(store.bm25.get_scores(query_tokens))
        bm25_rank = [int(i) for i in np.argsort(-bm25) if allowed[i] and bm25[i] > 0]
        for r, i in enumerate(bm25_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

    ranked = sorted(fused, key=fused.get, reverse=True)

    if reranker is not None:
        candidates = ranked[: max(k * 3, 20)]
        scores = reranker.predict([(query, store.index_text(i)) for i in candidates], show_progress_bar=False)
        probs = np.asarray(scores, dtype="float64")
        if probs.min() < 0 or probs.max() > 1:  # raw logits (models without a sigmoid head)
            probs = 1.0 / (1.0 + np.exp(-probs))
        order = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        picked = [(i, float(p)) for i, p in order if p >= min_score][:k]
    else:
        picked = [(i, dense.get(i, 0.0)) for i in ranked if dense.get(i, 0.0) >= min_score][:k]

    results = []
    for i, score in picked:
        meta = store.chunk_meta[i] if i < len(store.chunk_meta) else {}
        results.append({
            "idx": i,
            "text": store.chunks[i],
            "doc": store.doc_names[i],
            "page": meta.get("page"),
            "section": meta.get("section", ""),
            "type": meta.get("type", "text"),
            "score": score,
            "dense": dense.get(i, 0.0),
        })
    return results


# ==================== PERSISTENCE ====================
def save_store(store: RAGStore, name: str) -> Path:
    """Persist FAISS index + chunks + metadata under data/<name>/."""
    if store.index is None:
        raise ValueError("Cannot save an empty store.")
    name = re.sub(r"[^\w\-. ]", "_", name).strip() or "index"
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(folder / "faiss.index"))
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "embedding_model": store.embedding_model,
                "chunks": store.chunks,
                "doc_names": store.doc_names,
                "chunk_meta": store.chunk_meta,
                "pdf_metadata": store.pdf_metadata,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return folder


def load_store(name: str) -> RAGStore:
    """Load a saved store (v1 stores without page metadata still load). BM25 is rebuilt from chunks.

    Call ``ensure_compatible`` afterwards so an index built with a different embedding model is re-embedded.
    """
    folder = DATA_DIR / name
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    chunks = meta["chunks"]
    store = RAGStore(
        chunks=chunks,
        doc_names=meta["doc_names"],
        chunk_meta=meta.get("chunk_meta") or [{"page": None, "section": "", "type": "text"} for _ in chunks],
        pdf_metadata=meta["pdf_metadata"],
        embedding_model=meta.get("embedding_model", ""),
    )
    store.index = faiss.read_index(str(folder / "faiss.index"))
    if store.chunks:
        store.bm25 = BM25Okapi([tokenize(store.index_text(i)) or ["\u2205"] for i in range(len(store.chunks))])
    return store


def list_saved_indexes() -> List[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / "faiss.index").exists()
    )


def delete_store(name: str) -> None:
    folder = DATA_DIR / name
    if folder.exists():
        shutil.rmtree(folder)
