# DocuMind AI

> A modern, privacy-first Retrieval-Augmented Generation (RAG) system. Upload multi-page technical manuals, research papers, contracts, and data tables to get grounded, verifiable answers with exact page citations — accelerated by local Ollama LLMs and hybrid FAISS + BM25 search.

**Author:** Abhijeet ([@abhijeetandchoudhary-dev](https://github.com/abhijeetandchoudhary-dev))  
**License:** [MIT License](LICENSE)  
**Technologies:** Python, Streamlit, Ollama, PyMuPDF, FAISS, BM25, FPDF2  

---

## 🌟 Key Highlights

- **Multi-Column & Table-Aware Extraction:** Uses PyMuPDF layout analysis to parse multi-column reading order and transform tabular data into clean Markdown format.
- **Structure-Aware Chunking:** Paragraphs never cross page boundaries, ensuring strict, reliable page number attribution.
- **Dual Hybrid Retrieval (RRF):** Fuses dense semantic vector matching (FAISS `IndexFlatIP`) and sparse keyword matching (BM25Okapi) using Reciprocal Rank Fusion ($k=60$).
- **Anti-Hallucination Evidence Floor:** Queries that lack sufficient grounding in the uploaded files are declined automatically before spending inference cycles.
- **100% Offline & Private:** Powered locally by Ollama (`llama3.2:1b` and `nomic-embed-text`) with zero data leaving your machine.
- **Modern Split-Studio Interface:** High-contrast, responsive two-column dashboard with real-time token streaming and prompt chips.
- **Index Library Persistence:** Save and reload processed document vector stores without re-extracting or re-embedding.
- **Transcript Export:** Export verified conversations with source references to Markdown and formatted PDF.

---

## 🏗️ Architecture

```
                       ┌──────────────────────────┐
        PDF upload ───►│  PyMuPDF (Tables, OCR)   │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Structure-Aware Chunks  │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────┴─────────────┐
                       ▼                          ▼
              ┌──────────────────┐      ┌──────────────────┐
              │ Local Embeddings │      │  BM25 Tokens     │
              │(nomic-embed-text)│      │  (rank-bm25)     │
              └────────┬─────────┘      └────────┬─────────┘
                       ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │ FAISS IndexFlatIP│      │  BM25 Sparse Idx │
              └────────┬─────────┘      └────────┬─────────┘
                       └───────┬─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  RRF Hybrid Fusion   │
                    │  + Evidence Floor    │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Ollama Local LLM    │
                    │  (llama3.2:1b)       │
                    └──────────┬───────────┘
                               ▼
                        Grounded Answer
                        + Page Citations
```

**Project Layout:**

```
DocuMind-AI/
├── app.py            # Modern Streamlit Studio interface
├── rag.py            # PDF extraction, chunking, embeddings, FAISS, BM25, RRF
├── providers.py      # Local Ollama streaming and cloud fallback gateways
├── requirements.txt  # Core dependencies
├── .env.example      # Configuration template
├── .gitignore        # Git ignore rules
├── LICENSE           # MIT License (Abhijeet)
└── README.md         # Documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

```bash
# Pull the recommended lightweight models in Ollama
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 2. Clone and Setup Environment

```bash
git clone https://github.com/abhijeetandchoudhary-dev/DocuMind-AI.git
cd DocuMind-AI

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application

```bash
streamlit run app.py --server.port 8502
```

Open your browser at `http://localhost:8502`.

---

## ⚙️ Configuration (.env)

All settings are optional and can also be adjusted via the sidebar:

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Local Ollama endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model | `llama3.2:1b` |
| `EMBEDDING_MODEL` | Embedding model identifier | `nomic-embed-text` |
| `CHUNK_SIZE` | Target words per chunk | `220` |
| `CHUNK_OVERLAP` | Overlapping words between adjacent chunks | `40` |
| `MAX_CONTEXT_CHARS` | Context character limit fed to LLM | `14000` |
| `DATA_DIR` | Saved vector store directory | `data` |

---

## 📄 License

This project is open-source and released under the [MIT License](LICENSE) by **Abhijeet**.
