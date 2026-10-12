# Version 1: Basic RAG (Retrieval-Augmented Generation)

A baseline implementation of a Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python, FAISS, Sentence Transformers, and Ollama.

---

## 📌 Overview

This project demonstrates the core fundamentals of RAG without complex framework abstractions:
1. **Ingest & Chunk**: Loads a raw document (`data_source.txt`) and splits it into sections using delimiter-based chunking (`\n##`).
2. **Embedding**: Converts text chunks into dense vector representations using `all-MiniLM-L6-v2`.
3. **Indexing**: Stores and indexes vectors in an in-memory **FAISS** vector store (`IndexFlatL2`).
4. **Retrieval**: Finds the top 5 most relevant document chunks based on Euclidean (L2) distance for any user query.
5. **Generation**: Augments an LLM prompt with retrieved chunks and generates grounded answers using a local **Ollama** model (`qwen2.5:3b`).

---

## 🏗️ Architecture & Pipeline Flow

```
[ data_source.txt ]
        │
        ▼
   split_text() (delimiter: "\n##")
        │
        ▼
SentenceTransformer ("all-MiniLM-L6-v2") ──► Chunk Embeddings
                                                    │
                                                    ▼
                                           FAISS Vector Index (L2)
                                                    │
User Query ──► SentenceTransformer Embed ──► Search (Top 5 Chunks)
                                                    │
                                                    ▼
                                           Retrieved Context
                                                    │
                                                    ▼
                                            Ollama (qwen2.5:3b)
                                           [Prompt + Ground Rules]
                                                    │
                                                    ▼
                                             Jarvis Response
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+ (managed via `uv`)
- **Embedding Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Search:** `faiss-cpu` (`IndexFlatL2`)
- **Local LLM Runner:** [Ollama](https://ollama.ai/)
- **LLM Model:** `qwen2.5:3b`
- **Array Math:** `numpy`

---

## 📁 Project Structure

```
Version_1_Basic_RAG/
├── data_source.txt      # Source knowledge document
├── rag.py               # Document loading, embedding, FAISS indexing, and retrieval
├── chatbot.py           # Ollama LLM integration and prompt formulation
├── main.py              # Interactive CLI loop for chatting
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Lockfile for reproducible environment
└── README.md            # Version 1 documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10 or higher (or [uv package manager](https://docs.astral.sh/uv/))
- **Ollama**: Installed and running locally
  - Pull the model:
    ```bash
    ollama run qwen2.5:3b
    ```

### 2. Installation

Using `uv`:
```bash
uv sync
```
*Or using standard pip:*
```bash
pip install -r pyproject.toml
```

### 3. Run the Chatbot

Activate the environment and start the interactive CLI:

```bash
uv run python main.py
```

---

## 💬 Usage Example

```text
Indexing Completed
You😉: What is RAG?
📑 Context: ['What is Retrieval-Augmented Generation...', ...]

Jarvis🤖: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation...

You😉: exit
Jarvis🤖: See Yaa
```

---

## ⚠️ Limitations of Version 1

- **Fixed Delimiter Splitting:** Relies on manual `\n##` delimiters instead of dynamic or recursive character/token chunking with overlap.
- **In-Memory FAISS Index:** Embeddings and index are computed on every application startup rather than persisted to disk.
- **Euclidean Distance (L2):** Uses raw L2 distance without embedding normalization or cosine similarity.
- **Single Document:** Supports only a single `.txt` file loaded synchronously.

---

## 🗺️ Next Steps (Roadmap)

- [ ] **Phase 1: Smarter Chunking** — Overlapping token/sentence chunking preserving headers.
- [ ] **Phase 2: Cleaner Retrieval** — Normalized embeddings, cosine similarity, and similarity score thresholds.
- [ ] **Phase 3: Citations & Metadata** — Source tracking and chunk references in responses.
- [ ] **Phase 4: Persistent Vector Index** — Saving/loading FAISS indexes to avoid re-embedding on startup.
- [ ] **Phase 5: Evaluation & UI** — Benchmark Q&A test set and optional FastAPI / web UI.
