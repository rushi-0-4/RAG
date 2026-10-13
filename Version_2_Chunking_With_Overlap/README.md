# Version 2: Smarter Chunking with Sliding Overlap

An enhanced iteration of our Retrieval-Augmented Generation (RAG) pipeline, introducing **automated text preprocessing** and **sliding window chunking with overlap** to eliminate boundary cut-offs and deliver higher-quality retrieved context.

---

## 📌 What's New in Version 2

In **Version 1**, document chunking relied on an arbitrary delimiter (`\n##`). This led to unpredictable chunk lengths and broke semantic ideas right at section boundaries.

**Key Upgrades in Version 2:**
1. **Text Preprocessing & Cleaning (`clean_text`)**:
   - Strips null bytes, normalizes line breaks (`\r\n` $\rightarrow$ `\n`), and collapses excessive empty lines (`\n{3,}` $\rightarrow$ `\n\n`).
   - Cleans markdown noise (headers, bold/italic asterisks) so embeddings focus on clean content rather than formatting artifacts.
2. **Sliding Window Chunking (`split_text`)**:
   - Replaces manual delimiters with a configurable sliding window:
     - **`chunk_size = 500` characters** (consistent chunk density).
     - **`chunk_overlap = 100` characters** (shared bridge between consecutive chunks).
     - **`step = chunk_size - chunk_overlap = 400` characters**.
3. **Robustness & Code Quality**:
   - Added parameter validation (`chunk_size > 0`, `0 <= chunk_overlap < chunk_size`).
   - Type annotations (`faiss.IndexFlatL2 | None`, `list[str]`, etc.).
   - Boundary checks during retrieval (`if 0 <= i < len(chunks)`).
   - Lazy indexing fallback check in `search()`.

---

## 🔍 How Sliding Window Overlap Works

```text
Full Document Stream:
[--- Topic A --- Fact 1 --- Fact 2 --- Topic B --- Fact 3 ---]

Chunk 1 (0 to 500):
|=============================|
[Topic A ....... Fact 1 ...... Fact 2]

Chunk 2 (400 to 900):
                |=============================|
                [Fact 2 ....... Topic B ...... Fact 3]
                └─── 100 char shared bridge ──┘
```

> **The Advantage:** If a user asks a question connecting **Fact 2** and **Topic B**, the answer is preserved intact within Chunk 2 rather than severed across a rigid split point.

---

## 🏗️ Architecture & Pipeline Flow

```
[ ../data_source.txt ]
          │
          ▼
     load_data()
          │
          ▼
     clean_text() ──► Normalizes whitespace, strips markdown artifacts
          │
          ▼
     split_text() ──► Sliding window: size=500, overlap=100
          │
          ▼
SentenceTransformer ("all-MiniLM-L6-v2") ──► Chunk Embeddings
                                                    │
                                                    ▼
                                           FAISS Vector Index (L2)
                                                    │
User Query ──► SentenceTransformer Embed ──► search() (Top 5 Chunks)
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
- **Regex & Math:** `re`, `numpy`

---

## 📁 Directory Structure

```
Learn_RAG/
├── data_source.txt                  # Shared source knowledge document
├── pyproject.toml                   # Shared project dependencies
├── uv.lock                          # Shared lockfile
├── plan/
│   └── plan.md                      # Evolution roadmap
├── Version_1_Basic_RAG/             # Baseline (fixed delimiter)
└── Version_2_Chunking_With_Overlap/ # Version 2 (current)
    ├── rag.py                       # Preprocessing, sliding window chunking, FAISS index & search
    ├── chatbot.py                   # Ollama integration with Jarvis prompt
    ├── main.py                      # Interactive terminal chat loop
    └── README.md                    # Version 2 documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Ollama**: Installed and running locally
  ```bash
  ollama run qwen2.5:3b
  ```
- **uv Package Manager**: [Astral uv](https://docs.astral.sh/uv/)

### 2. Run the Interactive Chatbot

From the repository root or the `Version_2_Chunking_With_Overlap` folder:

```bash
uv run python Version_2_Chunking_With_Overlap/main.py
```

---

## 💬 Example CLI Interaction

```text
Loading weights: 100%|██████████| 103/103 [00:00<00:00]
Indexing Completed

You😉: Hello
📑 Context: ["The Ultimate Encyclopedia of Python...", ...]
Jarvis🤖: Hello! How can I assist you today?

You😉: what is Broadcasting?
📑 Context: ['multidimensional array.\n- Broadcasting: Powerful functions for performing operations on arrays of different shapes.\n- Linear Algebra: Integrated tools for matrix operations...']

Jarvis🤖: Broadcasting is a powerful feature in NumPy that allows for operations on arrays of different shapes. It automatically expands the smaller array along matching axes so mathematical operations can be performed element-wise without copying data.

You😉: bye
Jarvis🤖: See Yaa
```

---

## 📈 Comparison: Version 1 vs. Version 2

| Metric / Feature | Version 1 (Basic) | Version 2 (Smarter Chunking) |
| :--- | :--- | :--- |
| **Splitting Method** | Static `\n##` delimiter | Sliding window (`chunk_size=500`, `chunk_overlap=100`) |
| **Boundary Preservation** | None (hard cuts) | 100-character overlap bridge |
| **Text Preprocessing** | Raw file string | Markdown cleaning, whitespace & line-ending normalization |
| **Type Safety** | None | Full type hinting (`faiss.IndexFlatL2 | None`, `list[str]`) |
| **Error Handling** | Basic prints | `ValueError`, `RuntimeError`, bounds-checked index lookups |

---

## 🗺️ What's Next? (Version 3)

According to [plan.md](../plan/plan.md):
- **Version 3: Cosine Similarity & Score Display**:
  - Normalize embedding vectors before indexing.
  - Switch FAISS to inner product (`IndexFlatIP`) for true cosine similarity.
  - Print similarity confidence scores alongside retrieved chunks in debug output.
