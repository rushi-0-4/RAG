# Version 3: Cosine Similarity & Score Display

Welcome to **Version 3** of the Learn_RAG series!

---

## 🎯 Objective

In **Version 1** and **Version 2**, we used FAISS's default Euclidean distance (`IndexFlatL2`).

While Euclidean distance measures the geometric distance between points, **Cosine Similarity** measures the angle between vectors — capturing the _direction of meaning_ rather than magnitude.

**Version 3 focus:**

1. **Vector Normalization**: Normalize both chunk embeddings and query embeddings to unit length (L2 norm = 1.0).
2. **Inner Product Index (`IndexFlatIP`)**: When vectors are L2-normalized, the inner product (dot product) equals exact **Cosine Similarity**.
3. **Score Visibility**: Return and display cosine similarity scores alongside retrieved chunks so you can compare their relative match strength. Cosine similarity ranges from `0` to `1`; it is a similarity measure, not a probability or guaranteed confidence score.

---

## 📁 Directory Structure

```
Version_3_Cosine_Similarity/
├── rag.py               # Preprocessing, sliding chunking, normalized embeddings & FAISS IndexFlatIP
├── chatbot.py           # Ollama LLM integration
├── main.py              # Interactive CLI loop with score display
└── README.md            # Version 3 documentation
```

---

## 🚀 How to Run

From this directory:

```bash
uv run python main.py
```
