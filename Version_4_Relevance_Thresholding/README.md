# Version 4: Relevance Thresholding

Version 4 builds on Version 3 by filtering retrieved chunks using their cosine similarity scores. Only chunks at or above the configured threshold are returned for answering.

## What changed

- `search()` in `rag.py` retrieves the top results and keeps only those with a score greater than or equal to `threshold` (default: `0.5`).
- `respond()` in `chatbot.py` checks whether any chunks passed the filter. If none did, it returns a document-not-found response without calling Ollama.
- The CLI continues to display the retrieved chunks and their similarity scores.

Cosine similarity is a ranking signal, not a probability or guaranteed confidence score. Treat `0.5` as a starting value and tune it against relevant and irrelevant queries for your documents.

## Project files

```
Version_4_Relevance_Thresholding/
├── rag.py       # Text processing, cosine-similarity search, and score thresholding
├── chatbot.py   # Builds context or returns early when no chunks pass the threshold
├── main.py      # Interactive CLI with retrieved chunk score display
└── README.md
```

## Run

From this directory:

```bash
uv run python main.py
```
