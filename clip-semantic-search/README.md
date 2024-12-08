# CLIP Semantic Image Search
## Natural language image search with OpenAI CLIP + FAISS

Search your photo collection using natural language queries. Powered by OpenAI's CLIP model for zero-shot vision-language understanding and FAISS for efficient similarity search over high-dimensional embeddings.

## Features

- Index a folder of images into a FAISS vector database
- Search by natural language: "a dog playing in the snow", "sunset over mountains"
- Combine text + image queries for multimodal search
- Streamlit web dashboard with gallery view
- Incremental index updates (add new images without reindexing)
- Export/import embeddings for faster reload
- Embedding caching to avoid recomputation

## Architecture

```
                    ┌─────────┐
Image folder ─────►│  CLIP   │────► Embedding vectors ──► FAISS Index
                   │ Encoder │
Text query ───────►│  CLIP   │────► Query vector ──────► FAISS Search ──► Results
                   └─────────┘
```

## Quick Start

```bash
pip install -r requirements.txt
python src/indexer.py --image-dir ./photos      # index your images
streamlit run src/app.py                         # launch search UI
```

## Project Structure

```
clip-semantic-search/
├── src/
│   ├── app.py              # Streamlit search UI
│   ├── indexer.py          # CLIP embedding + FAISS indexing
│   ├── search.py           # Search engine abstraction
│   └── model.py            # CLIP model wrapper
├── tests/
│   └── test_search.py
├── notebooks/
│   └── benchmark.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- FAISS: https://github.com/facebookresearch/faiss
