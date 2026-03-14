# Let's Build a Vector Database from Scratch

> *"The best way to understand something is to build it from nothing."*

## The Core Idea: Meaning is Geometry

**Words that mean similar things end up at nearby coordinates in a high-dimensional space.**

When you run *"The cat sat on the mat"* through an embedding model, you get 768 floats — a coordinate in 768-dimensional space. *"A kitten rested on the rug"* maps to a nearby point. *"Quarterly revenue exceeded projections"* is far away.

This isn't metaphorical. It's literal geometry. Similarity search is "find the nearest points." A vector database is the system that stores these coordinates and answers that question.

## Architecture: Index vs Database

Here's the key insight that separates this project from a toy:

**FAISS is not a database. It's a search index.** An index only knows about float arrays. A database wraps an index and adds IDs, text storage, metadata filtering, and CRUD operations.

```
┌────────────────────────────────────────────┐
│             SimpleVectorDB                 │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Document Store (dict)               │  │
│  │  id → {text, metadata, position}     │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Vector Index (swappable)            │  │
│  │  VanillaIndex  OR  FAISSIndex        │  │
│  │  (only knows about float arrays)     │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  ID Mapping                          │  │
│  │  integer position ↔ string ID        │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

This is the same pattern ChromaDB, Pinecone, and Weaviate use — a document store + a pluggable vector index. We just build it in ~200 lines so you can see every seam.

## Project Structure

```
vectordb/
├── indexes.py      # Swappable backends: VanillaIndex (NumPy) & FAISSIndex (C++)
├── db.py           # SimpleVectorDB — our mini ChromaDB with IDs, metadata, CRUD
├── embed.py        # Text → 768-D float arrays via Ollama
└── chroma_store.py # Real ChromaDB wrapper for comparison

demo.py             # 100 real docs via Ollama → all three databases → search + filter
benchmark.py        # 10K random vectors → timed across all three
pyproject.toml      # uv-installable package
```

## Setup

```bash
# Install dependencies
uv sync

# Run the benchmark (no Ollama needed — uses random vectors)
uv run python benchmark.py

# For the real demo with text embeddings:
ollama pull nomic-embed-text
ollama serve
uv run python demo.py
```

## Quick Start

```python
from vectordb import SimpleVectorDB, get_embedding

# Pick your backend: "vanilla" (NumPy) or "faiss" (C++)
db = SimpleVectorDB(backend="faiss")

# Add documents with metadata
docs = [
    ("The cat sat on the mat",       {"topic": "animals"}),
    ("Python is great for ML",       {"topic": "tech"}),
    ("Risotto needs constant stirring", {"topic": "cooking"}),
]

for text, meta in docs:
    vec = get_embedding(text)
    db.add(text=text, vector=vec, metadata=meta)

# Semantic search
results = db.search(get_embedding("a kitten on the floor"), top_k=3)
for r in results:
    print(f"  {r.score:.4f}  [{r.metadata['topic']}]  {r.text}")

# Metadata filtering — only search within a topic
results = db.search(
    get_embedding("how to prepare food"),
    top_k=3,
    where={"topic": "cooking"},
)
```

## The Demo (demo.py)

Embeds 100 real sentences across 10 topics through Ollama, loads them into `SimpleVectorDB(vanilla)`, `SimpleVectorDB(faiss)`, and ChromaDB, then runs queries that share zero words with the best matches. Also demonstrates metadata filtering — a feature that lives in the database layer, not the index.

## The Benchmark (benchmark.py)

10,000 random 768-D vectors, 100 queries, three databases side by side:

```
  Metric                  Ours (vanilla)     Ours (faiss)         ChromaDB
  ---------------------- ----------------- ----------------- -----------------
  Avg latency (ms)               X.XXX           X.XXX           X.XXX
  Queries/min                   XX,XXX         XXX,XXX         XXX,XXX
  Speedup vs vanilla              1.0x            X.Xx            X.Xx
```

## File-by-File Guide

| File | What It Teaches |
|------|----------------|
| `indexes.py` | The abstract BaseIndex interface, VanillaIndex with explicit cosine math, FAISSIndex with L2→cosine conversion |
| `db.py` | How a database wraps an index: Document storage, ID mapping, metadata filtering, batch insert, soft delete |
| `embed.py` | How text becomes coordinates (Ollama HTTP call, NumPy conversion, why float32) |
| `chroma_store.py` | The production alternative: ChromaDB with HNSW, batch chunking for large inserts |
| `demo.py` | Real text + real embeddings + metadata filtering across all three databases |
| `benchmark.py` | The empirical proof: 10K vectors, three databases, same math, different speed |

## Key Concepts

**Index vs Database** — An index (VanillaIndex, FAISSIndex) only searches float arrays. A database (SimpleVectorDB) adds document storage, IDs, metadata, and CRUD on top. FAISS is an index. ChromaDB is a database. We build both layers.

**Cosine Similarity** — `dot(A, B) / (norm(A) * norm(B))`. Measures the angle between two vectors. 1 = identical, 0 = unrelated, -1 = opposite.

**L2 Distance** — Euclidean distance. For unit vectors: `L2² = 2 - 2*cos(θ)`. FAISS uses L2; we normalize to make L2 equivalent to cosine.

**HNSW** — Hierarchical Navigable Small World graph (inside ChromaDB). Approximate nearest neighbors in O(log N) instead of O(N).

**Metadata Filtering** — The database fetches extra candidates from the index, then filters by key-value metadata. This is post-filtering — the same strategy ChromaDB uses internally.
