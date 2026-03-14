# Let's Build a Vector Database from Scratch

> *"The best way to understand something is to build it from nothing."*

This repository is a hands-on, progressive walkthrough of building a Vector Database from scratch using Python, SQLite, FAISS, and Ollama. 

Instead of jumping straight to complex production systems like ChromaDB, Pinecone, or Weaviate, this project builds one step-by-step to help you understand the core concepts.

## Project Structure

- `walkthrough.ipynb`: The main notebook. It walks you through:
  1. **What is an embedding?** Calling Ollama's `nomic-embed-text` to generate vectors.
  2. **Cosine Similarity**: Comparing vectors manually using NumPy.
  3. **Building the Database (SQLite)**: Storing text, metadata, and binary floating-point vectors.
  4. **Basic Search**: Performing O(N) exhaustive search directly from SQLite.
  5. **Persistence**: Demonstrating how SQLite inherently gives us a persistent database.
  6. **Performance Scaling**: Showing where O(N) search bottlenecks.
  7. **Adding FAISS**: Integrating pure C++ vector indices to speed up searches.
- `test.py`: A simple script to test text embedding with Ollama and calculate the cosine similarity between generated mathematical vectors.
- `pyproject.toml` / `uv.lock`: Dependency definitions for the project.

## Setup

```bash
# Install dependencies using uv
uv sync

# Start your local Ollama instance (requires the nomic-embed-text model)
ollama pull nomic-embed-text
ollama serve

# Open the Jupyter Notebook
uv run jupyter notebook walkthrough.ipynb
```

## Core Concepts Explored

- **Embeddings**: Converting sentences to 768-D float arrays representing semantic meaning.
- **Cosine Similarity**: The core mathematical operation `dot(A, B) / (norm(A) * norm(B))` to measure distance between vectors.
- **Index vs Database Architecture**: Why a vector index (like FAISS) is different from a Vector Database (like Chroma). The database wraps the index with ID management, metadata filtering, and document storage.
