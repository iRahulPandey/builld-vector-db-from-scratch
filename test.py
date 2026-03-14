# use ollama for text eembedding
import requests
import numpy as np

def embed_text(text)->np.ndarray:
    response = requests.post("http://localhost:11434/api/embed", json={"model": "nomic-embed-text", "input": text})
    return np.array(response.json()["embeddings"][0], dtype=np.float32)

sentences = [
    "The cat sat on the mat",          # baseline
    "A kitten rested on the rug",      # same meaning, ZERO shared words
    "The dog played in the park",      # related (animal) but different
    "The stock market crashed today",  # totally unrelated
]

embeddings = [embed_text(sentence) for sentence in sentences]

#print(embeddings)

# calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# calculate cosine similarity between all pairs of vectors
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        print(f"Similarity between {sentences[i]} and {sentences[j]}: {cosine_similarity(embeddings[i], embeddings[j])}")
