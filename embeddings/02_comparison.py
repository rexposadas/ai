#!/usr/bin/env python3
"""
Part 1: Basic Embedding Demonstration
Shows how text is converted to numerical vectors (embeddings)
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a programming language",
    "Cats are independent animals",
    "Dogs are loyal pets",
]

# Convert sentences to embeddings (vectors of numbers)
embeddings = model.encode(documents)

# Display the embeddings
for i, (sentence, embedding) in enumerate(zip(documents, embeddings), 1):
    print(f"{i}. {sentence}")
    print(f"   Vector: [{', '.join(f'{x:.4f}' for x in embedding[:10])}...]")
    print(f"   (showing 10 of {len(embedding)} numbers)")
    print()

# Calculate similarity between embeddings
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        score = similarities[i][j]
        print(f"Sentence {i + 1} vs Sentence {j + 1}: {score:.3f}")
        print(f"  '{documents[i]}'")
        print(f"  '{documents[j]}'")
        print()
