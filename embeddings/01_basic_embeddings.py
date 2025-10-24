#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a programming language",
    "Cats are independent animals",
    "Dogs are loyal pets",
]

# Convert sentences to embeddings (vectors of numbers)
embeddings = model.encode(documents)

for i, (doc, embedding) in enumerate(zip(documents, embeddings), 1):
    print(f"{i}. {doc}")
    print(f"   Vector: [{', '.join(f'{x:.4f}' for x in embedding[:10])}...]")
    print(f"   (showing 10 of {len(embedding)} numbers)")
    print()
