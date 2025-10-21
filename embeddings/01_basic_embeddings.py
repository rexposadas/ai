#!/usr/bin/env python3
"""
Part 1: Basic Embedding Demonstration
Shows how text is converted to numerical vectors (embeddings)
"""

from sentence_transformers import SentenceTransformer

# Initialize the embedding model
# This is a small, fast model good for demonstrations
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences to demonstrate embeddings
# Each sentence will be converted to a vector of 384 numbers
sentences = [
    "The cat sits on the mat",
    "A dog runs in the park",
    "Python is a programming language"
]

print("=" * 60)
print("PART 1: Understanding Embeddings")
print("=" * 60)
print()

# Show the original sentences
print("Original Sentences:")
print("-" * 60)
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")
print()

# Convert sentences to embeddings (vectors of numbers)
print("Converting to embeddings...")
embeddings = model.encode(sentences)
print(f"✓ Done! Each sentence is now represented by {len(embeddings[0])} numbers")
print()

# Display the embeddings
print("Embeddings (first 10 numbers of each):")
print("-" * 60)
for i, (sentence, embedding) in enumerate(zip(sentences, embeddings), 1):
    print(f"{i}. {sentence}")
    print(f"   Vector: [{', '.join(f'{x:.4f}' for x in embedding[:10])}...]")
    print(f"   (showing 10 of {len(embedding)} numbers)")
    print()

# Calculate similarity between embeddings
from sklearn.metrics.pairwise import cosine_similarity

print("Similarity Scores:")
print("-" * 60)
print("These numbers show how similar the sentences are (1.0 = identical, 0.0 = unrelated)")
print()

similarities = cosine_similarity(embeddings)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarities[i][j]
        print(f"Sentence {i+1} vs Sentence {j+1}: {score:.3f}")
        print(f"  '{sentences[i]}'")
        print(f"  '{sentences[j]}'")
        print()

print("=" * 60)
print("KEY TAKEAWAY:")
print("Similar meanings → Similar numbers")
print("This is how computers understand the 'meaning' of text!")
print("=" * 60)
