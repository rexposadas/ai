#!/usr/bin/env python3
"""
Part 2: Semantic Search Engine
Demonstrates how to search documents by meaning, not just keywords
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Document collection - hardcoded in memory for simplicity
# In a real system, these would come from a database or files
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Dogs are loyal pets",
    "JavaScript runs in web browsers",
    "AI and deep learning are related",
    "Cats are independent animals",
    "Java is used for enterprise applications",
    "Python is popular for data science",
    "Neural networks mimic the brain",
    "Pets need food and water",
]

print("=" * 60)
print("PART 2: Semantic Search Engine")
print("=" * 60)
print()

# Step 1: Show the document collection
print("Document Collection:")
print("-" * 60)
for i, doc in enumerate(documents, 1):
    print(f"{i:2d}. {doc}")
print()

# Step 2: Convert all documents to embeddings
print("Step 1: Converting all documents to embeddings...")
document_embeddings = model.encode(documents)
print(f"✓ Created {len(document_embeddings)} document embeddings")
print(f"  Each document is now a vector of {len(document_embeddings[0])} numbers")
print()

# Step 3: Define search queries
queries = [
    "What is machine learning?",
    "Tell me about programming languages",
    "Information about animals",
]

# Process each query
for query in queries:
    print("=" * 60)
    print(f"SEARCH QUERY: '{query}'")
    print("=" * 60)
    print()

    # Step 4: Convert query to embedding
    print("Step 2: Converting query to embedding...")
    query_embedding = model.encode([query])
    print("✓ Query converted to vector")
    print()

    # Step 5: Calculate similarity scores
    print("Step 3: Comparing query to all documents...")
    scores = cosine_similarity([query_embedding[0]], document_embeddings)[0]
    print("✓ Calculated similarity scores")
    print()

    # Step 6: Get top 3 results
    print("Step 4: Top 3 Most Relevant Results:")
    print("-" * 60)

    # Sort indices by score (highest to lowest)
    top_3_indices = np.argsort(scores)[-3:][::-1]

    for rank, idx in enumerate(top_3_indices, 1):
        print(f"{rank}. {documents[idx]}")
        print(f"   Similarity Score: {scores[idx]:.3f}")

        # Add explanation based on score
        if scores[idx] > 0.7:
            print("   ↳ Very relevant!")
        elif scores[idx] > 0.5:
            print("   ↳ Somewhat relevant")
        else:
            print("   ↳ Less relevant")
        print()

print("=" * 60)
print("KEY INSIGHT:")
print("Notice how the search understands MEANING, not just exact words!")
print()
print("For 'machine learning' → finds 'algorithms', 'AI', 'neural networks'")
print("For 'programming languages' → finds 'Python', 'JavaScript', 'Java'")
print("For 'animals' → finds 'dogs', 'cats', 'pets'")
print()
print("This is SEMANTIC SEARCH - understanding what you mean!")
print("=" * 60)
