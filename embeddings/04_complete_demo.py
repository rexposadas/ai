#!/usr/bin/env python3
"""
Complete Demo: Understanding Embeddings and Semantic Search
A comprehensive demonstration combining both concepts for a 15-minute video

This script demonstrates:
1. What embeddings are (text → numbers)
2. How similarity works (comparing vectors)
3. Building a semantic search engine
4. Why this matters (understanding meaning vs keywords)
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

def print_section_header(title):
    """Helper function to print consistent section headers"""
    print("\n")
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()

def pause(seconds=1):
    """Helper function for demonstration pacing"""
    time.sleep(seconds)

# Initialize the model once
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model ready!")

# ============================================================================
# PART 1: Understanding Embeddings (3-4 minutes)
# ============================================================================

print_section_header("PART 1: What Are Embeddings?")

# Step 1: Show sample sentences
print("Let's start with some simple sentences:")
print("-" * 70)
sentences = [
    "The cat sits on the mat",
    "A dog runs in the park",
    "Python is a programming language",
    "I love my pet cat"
]

for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")
print()

pause(1)

# Step 2: Convert to embeddings
print("Now, let's convert these sentences into numbers (embeddings)...")
pause(1)
embeddings = model.encode(sentences)
print(f"✓ Done! Each sentence is now a vector of {len(embeddings[0])} numbers")
print()

# Step 3: Show a sample embedding
print("Here's what sentence 1 looks like as numbers (first 15):")
print("-" * 70)
print(f"'{sentences[0]}'")
print(f"→ [{', '.join(f'{x:.4f}' for x in embeddings[0][:15])}...]")
print()

pause(1)

# Step 4: Calculate and show similarity
print("Now let's see which sentences are SIMILAR to each other:")
print("-" * 70)

similarities = cosine_similarity(embeddings)

# Compare specific pairs
pairs_to_compare = [
    (0, 3, "Both about cats"),
    (0, 1, "Different animals"),
    (2, 0, "Completely different topics")
]

for idx1, idx2, explanation in pairs_to_compare:
    score = similarities[idx1][idx2]
    print(f"\nComparing:")
    print(f"  • '{sentences[idx1]}'")
    print(f"  • '{sentences[idx2]}'")
    print(f"  Similarity: {score:.3f} ({explanation})")

    # Visual bar representation
    bar_length = int(score * 40)
    print(f"  [{'█' * bar_length}{'-' * (40 - bar_length)}]")

print()
print("💡 Key Insight: Similar meanings → Similar numbers!")
print("   That's how computers understand the 'meaning' of text.")

pause(2)

# ============================================================================
# PART 2: Building a Semantic Search Engine (8-10 minutes)
# ============================================================================

print_section_header("PART 2: Semantic Search Engine")

# Document collection
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn from data",
    "Dogs are loyal and friendly pets",
    "JavaScript runs in web browsers",
    "Artificial intelligence and deep learning are related fields",
    "Cats are independent animals that make great companions",
    "Java is widely used for enterprise applications",
    "Python is extremely popular for data science and AI",
    "Neural networks are inspired by the human brain",
    "Pets like dogs and cats need food, water, and love",
    "Web development uses HTML, CSS, and JavaScript",
    "Machine learning models need training data",
    "Golden retrievers are friendly dogs",
    "React is a JavaScript library for building user interfaces",
    "Deep learning is a subset of machine learning"
]

print("Our Document Collection (15 documents):")
print("-" * 70)
for i, doc in enumerate(documents, 1):
    print(f"{i:2d}. {doc}")
print()

pause(1)

# Convert all documents to embeddings
print("Step 1: Converting all documents to embeddings...")
pause(1)
document_embeddings = model.encode(documents)
print(f"✓ Created {len(document_embeddings)} embeddings")
print(f"  Each document → {len(document_embeddings[0])} numbers")
print()

# Define test queries that demonstrate semantic understanding
test_queries = [
    ("What is machine learning?",
     "Notice it finds 'algorithms', 'AI', 'neural networks' - not just exact words!"),

    ("programming languages for beginners",
     "Finds Python, JavaScript, Java - understands the concept!"),

    ("information about dogs",
     "Retrieves dog-related content, including breeds and pet care!"),
]

# Process each query
for query_num, (query, insight) in enumerate(test_queries, 1):
    print_section_header(f"Search Example {query_num}: '{query}'")

    # Step 2: Convert query to embedding
    print("Step 2: Converting query to embedding...")
    pause(0.5)
    query_embedding = model.encode([query])
    print("✓ Query converted to vector")
    print()

    # Step 3: Calculate similarity
    print("Step 3: Comparing query to all documents...")
    pause(0.5)
    scores = cosine_similarity([query_embedding[0]], document_embeddings)[0]
    print("✓ Similarity calculated for all documents")
    print()

    # Step 4: Display top 3 results
    print("Step 4: Top 3 Most Relevant Documents:")
    print("-" * 70)

    top_3_indices = np.argsort(scores)[-3:][::-1]

    for rank, idx in enumerate(top_3_indices, 1):
        score = scores[idx]
        print(f"\n{rank}. [{score:.3f}] {documents[idx]}")

        # Visual relevance indicator
        if score > 0.7:
            print("   ⭐⭐⭐ Highly relevant!")
        elif score > 0.5:
            print("   ⭐⭐ Moderately relevant")
        else:
            print("   ⭐ Somewhat relevant")

    print()
    print(f"💡 {insight}")
    pause(2)

# ============================================================================
# PART 3: Why This Matters (2 minutes)
# ============================================================================

print_section_header("PART 3: Why Embeddings Matter")

print("The Traditional Way (Keyword Search):")
print("-" * 70)
print("❌ Search: 'ML'")
print("   Only finds documents with exact letters 'ML'")
print("   Misses: 'machine learning', 'artificial intelligence', 'neural networks'")
print()

print("The Semantic Search Way (Using Embeddings):")
print("-" * 70)
print("✓ Search: 'ML'")
print("   UNDERSTANDS you mean 'machine learning'")
print("   Finds: 'algorithms', 'AI', 'deep learning', 'neural networks'")
print()

pause(1)

print("Real-World Applications:")
print("-" * 70)
applications = [
    ("Search Engines", "Google, Bing - understand what you're looking for"),
    ("Recommendation Systems", "Netflix, Spotify - suggest similar content"),
    ("Chatbots", "Customer service bots that understand questions"),
    ("Document Search", "Find similar documents even with different wording"),
    ("Question Answering", "AI assistants that understand context"),
]

for app, description in applications:
    print(f"• {app}")
    print(f"  → {description}")
    print()

print("=" * 70)
print("🎯 THE KEY CONCEPT:")
print("=" * 70)
print("""
Embeddings transform text into numbers that capture MEANING.
Similar meanings → Similar numbers → Easy to compare!

This is the foundation for:
  • Semantic search
  • Recommendation systems
  • AI chatbots
  • Vector databases
  • And much more!

Without embeddings, computers can only match exact words.
With embeddings, computers can understand what you MEAN! 🚀
""")
print("=" * 70)
