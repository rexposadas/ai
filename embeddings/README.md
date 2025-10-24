# Embeddings & Semantic Search Demo

Educational demonstrations of text embeddings and semantic search using Python.

## Quick Start

### Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or: venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Scripts Overview

### 01_basic_embeddings.py
**Key Concept**: Text → Numerical Vectors

A minimal script showing the core concept of embeddings:
- Converts 3 sentences to vectors
- Displays the first 10 dimensions of each vector
- Perfect for understanding the basics

**Usage:**
```bash
python 01_basic_embeddings.py
```

### 02_comparison.py
**Key Concept**: Embedding Similarity

Extends the basic concepts with similarity analysis:
- Creates embeddings for sentences
- Calculates cosine similarity between all pairs
- Shows which sentences are semantically similar
- Good for understanding how similarity works

**Usage:**
```bash
python 02_comparison.py
```

### 03_semantic_search.py
**Key Concept**: Semantic Search Engine

Implements a practical search engine:
- 10 hardcoded documents (no database needed)
- 3 example search queries demonstrating semantic understanding
- Shows top 3 results with similarity scores
- Highlights how semantic search finds related concepts, not just keywords

**Usage:**
```bash
python 03_semantic_search.py
```

### 04_complete_demo.py
**Key Concept**: Full Educational Workflow

Comprehensive demonstration combining all concepts:
- Part 1: Understanding embeddings
- Part 2: Measuring similarity
- Part 3: Building a search engine
- Includes formatted output and pacing for presentations

**Usage:**
```bash
python 04_complete_demo.py
```

## How It Works

### The Embedding Process
1. **Text → Numbers**: Sentences are converted to vectors (arrays of numbers)
2. **Meaning → Dimensions**: Similar meanings produce similar vectors
3. **Similarity Scoring**: Compare vectors using cosine similarity (0.0 to 1.0)

### Why This Matters
- **Semantic Understanding**: Finds conceptually related content, not just keywords
- **Fast Search**: Vectorized comparison is computationally efficient
- **Scalable**: Can work with thousands of documents in production

## Model Information

Scripts use `sentence-transformers/all-MiniLM-L6-v2`:
- **Speed**: Fast inference (~100ms per document)
- **Size**: Lightweight (~90MB, downloads on first run)
- **Dimensions**: 384-dimensional vectors
- **Quality**: Good balance of speed and semantic understanding

## Key Design Principles

- **In-Memory Documents**: All text hardcoded for simplicity (no files/databases)
- **Minimal Dependencies**: Only sentence-transformers and scikit-learn
- **Progressive Complexity**: Scripts build from basics to practical applications
- **Educational Focus**: Optimized for clarity and understanding

## Customization

To adapt these scripts:

**Change Documents**:
```python
documents = [
    "Your text here",
    "More text",
]
```

**Change Model** (alternative models from Hugging Face):
```python
model = SentenceTransformer("model-name")
```

**Adjust Results**:
- Modify similarity thresholds (0.7 for "relevant", 0.5 for "somewhat")
- Change `top_3_indices` to show more/fewer results
- Add more documents to search across

## Troubleshooting

**Model Download Issues**:
- First run downloads ~90MB model
- Requires internet connection
- Model is cached locally after download

**Slow Performance**:
- First query is slower (model initialization)
- Subsequent queries run faster
- For production: batch encode documents

**Different Results**:
- Similarity scores vary based on random seed
- Transformer models have some inherent variation
- Results should be directionally consistent
