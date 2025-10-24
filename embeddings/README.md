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
**Concept**: Text → Numerical Vectors

Minimal demonstration of embedding basics:
- Converts 3 sentences to vectors
- Displays first 10 dimensions of each
- Shows how text becomes numbers

```bash
python 01_basic_embeddings.py
```

### 02_comparison.py
**Concept**: Embedding Similarity

Similarity analysis between sentences:
- Creates embeddings for multiple sentences
- Calculates cosine similarity between all pairs
- Shows which sentences are semantically similar

```bash
python 02_comparison.py
```

### 03_semantic_search.py
**Concept**: Semantic Search Engine

Practical search implementation:
- 10 hardcoded documents (no database)
- 3 search queries with results
- Shows top 3 results with similarity scores

```bash
python 03_semantic_search.py
```

### 04_complete_demo.py
**Concept**: Full Demonstration

Comprehensive workflow combining all concepts:
- Complete embeddings to search pipeline
- Formatted output with presentation flow
- 15 diverse documents with detailed examples

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

## Design Principles

- **In-Memory Documents**: All text hardcoded (no files/databases)
- **Minimal Dependencies**: Only sentence-transformers and scikit-learn
- **Progressive Complexity**: Build from basics to practical applications
- **Clarity First**: Clean, readable code focused on concepts

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
