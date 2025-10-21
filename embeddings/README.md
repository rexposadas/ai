# Embedding Demo Scripts

Scripts for creating a YouTube video about embeddings and semantic search.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Scripts Overview

### 01_basic_embeddings.py
**Duration: ~3-4 minutes**
- Shows how text transforms into numerical vectors
- Demonstrates similarity calculation
- Visualizes the concept of "similar meanings = similar numbers"
- Good for introducing the core concept

**Usage:**
```bash
python 01_basic_embeddings.py
```

### 02_semantic_search.py
**Duration: ~8-10 minutes**
- Builds a complete semantic search engine
- Uses a collection of 10 hardcoded documents
- Demonstrates 3 different search queries
- Shows how semantic search understands meaning vs keywords

**Usage:**
```bash
python 02_semantic_search.py
```

### 03_complete_demo.py
**Duration: ~15 minutes**
- Comprehensive demonstration combining both concepts
- Part 1: Understanding embeddings (3-4 min)
- Part 2: Building semantic search (8-10 min)
- Part 3: Why it matters (2 min)
- Includes pacing with time.sleep() for better presentation flow

**Usage:**
```bash
python 03_complete_demo.py
```

## Video Recording Tips

1. **Terminal Setup**:
   - Use a large font (18-20pt) for visibility
   - Dark theme with good contrast
   - Clear any clutter from terminal before starting

2. **Execution Flow**:
   - Run `03_complete_demo.py` for the full video
   - Or run individual scripts for specific segments
   - The scripts include pauses for better pacing

3. **Key Points to Emphasize**:
   - Text → Numbers transformation
   - Similarity scores (1.0 = identical, 0.0 = unrelated)
   - Semantic understanding vs keyword matching
   - Real-world applications

4. **What Makes This Simple**:
   - All documents hardcoded in memory
   - No database setup required
   - No file I/O operations
   - Just run and demonstrate!

## Model Information

The scripts use `all-MiniLM-L6-v2`:
- Fast and lightweight
- 384-dimensional embeddings
- Good for demonstrations
- First run will download the model (~90MB)

## Script Features

All scripts include:
- ✓ Clear section headers
- ✓ Step-by-step explanations
- ✓ Hardcoded documents (no external files)
- ✓ Visual indicators (scores, bars, emojis)
- ✓ Progressive difficulty (basic → advanced)
- ✓ Comments explaining key concepts

## Customization

To modify the demonstrations:
- **Change documents**: Edit the `documents` list in each script
- **Adjust queries**: Modify the `queries` or `test_queries` lists
- **Change model**: Replace `'all-MiniLM-L6-v2'` with another sentence-transformers model
- **Add more results**: Change `top_3_indices` to show more results
