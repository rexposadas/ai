# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains demonstration scripts for embedding and semantic search concepts, designed for educational video content. The project uses Python with sentence-transformers and scikit-learn to showcase how text can be transformed into numerical vectors and used for semantic search.

**Project Goal**: Create clear, progressive demonstrations of embeddings and semantic search for a YouTube video series.

## Quick Start

### Setup
1. Ensure Python 3.8+ is available
2. Create and activate a virtual environment:
   ```bash
   cd embeddings
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Scripts

The project includes three progressively complex demonstration scripts:

```bash
# Basic embedding concepts only (~3-4 minutes)
python 01_basic_embeddings.py

# Semantic search engine only (~8-10 minutes)
python 02_semantic_search.py

# Complete comprehensive demo (~15 minutes)
python 03_complete_demo.py
```

The first run will download the embedding model (~90MB) and cache it locally.

## Architecture

### Script Organization

**Progressive Complexity Pattern**: Each script builds on concepts from the previous one, designed for educational video pacing.

- **01_basic_embeddings.py**: Foundational concepts
  - Text → vector transformation
  - Similarity calculation between embeddings
  - Visualization of "similar meanings = similar numbers"

- **02_semantic_search.py**: Practical application
  - Building a search engine from scratch
  - 10 hardcoded documents for in-memory operation
  - Three diverse search queries demonstrating semantic understanding

- **03_complete_demo.py**: Full narrative
  - Combines both concepts with polished presentation
  - 15 documents with more diverse content
  - Three query examples with insights
  - Includes timing/pacing with `time.sleep()` for presentation flow
  - Helper functions for consistent formatting

### Key Design Decisions

**In-Memory Operation**: All documents and queries are hardcoded to eliminate:
- Database setup complexity
- File I/O operations
- External dependencies beyond ML libraries
- Recording interruptions

This approach prioritizes demonstration clarity over scalability.

**Model Choice**: Uses `all-MiniLM-L6-v2`:
- Fast inference suitable for live demos
- 384-dimensional embeddings (good balance of speed/accuracy)
- Lightweight (~90MB download)
- Well-documented and community-tested

**Similarity Metric**: Cosine similarity via scikit-learn
- Standard NLP approach (1.0 = identical, 0.0 = unrelated)
- Intuitive for explaining relevance scores
- Computationally efficient for demonstration scale

## Dependencies

**Core Libraries**:
- `sentence-transformers`: Embedding model and encoding
- `scikit-learn`: Cosine similarity calculations
- `torch`: Underlying computation framework
- `numpy`: Numerical operations

All specified in `embeddings/requirements.txt` with minimum versions for API compatibility.

## Customization

### Modifying Content

**Change Documents**:
- Edit the `documents` list in `02_semantic_search.py` or `03_complete_demo.py`
- Semantic search will automatically adapt to new content

**Change Queries**:
- Modify `queries` list in `02_semantic_search.py`
- Modify `test_queries` tuple in `03_complete_demo.py`
- Each query includes explanation text for video narration

**Change Model**:
- Replace `'all-MiniLM-L6-v2'` with any sentence-transformers model
- See [Hugging Face model cards](https://huggingface.co/sentence-transformers) for alternatives
- Larger models improve semantic understanding but increase runtime

**Adjust Results Display**:
- Change `top_3_indices` to `top_5_indices` etc. to show more results
- Modify similarity thresholds (currently 0.7 for "very relevant", 0.5 for "somewhat relevant")

### Customization Code Patterns

All scripts use helper functions for easy modification:

```python
# In 03_complete_demo.py:
print_section_header(title)    # Consistent formatting
pause(seconds=1)               # Pacing control
```

These can be extended for additional features without modifying core logic.

## Development Notes

### Video Recording Considerations

**Terminal Setup**:
- Use large font (18-20pt) for viewer readability
- Dark theme with good contrast
- Clear terminal before starting for professional appearance

**Execution Strategy**:
- Use `03_complete_demo.py` for full 15-minute video
- Use individual scripts for segment videos
- Scripts include natural pauses for narration timing

**Demonstrating Semantic Understanding**:
- The search examples show that semantic search finds conceptually related documents
- Not just keyword matching (e.g., "machine learning" finds "algorithms", "AI", "neural networks")
- This is the key differentiator from traditional keyword search

### Key Demonstrations

**Embedding Concept**:
- Text is converted to 384 numbers
- Similar texts produce similar vectors
- Similarity is measured by comparing vector dimensions

**Search Intelligence**:
- Query is converted to same embedding space
- Compared against all document embeddings
- Ranked by cosine similarity score
- User sees why semantic search understands meaning

## Testing and Validation

**Verification**:
- Run each script to ensure model downloads successfully
- Verify output formatting is terminal-friendly
- Check that similarity scores make intuitive sense
- Confirm pacing allows for narration (3-4s per major section)

**Quality Checks**:
- All sentences should render without truncation
- Similarity scores should range 0.0-1.0
- Related concepts should have high scores (>0.7)
- Unrelated topics should have lower scores
