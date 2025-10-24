# AI & ML Projects

Educational demonstrations and experiments in artificial intelligence and machine learning.

## Projects

### embeddings/
Demonstrations of text embeddings and semantic search using sentence transformers.

**Quick Start**:
```bash
cd embeddings
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 01_basic_embeddings.py
```

**Scripts**:
- `01_basic_embeddings.py` - Convert text to numerical vectors
- `02_comparison.py` - Calculate similarity between embeddings
- `03_semantic_search.py` - Build a semantic search engine
- `04_complete_demo.py` - Full workflow demonstration

See [embeddings/README.md](embeddings/README.md) for detailed information.

## Technology Stack

- **Python 3.8+** - Primary language
- **sentence-transformers** - Text embedding models
- **scikit-learn** - Machine learning utilities
- **numpy** - Numerical operations
- **torch** - Deep learning framework

## Getting Started

Each project is self-contained with its own virtual environment:

```bash
cd [project-name]
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Structure

- `embeddings/` - Text embedding demonstrations and semantic search
- `requirements.txt` - Project dependencies
