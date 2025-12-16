# My Atlas - RAG Chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot with advanced document processing, hybrid search, and intelligent answer validation.

## Features

- **Multi-format Document Ingestion**: PDF, DOCX, and scanned documents (with OCR)
- **Hybrid Retrieval**: Combines semantic search (Milvus) and keyword search (BM25) with Reciprocal Rank Fusion
- **Advanced Reranking**: Jina Reranker v2 for optimal result ordering
- **Smart Answer Validation**: Validates answers and falls back to general knowledge when needed
- **CLI Interface**: Easy-to-use command-line interface
- **Cost-Optimized**: Uses Voyage AI embeddings and Jina Reranker (open-source) for maximum cost efficiency

## Architecture

See [docu/001-architecture-and-benchmarks.md](docu/001-architecture-and-benchmarks.md) for complete architecture details and benchmarks.

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for Milvus)
- CUDA 11.8 (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/my_atlas
   ```

2. **Install PaddlePaddle (for OCR)**

   For CUDA 11.8 (GPU):
   ```bash
   pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   ```

   For CPU:
   ```bash
   pip install paddlepaddle==3.2.0
   ```

3. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

5. **Set up Milvus (Vector Database)**
   ```bash
   cd docker
   docker-compose up -d
   ```

6. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # - VOYAGE_API_KEY (get from https://voyageai.com)
   # - OPENAI_API_KEY (get from https://platform.openai.com)
   ```

7. **Download NLTK data (one-time)**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

## Usage

### Ingest Documents

```bash
# Ingest all documents from a directory
my-atlas ingest ./data/documents

# Ingest a single PDF
my-atlas ingest ./data/mydoc.pdf

# Ingest scanned PDFs with OCR
my-atlas ingest ./data/scanned --use-ocr

# Ingest only DOCX files
my-atlas ingest ./data/documents --doc-type docx
```

### Query the Knowledge Base

```bash
# Simple query
my-atlas query "What is RAG?"

# Verbose mode (shows retrieval details)
my-atlas query "How does hybrid search work?" --verbose

# Specify number of chunks to retrieve
my-atlas query "Explain embeddings" --top-k 10

# Disable fallback to general knowledge
my-atlas query "Company policy on X?" --no-fallback
```

### Check System Status

```bash
# Show general status
my-atlas status

# Show detailed database info
my-atlas status --db

# Show all configuration
my-atlas config --show
```

### Clear Knowledge Base

```bash
# Clear all data (with confirmation)
my-atlas clear

# Clear without confirmation
my-atlas clear --confirm
```

## Project Structure

```
my_atlas/
├── docu/                      # Documentation
│   └── 001-architecture-and-benchmarks.md
├── src/                       # Source code
│   ├── cli/                   # CLI commands
│   ├── database/              # Milvus client
│   ├── ingestion/             # Document processing pipeline
│   ├── inference/             # LLM and validation
│   ├── retrieval/             # Hybrid search and reranking
│   └── utils/                 # Configuration and logging
├── data/                      # Input documents
├── ext/                       # Processed data
│   ├── raw_text/              # Extracted text
│   ├── cleaned_chunks/        # Text chunks
│   └── emb_chunks/            # Embedded chunks
├── docker/                    # Docker configurations
│   └── docker-compose.yml     # Milvus deployment
├── logs/                      # Application logs
├── .env                       # Configuration (create from .env.example)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

All configuration is managed through the `.env` file. Key settings:

- **API Keys**: VOYAGE_API_KEY, OPENAI_API_KEY
- **Chunking**: CHUNK_SIZE (default: 512 tokens), CHUNK_OVERLAP (default: 102 tokens)
- **Retrieval**: HYBRID_TOP_K (default: 20), RERANK_TOP_K (default: 5)
- **Models**: EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL
- **OCR**: PADDLEOCR_USE_GPU, OCR_CONFIDENCE_THRESHOLD

See `.env.example` for all available options.

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Embeddings | Voyage AI (voyage-3.5-lite) | Best cost-performance ratio |
| Vector DB | Milvus (local) | Fast indexing, billions of vectors support |
| OCR | PaddleOCR | 92% accuracy, lightweight |
| Reranking | Jina Reranker v2 | Open-source, state-of-the-art |
| LLM | OpenAI GPT-4o-mini | Cost-efficient, vision support |
| BM25 | rank-bm25 | Pure Python, fast |

## Cost Estimate

Based on benchmarks (see documentation):

- **Per query cost**: ~$0.00054
- **Per 1000 queries**: ~$0.54
- **Per month (10K queries)**: ~$5.40 API costs + $20-50 compute

## Troubleshooting

### Milvus Connection Error

```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus
cd docker
docker-compose restart
```

### PaddleOCR Installation Issues

```bash
# For CUDA 11.8, make sure to use the correct installation command
pip uninstall paddlepaddle paddlepaddle-gpu
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### NLTK Data Missing

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### API Key Not Set

Make sure your `.env` file contains:
```bash
VOYAGE_API_KEY=your_actual_key
OPENAI_API_KEY=your_actual_key
```

## Development

To contribute or modify:

1. Install in development mode:
   ```bash
   pip install -e .
   ```

2. Run tests (TODO):
   ```bash
   pytest tests/
   ```

3. Check logs:
   ```bash
   tail -f logs/app.log
   ```

## License

[To be determined]

## References

- [Architecture & Benchmarks](docu/001-architecture-and-benchmarks.md)
- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [Jina Reranker](https://jina.ai/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)
