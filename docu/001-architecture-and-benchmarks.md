# 001 - RAG Chatbot Architecture & Benchmark Analysis

**Document Version:** 1.0
**Last Updated:** 2025-12-16
**Implementation Stage:** CLI-based backend (frontend to be added later)

---

## Table of Contents

1. [Executive Summary & Recommendations](#1-executive-summary--recommendations)
2. [Benchmark Results & Analysis](#2-benchmark-results--analysis)
3. [Complete System Architecture](#3-complete-system-architecture)
4. [Detailed Component Specifications](#4-detailed-component-specifications)
5. [Data Flow & Directory Structure](#5-data-flow--directory-structure)
6. [Implementation Recommendations](#6-implementation-recommendations)
7. [Deployment Considerations](#7-deployment-considerations)
8. [Alternative Configurations](#8-alternative-configurations)
9. [Sources](#9-sources)

---

## 1. EXECUTIVE SUMMARY & RECOMMENDATIONS

### **Key Technology Choices (Based on 2025 Benchmarks):**

✅ **Embedding API**: **Voyage AI (voyage-3.5-lite)**
- **9.74% better** performance than OpenAI text-embedding-3-large
- **20.71% better** than Cohere embed-v4.0
- **1/6th the cost** of Cohere with comparable quality
- **$0.02 per million tokens** vs OpenAI's higher pricing
- 32K context window vs OpenAI's 8K

✅ **Vector Database**: **Milvus (Local deployment)**
- **Best choice validated**: Fast indexing, supports billions of vectors
- **<10ms p50 latency** for queries
- **Fastest indexing time** among competitors
- Most comprehensive indexing strategies
- Excellent for local/self-hosted deployment

✅ **OCR Solution**: **PaddleOCR**
- **Best choice validated**: 92% accuracy vs Tesseract's 85%
- **Lightweight** (<10MB) and very fast with GPU acceleration
- Excellent for multilingual documents and complex layouts
- Supports 80+ languages
- Advanced layout detection capabilities

✅ **Reranking**: **Jina Reranker v2 (Open-Source)** 🎯 **Cost Optimization Choice**
- **Open-source** and self-hostable (no API costs)
- **+33.7% MRR improvement** over baseline retrieval
- **State-of-the-art performance** on AirBench leaderboard
- **+7.9% Hit Rate improvement** (from 0.7908 to 0.8553)
- Excellent multilingual support (26+ languages)
- **Trade-off**: ~800ms latency vs Cohere's 595ms (acceptable for most use cases)
- **Cost savings**: $0 vs $2 per 1000 searches with Cohere

✅ **LLM Inference**: **GPT-4o-mini**
- **Most cost-efficient** vision-capable model
- **$0.15 input / $0.60 output** per 1M tokens
- 128K context window, supports text and vision
- 60%+ cheaper than GPT-3.5 Turbo
- Sufficient quality for RAG applications

✅ **Implementation Approach**: **CLI-based Backend First**
- Focus on core functionality and pipeline
- RESTful API design for easy frontend integration later
- Command-line interface for testing and administration
- Modular architecture for future expansion

---

## 2. BENCHMARK RESULTS & ANALYSIS

### 2.1 Embedding APIs Comparison

| Provider | Model | Accuracy vs Baseline | Cost per 1M tokens | Context Window | Best For |
|----------|-------|---------------------|-------------------|----------------|----------|
| **Voyage AI** ⭐ | voyage-3.5-lite | +6.34% vs OpenAI | **$0.02** | 32K | **Cost-performance balance** |
| Voyage AI | voyage-3-large | +9.74% vs OpenAI | $0.06 | 32K | Maximum accuracy |
| OpenAI | text-embedding-3-large | Baseline | ~$0.13 | 8K | General purpose |
| Cohere | embed-v4.0 | -14.37% vs Voyage | $0.12 (6x Voyage) | 128K | Long context needs |
| Mistral | mistral-embed | 77.8% accuracy | Varies | N/A | Specialized tasks |

**Key Insights:**
- Voyage-3.5-lite delivers near-identical quality to Cohere-v4 at **1/6th the cost**
- Voyage models consistently outperform OpenAI across 100+ diverse datasets
- Binary embeddings with Voyage-3-large achieve 200x storage reduction with better accuracy

**Recommendation**: **Voyage-3.5-lite** - Best cost-performance ratio for production RAG systems.

---

### 2.2 Vector Database Comparison

| Database | Query Latency (p50) | Indexing Speed | Scalability | Filtering | Best For |
|----------|---------------------|----------------|-------------|-----------|----------|
| **Milvus** ⭐ | **<10ms** | **Fastest** | Billions of vectors | Good | **Large-scale, local deployment** |
| Qdrant | 20-50ms | Fast | High | **Best** | Complex metadata filtering |
| Weaviate | 50ms+ | Medium | High | Good | Hybrid search features |
| Chroma | ~20ms | Fast | Small-Medium | Basic | Prototyping only |

**Key Insights:**
- Milvus excels at **indexing performance** while maintaining good precision
- For 768-dim embeddings: Milvus <10ms, Pinecone/Qdrant 20-50ms
- Milvus supports **distributed deployments** on Kubernetes
- Offers **more indexing strategies** than competitors (HNSW, IVF_FLAT, IVF_SQ8, etc.)

**Recommendation**: **Milvus** - Ideal for local deployment with enterprise-grade performance.

---

### 2.3 OCR Solutions Comparison

| Solution | Accuracy | Speed | Size | GPU Support | Languages | Best For |
|----------|----------|-------|------|-------------|-----------|----------|
| **PaddleOCR** ⭐ | **92%** | **Very Fast** | **<10MB** | Yes | 80+ | **Modern documents, layouts** |
| EasyOCR | 88-90% | Fast | ~50MB | Yes | 70+ | Scene text, GPU available |
| Tesseract | 85% | Medium | Varies | No | 100+ | Simple documents, legacy |

**Key Insights:**
- PaddleOCR demonstrates **superior precision** on complex documents vs Tesseract
- **GPU-accelerated**: Processes documents several times faster than CPU-only
- **Slanted bounding boxes**: Only PaddleOCR supports non-straight text detection
- Better performance on **tables and complex layouts**
- Superior on **multilingual documents** (especially Arabic, Asian languages)

**Recommendation**: **PaddleOCR** - Best accuracy-speed-size balance for production OCR.

---

### 2.4 Reranking Models Comparison

| Provider | Model | Type | Latency | Context Length | Improvement | Cost |
|----------|-------|------|---------|----------------|-------------|------|
| **Jina** ⭐ | Reranker v2 | Open-Source | ~800ms | Flexible | **+33.7% MRR** | **$0 (self-hosted)** |
| Cohere | Rerank 3.5 | API | 595ms | 4096 tokens | +48% quality | $2 / 1000 searches |
| Voyage | Rerank 2.5 | API | 603ms | Flexible | High | ~$1-2 / 1000 |

**Key Insights - Jina Reranker v2:**
- **Hit Rate improvement**: 0.7908 → 0.8553 (+7.9%)
- **MRR improvement**: 0.5307 → 0.7091 (+33.7%)
- **State-of-the-art** on AirBench leaderboard for RAG
- **Multilingual excellence**: Strong on MKQA dataset (26 languages)
- **Self-hostable**: Run on your own infrastructure
- **No API limits**: No rate limiting or usage costs

**Cost Optimization Decision**:
Using **Jina Reranker v2** saves **$2 per 1000 searches** compared to Cohere. For 100K searches/month, this translates to **$200/month savings** with comparable quality. The ~200ms additional latency (800ms vs 595ms) is acceptable for most RAG applications.

---

### 2.5 Hybrid Search & RAG Strategy

| Approach | Recall | Precision | Implementation Complexity | Improvement |
|----------|--------|-----------|--------------------------|-------------|
| Semantic only | Medium | Medium | Low | Baseline |
| Keyword (BM25) only | Medium | Low | Low | -10% to baseline |
| **Hybrid (Semantic + BM25)** | **High** | Medium | Medium | **+15-25%** |
| **Hybrid + Reranking** ⭐ | **High** | **High** | Medium-High | **+25-48%** |

**Key Insights:**
- **Hybrid retrieval** captures both semantic meaning and exact keyword matches
- **BM25 struggles** with semantic understanding; **dense retrieval overlooks** keywords
- **Reranking** can improve retrieval quality by **up to 48%** (Databricks research)
- **Reciprocal Rank Fusion (RRF)** is economical and fast for combining results
- **Cross-encoder reranking** significantly improves final ranking quality

**Implementation**: 3-stage pipeline (BM25 + Semantic → RRF → Rerank) maximizes both recall and precision.

---

## 3. COMPLETE SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MY_ATLAS RAG CHATBOT SYSTEM                          │
│                        (CLI-Based Backend)                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Documents (PDF, DOCX, scanned PDFs, etc.)                      │
│         │                                                                │
│         ├──────► Document Type Detection                                │
│         │        (check if scanned or text-based)                       │
│         │                                                                │
│         ├──► Regular PDF/DOCX ──► PyMuPDF/python-docx                  │
│         │         │                                                      │
│         │         └──► ext/raw_text/{doc}_{page}.json                   │
│         │              {                                                 │
│         │                "content": "extracted text",                   │
│         │                "metadata": {                                  │
│         │                  "source_path": "/path/to/doc.pdf",           │
│         │                  "document_type": "pdf",                      │
│         │                  "version": "1.0",                            │
│         │                  "created_date": "ISO-8601",                  │
│         │                  "modified_date": "ISO-8601",                 │
│         │                  "page_number": 1                             │
│         │                }                                               │
│         │              }                                                 │
│         │                                                                │
│         └──► Scanned PDF ──► PaddleOCR (PP-OCRv4)                      │
│                   │           - Layout detection                         │
│                   │           - Text recognition                         │
│                   │           - Reading order determination              │
│                   │                                                      │
│                   └──► ext/raw_text/{doc}_{page}.json                   │
│                        (same format as above)                            │
│                                                                          │
│         ▼                                                                │
│   Text Cleaning & Chunking                                              │
│         │                                                                │
│         ├──► Recursive Character Text Splitter                          │
│         │    - Chunk size: ENV configurable (default: 512 tokens)      │
│         │    - Overlap: 20% of chunk size (~102 tokens)                │
│         │    - Separators: ["\n\n", "\n", ". ", " ", ""]               │
│         │                                                                │
│         └──► ext/cleaned_chunks/{doc}_{chunk_id}.json                   │
│              {                                                           │
│                "content": "cleaned chunk text",                         │
│                "metadata": {                                            │
│                  "source_path": "/path/to/doc.pdf",                    │
│                  "document_type": "pdf",                               │
│                  "page_number": 1,                                     │
│                  "chunk_id": "doc_001_chunk_005"                       │
│                }                                                         │
│              }                                                           │
│                                                                          │
│         ▼                                                                │
│   Embedding Generation (Voyage AI)                                      │
│         │                                                                │
│         ├──► voyage-3.5-lite API                                        │
│         │    - Batch processing: 128 texts/request                     │
│         │    - Dimensions: 1024                                         │
│         │    - Context window: 32K tokens                              │
│         │                                                                │
│         ├──► ext/emb_chunks/{doc}_{chunk_id}.json                       │
│         │    {                                                           │
│         │      "content": "chunk text",                                 │
│         │      "metadata": {...},                                       │
│         │      "embedding": [0.123, -0.456, ...] // 1024-dim           │
│         │    }                                                           │
│         │                                                                │
│         └──► Milvus Vector Database                                     │
│              Collection: "documents"                                     │
│              Index: HNSW (M=16, efConstruction=200)                     │
│              Metric: COSINE                                             │
│              Fields:                                                     │
│                - id: INT64 (primary, auto_increment)                    │
│                - content: VARCHAR(65535)                                │
│                - embedding: FLOAT_VECTOR(1024)                          │
│                - metadata: JSON                                         │
│                - created_at: DATETIME                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE (Hybrid RAG)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Query (CLI input)                                                 │
│      │                                                                   │
│      ├──► Query Embedding (Voyage AI voyage-3.5-lite)                   │
│      │                                                                   │
│      └──► Parallel Retrieval (Hybrid Search)                            │
│            │                                                             │
│            ├──► 1. SEMANTIC SEARCH (Vector Similarity)                  │
│            │    ├─ Milvus search with query embedding                  │
│            │    ├─ Metric: Cosine similarity                           │
│            │    ├─ Search params: ef=100 (HNSW)                        │
│            │    └─ Output: Top K=20 results with scores                │
│            │                                                             │
│            ├──► 2. KEYWORD SEARCH (BM25)                                │
│            │    ├─ In-memory BM25 index (rank-bm25)                    │
│            │    ├─ Parameters: k1=1.5, b=0.75                          │
│            │    └─ Output: Top K=20 results with scores                │
│            │                                                             │
│            └──► 3. RECIPROCAL RANK FUSION (RRF)                         │
│                 ├─ Formula: RRF_score = Σ(1 / (k + rank_i))            │
│                 ├─ k = 60 (standard parameter)                          │
│                 ├─ Combine results from both retrievers                │
│                 ├─ Deduplicate by chunk_id                             │
│                 └─ Output: Top K=20 fused results                       │
│                      │                                                   │
│                      ▼                                                   │
│            RERANKING (Jina Reranker v2 - Local)                         │
│                      │                                                   │
│                      ├──► Cross-encoder scoring                         │
│                      │    - Model: jina-reranker-v2-base-multilingual  │
│                      │    - Input: Query + 20 candidates                │
│                      │    - Self-hosted (no API costs)                  │
│                      │                                                   │
│                      └──► Final Top K=5 chunks                          │
│                           (with relevance scores)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              INFERENCE & ANSWER VALIDATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Top 5 Reranked Chunks + User Query                                     │
│         │                                                                │
│         ├──► Context Formation                                          │
│         │    - Combine top 5 chunks                                     │
│         │    - Add source citations (page numbers, doc names)           │
│         │    - Format for LLM consumption                               │
│         │                                                                │
│         └──► Primary LLM Inference (GPT-4o-mini)                        │
│              ├─ Model: gpt-4o-mini                                      │
│              ├─ Temperature: 0.1 (low for factual responses)            │
│              ├─ Max tokens: 1000                                        │
│              ├─ System prompt: "Answer based ONLY on provided context" │
│              ├─ Cost: $0.15 input / $0.60 output per 1M tokens         │
│              │                                                           │
│              └──► Generated Answer                                      │
│                      │                                                   │
│                      ▼                                                   │
│              Answer Validation Step                                     │
│              (GPT-4o-mini as validator)                                 │
│                      │                                                   │
│         ┌────────────┴────────────┐                                     │
│         │                          │                                     │
│    Validates? ✓              Validates? ✗                               │
│         │                          │                                     │
│         ▼                          ▼                                     │
│  Return Grounded Answer    Fallback Inference                           │
│  + Source Citations        (GPT-4o-mini, NO context)                    │
│  + Page numbers                   │                                     │
│  + Confidence score               └──► Generic Answer                   │
│                                         with warning:                    │
│                                         "⚠️ No answer found in           │
│                                         knowledge base. Here's           │
│                                         general information              │
│                                         (may not be up-to-date)..."      │
│                                                                          │
│  Output to CLI                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 CLI Interface Structure

```
my-atlas-cli
│
├── ingest <path-to-documents>     # Ingest documents into knowledge base
│   ├── --doc-type [pdf|docx|all]
│   ├── --use-ocr [true|false]
│   └── --batch-size <number>
│
├── query "<question>"              # Query the knowledge base
│   ├── --top-k <number>
│   ├── --verbose [show retrieval details]
│   └── --no-fallback [disable generic answers]
│
├── status                          # Show system status
│   ├── --docs [document count]
│   ├── --chunks [chunk statistics]
│   └── --db [database health]
│
├── clear                           # Clear knowledge base
│   └── --confirm
│
└── config                          # Show/update configuration
    ├── --show
    └── --set <key=value>
```

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### 4.1 Document Processing Components

#### 4.1.1 Text Extraction (Regular PDFs/Documents)

**Libraries:**
```python
# PDF extraction
- PyMuPDF (fitz): Best for text + images + metadata
- fallback: PyPDF2 for simple PDFs

# Office documents
- python-docx: DOCX files
- python-pptx: PPT files (optional)

# Plain text
- Standard file I/O for TXT, MD files
```

**Output Format:**
```json
// ext/raw_text/{document_name}_page_{page_num}.json
{
  "content": "Extracted text content from the page...",
  "metadata": {
    "source_path": "/absolute/path/to/document.pdf",
    "document_type": "pdf",
    "document_name": "document.pdf",
    "version": "1.0",
    "created_date": "2025-01-15T10:30:00Z",
    "modified_date": "2025-01-16T14:20:00Z",
    "page_number": 1,
    "total_pages": 10
  }
}
```

**Implementation Notes:**
- Preserve formatting markers (headings, lists) for better chunking
- Extract metadata directly from PDF properties
- Handle multi-column layouts
- Detect and skip image-only pages (flag for OCR)

---

#### 4.1.2 OCR Processing (Scanned PDFs)

**PaddleOCR Configuration:**
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,      # Enable angle classification
    lang='en',                # Primary language (configurable)
    use_gpu=True,             # GPU acceleration
    show_log=False,           # Reduce logging
    det_model_dir=None,       # Use default PP-OCRv4 detection
    rec_model_dir=None,       # Use default PP-OCRv4 recognition
    cls_model_dir=None,       # Use default angle classifier
    det_db_thresh=0.3,        # Detection threshold
    det_db_box_thresh=0.6,    # Box threshold
    rec_batch_num=6           # Batch size for recognition
)
```

**Processing Pipeline:**
1. **Layout Detection**: Identify text regions, tables, images
2. **Text Recognition**: OCR each region with confidence scoring
3. **Reading Order**: Determine logical reading sequence
4. **Confidence Filtering**: Drop results below threshold (default: 0.7)
5. **Output**: Same JSON format as regular extraction

**Output Format:** Identical to 4.1.1 (ext/raw_text/)

**Performance Optimization:**
- Process pages in parallel batches (4-8 pages)
- Use GPU if available (10x+ speedup)
- Cache OCR results per document hash
- Skip already processed pages

---

### 4.2 Text Chunking Strategy

**Configuration (.env):**
```bash
CHUNK_SIZE=512              # tokens (optimal for RAG: 256-512)
CHUNK_OVERLAP=102           # 20% of chunk size
CHUNKING_METHOD=recursive   # or 'semantic' (future)
```

**Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Use tiktoken for accurate token counting (matches OpenAI/Voyage)
encoding = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text: str) -> int:
    tokens = encoding.encode(text, disallowed_special=())
    return len(tokens)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv('CHUNK_SIZE', 512)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 102)),
    length_function=tiktoken_len,
    separators=[
        "\n\n",   # Paragraph breaks (highest priority)
        "\n",     # Line breaks
        ". ",     # Sentences
        ", ",     # Clauses
        " ",      # Words
        ""        # Characters (fallback)
    ],
    keep_separator=True
)
```

**Chunking Process:**
1. Load raw text from ext/raw_text/
2. Apply cleaning (remove excessive whitespace, fix encoding)
3. Split into chunks with overlap
4. Generate unique chunk IDs
5. Preserve metadata from parent document
6. Save to ext/cleaned_chunks/

**Output Format:**
```json
// ext/cleaned_chunks/{document_name}_chunk_{chunk_num}.json
{
  "content": "This is the cleaned chunk text content...",
  "metadata": {
    "source_path": "/absolute/path/to/document.pdf",
    "document_type": "pdf",
    "document_name": "document.pdf",
    "page_number": 1,
    "chunk_id": "document_chunk_0005",
    "chunk_index": 5,
    "char_start": 1024,
    "char_end": 1536,
    "token_count": 512
  }
}
```

**Best Practices:**
- Smaller chunks (256-384) for Q&A tasks
- Larger chunks (512-1024) for summarization
- Maintain context with 10-20% overlap
- Preserve sentence boundaries when possible

---

### 4.3 Embedding Generation & Vector Storage

#### 4.3.1 Voyage AI Embedding

**Configuration:**
```python
import voyageai
import os

vo = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))

EMBEDDING_CONFIG = {
    'model': 'voyage-3.5-lite',
    'input_type': 'document',  # or 'query' for search queries
    'batch_size': 128,          # API allows up to 128 texts/request
    'dimensions': 1024,         # Default for voyage-3.5-lite
}
```

**Batch Processing:**
```python
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed chunks in batches for efficiency."""
    embeddings = []

    for i in range(0, len(chunks), EMBEDDING_CONFIG['batch_size']):
        batch = chunks[i:i + EMBEDDING_CONFIG['batch_size']]

        result = vo.embed(
            texts=batch,
            model=EMBEDDING_CONFIG['model'],
            input_type=EMBEDDING_CONFIG['input_type']
        )

        embeddings.extend(result.embeddings)

    return embeddings
```

**Output Format:**
```json
// ext/emb_chunks/{document_name}_chunk_{chunk_num}.json
{
  "content": "Chunk text content...",
  "metadata": {
    "source_path": "/absolute/path/to/document.pdf",
    "document_type": "pdf",
    "document_name": "document.pdf",
    "page_number": 1,
    "chunk_id": "document_chunk_0005",
    "chunk_index": 5,
    "embedding_model": "voyage-3.5-lite",
    "embedding_dim": 1024,
    "created_at": "2025-12-16T10:30:00Z"
  },
  "embedding": [0.0234, -0.0567, 0.0891, ...]  // 1024-dimensional vector
}
```

**Cost Tracking:**
- Log tokens processed per batch
- Accumulate total embedding costs
- Cache embeddings to avoid re-embedding

---

#### 4.3.2 Milvus Vector Database

**Collection Schema:**
```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="created_at", dtype=DataType.INT64)  # Unix timestamp
]

schema = CollectionSchema(
    fields=fields,
    description="Document chunks with embeddings"
)

collection = Collection(
    name="documents",
    schema=schema
)
```

**Index Configuration:**
```python
# HNSW Index (best accuracy-speed trade-off)
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,                # Max connections per layer (8-64, higher = better recall)
        "efConstruction": 200   # Build-time accuracy (100-500, higher = better index)
    }
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

# Alternative: IVF_FLAT for larger datasets
# index_params = {
#     "index_type": "IVF_FLAT",
#     "metric_type": "COSINE",
#     "params": {"nlist": 1024}  # Number of clusters
# }
```

**Search Configuration:**
```python
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 100  # Search-time accuracy (higher = better recall, slower)
    }
}

# For IVF_FLAT:
# search_params = {
#     "metric_type": "COSINE",
#     "params": {"nprobe": 10}  # Clusters to search
# }
```

**Insertion Example:**
```python
# Prepare data for insertion
entities = [
    chunk_ids,      # List[str]
    contents,       # List[str]
    embeddings,     # List[List[float]]
    metadatas,      # List[dict]
    timestamps      # List[int]
]

# Insert into Milvus
collection.insert(entities)
collection.flush()  # Ensure data is persisted
```

---

### 4.4 Hybrid Retrieval System

#### 4.4.1 Semantic Search (Milvus)

```python
from pymilvus import Collection

def semantic_search(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    """Perform vector similarity search."""

    collection = Collection("documents")
    collection.load()  # Load collection to memory

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "content", "metadata"]
    )

    # Format results
    candidates = []
    for hit in results[0]:
        candidates.append({
            "chunk_id": hit.entity.get("chunk_id"),
            "content": hit.entity.get("content"),
            "metadata": hit.entity.get("metadata"),
            "score": hit.score,
            "source": "semantic"
        })

    return candidates
```

---

#### 4.4.2 Keyword Search (BM25)

```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
# nltk.download('punkt')

class BM25Index:
    def __init__(self, documents: list[dict]):
        """Initialize BM25 index with documents.

        Args:
            documents: List of dicts with 'chunk_id', 'content', 'metadata'
        """
        self.documents = documents
        self.chunk_ids = [doc['chunk_id'] for doc in documents]

        # Tokenize all documents
        tokenized_docs = [
            word_tokenize(doc['content'].lower())
            for doc in documents
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=1.5,  # Term frequency saturation
            b=0.75   # Length normalization
        )

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search using BM25 keyword matching."""

        # Tokenize query
        tokenized_query = word_tokenize(query.lower())

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top K results
        top_indices = scores.argsort()[-top_k:][::-1]

        candidates = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                candidates.append({
                    "chunk_id": self.documents[idx]['chunk_id'],
                    "content": self.documents[idx]['content'],
                    "metadata": self.documents[idx]['metadata'],
                    "score": float(scores[idx]),
                    "source": "bm25"
                })

        return candidates
```

---

#### 4.4.3 Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    alpha: float = 0.5
) -> list[dict]:
    """Combine results using Reciprocal Rank Fusion.

    Args:
        semantic_results: Results from vector search
        bm25_results: Results from BM25 search
        k: RRF constant (default: 60)
        alpha: Weight for semantic vs keyword (0.5 = equal weight)

    Returns:
        Fused and deduplicated results sorted by RRF score
    """

    rrf_scores = {}

    # Process semantic results
    for rank, result in enumerate(semantic_results):
        chunk_id = result['chunk_id']
        rrf_score = alpha * (1.0 / (k + rank + 1))

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                'score': 0,
                'content': result['content'],
                'metadata': result['metadata'],
                'sources': []
            }

        rrf_scores[chunk_id]['score'] += rrf_score
        rrf_scores[chunk_id]['sources'].append('semantic')

    # Process BM25 results
    for rank, result in enumerate(bm25_results):
        chunk_id = result['chunk_id']
        rrf_score = (1 - alpha) * (1.0 / (k + rank + 1))

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                'score': 0,
                'content': result['content'],
                'metadata': result['metadata'],
                'sources': []
            }

        rrf_scores[chunk_id]['score'] += rrf_score
        rrf_scores[chunk_id]['sources'].append('bm25')

    # Sort by RRF score
    fused_results = [
        {
            'chunk_id': chunk_id,
            'content': data['content'],
            'metadata': data['metadata'],
            'rrf_score': data['score'],
            'sources': list(set(data['sources']))
        }
        for chunk_id, data in rrf_scores.items()
    ]

    fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)

    return fused_results
```

---

#### 4.4.4 Reranking with Jina Reranker v2

**Model Setup:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class JinaReranker:
    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
        """Initialize Jina Reranker v2 model (self-hosted)."""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: User query
            candidates: List of candidate chunks from RRF
            top_k: Number of top results to return

        Returns:
            Top K reranked results with relevance scores
        """

        # Prepare input pairs
        pairs = [[query, candidate['content']] for candidate in candidates]

        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get relevance scores
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # Attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])

        # Sort by rerank score and return top K
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        return candidates[:top_k]
```

**Usage:**
```python
# Initialize reranker (do this once at startup)
reranker = JinaReranker()

# Rerank fusion results
top_reranked = reranker.rerank(
    query="What is RAG?",
    candidates=rrf_results[:20],  # Top 20 from RRF
    top_k=5
)
```

**Model Information:**
- **Model**: jinaai/jina-reranker-v2-base-multilingual
- **Size**: ~560MB
- **Languages**: 89+ languages
- **Context Length**: 512 tokens per pair (query + document)
- **Performance**: State-of-the-art on AirBench for RAG tasks

**Cost Optimization Notes:**
- **Self-hosted**: No API costs, only compute resources
- **GPU recommended**: ~800ms on GPU vs ~2-3s on CPU per batch
- **Batch processing**: Can rerank up to 32 pairs in parallel
- **Alternative**: jina-reranker-v2-base-en for English-only (faster)

---

### 4.5 LLM Inference & Answer Validation

#### 4.5.1 Primary Inference (GPT-4o-mini)

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

Instructions:
1. Answer the question using ONLY information from the context below
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question"
3. Always cite the source page number when providing information
4. Be concise and factual
5. Do not use external knowledge or make assumptions"""

def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """Generate answer using GPT-4o-mini with context.

    Args:
        query: User question
        context_chunks: Top K reranked chunks

    Returns:
        dict with answer, sources, and metadata
    """

    # Format context with citations
    context_text = "\n\n".join([
        f"[Source: {chunk['metadata']['document_name']}, "
        f"Page {chunk['metadata']['page_number']}]\n{chunk['content']}"
        for chunk in context_chunks
    ])

    user_message = f"""Context:
{context_text}

Question: {query}

Answer:"""

    # Call GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,  # Low temperature for factual responses
        max_tokens=1000,
        top_p=0.95
    )

    answer = response.choices[0].message.content

    # Extract sources
    sources = list(set([
        f"{chunk['metadata']['document_name']} (p. {chunk['metadata']['page_number']})"
        for chunk in context_chunks
    ]))

    return {
        "answer": answer,
        "sources": sources,
        "model": "gpt-4o-mini",
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }
```

---

#### 4.5.2 Answer Validation

```python
VALIDATION_PROMPT = """You are an answer quality validator.

Given a question and an answer, determine if the answer adequately addresses the question.

An answer is VALID if:
- It directly addresses the question asked
- It provides specific information (not just "I don't know")
- It's based on the provided context

An answer is INVALID if:
- It says there's not enough information
- It doesn't address the question
- It's too vague or generic

Respond with ONLY "VALID" or "INVALID" followed by a brief one-sentence explanation."""

def validate_answer(query: str, answer: str) -> dict:
    """Validate if answer adequately addresses the query.

    Args:
        query: Original user question
        answer: Generated answer

    Returns:
        dict with is_valid bool and explanation
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": VALIDATION_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"}
        ],
        temperature=0.0,
        max_tokens=100
    )

    validation_result = response.choices[0].message.content

    is_valid = validation_result.upper().startswith("VALID")

    return {
        "is_valid": is_valid,
        "explanation": validation_result
    }
```

---

#### 4.5.3 Fallback Inference

```python
FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant with general knowledge.

IMPORTANT: The user's question could not be answered from their knowledge base.
Provide a general answer using your training data, but:
1. Clearly state this is general information, not from their documents
2. Note that the information may not be up-to-date
3. Keep the answer concise
4. Suggest they verify from authoritative sources"""

def generate_fallback_answer(query: str) -> dict:
    """Generate generic answer when grounded answer fails validation.

    Args:
        query: User question

    Returns:
        dict with fallback answer and warning
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.3,  # Slightly higher for general knowledge
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Add warning prefix
    warning = "⚠️ No answer found in your knowledge base. Here's general information (may not be up-to-date):\n\n"

    return {
        "answer": warning + answer,
        "sources": ["General AI knowledge (not from documents)"],
        "model": "gpt-4o-mini",
        "is_fallback": True,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }
```

---

#### 4.5.4 Complete Query Pipeline

```python
def process_query(query: str, retriever, reranker) -> dict:
    """Complete end-to-end query processing pipeline.

    Args:
        query: User question
        retriever: HybridRetriever instance
        reranker: JinaReranker instance

    Returns:
        dict with final answer and metadata
    """

    # Step 1: Hybrid retrieval
    rrf_results = retriever.search(query, top_k=20)

    # Step 2: Reranking
    top_chunks = reranker.rerank(query, rrf_results, top_k=5)

    # Step 3: Generate answer with context
    result = generate_answer(query, top_chunks)

    # Step 4: Validate answer
    validation = validate_answer(query, result['answer'])

    # Step 5: Fallback if invalid
    if not validation['is_valid']:
        result = generate_fallback_answer(query)

    # Add metadata
    result['query'] = query
    result['validation'] = validation
    result['retrieved_chunks'] = len(top_chunks)

    return result
```

---

## 5. DATA FLOW & DIRECTORY STRUCTURE

### 5.1 Project Directory Structure

```
my_atlas/
├── .env                           # Environment variables & configuration
├── .env.example                   # Example environment file
├── .gitignore                     # Git ignore file
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
│
├── docu/                          # Documentation
│   └── 001-architecture-and-benchmarks.md
│
├── ext/                           # External data storage (not in git)
│   ├── raw_text/                  # Raw extracted text (1 JSON per page)
│   │   ├── doc1_page_001.json
│   │   ├── doc1_page_002.json
│   │   └── ...
│   │
│   ├── cleaned_chunks/            # Cleaned text chunks
│   │   ├── doc1_chunk_0001.json
│   │   ├── doc1_chunk_0002.json
│   │   └── ...
│   │
│   └── emb_chunks/                # Embedded chunks (content + vector)
│       ├── doc1_chunk_0001.json
│       ├── doc1_chunk_0002.json
│       └── ...
│
├── data/                          # Input documents (not in git)
│   ├── documents/                 # PDF, DOCX, etc.
│   └── scanned/                   # Scanned PDFs for OCR
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── ingestion/                 # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py       # Abstract base class
│   │   │   ├── pdf_extractor.py        # PyMuPDF-based extraction
│   │   │   ├── ocr_extractor.py        # PaddleOCR integration
│   │   │   └── docx_extractor.py       # DOCX extraction
│   │   ├── chunker.py                   # Text chunking logic
│   │   ├── embedder.py                  # Voyage AI embedding
│   │   └── pipeline.py                  # Orchestrate ingestion
│   │
│   ├── retrieval/                 # Retrieval pipeline
│   │   ├── __init__.py
│   │   ├── vector_search.py             # Milvus semantic search
│   │   ├── bm25_search.py               # BM25 keyword search
│   │   ├── hybrid_retriever.py          # RRF fusion
│   │   └── reranker.py                  # Jina Reranker v2
│   │
│   ├── inference/                 # LLM inference
│   │   ├── __init__.py
│   │   ├── llm_client.py                # OpenAI GPT-4o-mini
│   │   └── validator.py                 # Answer validation
│   │
│   ├── database/                  # Database clients
│   │   ├── __init__.py
│   │   └── milvus_client.py             # Milvus operations
│   │
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                    # Config management
│   │   ├── logger.py                    # Logging setup
│   │   └── file_utils.py                # File operations
│   │
│   └── cli/                       # CLI interface
│       ├── __init__.py
│       ├── main.py                      # CLI entry point
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── ingest.py                # Ingest command
│       │   ├── query.py                 # Query command
│       │   ├── status.py                # Status command
│       │   └── config.py                # Config command
│       └── utils.py                     # CLI helpers
│
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_ingestion/
│   ├── test_retrieval/
│   └── test_inference/
│
├── docker/                        # Docker configurations
│   └── docker-compose.yml         # Milvus deployment
│
└── logs/                          # Application logs (not in git)
    ├── ingestion.log
    ├── retrieval.log
    └── app.log
```

### 5.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION FLOW                          │
└─────────────────────────────────────────────────────────────────┘

data/documents/               ext/raw_text/              ext/cleaned_chunks/
      │                             │                            │
      ├─ doc1.pdf ────┐            │                            │
      ├─ doc2.docx ───┼───► Extractor ──► doc1_page_001.json   │
      └─ doc3.pdf ────┘   (PyMuPDF/      doc1_page_002.json    │
         (scanned)         PaddleOCR)            │              │
                                                 │              │
                                                 └───► Chunker ──► doc1_chunk_0001.json
                                                      (LangChain)  doc1_chunk_0002.json
                                                                         │
                                                                         │
   ext/emb_chunks/                                                      │
         │                                                               │
         ├─ doc1_chunk_0001.json ◄───── Embedder ◄────────────────────┘
         ├─ doc1_chunk_0002.json        (Voyage AI)
         │                                    │
         │                                    │
    Milvus Vector DB ◄──────────────────────┘
    (documents collection)

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY FLOW                             │
└─────────────────────────────────────────────────────────────────┘

User Query (CLI)
      │
      ├──► Voyage AI ──► Query Embedding
      │                        │
      │                        │
      └────────────────────────┼──────────────┐
                               │              │
                               ▼              ▼
                       Milvus Search      BM25 Search
                       (Semantic)         (Keyword)
                            │                  │
                            └────┬─────────────┘
                                 │
                                 ▼
                         RRF Fusion (Top 20)
                                 │
                                 ▼
                      Jina Reranker v2 (Top 5)
                                 │
                                 ▼
                      GPT-4o-mini + Context
                                 │
                                 ▼
                         Answer Validation
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             Valid Answer              Invalid Answer
                    │                         │
                    │                         ▼
                    │              GPT-4o-mini (No Context)
                    │                         │
                    └────────┬────────────────┘
                             │
                             ▼
                      Final Answer (CLI)
```

---

## 6. IMPLEMENTATION RECOMMENDATIONS

### 6.1 Complete Technology Stack

| Layer | Component | Technology | Rationale |
|-------|-----------|------------|-----------|
| **CLI** | Interface | Click or Typer | Best Python CLI frameworks |
| **Extraction** | PDF | PyMuPDF (fitz) | Fast, comprehensive metadata |
| | DOCX | python-docx | Standard library |
| | OCR | PaddleOCR | Best accuracy, lightweight |
| **Chunking** | Text splitting | LangChain RecursiveCharacterTextSplitter | Production-ready, flexible |
| **Embedding** | API | Voyage AI (voyage-3.5-lite) | Best cost-performance |
| **Vector DB** | Storage | Milvus (local) | Fast indexing, scalable |
| **Retrieval** | Keyword | rank-bm25 | Pure Python, fast |
| | Reranking | Jina Reranker v2 | Open-source, state-of-the-art |
| **LLM** | Inference | OpenAI GPT-4o-mini | Cost-efficient, vision support |
| **Orchestration** | Framework | Custom (Python 3.11+) | Full control, lightweight |
| **Config** | Management | python-dotenv | Standard .env handling |
| **Logging** | System | Python logging | Built-in, flexible |

---

### 6.2 Environment Configuration (.env)

```bash
# ========================================
# My Atlas RAG Chatbot Configuration
# ========================================

# ----- API Keys -----
VOYAGE_API_KEY=your_voyage_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# ----- Embedding Configuration -----
EMBEDDING_MODEL=voyage-3.5-lite
EMBEDDING_BATCH_SIZE=128
EMBEDDING_DIMENSIONS=1024

# ----- Vector Database (Milvus) -----
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=documents
MILVUS_INDEX_TYPE=HNSW          # or IVF_FLAT
MILVUS_METRIC_TYPE=COSINE

# HNSW Index Parameters
MILVUS_HNSW_M=16
MILVUS_HNSW_EF_CONSTRUCTION=200
MILVUS_HNSW_EF_SEARCH=100

# ----- Chunking Configuration -----
CHUNK_SIZE=512                  # tokens
CHUNK_OVERLAP=102               # ~20% of chunk size
CHUNKING_METHOD=recursive       # or semantic (future)

# ----- Retrieval Configuration -----
HYBRID_TOP_K=20                 # Candidates for reranking
RERANK_TOP_K=5                  # Final chunks for LLM
RRF_K=60                        # RRF constant
RRF_ALPHA=0.5                   # Semantic vs keyword weight (0.5 = equal)

# ----- Reranking (Jina) -----
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKER_DEVICE=cuda            # or cpu
RERANKER_MAX_LENGTH=512         # tokens per pair

# ----- LLM Configuration -----
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
LLM_TOP_P=0.95

# Fallback LLM (when validation fails)
FALLBACK_LLM_TEMPERATURE=0.3
FALLBACK_LLM_MAX_TOKENS=500

# ----- OCR Configuration -----
PADDLEOCR_LANG=en               # Language code
PADDLEOCR_USE_GPU=true
PADDLEOCR_USE_ANGLE_CLS=true
OCR_CONFIDENCE_THRESHOLD=0.7
OCR_BATCH_SIZE=6

# ----- Storage Paths -----
DATA_DIR=./data
EXT_RAW_TEXT_DIR=./ext/raw_text
EXT_CLEANED_CHUNKS_DIR=./ext/cleaned_chunks
EXT_EMB_CHUNKS_DIR=./ext/emb_chunks
LOGS_DIR=./logs

# ----- Logging -----
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json                 # json or text

# ----- Performance -----
MAX_WORKERS=4                   # Parallel processing workers
EMBEDDING_CACHE=true            # Cache embeddings to avoid re-computation
```

---

### 6.3 Dependencies (requirements.txt)

```txt
# Core
python-dotenv==1.0.0
click==8.1.7                    # CLI framework
pydantic==2.5.0                 # Data validation
python-json-logger==2.0.7       # JSON logging

# Document Processing
PyMuPDF==1.23.8                 # PDF extraction
python-docx==1.1.0              # DOCX extraction
paddleocr==2.7.3                # OCR
paddlepaddle-gpu==2.6.0         # PaddlePaddle (GPU version)
# paddlepaddle==2.6.0           # Use this for CPU-only

# Text Processing
langchain==0.1.0                # Text splitting
langchain-text-splitters==0.0.1
tiktoken==0.5.2                 # Token counting
nltk==3.8.1                     # Tokenization for BM25
rank-bm25==0.2.2                # BM25 algorithm

# Embeddings
voyageai==0.2.0                 # Voyage AI SDK

# Vector Database
pymilvus==2.3.4                 # Milvus client

# Reranking
transformers==4.36.0            # Hugging Face transformers
torch==2.1.2                    # PyTorch
# torch==2.1.2+cu118            # Use this for GPU support

# LLM
openai==1.6.1                   # OpenAI SDK

# Utilities
tqdm==4.66.1                    # Progress bars
requests==2.31.0                # HTTP requests
```

**Installation:**
```bash
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Download Jina Reranker model (first run)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-reranker-v2-base-multilingual')"
```

---

### 6.4 Cost Analysis (Updated with Jina Reranker)

**Assumptions:**
- 1000 queries
- Average query: 10 tokens
- Average context retrieval: 2000 tokens (5 chunks × 400 tokens)
- Average response: 300 tokens
- Self-hosted Jina Reranker (GPU)

| Component | Unit Cost | Usage per Query | Cost per 1000 Queries |
|-----------|-----------|----------------|----------------------|
| **Query Embedding** | $0.02 / 1M tokens | 10 tokens | $0.0002 |
| **Document Embedding** (one-time) | $0.02 / 1M tokens | N/A (cached) | $0.00 |
| **Reranking (Jina v2)** | **$0** (self-hosted) | 1 search | **$0.00** ✅ |
| **LLM Input** | $0.15 / 1M tokens | 2010 tokens | $0.30 |
| **LLM Output** | $0.60 / 1M tokens | 300 tokens | $0.18 |
| **Validation Input** | $0.15 / 1M tokens | 320 tokens | $0.05 |
| **Validation Output** | $0.60 / 1M tokens | 20 tokens | $0.01 |
| **TOTAL** | | | **$0.54 / 1000 queries** |

**Per Query Cost: ~$0.00054** 🎯

**Cost Comparison:**
- **With Cohere Rerank**: $2.48 per 1000 queries
- **With Jina Reranker v2**: $0.54 per 1000 queries
- **Savings**: **78% reduction** ($1.94 per 1000 queries)

**Monthly Estimates (10K queries/month):**
- API costs: ~$5.40/month
- Compute costs (GPU for reranker): ~$20-50/month (depending on cloud provider)
- **Total**: ~$25-55/month

**One-Time Costs:**
- Document embedding (100K chunks): ~$2-5 (one-time)

---

### 6.5 Performance Optimization Strategies

#### 6.5.1 Ingestion Optimizations

1. **Parallel Processing**
   - Process multiple documents simultaneously (4-8 workers)
   - Batch embed chunks (128 per API call)
   - Async I/O for file operations

2. **Caching**
   - Hash-based duplicate detection (skip re-processing)
   - Cache document metadata
   - Store raw text to avoid re-extraction

3. **Incremental Updates**
   - Only process new/modified documents
   - Track document versions and timestamps
   - Update embeddings only for changed chunks

#### 6.5.2 Retrieval Optimizations

1. **Index Optimization**
   - Tune HNSW parameters for dataset size
   - Use IVF_FLAT for >1M vectors
   - Regularly rebuild indexes for optimal performance

2. **Caching**
   - Cache frequent queries (LRU cache)
   - Cache query embeddings
   - Pre-warm BM25 index on startup

3. **Batching**
   - Batch multiple queries when possible
   - Reuse reranker model in memory

#### 6.5.3 Inference Optimizations

1. **Prompt Caching**
   - Cache identical system prompts
   - Reuse context for follow-up questions

2. **Model Optimization**
   - Use FP16 for Jina Reranker (2x faster, minimal quality loss)
   - Quantize models if on CPU (INT8)

3. **Parallel Processing**
   - Run validation concurrently with answer generation
   - Async API calls

---

### 6.6 Quality Improvement Strategies

#### 6.6.1 Retrieval Quality

1. **Contextual Chunk Enrichment**
   ```python
   # Add document context to each chunk
   enriched_chunk = f"""Document: {doc_title}
   Section: {section_name}

   {chunk_content}"""
   ```

2. **Metadata Filtering**
   - Filter by document type, date range, source
   - Boost recent documents
   - Filter by user permissions (future)

3. **Query Expansion**
   - Generate related queries
   - Use synonyms for keyword search
   - Rephrase ambiguous queries

#### 6.6.2 Answer Quality

1. **Citation Accuracy**
   - Return exact page numbers
   - Highlight relevant sentences
   - Link to original documents

2. **Confidence Scoring**
   - Show retrieval confidence
   - Display rerank scores
   - Indicate validation results

3. **Multi-Step Reasoning**
   - Chain-of-thought prompting
   - Break complex questions into sub-questions
   - Aggregate answers from multiple chunks

#### 6.6.3 User Feedback Loop

1. **Rating System**
   - Thumbs up/down on answers
   - Relevance ratings (1-5 stars)
   - Report incorrect information

2. **Active Learning**
   - Identify low-confidence queries
   - Collect user corrections
   - Retrain/tune based on feedback

3. **Analytics**
   - Track query patterns
   - Identify knowledge gaps
   - Monitor answer quality metrics

---

## 7. DEPLOYMENT CONSIDERATIONS

### 7.1 Milvus Deployment (Docker Compose)

**File:** `docker/docker-compose.yml`

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

**Startup:**
```bash
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f milvus
```

---

### 7.2 System Requirements

#### 7.2.1 Minimum Requirements (Development)

- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: Optional (for faster OCR and reranking)
- **Storage**: 20GB free space
- **OS**: Linux, macOS, Windows (WSL2)

#### 7.2.2 Recommended Requirements (Production)

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for PaddleOCR + Jina Reranker)
- **Storage**: 100GB+ SSD
- **OS**: Linux (Ubuntu 20.04+ or similar)

#### 7.2.3 Scaling Considerations

| Scale | Documents | Chunks | RAM | Storage | Notes |
|-------|-----------|--------|-----|---------|-------|
| **Small** | <1,000 | <100K | 8GB | 20GB | Single instance, in-memory BM25 |
| **Medium** | 1K-10K | 100K-1M | 16GB | 100GB | Milvus cluster, persistent BM25 |
| **Large** | 10K-100K | 1M-10M | 32GB+ | 500GB+ | Distributed Milvus, Elasticsearch |
| **Enterprise** | >100K | >10M | 64GB+ | 1TB+ | Multi-node cluster, load balancing |

---

### 7.3 Monitoring & Maintenance

#### 7.3.1 Health Checks

```python
# Milvus health
from pymilvus import utility

def check_milvus_health():
    return utility.get_server_version()

# Reranker health
def check_reranker_health():
    return reranker.model is not None

# API health
import openai
import voyageai

def check_api_health():
    # Test OpenAI
    openai.models.list()

    # Test Voyage
    vo.embed(texts=["test"], model="voyage-3.5-lite")
```

#### 7.3.2 Metrics to Track

1. **Ingestion Metrics**
   - Documents processed per hour
   - Extraction errors
   - Embedding costs
   - Storage usage

2. **Retrieval Metrics**
   - Queries per second
   - Average latency (p50, p95, p99)
   - Retrieval quality (MRR, NDCG)
   - Cache hit rate

3. **Inference Metrics**
   - Answer quality (user ratings)
   - Validation pass rate
   - Fallback frequency
   - Token usage and costs

#### 7.3.3 Maintenance Tasks

1. **Daily**
   - Monitor error logs
   - Check API usage/costs
   - Verify database connectivity

2. **Weekly**
   - Review user feedback
   - Analyze query patterns
   - Check storage usage

3. **Monthly**
   - Rebuild Milvus indexes
   - Archive old logs
   - Review and tune hyperparameters
   - Update dependencies

---

## 8. ALTERNATIVE CONFIGURATIONS

### 8.1 Budget-Conscious Configuration

**Goal**: Minimize costs while maintaining acceptable quality

| Component | Change | Cost Impact | Quality Impact |
|-----------|--------|-------------|----------------|
| Embedding | OpenAI text-embedding-3-small | +$0.01/1M → Same as Voyage | -5% accuracy |
| Vector DB | Chroma (embedded) | $0 (no Docker) | -20% speed for >100K docs |
| Reranker | **Jina v2** ✅ (already chosen) | $0 (self-hosted) | Same |
| LLM | Keep GPT-4o-mini | Same | Same |

**Total Savings**: ~15% cost reduction, acceptable quality trade-off for <50K documents

---

### 8.2 Maximum Quality Configuration

**Goal**: Best possible accuracy and performance

| Component | Change | Cost Impact | Quality Impact |
|-----------|--------|-------------|----------------|
| Embedding | Voyage-3-large | +$0.04/1M | +9.74% accuracy |
| Vector DB | Qdrant | Similar | Better filtering |
| Reranker | Cohere Rerank 3.5 | +$2/1K searches | +15% speed, +5-10% accuracy |
| LLM | GPT-4o | +$2.50/$10.00 per 1M | Significantly better reasoning |

**Total Cost**: ~5-6x current setup, significantly better results for critical applications

---

### 8.3 Fully Open-Source Configuration

**Goal**: No API dependencies, complete data privacy

| Component | Technology | Notes |
|-----------|-----------|-------|
| Embedding | sentence-transformers/all-MiniLM-L6-v2 | Local, 384-dim, fast |
| Vector DB | Qdrant or Chroma | Both support local deployment |
| Reranker | **Jina Reranker v2** ✅ (already chosen) | Already open-source |
| LLM | Llama 3.1 8B (via Ollama) | Local inference, requires GPU |
| OCR | **PaddleOCR** ✅ (already chosen) | Already open-source |

**Pros**:
- $0 API costs (only compute)
- Complete data privacy
- No rate limits
- Full control

**Cons**:
- Requires powerful GPU (12GB+ VRAM)
- Lower quality than commercial models
- Slower inference
- More maintenance overhead

**Cost**: $0 API fees, but requires hardware (~$500-2000 one-time GPU investment or $50-200/month cloud GPU)

---

### 8.4 Hybrid Configuration (Recommended for Most)

**Goal**: Balance cost, quality, and privacy

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Embedding | **Voyage-3.5-lite** ✅ | Best cost-performance (our choice) |
| Vector DB | **Milvus local** ✅ | Self-hosted, no API costs |
| Reranker | **Jina v2 local** ✅ | Open-source, self-hosted |
| LLM | GPT-4o-mini | Only component using external API |
| OCR | **PaddleOCR local** ✅ | Self-hosted |

**Benefits**:
- Only LLM and embedding use APIs (minimal cost)
- Retrieval and reranking are private (self-hosted)
- Easy to swap LLM for local model later
- Balanced cost (~$0.54 per 1000 queries)

**This is our chosen configuration** ✅

---

## 9. SOURCES

### Embedding Benchmarks
- [Text Embedding Models Compared: OpenAI, Voyage, Cohere & More](https://document360.com/blog/text-embedding-model-analysis/)
- [13 Best Embedding Models in 2025: OpenAI vs Voyage AI vs Ollama](https://elephas.app/blog/best-embedding-models)
- [voyage-3-large: the new state-of-the-art general-purpose embedding model](https://blog.voyageai.com/2025/01/07/voyage-3-large/)
- [voyage-3.5 and voyage-3.5-lite: improved quality for a new retrieval frontier](https://blog.voyageai.com/2025/05/20/voyage-3-5/)
- [We Benchmarked 20+ Embedding APIs with Milvus](https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md)
- [Benchmarking API latency of embedding providers](https://nixiesearch.substack.com/p/benchmarking-api-latency-of-embedding)

### Vector Database Benchmarks
- [Vector Database Comparison: Pinecone vs Weaviate vs Qdrant vs FAISS vs Milvus vs Chroma (2025)](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Vector Database Benchmarks - Qdrant](https://qdrant.tech/benchmarks/)
- [Best Vector Database For RAG In 2025](https://digitaloneagency.com.au/best-vector-database-for-rag-in-2025-pinecone-vs-weaviate-vs-qdrant-vs-milvus-vs-chroma/)
- [Top 5 Open Source Vector Databases for 2025](https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3)
- [An Honest Comparison of Open Source Vector Databases](https://www.kdnuggets.com/an-honest-comparison-of-open-source-vector-databases)

### OCR Benchmarks
- [OCR comparison: Tesseract versus EasyOCR vs PaddleOCR vs MMOCR](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66)
- [PaddleOCR vs Tesseract: Which is the best open source OCR?](https://www.koncile.ai/en/ressources/paddleocr-analyse-avantages-alternatives-open-source)
- [Comparison of Paddle OCR, EasyOCR, KerasOCR, and Tesseract OCR](https://www.plugger.ai/blog/comparison-of-paddle-ocr-easyocr-kerasocr-and-tesseract-ocr)
- [Thorough Comparison of 6 Free and Open Source OCR Tools 2025](https://www.cisdem.com/resource/open-source-ocr.html)

### Hybrid Search & Reranking
- [Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Ultimate Guide to Choosing the Best Reranking Model in 2025](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Building Contextual RAG Systems with Hybrid Search and Reranking](https://www.analyticsvidhya.com/blog/2024/12/contextual-rag-systems-with-hybrid-search-and-reranking/)
- [Top 7 Rerankers for RAG](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)
- [Maximizing Search Relevance and RAG Accuracy with Jina Reranker](https://jina.ai/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)
- [Reranker Leaderboard - Agentset](https://agentset.ai/rerankers)
- [Boosting RAG: Picking the Best Embedding & Reranker models](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)

### OpenAI Pricing
- [OpenAI API Pricing](https://openai.com/api/pricing/)
- [GPT-4o mini: advancing cost-efficient intelligence](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
- [AI API Pricing Comparison (2025): Grok, Gemini, ChatGPT & Claude](https://intuitionlabs.ai/articles/ai-api-pricing-comparison-grok-gemini-openai-claude)
- [OpenAI GPT-4o-Mini Pricing (Updated 2025)](https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini)

---

**End of Document**

_This architecture provides a production-ready foundation for building a high-quality RAG chatbot with optimal cost-performance trade-offs based on 2025 benchmarks and best practices._
