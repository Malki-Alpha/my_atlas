"""Configuration management for My Atlas RAG Chatbot."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv('VOYAGE_API_KEY', ''))
    model: str = Field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'voyage-3.5-lite'))
    batch_size: int = Field(default_factory=lambda: int(os.getenv('EMBEDDING_BATCH_SIZE', '128')))
    dimensions: int = Field(default_factory=lambda: int(os.getenv('EMBEDDING_DIMENSIONS', '1024')))


class MilvusConfig(BaseModel):
    """Milvus database configuration."""
    host: str = Field(default_factory=lambda: os.getenv('MILVUS_HOST', 'localhost'))
    port: int = Field(default_factory=lambda: int(os.getenv('MILVUS_PORT', '19530')))
    collection: str = Field(default_factory=lambda: os.getenv('MILVUS_COLLECTION', 'documents'))
    index_type: str = Field(default_factory=lambda: os.getenv('MILVUS_INDEX_TYPE', 'HNSW'))
    metric_type: str = Field(default_factory=lambda: os.getenv('MILVUS_METRIC_TYPE', 'COSINE'))

    # HNSW parameters
    hnsw_m: int = Field(default_factory=lambda: int(os.getenv('MILVUS_HNSW_M', '16')))
    hnsw_ef_construction: int = Field(default_factory=lambda: int(os.getenv('MILVUS_HNSW_EF_CONSTRUCTION', '200')))
    hnsw_ef_search: int = Field(default_factory=lambda: int(os.getenv('MILVUS_HNSW_EF_SEARCH', '100')))


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    chunk_size: int = Field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '512')))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '102')))
    method: str = Field(default_factory=lambda: os.getenv('CHUNKING_METHOD', 'recursive'))


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    hybrid_top_k: int = Field(default_factory=lambda: int(os.getenv('HYBRID_TOP_K', '20')))
    rerank_top_k: int = Field(default_factory=lambda: int(os.getenv('RERANK_TOP_K', '5')))
    rrf_k: int = Field(default_factory=lambda: int(os.getenv('RRF_K', '60')))
    rrf_alpha: float = Field(default_factory=lambda: float(os.getenv('RRF_ALPHA', '0.5')))


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    model: str = Field(default_factory=lambda: os.getenv('RERANKER_MODEL', 'jinaai/jina-reranker-v2-base-multilingual'))
    device: str = Field(default_factory=lambda: os.getenv('RERANKER_DEVICE', 'cuda'))
    max_length: int = Field(default_factory=lambda: int(os.getenv('RERANKER_MAX_LENGTH', '512')))


class LLMConfig(BaseModel):
    """LLM configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    model: str = Field(default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-4o-mini'))
    temperature: float = Field(default_factory=lambda: float(os.getenv('LLM_TEMPERATURE', '0.1')))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv('LLM_MAX_TOKENS', '1000')))
    top_p: float = Field(default_factory=lambda: float(os.getenv('LLM_TOP_P', '0.95')))

    # Fallback configuration
    fallback_temperature: float = Field(default_factory=lambda: float(os.getenv('FALLBACK_LLM_TEMPERATURE', '0.3')))
    fallback_max_tokens: int = Field(default_factory=lambda: int(os.getenv('FALLBACK_LLM_MAX_TOKENS', '500')))


class OCRConfig(BaseModel):
    """OCR configuration."""
    lang: str = Field(default_factory=lambda: os.getenv('PADDLEOCR_LANG', 'en'))
    use_gpu: bool = Field(default_factory=lambda: os.getenv('PADDLEOCR_USE_GPU', 'true').lower() == 'true')
    use_angle_cls: bool = Field(default_factory=lambda: os.getenv('PADDLEOCR_USE_ANGLE_CLS', 'true').lower() == 'true')
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.7')))
    batch_size: int = Field(default_factory=lambda: int(os.getenv('OCR_BATCH_SIZE', '6')))


class PathConfig(BaseModel):
    """Storage paths configuration."""
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv('DATA_DIR', './data')))
    raw_text_dir: Path = Field(default_factory=lambda: Path(os.getenv('EXT_RAW_TEXT_DIR', './ext/raw_text')))
    cleaned_chunks_dir: Path = Field(default_factory=lambda: Path(os.getenv('EXT_CLEANED_CHUNKS_DIR', './ext/cleaned_chunks')))
    emb_chunks_dir: Path = Field(default_factory=lambda: Path(os.getenv('EXT_EMB_CHUNKS_DIR', './ext/emb_chunks')))
    logs_dir: Path = Field(default_factory=lambda: Path(os.getenv('LOGS_DIR', './logs')))

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.raw_text_dir,
            self.cleaned_chunks_dir,
            self.emb_chunks_dir,
            self.logs_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    max_workers: int = Field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    embedding_cache: bool = Field(default_factory=lambda: os.getenv('EMBEDDING_CACHE', 'true').lower() == 'true')


class LogConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    format: str = Field(default_factory=lambda: os.getenv('LOG_FORMAT', 'json'))


class Config(BaseModel):
    """Main configuration class."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure all directories exist
        self.paths.ensure_dirs()

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are set.

        Returns:
            List of missing API keys
        """
        missing = []
        if not self.embedding.api_key:
            missing.append('VOYAGE_API_KEY')
        if not self.llm.api_key:
            missing.append('OPENAI_API_KEY')
        return missing


# Global config instance
config = Config()
