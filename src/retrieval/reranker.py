"""Reranking using Jina Reranker v2."""

from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class JinaReranker:
    """Rerank search results using Jina Reranker v2 cross-encoder model."""

    def __init__(self):
        """Initialize Jina Reranker v2 model."""
        self.config = config.reranker

        logger.info(f"Loading Jina Reranker model: {self.config.model}")

        # Determine device
        self.device = self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Load model and tokenizer
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True
            )

            self.model.eval()

            logger.info(f"Jina Reranker loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Jina Reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: User query
            candidates: List of candidate chunks from retrieval
            top_k: Number of top results to return (default from config)

        Returns:
            Top K reranked results with relevance scores
        """
        if top_k is None:
            top_k = config.retrieval.rerank_top_k

        if not candidates:
            return []

        logger.debug(f"Reranking {len(candidates)} candidates...")

        # Prepare input pairs (query, document)
        pairs = [[query, candidate['content']] for candidate in candidates]

        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Get relevance scores
            outputs = self.model(**inputs)

            # Extract logits and convert to scores
            if hasattr(outputs, 'logits'):
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            else:
                # Some models might return scores directly
                scores = outputs[0].squeeze(-1).cpu().numpy()

        # Attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])

        # Sort by rerank score and return top K
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        logger.debug(f"Reranking complete, returning top {top_k} results")

        return candidates[:top_k]
