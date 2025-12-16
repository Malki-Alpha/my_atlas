"""BM25 keyword search implementation."""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class BM25Search:
    """BM25 keyword-based search."""

    def __init__(self, documents: List[Dict[str, Any]]):
        """Initialize BM25 index with documents.

        Args:
            documents: List of document dicts with 'chunk_id', 'content', 'metadata'
        """
        self.documents = documents
        self.chunk_ids = [doc['chunk_id'] for doc in documents]

        logger.info(f"Building BM25 index for {len(documents)} documents...")

        # Tokenize all documents
        tokenized_docs = []
        for doc in documents:
            tokens = word_tokenize(doc['content'].lower())
            tokenized_docs.append(tokens)

        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=1.5,  # Term frequency saturation parameter
            b=0.75   # Length normalization parameter
        )

        logger.info("BM25 index created successfully")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using BM25 keyword matching.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of search results with scores
        """
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

        logger.debug(f"BM25 search returned {len(candidates)} results")
        return candidates

    def update(self, documents: List[Dict[str, Any]]):
        """Update BM25 index with new documents.

        Args:
            documents: New list of documents
        """
        self.__init__(documents)  # Rebuild index
