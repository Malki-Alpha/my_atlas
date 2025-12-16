"""Hybrid retriever combining semantic and keyword search with RRF."""

from typing import List, Dict, Any
from ..database.milvus_client import MilvusClient
from .bm25_search import BM25Search
from ..ingestion.embedder import Embedder
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval using semantic search + BM25 + RRF fusion."""

    def __init__(self):
        """Initialize hybrid retriever."""
        self.config = config.retrieval

        # Initialize components
        self.db_client = MilvusClient()
        self.embedder = Embedder()

        # Load all documents for BM25
        logger.info("Loading documents for BM25 index...")
        all_docs = self.db_client.get_all_chunks()

        if not all_docs:
            logger.warning("No documents found in database for BM25 index")
            self.bm25 = None
        else:
            self.bm25 = BM25Search(all_docs)

        logger.info("HybridRetriever initialized")

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
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

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return (default from config)

        Returns:
            List of search results after RRF fusion
        """
        if top_k is None:
            top_k = self.config.hybrid_top_k

        logger.info(f"Performing hybrid search for query: '{query[:50]}...'")

        # Step 1: Generate query embedding
        logger.debug("Generating query embedding...")
        query_embedding = self.embedder.embed_text(query, input_type="query")

        # Step 2: Semantic search
        logger.debug("Performing semantic search...")
        semantic_results = self.db_client.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Step 3: BM25 keyword search
        if self.bm25 is None:
            logger.warning("BM25 index not available, using semantic results only")
            return semantic_results[:top_k]

        logger.debug("Performing BM25 keyword search...")
        bm25_results = self.bm25.search(query, top_k=top_k)

        # Step 4: Reciprocal Rank Fusion
        logger.debug("Applying Reciprocal Rank Fusion...")
        fused_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            k=self.config.rrf_k,
            alpha=self.config.rrf_alpha
        )

        logger.info(
            f"Hybrid search complete: {len(fused_results)} results "
            f"(semantic: {len(semantic_results)}, bm25: {len(bm25_results)})"
        )

        return fused_results[:top_k]

    def refresh_bm25(self):
        """Refresh BM25 index with latest documents from database."""
        logger.info("Refreshing BM25 index...")
        all_docs = self.db_client.get_all_chunks()

        if not all_docs:
            logger.warning("No documents found in database")
            self.bm25 = None
        else:
            self.bm25 = BM25Search(all_docs)
            logger.info(f"BM25 index refreshed with {len(all_docs)} documents")
