"""Milvus vector database client."""

from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MilvusClient:
    """Client for interacting with Milvus vector database."""

    def __init__(self):
        """Initialize Milvus client and connect to database."""
        self.config = config.milvus
        self.embedding_dim = config.embedding.dimensions
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=str(self.config.port)
            )
            logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self) -> Collection:
        """Create the documents collection with schema and index.

        Returns:
            Created Collection instance
        """
        collection_name = self.config.collection

        # Drop existing collection if exists
        if utility.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} already exists. Using existing collection.")
            self.collection = Collection(collection_name)
            return self.collection

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings for RAG"
        )

        # Create collection
        logger.info(f"Creating collection: {collection_name}")
        self.collection = Collection(
            name=collection_name,
            schema=schema
        )

        # Create index
        self._create_index()

        logger.info(f"Collection {collection_name} created successfully")
        return self.collection

    def _create_index(self):
        """Create index on embedding field."""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {
                "M": self.config.hnsw_m,
                "efConstruction": self.config.hnsw_ef_construction
            }
        }

        logger.info(f"Creating {self.config.index_type} index...")
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        logger.info("Index created successfully")

    def get_collection(self) -> Collection:
        """Get or create the documents collection.

        Returns:
            Collection instance
        """
        if self.collection is None:
            collection_name = self.config.collection
            if utility.has_collection(collection_name):
                self.collection = Collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            else:
                self.create_collection()

        return self.collection

    def insert(
        self,
        chunk_ids: List[str],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        timestamps: List[int]
    ) -> Dict[str, Any]:
        """Insert chunks into Milvus.

        Args:
            chunk_ids: List of chunk IDs
            contents: List of chunk contents
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            timestamps: List of timestamps (Unix time)

        Returns:
            Insert result with inserted IDs
        """
        collection = self.get_collection()

        entities = [
            chunk_ids,
            contents,
            embeddings,
            metadatas,
            timestamps
        ]

        logger.info(f"Inserting {len(chunk_ids)} chunks into Milvus...")
        insert_result = collection.insert(entities)
        collection.flush()

        logger.info(f"Inserted {len(insert_result.primary_keys)} chunks successfully")

        return {
            "inserted_count": len(insert_result.primary_keys),
            "primary_keys": insert_result.primary_keys
        }

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            output_fields: Fields to include in results

        Returns:
            List of search results with scores and metadata
        """
        collection = self.get_collection()
        collection.load()

        if output_fields is None:
            output_fields = ["chunk_id", "content", "metadata"]

        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"ef": self.config.hnsw_ef_search}
        }

        logger.debug(f"Searching for top {top_k} similar chunks...")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )

        # Format results
        candidates = []
        for hit in results[0]:
            candidates.append({
                "chunk_id": hit.entity.get("chunk_id"),
                "content": hit.entity.get("content"),
                "metadata": hit.entity.get("metadata"),
                "score": float(hit.score),
                "source": "semantic"
            })

        logger.debug(f"Found {len(candidates)} results")
        return candidates

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all chunks from the collection.

        Returns:
            List of all chunks with content and metadata
        """
        collection = self.get_collection()
        collection.load()

        # Query all entities
        expr = "id > 0"  # Match all
        results = collection.query(
            expr=expr,
            output_fields=["chunk_id", "content", "metadata"]
        )

        logger.info(f"Retrieved {len(results)} chunks from Milvus")
        return results

    def count(self) -> int:
        """Get the number of chunks in the collection.

        Returns:
            Number of chunks
        """
        collection = self.get_collection()
        return collection.num_entities

    def delete_all(self):
        """Delete all data from the collection."""
        collection_name = self.config.collection
        if utility.has_collection(collection_name):
            logger.warning(f"Dropping collection: {collection_name}")
            utility.drop_collection(collection_name)
            self.collection = None
            logger.info("Collection dropped successfully")
        else:
            logger.warning(f"Collection {collection_name} does not exist")

    def get_status(self) -> Dict[str, Any]:
        """Get database status and statistics.

        Returns:
            Status dictionary
        """
        try:
            version = utility.get_server_version()
            collection_name = self.config.collection

            status = {
                "connected": True,
                "server_version": version,
                "host": self.config.host,
                "port": self.config.port,
                "collection_exists": utility.has_collection(collection_name)
            }

            if status["collection_exists"]:
                collection = self.get_collection()
                status["total_chunks"] = collection.num_entities
                status["collection_name"] = collection_name

            return status

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {
                "connected": False,
                "error": str(e)
            }

    def close(self):
        """Close connection to Milvus."""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
