"""Embedding generation using Voyage AI."""

import json
from pathlib import Path
from typing import List, Dict, Any
import voyageai
import time
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Generate embeddings using Voyage AI."""

    def __init__(self):
        """Initialize Voyage AI client."""
        self.embedding_config = config.embedding

        if not self.embedding_config.api_key:
            raise ValueError("VOYAGE_API_KEY not set in environment")

        self.client = voyageai.Client(api_key=self.embedding_config.api_key)

        logger.info(
            f"Embedder initialized: model={self.embedding_config.model}, "
            f"batch_size={self.embedding_config.batch_size}"
        )

    def embed_text(
        self,
        text: str,
        input_type: str = "document"
    ) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            input_type: Type of input ('document' or 'query')

        Returns:
            Embedding vector
        """
        result = self.client.embed(
            texts=[text],
            model=self.embedding_config.model,
            input_type=input_type
        )

        return result.embeddings[0]

    def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document"
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            input_type: Type of input ('document' or 'query')

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        batch_size = self.embedding_config.batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                logger.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")

                result = self.client.embed(
                    texts=batch,
                    model=self.embedding_config.model,
                    input_type=input_type
                )

                all_embeddings.extend(result.embeddings)

                # Rate limiting: small delay between batches
                if i + batch_size < len(texts):
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                raise

        return all_embeddings

    def process_chunk_file(
        self,
        input_file: Path,
        output_dir: Path
    ) -> Path:
        """Process a chunk file and add embeddings.

        Args:
            input_file: Path to chunk JSON file
            output_dir: Directory to save embedded chunks

        Returns:
            Path to output file with embeddings
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load chunk
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        content = data["content"]
        metadata = data["metadata"]

        # Generate embedding
        logger.debug(f"Generating embedding for {input_file.name}")
        embedding = self.embed_text(content, input_type="document")

        # Add embedding metadata
        metadata["embedding_model"] = self.embedding_config.model
        metadata["embedding_dim"] = len(embedding)
        metadata["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Create embedded chunk data
        embedded_data = {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }

        # Save
        output_file = output_dir / input_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_data, f, ensure_ascii=False, indent=2)

        return output_file

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> List[Path]:
        """Process all chunk files in a directory and add embeddings.

        Args:
            input_dir: Directory containing chunk JSON files
            output_dir: Directory to save embedded chunks

        Returns:
            List of paths to embedded chunk files
        """
        input_files = list(input_dir.glob("*.json"))

        if not input_files:
            logger.warning(f"No chunk files found in {input_dir}")
            return []

        logger.info(f"Generating embeddings for {len(input_files)} chunks...")

        # Option 1: Batch embedding (more efficient)
        # Load all chunks
        chunks = []
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks.append({
                    "file": input_file,
                    "content": data["content"],
                    "metadata": data["metadata"]
                })

        # Extract texts
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings in batches
        logger.info("Generating embeddings in batches...")
        embeddings = self.embed_batch(texts, input_type="document")

        # Save embedded chunks
        output_files = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk["metadata"]
            metadata["embedding_model"] = self.embedding_config.model
            metadata["embedding_dim"] = len(embedding)
            metadata["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            embedded_data = {
                "content": chunk["content"],
                "metadata": metadata,
                "embedding": embedding
            }

            output_file = output_dir / chunk["file"].name
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_data, f, ensure_ascii=False, indent=2)

            output_files.append(output_file)

        logger.info(f"Generated embeddings for {len(output_files)} chunks")

        return output_files
