"""Text chunking module for splitting documents into chunks."""

import json
from pathlib import Path
from typing import List, Dict, Any
import tiktoken
from langchain.text_splitters import RecursiveCharacterTextSplitter
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Split text into chunks for embedding and retrieval."""

    def __init__(self):
        """Initialize text chunker with configuration."""
        self.chunk_config = config.chunking

        # Initialize tiktoken for token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            keep_separator=True
        )

        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_config.chunk_size}, "
            f"overlap={self.chunk_config.chunk_overlap}"
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        tokens = self.encoding.encode(text, disallowed_special=())
        return len(tokens)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        return chunks

    def process_extracted_file(
        self,
        input_file: Path,
        output_dir: Path
    ) -> List[Path]:
        """Process an extracted text file and create chunks.

        Args:
            input_file: Path to extracted text JSON file
            output_dir: Directory to save chunks

        Returns:
            List of paths to generated chunk JSON files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load extracted text
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        content = data["content"]
        metadata = data["metadata"]

        # Chunk the text
        chunks = self.chunk_text(content)

        if not chunks:
            logger.warning(f"No chunks generated from {input_file.name}")
            return []

        # Save chunks
        output_files = []
        doc_name = metadata.get("document_name", input_file.stem).replace("_page_", "_")
        base_name = doc_name.split("_page_")[0] if "_page_" in doc_name else doc_name
        page_num = metadata.get("page_number", 0)

        for idx, chunk_text in enumerate(chunks):
            # Create chunk ID
            chunk_id = f"{base_name}_chunk_{page_num:03d}_{idx:04d}"

            # Chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks_in_page"] = len(chunks)
            chunk_metadata["token_count"] = self._count_tokens(chunk_text)

            # Create chunk data
            chunk_data = {
                "content": chunk_text,
                "metadata": chunk_metadata
            }

            # Save chunk
            output_file = output_dir / f"{chunk_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

            output_files.append(output_file)

        logger.debug(f"Created {len(output_files)} chunks from {input_file.name}")

        return output_files

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> List[Path]:
        """Process all extracted text files in a directory.

        Args:
            input_dir: Directory containing extracted text JSON files
            output_dir: Directory to save chunks

        Returns:
            List of paths to all generated chunk files
        """
        input_files = list(input_dir.glob("*.json"))

        if not input_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return []

        logger.info(f"Processing {len(input_files)} extracted files...")

        all_output_files = []

        for input_file in input_files:
            try:
                output_files = self.process_extracted_file(input_file, output_dir)
                all_output_files.extend(output_files)
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {e}")
                continue

        logger.info(f"Generated {len(all_output_files)} chunks total")

        return all_output_files
