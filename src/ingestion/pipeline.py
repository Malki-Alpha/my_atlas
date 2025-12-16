"""Ingestion pipeline orchestrator."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DOCXExtractor
from .extractors.ocr_extractor import OCRExtractor
from .chunker import TextChunker
from .embedder import Embedder
from ..database.milvus_client import MilvusClient
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrate the document ingestion pipeline."""

    def __init__(self, use_ocr: bool = False):
        """Initialize the ingestion pipeline.

        Args:
            use_ocr: Whether to use OCR for PDF extraction
        """
        self.use_ocr = use_ocr

        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DOCXExtractor()
        self.ocr_extractor = OCRExtractor() if use_ocr else None
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.db_client = MilvusClient()

        # Paths
        self.paths = config.paths

        logger.info(f"IngestionPipeline initialized (OCR: {use_ocr})")

    def get_extractor(self, file_path: Path):
        """Get appropriate extractor for a file.

        Args:
            file_path: Path to the document

        Returns:
            Appropriate extractor instance
        """
        if self.use_ocr and file_path.suffix.lower() == '.pdf':
            return self.ocr_extractor

        if self.pdf_extractor.can_extract(file_path):
            return self.pdf_extractor
        elif self.docx_extractor.can_extract(file_path):
            return self.docx_extractor
        else:
            raise ValueError(f"No extractor available for file type: {file_path.suffix}")

    def ingest_document(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a single document through the complete pipeline.

        Args:
            file_path: Path to the document to ingest

        Returns:
            Ingestion results dictionary
        """
        start_time = time.time()
        logger.info(f"Starting ingestion of: {file_path.name}")

        results = {
            "document": file_path.name,
            "status": "success",
            "pages_extracted": 0,
            "chunks_created": 0,
            "chunks_embedded": 0,
            "chunks_inserted": 0,
            "processing_time": 0,
            "error": None
        }

        try:
            # Step 1: Extract text
            logger.info(f"[1/4] Extracting text from {file_path.name}...")
            extractor = self.get_extractor(file_path)
            extracted_files = extractor.extract(file_path, self.paths.raw_text_dir)
            results["pages_extracted"] = len(extracted_files)
            logger.info(f"Extracted {len(extracted_files)} pages")

            # Step 2: Chunk text
            logger.info(f"[2/4] Chunking text...")
            chunked_files = []
            for extracted_file in extracted_files:
                chunk_files = self.chunker.process_extracted_file(
                    extracted_file,
                    self.paths.cleaned_chunks_dir
                )
                chunked_files.extend(chunk_files)

            results["chunks_created"] = len(chunked_files)
            logger.info(f"Created {len(chunked_files)} chunks")

            # Step 3: Generate embeddings
            logger.info(f"[3/4] Generating embeddings...")
            embedded_files = []
            for chunk_file in chunked_files:
                embedded_file = self.embedder.process_chunk_file(
                    chunk_file,
                    self.paths.emb_chunks_dir
                )
                embedded_files.append(embedded_file)

            results["chunks_embedded"] = len(embedded_files)
            logger.info(f"Generated embeddings for {len(embedded_files)} chunks")

            # Step 4: Insert into Milvus
            logger.info(f"[4/4] Inserting into vector database...")
            chunk_ids = []
            contents = []
            embeddings = []
            metadatas = []
            timestamps = []

            for embedded_file in embedded_files:
                with open(embedded_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                chunk_ids.append(data["metadata"]["chunk_id"])
                contents.append(data["content"])
                embeddings.append(data["embedding"])
                metadatas.append(data["metadata"])
                timestamps.append(int(time.time()))

            insert_result = self.db_client.insert(
                chunk_ids=chunk_ids,
                contents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
                timestamps=timestamps
            )

            results["chunks_inserted"] = insert_result["inserted_count"]
            logger.info(f"Inserted {insert_result['inserted_count']} chunks into database")

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"Error during ingestion of {file_path.name}: {e}")

        results["processing_time"] = time.time() - start_time
        logger.info(
            f"Ingestion completed in {results['processing_time']:.2f}s: "
            f"{results['chunks_inserted']} chunks inserted"
        )

        return results

    def ingest_directory(
        self,
        directory: Path,
        doc_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """Ingest all documents in a directory.

        Args:
            directory: Directory containing documents
            doc_type: Type of documents to process ('pdf', 'docx', or 'all')

        Returns:
            List of ingestion results for each document
        """
        # Get files based on type
        if doc_type == "pdf":
            patterns = ["*.pdf"]
        elif doc_type == "docx":
            patterns = ["*.docx"]
        else:  # all
            patterns = ["*.pdf", "*.docx"]

        files = []
        for pattern in patterns:
            files.extend(list(directory.glob(pattern)))

        if not files:
            logger.warning(f"No documents found in {directory}")
            return []

        logger.info(f"Found {len(files)} documents to ingest")

        results = []
        for file_path in files:
            result = self.ingest_document(file_path)
            results.append(result)

        # Summary
        total_chunks = sum(r["chunks_inserted"] for r in results)
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        logger.info(
            f"\nIngestion summary:\n"
            f"  Total documents: {len(results)}\n"
            f"  Successful: {successful}\n"
            f"  Failed: {failed}\n"
            f"  Total chunks inserted: {total_chunks}"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion pipeline statistics.

        Returns:
            Statistics dictionary
        """
        raw_files = list(self.paths.raw_text_dir.glob("*.json"))
        chunk_files = list(self.paths.cleaned_chunks_dir.glob("*.json"))
        embedded_files = list(self.paths.emb_chunks_dir.glob("*.json"))
        db_count = self.db_client.count()

        return {
            "raw_text_files": len(raw_files),
            "chunk_files": len(chunk_files),
            "embedded_files": len(embedded_files),
            "database_chunks": db_count
        }
