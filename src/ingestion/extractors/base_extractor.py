"""Base extractor class for document processing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""

    @abstractmethod
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file.

        Args:
            file_path: Path to the document

        Returns:
            True if this extractor can process the file
        """
        pass

    @abstractmethod
    def extract(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Extract text from document and save to output directory.

        Args:
            file_path: Path to input document
            output_dir: Directory to save extracted text

        Returns:
            List of paths to generated JSON files
        """
        pass

    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Metadata dictionary
        """
        stat = file_path.stat()
        return {
            "source_path": str(file_path.absolute()),
            "document_type": file_path.suffix[1:],  # Remove the dot
            "document_name": file_path.name,
            "version": "1.0",
            "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
        }

    def _save_page(
        self,
        content: str,
        metadata: Dict[str, Any],
        output_dir: Path,
        doc_name: str,
        page_num: int
    ) -> Path:
        """Save extracted page content to JSON file.

        Args:
            content: Extracted text content
            metadata: Page metadata
            output_dir: Output directory
            doc_name: Document name (without extension)
            page_num: Page number

        Returns:
            Path to saved JSON file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{doc_name}_page_{page_num:03d}.json"

        data = {
            "content": content,
            "metadata": metadata
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_file
