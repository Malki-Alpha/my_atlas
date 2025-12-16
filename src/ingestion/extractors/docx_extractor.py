"""DOCX document extractor."""

from pathlib import Path
from typing import List
from docx import Document
from .base_extractor import BaseExtractor
from ...utils.logger import get_logger

logger = get_logger(__name__)


class DOCXExtractor(BaseExtractor):
    """Extract text from DOCX files."""

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is a DOCX.

        Args:
            file_path: Path to the document

        Returns:
            True if file is a DOCX
        """
        return file_path.suffix.lower() == '.docx'

    def extract(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Extract text from DOCX and save each page as JSON.

        Note: DOCX doesn't have pages like PDF, so we simulate pages
        by grouping paragraphs (approximately 500 words per page).

        Args:
            file_path: Path to DOCX file
            output_dir: Directory to save extracted text

        Returns:
            List of paths to generated JSON files
        """
        logger.info(f"Extracting text from DOCX: {file_path.name}")

        doc_name = file_path.stem
        base_metadata = self._get_file_metadata(file_path)

        output_files = []

        try:
            doc = Document(file_path)

            # Extract all paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Skip empty paragraphs
                    paragraphs.append(text)

            # Group paragraphs into "pages" (approx 500 words each)
            words_per_page = 500
            current_page_text = []
            current_word_count = 0
            page_num = 1

            for para in paragraphs:
                para_words = len(para.split())
                current_page_text.append(para)
                current_word_count += para_words

                # If we've accumulated enough words, save as a page
                if current_word_count >= words_per_page:
                    content = "\n\n".join(current_page_text)

                    # Page-specific metadata
                    page_metadata = base_metadata.copy()
                    page_metadata["page_number"] = page_num
                    page_metadata["total_pages"] = "unknown"  # Will update at end

                    # Save page
                    output_file = self._save_page(
                        content=content,
                        metadata=page_metadata,
                        output_dir=output_dir,
                        doc_name=doc_name,
                        page_num=page_num
                    )

                    output_files.append(output_file)

                    # Reset for next page
                    current_page_text = []
                    current_word_count = 0
                    page_num += 1

            # Save any remaining content as final page
            if current_page_text:
                content = "\n\n".join(current_page_text)

                page_metadata = base_metadata.copy()
                page_metadata["page_number"] = page_num
                page_metadata["total_pages"] = page_num

                output_file = self._save_page(
                    content=content,
                    metadata=page_metadata,
                    output_dir=output_dir,
                    doc_name=doc_name,
                    page_num=page_num
                )

                output_files.append(output_file)

            logger.info(f"Extracted {len(output_files)} pages from {file_path.name}")

        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path.name}: {e}")
            raise

        return output_files
