"""PDF text extractor using PyMuPDF."""

from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from .base_extractor import BaseExtractor
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract text from regular (text-based) PDF files."""

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is a PDF.

        Args:
            file_path: Path to the document

        Returns:
            True if file is a PDF
        """
        return file_path.suffix.lower() == '.pdf'

    def _is_scanned_pdf(self, doc: fitz.Document) -> bool:
        """Detect if PDF is scanned (image-based).

        Args:
            doc: PyMuPDF document

        Returns:
            True if PDF appears to be scanned
        """
        # Check first few pages
        pages_to_check = min(3, len(doc))
        text_chars = 0
        total_pages = 0

        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()
            text_chars += len(text.strip())
            total_pages += 1

        # If average text per page is very low, likely scanned
        avg_chars = text_chars / total_pages if total_pages > 0 else 0
        is_scanned = avg_chars < 50  # Threshold for scanned detection

        if is_scanned:
            logger.warning(f"PDF appears to be scanned (avg {avg_chars:.0f} chars/page). Consider using OCR.")

        return is_scanned

    def extract(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Extract text from PDF and save each page as JSON.

        Args:
            file_path: Path to PDF file
            output_dir: Directory to save extracted text

        Returns:
            List of paths to generated JSON files
        """
        logger.info(f"Extracting text from PDF: {file_path.name}")

        doc_name = file_path.stem
        base_metadata = self._get_file_metadata(file_path)

        output_files = []

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            base_metadata["total_pages"] = total_pages

            # Check if scanned
            is_scanned = self._is_scanned_pdf(doc)

            logger.info(f"Processing {total_pages} pages...")

            for page_num in range(total_pages):
                page = doc[page_num]

                # Extract text
                text = page.get_text()

                # Page-specific metadata
                page_metadata = base_metadata.copy()
                page_metadata["page_number"] = page_num + 1
                page_metadata["is_scanned"] = is_scanned

                # Save page
                output_file = self._save_page(
                    content=text,
                    metadata=page_metadata,
                    output_dir=output_dir,
                    doc_name=doc_name,
                    page_num=page_num + 1
                )

                output_files.append(output_file)

            doc.close()

            logger.info(f"Extracted {len(output_files)} pages from {file_path.name}")

            if is_scanned:
                logger.warning(
                    f"PDF {file_path.name} may need OCR processing for better quality. "
                    "Use the OCR extractor instead."
                )

        except Exception as e:
            logger.error(f"Error extracting PDF {file_path.name}: {e}")
            raise

        return output_files
