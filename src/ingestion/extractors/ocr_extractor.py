"""OCR extractor using PaddleOCR for scanned documents."""

from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF for PDF to image conversion
from paddleocr import PaddleOCR
from .base_extractor import BaseExtractor
from ...utils.config import config
from ...utils.logger import get_logger

logger = get_logger(__name__)


class OCRExtractor(BaseExtractor):
    """Extract text from scanned PDFs using PaddleOCR."""

    def __init__(self):
        """Initialize PaddleOCR."""
        self.ocr_config = config.ocr

        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=self.ocr_config.use_angle_cls,
            lang=self.ocr_config.lang,
            use_gpu=self.ocr_config.use_gpu,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            rec_batch_num=self.ocr_config.batch_size
        )
        logger.info("PaddleOCR initialized successfully")

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is a PDF (for OCR processing).

        Args:
            file_path: Path to the document

        Returns:
            True if file is a PDF
        """
        return file_path.suffix.lower() == '.pdf'

    def _pdf_page_to_image(self, page: fitz.Page) -> bytes:
        """Convert PDF page to image bytes.

        Args:
            page: PyMuPDF page object

        Returns:
            Image bytes (PNG format)
        """
        # Render page to pixmap (image)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_bytes = pix.tobytes("png")
        return img_bytes

    def _extract_text_from_ocr_result(self, ocr_result: List) -> str:
        """Extract and format text from PaddleOCR result.

        Args:
            ocr_result: PaddleOCR detection result

        Returns:
            Extracted text
        """
        if not ocr_result or not ocr_result[0]:
            return ""

        lines = []
        for line in ocr_result[0]:
            text = line[1][0]  # Get text content
            confidence = line[1][1]  # Get confidence score

            # Filter by confidence threshold
            if confidence >= self.ocr_config.confidence_threshold:
                lines.append(text)

        return "\n".join(lines)

    def extract(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Extract text from scanned PDF using OCR.

        Args:
            file_path: Path to scanned PDF file
            output_dir: Directory to save extracted text

        Returns:
            List of paths to generated JSON files
        """
        logger.info(f"Extracting text from scanned PDF using OCR: {file_path.name}")

        doc_name = file_path.stem
        base_metadata = self._get_file_metadata(file_path)
        base_metadata["extraction_method"] = "OCR"
        base_metadata["ocr_model"] = "PaddleOCR"
        base_metadata["ocr_language"] = self.ocr_config.lang

        output_files = []

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            base_metadata["total_pages"] = total_pages

            logger.info(f"Processing {total_pages} pages with OCR...")

            for page_num in range(total_pages):
                page = doc[page_num]

                # Convert page to image
                img_bytes = self._pdf_page_to_image(page)

                # Perform OCR
                logger.debug(f"Running OCR on page {page_num + 1}/{total_pages}")
                ocr_result = self.ocr.ocr(img_bytes, cls=True)

                # Extract text from result
                text = self._extract_text_from_ocr_result(ocr_result)

                # Page-specific metadata
                page_metadata = base_metadata.copy()
                page_metadata["page_number"] = page_num + 1

                # Save page
                output_file = self._save_page(
                    content=text,
                    metadata=page_metadata,
                    output_dir=output_dir,
                    doc_name=doc_name,
                    page_num=page_num + 1
                )

                output_files.append(output_file)

                logger.debug(f"Page {page_num + 1} processed, extracted {len(text)} characters")

            doc.close()

            logger.info(f"OCR extraction complete: {len(output_files)} pages from {file_path.name}")

        except Exception as e:
            logger.error(f"Error during OCR extraction of {file_path.name}: {e}")
            raise

        return output_files
