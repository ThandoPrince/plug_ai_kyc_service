"""
Generic document loading, preprocessing, and OCR extraction.

This module is deliberately unaware of *what kind* of document it's
looking at (affidavit vs ID vs anything else) — that judgment belongs to
callers like KYCProcessor. DocumentProcessor's job is just: given a file
path, reliably produce page images and extracted text.
"""

from typing import Dict, List

import numpy as np

from src.api.utils.image_utils import (
    get_document_extension,
    load_image_pages,
    preprocess_for_ocr,
)
from src.api.utils.ocr_utils import run_ocr


class DocumentProcessor:
    def process(self, document_path: str, pdf_dpi: int = 300, preprocess: bool = True) -> Dict:
        """
        Load a document (PDF/JPG/PNG/WEBP), OCR every page, and return a
        unified result. Never raises — failures are reported in the
        returned dict so callers don't need a try/except at every call site.
        """
        result = {
            "success": False,
            "document_type": None,
            "page_count": 0,
            "pages": [],
            "raw_text": "",
            "extracted_lines": [],
            "error": None,
        }

        try:
            extension = get_document_extension(document_path)
            page_images = load_image_pages(document_path, pdf_dpi=pdf_dpi)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            result["error"] = str(e)
            return result

        result["document_type"] = "PDF" if extension == ".pdf" else "IMAGE"
        result["page_count"] = len(page_images)

        combined_text_parts: List[str] = []

        for index, image in enumerate(page_images):
            try:
                ocr_input = preprocess_for_ocr(image) if preprocess else image
                page_text = run_ocr(ocr_input)
            except Exception as e:
                page_text = ""
                result.setdefault("page_errors", []).append(
                    {"page": index, "error": str(e)}
                )

            combined_text_parts.append(page_text)

            result["pages"].append(
                {
                    "index": index,
                    "image": image,  # in-memory array; caller decides whether to persist it
                    "raw_text": page_text,
                    "extracted_lines": [
                        line.strip() for line in page_text.splitlines() if line.strip()
                    ],
                }
            )

        raw_text = "\n".join(combined_text_parts)
        result["raw_text"] = raw_text
        result["extracted_lines"] = [
            line.strip() for line in raw_text.splitlines() if line.strip()
        ]

        if raw_text.strip():
            result["success"] = True
        else:
            result["error"] = "No text could be extracted from the document."

        return result

    def get_primary_page_image(self, processed_result: Dict) -> np.ndarray:
        """
        Convenience accessor for CV checks that only care about one
        representative page (e.g. affidavit vision checks run on page 1).
        """
        pages = processed_result.get("pages") or []
        if not pages:
            raise ValueError("No pages available in processed document result.")
        return pages[0]["image"]