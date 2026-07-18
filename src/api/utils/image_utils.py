"""
Image loading and preprocessing utilities shared across document
processing and computer-vision checks.
"""

import os
from typing import List

import cv2
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - only hit if PyMuPDF isn't installed
    fitz = None

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


def get_document_extension(document_path: str) -> str:
    return os.path.splitext(document_path)[1].lower()


def is_pdf(document_path: str) -> bool:
    return get_document_extension(document_path) == ".pdf"


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image (any mode) to an OpenCV BGR array."""
    rgb_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def load_image_pages(document_path: str, pdf_dpi: int = 300) -> List[np.ndarray]:
    """
    Load a document from disk and return a list of page images as OpenCV
    (BGR) arrays. Single-page images return a list with one element.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the extension is unsupported or the file can't be read.
        RuntimeError: if a PDF is given but PyMuPDF isn't installed.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document file not found: {document_path}")

    extension = get_document_extension(document_path)

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type '{extension}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        return _load_pdf_pages(document_path, dpi=pdf_dpi)

    return [_load_single_image(document_path)]


def _load_single_image(document_path: str) -> np.ndarray:
    # Route through PIL first so WEBP (whose cv2 support varies by build)
    # is handled consistently, then convert to the BGR array the rest of
    # the pipeline (OpenCV) expects.
    try:
        pil_image = Image.open(document_path)
    except Exception as e:
        raise ValueError(f"Could not open image: {e}") from e

    return pil_to_cv2(pil_image)


def _load_pdf_pages(document_path: str, dpi: int = 300) -> List[np.ndarray]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to process PDF documents but is not installed."
        )

    pages: List[np.ndarray] = []

    pdf = fitz.open(document_path)
    try:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pil_to_cv2(pil_image))
    finally:
        pdf.close()

    if not pages:
        raise ValueError("PDF contains no pages.")

    return pages


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Light preprocessing to improve OCR accuracy: grayscale, denoise, and
    adaptive thresholding. Returns a single-channel image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )
    return thresh


def resize_max_dimension(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """Downscale an image if either side exceeds max_dimension, preserving aspect ratio."""
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image

    scale = max_dimension / longest_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)