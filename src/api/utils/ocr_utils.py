"""
OCR execution and fuzzy text-matching utilities for verifying document
content (e.g. affidavit declarations, applicant name/ID/service matches).
"""
from difflib import SequenceMatcher
import re
from difflib import SequenceMatcher
from typing import Dict, List

import numpy as np
import pytesseract
from PIL import Image

from src.api.utils.image_utils import cv2_to_pil


def run_ocr(image: np.ndarray) -> str:
    """Run Tesseract OCR on an image array (BGR color or single-channel)."""
    pil_image = cv2_to_pil(image) if image.ndim == 3 else Image.fromarray(image)
    return pytesseract.image_to_string(pil_image)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation noise, and collapse whitespace for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_contains(haystack: str, phrase: str, threshold: float = 0.78) -> bool:
    """
    Check whether `phrase` appears in `haystack`, tolerating OCR noise.

    Tries exact substring first (cheap, common case), then falls back to
    a sliding-window similarity check via difflib so minor OCR errors
    (e.g. "1dentity" vs "identity") don't cause a false negative.
    """
    haystack = normalize_text(haystack)
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    if phrase in haystack:
        return True

    phrase_len = len(phrase)
    if len(haystack) < phrase_len:
        return False

    window = phrase_len + max(6, phrase_len // 4)
    step = max(1, phrase_len // 4)

    for start in range(0, len(haystack) - phrase_len + 1, step):
        snippet = haystack[start:start + window]
        ratio = SequenceMatcher(None, snippet, phrase).ratio()
        if ratio >= threshold:
            return True

    return False


def match_declarations(
    text: str,
    required_declarations: Dict[str, List[str]],
    min_keywords_matched: int = 2,
) -> Dict[str, dict]:
    """
    Check OCR'd affidavit text against the required declaration categories.

    A category counts as "matched" if at least `min_keywords_matched` of
    its keyword phrases are found (fuzzy) — requiring every keyword is
    brittle against OCR noise and minor template wording differences.
    """
    results = {}

    for category, phrases in required_declarations.items():
        found = [p for p in phrases if fuzzy_contains(text, p)]
        missing = [p for p in phrases if p not in found]

        results[category] = {
            "found": found,
            "missing": missing,
            "matched": len(found) >= min(min_keywords_matched, len(phrases)),
        }

    return results


def find_identifier(text: str, value: str) -> bool:
    """
    Loose match for a name/ID/service string within OCR'd text. Checked
    more strictly than declaration phrases since a false positive here
    has real identity-verification consequences.
    """
    if not value:
        return False

    normalized_text = normalize_text(text)
    normalized_value = normalize_text(value)

    if not normalized_value:
        return False

    if normalized_value in normalized_text:
        return True

    # For multi-word values (e.g. full names), allow one missing token —
    # handles OCR dropping a middle name or misreading a single character.
    tokens = normalized_value.split()
    if len(tokens) > 1:
        matched_tokens = sum(1 for t in tokens if t in normalized_text)
        return matched_tokens >= max(1, len(tokens) - 1)

    return False




def fuzzy_name_match(raw_text: str, full_name: str, threshold: float = 0.82) -> bool:
    """Similarity-based name match for lower-quality scans. Compares
    against sliding windows of raw_text roughly the length of full_name,
    rather than requiring an exact substring."""
    normalized_name = " ".join(full_name.upper().split())
    normalized_text = " ".join(raw_text.upper().split())

    words = normalized_text.split()
    name_word_count = len(normalized_name.split())

    best_ratio = 0.0
    for i in range(len(words) - name_word_count + 1):
        window = " ".join(words[i:i + name_word_count])
        ratio = SequenceMatcher(None, normalized_name, window).ratio()
        best_ratio = max(best_ratio, ratio)

    return best_ratio >= threshold

_SAPS_STAMP_KEYWORDS = [
    "south african police service",
    "suid-afrikaanse polisiediens",
    "saps",
]


def detect_saps_stamp_text(raw_text: str) -> dict:
    """Check OCR'd text for wording that appears on a SAPS commissioner-
    of-oaths stamp, in English and/or Afrikaans. This is a text-content
    signal, complementary to AffidavitVision.detect_stamp(), which only
    detects a circular stamp *shape* without reading what it says — a
    circle with no SAPS wording shouldn't count as a valid stamp."""
    normalized = " ".join(raw_text.lower().split())
    matched = [kw for kw in _SAPS_STAMP_KEYWORDS if kw in normalized]
    return {
        "detected": len(matched) > 0,
        "matched_keywords": matched,
    }