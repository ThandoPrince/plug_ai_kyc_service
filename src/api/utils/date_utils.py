"""
Date extraction and recency validation for affidavit documents.
"""
import re
from datetime import date, timedelta
from typing import List, Optional

MAX_AFFIDAVIT_AGE_DAYS = 90  # ~3 months

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_TEXT_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b")


def _normalize_year(year: int) -> int:
    if year < 100:
        current_yy = date.today().year % 100
        return 2000 + year if year <= current_yy else 1900 + year
    return year


def extract_dates(raw_text: str) -> List[date]:
    """Find every plausible calendar date in the OCR'd text. Affidavits
    typically contain more than one date (deponent signing date,
    commissioner's stamp date) — callers decide which to use."""
    found = []

    for day_s, month_s, year_s in _NUMERIC_DATE_RE.findall(raw_text):
        try:
            day, month, year = int(day_s), int(month_s), _normalize_year(int(year_s))
            found.append(date(year, month, day))
        except ValueError:
            continue

    for day_s, month_name, year_s in _TEXT_DATE_RE.findall(raw_text):
        month = _MONTH_NAMES.get(month_name.lower())
        if month is None:
            continue
        try:
            found.append(date(int(year_s), month, int(day_s)))
        except ValueError:
            continue

    return found


def most_likely_document_date(raw_text: str, upload_date: Optional[date] = None) -> Optional[date]:
    """Pick the most recent extracted date that isn't after upload_date
    (with a 1-day buffer for timezone edge cases). A date after the
    upload date is almost always an OCR misread rather than the real
    signing/stamp date, so those candidates are discarded."""
    upload_date = upload_date or date.today()
    candidates = [d for d in extract_dates(raw_text) if d <= upload_date + timedelta(days=1)]
    return max(candidates) if candidates else None


def is_date_recent(
    document_date: date,
    upload_date: Optional[date] = None,
    max_age_days: int = MAX_AFFIDAVIT_AGE_DAYS,
) -> bool:
    upload_date = upload_date or date.today()
    age = (upload_date - document_date).days
    return 0 <= age <= max_age_days