"""
South African ID number validation and parsing. Applies equally to a
13-digit number sourced from a Smart ID Card or the green barcoded ID
book — the number format is identical, only the physical document differs.
"""
import re
from datetime import date
from typing import Dict, Optional

SA_ID_PATTERN = re.compile(r"\b(\d{13})\b")


def extract_id_number(raw_text: str) -> Optional[str]:
    """Return the first 13-digit sequence in the OCR text that also
    passes the Luhn checksum. Guards against OCR noise (dates, phone
    numbers, etc.) being mistaken for the ID number."""
    for match in SA_ID_PATTERN.finditer(raw_text.replace(" ", "")):
        candidate = match.group(1)
        if luhn_checksum_valid(candidate):
            return candidate
    return None


def luhn_checksum_valid(id_number: str) -> bool:
    if not id_number.isdigit() or len(id_number) != 13:
        return False

    digits = [int(d) for d in id_number]
    check_digit = digits[-1]
    body = digits[:-1]

    total = 0
    for i, digit in enumerate(reversed(body)):
        if i % 2 == 0:
            total += digit
        else:
            doubled = digit * 2
            total += doubled // 10 + doubled % 10

    computed_check = (10 - (total % 10)) % 10
    return computed_check == check_digit


def parse_sa_id_number(id_number: str) -> Dict:
    """Derive DOB, gender, and citizenship. Does NOT itself validate the
    checksum — call luhn_checksum_valid first, or use extract_id_number,
    which already filters on it."""
    if not id_number.isdigit() or len(id_number) != 13:
        raise ValueError("SA ID number must be exactly 13 digits")

    yy, mm, dd = int(id_number[0:2]), int(id_number[2:4]), int(id_number[4:6])
    gender_seq = int(id_number[6:10])
    citizenship_digit = id_number[10]

    # 13-digit numbers predate 1980-ish; roll yy into 1900s if it's
    # later than the current two-digit year, else 2000s.
    current_yy = date.today().year % 100
    century = 2000 if yy <= current_yy else 1900

    try:
        dob = date(century + yy, mm, dd)
    except ValueError:
        raise ValueError("ID number contains an invalid date of birth")

    return {
        "date_of_birth": dob.isoformat(),
        "gender": "female" if gender_seq < 5000 else "male",
        "citizenship": "citizen" if citizenship_digit == "0" else "permanent_resident",
    }


_OCR_DIGIT_CONFUSIONS = {
    "O": "0", "o": "0", "D": "0",
    "I": "1", "l": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}


def extract_id_number(raw_text: str, lenient: bool = False) -> Optional[str]:
    """Return the first 13-digit sequence that passes the Luhn check.
    If lenient=True and no clean match is found, retry after correcting
    common OCR digit confusions (O/0, I/1, S/5 etc.) — useful for green
    ID books, where typewriter-style print causes more misreads than
    the Smart ID Card's machine-readable zone."""
    cleaned = raw_text.replace(" ", "")

    for match in SA_ID_PATTERN.finditer(cleaned):
        candidate = match.group(1)
        if luhn_checksum_valid(candidate):
            return candidate

    if not lenient:
        return None

    corrected = "".join(_OCR_DIGIT_CONFUSIONS.get(ch, ch) for ch in cleaned)
    for match in SA_ID_PATTERN.finditer(corrected):
        candidate = match.group(1)
        if luhn_checksum_valid(candidate):
            return candidate

    return None