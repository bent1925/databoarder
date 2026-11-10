"""Decides what the AI is allowed to see. Identifying values are replaced before anything leaves the server.

A masked ID keeps its format, so C-00123 becomes X-99999. That is enough to see that two files write
IDs differently, without revealing anyone's actual ID. Group labels and amounts are left as they are,
because the AI needs them and they don't identify anyone.
"""
from __future__ import annotations

import re

import pandas as pd

MAX_EXAMPLES = 6
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_ID_NAME = re.compile(r"(^|[\s_-])id$|^id[\s_-]|uuid|^user$|^customer$|^account$|customer.?(ref|code|no|num)|^cust", re.I)
_SENSITIVE_NAME = re.compile(r"name|e-?mail|phone|mobile|address|postcode|postal|zip|ssn|birth|dob|ip.?addr|cookie|device", re.I)
# Always about one person, however often the values repeat (an events table repeats each customer's phone).
# Names like zip_code or campaign_name can hold harmless groups, so they keep the uniqueness rule below.
_PERSONAL_NAME = re.compile(r"e-?mail|phone|mobile|^tel|ssn|birth|dob|ip.?addr|street|"
                            r"(first|last|full|given|family|sur|customer|person|contact).?name|^name$", re.I)


def _is_mostly_dates(text: pd.Series) -> bool:
    if not text.str.contains(r"\d").all():
        return False
    return bool(pd.to_datetime(text, errors="coerce", format="mixed").notna().mean() >= 0.9)


def should_mask(name: str, series: pd.Series) -> bool:
    """True when a column's values could point to a particular person or free-form text about them."""
    values = series.dropna()
    if values.empty:
        return False
    sample = values.astype(str).head(1000)
    if sample.str.contains(_EMAIL).any() or _ID_NAME.search(str(name)) or _PERSONAL_NAME.search(str(name)):
        return True
    rows, unique = len(values), int(values.nunique())
    if _SENSITIVE_NAME.search(str(name)) and (unique > 20 or unique >= 0.9 * rows):
        return True
    phones = sample.str.fullmatch(_PHONE) & ~sample.str.fullmatch(r"[+-]?\d*\.\d+")    # not decimals like 0.123456789
    if phones.any() and not _is_mostly_dates(sample):
        return True
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all():
        return bool((numbers == numbers.round()).all() and rows >= 20 and unique >= 0.9 * rows)
    if _is_mostly_dates(sample):
        return False
    return unique >= max(20, 0.5 * rows)


def shape_of(value) -> str:
    """Keeps a value's format and hides its content."""
    text = str(value)
    if _EMAIL.search(text):
        return "<email>"
    if _PHONE.fullmatch(text):
        return "<phone>"
    return re.sub(r"\d", "9", re.sub(r"[A-Z]", "X", re.sub(r"[^\W\d_A-Z]", "x", text)))[:24]


def safe_examples(name: str, series: pd.Series, limit: int = MAX_EXAMPLES) -> tuple[list[str], bool]:
    """Example values that are safe to show the AI, and whether they were masked."""
    examples = [str(value) for value in series.dropna().unique()[:limit]]
    if should_mask(name, series):
        return list(dict.fromkeys(shape_of(value) for value in examples)), True
    return examples, False


def masked_profile(df: pd.DataFrame, profile: list[dict]) -> list[dict]:
    """A copy of a column profile for the AI. The original, unmasked one is only ever shown in the browser."""
    result = []
    for column in profile:
        examples, masked = safe_examples(column["name"], df[column["name"]])
        result.append({**column, "examples": examples, **({"masked": True} if masked else {})})
    return result
