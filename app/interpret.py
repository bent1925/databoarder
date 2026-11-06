"""Small helpers that turn numbers into plain language, plus a number auditor.

The auditor is the safety net for any future LLM-written prose: every number in
a sentence must be one the analysis actually produced.
"""
from __future__ import annotations

import re


def humanize(col: str) -> str:
    return col.replace("_", " ").strip()


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def chance_phrase(p: float) -> str:
    """Turn a p-value into 'about N in 100 times' language."""
    if p < 0.001:
        return "fewer than 1 in 1,000 times"
    if p < 0.01:
        return f"about {p * 1000:.0f} in 1,000 times"
    return f"about {p * 100:.0f} in 100 times"


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text: str) -> set[str]:
    return {m.group(0).rstrip(".,") for m in _NUM.finditer(text)}


def unsupported_numbers(text: str, allowed_text: str) -> set[str]:
    """Numbers in `text` that never appear in `allowed_text` (the engine's own output)."""
    allowed = numbers_in(allowed_text)
    return {n for n in numbers_in(text) if n not in allowed}
