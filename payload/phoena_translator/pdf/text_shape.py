"""Text normalisers and shape predicates shared by every PDF pass."""

from __future__ import annotations

import re


def _plain_text(text: str) -> str:
    """Remove lightweight inline markup for translation checks."""
    # PDF rich text only emits a small, known markup vocabulary.  A generic
    # ``<...>`` stripper corrupts ordinary comparison prose such as
    # ``small (<£100,000) ... large (>£1,000,000)`` by treating the range as
    # one HTML tag.  Strip only actual lightweight tags and preserve literal
    # less-than / greater-than operators for translation and formula audit.
    return re.sub(
        r"(?is)<\s*/?\s*(?:b|strong|i|em|sup|sub|span|a)\b[^>]*>",
        "",
        text or "",
    )


def _collapsed_plain_text(text: str) -> str:
    """Return ``_plain_text`` with whitespace runs collapsed and ends trimmed."""
    return re.sub(r"\s+", " ", _plain_text(text)).strip()


def _looks_like_pdf_list_marker(text: str) -> bool:
    marker = re.sub(r'\s+', ' ', text or '').strip()
    if not marker or len(marker) > 12:
        return False
    patterns = [
        r'^[\u2022\u25AA\u25CF\u25E6\u2043\u2219\uF0B7\-\*\u00B7]+$',
        r'^\(?\d{1,3}[.)]\)?$',
        r'^\(?[A-Za-z][.)]\)?$',
        r'^\(?[ivxlcdmIVXLCDM]{1,8}[.)]\)?$',
        r'^[\u4e00-\u9fff]{1,3}[、.)]$',
    ]
    return any(re.fullmatch(pattern, marker) for pattern in patterns)
