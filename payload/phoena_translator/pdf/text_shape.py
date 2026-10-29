"""Text normalisers and shape predicates shared by every PDF pass."""

from __future__ import annotations

import re

from phoena_translator.pdf.patterns import _PDF_PARAGRAPH_MARKER_ATOM

_PDF_NUMBERED_PARAGRAPH_LEAD_RE = re.compile(
    rf"^\s*(?P<marker>(?:{_PDF_PARAGRAPH_MARKER_ATOM}[.)]|"
    rf"\({_PDF_PARAGRAPH_MARKER_ATOM}\)))\s+(?=\S)"
)
_PDF_BULLETED_PARAGRAPH_LEAD_RE = re.compile(
    r"^\s*[\u2022\u25aa\u25cf\u25e6\u2043\u2219\uf0b7\-*\u00b7]+\s+(?=\S)"
)


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


def _join_pdf_line_fragments(lines: list[str]) -> str:
    cleaned = [
        re.sub(r"\s+", " ", line).strip() for line in lines if line and line.strip()
    ]
    if not cleaned:
        return ""
    return " ".join(cleaned).strip()


def _looks_like_pdf_paragraph_lead(text: str) -> bool:
    """Return whether a visual line starts a numbered/bulleted paragraph.

    Legal and regulatory PDFs often encode every visual line as a separate
    text block.  A lead such as ``7.``, ``7.1.`` or ``7.2.1.`` identifies the
    first line of one semantic paragraph; its indented following lines belong
    to the same translation and render box.
    """
    plain = _collapsed_plain_text(text or "")
    numbered_match = _PDF_NUMBERED_PARAGRAPH_LEAD_RE.match(plain)
    if numbered_match and re.fullmatch(
        r"\((?:18|19|20)\d{2}\)",
        numbered_match.group("marker"),
    ):
        # A citation year at the beginning of a wrapped visual line, e.g.
        # ``(2016) show ...``, continues the preceding prose paragraph.
        numbered_match = None
    return bool(numbered_match or _PDF_BULLETED_PARAGRAPH_LEAD_RE.match(plain))


def _pdf_has_terminal_sentence_boundary(text: str) -> bool:
    plain = _collapsed_plain_text(text or "")
    return bool(re.search(r"[.!?。！？][\"'’”）)\]]*$", plain))


def _bucket_pdf_fontsize(value: float) -> float:
    return round(max(float(value), 1.0) * 2.0) / 2.0


def _pick_pdf_dominant_value(weighted_values: list[tuple[object, float]], default):
    if not weighted_values:
        return default

    scores = {}
    for value, weight in weighted_values:
        if value is None:
            continue
        scores[value] = scores.get(value, 0.0) + max(float(weight), 0.1)

    if not scores:
        return default
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _pdf_text_colors_semantically_compatible(first: int, second: int) -> bool:
    """Treat tiny neutral-black PDF color variations as the same body style."""
    first = int(first or 0)
    second = int(second or 0)
    if first == second:
        return True
    first_rgb = ((first >> 16) & 0xFF, (first >> 8) & 0xFF, first & 0xFF)
    second_rgb = ((second >> 16) & 0xFF, (second >> 8) & 0xFF, second & 0xFF)
    first_neutral = max(first_rgb) - min(first_rgb) <= 6 and max(first_rgb) <= 80
    second_neutral = max(second_rgb) - min(second_rgb) <= 6 and max(second_rgb) <= 80
    return bool(
        first_neutral
        and second_neutral
        and max(abs(a - b) for a, b in zip(first_rgb, second_rgb)) <= 40
    )


def _looks_like_heading_text(text: str) -> bool:
    """Heading signal that works even when the PDF fakes headings without a
    bold font: a short line whose letters are overwhelmingly uppercase."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 130:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.8


def _is_heading_like_elem(elem: dict) -> bool:
    if elem.get("bold"):
        return True
    content = _plain_text(elem.get("content", ""))
    lines = [line for line in content.splitlines() if line.strip()]
    return bool(lines) and all(_looks_like_heading_text(line) for line in lines)


def _ends_with_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary (i.e. does NOT need merging).
    Returns True if the text ends with any punctuation EXCEPT comma-like marks.
    Returns False (needs merging) if text has no trailing punctuation or ends with comma."""
    stripped = text.rstrip()
    if not stripped:
        return True
    last = stripped[-1]
    # Comma-like marks: sentence is mid-flow, need to merge
    if last in ",，、":
        return False
    # Any other punctuation: sentence boundary, no merge needed
    import unicodedata

    if unicodedata.category(last).startswith("P"):
        return True
    # No punctuation at all: incomplete sentence, need to merge
    return False
