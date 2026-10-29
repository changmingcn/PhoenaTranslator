"""Table-of-contents rows: dotted leaders and page references."""

from __future__ import annotations

import re

from phoena_translator.pdf.text_shape import _collapsed_plain_text
from phoena_translator.pdf.translate.inline_markup import _normalize_pdf_translation

_PDF_TOC_ENTRY_RE = re.compile(
    r"(?P<leader>(?:[.…·]\s*){4,})\s*(?P<page>\d+)\b"
)


def _pdf_toc_entries(text: str) -> list[tuple[str, str]]:
    """Return ordered ``(label, page)`` entries from a dotted-leader block."""
    plain = _collapsed_plain_text(text)
    entries = []
    cursor = 0
    for match in _PDF_TOC_ENTRY_RE.finditer(plain):
        label = plain[cursor:match.start()].strip()
        if label:
            entries.append((label, match.group("page")))
        cursor = match.end()
    return entries


def _pdf_toc_trailing_text(text: str) -> str:
    plain = _collapsed_plain_text(text)
    matches = list(_PDF_TOC_ENTRY_RE.finditer(plain))
    return plain[matches[-1].end():].strip() if matches else plain


def _parse_pdf_toc_leader(text: str) -> dict | None:
    plain = _collapsed_plain_text(text)
    if not plain:
        return None

    match = re.match(
        r"^(?P<label>.+?)\s*(?:[.\u2024\u2025\u2026\u00b7·]{6,})\s*(?P<page>(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,12}|[A-Za-z]-?\d{1,4}))\s*$",
        plain,
    )
    if not match:
        return None

    label = match.group("label").strip()
    page = match.group("page").strip()
    if not label or not page:
        return None
    return {"label": label, "page": page}


def _strip_pdf_toc_label(text: str, page_ref: str) -> str:
    plain = _collapsed_plain_text(_normalize_pdf_translation(text))
    if not plain:
        return ""

    plain = re.sub(
        rf"\s*(?:[.\u2024\u2025\u2026\u00b7·]{{3,}}\s*)?{re.escape(page_ref)}\s*$",
        "",
        plain,
    ).strip()
    plain = re.sub(r"[.\u2024\u2025\u2026\u00b7·]{3,}\s*$", "", plain).strip()
    return plain
