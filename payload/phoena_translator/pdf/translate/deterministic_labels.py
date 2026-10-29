"""Offline translations for fixed short labels and full dates, applied
without a model call."""

from __future__ import annotations

import re

from phoena_translator.pdf.semantics.references import _translate_pdf_citation_labels
from phoena_translator.pdf.text_shape import _collapsed_plain_text
from phoena_translator.pdf.translate.inline_markup import (
    _restore_pdf_superscript_markup,
)

PDF_FIXED_SHORT_LABEL_TRANSLATIONS = {
    "all†": "全部†",
    "day": "日盘",
    "half": "一半",
    "internal": "内部",
    "max": "最大值",
    "min": "最小值",
    "external": "外部",
    "other": "其他",
    "others": "其他",
    "night": "夜间",
    "received by": "接收方",
    "enter*": "输入*",
}
_PDF_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
_PDF_DATE_DMY_RE = re.compile(r"(?i)(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_PDF_DATE_MDY_RE = re.compile(r"(?i)([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})")


def _pdf_full_date_label_translation(text: str) -> str | None:
    """Deterministically translate an element that is exactly one full date.

    ``05 August 2026`` runs as a page header through entire research series,
    so every page pays an API call — or, when the model echoes it, a full
    retry ladder — for a closed, unambiguous transformation.  Only complete
    ``day month year`` / ``month day, year`` elements qualify; partial dates
    and dates inside prose keep going to the model.
    """
    plain = _collapsed_plain_text(text or "").rstrip(".")
    match = _PDF_DATE_DMY_RE.fullmatch(plain)
    if match:
        day, month_word, year = match.group(1), match.group(2), match.group(3)
    else:
        match = _PDF_DATE_MDY_RE.fullmatch(plain)
        if not match:
            return None
        month_word, day, year = match.group(1), match.group(2), match.group(3)
    month = _PDF_MONTH_NUMBERS.get(month_word.casefold())
    if month is None or not 1 <= int(day) <= 31:
        return None
    return f"{year}年{month}月{int(day)}日"


def _translate_pdf_fixed_short_label(text: str) -> str:
    """Translate a closed set of standalone chart/table labels offline.

    The ordinary tiny-label exemption protects company names, series codes,
    and mathematical identifiers from context-free API translation.  A small
    controlled vocabulary of unambiguous functional labels can still be
    translated deterministically, provided the *whole* element matches.
    """
    raw = text or ""
    plain = _collapsed_plain_text(raw)
    replacement = PDF_FIXED_SHORT_LABEL_TRANSLATIONS.get(plain.casefold())
    if replacement is not None:
        return _restore_pdf_superscript_markup(raw, replacement)

    date_translation = _pdf_full_date_label_translation(plain)
    if date_translation is not None:
        return _restore_pdf_superscript_markup(raw, date_translation)

    band_match = re.fullmatch(
        r"(?i)band\s+(\d+)(\s*\([^)]*\))?",
        plain,
    )
    if band_match:
        suffix = band_match.group(2) or ""
        return f"第{band_match.group(1)}档{suffix}"

    times_match = re.fullmatch(
        r"([+−\-–—]?\d+(?:[.,]\d+)?(?:\s*[−\-–—]\s*\d+(?:[.,]\d+)?)?)\s+times?",
        plain,
        re.IGNORECASE,
    )
    if times_match:
        return f"{times_match.group(1)}倍"
    return raw


def _translate_pdf_deterministic_labels(text: str) -> str:
    """Apply every context-free PDF label translation in one stable order."""
    return _translate_pdf_fixed_short_label(
        _translate_pdf_citation_labels(text or "")
    )
