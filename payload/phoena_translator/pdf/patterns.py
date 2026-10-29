"""Lexical patterns and word lists shared by several PDF passes."""

from __future__ import annotations

import re


PDF_FORMULA_GUARD_PADDING = 0.5
_PDF_PARAGRAPH_MARKER_ATOM = (
    r"(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdmIVXLCDM]{1,8})"
)
_PDF_BARE_FOOTNOTE_LEAD_RE = re.compile(r"^\d{1,3}\s+(?=[A-Z])")
_PRESERVED_URI_BODY = r"[A-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]+"
_PRESERVED_URI_LAYOUT_WRAPS = (
    rf"(?:"
    rf"(?<=[/_?&=%\-])\s+{_PRESERVED_URI_BODY}"
    rf"|(?<=\.)\s+(?=[A-Z0-9._~\-]+[/?:#]){_PRESERVED_URI_BODY}"
    rf")*"
)
_PRESERVED_IDENTIFIER_RE = re.compile(
    rf"(?i)(?:"
    rf"(?:https?|ftp)://{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf"|www\.{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf"|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{{2,}}"
    rf"|(?:doi:\s*)?10\.\d{{4,9}}/{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf")"
)
_PDF_PROPER_NAME_CONNECTORS = {
    "and", "at", "by", "de", "del", "der", "di", "dos", "du", "et",
    "for", "in", "jr", "la", "le", "of", "on", "sr", "the", "van",
    "von",
}
PDF_TRANSLATABLE_CITATION_LABELS = (
    (re.compile(r"(?i)\blast\s+accessed\s*:\s*"), "最后访问日期："),
    (re.compile(r"(?i)\baccessed\s*:\s*"), "访问日期："),
    (re.compile(r"(?i)\bretrieved(?:\s+on)?\s*:\s*"), "检索日期："),
    (re.compile(r"(?i)\bavailable\s+at\s*:\s*"), "可获取地址："),
    (re.compile(r"(?i)\bavailable\s*:\s*"), "可获取："),
)

