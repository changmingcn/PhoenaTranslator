"""Entity directories: aligned firm/participant names introduced by
explicit list or directory prose, which keep their source text."""

from __future__ import annotations

import re
from phoena_translator.pdf.patterns import (
    _PDF_PROPER_NAME_CONNECTORS,
    _PRESERVED_IDENTIFIER_RE,
)
from phoena_translator.pdf.translation_targets import _SHORT_CITATION_WORD_RE
from phoena_translator.pdf.model import ElementType, LayoutClass
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.geometry import _get_pdf_source_ink_rect


def _looks_like_pdf_entity_directory_row(text: str) -> bool:
    """Return whether one short line can be an entity in a declared directory.

    This deliberately does not decide preservation on its own.  Generic title-
    cased phrases such as ``Market Liquidity Crisis`` still need translation.
    A caller must additionally establish page-level directory language and a
    sufficiently long, aligned run of peer rows.
    """
    plain = _collapsed_plain_text(text or "")
    if not plain or len(plain) > 90 or _PRESERVED_IDENTIFIER_RE.search(plain):
        return False
    # A detached one-letter footnote marker is part of several company rows
    # (for example ``Knight Capital a``), not a lower-case prose word.
    lexical = re.sub(r"\s+[a-z]\s*$", "", plain).strip()
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(lexical)
    ]
    if not 1 <= len(words) <= 10:
        return False
    if not any(any(char.isalpha() for char in word) for word in words):
        return False
    for word in words:
        normalized = word.strip("&.")
        if not normalized:
            continue
        if normalized.casefold() in _PDF_PROPER_NAME_CONNECTORS:
            continue
        if not (normalized[:1].isupper() or normalized.isupper()):
            return False
    residue = _SHORT_CITATION_WORD_RE.sub("", lexical)
    return bool(re.fullmatch(r"[\s,&'’()\-./]*", residue))


def _mark_pdf_entity_directory_elements(elements: list[dict]) -> int:
    """Mark aligned entity-name rows under explicit list/directory prose.

    Some PDFs contain an explanatory paragraph followed by a one-column list
    of firms.  Individual names such as ``All Options International`` are
    linguistically ambiguous without that page context and were repeatedly
    sent to the LLM.  Require both explicit directory language and at least
    five tightly aligned peer rows so ordinary short headings remain outside
    this preservation rule.
    """
    page_plain = re.sub(
        r"\s+",
        " ",
        " ".join(
            _plain_text(element.get("content", ""))
            for element in elements or []
            if element.get("type") == ElementType.TEXT
        ),
    ).strip()
    page_folded = page_plain.casefold()
    entity_noun = (
        r"(?:companies|company|entities|firms|hfts|institutions|members|"
        r"organisations|organizations|participants|traders)"
    )
    has_directory_context = bool(
        re.search(rf"\b(?:list|directory)\s+of\s+(?:the\s+)?{entity_noun}\b", page_folded)
        or re.search(
            r"\bmembers?\s+of\s+(?:the\s+)?(?:study|working|advisory|"
            r"steering)\s+group\b",
            page_folded,
        )
        or re.search(
            rf"\b{entity_noun}\b[^.!?]{{0,100}}\b(?:presented|listed|ordered)\s+"
            r"in\s+alphabetical\s+order\b",
            page_folded,
        )
    )
    if not has_directory_context:
        return 0

    def row_candidate(element: dict) -> bool:
        return bool(
            element.get("type") == ElementType.TEXT
            and element.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED}
            and not element.get("bold")
            and _looks_like_pdf_entity_directory_row(element.get("content", ""))
        )

    candidates = [
        element for element in (elements or []) if row_candidate(element)
    ]
    columns: list[list[dict]] = []
    for element in sorted(
        candidates,
        key=lambda candidate: (
            _get_pdf_source_ink_rect(candidate).x0,
            _get_pdf_source_ink_rect(candidate).y0,
        ),
    ):
        rect = _get_pdf_source_ink_rect(element)
        fontsize = max(float(element.get("fontsize", 9.0)), 1.0)
        compatible_columns = [
            column
            for column in columns
            if abs(
                rect.x0 - _get_pdf_source_ink_rect(column[0]).x0
            ) <= max(12.0, fontsize * 1.5)
        ]
        if compatible_columns:
            min(
                compatible_columns,
                key=lambda column: abs(
                    rect.x0 - _get_pdf_source_ink_rect(column[0]).x0
                ),
            ).append(element)
        else:
            columns.append([element])

    runs: list[list[dict]] = []
    for column in columns:
        current: list[dict] = []
        for element in sorted(
            column,
            key=lambda candidate: _get_pdf_source_ink_rect(candidate).y0,
        ):
            rect = _get_pdf_source_ink_rect(element)
            if current:
                previous = current[-1]
                previous_rect = _get_pdf_source_ink_rect(previous)
                fontsize = max(
                    float(element.get("fontsize", 9.0)),
                    float(previous.get("fontsize", 9.0)),
                    1.0,
                )
                compact_gap = (
                    rect.y0 - previous_rect.y1
                    <= max(42.0, fontsize * 5.0)
                )
                forward = rect.y0 >= previous_rect.y0 - 0.5
                if not (compact_gap and forward):
                    runs.append(current)
                    current = []
            current.append(element)
        if current:
            runs.append(current)

    marked = 0
    for run in runs:
        if len(run) < 5:
            continue
        for element in run:
            if not element.get("entity_directory_hint"):
                element["entity_directory_hint"] = True
                marked += 1
    return marked
