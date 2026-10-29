"""Optionally keep whole tables in their source language: mark table
pages, sweep their grid and roll back marks that caught body prose."""

from __future__ import annotations

import fitz
from phoena_translator.pdf.semantic_cross_page import _cross_page_dominant_fontsize
from phoena_translator.pdf.model import (
    ElementType,
    LayoutClass,
    PageExtraction,
    SkipReason,
)
from phoena_translator.pdf.layout_classifier import _pdf_element_is_body_prose
from phoena_translator.pdf.geometry import _get_pdf_elem_rect


def _pdf_element_in_page_footer_band(elem: dict, page_rect) -> bool:
    """True for the running footer band under the last body/table line.

    Only the foot of the page is fenced off, not the classifier's symmetric
    header band: a page can legitimately open with a chart box, and its title
    sits inside the top band.
    """
    if page_rect is None:
        return False
    height = float(getattr(page_rect, "height", 0.0) or 0.0)
    if height <= 0.0:
        return False
    try:
        rect = _get_pdf_elem_rect(elem)
    except Exception:
        return False
    if rect.is_empty:
        return False
    center_y = ((rect.y0 + rect.y1) / 2.0 - float(page_rect.y0)) / height
    return center_y >= 0.92


def _mark_pdf_table_preserved_elements(
    page_extractions: dict[int, PageExtraction],
    rollbacks: list[dict] | None = None,
    page_rects: dict | None = None,
) -> list[dict]:
    """Mark table elements to keep their original ink, like formula risks.

    Tables rarely survive translation with row/column alignment intact
    (2026-07-24 policy: preserve them verbatim, exactly like dense math).
    Region-detected cells (``table_hint``) are always preserved; classifier
    ``layout_class == "table"`` elements are preserved only on
    table-dominated pages, because the layout classifier also stamps a few
    stray short fragments per ordinary prose page and those must keep
    translating.  Body prose is never eligible for either heuristic trigger,
    and a page whose prose would end up frozen anyway rolls the whole page's
    marks back."""
    preserved = []
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions or {})
    for page_num, info in page_extractions.items():
        elements = info.get("elements") or []
        text_indices = [
            index
            for index, elem in enumerate(elements)
            if elem.get("type") == ElementType.TEXT
        ]
        prose_indices = {
            index
            for index in text_indices
            if _pdf_element_is_body_prose(elements[index], dominant_fontsize)
        }
        # Region geometry stays authoritative: a cell inside a detected grid
        # is preserved whatever it reads like.  The classifier is the noisy
        # signal, so it may not speak for prose.
        table_like = [
            index
            for index in text_indices
            if elements[index].get("table_hint")
            or (
                elements[index].get("layout_class") == LayoutClass.TABLE
                and index not in prose_indices
            )
        ]
        count = len(table_like)
        # Real data tables carry numbers; a title page's centered
        # author/date block also gets classified "table" but is mostly
        # digit-free prose and must keep translating.
        digit_bearing = sum(
            1
            for index in table_like
            if any(
                char.isdigit()
                for char in str(elements[index].get("content", ""))
            )
        )
        numeric_enough = digit_bearing * 5 >= count * 2
        page_is_tabular = numeric_enough and (
            count >= 10 or (count >= 4 and count * 2 >= len(text_indices))
        )
        marked_rects = []
        for index in table_like:
            elem = elements[index]
            if elem.get("skip_translate_reason"):
                continue
            if not (elem.get("table_hint") or page_is_tabular):
                continue
            elem["skip_translate_reason"] = SkipReason.TABLE_PRESERVED
            preserved.append(
                {
                    "page": int(page_num) + 1,
                    "element": index,
                    "trigger": (
                        "region" if elem.get("table_hint") else "page-density"
                    ),
                }
            )
            try:
                rect = _get_pdf_elem_rect(elem)
            except Exception:
                continue
            if not rect.is_empty:
                marked_rects.append(rect)
        # Grid sweep: the classifier occasionally stamps individual cells (a
        # column header, a first data row, the second line of a table note)
        # as scattered/body.  Each unmarked element is compared against the
        # marked elements themselves rather than against one box grown from
        # all of them: a single box merges a table in the left text column
        # with a chart in the right one, spans the page, and then walks up
        # both text columns row by row (2026-07-27: a whole page of a
        # two-column paper delivered in English).  Chaining locally cannot
        # jump columns, because every step needs horizontal overlap with the
        # element it grows from, and cannot cross a paragraph, because prose
        # is not eligible.
        page_rect = (page_rects or {}).get(page_num)
        if len(marked_rects) >= 4:
            changed = True
            while changed:
                changed = False
                anchors = [
                    (index, fitz.Rect(elements[index].get("bbox", elements[index].get("rect"))))
                    for index in text_indices
                    if elements[index].get("skip_translate_reason")
                    == SkipReason.TABLE_PRESERVED
                ]
                for index in text_indices:
                    elem = elements[index]
                    if elem.get("skip_translate_reason"):
                        continue
                    if index in prose_indices:
                        continue
                    # The sweep exists to recover cells the classifier stamped
                    # inconsistently.  ``body`` is not an inconsistent stamp,
                    # it is the classifier saying "running text" — and a page
                    # of stacked reference entries reads to every other signal
                    # exactly like a page of table rows.
                    if elem.get("layout_class") == LayoutClass.BODY:
                        continue
                    # The running footer sits a few points under the last
                    # note line of a chart box in the same column; it belongs
                    # to the page, not to the table.
                    if _pdf_element_in_page_footer_band(elem, page_rect):
                        continue
                    try:
                        rect = _get_pdf_elem_rect(elem)
                    except Exception:
                        continue
                    if rect.is_empty:
                        continue
                    # Intra-table row gaps run a few points; a paragraph gap is
                    # wider and stays outside.  Deliberately the same limit the
                    # box-growing sweep used: this rewrite is here to stop the
                    # sweep reaching ACROSS a page, not to let it reach further.
                    gap_limit = max(
                        4.0, 0.45 * float(elem.get("fontsize", 12.0) or 12.0)
                    )
                    attached = False
                    for anchor_index, anchor in anchors:
                        if anchor.is_empty:
                            continue
                        overlap = min(rect.x1, anchor.x1) - max(
                            rect.x0, anchor.x0
                        )
                        if overlap < min(rect.width, anchor.width) * 0.5:
                            continue
                        gap = max(anchor.y0 - rect.y1, rect.y0 - anchor.y1)
                        if gap > gap_limit:
                            continue
                        attached = True
                        break
                    if not attached:
                        continue
                    elem["skip_translate_reason"] = SkipReason.TABLE_PRESERVED
                    changed = True
                    preserved.append(
                        {
                            "page": int(page_num) + 1,
                            "element": index,
                            "trigger": "grid-union",
                        }
                    )
        # Circuit breaker.  Verbatim preservation exists so a table keeps its
        # geometry; it must never be the reason a reader gets a page of
        # untranslated prose.  If this page's marks would freeze most of its
        # running text anyway — a table region mis-detected over a whole
        # column, some layout nobody has seen yet — drop every mark the policy
        # made here and let the page translate.  Delivering a translated page
        # with a reflowed table beats delivering the source page.
        prose_total = sum(
            len(str(elements[index].get("content", "")))
            for index in prose_indices
        )
        prose_frozen = sum(
            len(str(elements[index].get("content", "")))
            for index in prose_indices
            if elements[index].get("skip_translate_reason") == SkipReason.TABLE_PRESERVED
        )
        if prose_total >= 200 and prose_frozen * 2 > prose_total:
            page_marks = [
                entry
                for entry in preserved
                if entry["page"] == int(page_num) + 1
            ]
            for entry in page_marks:
                elem = elements[entry["element"]]
                if elem.get("skip_translate_reason") == SkipReason.TABLE_PRESERVED:
                    elem.pop("skip_translate_reason", None)
                preserved.remove(entry)
            if rollbacks is not None:
                rollbacks.append({
                    "page": int(page_num) + 1,
                    "reason": "table-preservation-would-freeze-page-prose",
                    "released_elements": len(page_marks),
                    "prose_chars": int(prose_total),
                    "frozen_prose_chars": int(prose_frozen),
                })
    return preserved
