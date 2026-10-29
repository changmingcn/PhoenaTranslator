"""Optionally keep whole tables in their source language: mark table
pages, sweep their grid and roll back marks that caught body prose."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import fitz

from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import (
    ElementType,
    LayoutClass,
    PageExtraction,
    SkipReason,
)
from phoena_translator.pdf.semantics.cross_page import _cross_page_dominant_fontsize
from phoena_translator.pdf.semantics.layout_classifier import _pdf_element_is_body_prose


@dataclass(frozen=True)
class TablePreservationThresholds:
    # A page is tabular when its table-like cells are numeric enough and either
    # numerous or at least half of the page's text elements.
    min_digit_bearing_share: Fraction = Fraction(2, 5)  # exact: compared with counts
    dense_page_cells: int = 10
    min_page_cells: int = 4
    min_page_cell_share: float = 0.5
    min_marks_for_grid_sweep: int = 4
    # The grid sweep attaches a cell that overlaps a mark by half its width
    # and sits within a row gap of it.
    min_column_overlap: float = 0.5
    row_gap_min: float = 4.0
    row_gap_em: float = 0.45
    # Circuit breaker: release a page whose marks freeze most of its prose.
    breaker_min_prose_chars: int = 200
    breaker_frozen_share: float = 0.5


TABLES = TablePreservationThresholds()


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
        preserved.extend(
            _preserve_page_tables(
                int(page_num) + 1,
                info.get("elements") or [],
                (page_rects or {}).get(page_num),
                dominant_fontsize,
                rollbacks,
            )
        )
    return preserved


def _preserve_page_tables(
    page_number: int,
    elements: list[dict],
    page_rect,
    dominant_fontsize: float,
    rollbacks: list[dict] | None,
) -> list[dict]:
    """Apply the table policy to one page; return its preservation entries."""
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
    page_is_tabular = _pdf_page_is_tabular(elements, table_like, len(text_indices))
    entries, marked_rects = _seed_pdf_table_marks(
        page_number, elements, table_like, page_is_tabular
    )
    if len(marked_rects) >= TABLES.min_marks_for_grid_sweep:
        entries.extend(
            _sweep_pdf_table_grid(page_number, elements, text_indices, prose_indices, page_rect)
        )
    if _release_pdf_table_marks_freezing_prose(
        page_number, elements, prose_indices, entries, rollbacks
    ):
        return []
    return entries


def _pdf_page_is_tabular(elements: list[dict], table_like: list[int], text_count: int) -> bool:
    count = len(table_like)
    # Real data tables carry numbers; a title page's centered
    # author/date block also gets classified "table" but is mostly
    # digit-free prose and must keep translating.
    digit_bearing = sum(
        1
        for index in table_like
        if any(char.isdigit() for char in str(elements[index].get("content", "")))
    )
    numeric_enough = digit_bearing >= count * TABLES.min_digit_bearing_share
    return numeric_enough and (
        count >= TABLES.dense_page_cells
        or (count >= TABLES.min_page_cells and count >= text_count * TABLES.min_page_cell_share)
    )


def _seed_pdf_table_marks(
    page_number: int,
    elements: list[dict],
    table_like: list[int],
    page_is_tabular: bool,
) -> tuple[list[dict], list[fitz.Rect]]:
    """Mark region cells, and classifier table cells on tabular pages."""
    entries = []
    marked_rects = []
    for index in table_like:
        elem = elements[index]
        if elem.get("skip_translate_reason"):
            continue
        if not (elem.get("table_hint") or page_is_tabular):
            continue
        elem["skip_translate_reason"] = SkipReason.TABLE_PRESERVED
        entries.append(
            {
                "page": page_number,
                "element": index,
                "trigger": "region" if elem.get("table_hint") else "page-density",
            }
        )
        try:
            rect = _get_pdf_elem_rect(elem)
        except Exception:
            continue
        if not rect.is_empty:
            marked_rects.append(rect)
    return entries, marked_rects


def _sweep_pdf_table_grid(
    page_number: int,
    elements: list[dict],
    text_indices: list[int],
    prose_indices: set[int],
    page_rect,
) -> list[dict]:
    """Grow marks to cells the classifier stamped inconsistently.

    The classifier occasionally stamps individual cells (a column header, a
    first data row, the second line of a table note) as scattered/body.  Each
    unmarked element is compared against the marked elements themselves
    rather than against one box grown from all of them: a single box merges a
    table in the left text column with a chart in the right one, spans the
    page, and then walks up both text columns row by row (2026-07-27: a whole
    page of a two-column paper delivered in English).  Chaining locally
    cannot jump columns, because every step needs horizontal overlap with the
    element it grows from, and cannot cross a paragraph, because prose is not
    eligible.
    """
    entries = []
    changed = True
    while changed:
        changed = False
        anchors = [
            (index, _get_pdf_elem_rect(elements[index]))
            for index in text_indices
            if elements[index].get("skip_translate_reason") == SkipReason.TABLE_PRESERVED
        ]
        for index in text_indices:
            elem = elements[index]
            if elem.get("skip_translate_reason") or index in prose_indices:
                continue
            # The sweep exists to recover cells the classifier stamped
            # inconsistently.  ``body`` is not an inconsistent stamp, it is
            # the classifier saying "running text" — and a page of stacked
            # reference entries reads to every other signal exactly like a
            # page of table rows.
            if elem.get("layout_class") == LayoutClass.BODY:
                continue
            # The running footer sits a few points under the last note line
            # of a chart box in the same column; it belongs to the page, not
            # to the table.
            if _pdf_element_in_page_footer_band(elem, page_rect):
                continue
            try:
                rect = _get_pdf_elem_rect(elem)
            except Exception:
                continue
            if rect.is_empty or not _pdf_rect_attaches_to_table_anchor(rect, elem, anchors):
                continue
            elem["skip_translate_reason"] = SkipReason.TABLE_PRESERVED
            changed = True
            entries.append({"page": page_number, "element": index, "trigger": "grid-union"})
    return entries


def _pdf_rect_attaches_to_table_anchor(rect: fitz.Rect, elem: dict, anchors) -> bool:
    # Intra-table row gaps run a few points; a paragraph gap is wider and
    # stays outside.  Deliberately the same limit the box-growing sweep used:
    # the local sweep is here to stop reaching ACROSS a page, not to let it
    # reach further.
    gap_limit = max(
        TABLES.row_gap_min,
        TABLES.row_gap_em * float(elem.get("fontsize", 12.0) or 12.0),
    )
    for _anchor_index, anchor in anchors:
        if anchor.is_empty:
            continue
        overlap = min(rect.x1, anchor.x1) - max(rect.x0, anchor.x0)
        if overlap < min(rect.width, anchor.width) * TABLES.min_column_overlap:
            continue
        gap = max(anchor.y0 - rect.y1, rect.y0 - anchor.y1)
        if gap <= gap_limit:
            return True
    return False


def _release_pdf_table_marks_freezing_prose(
    page_number: int,
    elements: list[dict],
    prose_indices: set[int],
    entries: list[dict],
    rollbacks: list[dict] | None,
) -> bool:
    """Circuit breaker: undo a page's marks when they would freeze its prose.

    Verbatim preservation exists so a table keeps its geometry; it must never
    be the reason a reader gets a page of untranslated prose.  If this page's
    marks would freeze most of its running text anyway — a table region
    mis-detected over a whole column, some layout nobody has seen yet — drop
    every mark the policy made here and let the page translate.  Delivering a
    translated page with a reflowed table beats delivering the source page.
    """
    prose_total = sum(len(str(elements[index].get("content", ""))) for index in prose_indices)
    prose_frozen = sum(
        len(str(elements[index].get("content", "")))
        for index in prose_indices
        if elements[index].get("skip_translate_reason") == SkipReason.TABLE_PRESERVED
    )
    if not (
        prose_total >= TABLES.breaker_min_prose_chars
        and prose_frozen > prose_total * TABLES.breaker_frozen_share
    ):
        return False
    for entry in entries:
        elem = elements[entry["element"]]
        if elem.get("skip_translate_reason") == SkipReason.TABLE_PRESERVED:
            elem.pop("skip_translate_reason", None)
    if rollbacks is not None:
        rollbacks.append({
            "page": page_number,
            "reason": "table-preservation-would-freeze-page-prose",
            "released_elements": len(entries),
            "prose_chars": int(prose_total),
            "frozen_prose_chars": int(prose_frozen),
        })
    return True
