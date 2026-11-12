"""Bound the figure or table around a text element whose rendering failed.

When a text element cannot be typeset at all (its translation does not fit
and putting the source text back in place fails too), the renderer would
give up the whole page.  If that element belongs to a figure or a table, the
renderer instead keeps that one figure or table in its exact source ink and
translates the rest of the page.  This module decides what "that figure or
table" is, and answers only when it is confident: ``None`` keeps the
page-level fallback.

Two rules, tried in order:

``table-region``
    The element lies inside a region PyMuPDF's table finder detected (a
    ``table_image`` element).  The group is every text element whose centre
    lies in that region.  A region that holds body prose is a mis-detection
    and is not used.

``caption-band``
    Tables and charts the finder missed are framed by a numbered title
    ("Figure 3:", "Exhibit 2.", "Table 1") and set in type smaller than the
    body.  The band runs from the nearest title above the element down to the
    first frame end: a source or note line, another title, body prose, text
    in body type or larger, a footnote, the footer band.  (A caption below
    the element frames the band upward the same way.)  Titles side by side
    split the page width between them.  The group is every text element whose
    centre lies in the band.  It is used only when most of its members are
    cells or labels by the layout classifier's own reading.

Neither rule takes an accepted cross-page merge endpoint or a merged-away
element into a group: a merge spans pages, so its fallback stays the
page-level closure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz

from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import ElementType, LayoutClass, SkipReason
from phoena_translator.pdf.semantics.layout_classifier import _pdf_element_is_body_prose
from phoena_translator.pdf.semantics.table_preservation import (
    _pdf_element_in_page_footer_band,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text


@dataclass(frozen=True)
class FigureTableGroupRules:
    # A detected table region is widened by this many points before testing
    # whether an element's centre lies inside it.
    region_margin: float = 2.0
    # Text at or above this share of the document's dominant type size is
    # body type and ends a caption band; the same boundary
    # ``_pdf_element_is_body_prose`` draws.
    body_type_share: float = 0.92
    # Vertical slack, in points, when comparing a title or frame end with
    # the element it frames.
    edge_tolerance: float = 0.5
    # A band with fewer members, or one the classifier mostly did not call
    # table cells, is not trusted as a figure or table.
    min_band_members: int = 2
    min_band_table_share: float = 0.5
    # Hard ceiling on figures/tables narrowed on one page.
    max_groups_per_page: int = 12


RULES = FigureTableGroupRules()

TABLE_REGION = "table-region"
CAPTION_BAND = "caption-band"

# A numbered figure/table title: the number is followed by a separator or
# ends the line, so a sentence such as "Table 1 shows ..." is not a title.
_TITLE_RE = re.compile(
    r"^\s*(?:figure|fig\.?|exhibit|chart|table|graph)\s*"
    r"[A-Z]?\d+[A-Za-z]?(?:[.\-]\d+)*\s*(?:[:.|\-–—]|$)",
    re.IGNORECASE,
)
_SOURCE_NOTE_RE = re.compile(r"^\s*(?:sources?|notes?)\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class FigureTableGroup:
    """One figure or table kept in its exact source ink."""

    kind: str
    trigger_element: int
    reason: str
    members: frozenset[int]
    bbox: tuple[float, float, float, float]


def _is_text(elem: dict) -> bool:
    return (
        elem.get("type") == ElementType.TEXT
        and elem.get("skip_translate_reason") != SkipReason.WATERMARK
    )


def _safe_rect(elem: dict) -> fitz.Rect | None:
    try:
        rect = _get_pdf_elem_rect(elem)
    except Exception:
        return None
    return None if rect.is_empty or rect.is_infinite else rect


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)


def _union(rects: list[fitz.Rect]) -> tuple[float, float, float, float]:
    box = fitz.Rect(rects[0])
    for rect in rects[1:]:
        box |= rect
    return (
        round(float(box.x0), 3),
        round(float(box.y0), 3),
        round(float(box.x1), 3),
        round(float(box.y1), 3),
    )


def _plain(elem: dict) -> str:
    return _collapsed_plain_text(elem.get("content") or "")


def _band_members(
    elements: list[dict],
    band: fitz.Rect,
    dominant_fontsize: float,
) -> set[int] | None:
    """Text elements centred in ``band``; ``None`` if the band holds prose or a merge."""
    members = set()
    for index, elem in enumerate(elements):
        rect = _safe_rect(elem)
        if rect is None or not band.contains(_center(rect)):
            continue
        if elem.get("type") == ElementType.MERGED_AWAY:
            return None
        if not _is_text(elem):
            continue
        if _pdf_element_is_body_prose(elem, dominant_fontsize):
            return None
        members.add(index)
    return members


def _table_region_members(
    elements: list[dict],
    seed: int,
    seed_rect: fitz.Rect,
    dominant_fontsize: float,
) -> set[int] | None:
    """Text elements of the smallest detected table region holding the seed."""
    seed_center = _center(seed_rect)
    margin = RULES.region_margin
    regions = []
    for elem in elements:
        if elem.get("type") != ElementType.TABLE_IMAGE:
            continue
        rect = _safe_rect(elem)
        if rect is None:
            continue
        widened = fitz.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)
        if widened.contains(seed_center):
            regions.append(widened)
    for region in sorted(regions, key=lambda rect: (rect.width * rect.height, tuple(rect))):
        members = _band_members(elements, region, dominant_fontsize)
        if members and seed in members:
            return members
    return None


def _is_title(elem: dict, dominant_fontsize: float) -> bool:
    return (
        _is_text(elem)
        and bool(_TITLE_RE.match(_plain(elem)))
        and not _pdf_element_is_body_prose(elem, dominant_fontsize)
    )


def _is_frame_end(elem: dict, page_rect, dominant_fontsize: float) -> bool:
    """Whether ``elem`` closes a caption band instead of belonging to it."""
    if elem.get("type") == ElementType.MERGED_AWAY:
        return True
    if not _is_text(elem):
        return False
    plain = _plain(elem)
    if not plain:
        return False
    if _TITLE_RE.match(plain) or _SOURCE_NOTE_RE.match(plain):
        return True
    if elem.get("footnote_hint") or elem.get("reference_entry_hint"):
        return True
    if _pdf_element_is_body_prose(elem, dominant_fontsize):
        return True
    try:
        size = float(elem.get("fontsize") or 0.0)
    except (TypeError, ValueError):
        size = 0.0
    if size >= dominant_fontsize * RULES.body_type_share:
        return True
    return _pdf_element_in_page_footer_band(elem, page_rect)


def _title_span(
    title: int,
    title_rects: dict[int, fitz.Rect],
    page_rect: fitz.Rect,
) -> tuple[float, float]:
    """The page width a title owns, split with titles in the same row."""
    rect = title_rects[title]
    row = sorted(
        (other for other, other_rect in title_rects.items()
         if min(rect.y1, other_rect.y1) - max(rect.y0, other_rect.y0) > 0),
        key=lambda other: (title_rects[other].x0, other),
    )
    position = row.index(title)
    left = float(page_rect.x0)
    right = float(page_rect.x1)
    if position > 0:
        previous = title_rects[row[position - 1]]
        left = (previous.x1 + rect.x0) / 2.0
    if position + 1 < len(row):
        following = title_rects[row[position + 1]]
        right = (rect.x1 + following.x0) / 2.0
    return left, right


def _caption_band_members(
    elements: list[dict],
    seed: int,
    seed_rect: fitz.Rect,
    page_rect,
    dominant_fontsize: float,
) -> set[int] | None:
    """Text elements between the seed's figure title and its first frame end."""
    if page_rect is None:
        return None
    page_rect = fitz.Rect(page_rect)
    rects = {index: _safe_rect(elem) for index, elem in enumerate(elements)}
    title_rects = {
        index: rect
        for index, rect in rects.items()
        if rect is not None and _is_title(elements[index], dominant_fontsize)
    }
    frame_ends = {
        index: rect
        for index, rect in rects.items()
        if rect is not None and _is_frame_end(elements[index], page_rect, dominant_fontsize)
    }
    if seed in frame_ends:
        return None
    tolerance = RULES.edge_tolerance
    seed_center = _center(seed_rect)

    def owns(title: int) -> tuple[float, float] | None:
        left, right = _title_span(title, title_rects, page_rect)
        return (left, right) if left <= seed_center.x <= right else None

    def in_span(rect: fitz.Rect, span: tuple[float, float]) -> bool:
        return min(rect.x1, span[1]) - max(rect.x0, span[0]) > 0

    above = sorted(
        (index for index, rect in title_rects.items() if rect.y1 <= seed_rect.y0 + tolerance),
        key=lambda index: (-title_rects[index].y1, index),
    )
    for title in above:
        span = owns(title)
        if span is None:
            continue
        top = title_rects[title].y1
        ends = [
            rect.y0
            for index, rect in frame_ends.items()
            if index != title and rect.y0 >= top - tolerance and in_span(rect, span)
        ]
        bottom = min(ends) if ends else float(page_rect.y1)
        if seed_rect.y1 <= bottom + tolerance:
            return _band_members(
                elements, fitz.Rect(span[0], top, span[1], bottom), dominant_fontsize
            )
        break

    below = sorted(
        (index for index, rect in title_rects.items() if rect.y0 >= seed_rect.y1 - tolerance),
        key=lambda index: (title_rects[index].y0, index),
    )
    for title in below:
        span = owns(title)
        if span is None:
            continue
        bottom = title_rects[title].y0
        ends = [
            rect.y1
            for index, rect in frame_ends.items()
            if index != title and rect.y1 <= bottom + tolerance and in_span(rect, span)
        ]
        if not ends:
            # Nothing frames the band from above (a running header is small
            # type too): the figure cannot be bounded with confidence.
            return None
        top = max(ends)
        if seed_rect.y0 >= top - tolerance:
            return _band_members(
                elements, fitz.Rect(span[0], top, span[1], bottom), dominant_fontsize
            )
        break
    return None


def _trusted_band(elements: list[dict], members: set[int] | None, seed: int) -> bool:
    if not members or seed not in members or len(members) < RULES.min_band_members:
        return False
    table_cells = sum(
        1
        for index in members
        if elements[index].get("table_hint")
        or elements[index].get("layout_class") == LayoutClass.TABLE
    )
    return table_cells >= len(members) * RULES.min_band_table_share


def find_figure_table_group(
    elements: list[dict],
    element_index: int,
    *,
    page_rect,
    dominant_fontsize: float,
    reason: str = "",
    protected_indices: set[int] | frozenset[int] = frozenset(),
    kept_indices: set[int] | frozenset[int] = frozenset(),
) -> FigureTableGroup | None:
    """Return the figure or table holding ``element_index``, or ``None``.

    ``protected_indices`` (accepted cross-page merge endpoints) may never be
    kept as source by this narrowing.  ``kept_indices`` are elements an
    earlier group on the same page already keeps; a failure among them cannot
    be narrowed further, which bounds the caller's loop.
    """
    elements = list(elements or [])
    try:
        seed = int(element_index)
    except (TypeError, ValueError):
        return None
    if not 0 <= seed < len(elements):
        return None
    if seed in kept_indices or seed in protected_indices:
        return None
    seed_elem = elements[seed]
    seed_rect = _safe_rect(seed_elem)
    if seed_rect is None or not _is_text(seed_elem):
        return None
    if _pdf_element_is_body_prose(seed_elem, dominant_fontsize):
        return None

    region = _table_region_members(elements, seed, seed_rect, dominant_fontsize)
    band = _caption_band_members(elements, seed, seed_rect, page_rect, dominant_fontsize)
    if not _trusted_band(elements, band, seed):
        band = None
    # A detected region inside a captioned figure is a fragment of it (a
    # legend grid, one panel): the caption band is the whole figure.
    if band is not None and (region is None or region <= band):
        kind, members = CAPTION_BAND, band
    elif region:
        kind, members = TABLE_REGION, region
    else:
        return None
    if members & set(protected_indices):
        return None
    return FigureTableGroup(
        kind=kind,
        trigger_element=seed,
        reason=str(reason or "")[:500],
        members=frozenset(members),
        bbox=_union([_get_pdf_elem_rect(elements[index]) for index in sorted(members)]),
    )


__all__ = [
    "CAPTION_BAND",
    "RULES",
    "TABLE_REGION",
    "FigureTableGroup",
    "FigureTableGroupRules",
    "find_figure_table_group",
]
