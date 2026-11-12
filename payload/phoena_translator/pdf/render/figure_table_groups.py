"""Bound the figure or table around a text element whose rendering failed.

When a text element's translation cannot be placed, the renderer would
re-typeset its source text in place, or give up the whole page when that
fails too.  If that element belongs to a figure or a table, the renderer
instead keeps that one figure or table in its exact source ink and
translates the rest of the page.  This module decides what "that figure or
table" is, and answers only when it is confident: ``None`` keeps the
page-level fallback.

Two rules, tried in order:

``table-region``
    The element lies inside a region PyMuPDF's table finder detected (a
    ``table_image`` element).  The group is the region's text elements, less
    the text that is never a member (below).

``caption-band``
    Tables and charts the finder missed are framed by a numbered title
    ("Figure 3:", "Exhibit 2.", "Table 1") and set in type smaller than the
    body.  The band runs from the nearest title above the element down to the
    first frame end: a source or note line, another title, prose, text in
    body type or larger, a footnote, the running header or footer band.  (A
    caption below the element frames the band upward the same way.)  A band
    that nothing closes is not used.  Titles side by side, level or slightly
    staggered, split the page width between them.  Across the band the
    figure extends over its title and the element, and over the vector
    graphics and images joined to them inside the band; text beside that
    extent is not taken, and text that straddles it, or sits on the same
    lines as the figure's text, makes the extent undecidable.  A band is used
    only when most of its members are cells or labels by the layout
    classifier's own reading.

Never a member under either rule: prose (text classed ``body``, or reading
as running prose, that stands alone on its lines: no other text beside it),
titles and captions, source and note lines with their continuation lines,
footnotes, the running header and footer bands.  Text that reads long but
sits in a row with other cells is table text.  Neither rule takes an
accepted cross-page merge endpoint or a merged-away element into a group: a
merge spans pages, so its fallback stays the page-level closure.  When a
rule cannot decide, it answers ``None`` rather than a wider group.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz

from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import ElementType, LayoutClass, SkipReason
from phoena_translator.pdf.model import DEFAULT_FONTSIZE
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
    # The running header band: centre in the top share of the page (the
    # footer band is the bottom 8 %, ``_pdf_element_in_page_footer_band``).
    header_band_share: float = 0.08
    # Running prose: at least this many words, or the smaller count wrapped
    # over more than ``prose_wrap_lines`` lines (the shape test of
    # ``_pdf_element_is_body_prose`` without its type-size gate).
    prose_long_words: int = 12
    prose_wrapped_words: int = 8
    prose_wrap_lines: float = 1.6
    # Two boxes share a line when they overlap vertically by at least this
    # share of the shorter one.  Table cells with different line counts are
    # centred in their row, so a one-line cell beside a two-line cell
    # overlaps it by under half a line (case 1 page 3: 0.44).
    same_line_share: float = 0.25
    # Side-by-side titles count as one row when the vertical gap between
    # them is at most this share of the taller title.
    staggered_title_share: float = 1.0
    # A source or note line continues onto the next line when it keeps its
    # type size and left edge and the gap is at most this share of the size.
    note_continuation_gap_share: float = 0.6
    # A band with fewer members, or one the classifier mostly did not call
    # table cells, is not trusted as a figure or table.
    min_band_members: int = 2
    min_band_table_share: float = 0.5
    # Hard ceiling on figures/tables narrowed on one page.
    max_groups_per_page: int = 12


RULES = FigureTableGroupRules()

TABLE_REGION = "table-region"
CAPTION_BAND = "caption-band"

# What made the renderer give the group up: the page could not be rendered
# at all, or an element would have had its source text re-typeset in place.
PAGE_FAILURE = "page-failure"
ELEMENT_RESTORE = "element-restore"

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
    trigger: str = PAGE_FAILURE


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


def _fontsize(elem: dict) -> float:
    try:
        return float(elem.get("fontsize") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _x_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    return min(a.x1, b.x1) - max(a.x0, b.x0)


def _same_line(a: fitz.Rect, b: fitz.Rect) -> bool:
    """``b`` sits beside ``a`` on the same lines (left or right of it)."""
    shorter = max(min(a.height, b.height), 0.1)
    vertical = min(a.y1, b.y1) - max(a.y0, b.y0)
    return vertical >= shorter * RULES.same_line_share and _x_overlap(a, b) <= 0


def _reads_as_prose(elem: dict) -> bool:
    words = len(re.findall(r"[A-Za-z][A-Za-z'’-]*", _plain(elem)))
    if words >= RULES.prose_long_words:
        return True
    if words < RULES.prose_wrapped_words:
        return False
    rect = _safe_rect(elem)
    if rect is None:
        return False
    size = _fontsize(elem) or DEFAULT_FONTSIZE
    line_height = max(float(elem.get("line_height", 0.0) or 0.0), size * 1.2, 8.0)
    return rect.height > line_height * RULES.prose_wrap_lines


def _prose_shaped(elem: dict, dominant_fontsize: float) -> bool:
    """Classed ``body``, or reading as running prose, at any type size."""
    return bool(
        _is_text(elem)
        and _plain(elem)
        and (
            elem.get("layout_class") == LayoutClass.BODY
            or _reads_as_prose(elem)
            or _pdf_element_is_body_prose(elem, dominant_fontsize)
        )
    )


def _is_prose(
    elements: list[dict],
    index: int,
    neighbours,
    dominant_fontsize: float,
) -> bool:
    """Prose at any type size: prose-shaped and alone on its lines among
    ``neighbours`` (indices of nearby text).  Long text in a row of other
    cells is table text."""
    elem = elements[index]
    if not _prose_shaped(elem, dominant_fontsize):
        return False
    rect = _safe_rect(elem)
    if rect is None:
        return True
    for other in neighbours:
        if other == index:
            continue
        other_elem = elements[other]
        other_rect = _safe_rect(other_elem)
        if (
            other_rect is not None
            and _is_text(other_elem)
            and _plain(other_elem)
            and _same_line(rect, other_rect)
        ):
            return False
    return True


def _in_header_band(elem: dict, page_rect) -> bool:
    if page_rect is None:
        return False
    rect = _safe_rect(elem)
    height = float(getattr(page_rect, "height", 0.0) or 0.0)
    if rect is None or height <= 0.0:
        return False
    center_y = ((rect.y0 + rect.y1) / 2.0 - float(page_rect.y0)) / height
    return center_y <= RULES.header_band_share


def _note_lines(elements: list[dict]) -> set[int]:
    """Source and note lines, with the lines that continue them."""
    notes = {
        index
        for index, elem in enumerate(elements)
        if _is_text(elem) and _SOURCE_NOTE_RE.match(_plain(elem))
    }
    frontier = sorted(notes)
    while frontier:
        grown = []
        for note in frontier:
            note_elem = elements[note]
            note_rect = _safe_rect(note_elem)
            size = _fontsize(note_elem)
            if note_rect is None or size <= 0.0:
                continue
            for index, elem in enumerate(elements):
                if index in notes or not _is_text(elem) or not _plain(elem):
                    continue
                rect = _safe_rect(elem)
                if rect is None or abs(_fontsize(elem) - size) > 0.25:
                    continue
                gap = rect.y0 - note_rect.y1
                if (
                    abs(rect.x0 - note_rect.x0) <= 1.5
                    and -RULES.edge_tolerance <= gap <= size * RULES.note_continuation_gap_share
                ):
                    notes.add(index)
                    grown.append(index)
        frontier = grown
    return notes


def _never_member(
    elements: list[dict],
    index: int,
    page_rect,
    note_lines: set[int],
) -> bool:
    """Titles, captions, source/note lines, footnotes and page bands."""
    elem = elements[index]
    plain = _plain(elem)
    return bool(
        index in note_lines
        or _TITLE_RE.match(plain)
        or _SOURCE_NOTE_RE.match(plain)
        or elem.get("footnote_hint")
        or elem.get("reference_entry_hint")
        or _in_header_band(elem, page_rect)
        or _pdf_element_in_page_footer_band(elem, page_rect)
    )


def _text_indices(elements: list[dict]) -> list[int]:
    return [
        index
        for index, elem in enumerate(elements)
        if _is_text(elem) and _plain(elem) and _safe_rect(elem) is not None
    ]


def _table_region_members(
    elements: list[dict],
    seed: int,
    seed_rect: fitz.Rect,
    page_rect,
    dominant_fontsize: float,
) -> set[int] | None:
    """Member text of the smallest detected table region holding the seed."""
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
    note_lines = _note_lines(elements)
    for region in sorted(regions, key=lambda rect: (rect.width * rect.height, tuple(rect))):
        inside = [
            index
            for index, elem in enumerate(elements)
            if (rect := _safe_rect(elem)) is not None and region.contains(_center(rect))
        ]
        if any(elements[index].get("type") == ElementType.MERGED_AWAY for index in inside):
            continue
        texts = [index for index in inside if _is_text(elements[index]) and _plain(elements[index])]
        members = {
            index
            for index in texts
            if not _never_member(elements, index, page_rect, note_lines)
            and not _is_prose(elements, index, texts, dominant_fontsize)
        }
        if seed in members:
            return members
    return None


def _is_title(elem: dict, dominant_fontsize: float) -> bool:
    return (
        _is_text(elem)
        and bool(_TITLE_RE.match(_plain(elem)))
        and not _pdf_element_is_body_prose(elem, dominant_fontsize)
    )


def _title_span(
    title: int,
    title_rects: dict[int, fitz.Rect],
    page_rect: fitz.Rect,
) -> tuple[float, float]:
    """The page width a title may claim, split with titles in the same row.

    Titles side by side (no horizontal overlap) share a row when they overlap
    vertically or are staggered by at most one title height."""
    rect = title_rects[title]

    def same_row(other_rect: fitz.Rect) -> bool:
        if other_rect is rect:
            return True
        if _x_overlap(rect, other_rect) > 0:
            return False
        gap = max(other_rect.y0 - rect.y1, rect.y0 - other_rect.y1)
        return gap <= max(rect.height, other_rect.height) * RULES.staggered_title_share

    row = sorted(
        (other for other, other_rect in title_rects.items() if same_row(other_rect)),
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


def _figure_extent(
    core: tuple[float, float],
    top: float,
    bottom: float,
    span: tuple[float, float],
    graphics: list[fitz.Rect],
) -> tuple[float, float]:
    """The figure's horizontal extent: its title and the element, grown by
    every graphic inside the band that joins what is already covered."""
    tolerance = RULES.edge_tolerance
    left, right = core
    inside = [
        rect
        for rect in graphics
        if rect.y0 >= top - tolerance
        and rect.y1 <= bottom + tolerance
        and rect.x0 >= span[0] - tolerance
        and rect.x1 <= span[1] + tolerance
    ]
    changed = True
    while changed:
        changed = False
        for rect in inside:
            if min(rect.x1, right) - max(rect.x0, left) > 0 and (
                rect.x0 < left or rect.x1 > right
            ):
                left, right = min(left, rect.x0), max(right, rect.x1)
                changed = True
    return left, right


def _caption_band_members(
    elements: list[dict],
    seed: int,
    seed_rect: fitz.Rect,
    page_rect,
    dominant_fontsize: float,
    graphics: list[fitz.Rect],
) -> set[int] | None:
    """Member text of the seed's captioned figure, or ``None`` if undecidable."""
    if page_rect is None:
        return None
    page_rect = fitz.Rect(page_rect)
    tolerance = RULES.edge_tolerance
    texts = _text_indices(elements)
    note_lines = _note_lines(elements)
    rects = {index: _safe_rect(elem) for index, elem in enumerate(elements)}
    title_rects = {
        index: rect
        for index, rect in rects.items()
        if rect is not None and _is_title(elements[index], dominant_fontsize)
    }

    def frame_end(index: int, neighbours) -> bool:
        elem = elements[index]
        if elem.get("type") == ElementType.MERGED_AWAY:
            return True
        if index not in neighbours:
            return False
        return (
            _never_member(elements, index, page_rect, note_lines)
            or _fontsize(elem) >= dominant_fontsize * RULES.body_type_share
            or _is_prose(elements, index, neighbours, dominant_fontsize)
        )

    seed_center = _center(seed_rect)

    def band_for(title: int, upward: bool) -> set[int] | None:
        left, right = _title_span(title, title_rects, page_rect)
        if not left <= seed_center.x <= right:
            return None
        span = (left, right)
        title_rect = title_rects[title]
        in_span = [
            index for index in texts if left <= _center(rects[index]).x <= right
        ]
        # A frame end only has to reach into the span: body text across both
        # of two side-by-side figures closes both.
        reaching = [
            index for index in texts if min(rects[index].x1, right) - max(rects[index].x0, left) > 0
        ]
        merged = [
            index
            for index, elem in enumerate(elements)
            if elem.get("type") == ElementType.MERGED_AWAY and rects[index] is not None
        ]
        candidates = [index for index in [*reaching, *merged] if index != title]
        ends = [index for index in candidates if frame_end(index, reaching)]
        if seed in ends:
            return None
        core = (
            max(left, min(title_rect.x0, seed_rect.x0)),
            min(right, max(title_rect.x1, seed_rect.x1)),
        )
        extent = core
        if upward:
            edge, limit = title_rect.y0, None
        else:
            edge, limit = title_rect.y1, None
        # Close the band at the nearest frame end over the extent; the
        # extent grows with the graphics of the band it bounds, so repeat
        # until both stop changing (the band can only shrink).
        for _ in range(len(elements) + 2):
            if upward:
                closing = [
                    rects[index].y1
                    for index in ends
                    if rects[index].y1 <= edge + tolerance
                    and min(rects[index].x1, extent[1]) - max(rects[index].x0, extent[0]) > 0
                ]
                new_limit = max(closing) if closing else None
                if new_limit is None:
                    return None  # nothing frames the band
                top, bottom = new_limit, edge
            else:
                closing = [
                    rects[index].y0
                    for index in ends
                    if rects[index].y0 >= edge - tolerance
                    and min(rects[index].x1, extent[1]) - max(rects[index].x0, extent[0]) > 0
                ]
                new_limit = min(closing) if closing else None
                if new_limit is None:
                    return None  # nothing closes the band
                top, bottom = edge, new_limit
            new_extent = _figure_extent(core, top, bottom, span, graphics)
            if new_limit == limit and new_extent == extent:
                break
            limit, extent = new_limit, new_extent
        else:
            return None
        if not (top - tolerance <= seed_rect.y0 and seed_rect.y1 <= bottom + tolerance):
            return None
        band_texts = [
            index
            for index in in_span
            if index != title and top <= _center(rects[index]).y <= bottom
        ]
        if any(top <= _center(rects[index]).y <= bottom
               and extent[0] <= _center(rects[index]).x <= extent[1] for index in merged):
            return None
        members, outside = set(), []
        for index in band_texts:
            rect = rects[index]
            if index in ends:
                continue
            if rect.x0 >= extent[0] - tolerance and rect.x1 <= extent[1] + tolerance:
                members.add(index)
            elif rect.x1 <= extent[0] + tolerance or rect.x0 >= extent[1] - tolerance:
                outside.append(index)
            else:
                return None  # straddles the figure's extent
        for index in outside:
            # Running text beside the figure is another column, not figure
            # text; a short line on the figure's own lines could be either.
            if _prose_shaped(elements[index], dominant_fontsize):
                continue
            if any(_same_line(rects[index], rects[member]) for member in members):
                return None
        return members

    above = sorted(
        (index for index, rect in title_rects.items() if rect.y1 <= seed_rect.y0 + tolerance),
        key=lambda index: (-title_rects[index].y1, index),
    )
    for title in above:
        left, right = _title_span(title, title_rects, page_rect)
        if left <= seed_center.x <= right:
            members = band_for(title, upward=False)
            if members is not None:
                return members
            break

    below = sorted(
        (index for index, rect in title_rects.items() if rect.y0 >= seed_rect.y1 - tolerance),
        key=lambda index: (title_rects[index].y0, index),
    )
    for title in below:
        left, right = _title_span(title, title_rects, page_rect)
        if left <= seed_center.x <= right:
            return band_for(title, upward=True)
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


def element_is_prose(elements: list[dict], index: int, dominant_fontsize: float) -> bool:
    """Prose by the grouping rule's definition, judged against the whole page."""
    return _is_prose(elements, index, _text_indices(elements), dominant_fontsize)


def find_figure_table_group(
    elements: list[dict],
    element_index: int,
    *,
    page_rect,
    dominant_fontsize: float,
    reason: str = "",
    protected_indices: set[int] | frozenset[int] = frozenset(),
    kept_indices: set[int] | frozenset[int] = frozenset(),
    graphics=(),
) -> FigureTableGroup | None:
    """Return the figure or table holding ``element_index``, or ``None``.

    ``protected_indices`` (accepted cross-page merge endpoints) may never be
    kept as source by this narrowing.  ``kept_indices`` are elements an
    earlier group on the same page already keeps; a failure among them cannot
    be narrowed further, which bounds the caller's loop.  ``graphics`` are
    the page's visible vector drawing rectangles, evidence of a figure's
    extent beyond its title.
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
    graphic_rects = [
        fitz.Rect(rect) for rect in graphics or () if not fitz.Rect(rect).is_infinite
    ]
    graphic_rects.extend(
        rect
        for elem in elements
        if elem.get("type") in {ElementType.IMAGE, ElementType.TABLE_IMAGE}
        and (rect := _safe_rect(elem)) is not None
    )

    region = _table_region_members(elements, seed, seed_rect, page_rect, dominant_fontsize)
    band = _caption_band_members(
        elements, seed, seed_rect, page_rect, dominant_fontsize, graphic_rects
    )
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
    "ELEMENT_RESTORE",
    "PAGE_FAILURE",
    "RULES",
    "TABLE_REGION",
    "element_is_prose",
    "FigureTableGroup",
    "FigureTableGroupRules",
    "find_figure_table_group",
]
