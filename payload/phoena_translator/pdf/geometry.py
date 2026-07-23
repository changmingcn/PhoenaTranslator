"""Geometry for deterministic PDF processing."""

from __future__ import annotations

import re

import fitz

from phoena_translator.pdf.types import (
    PDF_FORMULA_GUARD_PADDING,
)

def _pdf_rotation_from_direction(direction) -> int:
    """Map a native PDF text direction to PyMuPDF's textbox rotation.

    ``line['dir'] == (0, -1)`` is the common bottom-to-top chart-axis
    orientation.  Keeping the discrete rotation with the extracted element
    lets assembly redraw translated labels in the same direction instead of
    squeezing horizontal CJK into an eight-point-wide vertical rectangle.
    """
    try:
        dx, dy = float(direction[0]), float(direction[1])
    except (IndexError, TypeError, ValueError):
        return 0
    if abs(dy) > abs(dx):
        return 90 if dy < 0 else 270
    if dx < 0:
        return 180
    return 0


def _subtract_pdf_protected_rects(
    removal_rect,
    protected_rects: list,
    padding: float = PDF_FORMULA_GUARD_PADDING,
) -> list[fitz.Rect]:
    """Subtract protected formula geometry from a text-redaction rectangle."""
    pieces = [fitz.Rect(removal_rect)]
    for protected in protected_rects or []:
        guard = fitz.Rect(protected)
        guard = fitz.Rect(
            guard.x0 - padding,
            guard.y0 - padding,
            guard.x1 + padding,
            guard.y1 + padding,
        )
        next_pieces = []
        for piece in pieces:
            inter = piece & guard
            if inter.is_empty or inter.get_area() <= 0:
                next_pieces.append(piece)
                continue
            candidates = (
                fitz.Rect(piece.x0, piece.y0, piece.x1, inter.y0),
                fitz.Rect(piece.x0, inter.y1, piece.x1, piece.y1),
                fitz.Rect(piece.x0, inter.y0, inter.x0, inter.y1),
                fitz.Rect(inter.x1, inter.y0, piece.x1, inter.y1),
            )
            next_pieces.extend(
                candidate for candidate in candidates
                if not candidate.is_empty
                and candidate.width >= 0.2
                and candidate.height >= 0.2
            )
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def _pdf_rect_intersects_protected(
    candidate_rect,
    protected_rects: list,
    padding: float = PDF_FORMULA_GUARD_PADDING,
) -> bool:
    """Return whether a redraw rectangle enters immutable formula geometry."""
    candidate = fitz.Rect(candidate_rect)
    if candidate.is_empty:
        return False
    for protected in protected_rects or []:
        guard = fitz.Rect(protected)
        guard = fitz.Rect(
            guard.x0 - padding,
            guard.y0 - padding,
            guard.x1 + padding,
            guard.y1 + padding,
        )
        inter = candidate & guard
        if not inter.is_empty and inter.get_area() > 0:
            return True
    return False


def _safe_bbox_rect(value) -> fitz.Rect | None:
    """Convert an untrusted bbox value to a non-empty Rect, or ``None``.

    Element/paragraph metadata may legitimately lack geometry fields.  PyMuPDF
    does not raise a uniform exception type for malformed input (this build
    raises ``AssertionError`` for ``Rect(None)``), so the conversion must be
    guarded broadly instead of enumerating exception classes.
    """
    if value is None:
        return None
    try:
        rect = fitz.Rect(value)
    except Exception:
        return None
    return None if rect.is_empty else rect


def _get_pdf_source_ink_rects(elem: dict) -> list[fitz.Rect]:
    """Return each recorded source-glyph rectangle without filling gaps.

    A semantic sentence can occupy the right side of one visual row and the
    left side of the next while an unrelated formula occupies the remaining
    space.  Collapsing those source lines into one outer rectangle fabricates
    an overlap with the formula.  Preserve the recorded line/fragment
    rectangles so overlap checks and redaction operate on actual source ink.
    """
    source_rects: list[fitz.Rect] = []
    for paragraph in elem.get("paragraphs") or []:
        paragraph_rects: list[fitz.Rect] = []
        for line in paragraph.get("source_lines") or []:
            fragment_rects = [
                rect
                for fragment in line.get("same_baseline_fragments") or []
                if (rect := _safe_bbox_rect(fragment.get("bbox"))) is not None
            ]
            if fragment_rects:
                paragraph_rects.extend(fragment_rects)
                continue
            line_rect = _safe_bbox_rect(line.get("bbox"))
            if line_rect is not None:
                paragraph_rects.append(line_rect)

        if not paragraph_rects:
            paragraph_rects = [
                rect
                for bbox in paragraph.get("source_line_bboxes") or []
                if (rect := _safe_bbox_rect(bbox)) is not None
            ]

        if not paragraph_rects:
            source_rect = _safe_bbox_rect(paragraph.get("source_bbox"))
            if source_rect is not None:
                paragraph_rects.append(source_rect)
        source_rects.extend(paragraph_rects)

    return source_rects or [_get_pdf_elem_rect(elem)]


def _get_pdf_source_ink_rect(elem: dict) -> fitz.Rect:
    """Return the tight outer union of source glyph geometry.

    PDF text blocks can contain whitespace-only positioning objects far away
    from the visible glyphs.  Those objects must never enlarge a redaction or
    overlap-repair rectangle: doing so can erase unrelated table cells that
    merely share the same source block.
    """
    source_rects = _get_pdf_source_ink_rects(elem)
    result = fitz.Rect(source_rects[0])
    for rect in source_rects[1:]:
        result |= rect
    return result


def _pdf_elem_last_source_line_rect(elem: dict) -> fitz.Rect | None:
    """Tight bbox of the element's final visual source line, if recorded."""
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if not paragraphs:
        return None
    lines = [
        line
        for line in (paragraphs[-1].get("source_lines") or [])
        if line.get("bbox") and (line.get("plain") or "").strip()
    ]
    if not lines:
        return None
    return _safe_bbox_rect(lines[-1]["bbox"])


def _cluster_pdf_line_spans(span_entries: list[dict], fontsize: float) -> list[list[dict]]:
    if not span_entries:
        return []

    gap_threshold = max(fontsize * 5.0, 36.0)
    clusters = [[span_entries[0]]]
    for span in span_entries[1:]:
        prev = clusters[-1][-1]
        if span["x0"] - prev["x1"] > gap_threshold:
            clusters.append([span])
        else:
            clusters[-1].append(span)
    return clusters


def _split_pdf_line_layout_cells(clusters: list[list[dict]], line_bbox) -> list[dict]:
    if not clusters:
        return []

    line_x0 = float(line_bbox[0])
    line_x1 = float(line_bbox[2])
    cells = []
    for idx, cluster in enumerate(clusters):
        if idx == 0:
            cell_x0 = line_x0
        else:
            prev = clusters[idx - 1][-1]
            cell_x0 = (prev["x1"] + cluster[0]["x0"]) / 2.0

        if idx == len(clusters) - 1:
            cell_x1 = line_x1
        else:
            nxt = clusters[idx + 1][0]
            cell_x1 = (cluster[-1]["x1"] + nxt["x0"]) / 2.0

        if len(clusters) == 1:
            align = "left"
        elif idx == 0:
            align = "left"
        elif idx == len(clusters) - 1:
            align = "right"
        else:
            align = "center"

        cells.append({
            "cluster": cluster,
            "x0": float(cell_x0),
            "x1": float(cell_x1),
            "align": align,
        })
    return cells


def _detect_pdf_text_align(lines: list[dict], block_rect, fontsize: float) -> str:
    if not lines:
        return "left"

    # A single PDF text line normally has a tight block rectangle whose left,
    # right, and centre gaps are all zero.  That geometry contains no evidence
    # of alignment; treating it as centred was the reason ordinary prose in
    # documents such as the SEBI order and ar2026 was redrawn centred/right.
    # Keep the safe document default, except when the containing block itself
    # provides strong asymmetric whitespace (for example a page number placed
    # at the far right of a shared footer block).
    if len(lines) == 1:
        left_gap = max(0.0, float(lines[0]["x0"]) - float(block_rect.x0))
        right_gap = max(0.0, float(block_rect.x1) - float(lines[0]["x1"]))
        evidence_gap = max(fontsize * 2.0, 14.0)
        edge_gap = max(fontsize * 0.6, 5.0)
        if left_gap >= evidence_gap and right_gap <= edge_gap:
            return "right"
        return "left"

    left_gaps = [max(0.0, line["x0"] - block_rect.x0) for line in lines]
    right_gaps = [max(0.0, block_rect.x1 - line["x1"]) for line in lines]
    avg_left_gap = sum(left_gaps) / len(left_gaps)
    avg_right_gap = sum(right_gaps) / len(right_gaps)
    x0_spread = max(line["x0"] for line in lines) - min(line["x0"] for line in lines)
    x1_spread = max(line["x1"] for line in lines) - min(line["x1"] for line in lines)
    center_offset = max(
        abs(((line["x0"] + line["x1"]) / 2.0) - ((block_rect.x0 + block_rect.x1) / 2.0))
        for line in lines
    )
    short_lines = all(len(re.sub(r"\s+", " ", line["plain"]).strip()) <= 80 for line in lines)

    # Right-aligned author/byline blocks often look "centered" if we only inspect
    # the block midpoint. Prefer right alignment when right edges are much more
    # stable than left edges.
    if (
        short_lines
        and x1_spread <= max(fontsize * 0.7, 6.0)
        and x0_spread >= x1_spread + max(fontsize * 0.8, 6.0)
        and avg_right_gap <= max(fontsize * 0.45, 4.0)
    ):
        return "right"
    if (
        short_lines
        and center_offset <= max(fontsize * 1.5, 18.0)
        and abs(avg_left_gap - avg_right_gap) <= max(fontsize * 0.9, 10.0)
    ):
        return "center"
    if x0_spread <= max(fontsize * 0.7, 6.0) and avg_left_gap <= avg_right_gap + max(fontsize * 0.7, 6.0):
        return "left"
    if x1_spread <= max(fontsize * 0.7, 6.0) and avg_right_gap < avg_left_gap:
        return "right"
    return "left"


def _get_pdf_render_bbox(block_rect, page_rect, paragraphs: list[dict], fontsize: float) -> list[float]:
    if not paragraphs:
        return [block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1]

    plain_lengths = [len(re.sub(r"\s+", " ", p.get("plain", "")).strip()) for p in paragraphs]
    max_len = max(plain_lengths) if plain_lengths else 0
    aligns = {p.get("text_align", "left") for p in paragraphs}
    all_nowrap = all(bool(p.get("nowrap")) for p in paragraphs)

    rect = fitz.Rect(block_rect)
    if all_nowrap and max_len <= 32 and aligns == {"right"}:
        expand = min(page_rect.width * 0.28, max(120.0, fontsize * 12.0))
        rect.x0 = max(page_rect.x0 + 72.0, rect.x0 - expand)
    elif all_nowrap and max_len <= 36 and aligns == {"center"}:
        expand = min(page_rect.width * 0.18, max(72.0, fontsize * 6.0))
        rect.x0 = max(page_rect.x0 + 72.0, rect.x0 - expand)
        rect.x1 = min(page_rect.x1 - 72.0, rect.x1 + expand)

    return [rect.x0, rect.y0, rect.x1, rect.y1]


def _get_pdf_elem_rect(elem: dict) -> fitz.Rect:
    bbox = elem.get("bbox", elem.get("rect"))
    return fitz.Rect(bbox) if isinstance(bbox, list) else fitz.Rect(bbox)


def _line_overlaps_pdf_table_rects(line_bbox, table_rects: list[fitz.Rect]) -> bool:
    if not table_rects:
        return False

    line_rect = fitz.Rect(line_bbox)
    probe_rect = fitz.Rect(line_rect.x0 - 2.0, line_rect.y0 - 2.0, line_rect.x1 + 2.0, line_rect.y1 + 2.0)
    probe_area = max(probe_rect.width * probe_rect.height, 1.0)
    center_x = (probe_rect.x0 + probe_rect.x1) / 2.0
    center_y = (probe_rect.y0 + probe_rect.y1) / 2.0

    for rect in table_rects:
        expanded = fitz.Rect(rect.x0 - 12.0, rect.y0 - 12.0, rect.x1 + 12.0, rect.y1 + 12.0)
        inter = probe_rect & expanded
        if not inter.is_empty and (inter.width * inter.height) / probe_area >= 0.12:
            return True
        if expanded.y0 <= center_y <= expanded.y1 and expanded.x0 - 24.0 <= center_x <= expanded.x1 + 24.0:
            return True
    return False


def _get_pdf_primary_paragraph_rect(elem: dict) -> fitz.Rect:
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) == 1 and paragraphs[0].get("source_bbox"):
        return fitz.Rect(paragraphs[0]["source_bbox"])
    return _get_pdf_elem_rect(elem)
