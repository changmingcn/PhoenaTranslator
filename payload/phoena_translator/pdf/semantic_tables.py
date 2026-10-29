"""Semantic tables helpers for deterministic PDF processing."""

from __future__ import annotations

import logging

import fitz

from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
    _get_pdf_source_ink_rect,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text
from phoena_translator.pdf.text_shape import _join_pdf_line_fragments
from phoena_translator.pdf.toc import _parse_pdf_toc_leader
from phoena_translator.pdf.model import ElementType

log = logging.getLogger("translator")


def _merge_pdf_table_cell_group(
    group: list[dict],
    page_rect: fitz.Rect,
    *,
    cell_rect: fitz.Rect | None = None,
    glossary_cell_hint: bool = False,
    semantic_native_table_cell: bool = False,
) -> dict:
    """Coalesce visual fragments belonging to one semantic table cell."""
    ordered = sorted(
        group,
        key=lambda elem: (
            _get_pdf_elem_rect(elem).y0,
            _get_pdf_elem_rect(elem).x0,
        ),
    )
    merged = dict(ordered[0])
    source_rect = _get_pdf_elem_rect(ordered[0])
    for elem in ordered[1:]:
        source_rect |= _get_pdf_elem_rect(elem)
    rect = fitz.Rect(cell_rect) if cell_rect is not None else fitz.Rect(source_rect)
    if rect.is_empty:
        rect = fitz.Rect(source_rect)

    plain_parts = [
        _collapsed_plain_text(elem.get("content", ""))
        for elem in ordered
    ]
    plain_parts = [part for part in plain_parts if part]
    rich_parts = [
        (elem.get("rich_content") or elem.get("content", "")).strip()
        for elem in ordered
        if (elem.get("rich_content") or elem.get("content", "")).strip()
    ]
    plain = _join_pdf_line_fragments(plain_parts)
    rich = _join_pdf_line_fragments(rich_parts)
    source_lines = []
    for elem in ordered:
        paragraphs = [
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ]
        appended = False
        for paragraph in paragraphs:
            for line in paragraph.get("source_lines") or []:
                if not (line.get("plain") or "").strip():
                    continue
                source_lines.append(dict(line))
                appended = True
        if appended:
            continue
        elem_rect = _get_pdf_source_ink_rect(elem)
        source_lines.append(
            {
                "plain": elem.get("content", ""),
                "rich": elem.get("rich_content") or elem.get("content", ""),
                "bbox": [float(value) for value in elem_rect],
            }
        )
    paragraph = {
        "plain": plain,
        "rich": rich or plain,
        "source_bbox": [float(value) for value in source_rect],
        "source_line_bboxes": [
            [float(value) for value in line["bbox"]] for line in source_lines
        ],
        "source_lines": source_lines,
        "margin_left": 0.0,
        "text_indent": 0.0,
        "gap_before": 0.0,
        "text_align": "left",
        "nowrap": None,
        "first_line_indent": False,
        "toc_leader": _parse_pdf_toc_leader(plain),
    }
    merged.update(
        {
            "y": float(rect.y0),
            "x": float(rect.x0),
            "rect": fitz.Rect(rect),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "render_bbox": _get_pdf_render_bbox(
                rect,
                page_rect,
                [paragraph],
                float(merged.get("fontsize", 9.0)),
            ),
            "content": plain,
            "rich_content": rich
            if any(elem.get("rich_content") for elem in ordered)
            else None,
            "paragraphs": [paragraph],
            "superscript_runs": [
                run for elem in ordered for run in (elem.get("superscript_runs") or [])
            ],
            "top_padding": min(
                (float(elem.get("top_padding", 0.0)) for elem in ordered),
                default=0.0,
            ),
            "line_height": max(
                (float(elem.get("line_height", 0.0)) for elem in ordered),
                default=float(merged.get("fontsize", 9.0)) * 1.05,
            ),
            "bold": all(bool(elem.get("bold")) for elem in ordered),
            "preserve_source_style": True,
            "table_hint": all(bool(elem.get("table_hint")) for elem in ordered),
            "glossary_cell_hint": bool(glossary_cell_hint),
            "semantic_native_table_cell": bool(semantic_native_table_cell),
            "semantic_table_cell": len(ordered) > 1,
            "merged_visual_line_count": sum(
                max(int(elem.get("merged_visual_line_count", 1)), 1) for elem in ordered
            ),
            "source_line_bboxes": [
                [float(value) for value in _get_pdf_elem_rect(elem)] for elem in ordered
            ],
            "inline_math_fragments": [
                dict(fragment)
                for elem in ordered
                for fragment in (elem.get("inline_math_fragments") or [])
            ],
        }
    )
    return merged


def _pdf_table_visual_line_bands(
    candidates: list[tuple[int, dict]],
) -> list[list[tuple[int, dict]]]:
    """Cluster table text elements that share one visual baseline."""
    bands: list[list[tuple[int, dict]]] = []
    for item in sorted(
        candidates,
        key=lambda pair: (
            _get_pdf_elem_rect(pair[1]).y0,
            _get_pdf_elem_rect(pair[1]).x0,
        ),
    ):
        rect = _get_pdf_elem_rect(item[1])
        fontsize = max(float(item[1].get("fontsize", 9.0)), 1.0)
        if bands:
            previous_y = sum(
                _get_pdf_elem_rect(previous[1]).y0 for previous in bands[-1]
            ) / len(bands[-1])
            if abs(rect.y0 - previous_y) <= max(2.5, fontsize * 0.42):
                bands[-1].append(item)
                continue
        bands.append([item])
    return bands


def _pdf_compact_open_table_cell_groups(
    candidates: list[tuple[int, dict]],
) -> list[list[tuple[int, dict]]]:
    """Split a coarse open-table cell into compact visual-line clusters.

    Text/line fallback tables can recover accurate column and header bounds,
    but a table without internal horizontal rules may expose the whole data
    body as one tall cell.  Consecutive wrapped header lines have a normal
    line pitch; distinct statistical rows have a larger pitch.  Preserve that
    distinction instead of turning an entire numeric column into one request.
    """
    bands = _pdf_table_visual_line_bands(candidates)
    if not bands:
        return []

    font_sizes = [max(float(elem.get("fontsize", 9.0)), 1.0) for _, elem in candidates]
    dominant_fontsize = sorted(font_sizes)[len(font_sizes) // 2]
    compact_pitch_limit = max(4.5, dominant_fontsize * 1.20)
    grouped_bands: list[list[list[tuple[int, dict]]]] = [[bands[0]]]
    previous_y = sum(_get_pdf_elem_rect(elem).y0 for _, elem in bands[0]) / len(
        bands[0]
    )
    for band in bands[1:]:
        current_y = sum(_get_pdf_elem_rect(elem).y0 for _, elem in band) / len(band)
        if current_y - previous_y <= compact_pitch_limit:
            grouped_bands[-1].append(band)
        else:
            grouped_bands.append([band])
        previous_y = current_y

    return [
        [item for band in band_group for item in band] for band_group in grouped_bands
    ]


def _pdf_semantic_table_region_candidates(
    elements: list[dict],
    table_rect: fitz.Rect,
    claimed: set[int],
) -> list[tuple[int, dict]]:
    """Select unclaimed horizontal text elements centered in one table."""
    expanded = fitz.Rect(
        table_rect.x0 - 4.0,
        table_rect.y0 - 4.0,
        table_rect.x1 + 4.0,
        table_rect.y1 + 4.0,
    )
    candidates = []
    for index, elem in enumerate(elements):
        if (
            index in claimed
            or elem.get("type") != ElementType.TEXT
            or not elem.get("table_hint")
            or elem.get("non_horizontal")
        ):
            continue
        rect = _get_pdf_elem_rect(elem)
        center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
        if center in expanded:
            candidates.append((index, elem))
    return candidates


def _pdf_semantic_table_cell_candidates(
    candidates: list[tuple[int, dict]],
    cell_rect: fitz.Rect,
    claimed: set[int],
) -> list[tuple[int, dict]]:
    """Select candidate fragments centered in one detected fallback cell."""
    expanded_cell = fitz.Rect(
        cell_rect.x0 - 2.0,
        cell_rect.y0 - 2.0,
        cell_rect.x1 + 2.0,
        cell_rect.y1 + 2.0,
    )
    cell_candidates = []
    for item in candidates:
        index, elem = item
        if index in claimed:
            continue
        rect = _get_pdf_elem_rect(elem)
        center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
        if center in expanded_cell:
            cell_candidates.append(item)
    return cell_candidates


def _pdf_semantic_table_group_rect(group: list[dict]) -> fitz.Rect:
    """Return the union of the source rectangles for a non-empty group."""
    group_source_rect = _get_pdf_elem_rect(group[0])
    for elem in group[1:]:
        group_source_rect |= _get_pdf_elem_rect(elem)
    return group_source_rect


def _record_pdf_semantic_table_cell_merge(
    group_items: list[tuple[int, dict]],
    page_rect: fitz.Rect,
    cell_rect: fitz.Rect,
    replacements: dict[int, dict],
    removed: set[int],
    claimed: set[int],
    *,
    reject_claimed: bool,
    semantic_native_table_cell: bool = False,
) -> bool:
    """Apply one non-overlapping semantic-cell merge to the result ledger."""
    if len(group_items) < 2:
        return False
    ordered_items = sorted(
        group_items,
        key=lambda item: (
            _get_pdf_elem_rect(item[1]).y0,
            _get_pdf_elem_rect(item[1]).x0,
        ),
    )
    group_indices = [index for index, _ in ordered_items]
    if any(
        (reject_claimed and index in claimed)
        or index in removed
        or index in replacements
        for index in group_indices
    ):
        return False
    group = [elem for _, elem in ordered_items]
    replacements[group_indices[0]] = _merge_pdf_table_cell_group(
        group,
        page_rect,
        cell_rect=cell_rect,
        semantic_native_table_cell=semantic_native_table_cell,
    )
    removed.update(group_indices[1:])
    claimed.update(group_indices)
    return True


def _merge_pdf_exact_table_cells(
    region: dict,
    table_rect: fitz.Rect,
    candidates: list[tuple[int, dict]],
    page_rect: fitz.Rect,
    replacements: dict[int, dict],
    removed: set[int],
    claimed: set[int],
) -> int:
    """Merge fragments inside deterministic detected table-cell bounds.

    Native table detection exposes exact row/cell rectangles too.  Ignoring
    those rectangles and relying only on visual-line pitch can merge the last
    row of one table section, the next ``(ii)`` section header, and the first
    row of that section into one element.  The paragraph-boundary invariant
    then fails and the recovery policy preserves the entire page in English.
    Exact native cells are authoritative, so keep all of their wrapped lines
    together; only coarse open-table fallback cells need pitch-based splitting.
    """
    fallback_open = bool(region.get("fallback_open"))
    raw_cells = region.get("cells") or []
    if not fallback_open:
        row_count = int(region.get("row_count", 0) or 0)
        col_count = int(region.get("col_count", 0) or 0)
        # Sparse native grids commonly come from charts or merged-cell
        # layouts.  Their detector rectangles are not authoritative enough
        # to replace the established pitch-based inference.  A complete
        # rectangular grid is the narrow case where every row boundary is
        # deterministic and exact-cell grouping prevents cross-row merges.
        if (
            row_count < 1
            or col_count < 1
            or len(raw_cells) != row_count * col_count
        ):
            return 0

    merge_count = 0
    for raw_cell_rect in raw_cells:
        cell_rect = fitz.Rect(raw_cell_rect)
        if cell_rect.is_empty:
            continue
        cell_candidates = _pdf_semantic_table_cell_candidates(
            candidates,
            cell_rect,
            claimed,
        )
        if len(cell_candidates) < 2:
            continue

        is_open_header_cell = bool(
            fallback_open
            and int(region.get("row_count", 0) or 0) >= 2
            and abs(cell_rect.y0 - table_rect.y0) <= 2.5
            and cell_rect.y1 < table_rect.y1 - 2.5
        )
        if not fallback_open or is_open_header_cell:
            # A detected top row is a real header cell.  Its lines may be
            # vertically staggered relative to neighbouring columns, so keep
            # the whole cell even when one pitch is slightly larger.  Native
            # detector cells have exact row bounds and are equally safe.
            cell_groups = [cell_candidates]
        else:
            cell_groups = _pdf_compact_open_table_cell_groups(cell_candidates)

        for group_items in cell_groups:
            if len(group_items) < 2:
                continue
            group = [elem for _, elem in group_items]
            group_source_rect = _pdf_semantic_table_group_rect(group)
            dominant_fontsize = sorted(
                max(float(elem.get("fontsize", 9.0)), 1.0) for elem in group
            )[len(group) // 2]
            render_cell_rect = fitz.Rect(
                min(cell_rect.x0, group_source_rect.x0),
                group_source_rect.y0,
                max(
                    cell_rect.x1 - max(2.0, dominant_fontsize * 0.30),
                    group_source_rect.x1,
                ),
                max(
                    cell_rect.y1 - max(2.0, dominant_fontsize * 0.25),
                    group_source_rect.y1,
                ),
            )
            merge_count += int(
                _record_pdf_semantic_table_cell_merge(
                    group_items,
                    page_rect,
                    render_cell_rect,
                    replacements,
                    removed,
                    claimed,
                    reject_claimed=True,
                    semantic_native_table_cell=not fallback_open,
                )
            )
    return merge_count


def _pdf_semantic_table_row_plan(
    candidates: list[tuple[int, dict]],
) -> (
    tuple[
        list[list[tuple[int, dict]]],
        list[float],
        list[int],
        float,
    ]
    | None
):
    """Infer logical row boundaries from visual-line pitch."""
    bands = _pdf_table_visual_line_bands(candidates)
    if len(bands) < 2:
        return None

    band_y = [
        sum(_get_pdf_elem_rect(elem).y0 for _, elem in band) / len(band)
        for band in bands
    ]
    gaps = [
        band_y[position + 1] - band_y[position]
        for position in range(len(band_y) - 1)
        if band_y[position + 1] > band_y[position]
    ]
    if not gaps:
        return None
    sorted_gaps = sorted(gaps)
    lower_count = max(1, len(sorted_gaps) // 2)
    compact_pitch = sorted_gaps[lower_count // 2]
    font_sizes = [max(float(elem.get("fontsize", 9.0)), 1.0) for _, elem in candidates]
    dominant_fontsize = sorted(font_sizes)[len(font_sizes) // 2]
    # A genuine continuation line is bounded by normal text leading.  Without
    # a distinct gap, each visual band remains an independent logical row.
    compact_pitch = min(compact_pitch, dominant_fontsize * 1.40)
    row_gap_threshold = max(
        compact_pitch * 1.35,
        compact_pitch + dominant_fontsize * 0.30,
    )
    if any(gap > row_gap_threshold for gap in gaps):
        row_start_positions = [0] + [
            position + 1 for position, gap in enumerate(gaps) if gap > row_gap_threshold
        ]
    else:
        row_start_positions = list(range(len(bands)))
    return bands, band_y, row_start_positions, dominant_fontsize


def _pdf_semantic_table_row_anchors(
    first_band: list[tuple[int, dict]],
    dominant_fontsize: float,
) -> list[float]:
    """Derive stable column origins from the first visual line of a row."""
    anchors: list[float] = []
    for _, elem in sorted(
        first_band,
        key=lambda pair: _get_pdf_elem_rect(pair[1]).x0,
    ):
        x0 = float(_get_pdf_elem_rect(elem).x0)
        if not anchors or x0 - anchors[-1] > max(12.0, dominant_fontsize * 1.4):
            anchors.append(x0)
    return anchors


def _pdf_semantic_table_anchor_groups(
    row_items: list[tuple[int, dict]],
    anchors: list[float],
    dominant_fontsize: float,
) -> dict[int, list[tuple[int, dict]]]:
    """Assign vertical continuations to the nearest compatible column."""
    by_anchor: dict[int, list[tuple[int, dict]]] = {
        position: [] for position in range(len(anchors))
    }
    for item in row_items:
        x0 = float(_get_pdf_elem_rect(item[1]).x0)
        anchor_position = min(
            range(len(anchors)),
            key=lambda position: abs(x0 - anchors[position]),
        )
        # Multi-level headers can put unrelated labels on adjacent baselines.
        # Only continuations tied to essentially the same origin may merge.
        if abs(x0 - anchors[anchor_position]) > max(
            24.0,
            dominant_fontsize * 2.8,
        ):
            continue
        by_anchor[anchor_position].append(item)
    return by_anchor


def _pdf_semantic_table_inferred_cell_rect(
    group: list[dict],
    anchors: list[float],
    anchor_position: int,
    dominant_fontsize: float,
    table_rect: fitz.Rect,
    next_row_y: float,
) -> fitz.Rect:
    """Calculate the render bounds for one heuristically inferred cell."""
    group_source_rect = _pdf_semantic_table_group_rect(group)
    cell_x0 = min(anchors[anchor_position], group_source_rect.x0)
    if anchor_position + 1 < len(anchors):
        cell_x1 = anchors[anchor_position + 1] - max(
            3.0,
            dominant_fontsize * 0.65,
        )
    else:
        cell_x1 = table_rect.x1 - max(2.0, dominant_fontsize * 0.30)
    cell_y1 = min(
        table_rect.y1,
        next_row_y - max(2.0, dominant_fontsize * 0.25),
    )
    return fitz.Rect(
        cell_x0,
        group_source_rect.y0,
        max(cell_x1, group_source_rect.x1),
        max(cell_y1, group_source_rect.y1),
    )


def _merge_pdf_inferred_table_cells(
    candidates: list[tuple[int, dict]],
    table_rect: fitz.Rect,
    page_rect: fitz.Rect,
    replacements: dict[int, dict],
    removed: set[int],
    claimed: set[int],
) -> int:
    """Merge wrapped cells inferred from row pitch and column origins."""
    row_plan = _pdf_semantic_table_row_plan(candidates)
    if row_plan is None:
        return 0
    bands, band_y, row_start_positions, dominant_fontsize = row_plan
    merge_count = 0

    for row_number, start_position in enumerate(row_start_positions):
        end_position = (
            row_start_positions[row_number + 1]
            if row_number + 1 < len(row_start_positions)
            else len(bands)
        )
        if end_position - start_position < 2:
            continue
        anchors = _pdf_semantic_table_row_anchors(
            bands[start_position],
            dominant_fontsize,
        )
        if len(anchors) < 2:
            continue

        row_items = [
            item for band in bands[start_position:end_position] for item in band
        ]
        by_anchor = _pdf_semantic_table_anchor_groups(
            row_items,
            anchors,
            dominant_fontsize,
        )
        next_row_y = (
            band_y[row_start_positions[row_number + 1]]
            if row_number + 1 < len(row_start_positions)
            else table_rect.y1
        )
        for anchor_position, group_items in by_anchor.items():
            if len(group_items) < 2:
                continue
            group = [elem for _, elem in group_items]
            cell_rect = _pdf_semantic_table_inferred_cell_rect(
                group,
                anchors,
                anchor_position,
                dominant_fontsize,
                table_rect,
                next_row_y,
            )
            merge_count += int(
                _record_pdf_semantic_table_cell_merge(
                    group_items,
                    page_rect,
                    cell_rect,
                    replacements,
                    removed,
                    claimed,
                    reject_claimed=False,
                )
            )
    return merge_count


def _apply_pdf_semantic_table_replacements(
    elements: list[dict],
    replacements: dict[int, dict],
    removed: set[int],
) -> list[dict]:
    """Materialize and restore reading order for the table merge ledger."""
    merged_elements = [
        replacements.get(index, elem)
        for index, elem in enumerate(elements)
        if index not in removed
    ]
    merged_elements.sort(
        key=lambda elem: (
            float(elem.get("y", _get_pdf_elem_rect(elem).y0)),
            float(elem.get("x", _get_pdf_elem_rect(elem).x0)),
        )
    )
    return merged_elements


def _merge_pdf_semantic_table_cells(
    elements: list[dict],
    table_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
    table_regions: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Restore wrapped table cells before translation.

    Recover deterministic candidate groups from exact fallback cells or from
    compact row pitch and column anchors, then apply each non-overlapping merge.
    """
    if not table_rects:
        return elements, 0

    replacements: dict[int, dict] = {}
    removed: set[int] = set()
    claimed: set[int] = set()
    merge_count = 0
    regions = table_regions or [
        {
            "rect": fitz.Rect(raw_table_rect),
            "cells": [],
            "fallback_open": False,
        }
        for raw_table_rect in table_rects
    ]

    for region in regions:
        table_rect = fitz.Rect(region.get("rect", fitz.Rect()))
        if table_rect.is_empty:
            continue
        candidates = _pdf_semantic_table_region_candidates(
            elements,
            table_rect,
            claimed,
        )
        merge_count += _merge_pdf_exact_table_cells(
            region,
            table_rect,
            candidates,
            page_rect,
            replacements,
            removed,
            claimed,
        )
        if region.get("fallback_open"):
            # Exact fallback columns are already compact-clustered; the global
            # heuristic would recombine deliberately separated statistical rows.
            continue
        unclaimed_candidates = [item for item in candidates if item[0] not in claimed]
        merge_count += _merge_pdf_inferred_table_cells(
            unclaimed_candidates,
            table_rect,
            page_rect,
            replacements,
            removed,
            claimed,
        )

    if not replacements:
        return elements, 0
    return (
        _apply_pdf_semantic_table_replacements(elements, replacements, removed),
        merge_count,
    )


