"""Find genuine table regions among PyMuPDF table candidates, rejecting
body-text false positives and sparse layout wrappers."""

from __future__ import annotations

import logging
import re

import fitz

log = logging.getLogger("translator")

def _pdf_table_list(table_finder) -> list:
    tables = getattr(table_finder, "tables", table_finder)
    return list(tables or [])


def _is_compact_open_pdf_table(table, page_rect: fitz.Rect) -> bool:
    """Accept conservative text/line fallback detections.

    PyMuPDF's default table finder misses compact tables that have only top
    and bottom horizontal rules. Its text/line strategy can recover those,
    but also mistakes multi-column charts and body text for tall tables. The
    geometry below admits the compact, wide case while rejecting those false
    positives.
    """
    rect = fitz.Rect(table.bbox)
    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    row_count = int(getattr(table, "row_count", 0) or 0)
    col_count = int(getattr(table, "col_count", 0) or 0)

    # A hyperlink underline plus the footer rule can look like a one-row,
    # four-column open table even though only one inferred cell contains
    # text.  Such a rectangle may cover the second line of a footnote and
    # force one semantic sentence into separate translation units.  A real
    # multi-column table must populate at least two inferred columns.
    occupied_columns: set[int] = set()
    nonempty_cell_count = 0
    try:
        rows = list(table.extract() or [])
    except Exception:
        rows = []
    for row in rows:
        for column_index, cell in enumerate(row or []):
            if re.sub(r"\s+", " ", str(cell or "")).strip():
                nonempty_cell_count += 1
                occupied_columns.add(column_index)
    if rows and (
        nonempty_cell_count < 2
        or len(occupied_columns) < 2
    ):
        return False

    return (
        row_count >= 1
        and col_count >= 3
        and rect.width >= page_width * 0.42
        and rect.height >= 2.0
        and rect.height <= page_height * 0.28
        and rect.width / max(rect.height, 1.0) >= 2.4
    )


def _looks_like_pdf_body_text_false_table(table) -> bool:
    """Reject prose paragraphs that PyMuPDF mistakes for sparse tables.

    Fully justified body text can create recurring vertical whitespace bands.
    The default table finder may then report a many-column table even though
    almost every visual row contains only one long prose cell.  Treating that
    rectangle as a table forces one LLM request and one redraw per visual line.
    """
    row_count = int(getattr(table, "row_count", 0) or 0)
    col_count = int(getattr(table, "col_count", 0) or 0)
    if row_count < 3 or col_count < 3:
        return False
    try:
        rows = list(table.extract() or [])
    except Exception:
        return False
    if len(rows) < 3:
        return False

    nonempty_counts = []
    cell_texts = []
    for row in rows:
        nonempty = [
            re.sub(r"\s+", " ", str(cell or "")).strip()
            for cell in (row or [])
            if re.sub(r"\s+", " ", str(cell or "")).strip()
        ]
        nonempty_counts.append(len(nonempty))
        cell_texts.extend(nonempty)
    if not cell_texts:
        return False

    single_cell_rows = sum(count == 1 for count in nonempty_counts)
    long_prose_cells = sum(
        len(re.findall(r"[A-Za-z][A-Za-z'’-]*", text)) >= 8
        for text in cell_texts
    )
    numeric_cells = sum(
        bool(re.fullmatch(r"[\s()\[\]+\-−–—.,:%$€£¥0-9*]+", text))
        for text in cell_texts
    )
    return bool(
        single_cell_rows / len(rows) >= 0.70
        and long_prose_cells / len(cell_texts) >= 0.70
        and numeric_cells / len(cell_texts) <= 0.15
    )


def _looks_like_sparse_pdf_table_wrapper(outer, inner) -> bool:
    """Detect a sparse layout wrapper around a denser real table.

    Two-column reports can fool PyMuPDF into treating the page-bottom rule,
    column divider, body prose, and a neighboring chart as one giant 1x2
    table.  The genuine chart table is also returned as a denser nested
    candidate.  Keeping the wrapper marks ordinary prose as table content and
    suppresses the nested table, so prefer the structurally specific child.
    """
    if outer is inner:
        return False
    outer_rows = int(getattr(outer, "row_count", 0) or 0)
    outer_cols = int(getattr(outer, "col_count", 0) or 0)
    inner_rows = int(getattr(inner, "row_count", 0) or 0)
    inner_cols = int(getattr(inner, "col_count", 0) or 0)
    if (
        outer_rows > 2
        or outer_cols > 2
        or inner_rows < 2
        or inner_cols < 2
        or inner_rows * inner_cols < 6
    ):
        return False

    outer_rect = fitz.Rect(outer.bbox)
    inner_rect = fitz.Rect(inner.bbox)
    if outer_rect.is_empty or inner_rect.is_empty:
        return False
    intersection = outer_rect & inner_rect
    if (
        intersection.is_empty
        or intersection.get_area() / max(inner_rect.get_area(), 1.0) < 0.98
        or outer_rect.get_area() < inner_rect.get_area() * 1.15
    ):
        return False

    try:
        outer_rows_text = list(outer.extract() or [])
    except Exception:
        return False
    cell_texts = [
        re.sub(r"\s+", " ", str(cell or "")).strip()
        for row in outer_rows_text
        for cell in (row or [])
        if re.sub(r"\s+", " ", str(cell or "")).strip()
    ]
    word_counts = [
        len(re.findall(r"[A-Za-z][A-Za-z'’-]*", text))
        for text in cell_texts
    ]
    return bool(
        len(word_counts) >= 2
        and max(word_counts) >= 20
        and sum(word_counts) >= 40
    )


def _prune_sparse_pdf_table_wrappers(tables: list) -> list:
    """Keep nested real tables while dropping sparse page-layout wrappers."""
    return [
        table
        for table in tables
        if not any(
            _looks_like_sparse_pdf_table_wrapper(table, nested)
            for nested in tables
            if nested is not table
        )
    ]


def _has_compact_open_pdf_table_rules(page) -> bool:
    """Cheaply gate the slower text/line table finder by horizontal rules."""
    page_rect = fitz.Rect(page.rect)
    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    rule_parts = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return False

    for drawing in drawings:
        rect = fitz.Rect(drawing.get("rect", fitz.Rect()))
        if (
            rect.width >= max(page_width * 0.015, 8.0)
            and rect.height <= 3.5
        ):
            rule_parts.append(((rect.y0 + rect.y1) / 2.0, rect.x0, rect.x1))

    if not rule_parts:
        return False

    bands = []
    for y, x0, x1 in sorted(rule_parts):
        if not bands or y - bands[-1]["last_y"] > 3.5:
            bands.append({"ys": [y], "last_y": y, "intervals": [(x0, x1)]})
        else:
            bands[-1]["ys"].append(y)
            bands[-1]["last_y"] = y
            bands[-1]["intervals"].append((x0, x1))

    summaries = []
    for band in bands:
        intervals = sorted(band["intervals"])
        merged = []
        for x0, x1 in intervals:
            if not merged or x0 > merged[-1][1] + 2.0:
                merged.append([x0, x1])
            else:
                merged[-1][1] = max(merged[-1][1], x1)
        coverage = sum(max(x1 - x0, 0.0) for x0, x1 in merged)
        if coverage < page_width * 0.40:
            continue
        summaries.append({
            "y": sum(band["ys"]) / len(band["ys"]),
            "x0": min(x0 for x0, _ in merged),
            "x1": max(x1 for _, x1 in merged),
            "coverage": coverage,
        })

    for idx, top in enumerate(summaries):
        for bottom in summaries[idx + 1:]:
            gap = bottom["y"] - top["y"]
            overlap = max(0.0, min(top["x1"], bottom["x1"]) - max(top["x0"], bottom["x0"]))
            if (
                max(18.0, page_height * 0.025) <= gap <= page_height * 0.28
                and overlap >= page_width * 0.40
                and overlap / max(gap, 1.0) >= 2.4
            ):
                return True
    return False


def _pdf_table_cell_rects(table) -> list[fitz.Rect]:
    """Return stable, de-duplicated cell rectangles from a PyMuPDF table."""
    rects: list[fitz.Rect] = []
    for row in list(getattr(table, "rows", []) or []):
        for raw_cell in list(getattr(row, "cells", []) or []):
            if raw_cell is None:
                continue
            rect = fitz.Rect(raw_cell)
            if rect.is_empty:
                continue
            if any(
                abs(rect.x0 - existing.x0) <= 0.5
                and abs(rect.y0 - existing.y0) <= 0.5
                and abs(rect.x1 - existing.x1) <= 0.5
                and abs(rect.y1 - existing.y1) <= 0.5
                for existing in rects
            ):
                continue
            rects.append(rect)
    return rects


def _find_pdf_table_regions(page) -> list[dict]:
    """Detect tables while retaining their cell geometry for text recovery."""
    standard_candidates = [
        table
        for table in _pdf_table_list(page.find_tables())
        if not _looks_like_pdf_body_text_false_table(table)
    ]
    standard_tables = _prune_sparse_pdf_table_wrappers(standard_candidates)
    if len(standard_tables) < len(standard_candidates):
        log.info(
            "Page %s: ignored %s sparse table wrapper(s) around denser "
            "nested table geometry",
            page.number + 1,
            len(standard_candidates) - len(standard_tables),
        )
    tables = [(table, False) for table in standard_tables]

    if not standard_tables and _has_compact_open_pdf_table_rules(page):
        fallback_finder = page.find_tables(
            vertical_strategy="text",
            horizontal_strategy="lines",
        )
        fallback_tables = [
            table
            for table in _pdf_table_list(fallback_finder)
            if (
                _is_compact_open_pdf_table(table, page.rect)
                and not _looks_like_pdf_body_text_false_table(table)
            )
        ]
        if fallback_tables:
            log.info(
                "Page %s: detected %s compact open table(s) with text/line fallback",
                page.number + 1,
                len(fallback_tables),
            )
            tables.extend((table, True) for table in fallback_tables)

    regions = []
    for table, fallback_open in tables:
        rect = fitz.Rect(table.bbox)
        if any(
            (rect & existing["rect"]).get_area() >= rect.get_area() * 0.9
            for existing in regions
        ):
            continue
        regions.append({
            "rect": rect,
            "cells": _pdf_table_cell_rects(table),
            "fallback_open": bool(fallback_open),
            "row_count": int(getattr(table, "row_count", 0) or 0),
            "col_count": int(getattr(table, "col_count", 0) or 0),
        })
    return regions
