"""Extraction for deterministic PDF processing."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import subprocess

import fitz

from phoena_translator.pdf.types import (
    PDF_VECTOR_OCR_DPI,
    PDF_VECTOR_OCR_MIN_ALPHA_WORDS,
    SKIP_SECTION_HEADING_RE,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_source_ink_rect,
)
from phoena_translator.pdf.math_detection import (
    _plain_text,
)

log = logging.getLogger("translator")

def _is_skip_page(page_text: str) -> bool:
    """Check if page is a bibliography/references/index section heading page.
    Only matches if:
    - The keyword appears in one of the first 3 short lines (headings)
    - The page has few content lines (<=10), to avoid false positives on
      table-of-contents pages or body text pages that mention these words."""
    lines = [l.strip() for l in page_text.strip().splitlines() if l.strip()]
    # If page has many lines, it's likely a TOC or body text page, not a section start
    if len(lines) > 10:
        return False
    for line in lines[:3]:
        if len(line) > 80:
            continue
        # A short chart page can have only a handful of extracted text lines.
        # Searching anywhere in those lines made titles such as "European
        # index futures" look like an Index section and silently skipped the
        # whole page.  A section boundary must itself be a heading, optionally
        # with numbering or a conventional "further reading" suffix.
        normalized_heading = re.sub(r"\s+", " ", line).strip(" \t\r\n-–—:;")
        if SKIP_SECTION_HEADING_RE.fullmatch(normalized_heading):
            return True
    return False


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
    standard_tables = [
        table
        for table in _pdf_table_list(page.find_tables())
        if not _looks_like_pdf_body_text_false_table(table)
    ]
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


def _find_pdf_table_rects(page) -> list[fitz.Rect]:
    """Compatibility wrapper for callers that only need table bounds."""
    return [
        fitz.Rect(region["rect"])
        for region in _find_pdf_table_regions(page)
    ]


def _looks_like_pdf_hidden_text_artifact_span(
    span: dict,
    page_rect: fitz.Rect,
) -> bool:
    """Detect an impossible oversized span hidden by clipping/content order."""
    text = span.get("text") or ""
    letters = sum(1 for char in text if char.isalpha())
    size = float(span.get("size", 0.0))
    if letters < 40 or size < max(48.0, page_rect.height * 0.055):
        return False

    span_rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
    if span_rect.is_empty:
        return False
    estimated_condensed_width = size * letters * 0.11
    horizontally_impossible = (
        estimated_condensed_width > max(span_rect.width, 1.0) * 6.0
    )
    materially_outside_page = (
        span_rect.y0 < page_rect.y0 - size * 0.20
        or span_rect.y1 > page_rect.y1 + size * 0.20
        or span_rect.x0 < page_rect.x0 - size * 0.20
        or span_rect.x1 > page_rect.x1 + size * 0.20
    )
    implausibly_large = size > page_rect.height * 0.10
    return horizontally_impossible and (
        materially_outside_page or implausibly_large
    )


def _sanitize_pdf_hidden_text_artifact_spans(
    block: dict,
    page_rect: fitz.Rect,
) -> tuple[dict | None, int]:
    """Remove only impossible spans while retaining legitimate peers/lines."""
    if block.get("type") != 0:
        return block, 0

    kept_lines = []
    removed = 0
    for line in block.get("lines", []):
        kept_spans = []
        for span in line.get("spans", []):
            if _looks_like_pdf_hidden_text_artifact_span(span, page_rect):
                removed += 1
                continue
            kept_spans.append(span)
        if not kept_spans:
            continue
        line_copy = dict(line)
        line_copy["spans"] = kept_spans
        line_rect = fitz.Rect(kept_spans[0].get("bbox", (0, 0, 0, 0)))
        for span in kept_spans[1:]:
            line_rect |= fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
        line_copy["bbox"] = tuple(line_rect)
        kept_lines.append(line_copy)

    if not kept_lines:
        return None, removed
    if not removed:
        return block, 0

    block_copy = dict(block)
    block_copy["lines"] = kept_lines
    block_rect = fitz.Rect(kept_lines[0]["bbox"])
    for line in kept_lines[1:]:
        block_rect |= fitz.Rect(line["bbox"])
    block_copy["bbox"] = tuple(block_rect)
    return block_copy, removed


def _mark_pdf_embedded_thumbnail_text_elements(
    elements: list[dict],
    image_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
) -> int:
    """Preserve tiny vector text printed inside a decorative image thumbnail.

    Some reports place a miniature copy of their cover in a diagram. The
    cover artwork is a small raster image, while its one- to three-point title
    is repeated as separate vector text objects over that image. Translating
    those objects independently makes several normal-sized Chinese strings
    overlap on top of the thumbnail. Keep the source artwork intact when at
    least two tiny text objects are almost wholly contained by the same small
    image. Full-page scans and ordinary image captions deliberately do not
    meet these size gates.
    """
    page_area = max(page_rect.get_area(), 1.0)
    thumbnail_rects: list[fitz.Rect] = []
    for raw_rect in image_rects or []:
        rect = fitz.Rect(raw_rect)
        if rect.is_empty:
            continue
        area_ratio = rect.get_area() / page_area
        if not (
            0.0005 <= area_ratio <= 0.04
            and rect.width <= page_rect.width * 0.25
            and rect.height <= page_rect.height * 0.25
        ):
            continue
        if not any(
            all(abs(a - b) <= 0.25 for a, b in zip(rect, existing))
            for existing in thumbnail_rects
        ):
            thumbnail_rects.append(rect)

    marked = 0
    for image_rect in thumbnail_rects:
        contained: list[dict] = []
        for element in elements or []:
            if (
                element.get("type") != "text"
                or element.get("skip_translate_reason")
                or float(element.get("fontsize", 0.0)) > 4.0
            ):
                continue
            text_rect = _get_pdf_source_ink_rect(element)
            if text_rect.is_empty:
                continue
            intersection = text_rect & image_rect
            if (
                intersection.is_empty
                or intersection.get_area() / max(text_rect.get_area(), 1.0) < 0.97
            ):
                continue
            contained.append(element)

        if len(contained) < 2:
            continue
        for element in contained:
            element["skip_translate_reason"] = "embedded_thumbnail_text"
            marked += 1
    return marked


def _pdf_vector_ocr_normalized_match_text(text: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", _plain_text(text or "").casefold())
    )


def _normalize_pdf_vector_ocr_text(text: str) -> str:
    """Repair only high-confidence OCR confusions in diagram control labels."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    normalized = re.sub(r"\bNote\s*[|Il]\s*:", "Note 1:", normalized, flags=re.I)
    normalized = re.sub(
        r"\b(WR)\s*:\s*([0-9Il]+)\s*:",
        lambda match: f"{match.group(1).upper()}{match.group(2).translate(str.maketrans({'I': '1', 'l': '1'}))}:",
        normalized,
        flags=re.I,
    )

    def _repair_report_code(match: re.Match) -> str:
        prefix = match.group(1).upper()
        suffix = re.sub(r"\s+", "", match.group(2)).translate(
            str.maketrans({"I": "1", "i": "1", "l": "1"})
        )
        return f"{prefix}{suffix}:"

    normalized = re.sub(
        r"\b(EIA|SR|WR|IN|SC)\s*([0-9Il ]{1,4})\s*:",
        _repair_report_code,
        normalized,
        flags=re.I,
    )
    normalized = re.sub(r"\bI(?=\d{3}s\b)", "1", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"(?:—|–)\s*-", "—", normalized)
    return normalized.strip(" |")


def _join_pdf_vector_ocr_lines(lines: list[dict]) -> str:
    joined = ""
    for line in lines:
        part = _normalize_pdf_vector_ocr_text(line.get("text", ""))
        if not part:
            continue
        if joined.endswith("-") and re.match(r"^[a-z]", part):
            joined += part
        else:
            joined += (" " if joined else "") + part
    return re.sub(r"\s+", " ", joined).strip()


def _pdf_vector_ocr_line_is_artifact(line: dict) -> bool:
    text = _normalize_pdf_vector_ocr_text(line.get("text", ""))
    alpha = re.sub(r"[^A-Za-z]", "", text)
    if not re.search(r"[A-Za-z0-9]", text):
        return True
    if len(alpha) <= 1 and not re.search(r"\d", text):
        return True
    return text.casefold() in {"vv", "zn"}


def _pdf_vector_ocr_existing_line_match(line: dict, elements: list[dict]) -> bool:
    line_text = line.get("text", "")
    candidate = _pdf_vector_ocr_normalized_match_text(line_text)
    if len(candidate) < 4:
        return False
    line_rect = fitz.Rect(line.get("bbox", (0, 0, 0, 0)))
    for elem in elements or []:
        if elem.get("type") != "text":
            continue
        elem_rect = _get_pdf_source_ink_rect(elem)
        intersection = line_rect & elem_rect
        centers_close = (
            abs((line_rect.x0 + line_rect.x1) - (elem_rect.x0 + elem_rect.x1)) <= 16.0
            and abs((line_rect.y0 + line_rect.y1) - (elem_rect.y0 + elem_rect.y1)) <= 12.0
        )
        if (
            intersection.is_empty
            or intersection.get_area() / max(line_rect.get_area(), 1.0) < 0.20
        ) and not centers_close:
            continue
        existing = _pdf_vector_ocr_normalized_match_text(elem.get("content", ""))
        if not existing:
            continue
        if candidate in existing:
            return True
        if existing in candidate and len(existing) / max(len(candidate), 1) >= 0.82:
            return True
    return False


def _pdf_vector_ocr_parse_tsv(tsv_text: str, scale: float) -> list[dict]:
    words_by_line: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in (tsv_text or "").splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) < 12 or fields[0] != "5":
            continue
        text = fields[11].strip()
        if not text:
            continue
        try:
            confidence = float(fields[10])
            left, top, width, height = map(float, fields[6:10])
        except (TypeError, ValueError):
            continue
        # Outline fonts often give the short bold control code (``EIA11``)
        # confidence 0 while the prose on the same line scores above 90.
        # Dropping individual low-confidence words therefore deletes the most
        # important identifier.  Keep all actual word rows and reject noise at
        # the line/region semantic gates below.
        if confidence < 0.0 or width <= 0 or height <= 0:
            continue
        key = (fields[1], fields[2], fields[3], fields[4])
        words_by_line.setdefault(key, []).append({
            "text": text,
            "confidence": confidence,
            "bbox_px": [left, top, left + width, top + height],
        })

    lines = []
    for words in words_by_line.values():
        words.sort(key=lambda word: word["bbox_px"][0])
        x0 = min(word["bbox_px"][0] for word in words)
        y0 = min(word["bbox_px"][1] for word in words)
        x1 = max(word["bbox_px"][2] for word in words)
        y1 = max(word["bbox_px"][3] for word in words)
        text = _normalize_pdf_vector_ocr_text(" ".join(word["text"] for word in words))
        if not text:
            continue
        lines.append({
            "text": text,
            "bbox_px": [x0, y0, x1, y1],
            "bbox": [x0 / scale, y0 / scale, x1 / scale, y1 / scale],
            "confidence": sum(word["confidence"] for word in words) / len(words),
            "words": words,
        })
    lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    return lines


def _pdf_vector_ocr_candidate(page, image_rects: list[fitz.Rect]) -> tuple[bool, int, int]:
    """Cheap gate for pages whose visible glyphs may be vector outlines."""
    if image_rects:
        return False, 0, 0
    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0
    if drawing_count < 20:
        return False, drawing_count, 0
    extracted_alpha_words = sum(
        1
        for word in page.get_text("words")
        if len(re.sub(r"[^A-Za-z]", "", str(word[4]))) >= 2
    )
    return extracted_alpha_words <= 60, drawing_count, extracted_alpha_words


def _pdf_vector_ocr_detect_regions(image, lines: list[dict], scale: float) -> list[fitz.Rect]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for vector-outline PDF OCR") from exc

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_regions: list[fitz.Rect] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w < width * 0.14
            or h < scale * 12.0
            or h > height * 0.45
            or w > width * 0.97
        ):
            continue
        area_ratio = cv2.contourArea(contour) / max(w * h, 1)
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, perimeter * 0.015, True)
        if area_ratio < 0.90 or len(vertices) > 8:
            continue
        raw_regions.append(fitz.Rect(x / scale, y / scale, (x + w) / scale, (y + h) / scale))

    # Connected flowchart cells often form one giant contour because the
    # connector touches every box.  Recover each cell by pairing adjacent
    # horizontal borders with the same span and at least one OCR line between.
    strong_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(
        strong_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, int(width * 0.045)), 1)),
    )
    horizontal_contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segments = []
    for contour in horizontal_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= width * 0.18 and h <= max(12, int(scale * 4)):
            segments.append((x, y, w, h))

    segment_groups: list[list[tuple[int, int, int, int]]] = []
    for segment in sorted(segments, key=lambda item: (item[0], item[2], item[1])):
        target = None
        for group in segment_groups:
            anchor = group[0]
            if abs(segment[0] - anchor[0]) <= 8 and abs(segment[2] - anchor[2]) <= 12:
                target = group
                break
        if target is None:
            segment_groups.append([segment])
        else:
            target.append(segment)

    # The compact expression above only appends to existing groups; new
    # groups were already seeded.  Normalize and pair their consecutive lines.
    for group in segment_groups:
        group.sort(key=lambda item: item[1])
        for upper, lower in zip(group, group[1:]):
            top = upper[1]
            bottom = lower[1]
            gap = bottom - top
            if gap < scale * 12.0 or gap > height * 0.45:
                continue
            left_x = int(round((upper[0] + lower[0]) / 2.0))
            right_x = int(round(
                ((upper[0] + upper[2]) + (lower[0] + lower[2])) / 2.0
            ))
            y_start = max(0, min(height - 1, top))
            y_end = max(y_start + 1, min(height, bottom + lower[3]))

            def _vertical_border_ratio(x_position: int) -> float:
                x_start = max(0, x_position - 3)
                x_end = min(width, x_position + 4)
                band = strong_binary[y_start:y_end, x_start:x_end]
                if not band.size:
                    return 0.0
                return float((band.max(axis=1) > 0).mean())

            if (
                _vertical_border_ratio(left_x) < 0.55
                or _vertical_border_ratio(right_x) < 0.55
            ):
                continue
            candidate = fitz.Rect(
                min(upper[0], lower[0]) / scale,
                top / scale,
                max(upper[0] + upper[2], lower[0] + lower[2]) / scale,
                (bottom + lower[3]) / scale,
            )
            if any(
                candidate.contains(fitz.Point(
                    (fitz.Rect(line["bbox"]).x0 + fitz.Rect(line["bbox"]).x1) / 2.0,
                    (fitz.Rect(line["bbox"]).y0 + fitz.Rect(line["bbox"]).y1) / 2.0,
                ))
                for line in lines
            ):
                raw_regions.append(candidate)

    deduped: list[fitz.Rect] = []
    for region in sorted(raw_regions, key=lambda rect: (rect.y0, rect.x0, -rect.get_area())):
        duplicate_index = None
        for index, existing in enumerate(deduped):
            intersection = region & existing
            if intersection.is_empty:
                continue
            overlap = intersection.get_area() / max(min(region.get_area(), existing.get_area()), 1.0)
            if overlap >= 0.94:
                duplicate_index = index
                break
        if duplicate_index is None:
            deduped.append(region)
        elif region.get_area() > deduped[duplicate_index].get_area():
            deduped[duplicate_index] = region
    return deduped


def _pdf_vector_ocr_dominant_fill(image, rect: fitz.Rect, scale: float) -> list[float]:
    import numpy as np

    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(math.floor(rect.x0 * scale))))
    y0 = max(0, min(height - 1, int(math.floor(rect.y0 * scale))))
    x1 = max(x0 + 1, min(width, int(math.ceil(rect.x1 * scale))))
    y1 = max(y0 + 1, min(height, int(math.ceil(rect.y1 * scale))))
    pixels = image[y0:y1, x0:x1].reshape(-1, 3)
    if not len(pixels):
        return [1.0, 1.0, 1.0]
    quantized = (pixels // 16).astype(np.int16)
    keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    winner = int(np.bincount(keys).argmax())
    mask = keys == winner
    bgr = pixels[mask].mean(axis=0) if mask.any() else pixels.mean(axis=0)
    return [round(float(bgr[2]) / 255.0, 4), round(float(bgr[1]) / 255.0, 4), round(float(bgr[0]) / 255.0, 4)]


def _pdf_vector_ocr_semantic_paragraphs(
    lines: list[dict],
    region: fitz.Rect,
) -> list[list[dict]]:
    useful = [line for line in lines if not _pdf_vector_ocr_line_is_artifact(line)]
    if not useful:
        return []
    useful.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    if region.height <= 90.0:
        return [useful]

    title = []
    body = []
    for line in useful:
        rect = fitz.Rect(line["bbox"])
        if not body and (
            rect.x0 <= region.x0 + 13.0
            and rect.y0 <= region.y0 + 34.0
        ):
            title.append(line)
        else:
            body.append(line)
    if not title:
        title = [useful[0]]
        body = useful[1:]

    paragraphs = [title]
    if not body:
        return paragraphs
    heights = sorted(max(fitz.Rect(line["bbox"]).height, 1.0) for line in body)
    median_height = heights[len(heights) // 2]
    current = []
    previous = None
    for line in body:
        rect = fitz.Rect(line["bbox"])
        new_item = False
        if previous is not None:
            previous_rect = fitz.Rect(previous["bbox"])
            top_gap = rect.y0 - previous_rect.y0
            new_item = top_gap > max(14.0, median_height * 1.55)
        if new_item and current:
            paragraphs.append(current)
            current = []
        current.append(line)
        previous = line
    if current:
        paragraphs.append(current)
    return paragraphs


def _pdf_vector_ocr_outside_paragraphs(lines: list[dict]) -> list[list[dict]]:
    useful = [line for line in lines if not _pdf_vector_ocr_line_is_artifact(line)]
    useful.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    previous = None
    for line in useful:
        line_text = _normalize_pdf_vector_ocr_text(line.get("text", ""))
        rect = fitz.Rect(line["bbox"])
        starts_note = bool(re.match(r"(?i)^Note\s+\d+\s*:", line_text))
        large_gap = False
        if previous is not None:
            previous_rect = fitz.Rect(previous["bbox"])
            large_gap = rect.y0 - previous_rect.y1 > max(
                10.0,
                min(previous_rect.height, rect.height) * 1.25,
            )
        if current and (starts_note or large_gap):
            paragraphs.append(current)
            current = []
        current.append(line)
        previous = line
    if current:
        paragraphs.append(current)
    return paragraphs


def _make_pdf_vector_ocr_element(
    lines: list[dict],
    page_rect: fitz.Rect,
    image,
    scale: float,
    *,
    region: fitz.Rect | None = None,
) -> dict | None:
    useful = [line for line in lines if not _pdf_vector_ocr_line_is_artifact(line)]
    if not useful:
        return None
    source_union = fitz.Rect(useful[0]["bbox"])
    for line in useful[1:]:
        source_union |= fitz.Rect(line["bbox"])
    render_rect = fitz.Rect(region) if region is not None else fitz.Rect(source_union)
    if region is not None:
        inset = min(3.0, max(1.5, min(region.width, region.height) * 0.035))
        render_rect = fitz.Rect(
            region.x0 + inset,
            region.y0 + inset,
            region.x1 - inset,
            region.y1 - inset,
        )

    semantic_groups = (
        _pdf_vector_ocr_semantic_paragraphs(useful, fitz.Rect(region))
        if region is not None
        else _pdf_vector_ocr_outside_paragraphs(useful)
    )
    if not semantic_groups:
        return None
    paragraph_records = []
    plain_parts = []
    rich_parts = []
    for position, group in enumerate(semantic_groups):
        plain = _join_pdf_vector_ocr_lines(group)
        if not plain:
            continue
        if region is not None and region.height > 90.0 and position > 0:
            plain = f"• {plain}"
        rich = plain
        if position == 0 and region is not None and region.height > 90.0:
            rich = f"<b>{plain}</b>"
        elif re.match(r"^(?:EIA|SR|WR|IN|SC)\d*:", plain):
            rich = re.sub(
                r"^((?:EIA|SR|WR|IN|SC)\d*:)",
                r"<b>\1</b>",
                plain,
                count=1,
            )
        group_rect = fitz.Rect(group[0]["bbox"])
        for line in group[1:]:
            group_rect |= fitz.Rect(line["bbox"])
        source_lines = [
            {
                "plain": _normalize_pdf_vector_ocr_text(line.get("text", "")),
                "rich": _normalize_pdf_vector_ocr_text(line.get("text", "")),
                "bbox": [float(value) for value in fitz.Rect(line["bbox"])],
            }
            for line in group
        ]
        paragraph_records.append({
            "plain": plain,
            "rich": rich,
            "source_bbox": [float(value) for value in group_rect],
            "source_line_bboxes": [line["bbox"] for line in source_lines],
            "source_lines": source_lines,
            "margin_left": 0.0,
            "text_indent": 0.0,
            "gap_before": 0.0 if not paragraph_records else 2.0,
            "text_align": "left",
            "nowrap": False,
            "first_line_indent": False,
        })
        plain_parts.append(plain)
        rich_parts.append(rich)
    if not paragraph_records:
        return None

    line_heights = sorted(fitz.Rect(line["bbox"]).height for line in useful)
    median_height = line_heights[len(line_heights) // 2]
    fontsize = max(7.0, min(11.0, median_height * 1.10))
    fill_rects = []
    erase_rects = []
    if region is not None:
        fill_rect = fitz.Rect(render_rect)
        fill_rects.append({
            "bbox": [float(value) for value in fill_rect],
            "fill": _pdf_vector_ocr_dominant_fill(image, fill_rect, scale),
        })
    else:
        for line in useful:
            line_rect = fitz.Rect(line["bbox"])
            expanded = fitz.Rect(
                line_rect.x0 - 1.5,
                line_rect.y0 - 1.0,
                line_rect.x1 + 1.5,
                line_rect.y1 + 1.0,
            ) & page_rect
            erase_rects.append({
                "bbox": [float(value) for value in expanded],
                "fill": _pdf_vector_ocr_dominant_fill(image, expanded, scale),
            })
    background = (
        fill_rects[0]["fill"] if fill_rects
        else _pdf_vector_ocr_dominant_fill(image, source_union, scale)
    )
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    color = 0xFFFFFF if luminance < 0.55 else 0x000000
    plain_text = "\n\n".join(plain_parts)
    rich_text = "\n\n".join(rich_parts)
    code_fragments = [
        {"text": code, "font_kind": "vector-ocr-code"}
        for code in re.findall(r"\b(?:EIA|SR|WR|IN|SC)\d+\b", plain_text)
    ]
    identity_material = json.dumps({
        "content": plain_text,
        "bbox": [round(float(value), 3) for value in render_rect],
        "lines": [line.get("text", "") for line in useful],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "type": "text",
        "y": render_rect.y0,
        "x": render_rect.x0,
        "rect": fitz.Rect(render_rect),
        "bbox": [float(value) for value in render_rect],
        "render_bbox": [float(value) for value in render_rect],
        "content": plain_text,
        "rich_content": rich_text if rich_text != plain_text else None,
        "paragraphs": paragraph_records,
        "top_padding": 0.0,
        "fontsize": fontsize,
        "line_height": fontsize * 1.22,
        "color": color,
        "bold": False,
        "non_horizontal": False,
        "rotation": 0,
        "inline_math_fragments": code_fragments,
        "preserve_source_style": bool(region is not None),
        "table_hint": bool(region is not None),
        "diagram_cell_hint": bool(region is not None),
        "layout_class": "table" if region is not None else "scattered",
        "vector_ocr": True,
        "vector_ocr_source_signature": hashlib.sha256(identity_material.encode("utf-8")).hexdigest(),
        "vector_ocr_fill_rects": fill_rects,
        "vector_ocr_erase_rects": erase_rects,
        "merged_visual_line_count": len(useful),
    }


def _extract_pdf_vector_ocr_elements(
    page,
    elements: list[dict],
    image_rects: list[fitz.Rect],
) -> list[dict]:
    candidate, drawing_count, extracted_alpha_words = _pdf_vector_ocr_candidate(
        page, image_rects
    )
    if not candidate:
        return []
    if not shutil.which("tesseract"):
        raise RuntimeError(
            f"Page {page.number + 1}: suspicious vector-text page requires tesseract OCR"
        )
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            f"Page {page.number + 1}: suspicious vector-text page requires OpenCV OCR support"
        ) from exc

    scale = PDF_VECTOR_OCR_DPI / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
    )
    png_bytes = pixmap.tobytes("png")
    try:
        completed = subprocess.run(
            [
                "tesseract", "stdin", "stdout", "--dpi", str(PDF_VECTOR_OCR_DPI),
                "-l", "eng", "--psm", "11", "tsv",
            ],
            input=png_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Page {page.number + 1}: vector-text OCR failed ({type(exc).__name__})"
        ) from exc
    lines = _pdf_vector_ocr_parse_tsv(
        completed.stdout.decode("utf-8", errors="replace"), scale
    )
    ocr_alpha_words = sum(
        len(re.findall(r"[A-Za-z]{2,}", line.get("text", "")))
        for line in lines
    )
    if (
        ocr_alpha_words < PDF_VECTOR_OCR_MIN_ALPHA_WORDS
        or ocr_alpha_words < max(extracted_alpha_words * 2.2, PDF_VECTOR_OCR_MIN_ALPHA_WORDS)
    ):
        log.info(
            "Page %s: vector OCR gate rejected chart page "
            "(drawings=%s, extracted_words=%s, ocr_words=%s)",
            page.number + 1,
            drawing_count,
            extracted_alpha_words,
            ocr_alpha_words,
        )
        return []

    image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Page {page.number + 1}: vector OCR raster decode failed")
    regions = _pdf_vector_ocr_detect_regions(image, lines, scale)
    matched_indices = {
        index
        for index, line in enumerate(lines)
        if _pdf_vector_ocr_existing_line_match(line, elements)
    }
    assigned_indices: set[int] = set()
    additions: list[dict] = []
    for region in regions:
        contained = []
        contained_indices = []
        for index, line in enumerate(lines):
            line_rect = fitz.Rect(line["bbox"])
            center = fitz.Point(
                (line_rect.x0 + line_rect.x1) / 2.0,
                (line_rect.y0 + line_rect.y1) / 2.0,
            )
            if region.contains(center):
                contained.append(line)
                contained_indices.append(index)
        unmatched_words = sum(
            len(re.findall(r"[A-Za-z]{2,}", lines[index].get("text", "")))
            for index in contained_indices
            if index not in matched_indices
        )
        if unmatched_words < 2:
            continue
        element = _make_pdf_vector_ocr_element(
            contained,
            fitz.Rect(page.rect),
            image,
            scale,
            region=region,
        )
        if element is not None:
            additions.append(element)
            assigned_indices.update(contained_indices)

    remaining = [
        (index, line)
        for index, line in enumerate(lines)
        if index not in assigned_indices
        and index not in matched_indices
        and not _pdf_vector_ocr_line_is_artifact(line)
    ]
    outside_groups: list[list[tuple[int, dict]]] = []
    for item in remaining:
        _, line = item
        rect = fitz.Rect(line["bbox"])
        target = None
        best_score = None
        for group in outside_groups:
            group_rect = fitz.Rect(group[0][1]["bbox"])
            for _, grouped_line in group[1:]:
                group_rect |= fitz.Rect(grouped_line["bbox"])
            vertical_gap = rect.y0 - group_rect.y1
            center_delta = abs(
                (rect.x0 + rect.x1) / 2.0
                - (group_rect.x0 + group_rect.x1) / 2.0
            )
            horizontal_overlap = max(
                0.0, min(rect.x1, group_rect.x1) - max(rect.x0, group_rect.x0)
            )
            if vertical_gap < -3.0 or vertical_gap > 18.0:
                continue
            if horizontal_overlap <= 0 and center_delta > max(rect.width, group_rect.width) * 0.75 + 18.0:
                continue
            score = max(vertical_gap, 0.0) + center_delta * 0.08
            if best_score is None or score < best_score:
                target = group
                best_score = score
        if target is None:
            outside_groups.append([item])
        else:
            target.append(item)

    for group in outside_groups:
        group_lines = [line for _, line in group]
        if sum(len(re.findall(r"[A-Za-z]{2,}", line.get("text", ""))) for line in group_lines) < 2:
            continue
        element = _make_pdf_vector_ocr_element(
            group_lines,
            fitz.Rect(page.rect),
            image,
            scale,
        )
        if element is not None:
            additions.append(element)

    if not additions:
        raise RuntimeError(
            f"Page {page.number + 1}: vector OCR found {ocr_alpha_words} words but no semantic regions"
        )

    paint_rects = [
        fitz.Rect(spec["bbox"])
        for element in additions
        for field in ("vector_ocr_fill_rects", "vector_ocr_erase_rects")
        for spec in element.get(field, [])
    ]
    for element in elements:
        if element.get("type") != "text":
            continue
        source_rect = _get_pdf_source_ink_rect(element)
        if source_rect.is_empty:
            continue
        if any(
            not (source_rect & paint_rect).is_empty
            and (source_rect & paint_rect).get_area() / max(source_rect.get_area(), 1.0) >= 0.12
            for paint_rect in paint_rects
        ):
            element["type"] = "text_merged_away"
            element["vector_ocr_superseded"] = True

    log.warning(
        "Page %s: recovered %s semantic vector-outline text region(s) "
        "from %s OCR words (drawings=%s, native_words=%s)",
        page.number + 1,
        len(additions),
        ocr_alpha_words,
        drawing_count,
        extracted_alpha_words,
    )
    return additions
