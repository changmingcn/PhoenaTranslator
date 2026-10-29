"""Decide what native page content to ignore: appendix pages, hidden-text
artifacts and text inside embedded thumbnails."""

from __future__ import annotations

import logging
import re

import fitz

from phoena_translator.pdf.geometry import _get_pdf_source_ink_rect
from phoena_translator.pdf.model import ElementType, SkipReason

SKIP_SECTION_HEADING_RE = re.compile(
    r"(?i)^(?:section\s+)?(?:\d+(?:\.\d+)*[.):]?\s*)?"
    r"(?:bibliography|references|index|endnotes?|glossary|acronyms|abbreviations)"
    r"(?:\s*(?:&|and)\s*(?:further\s+reading|notes|references))?$"
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
                element.get("type") != ElementType.TEXT
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
            element["skip_translate_reason"] = SkipReason.EMBEDDED_THUMBNAIL_TEXT
            marked += 1
    return marked
