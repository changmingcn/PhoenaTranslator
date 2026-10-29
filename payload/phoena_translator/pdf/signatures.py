"""Stable fingerprints of formula regions, used to prove output fidelity."""

from __future__ import annotations

import fitz
import json
import re
import hashlib
from phoena_translator.pdf.geometry import (
    _pdf_extraction_rect_to_display,
    _pdf_page_extraction_rect,
)
from phoena_translator.pdf.math_detection import _normalize_pdf_font_name


PDF_SIGNATURE_GEOMETRY_QUANTUM = 0.25



def _pdf_signature_number(
    value: float,
    quantum: float = PDF_SIGNATURE_GEOMETRY_QUANTUM,
) -> float:
    """Quantize signature geometry and canonicalize signed zero.

    Incremental PyMuPDF saves can move an extracted coordinate by less than
    0.0001 pt. Decimal rounding is unstable exactly on a half-step (13.5500
    versus 13.5499), so use a quarter-point bucket instead.
    """
    quantum = max(float(quantum), 0.000001)
    rounded = round(round(float(value) / quantum) * quantum, 6)
    return 0.0 if rounded == 0 else rounded


def _pdf_formula_region_signature(page, bbox, page_dict: dict | None = None) -> dict:
    """Hash source text/font/geometry and raster pixels without exposing content."""
    rect = fitz.Rect(bbox) & _pdf_page_extraction_rect(page)
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        raise ValueError("empty formula region")
    if page_dict is None:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    records = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = re.sub(r"\s+", " ", span.get("text", "")).strip()
                if not span_text:
                    continue
                span_rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                if span_rect.is_empty:
                    continue
                inter = span_rect & rect
                center = fitz.Point(
                    (span_rect.x0 + span_rect.x1) / 2.0,
                    (span_rect.y0 + span_rect.y1) / 2.0,
                )
                if inter.is_empty and not rect.contains(center):
                    continue
                if not rect.contains(center) and inter.get_area() < span_rect.get_area() * 0.45:
                    continue
                records.append({
                    "text": span_text,
                    "font": _normalize_pdf_font_name(span.get("font", "")),
                    "size": round(float(span.get("size", 0.0)), 2),
                    "bbox": [
                        _pdf_signature_number(span_rect.x0 - rect.x0),
                        _pdf_signature_number(span_rect.y0 - rect.y0),
                        _pdf_signature_number(span_rect.x1 - rect.x0),
                        _pdf_signature_number(span_rect.y1 - rect.y0),
                    ],
                })
    records.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["font"], item["text"]))
    text_material = "\n".join(item["text"] for item in records)
    font_material = "\n".join(item["font"] for item in records)
    geometry_material = json.dumps(
        [{"font": item["font"], "size": item["size"], "bbox": item["bbox"]} for item in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    combined_material = json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2.0, 2.0),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=_pdf_extraction_rect_to_display(page, rect),
    )
    return {
        "signature_sha256": hashlib.sha256(combined_material.encode("utf-8")).hexdigest(),
        "text_sha256": hashlib.sha256(text_material.encode("utf-8")).hexdigest(),
        "font_sha256": hashlib.sha256(font_material.encode("utf-8")).hexdigest(),
        "geometry_sha256": hashlib.sha256(geometry_material.encode("utf-8")).hexdigest(),
        "raster_sha256": hashlib.sha256(pix.samples).hexdigest(),
        "span_count": len(records),
        "raster_width": pix.width,
        "raster_height": pix.height,
    }
