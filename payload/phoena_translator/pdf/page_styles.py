"""Derive a page's dominant body, heading and footnote styles for redraw."""

from __future__ import annotations

import fitz
from phoena_translator.pdf.text_shape import (
    _bucket_pdf_fontsize,
    _pick_pdf_dominant_value,
    _plain_text,
)
from phoena_translator.pdf.model import ElementType


def _derive_pdf_page_layout_styles(
    elements: list[dict], page_rect: fitz.Rect
) -> dict[str, dict]:
    text_elements = [
        elem
        for elem in elements
        if elem.get("type") == ElementType.TEXT and not elem.get("skip_translate_reason")
    ]
    if not text_elements:
        return {
            "body": {
                "fontsize": 11.0,
                "line_height": 15.4,
                "color": 0,
                "color_hex": "#000000",
                "padding_ratio": 0.34,
            },
            "table": {
                "fontsize": 9.0,
                "line_height": 11.0,
                "color": 0,
                "color_hex": "#000000",
                "padding_ratio": 0.28,
            },
            "scattered": {
                "fontsize": 10.0,
                "line_height": 12.8,
                "color": 0,
                "color_hex": "#000000",
                "padding_ratio": 0.30,
            },
        }

    overall_fontsize = _pick_pdf_dominant_value(
        [
            (
                _bucket_pdf_fontsize(elem.get("fontsize", 11.0)),
                max(len(_plain_text(elem.get("content", "")).strip()), 1),
            )
            for elem in text_elements
        ],
        11.0,
    )
    overall_color = _pick_pdf_dominant_value(
        [
            (
                elem.get("color", 0),
                max(len(_plain_text(elem.get("content", "")).strip()), 1),
            )
            for elem in text_elements
        ],
        0,
    )

    class_defaults = {
        "body": {
            "fallback_size": overall_fontsize,
            "line_height_ratio": 1.42,
            "padding_ratio": 0.34,
        },
        "table": {
            "fallback_size": max(
                7.0,
                min(
                    overall_fontsize,
                    overall_fontsize - 1.0
                    if overall_fontsize > 9.0
                    else overall_fontsize,
                ),
            ),
            "line_height_ratio": 1.24,
            "padding_ratio": 0.26,
        },
        "scattered": {
            "fallback_size": max(8.0, min(max(overall_fontsize, 10.0), 12.0)),
            "line_height_ratio": 1.30,
            "padding_ratio": 0.30,
        },
    }

    styles = {}
    for layout_class, defaults in class_defaults.items():
        class_elems = [
            elem for elem in text_elements if elem.get("layout_class") == layout_class
        ]
        size = _pick_pdf_dominant_value(
            [
                (
                    _bucket_pdf_fontsize(
                        elem.get("fontsize", defaults["fallback_size"])
                    ),
                    max(len(_plain_text(elem.get("content", "")).strip()), 1),
                )
                for elem in class_elems
            ],
            defaults["fallback_size"],
        )
        color = _pick_pdf_dominant_value(
            [
                (
                    elem.get("color", overall_color),
                    max(len(_plain_text(elem.get("content", "")).strip()), 1),
                )
                for elem in class_elems
            ],
            overall_color,
        )

        if layout_class == "body":
            size = max(9.0, min(float(size), 13.0))
        elif layout_class == "table":
            size = max(7.0, min(float(size), max(float(overall_fontsize), 9.5)))
        else:
            size = max(8.0, min(float(size), 12.0))

        r_val = (int(color) >> 16) & 0xFF
        g_val = (int(color) >> 8) & 0xFF
        b_val = int(color) & 0xFF
        styles[layout_class] = {
            "fontsize": float(size),
            "line_height": float(size) * defaults["line_height_ratio"],
            "color": int(color),
            "color_hex": f"#{r_val:02x}{g_val:02x}{b_val:02x}",
            "padding_ratio": defaults["padding_ratio"],
        }

    return styles
