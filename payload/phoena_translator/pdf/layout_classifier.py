"""Classify each text element of a page as body, table, scattered, heading
or footer from its geometry and neighbours."""

from __future__ import annotations

import fitz
import re
from phoena_translator.pdf.semantic_text import (
    _bucket_pdf_fontsize,
    _is_disclaimer_block,
    _normalize_pdf_footnote_alignment,
    _pdf_element_prefers_single_line_heading,
    _pick_pdf_dominant_value,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.semantic_glossary import (
    _mark_pdf_glossary_term_elements,
    _normalize_pdf_glossary_cells,
)
from phoena_translator.pdf.geometry import _get_pdf_elem_rect


def _classify_pdf_page_text_elements(elements: list[dict], page_rect: fitz.Rect) -> dict[int, str]:
    _normalize_pdf_glossary_cells(elements, page_rect)
    text_indices = [
        idx for idx, elem in enumerate(elements)
        if elem.get("type") == ElementType.TEXT
    ]
    if not text_indices:
        return {}

    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    overall_fontsize = _pick_pdf_dominant_value(
        [
            (_bucket_pdf_fontsize(elements[idx].get("fontsize", 11.0)), max(len(_plain_text(elements[idx].get("content", "")).strip()), 1))
            for idx in text_indices
        ],
        11.0,
    )

    for idx in text_indices:
        _normalize_pdf_footnote_alignment(elements[idx], page_rect)

    short_mid_candidates = []
    for idx in text_indices:
        elem = elements[idx]
        plain = _collapsed_plain_text(elem.get("content", ""))
        if not plain or _is_disclaimer_block(plain) or elem.get("skip_translate_reason") == SkipReason.WATERMARK:
            continue

        rect = _get_pdf_elem_rect(elem)
        center_y = ((rect.y0 + rect.y1) / 2.0 - page_rect.y0) / page_height
        width_ratio = rect.width / page_width
        paragraphs = [p for p in (elem.get("paragraphs") or []) if (p.get("plain") or "").strip()]
        para_count = len(paragraphs) or 1
        aligns = {p.get("text_align", "left") for p in paragraphs} or {"left"}
        nowrap = any(bool(p.get("nowrap")) for p in paragraphs)

        if (
            0.20 <= center_y <= 0.88
            and len(plain) <= 140
            and para_count <= 2
            and (width_ratio <= 0.45 or aligns != {"left"} or nowrap)
        ):
            short_mid_candidates.append(idx)

    dense_table_page = len(short_mid_candidates) >= 8
    table_zone = None
    for idx in short_mid_candidates:
        rect = _get_pdf_elem_rect(elements[idx])
        table_zone = rect if table_zone is None else (table_zone | rect)

    classifications = {}
    for idx in text_indices:
        elem = elements[idx]
        plain = _collapsed_plain_text(elem.get("content", ""))
        rect = _get_pdf_elem_rect(elem)
        center_y = ((rect.y0 + rect.y1) / 2.0 - page_rect.y0) / page_height
        width_ratio = rect.width / page_width
        fontsize = float(elem.get("fontsize", overall_fontsize))
        paragraphs = [p for p in (elem.get("paragraphs") or []) if (p.get("plain") or "").strip()]
        para_count = len(paragraphs) or 1
        aligns = {p.get("text_align", "left") for p in paragraphs} or {"left"}
        nowrap = any(bool(p.get("nowrap")) for p in paragraphs)
        para_lengths = [len(re.sub(r'\s+', ' ', p.get("plain", "")).strip()) for p in paragraphs if (p.get("plain") or "").strip()]
        short_block = len(plain) <= 140 and para_count <= 2
        header_footer = center_y <= 0.20 or center_y >= 0.90
        short_stacked_block = (
            para_count >= 2
            and len(plain) <= 140
            and width_ratio <= 0.45
            and para_lengths
            and max(para_lengths) <= 36
        )
        right_aligned_metadata = (
            center_y <= 0.32
            and width_ratio <= 0.40
            and aligns == {"right"}
            and len(plain) <= 180
        )

        is_body = (
            (para_count >= 2 and not short_stacked_block and not right_aligned_metadata)
            or len(plain) >= 180
            or (len(plain) >= 120 and width_ratio >= 0.42)
            or (len(plain) >= 90 and width_ratio >= 0.55 and rect.height >= max(fontsize * 2.0, 18.0))
        )

        in_table_zone = False
        if table_zone is not None:
            expanded_zone = fitz.Rect(table_zone.x0 - 12.0, table_zone.y0 - 12.0, table_zone.x1 + 12.0, table_zone.y1 + 12.0)
            in_table_zone = not (rect & expanded_zone).is_empty

        is_table = (
            not header_footer
            and not is_body
            and (
                in_table_zone
                or (short_block and dense_table_page)
                or (short_block and (width_ratio <= 0.35 or aligns != {"left"} or nowrap))
                or short_stacked_block
                or (len(plain) <= 40 and rect.width <= page_width * 0.25 and center_y >= 0.22)
                or (len(plain) <= 80 and fontsize <= overall_fontsize and width_ratio <= 0.30 and center_y >= 0.22)
            )
        )

        if elem.get("skip_translate_reason") == SkipReason.WATERMARK or _is_disclaimer_block(plain):
            layout_class = "scattered"
        elif elem.get("table_hint"):
            layout_class = "table"
        elif is_body and not (header_footer and len(plain) < 180):
            layout_class = "body"
        elif is_table:
            layout_class = "table"
        else:
            layout_class = "scattered"

        elem["layout_class"] = layout_class
        single_line_heading = _pdf_element_prefers_single_line_heading(elem, page_rect)
        elem["single_line_heading"] = single_line_heading
        if single_line_heading:
            for paragraph in elem.get("paragraphs") or []:
                paragraph["nowrap"] = True
        classifications[idx] = layout_class

    _mark_pdf_glossary_term_elements(elements, page_rect)
    return classifications
