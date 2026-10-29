"""Collect, at extraction time, what the output audit must later find
(superscripts, formula regions, recovered vector text)."""

from __future__ import annotations

import re
from phoena_translator.pdf.model import ElementType, PageExtraction
from phoena_translator.pdf.superscripts import _pdf_is_semantic_superscript_marker
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.inline_markup import _pdf_superscript_signature
from phoena_translator.math_text import is_math_block as _is_math_block
from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from collections import deque


def _collect_pdf_superscript_expectations(page_extractions: dict[int, PageExtraction]) -> list[dict]:
    expectations = []
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_idx, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != ElementType.TEXT:
                continue

            run_queues = {}
            for run in elem.get("superscript_runs") or []:
                marker = str(run.get("text", "")).strip()
                if _pdf_is_semantic_superscript_marker(marker):
                    run_queues.setdefault(marker, deque()).append(run)

            markers = []
            paragraphs = elem.get("paragraphs") or [{
                "plain": elem.get("content", ""),
                "rich": elem.get("rich_content") or elem.get("content", ""),
            }]
            original_plain = re.sub(
                r"\s+",
                " ",
                " ".join(
                    str(paragraph.get("plain") or "")
                    for paragraph in paragraphs
                    if isinstance(paragraph, dict)
                ),
            ).strip()
            live_plain = _collapsed_plain_text(elem.get("content") or "")
            if original_plain != live_plain:
                # Cross-page absorption deliberately keeps ``paragraphs`` as
                # immutable pre-merge evidence for the merge auditor. Marker
                # expectations must instead follow the live source currently
                # owned and rendered by this element.
                paragraphs = [{
                    "plain": elem.get("content", ""),
                    "rich": elem.get("rich_content") or elem.get("content", ""),
                }]
            for paragraph in paragraphs:
                plain = paragraph.get("plain") or ""
                rich = paragraph.get("rich") or plain
                paragraph_is_formula = _is_math_block(plain)
                for marker in _pdf_superscript_signature(rich):
                    queue_for_marker = run_queues.get(marker)
                    run = queue_for_marker.popleft() if queue_for_marker else {}
                    if paragraph_is_formula or not _pdf_is_semantic_superscript_marker(marker):
                        continue
                    markers.append({
                        "text": marker,
                        "scale": round(float(run.get("scale", 0.60)), 3),
                        "source": run.get("source") or "unknown",
                    })
            if markers:
                expectations.append({
                    "page": page_num + 1,
                    "element": elem_idx,
                    "markers": markers,
                })
    return expectations


def _collect_pdf_formula_expectations(page_extractions: dict[int, PageExtraction]) -> list[dict]:
    """Collect hash-only formula expectations suitable for a persisted audit."""
    expectations = []
    allowed_signature_fields = {
        "signature_sha256", "text_sha256", "font_sha256", "geometry_sha256",
        "raster_sha256", "span_count", "raster_width", "raster_height",
    }
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_idx, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != ElementType.FORMULA_IMAGE:
                continue
            bbox = _get_pdf_elem_rect(elem)
            signature = {
                key: value
                for key, value in (elem.get("math_signature") or {}).items()
                if key in allowed_signature_fields
            }
            expectations.append({
                "page": page_num + 1,
                "element": elem_idx,
                "bbox": [float(value) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)],
                "reasons": sorted(set(elem.get("math_reasons") or ["equation_structure"])),
                "mixed": bool(elem.get("math_mixed")),
                "line_count": int(elem.get("math_line_count", 1)),
                "candidate_symbol_count": int(elem.get("math_candidate_symbol_count", 0)),
                "signature": signature,
            })
    return expectations


def _pdf_vector_ocr_audit_phrases(text: str) -> list[str]:
    words = [
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z'’\-]*", _plain_text(text or ""))
        if len(re.sub(r"[^A-Za-z]", "", word)) >= 2
        and not re.fullmatch(r"(?i)(?:EIA|SR|WR|IN|SC)\d*", word)
    ]
    if len(words) == 1:
        return [words[0]] if len(words[0]) >= 5 else []
    if not words:
        return []
    if len(words) <= 7:
        return [" ".join(words)]
    window = 5
    starts = sorted({0, max(0, (len(words) - window) // 2), len(words) - window})
    return [" ".join(words[start:start + window]) for start in starts]


def _collect_pdf_vector_ocr_expectations(page_extractions: dict[int, PageExtraction]) -> list[dict]:
    expectations = []
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_index, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != ElementType.TEXT or not elem.get("vector_ocr"):
                continue
            phrases = _pdf_vector_ocr_audit_phrases(elem.get("content", ""))
            if not phrases:
                continue
            expectations.append({
                "page": page_num + 1,
                "element": elem_index,
                "phrases": phrases,
                "source_ascii_question_marks": elem.get("content", "").count("?"),
                "source_replacement_characters": elem.get("content", "").count("\ufffd"),
                "source_signature": elem.get("vector_ocr_source_signature"),
            })
    return expectations
