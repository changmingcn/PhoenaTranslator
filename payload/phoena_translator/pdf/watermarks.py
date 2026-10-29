"""Watermarks and disclaimers: repeated light-grey stamps to drop and
disclaimer blocks that keep their source text."""

from __future__ import annotations

import logging
import re
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.geometry import _get_pdf_elem_rect


DISCLAIMER_PATTERNS = re.compile(
    r'(?i)^(\s*\*?\s*(disclaimer|copyright|all rights reserved|legal notice|terms of use|'
    r'this (document|report|paper|publication) (is|was|has been)|'
    r'the views expressed|the opinions expressed|'
    r'for informational purposes only|not intended as|'
    r'source:|sources?:|\[\d+\]|footnote|endnote))',
    re.MULTILINE
)
_PDF_TRANSLATABLE_DISCLAIMER_LABELS = {
    "this report is intended for",
}
_PDF_TRANSLATABLE_REPORT_STRUCTURE_RE = re.compile(
    r"(?is)^\s*this\s+report\s+is\s+"
    r"(?:comprised|composed|organised|organized|structured|divided|arranged|made\s+up)\s+"
    r"(?:of|into)\b"
)
WATERMARK_PATTERNS = re.compile(
    r'(?i)\b(draft|confidential|sample|watermark|do not distribute|internal use|'
    r'for review only|preliminary|intended for|copy)\b'
)



log = logging.getLogger("translator")

def _is_pdf_translatable_disclaimer_label(text: str) -> bool:
    plain = _collapsed_plain_text(text or "")
    normalized = plain.rstrip(" \t\r\n:;,.!?").casefold()
    return normalized in _PDF_TRANSLATABLE_DISCLAIMER_LABELS


def _is_disclaimer_block(text: str) -> bool:
    """Check if a text block is a disclaimer/copyright notice.
    More robust than just regex: requires either the DISCLAIMER_PATTERNS match,
    or a copyright notice (©) that makes up a significant portion of the block."""
    if _is_pdf_translatable_disclaimer_label(text):
        return False
    # Structural prose such as ``This Report is comprised of eight chapters``
    # describes the document and is ordinary translatable body text.  The old
    # broad legal-notice prefix treated every ``This report is ...`` sentence
    # as a disclaimer and silently left entire paragraphs in English.
    if _PDF_TRANSLATABLE_REPORT_STRUCTURE_RE.search(_plain_text(text or "")):
        return False
    if DISCLAIMER_PATTERNS.search(text):
        return True
    # Check for © at line start, but only if the block is primarily a copyright notice
    if re.search(r"(?m)^\s*©", text):
        plain = re.sub(r"<[^>]+>", "", text).strip()
        # Only treat as disclaimer if it's a standalone copyright block (>60% of text
        # is the copyright line), not a header that happens to include ©
        lines = [l.strip() for l in plain.splitlines() if l.strip()]
        copyright_lines = [
            l
            for l in lines
            if l.startswith("©")
            or "all rights reserved" in l.lower()
            or "bridgewater" in l.lower()
            and "©" in l
        ]
        if len(copyright_lines) >= len(lines) * 0.5:
            return True
    return False


def _color_int_to_rgb(color_int: int) -> tuple[int, int, int]:
    return ((color_int >> 16) & 0xFF, (color_int >> 8) & 0xFF, color_int & 0xFF)


def _looks_light_gray(color_int: int) -> bool:
    r_val, g_val, b_val = _color_int_to_rgb(color_int)
    spread = max(r_val, g_val, b_val) - min(r_val, g_val, b_val)
    avg = (r_val + g_val + b_val) / 3
    return avg >= 125 and spread <= 28


def _normalize_watermark_text(text: str) -> str:
    plain = _collapsed_plain_text(text).casefold()
    return re.sub(r"\s+", " ", plain)


def _mark_watermark_elements(page_extractions, page_rects, total_pages: int):
    """Identify repeated watermark-like text blocks and skip translating them."""
    groups = {}
    min_repeat_pages = 2 if total_pages <= 6 else 3

    for page_num, info in page_extractions.items():
        elements = info.get("elements", [])
        page_rect = page_rects.get(page_num)
        if not page_rect:
            continue
        page_area = max(page_rect.width * page_rect.height, 1)

        for elem_idx, elem in enumerate(elements):
            if elem.get("type") != ElementType.TEXT:
                continue

            raw_text = elem.get("content", "")
            norm_text = _normalize_watermark_text(raw_text)
            if len(norm_text) < 4 or len(norm_text) > 120:
                continue

            rect = _get_pdf_elem_rect(elem)
            cx = ((rect.x0 + rect.x1) / 2) / max(page_rect.width, 1)
            cy = ((rect.y0 + rect.y1) / 2) / max(page_rect.height, 1)
            area_ratio = (rect.width * rect.height) / page_area
            pos_bucket = (round(cx, 1), round(cy, 1))
            groups.setdefault(norm_text, []).append(
                {
                    "page_num": page_num,
                    "elem_idx": elem_idx,
                    "bucket": pos_bucket,
                    "centerish": 0.18 <= cx <= 0.82 and 0.18 <= cy <= 0.82,
                    "header_footer": cy <= 0.14 or cy >= 0.86,
                    "light_gray": _looks_light_gray(elem.get("color", 0)),
                    "non_horizontal": bool(elem.get("non_horizontal")),
                    "large": elem.get("fontsize", 0) >= 18 or area_ratio >= 0.035,
                    "keyword": bool(WATERMARK_PATTERNS.search(raw_text)),
                    "text": raw_text,
                }
            )

    for norm_text, records in groups.items():
        pages = {record["page_num"] for record in records}
        if len(pages) < min_repeat_pages:
            continue

        bucket_counts = {}
        for record in records:
            bucket_counts.setdefault(record["bucket"], set()).add(record["page_num"])
        best_bucket, best_pages = max(
            bucket_counts.items(), key=lambda item: len(item[1])
        )
        if len(best_pages) < min_repeat_pages:
            continue

        for record in records:
            if record["bucket"] != best_bucket:
                continue

            # A keyword alone is not proof: repeated compliance footers such
            # as "For internal use only. Do not distribute." are printed in
            # normal black body type and are document content, not watermark
            # stamps.  The keyword arm therefore needs one visual watermark
            # trait; plain repeated keyword lines fall through to normal
            # translation instead of being erased.
            looks_like_watermark = (
                (
                    record["keyword"]
                    and (
                        record["light_gray"]
                        or record["non_horizontal"]
                        or record["large"]
                    )
                )
                or record["non_horizontal"]
                or (
                    record["light_gray"]
                    and (record["centerish"] or record["header_footer"])
                )
                or (record["centerish"] and record["large"])
            )
            if not looks_like_watermark:
                continue

            elem = page_extractions[record["page_num"]]["elements"][record["elem_idx"]]
            elem["skip_translate_reason"] = SkipReason.WATERMARK

    marked = sum(
        1
        for info in page_extractions.values()
        for elem in info.get("elements", [])
        if elem.get("skip_translate_reason") == SkipReason.WATERMARK
    )
    if marked:
        log.info(f"Marked {marked} text blocks as watermarks")
