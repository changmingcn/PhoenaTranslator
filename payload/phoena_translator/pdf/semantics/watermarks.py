"""Watermarks and disclaimers: repeated light-grey stamps to drop and
disclaimer blocks that keep their source text."""

from __future__ import annotations

import logging
import math
import re

from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text

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


# A text stamped across the page at one place (``_record_is_stamp``) on at
# least this share of the pages that carry elements, and on no fewer than
# ``WATERMARK_DOCUMENT_WIDE_MIN_PAGES`` of them (all of them in a shorter
# document), is a document-wide watermark: a per-copy stamp such as the
# recipient's e-mail address.  A stamp is on nearly every page; the share
# leaves room for pages where extraction holds it as formula geometry, and
# keeps out a title printed large on the covers or dividers of a short note.
WATERMARK_DOCUMENT_WIDE_PAGE_SHARE = 2 / 3
WATERMARK_DOCUMENT_WIDE_MIN_PAGES = 3


def _record_looks_like_watermark(record: dict) -> bool:
    # A keyword alone is not proof: repeated compliance footers such
    # as "For internal use only. Do not distribute." are printed in
    # normal black body type and are document content, not watermark
    # stamps.  The keyword arm therefore needs one visual watermark
    # trait; plain repeated keyword lines fall through to normal
    # translation instead of being erased.
    return bool(
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


def _record_is_stamp(record: dict) -> bool:
    """A stamp proper: large type across the middle of the page, or large
    type carrying a watermark keyword.

    Narrower than ``_record_looks_like_watermark`` on purpose.  Grey or white
    type, rotated type and small lines that happen to hold a keyword
    ("sample", "preliminary") are also how source lines, table headers on a
    dark band and chart axis titles are printed, and those repeat at several
    places on the pages of a chart book; they must not make their text a
    document-wide watermark.
    """
    return bool(record["large"] and (record["centerish"] or record["keyword"]))


def _mark_watermark_record(page_extractions, record: dict) -> None:
    elem = page_extractions[record["page_num"]]["elements"][record["elem_idx"]]
    elem["skip_translate_reason"] = SkipReason.WATERMARK


def _mark_watermark_elements(page_extractions, page_rects, total_pages: int):
    """Identify repeated watermark-like text blocks and skip translating them.

    A repeated text is judged at its dominant position: the place where it
    sits on the most pages.  Copies there that look like a watermark are
    removed.

    Portals also stamp one watermark at several places on a page (a faint
    diagonal stamp across the body plus a footer line with the same text),
    and either copy can be the dominant one.  So a text stamped across the
    page (``_record_is_stamp``) at one place on at least
    ``WATERMARK_DOCUMENT_WIDE_PAGE_SHARE`` of the pages that carry elements
    (and no fewer than ``WATERMARK_DOCUMENT_WIDE_MIN_PAGES``)
    is a document-wide watermark: every watermark-like copy of it that
    repeats at its own place is removed, dominant or not, and so is a single
    copy that shares the font size and colour of a removed one (the same
    stamp on a page of another size).

    Without such a stamp the dominant-position rule applies alone, as it
    always has: grey source lines, white table headers and rotated axis
    titles that repeat at several places lose no further copy.  A copy that
    does not look like a watermark is never removed, so a title or a black
    footer line that repeats the stamped text stays.
    """
    groups = {}
    min_repeat_pages = 2 if total_pages <= 6 else 3
    pages_with_elements = sum(
        1 for info in page_extractions.values() if info.get("elements")
    )
    document_wide_pages = max(
        min_repeat_pages,
        min(pages_with_elements, WATERMARK_DOCUMENT_WIDE_MIN_PAGES),
        math.ceil(pages_with_elements * WATERMARK_DOCUMENT_WIDE_PAGE_SHARE),
    )

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
                    "look": (
                        round(float(elem.get("fontsize", 0) or 0) * 2) / 2,
                        elem.get("color", 0),
                    ),
                    "text": raw_text,
                }
            )

    for norm_text, records in groups.items():
        pages = {record["page_num"] for record in records}
        if len(pages) < min_repeat_pages:
            continue

        bucket_pages = {}
        watermark_like_pages = {}
        stamp_pages = {}
        for record in records:
            bucket_pages.setdefault(record["bucket"], set()).add(record["page_num"])
            if _record_looks_like_watermark(record):
                watermark_like_pages.setdefault(record["bucket"], set()).add(
                    record["page_num"]
                )
            if _record_is_stamp(record):
                stamp_pages.setdefault(record["bucket"], set()).add(record["page_num"])
        best_bucket, best_pages = max(
            bucket_pages.items(), key=lambda item: len(item[1])
        )
        if len(best_pages) < min_repeat_pages:
            continue

        removed_buckets = {best_bucket}
        document_wide = any(
            len(stamped) >= document_wide_pages for stamped in stamp_pages.values()
        )
        if document_wide:
            removed_buckets.update(
                bucket
                for bucket, copies in watermark_like_pages.items()
                if len(copies) >= min_repeat_pages
            )

        removed_looks = set()
        single_copies = []
        for record in records:
            if not _record_looks_like_watermark(record):
                continue
            if record["bucket"] in removed_buckets:
                _mark_watermark_record(page_extractions, record)
                removed_looks.add(record["look"])
            else:
                single_copies.append(record)
        if document_wide:
            for record in single_copies:
                if record["look"] in removed_looks:
                    _mark_watermark_record(page_extractions, record)

    marked = sum(
        1
        for info in page_extractions.values()
        for elem in info.get("elements", [])
        if elem.get("skip_translate_reason") == SkipReason.WATERMARK
    )
    if marked:
        log.info(f"Marked {marked} text blocks as watermarks")
