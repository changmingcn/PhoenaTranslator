"""Semantic cross page helpers for deterministic PDF processing."""

from __future__ import annotations

import logging
import re

import fitz

from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
)
from phoena_translator.pdf.math_detection import (
    _pdf_superscript_signature,
    _plain_text,
)

log = logging.getLogger("translator")

from phoena_translator.pdf.semantic_text import (
    _bucket_pdf_fontsize,
    _ends_with_sentence_boundary,
    _is_disclaimer_block,
    _is_heading_like_elem,
    _pick_pdf_dominant_value,
)


def _split_trailing_fragment_for_merge(text: str) -> tuple[str, str]:
    """Split a text block into (keep_on_page, carry_to_next_page).
    Only the trailing incomplete sentence fragment should be carried forward.
    Large fully translated body text must stay on the original page.
    """
    stripped = text.rstrip()
    if not stripped or _ends_with_sentence_boundary(stripped):
        return stripped, ""

    import unicodedata

    for idx in range(len(stripped) - 1, -1, -1):
        ch = stripped[idx]
        if ch in ",，、":
            continue
        if unicodedata.category(ch).startswith("P"):
            keep_text = stripped[: idx + 1].rstrip()
            carry_text = stripped[idx + 1 :].strip()
            if carry_text:
                return keep_text, carry_text
            break

    # If there is no sentence-ending punctuation, fall back to the last line/
    # paragraph only; otherwise we risk moving an entire page block forward.
    parts = [p.strip() for p in re.split(r"\n\s*\n+|\n", stripped) if p.strip()]
    if len(parts) > 1:
        carry_text = parts[-1]
        keep_text = stripped[: -len(carry_text)].rstrip()
        if carry_text:
            return keep_text, carry_text

    # As a final safeguard, only move the whole block when it is genuinely short.
    if len(_plain_text(stripped)) <= 240:
        return "", stripped
    return stripped, ""


def _looks_like_right_aligned_signoff(elem: dict, page_rect: fitz.Rect | None) -> bool:
    text = _plain_text(elem.get("content", "")).strip()
    if not text or _ends_with_sentence_boundary(text):
        return False
    if page_rect is None:
        return False

    rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
    page_width = max(page_rect.width, 1.0)
    right_gap = max(0.0, page_rect.x1 - rect.x1)
    left_ratio = (rect.x0 - page_rect.x0) / page_width
    width_ratio = rect.width / page_width
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text]

    short_block = (
        len(text) <= 120 and len(lines) <= 4 and max(len(line) for line in lines) <= 60
    )
    right_aligned = (
        right_gap <= page_width * 0.08 and left_ratio >= 0.45 and width_ratio <= 0.5
    )

    date_like = bool(
        re.search(
            r"(?i)\b("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?"
            r")\b|\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
            text,
        )
    )
    title_like = bool(
        re.search(
            r"(?i)\b(chairman|president|founder|author|editor|translator|"
            r"professor|phd|ceo|cfo|chair|director|vice president|"
            r"foreword|preface|introduction)\b",
            text,
        )
    )
    name_like = all(
        re.fullmatch(r"[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,5}", line)
        for line in lines
    )

    return short_block and right_aligned and (date_like or title_like or name_like)


def _cross_page_dominant_fontsize(page_extractions) -> float:
    weighted = []
    for info in page_extractions.values():
        for elem in info.get("elements", []):
            if elem.get("type") != "text":
                continue
            text = _plain_text(elem.get("content", "")).strip()
            if not text:
                continue
            weighted.append(
                (_bucket_pdf_fontsize(elem.get("fontsize", 11.0)), len(text))
            )
    return float(_pick_pdf_dominant_value(weighted, 11.0))


def _is_cross_page_body_elem(elem, page_rect, dominant_fontsize: float) -> bool:
    """Cross-page merge candidates must be real body prose. Headers, footers,
    footnotes and page numbers never participate: they sit in the page's edge
    bands or use smaller type than the document's dominant body size, and a
    heading (larger type) can never be the middle of a sentence."""
    if elem.get("type") != "text" or elem.get("skip_translate_reason"):
        return False
    if elem.get("layout_class") != "body":
        return False
    if (
        elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
        or elem.get("footnote_hint")
    ):
        return False
    if _is_heading_like_elem(elem):
        return False
    text = _plain_text(elem.get("content", "")).strip()
    if not text or _is_disclaimer_block(text):
        return False
    if len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
        return False
    if _pdf_superscript_signature(elem.get("rich_content") or ""):
        return False
    paragraphs = elem.get("paragraphs") or []
    if any(
        paragraph.get("text_align", "left") not in {"left", "justify"}
        or bool(paragraph.get("nowrap"))
        for paragraph in paragraphs
    ):
        return False
    fontsize = float(elem.get("fontsize", dominant_fontsize))
    if fontsize < dominant_fontsize * 0.88:
        return False
    if page_rect is not None:
        rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
        height = max(page_rect.height, 1.0)
        if rect.y1 <= page_rect.y0 + 0.08 * height:
            return False
        if rect.y0 >= page_rect.y1 - 0.12 * height:
            return False
    return True


def _is_cross_page_heading_elem(elem, dominant_fontsize: float) -> bool:
    if elem.get("type") != "text":
        return False
    return (
        _is_heading_like_elem(elem)
        or float(elem.get("fontsize", dominant_fontsize)) > dominant_fontsize * 1.2
    )


def _carry_fragment_is_prose(carry_text: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", carry_text)
    return len(words) >= 2


def _first_cross_page_lexical_char(text: str) -> str:
    match = re.search(r"[A-Za-z0-9\u4e00-\u9fff]", _plain_text(text or ""))
    return match.group(0) if match else ""


_TRUSTED_PAGE_END_RUN_IN_START_EXCEPTION = "trusted-page-end-run-in"


def _cross_page_destination_start_kind(text: str) -> str:
    """Return the persisted lexical-start category for a merge destination."""
    first_char = _first_cross_page_lexical_char(text)
    if first_char.isdigit():
        return "digit"
    if first_char.isupper():
        return "uppercase"
    if first_char.islower():
        return "lowercase"
    return "other" if first_char else "none"


def _cross_page_destination_start_rejection(
    start_kind: str,
    start_exception: str | None = None,
) -> str | None:
    """Validate one destination start under the shared producer/audit policy.

    An uppercase destination is safe only when extraction proved that the
    carried fragment is a new sentence which started at the previous page end.
    Persisting that narrowly named exception lets the final auditor verify the
    same rule without weakening arbitrary uppercase-start rejection.
    """
    if start_exception is not None and not isinstance(start_exception, str):
        return "destination-start-exception-unknown"
    if start_exception not in (None, _TRUSTED_PAGE_END_RUN_IN_START_EXCEPTION):
        return "destination-start-exception-unknown"
    if start_kind == "none":
        return "destination-empty"
    if start_kind == "digit":
        return "destination-starts-with-digit"
    if start_kind == "lowercase":
        return (
            None
            if start_exception is None
            else "destination-start-exception-mismatch"
        )
    if start_kind == "uppercase":
        return (
            None
            if start_exception == _TRUSTED_PAGE_END_RUN_IN_START_EXCEPTION
            else "destination-starts-with-uppercase"
        )
    return "destination-does-not-start-with-lowercase"


def _has_trusted_pdf_same_baseline_page_end_run_in(elem: dict) -> bool:
    """Return whether extraction proved a new sentence starts at page end."""
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if not paragraphs:
        return False
    source_lines = [
        line
        for line in (paragraphs[-1].get("source_lines") or [])
        if (line.get("plain") or "").strip()
    ]
    if not source_lines:
        return False
    fragments = [
        fragment
        for fragment in (source_lines[-1].get("same_baseline_fragments") or [])
        if (fragment.get("plain") or "").strip()
    ]
    if len(fragments) < 2:
        return False
    final_fragment = _plain_text(fragments[-1].get("plain", "")).strip()
    content = _plain_text(elem.get("content", "")).strip()
    return bool(
        final_fragment
        and content.endswith(final_fragment)
        and not _ends_with_sentence_boundary(final_fragment)
        and any(
            _ends_with_sentence_boundary(fragment.get("plain", ""))
            for fragment in fragments[:-1]
        )
    )


def _cross_page_destination_start_evidence(
    source_elem: dict | None,
    destination_text: str,
) -> tuple[str, str | None]:
    """Derive, rather than trust, the lexical start and its narrow exception."""
    start_kind = _cross_page_destination_start_kind(destination_text)
    start_exception = (
        _TRUSTED_PAGE_END_RUN_IN_START_EXCEPTION
        if source_elem
        and start_kind == "uppercase"
        and _has_trusted_pdf_same_baseline_page_end_run_in(source_elem)
        else None
    )
    return start_kind, start_exception


def _cross_page_geometry_rejection(prev_page_rect, next_page_rect) -> str | None:
    if prev_page_rect is None or next_page_rect is None:
        return None

    prev_landscape = prev_page_rect.width > prev_page_rect.height
    next_landscape = next_page_rect.width > next_page_rect.height
    if prev_landscape != next_landscape:
        return "page-orientation-mismatch"

    width_delta = abs(prev_page_rect.width - next_page_rect.width) / max(
        prev_page_rect.width, next_page_rect.width, 1.0
    )
    height_delta = abs(prev_page_rect.height - next_page_rect.height) / max(
        prev_page_rect.height, next_page_rect.height, 1.0
    )
    if width_delta > 0.12 or height_delta > 0.12:
        return "page-size-mismatch"
    return None


def _cross_page_element_rejection(
    prev_elem: dict,
    next_elem: dict,
    prev_page_rect,
    next_page_rect,
    dominant_fontsize: float,
) -> str | None:
    geometry_rejection = _cross_page_geometry_rejection(prev_page_rect, next_page_rect)
    if geometry_rejection:
        return geometry_rejection

    start_kind, start_exception = _cross_page_destination_start_evidence(
        prev_elem,
        next_elem.get("content", ""),
    )
    start_rejection = _cross_page_destination_start_rejection(
        start_kind,
        start_exception,
    )
    if start_rejection:
        return start_rejection

    prev_size = float(prev_elem.get("fontsize", dominant_fontsize))
    next_size = float(next_elem.get("fontsize", dominant_fontsize))
    if abs(prev_size - next_size) > max(0.8, max(prev_size, next_size) * 0.10):
        return "font-size-mismatch"

    if prev_page_rect is None or next_page_rect is None:
        return None
    prev_rect = _get_pdf_elem_rect(prev_elem)
    next_rect = _get_pdf_elem_rect(next_elem)
    prev_x0 = (prev_rect.x0 - prev_page_rect.x0) / max(prev_page_rect.width, 1.0)
    prev_x1 = (prev_rect.x1 - prev_page_rect.x0) / max(prev_page_rect.width, 1.0)
    next_x0 = (next_rect.x0 - next_page_rect.x0) / max(next_page_rect.width, 1.0)
    next_x1 = (next_rect.x1 - next_page_rect.x0) / max(next_page_rect.width, 1.0)
    prev_width = max(prev_x1 - prev_x0, 0.001)
    next_width = max(next_x1 - next_x0, 0.001)
    overlap = max(0.0, min(prev_x1, next_x1) - max(prev_x0, next_x0))
    overlap_ratio = overlap / min(prev_width, next_width)
    center_delta = abs((prev_x0 + prev_x1) / 2.0 - (next_x0 + next_x1) / 2.0)
    width_ratio = max(prev_width, next_width) / min(prev_width, next_width)
    if overlap_ratio < 0.60 or center_delta > 0.15 or width_ratio > 1.8:
        return "column-geometry-mismatch"
    return None


def _merge_cross_page_sentences(
    page_extractions, total_pages, page_rects=None, audit_log=None
):
    """Merge body sentences broken across page boundaries.
    If a page's last BODY paragraph doesn't end with sentence-ending
    punctuation, its trailing fragment is prepended to the next page's first
    BODY paragraph so the sentence is translated and rendered as one unit.
    Footnotes, footers, headers and headings are never merge participants; a
    next page that opens with a heading means the sentence was complete."""
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions)
    pending_text = ""
    pending_keep_text = ""
    pending_page = -1
    pending_elem_idx = -1
    pending_page_rect = None

    def _orientation_name(rect) -> str | None:
        if rect is None:
            return None
        return "landscape" if rect.width > rect.height else "portrait"

    def _record_decision(
        decision: str,
        reason: str,
        destination_page: int | None,
        destination_idx=None,
        destination_elem=None,
    ):
        if audit_log is None or pending_page < 0:
            return
        source_info = page_extractions.get(pending_page, {})
        source_elem = None
        if "elements" in source_info and 0 <= pending_elem_idx < len(
            source_info["elements"]
        ):
            source_elem = source_info["elements"][pending_elem_idx]
        destination_rect = (
            page_rects.get(destination_page)
            if page_rects and destination_page is not None
            else None
        )
        start_kind, start_exception = _cross_page_destination_start_evidence(
            source_elem,
            destination_elem.get("content", "") if destination_elem else "",
        )
        audit_log.append(
            {
                "kind": "cross-page-merge",
                "source_page": pending_page + 1,
                "destination_page": destination_page + 1
                if destination_page is not None
                else None,
                "source_element": pending_elem_idx,
                "destination_element": destination_idx,
                "decision": decision,
                "reason": reason,
                "source_layout": source_elem.get("layout_class")
                if source_elem
                else None,
                "destination_layout": destination_elem.get("layout_class")
                if destination_elem
                else None,
                "source_fontsize": round(float(source_elem.get("fontsize", 0.0)), 2)
                if source_elem
                else None,
                "destination_fontsize": round(
                    float(destination_elem.get("fontsize", 0.0)), 2
                )
                if destination_elem
                else None,
                "source_orientation": _orientation_name(pending_page_rect),
                "destination_orientation": _orientation_name(destination_rect),
                "destination_start": start_kind,
                "destination_start_exception": start_exception,
                "carry_chars": len(_plain_text(pending_text)),
                "carry_words": len(re.findall(r"[A-Za-z]{2,}", pending_text)),
            }
        )

    def _drop_pending():
        nonlocal \
            pending_text, \
            pending_keep_text, \
            pending_page, \
            pending_elem_idx, \
            pending_page_rect
        pending_text = ""
        pending_keep_text = ""
        pending_page = -1
        pending_elem_idx = -1
        pending_page_rect = None

    for page_num in range(total_pages):
        info = page_extractions.get(page_num, {})
        if "elements" not in info:
            if pending_text:
                _record_decision("rejected", "destination-page-not-extracted", page_num)
            _drop_pending()
            continue

        page_rect = page_rects.get(page_num) if page_rects else None
        ordered = sorted(
            enumerate(info["elements"]),
            key=lambda item: (item[1].get("y", 0.0), item[1].get("x", 0.0)),
        )
        body_indices = [
            (i, e)
            for i, e in ordered
            if _is_cross_page_body_elem(e, page_rect, dominant_fontsize)
        ]
        if not body_indices:
            if pending_text:
                _record_decision(
                    "rejected", "destination-has-no-body-candidate", page_num
                )
            _drop_pending()
            continue

        # Commit the pending fragment into this page's first body paragraph —
        # unless a heading sits above it, which means the previous sentence
        # was actually complete (print layout never breaks a sentence around
        # a heading).
        if pending_text:
            first_idx, first_elem = body_indices[0]
            heading_above = any(
                _is_cross_page_heading_elem(e, dominant_fontsize)
                for i, e in ordered
                if i != first_idx and e.get("y", 0.0) < first_elem.get("y", 0.0)
            )
            prev_info = page_extractions.get(pending_page, {})
            prev_elem = None
            if "elements" in prev_info and 0 <= pending_elem_idx < len(
                prev_info["elements"]
            ):
                prev_elem = prev_info["elements"][pending_elem_idx]
            rejection = "heading-above-destination" if heading_above else None
            if rejection is None and prev_elem is not None:
                rejection = _cross_page_element_rejection(
                    prev_elem,
                    first_elem,
                    pending_page_rect,
                    page_rect,
                    dominant_fontsize,
                )
            if rejection is None and prev_elem is not None:
                _record_decision(
                    "accepted", "accepted", page_num, first_idx, first_elem
                )
                first_elem["content"] = pending_text + " " + first_elem["content"]
                first_elem["rich_content"] = first_elem["content"]
                if pending_keep_text:
                    prev_elem["content"] = pending_keep_text
                    prev_elem["rich_content"] = pending_keep_text
                else:
                    prev_elem["type"] = "text_merged_away"
            else:
                _record_decision(
                    "rejected",
                    rejection or "source-element-missing",
                    page_num,
                    first_idx,
                    first_elem,
                )
            _drop_pending()

        # Check if the last body paragraph ends mid-sentence
        last_idx, last_elem = body_indices[-1]
        if _looks_like_right_aligned_signoff(last_elem, page_rect):
            continue
        if page_rect is not None:
            last_rect = _get_pdf_elem_rect(last_elem)
            if last_rect.y1 < page_rect.y0 + page_rect.height * 0.45:
                continue
        keep_text, carry_text = _split_trailing_fragment_for_merge(last_elem["content"])
        if carry_text and _carry_fragment_is_prose(carry_text):
            pending_text = carry_text
            pending_keep_text = keep_text
            pending_page = page_num
            pending_elem_idx = last_idx
            pending_page_rect = page_rect

    if pending_text:
        _record_decision("rejected", "end-of-document", None)
