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


def _is_cross_page_body_elem(
    elem,
    page_rect,
    dominant_fontsize: float,
    *,
    allow_superscript_markers: bool = False,
) -> bool:
    """Cross-page merge candidates must be real body prose. Headers, footers,
    footnotes and page numbers never participate: they sit in the page's edge
    bands or use smaller type than the document's dominant body size, and a
    heading (larger type) can never be the middle of a sentence.

    Marker-bearing paragraphs are excluded by default because the forward
    carry rewrites ``rich_content`` wholesale and would flatten superscripts.
    The orphan-tail absorber only APPENDS at the end, which cannot damage
    markers, so it opts in via ``allow_superscript_markers``."""
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
    if not allow_superscript_markers and _pdf_superscript_signature(
        elem.get("rich_content") or ""
    ):
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


def _cross_page_pair_incompatibility(
    prev_elem: dict,
    next_elem: dict,
    prev_page_rect,
    next_page_rect,
    dominant_fontsize: float,
) -> str | None:
    """Font and column compatibility shared by both merge directions."""
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

    return _cross_page_pair_incompatibility(
        prev_elem,
        next_elem,
        prev_page_rect,
        next_page_rect,
        dominant_fontsize,
    )


CROSS_PAGE_ORPHAN_TAIL_MAX_CHARS = 80
CROSS_PAGE_ORPHAN_TAIL_MAX_WORDS = 10
CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS = 300
_TRAILING_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:\d{1,3}|[*†‡§¶]{1,3})$"
)
_SENTENCE_SPLIT_BOUNDARY_RE = re.compile(
    "[.!?。！？][\"'’”)\\]]*"
)


def _strip_trailing_footnote_marker(text: str) -> str:
    """Drop one trailing footnote marker so boundary tests see the sentence."""
    stripped = (text or "").rstrip()
    without = _TRAILING_FOOTNOTE_MARKER_RE.sub("", stripped).rstrip()
    return without or stripped


def _find_leading_sentence_tail(
    plain: str,
    max_chars: int,
) -> tuple[str, str] | None:
    """Split ``plain`` at its first real sentence boundary.

    A boundary counts only when followed by whitespace/end and a new-sentence
    opener (uppercase, digit or CJK), so abbreviations such as ``U.S.`` and
    decimals such as ``2.5`` do not split the tail early."""
    for match in _SENTENCE_SPLIT_BOUNDARY_RE.finditer(plain):
        end = match.end()
        if end > max_chars:
            return None
        rest = plain[end:]
        rest_stripped = rest.lstrip()
        if rest and not rest[:1].isspace() and rest_stripped:
            continue
        if rest_stripped and not (
            rest_stripped[0].isupper()
            or rest_stripped[0].isdigit()
            or ord(rest_stripped[0]) > 0x2E7F
        ):
            continue
        tail = plain[:end].strip()
        if not tail:
            return None
        return tail, rest_stripped
    return None


def _cross_page_whole_fragment_tail(elem: dict) -> tuple[str, str] | None:
    """Return (plain, rich) when an element is exactly one stranded tail.

    A stranded tail is the end of a sentence broken by a page turn: short,
    lowercase-starting, reaching a sentence boundary.  A trailing footnote
    marker (``exchange.4``) is tolerated and carried along inside the rich
    text so the reference survives next to its sentence."""
    if elem.get("type") != "text" or elem.get("skip_translate_reason"):
        return None
    if (
        elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
    ):
        return None
    if elem.get("layout_class") not in {"body", "scattered"}:
        return None
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) > 1:
        return None
    plain = re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
    if not plain or len(plain) > CROSS_PAGE_ORPHAN_TAIL_MAX_CHARS:
        return None
    if not _first_cross_page_lexical_char(plain).islower():
        return None
    if not _ends_with_sentence_boundary(_strip_trailing_footnote_marker(plain)):
        return None
    words = re.findall(r"[A-Za-z]{2,}", plain)
    if not 1 <= len(words) <= CROSS_PAGE_ORPHAN_TAIL_MAX_WORDS:
        return None
    rich = elem.get("rich_content") or elem.get("content", "")
    return plain, " ".join(str(rich).split())


def _merge_cross_page_sentences(
    page_extractions, total_pages, page_rects=None, audit_log=None
) -> int:
    """Complete sentences broken by a page turn on the page where they start.

    When page N's last BODY paragraph stops mid-sentence, the sentence's tail
    is taken from the top of page N+1 — either a standalone stranded fragment
    (``exchange.4``) or the leading words of the first body paragraph up to
    its first sentence boundary (``high frequency traders. Because ...``) —
    APPENDED to page N's paragraph so the sentence translates and renders as
    one unit where it started, and removed from page N+1 so that page begins
    at its first complete sentence.  Appending never rewrites existing markup,
    so marker-bearing source paragraphs participate safely; a destination is
    split only when its rich text is plain.  Footnotes, footers, headers and
    headings never participate; a next page that opens with a heading means
    the sentence was complete."""
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions)
    absorbed = 0

    def _record(source_page, source_idx, source_elem, dest_idx, dest_elem, tail):
        if audit_log is None:
            return
        audit_log.append(
            {
                "kind": "cross-page-orphan-tail",
                "source_page": source_page + 1,
                "destination_page": source_page + 2,
                "source_element": source_idx,
                "destination_element": dest_idx,
                "decision": "accepted",
                "reason": "accepted",
                "carried_tail": tail,
                "source_layout": source_elem.get("layout_class"),
                "destination_layout": dest_elem.get("layout_class"),
            }
        )

    for page_num in range(total_pages - 1):
        info = page_extractions.get(page_num, {})
        next_info = page_extractions.get(page_num + 1, {})
        if "elements" not in info or "elements" not in next_info:
            continue
        page_rect = page_rects.get(page_num) if page_rects else None
        next_page_rect = page_rects.get(page_num + 1) if page_rects else None
        if _cross_page_geometry_rejection(page_rect, next_page_rect):
            continue

        ordered = sorted(
            enumerate(info["elements"]),
            key=lambda item: (item[1].get("y", 0.0), item[1].get("x", 0.0)),
        )
        body_indices = [
            (index, elem)
            for index, elem in ordered
            if _is_cross_page_body_elem(
                elem,
                page_rect,
                dominant_fontsize,
                allow_superscript_markers=True,
            )
        ]
        if not body_indices:
            continue
        last_idx, last_elem = body_indices[-1]
        source_plain = re.sub(
            r"\s+", " ", _plain_text(last_elem.get("content", ""))
        ).strip()
        if not source_plain or _ends_with_sentence_boundary(source_plain):
            continue
        if _looks_like_right_aligned_signoff(last_elem, page_rect):
            continue
        if page_rect is not None:
            last_rect = _get_pdf_elem_rect(last_elem)
            if last_rect.y1 < page_rect.y0 + page_rect.height * 0.45:
                continue

        next_ordered = sorted(
            enumerate(next_info["elements"]),
            key=lambda item: (item[1].get("y", 0.0), item[1].get("x", 0.0)),
        )
        next_text_elements = [
            (index, elem)
            for index, elem in next_ordered
            if elem.get("type") == "text"
            and _plain_text(elem.get("content", "")).strip()
        ]
        if not next_text_elements:
            continue
        first_idx, first_elem = next_text_elements[0]
        heading_above = any(
            _is_cross_page_heading_elem(elem, dominant_fontsize)
            and elem.get("y", 0.0) < first_elem.get("y", 0.0)
            for _index, elem in next_ordered
        )
        if heading_above:
            continue

        source_size = float(last_elem.get("fontsize", dominant_fontsize))
        first_size = float(first_elem.get("fontsize", dominant_fontsize))
        if abs(source_size - first_size) > max(
            0.8, max(source_size, first_size) * 0.10
        ):
            continue

        fragment = _cross_page_whole_fragment_tail(first_elem)
        if fragment is not None:
            tail_plain, tail_rich = fragment
            if page_rect is not None and next_page_rect is not None:
                source_rect = _get_pdf_elem_rect(last_elem)
                orphan_rect = _get_pdf_elem_rect(first_elem)
                source_x0 = (source_rect.x0 - page_rect.x0) / max(
                    page_rect.width, 1.0
                )
                source_x1 = (source_rect.x1 - page_rect.x0) / max(
                    page_rect.width, 1.0
                )
                orphan_x0 = (orphan_rect.x0 - next_page_rect.x0) / max(
                    next_page_rect.width, 1.0
                )
                orphan_x1 = (orphan_rect.x1 - next_page_rect.x0) / max(
                    next_page_rect.width, 1.0
                )
                # A sentence tail wraps back to the column left edge and
                # cannot be wider than the column it continues.
                if (
                    abs(orphan_x0 - source_x0) > 0.03
                    or orphan_x1 > source_x1 + 0.03
                ):
                    continue
            last_elem["content"] = (
                last_elem["content"].rstrip() + " " + tail_plain
            )
            if last_elem.get("rich_content"):
                last_elem["rich_content"] = (
                    last_elem["rich_content"].rstrip() + " " + tail_rich
                )
            elif tail_rich != tail_plain:
                base = last_elem["content"][: -len(" " + tail_plain)]
                last_elem["rich_content"] = base + " " + tail_rich
            first_elem["type"] = "text_merged_away"
            absorbed += 1
            log.info(
                "Page %s: absorbed stranded sentence tail %r from page %s",
                page_num + 1,
                tail_plain[:60],
                page_num + 2,
            )
            _record(
                page_num, last_idx, last_elem, first_idx, first_elem, tail_plain
            )
            continue

        # Split the leading tail out of the first body paragraph.
        if not _is_cross_page_body_elem(
            first_elem,
            next_page_rect,
            dominant_fontsize,
        ):
            continue
        first_plain = re.sub(
            r"\s+", " ", _plain_text(first_elem.get("content", ""))
        ).strip()
        if not _first_cross_page_lexical_char(first_plain).islower():
            continue
        rich = first_elem.get("rich_content")
        if rich is not None and str(rich).strip() != str(
            first_elem.get("content", "")
        ).strip():
            continue
        if _cross_page_pair_incompatibility(
            last_elem,
            first_elem,
            page_rect,
            next_page_rect,
            dominant_fontsize,
        ):
            continue
        split = _find_leading_sentence_tail(
            first_plain,
            CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS,
        )
        if split is None:
            continue
        tail_plain, rest = split
        last_elem["content"] = last_elem["content"].rstrip() + " " + tail_plain
        if last_elem.get("rich_content"):
            last_elem["rich_content"] = (
                last_elem["rich_content"].rstrip() + " " + tail_plain
            )
        if rest:
            first_elem["content"] = rest
            first_elem["rich_content"] = None
        else:
            first_elem["type"] = "text_merged_away"
        absorbed += 1
        log.info(
            "Page %s: absorbed leading sentence tail %r from page %s",
            page_num + 1,
            tail_plain[:60],
            page_num + 2,
        )
        _record(page_num, last_idx, last_elem, first_idx, first_elem, tail_plain)
    return absorbed


def _cross_page_orphan_tail_fragment(elem: dict) -> str | None:
    """Compatibility wrapper returning only the plain fragment text."""
    fragment = _cross_page_whole_fragment_tail(elem)
    return fragment[0] if fragment else None


def _absorb_cross_page_orphan_tails(
    page_extractions,
    total_pages,
    page_rects=None,
    audit_log=None,
) -> int:
    """Compatibility alias: the backward absorber is the merge itself now."""
    return _merge_cross_page_sentences(
        page_extractions,
        total_pages,
        page_rects,
        audit_log,
    )
