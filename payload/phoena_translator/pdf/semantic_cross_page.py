"""Semantic cross page helpers for deterministic PDF processing."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

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
    allow_short_text: bool = False,
    allow_heading_shape: bool = False,
    allowed_layout_classes: tuple[str, ...] = ("body",),
) -> bool:
    """Cross-page merge candidates must be real body prose. Headers, footers,
    footnotes and page numbers never participate: they sit in the page's edge
    bands or use smaller type than the document's dominant body size, and a
    heading (larger type) can never be the middle of a sentence.

    Marker-bearing paragraphs are excluded by default because the forward
    carry rewrites ``rich_content`` wholesale and would flatten superscripts.
    The orphan-tail absorber only APPENDS at the end, which cannot damage
    markers, so it opts in via ``allow_superscript_markers``.

    A page-top continuation paragraph is often classified ``scattered``
    rather than ``body`` (it sits alone above the page's first heading), so
    merge destinations opt in via ``allowed_layout_classes`` — the same
    policy `_cross_page_whole_fragment_tail` already applies; the column
    geometry, font-size and lowercase-start gates still exclude captions
    and labels."""
    if elem.get("type") != "text" or elem.get("skip_translate_reason"):
        return False
    if elem.get("layout_class") not in allowed_layout_classes:
        return False
    if (
        elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
        or elem.get("footnote_hint")
    ):
        return False
    if not allow_heading_shape and _is_heading_like_elem(elem):
        return False
    text = _plain_text(elem.get("content", "")).strip()
    if not text or _is_disclaimer_block(text):
        return False
    if (
        not allow_short_text
        and len(re.findall(r"[A-Za-z]{2,}", text)) < 2
    ):
        return False
    if allow_short_text and not re.search(
        r"[A-Za-z0-9\u4e00-\u9fff]",
        text,
    ):
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
    if not allow_heading_shape and fontsize >= dominant_fontsize * 1.18:
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


_CROSS_PAGE_CAPTION_LEAD_RE = re.compile(
    r"^(?:Table|Figure|Graph|Panel|Appendix|Exhibit|Chart|Box|Note|Notes|Source|"
    r"Sources|Section|Chapter)\b"
    r"|^[IVXLCDM]+[.\uff0e]"
)
_CROSS_PAGE_DIRECT_LABEL_RE = re.compile(
    r"^(?:"
    r"(?:Table|Figure|Graph|Panel|Appendix|Exhibit|Chart|Box)"
    r"\s+(?:[A-Z]|\d|[IVXLCDM])[\w.\-]*"
    r"(?:\s*$|\s*[:.\-–—]\s*\S.*$)"
    r"|(?:Note|Notes|Source|Sources)\s*:.*"
    r")$",
    re.IGNORECASE,
)
_CROSS_PAGE_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)+|[IVXLCDM]+[.\uff0e])\s+\S",
    re.IGNORECASE,
)

_CROSS_PAGE_AMBIGUOUS_TERMINAL_ABBREVIATION_RE = re.compile(
    r"\bet\s+al\.\s*[\)\]\}\"'’”]*$",
    re.IGNORECASE,
)
_CROSS_PAGE_NON_BODY_SECTION_RE = re.compile(
    r"^(?:glossary|bibliography|references|index|contents?|"
    r"list of (?:figures|tables)|abbreviations)$",
    re.IGNORECASE,
)


def _cross_page_text_line_count(elem: dict) -> int:
    paragraphs = elem.get("paragraphs") or []
    source_lines = sum(
        len(
            [
                line
                for line in (paragraph.get("source_lines") or [])
                if _plain_text(line.get("plain", "")).strip()
            ]
        )
        for paragraph in paragraphs
    )
    if source_lines:
        return source_lines
    return len(
        [
            line
            for line in str(elem.get("content", "")).splitlines()
            if line.strip()
        ]
    ) or 1


def _cross_page_normalized_label(text: str) -> str:
    return re.sub(r"\s+", " ", _plain_text(text or "")).strip().casefold()


def _is_cross_page_non_body_context(
    elem: dict,
    page_elements: list[dict],
    dominant_fontsize: float,
    *,
    previous_page_elements: list[dict] | None = None,
) -> bool:
    """Reject chart titles, table labels, and repeated running labels.

    PDF extraction regularly classifies a chart title as ``scattered``. It
    can sit above the actual first body paragraph and even end in ``?``, which
    makes lexical sentence tests alone unsafe. A short scattered line in the
    title band immediately above a labelled table/graph is non-body. Exact
    text repeated on the previous page is likewise a heading/running label,
    not the continuation sentence.
    """
    text = _plain_text(elem.get("content", "")).strip()
    if not text:
        return True
    if _CROSS_PAGE_DIRECT_LABEL_RE.match(text):
        return True
    if (
        elem.get("layout_class") == "scattered"
        and len(text) <= 180
        and _cross_page_text_line_count(elem) <= 2
        and _CROSS_PAGE_NUMBERED_HEADING_RE.match(text)
    ):
        return True

    for page_element in page_elements:
        if page_element.get("type") != "text":
            continue
        page_heading = _plain_text(page_element.get("content", "")).strip()
        if not _CROSS_PAGE_NON_BODY_SECTION_RE.fullmatch(page_heading):
            continue
        try:
            heading_rect = fitz.Rect(
                page_element.get("bbox", page_element.get("rect"))
            )
        except (TypeError, ValueError, AssertionError):
            continue
        if (
            heading_rect.y0
            <= min(
                140.0,
                max(float(dominant_fontsize), 1.0) * 14.0,
            )
            and float(page_element.get("fontsize", dominant_fontsize))
            >= dominant_fontsize * 1.1
        ):
            return True

    normalized = _cross_page_normalized_label(text)
    if len(normalized) >= 8:
        for previous in previous_page_elements or []:
            if previous is elem or previous.get("type") != "text":
                continue
            previous_text = _cross_page_normalized_label(
                previous.get("content", "")
            )
            if (
                normalized == previous_text
                or (
                    len(normalized) >= 16
                    and re.search(
                        rf"(?<!\w){re.escape(normalized)}(?!\w)",
                        previous_text,
                    )
                )
            ):
                return True

    if (
        elem.get("layout_class") != "scattered"
        or _cross_page_text_line_count(elem) > 1
    ):
        return False
    try:
        candidate_rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
    except (TypeError, ValueError, AssertionError):
        return False
    band = max(float(dominant_fontsize), 1.0) * 6.0
    nearby_table = False
    nearby_label = False
    for other in page_elements:
        if other is elem or other.get("type") != "text":
            continue
        try:
            other_rect = fitz.Rect(other.get("bbox", other.get("rect")))
        except (TypeError, ValueError, AssertionError):
            continue
        if (
            other_rect.y0 >= candidate_rect.y0 - dominant_fontsize
            and other_rect.y0 <= candidate_rect.y0 + band
            and _CROSS_PAGE_DIRECT_LABEL_RE.match(
                _plain_text(other.get("content", "")).strip()
            )
        ):
            nearby_label = True
        if (
            (other.get("table_hint") or other.get("layout_class") == "table")
            and other_rect.y0 >= candidate_rect.y0
            and other_rect.y0 <= candidate_rect.y0 + band
        ):
            nearby_table = True
    return nearby_label and nearby_table


class SourceContinuationEvidence(StrEnum):
    """Stable values written to cross-page audit records."""

    DANGLING = "dangling"
    AMBIGUOUS_TERMINAL_ABBREVIATION = "ambiguous-terminal-abbreviation"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ContinuationDecision:
    """Typed immutable source evidence for one page-edge decision."""

    evidence: SourceContinuationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SourceContinuationEvidence):
            raise TypeError("continuation evidence must be SourceContinuationEvidence")

    @property
    def source_dangling(self) -> bool:
        return self.evidence is SourceContinuationEvidence.DANGLING

    @property
    def audit_value(self) -> str:
        return self.evidence.value


def _cross_page_continuation_decision(text: str) -> ContinuationDecision:
    """Classify immutable page-end evidence before considering the next page.

    This records only the previous page's signal.  The final decision follows
    the historical two-signal OR policy: a dangling source OR a next-page
    body opener whose first alphabetic character is not capitalized.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    boundary_text = _strip_trailing_footnote_marker(plain)
    if not _ends_with_sentence_boundary(boundary_text):
        return ContinuationDecision(SourceContinuationEvidence.DANGLING)
    if _CROSS_PAGE_AMBIGUOUS_TERMINAL_ABBREVIATION_RE.search(boundary_text):
        return ContinuationDecision(
            SourceContinuationEvidence.AMBIGUOUS_TERMINAL_ABBREVIATION
        )
    return ContinuationDecision(SourceContinuationEvidence.COMPLETE)


def _cross_page_source_continuation_reason(text: str) -> str:
    """Compatibility facade returning the historical serialized value."""

    return _cross_page_continuation_decision(text).audit_value


def _cross_page_tail_start_acceptable(
    text: str,
    *,
    source_dangling: bool,
    source_ends_capitalized: bool = False,
) -> bool:
    """Apply the required two-signal OR policy to a page-turn pair.

    Signal 1 is the previous page's final body sentence ending without
    non-comma punctuation. Signal 2 is the next page's first body text
    opening with a non-capitalized first alphabetic character. Once either
    signal is present, lexical shape must not veto the merge; headings,
    footnotes, watermarks and other non-body objects are filtered before this
    helper is called.

    ``source_ends_capitalized`` remains in the signature for compatibility
    with older callers and serialized tests, but the OR rule does not need it.
    """
    plain = _plain_text(text or "")
    first_alpha = re.search(r"[A-Za-z]", plain)
    destination_not_capitalized = bool(
        first_alpha and first_alpha.group(0).islower()
    )
    if first_alpha is None:
        first_lexical = _first_cross_page_lexical_char(plain)
        destination_not_capitalized = bool(
            first_lexical and not first_lexical.isupper()
        )
    return bool(source_dangling or destination_not_capitalized)


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
CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS = 4000
CROSS_PAGE_UNTERMINATED_TAIL_MAX_CHARS = 4000
CROSS_PAGE_WEAK_DESTINATION_TOP_RATIO = 0.32
_TRAILING_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:\d{1,3}|[*†‡§¶]{1,3}|[\ue000-\uf8ff])$"
)
_CROSS_PAGE_NONTERMINAL_ABBREVIATION_RE = re.compile(
    r"(?:"
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|Nos|Fig|Figs|Eq|Eqs|"
    r"Sec|Secs|Ch|Vol|vs|etc|e\.g|i\.e|et\s+al)\."
    r"|(?:[A-Z]\.){2,}"
    r")$",
    re.IGNORECASE,
)
_CROSS_PAGE_BOUNDARY_CLOSERS = "\"'’”)]}》〉」』"
_CROSS_PAGE_INLINE_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:\d{1,3}|[*†‡§¶]{1,3}|[\ue000-\uf8ff])"
)


def _strip_trailing_footnote_marker(text: str) -> str:
    """Drop one trailing footnote marker so boundary tests see the sentence."""
    stripped = (text or "").rstrip()
    without = _TRAILING_FOOTNOTE_MARKER_RE.sub("", stripped).rstrip()
    return without or stripped


def _cross_page_weak_destination_is_near_page_start(
    elem: dict,
    page_rect: fitz.Rect | None,
) -> bool:
    """Constrain lowercase-only continuation evidence to the page top.

    A dangling source sentence is strong evidence and may legitimately resume
    below a page-opening chart.  A *complete* source sentence is different:
    its only evidence is a lowercase destination opener, which can also be a
    wrapped line inside the destination page.  In that weak-evidence case,
    accepting a mid-page element moves an unrelated local sentence backward
    by one page.

    ``page_rect`` is always available in the production extraction pipeline.
    The ``None`` compatibility path preserves direct legacy callers that do
    not supply page geometry.
    """
    if page_rect is None:
        return True
    try:
        rect = _get_pdf_elem_rect(elem)
    except (TypeError, ValueError, AssertionError):
        return False
    page_height = max(float(page_rect.height), 1.0)
    top_ratio = (float(rect.y0) - float(page_rect.y0)) / page_height
    return top_ratio <= CROSS_PAGE_WEAK_DESTINATION_TOP_RATIO


def _cross_page_destination_has_local_predecessor(
    destination_index: int,
    page_elements: list[dict],
    dominant_fontsize: float,
) -> bool:
    """Return whether a nearby page-local line owns the destination text.

    This is the final guard against a page-local paragraph split.  A prose
    line immediately below an incomplete, same-size, same-column text line
    belongs to that local paragraph even if the upper line was separated by
    a bold-to-regular style boundary or accidentally labelled as a table
    cell.  It must never be consumed as the previous page's orphan tail.
    """
    if isinstance(destination_index, bool):
        return False
    try:
        resolved_index = int(destination_index)
        destination = page_elements[resolved_index]
        destination_rect = _get_pdf_elem_rect(destination)
    except (IndexError, TypeError, ValueError, AssertionError):
        return False
    if destination.get("type") != "text":
        return False

    destination_size = max(
        float(destination.get("fontsize", dominant_fontsize)),
        1.0,
    )
    for index, previous in enumerate(page_elements):
        if index == resolved_index or previous.get("type") != "text":
            continue
        previous_text = re.sub(
            r"\s+",
            " ",
            _plain_text(previous.get("content", "")),
        ).strip()
        if (
            len(previous_text) < 20
            or _ends_with_sentence_boundary(
                _strip_trailing_footnote_marker(previous_text)
            )
            or previous.get("non_horizontal")
            or previous.get("footnote_hint")
            or previous.get("skip_translate_reason") == "watermark"
        ):
            continue
        try:
            previous_rect = _get_pdf_elem_rect(previous)
        except (TypeError, ValueError, AssertionError):
            continue
        previous_size = max(
            float(previous.get("fontsize", dominant_fontsize)),
            1.0,
        )
        if abs(previous_size - destination_size) > max(
            previous_size * 0.10,
            0.8,
        ):
            continue

        vertical_gap = float(destination_rect.y0) - float(previous_rect.y1)
        if vertical_gap < -min(previous_rect.height, destination_rect.height) * 0.45:
            continue
        if vertical_gap > max(previous_size * 0.85, 8.0):
            continue
        if float(previous_rect.y0) >= float(destination_rect.y0):
            continue

        overlap = max(
            0.0,
            min(previous_rect.x1, destination_rect.x1)
            - max(previous_rect.x0, destination_rect.x0),
        )
        if overlap / max(min(previous_rect.width, destination_rect.width), 1.0) < 0.55:
            continue
        if abs(float(previous_rect.x0) - float(destination_rect.x0)) > max(
            previous_size * 1.2,
            12.0,
        ):
            continue
        return True
    return False


def _cross_page_source_already_ends_with_tail(
    source_text: str,
    tail_text: str,
) -> bool:
    """Detect duplicate extraction with or without a trailing footnote marker."""
    source = re.sub(r"\s+", " ", source_text or "").strip()
    tail = re.sub(r"\s+", " ", tail_text or "").strip()
    tail_without_marker = _strip_trailing_footnote_marker(tail)
    return bool(
        tail
        and (
            source.endswith(tail)
            or (
                tail_without_marker != tail
                and source.endswith(tail_without_marker)
            )
        )
    )


def _find_leading_sentence_tail(
    plain: str,
    max_chars: int,
) -> tuple[str, str] | None:
    """Split the next page at its first complete sentence.

    Commas never close the carried sentence. Other terminal punctuation
    closes it at the end of the body block, or when the next lexical character
    is capitalized (also accepting digits and CJK openers). Common
    abbreviations and dotted initialisms are not mistaken for the boundary.
    A directly attached footnote marker belongs to the carried sentence.
    """
    for index, char in enumerate(plain):
        category = unicodedata.category(char)
        if (
            char in ",，、"
            or not category.startswith("P")
            or category in {"Ps", "Pi", "Pd", "Pc"}
        ):
            continue

        end = index + 1
        while end < len(plain) and plain[end] in _CROSS_PAGE_BOUNDARY_CLOSERS:
            end += 1

        marker_match = _CROSS_PAGE_INLINE_FOOTNOTE_MARKER_RE.match(plain, end)
        if marker_match:
            marker_end = marker_match.end()
            if marker_end == len(plain) or plain[marker_end].isspace():
                end = marker_end

        if end > max_chars:
            return None
        rest = plain[end:]
        if rest and not rest[:1].isspace():
            continue
        rest_stripped = rest.lstrip()

        prefix_through_punctuation = plain[: index + 1].rstrip()
        if (
            rest_stripped
            and char == "."
            and _CROSS_PAGE_NONTERMINAL_ABBREVIATION_RE.search(
                prefix_through_punctuation
            )
        ):
            continue

        if rest_stripped:
            next_lexical = _first_cross_page_lexical_char(rest_stripped)
            if not next_lexical or not (
                next_lexical.isupper()
                or next_lexical.isdigit()
                or ord(next_lexical) > 0x2E7F
            ):
                continue

        tail = plain[:end].strip()
        if not tail:
            return None
        return tail, rest_stripped
    return None


_CROSS_PAGE_LIGHTWEIGHT_TAG_RE = re.compile(
    r"(?is)<\s*(?P<closing>/?)\s*"
    r"(?P<name>b|strong|i|em|sup|sub|span|a)\b[^>]*>"
)


def _split_cross_page_rich_prefix(
    rich: str,
    tail_plain: str,
    rest_plain: str,
) -> tuple[str, str] | None:
    """Split a rich destination at the proven plain-text boundary.

    PDF body paragraphs commonly carry footnote markers as ``<sup>`` markup.
    Walk the lightweight markup while counting only visible characters, then
    accept the split only when both outputs are balanced and reproduce the
    exact plain-text split. Markup spanning the boundary remains rejected.
    """
    normalized_rich = re.sub(r"\s+", " ", str(rich or "")).strip()
    expected_plain = tail_plain + ((" " + rest_plain) if rest_plain else "")
    if (
        not normalized_rich
        or re.sub(r"\s+", " ", _plain_text(normalized_rich)).strip()
        != expected_plain
    ):
        return None

    boundary = len(tail_plain)
    cursor = 0
    visible = 0
    open_tags: list[str] = []
    while cursor < len(normalized_rich) and visible < boundary:
        tag = _CROSS_PAGE_LIGHTWEIGHT_TAG_RE.match(normalized_rich, cursor)
        if tag:
            name = tag.group("name").casefold()
            token = tag.group(0)
            if tag.group("closing"):
                if not open_tags or open_tags[-1] != name:
                    return None
                open_tags.pop()
            elif not re.search(r"/\s*>$", token):
                open_tags.append(name)
            cursor = tag.end()
            continue
        cursor += 1
        visible += 1

    if visible != boundary:
        return None

    # A boundary immediately after a superscript marker sits before its
    # closing tag in the raw rich string. Carry those closing tags with the
    # visible prefix. Any remaining open tag spans the sentence boundary.
    while cursor < len(normalized_rich):
        tag = _CROSS_PAGE_LIGHTWEIGHT_TAG_RE.match(normalized_rich, cursor)
        if not tag or not tag.group("closing"):
            break
        name = tag.group("name").casefold()
        if not open_tags or open_tags[-1] != name:
            return None
        open_tags.pop()
        cursor = tag.end()
    if open_tags:
        return None

    tail_rich = normalized_rich[:cursor].rstrip()
    rest_rich = normalized_rich[cursor:].lstrip()
    if (
        re.sub(r"\s+", " ", _plain_text(tail_rich)).strip() != tail_plain
        or re.sub(r"\s+", " ", _plain_text(rest_rich)).strip() != rest_plain
        or (
            _pdf_superscript_signature(tail_rich)
            + _pdf_superscript_signature(rest_rich)
            != _pdf_superscript_signature(normalized_rich)
        )
    ):
        return None
    return tail_rich, rest_rich


def _cross_page_whole_fragment_tail(
    elem: dict,
    *,
    source_dangling: bool = False,
    source_ends_capitalized: bool = False,
) -> tuple[str, str] | None:
    """Return (plain, rich) when an element is exactly one stranded tail.

    A stranded tail is the end of a sentence broken by a page turn: short,
    reaching a sentence boundary, and opening per the two-signal policy of
    `_cross_page_tail_start_acceptable`.  A trailing footnote marker
    (``exchange.4``, ``respectively.10``) is tolerated and carried along
    inside the rich text so the reference survives next to its sentence."""
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
    if not _cross_page_tail_start_acceptable(
        plain,
        source_dangling=source_dangling,
        source_ends_capitalized=source_ends_capitalized,
    ):
        return None
    split = _find_leading_sentence_tail(
        plain,
        CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS,
    )
    if split is None or split[1]:
        return None
    if not _ends_with_sentence_boundary(
        _strip_trailing_footnote_marker(split[0])
    ):
        return None
    words = re.findall(r"[A-Za-z]{2,}", plain)
    if not 1 <= len(words) <= CROSS_PAGE_ORPHAN_TAIL_MAX_WORDS:
        return None
    rich = elem.get("rich_content") or elem.get("content", "")
    return plain, " ".join(str(rich).split())


def _cross_page_unterminated_fragment_tail(
    elem: dict,
    next_body_ordered: list,
    elem_idx: int,
    *,
    source_dangling: bool,
    source_last_word: str,
) -> tuple[str, str] | None:
    """Whole-fragment absorption for a tail whose final period is missing.

    Some sources omit the sentence-final punctuation (a typo in the original
    document). The page-top fragment is still the carried sentence when the
    pair satisfies either OR signal and the next BODY element starts a new
    capitalized sentence."""
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
    if not plain or len(plain) > CROSS_PAGE_UNTERMINATED_TAIL_MAX_CHARS:
        return None
    if _ends_with_sentence_boundary(_strip_trailing_footnote_marker(plain)):
        return None  # terminated fragments take the standard paths
    if not _cross_page_tail_start_acceptable(
        plain,
        source_dangling=source_dangling,
        source_ends_capitalized=bool(source_last_word[:1].isupper()),
    ):
        return None
    if len(re.findall(r"[A-Za-z]{2,}", plain)) < 2:
        return None
    followers = [
        other
        for index, other in next_body_ordered
        if index != elem_idx
        and other.get("type") == "text"
        and _plain_text(other.get("content", "")).strip()
        and float(other.get("y", 0.0)) > float(elem.get("y", 0.0))
    ]
    if not followers:
        return None
    follower_char = _first_cross_page_lexical_char(
        _plain_text(followers[0].get("content", ""))
    )
    if not follower_char or not (
        follower_char.isupper()
        or follower_char.isdigit()
        or ord(follower_char) > 0x2E7F
    ):
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
    at its first complete sentence. Appending never rewrites existing markup,
    so marker-bearing source paragraphs participate safely. A rich destination
    is split only at an exact, balanced lightweight-markup boundary, preserving
    footnote superscripts on both sides. Footnotes, footers, headers, headings
    and watermarks never participate; they are skipped while locating each
    page's first and last body text."""
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions)
    absorbed = 0

    def _record(
        source_page,
        source_idx,
        source_elem,
        dest_idx,
        dest_elem,
        tail,
        *,
        source_continuation_reason,
        unterminated=False,
    ):
        if audit_log is None:
            return
        entry = {
            "kind": "cross-page-orphan-tail",
            "source_page": source_page + 1,
            "destination_page": source_page + 2,
            "source_element": source_idx,
            "destination_element": dest_idx,
            "decision": "accepted",
            "reason": "accepted",
            "source_continuation_reason": source_continuation_reason,
            "carried_tail": tail,
            "source_layout": source_elem.get("layout_class"),
            "destination_layout": dest_elem.get("layout_class"),
        }
        if unterminated:
            entry["tail_unterminated"] = True
        audit_log.append(entry)

    for page_num in range(total_pages - 1):
        info = page_extractions.get(page_num, {})
        next_info = page_extractions.get(page_num + 1, {})
        if "elements" not in info or "elements" not in next_info:
            continue
        page_rect = page_rects.get(page_num) if page_rects else None
        next_page_rect = page_rects.get(page_num + 1) if page_rects else None
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
        if not source_plain:
            continue
        # Preserve the previous page's signal, then apply it together with the
        # destination opener as an OR. A complete source remains eligible only
        # for a genuine page-top opener; lowercase text may instead be a local
        # wrapped line split off by a bold/style or table-classification
        # boundary.
        continuation_decision = _cross_page_continuation_decision(source_plain)
        source_dangling = continuation_decision.source_dangling
        source_words = re.findall(r"[A-Za-z][A-Za-z'’-]*", source_plain)
        source_last_word = source_words[-1] if source_words else ""
        source_ends_capitalized = bool(
            source_dangling and source_last_word[:1].isupper()
        )
        if _looks_like_right_aligned_signoff(last_elem, page_rect):
            continue

        next_ordered = sorted(
            enumerate(next_info["elements"]),
            key=lambda item: (item[1].get("y", 0.0), item[1].get("x", 0.0)),
        )
        next_body_elements = [
            (index, elem)
            for index, elem in next_ordered
            if _is_cross_page_body_elem(
                elem,
                next_page_rect,
                dominant_fontsize,
                allow_superscript_markers=True,
                allow_short_text=True,
                allowed_layout_classes=("body", "scattered"),
            )
            and not _is_cross_page_non_body_context(
                elem,
                next_info["elements"],
                dominant_fontsize,
                previous_page_elements=info["elements"],
            )
        ]
        if not next_body_elements:
            continue
        first_idx, first_elem = next_body_elements[0]

        if _cross_page_destination_has_local_predecessor(
            first_idx,
            next_info["elements"],
            dominant_fontsize,
        ):
            log.warning(
                "Page %s: kept lowercase body fragment on page %s because "
                "an adjacent local line owns the continuation",
                page_num + 2,
                page_num + 2,
            )
            continue
        if (
            not source_dangling
            and _cross_page_tail_start_acceptable(
                first_elem.get("content", ""),
                source_dangling=False,
            )
            and not _cross_page_weak_destination_is_near_page_start(
                first_elem,
                next_page_rect,
            )
        ):
            log.warning(
                "Page %s: rejected mid-page lowercase orphan-tail candidate "
                "after a complete source sentence on page %s",
                page_num + 2,
                page_num + 1,
            )
            continue

        if first_elem.get("inline_math_fragments"):
            # Inline-math protection records are element-scoped.  Moving text
            # out of this element would leave records pointing at content the
            # residual no longer holds (deterministic validation failure →
            # source fallback) while the moved tail travels unprotected.
            continue

        fragment = _cross_page_whole_fragment_tail(
            first_elem,
            source_dangling=source_dangling,
            source_ends_capitalized=source_ends_capitalized,
        )
        fragment_unterminated = False
        if fragment is None:
            fragment = _cross_page_unterminated_fragment_tail(
                first_elem,
                next_body_elements,
                first_idx,
                source_dangling=source_dangling,
                source_last_word=source_last_word,
            )
            fragment_unterminated = fragment is not None
        if fragment is not None and _cross_page_source_already_ends_with_tail(
            source_plain,
            fragment[0],
        ):
            # The audit invariant "source original must not already end
            # with the tail" guards against double absorption; enforcing
            # it here keeps producer and auditor aligned instead of
            # failing the task later.
            fragment = None
        if fragment is not None:
            tail_plain, tail_rich = fragment
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
                page_num,
                last_idx,
                last_elem,
                first_idx,
                first_elem,
                tail_plain,
                source_continuation_reason=continuation_decision.audit_value,
                unterminated=fragment_unterminated,
            )
            continue

        # Split the leading tail out of the first body paragraph.  The element
        # already passed the destination-body filter above; do not reapply a
        # stricter short-text / heading-shape heuristic here and accidentally
        # reject the very first body sentence selected by that filter.
        first_plain = re.sub(
            r"\s+", " ", _plain_text(first_elem.get("content", ""))
        ).strip()
        if not _cross_page_tail_start_acceptable(
            first_plain,
            source_dangling=source_dangling,
            source_ends_capitalized=source_ends_capitalized,
        ):
            continue
        rich = first_elem.get("rich_content")
        split = _find_leading_sentence_tail(
            first_plain,
            CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS,
        )
        if split is None:
            continue
        tail_plain, rest = split
        if _cross_page_source_already_ends_with_tail(
            source_plain,
            tail_plain,
        ):
            # Same double-absorption guard as the whole-fragment path.
            continue
        if rest and not _carry_fragment_is_prose(rest):
            # The residual is only redacted when it re-renders as a
            # translation; a non-prose remainder keeps its source ink —
            # including the moved tail's leading line — visible on the next
            # page while the tail also renders translated on this page.
            continue
        tail_rich = tail_plain
        rest_rich = rest
        if rich is not None and str(rich).strip() != str(
            first_elem.get("content", "")
        ).strip():
            rich_split = _split_cross_page_rich_prefix(
                str(rich),
                tail_plain,
                rest,
            )
            if rich_split is None:
                continue
            tail_rich, rest_rich = rich_split

        source_content = last_elem["content"].rstrip()
        last_elem["content"] = source_content + " " + tail_plain
        if last_elem.get("rich_content"):
            last_elem["rich_content"] = (
                last_elem["rich_content"].rstrip() + " " + tail_rich
            )
        elif tail_rich != tail_plain:
            last_elem["rich_content"] = source_content + " " + tail_rich
        if rest:
            first_elem["content"] = rest
            first_elem["rich_content"] = (
                rest_rich if rest_rich != rest else None
            )
        else:
            first_elem["type"] = "text_merged_away"
        absorbed += 1
        log.info(
            "Page %s: absorbed leading sentence tail %r from page %s",
            page_num + 1,
            tail_plain[:60],
            page_num + 2,
        )
        _record(
            page_num,
            last_idx,
            last_elem,
            first_idx,
            first_elem,
            tail_plain,
            source_continuation_reason=continuation_decision.audit_value,
        )
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
