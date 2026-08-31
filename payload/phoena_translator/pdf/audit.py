"""Audit for deterministic PDF processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable
import math
import os
import re
import shutil
import subprocess
from collections import deque
import fitz
from phoena_translator.math_text import (
    is_math_block as _is_math_block,
    unicode_math_symbol_count as _unicode_math_symbol_count,
)
from phoena_translator.pdf.types import (
    PDFStructureValidationError,
    PDF_VECTOR_OCR_DPI,
)
from phoena_translator.pdf.math_detection import (
    _looks_like_pdf_identifier_line,
    _pdf_exact_source_native_superscript_matches,
    _pdf_formula_region_signature,
    _pdf_is_semantic_superscript_marker,
    _pdf_marker_pattern,
    _pdf_nearby_superscript_reference,
    _pdf_span_has_lexical_text,
    _pdf_superscript_signature,
    _plain_text,
)
from phoena_translator.pdf.extraction import _pdf_vector_ocr_normalized_match_text
from phoena_translator.pdf.targets import _pdf_element_requires_translation
from phoena_translator.pdf.cache import (
    _save_pdf_audit,
)
from phoena_translator.pdf.semantic_cross_page import (
    CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS,
    _cross_page_continuation_decision,
    _cross_page_destination_start_evidence,
    _cross_page_destination_start_rejection,
    _cross_page_dominant_fontsize,
    _cross_page_tail_start_acceptable,
    _first_cross_page_lexical_char,
    _is_cross_page_non_body_context,
    _strip_trailing_footnote_marker,
)
from phoena_translator.pdf.semantic_text import (
    _ends_with_sentence_boundary,
    _is_heading_like_elem,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _pdf_extraction_rect_to_display,
)

log = logging.getLogger("translator")

@dataclass(frozen=True)
class PDFAuditDependencies:
    extraction_max_concurrency: int
    extraction_semaphore: Any
    check_output_structure: Callable[..., dict]
    trim_process_memory: Callable[[], None]
    save_progress: Callable[[str, dict], None]
    tasks: dict[str, dict]

def _collect_pdf_superscript_expectations(page_extractions: dict) -> list[dict]:
    expectations = []
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_idx, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != "text":
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
            live_plain = re.sub(
                r"\s+",
                " ",
                _plain_text(elem.get("content") or ""),
            ).strip()
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


def _collect_pdf_formula_expectations(page_extractions: dict) -> list[dict]:
    """Collect hash-only formula expectations suitable for a persisted audit."""
    expectations = []
    allowed_signature_fields = {
        "signature_sha256", "text_sha256", "font_sha256", "geometry_sha256",
        "raster_sha256", "span_count", "raster_width", "raster_height",
    }
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_idx, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != "formula_image":
                continue
            bbox = fitz.Rect(elem.get("bbox", elem.get("rect")))
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


def _collect_pdf_vector_ocr_expectations(page_extractions: dict) -> list[dict]:
    expectations = []
    for page_num in sorted(page_extractions):
        info = page_extractions.get(page_num, {})
        for elem_index, elem in enumerate(info.get("elements", [])):
            if elem.get("type") != "text" or not elem.get("vector_ocr"):
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


def _pdf_vector_ocr_page_text(page) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract is unavailable for vector OCR output audit")
    scale = PDF_VECTOR_OCR_DPI / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
    )
    completed = subprocess.run(
        [
            "tesseract", "stdin", "stdout", "--dpi", str(PDF_VECTOR_OCR_DPI),
            "-l", "eng", "--psm", "11", "txt",
        ],
        input=pixmap.tobytes("png"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=60,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def _pdf_text_element_formula_audit_symbol_count(elem: dict) -> int:
    """Count only genuine math candidates in a translatable text element.

    Extraction deliberately leaves URLs and their query parameters as text so
    a complete citation can be translated as one semantic unit.  Formula QA
    must apply the same identifier exception instead of reclassifying each
    query ``=`` as an unprotected mathematical symbol.
    """
    source_lines = [
        line
        for paragraph in (elem.get("paragraphs") or [])
        for line in (paragraph.get("source_lines") or [])
        if (line.get("plain") or "").strip()
    ]
    if source_lines:
        return sum(
            0
            if _looks_like_pdf_identifier_line(line.get("plain", ""))
            else _unicode_math_symbol_count(
                _plain_text(line.get("plain", ""))
            )
            for line in source_lines
        )
    content = _plain_text(elem.get("content", ""))
    if _looks_like_pdf_identifier_line(content):
        return 0
    return _unicode_math_symbol_count(content)


def _pdf_prose_inline_math_symbol_tolerance(elem: dict) -> int:
    """How many residual unprotected math symbols a prose element may carry.

    A paragraph whose only mathematical content is ``p < 0.01`` or one Greek
    parameter survives translation verbatim; failing the whole page back to
    exact source over it deletes a page of perfectly translatable prose.
    Genuine formula fragments contain few long prose words and keep a zero
    tolerance, so they are still preserved as exact source.
    """
    plain = re.sub(r"\s+", " ", _plain_text(elem.get("content") or "")).strip()
    prose_words = len(re.findall(r"[A-Za-z]{4,}", plain))
    if prose_words < 4:
        return 0
    return min(8, 2 + prose_words // 60)


def _pdf_text_element_unprotected_math_symbols(elem: dict) -> int:
    """Count math symbols in a translatable element not covered by fragments."""
    symbol_count = _pdf_text_element_formula_audit_symbol_count(elem)
    if not symbol_count:
        return 0
    semantic_inline = min(
        symbol_count,
        sum(
            _unicode_math_symbol_count(fragment.get("text", ""))
            for fragment in (elem.get("inline_math_fragments") or [])
            if isinstance(fragment, dict)
        ),
    )
    return max(symbol_count - semantic_inline, 0)


def _preserve_pdf_formula_risk_text_elements(page_extractions: dict) -> list[dict]:
    """Mark math-dense translatable text elements as keep-original.

    Delivery policy is element-granular: rather than preserving a whole page
    as exact source when one text element still carries many unprotected math
    symbols, keep only that element's source ink and translate the rest of
    the page. Elements within the prose tolerance keep translating — one or
    two inline symbols ride through translation as literal characters.

    Runs BEFORE cross-page merging so that both operands of the decision come
    from the same text: the symbol count reads the immutable ``paragraphs``
    while the tolerance reads live ``content``, and a merge that shrinks
    ``content`` would otherwise drop the tolerance while the count still
    reflected the pre-merge block.
    """
    preserved = []
    for page_num in sorted(page_extractions or {}):
        info = page_extractions.get(page_num) or {}
        for elem_index, elem in enumerate(info.get("elements") or []):
            if elem.get("type") != "text":
                continue
            if not _pdf_element_requires_translation(elem):
                continue
            unprotected = _pdf_text_element_unprotected_math_symbols(elem)
            if not unprotected:
                continue
            if unprotected <= _pdf_prose_inline_math_symbol_tolerance(elem):
                continue
            elem["skip_translate_reason"] = "formula_risk_preserved"
            preserved.append({
                "page": int(page_num) + 1,
                "element": elem_index,
                "unprotected_math_symbols": int(unprotected),
                "text": re.sub(
                    r"\s+", " ", _plain_text(elem.get("content") or "")
                ).strip()[:120],
            })
    return preserved


# A stub is the leftover head of a sentence: one or two lines.  The two
# fixtures this rule exists for are 106 and 44 characters.
PDF_FORMULA_MID_SENTENCE_STUB_MAX_CHARS = 150


def _preserve_pdf_formula_mid_sentence_neighbors(
    page_extractions: dict,
) -> list[dict]:
    """Preserve prose whose sentence continues inside a protected formula line.

    Math detection can promote the marker-bearing lines of one paragraph to
    immutable formula regions, leaving the leading prose lines behind as a
    text element that stops mid-sentence ("... the Up phase and the Down
    phase").  Translating that stub alone truncates the sentence, so keep the
    stub as exact source: the whole paragraph then reads coherently in the
    original language, matching the conservative delivery policy.

    Runs BEFORE cross-page merging so the stub is seen with its pristine text
    and, once stamped, is refused as a merge endpoint by
    ``_is_cross_page_body_elem``.  Skipping the stamp for an already-merged
    endpoint would be unsafe here: a mid-sentence stub carries no math
    symbols, so nothing downstream would escalate and the truncated stub this
    function exists to protect would ship translated in isolation.
    """
    preserved = []
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions or {})
    for page_num in sorted(page_extractions or {}):
        info = page_extractions.get(page_num) or {}
        elements = info.get("elements") or []
        # Only a real display region can swallow the end of a sentence.  A
        # ``mixed`` region is an ordinary prose line that math detection
        # promoted because of one inline variable (``mixed = bool(reasons and
        # prose_words)``); the words after the variable ship as English ink
        # either way, so freezing the head in front of it cannot restore the
        # sentence — it only doubles the untranslated area.
        formula_rects = [
            _get_pdf_elem_rect(elem)
            for elem in elements
            if elem.get("type") == "formula_image"
            and not elem.get("math_mixed")
        ]
        if not formula_rects:
            continue
        for elem_index, elem in enumerate(elements):
            if elem.get("type") != "text":
                continue
            if not _pdf_element_requires_translation(elem):
                continue
            # Headings, table cells and rotated labels legitimately end
            # without punctuation; only running prose can continue into a
            # protected formula line.
            if (
                elem.get("table_hint")
                or elem.get("non_horizontal")
                or elem.get("layout_class") == "table"
                or _is_heading_like_elem(elem)
                or float(elem.get("fontsize", dominant_fontsize))
                > dominant_fontsize * 1.15
            ):
                continue
            plain = re.sub(
                r"\s+", " ", _plain_text(elem.get("content") or "")
            ).strip()
            # A footnote marker rides on the last character, so a finished
            # sentence ("... the USD/JPY spot market.18") reads as a stub to a
            # test that looks at the final glyph.  Strip the marker for the
            # boundary question only; ``_ends_with_sentence_boundary`` itself
            # is shared with cross-page merging and must not move.
            terminated = re.sub(r"(?<=[.!?])\s*\d{1,3}\s*$", "", plain)
            if not plain or _ends_with_sentence_boundary(terminated):
                continue
            # The function exists for a stub — a line or two left in front of
            # a display equation.  A whole paragraph that happens to run into
            # one is better translated: freezing it leaves the reader more
            # English than the truncation the stub rule was avoiding.
            if len(plain) > PDF_FORMULA_MID_SENTENCE_STUB_MAX_CHARS:
                continue
            rect = _get_pdf_elem_rect(elem)
            if rect.is_empty:
                continue
            line_height = max(
                float(elem.get("line_height", 0.0) or 0.0),
                float(elem.get("fontsize", 11.0) or 11.0) * 1.2,
                8.0,
            )
            coupled = False
            for formula_rect in formula_rects:
                gap = formula_rect.y0 - rect.y1
                if gap < -line_height * 0.5 or gap > line_height * 1.8:
                    continue
                overlap = min(rect.x1, formula_rect.x1) - max(
                    rect.x0, formula_rect.x0
                )
                if overlap < min(rect.width, formula_rect.width) * 0.5:
                    continue
                # The sentence only runs INTO the formula when nothing
                # translatable stands between them.  An inline math variable
                # splits its own visual line into separate spans, so the tail
                # after the variable is extracted as its own element nested
                # inside the parent's bbox; the parent then looks like it ends
                # mid-sentence while its continuation is ordinary prose two
                # lines above the formula.  Preserving it there is wrong twice
                # over: the prose stays English, and the preserved ink overlaps
                # its own tail, which drags it into the redraw closure and
                # fails the whole page.
                if any(
                    other is not elem
                    and other.get("type") == "text"
                    and not other.get("skip_translate_reason")
                    and _pdf_element_requires_translation(other)
                    and _get_pdf_elem_rect(other).y0 >= rect.y1 - line_height * 0.5
                    and _get_pdf_elem_rect(other).y1 <= formula_rect.y0 + 0.5
                    for other in elements
                ):
                    continue
                coupled = True
                break
            if not coupled:
                continue
            elem["skip_translate_reason"] = "formula_risk_preserved"
            preserved.append({
                "page": int(page_num) + 1,
                "element": elem_index,
                "reason": "mid-sentence-into-formula",
                "text": plain[:120],
            })
    return preserved


def _summarize_pdf_formula_protection(
    expectations: list[dict],
    total_pages: int,
    page_extractions: dict | None = None,
) -> dict:
    per_page = {
        str(page_number): {
            "protected_regions": 0,
            "mixed_regions": 0,
            "reasons": {},
            "candidate_math_symbols": 0,
            "protected_math_symbols": 0,
            "semantic_inline_math_symbols": 0,
            "unprotected_math_symbols_in_text": 0,
            "preserved_source_math_symbols": 0,
            "tolerated_inline_math_symbols": 0,
            "formula_queue_entries": 0,
        }
        for page_number in range(1, total_pages + 1)
    }
    reason_counts: dict[str, int] = {}
    mixed_total = 0
    protected_symbol_total = 0
    for expectation in expectations or []:
        page_key = str(int(expectation.get("page", 0)))
        if page_key not in per_page:
            continue
        per_page[page_key]["protected_regions"] += 1
        if expectation.get("mixed"):
            per_page[page_key]["mixed_regions"] += 1
            mixed_total += 1
        symbol_count = int(expectation.get("candidate_symbol_count", 0))
        protected_symbol_total += symbol_count
        per_page[page_key]["candidate_math_symbols"] += symbol_count
        per_page[page_key]["protected_math_symbols"] += symbol_count
        for reason in expectation.get("reasons") or []:
            reason = str(reason)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            page_reasons = per_page[page_key]["reasons"]
            page_reasons[reason] = page_reasons.get(reason, 0) + 1

    formula_queue_entries = None
    unprotected_symbol_total = None
    semantic_inline_symbol_total = 0
    preserved_symbol_total = 0
    tolerated_symbol_total = 0
    if page_extractions is not None:
        formula_queue_entries = 0
        unprotected_symbol_total = 0
        for page_num, info in page_extractions.items():
            page_key = str(int(page_num) + 1)
            for elem in info.get("elements", []):
                if elem.get("type") == "formula_image":
                    queued = int(_pdf_element_requires_translation(elem))
                    formula_queue_entries += queued
                    if page_key in per_page:
                        per_page[page_key]["formula_queue_entries"] += queued
                elif elem.get("type") == "text":
                    text_symbol_count = _pdf_text_element_formula_audit_symbol_count(
                        elem
                    )
                    if not _pdf_element_requires_translation(elem):
                        # Kept-original elements (labels, disclaimers,
                        # formula-risk preserved text) are never redacted or
                        # rewritten, so their source glyphs cannot be damaged
                        # by translation.  Track them separately instead of
                        # failing the page-level queue check over them.
                        preserved_symbol_total += text_symbol_count
                        if page_key in per_page:
                            per_page[page_key]["candidate_math_symbols"] += text_symbol_count
                            per_page[page_key]["preserved_source_math_symbols"] += text_symbol_count
                        continue
                    semantic_inline = min(
                        text_symbol_count,
                        sum(
                            _unicode_math_symbol_count(fragment.get("text", ""))
                            for fragment in (elem.get("inline_math_fragments") or [])
                            if isinstance(fragment, dict)
                        ),
                    )
                    unprotected = max(text_symbol_count - semantic_inline, 0)
                    tolerated = 0
                    if unprotected and unprotected <= _pdf_prose_inline_math_symbol_tolerance(elem):
                        # A prose paragraph with one or two literal inline
                        # symbols rides through translation unchanged; do not
                        # let it poison the whole page into a source fallback.
                        tolerated = unprotected
                        unprotected = 0
                    semantic_inline_symbol_total += semantic_inline
                    unprotected_symbol_total += unprotected
                    tolerated_symbol_total += tolerated
                    if page_key in per_page:
                        per_page[page_key]["candidate_math_symbols"] += text_symbol_count
                        per_page[page_key]["protected_math_symbols"] += semantic_inline
                        per_page[page_key]["semantic_inline_math_symbols"] += semantic_inline
                        per_page[page_key]["unprotected_math_symbols_in_text"] += unprotected
                        per_page[page_key]["tolerated_inline_math_symbols"] += tolerated
    return {
        "protected_regions": len(expectations or []),
        "mixed_regions": mixed_total,
        "candidate_math_symbols": (
            protected_symbol_total + unprotected_symbol_total
            if unprotected_symbol_total is not None else protected_symbol_total
        ),
        "protected_math_symbols": protected_symbol_total + semantic_inline_symbol_total,
        "semantic_inline_math_symbols": semantic_inline_symbol_total,
        "unprotected_math_symbols_in_text": unprotected_symbol_total,
        "preserved_source_math_symbols": preserved_symbol_total,
        "tolerated_inline_math_symbols": tolerated_symbol_total,
        "protected_regions_queued_for_translation": formula_queue_entries,
        "translation_queue_check": (
            "ok" if formula_queue_entries == 0 and unprotected_symbol_total == 0
            else "warning" if formula_queue_entries is not None
            else "unavailable"
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_page": per_page,
    }


def _pdf_formula_protection_fallback_pages(
    summary: dict,
    page_extractions: dict,
) -> list[int]:
    """Return one-based pages that cannot safely enter the translation queue.

    The formula audit already records page-level queue and unprotected-symbol
    counters.  Delivery recovery must use those counters instead of turning a
    local classification ambiguity into an unscoped whole-document failure.
    If an inconsistent summary reports a warning without a page, preserve all
    extracted content pages: an exact source PDF is preferable to no PDF.
    """
    pages = set()
    for page_key, details in (summary.get("per_page") or {}).items():
        if not isinstance(details, dict):
            continue
        try:
            page_number = int(page_key)
            queued = int(details.get("formula_queue_entries", 0) or 0)
            unprotected = int(
                details.get("unprotected_math_symbols_in_text", 0) or 0
            )
        except (TypeError, ValueError):
            continue
        if page_number > 0 and (queued > 0 or unprotected > 0):
            pages.add(page_number)

    if summary.get("translation_queue_check") != "ok" and not pages:
        pages.update(
            int(page_num) + 1
            for page_num, info in (page_extractions or {}).items()
            if isinstance(info, dict) and "elements" in info
        )
    return sorted(pages)


def _is_accepted_merge_endpoint(
    merge_decisions: list[dict] | None,
    page_number: int,
    element_index: int,
) -> bool:
    """Return whether an element was mutated by an accepted cross-page merge."""
    for decision in merge_decisions or []:
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        for page_key, element_key in (
            ("source_page", "source_element"),
            ("destination_page", "destination_element"),
        ):
            try:
                endpoint = (
                    int(decision.get(page_key, 0) or 0),
                    int(decision.get(element_key, -1)),
                )
            except (TypeError, ValueError):
                continue
            if endpoint == (page_number, element_index):
                return True
    return False


def _expand_pdf_fallback_pages_for_accepted_merges(
    page_numbers: list[int] | set[int],
    merge_decisions: list[dict],
) -> list[int]:
    """Preserve every page in an accepted cross-page merge component.

    Cross-page merging mutates both endpoint elements before the formula gate.
    Falling back only one endpoint could duplicate or drop the carried source
    fragment, so recovery expands through the small accepted-merge graph and
    keeps each connected page exact.
    """
    pages = {int(page) for page in page_numbers if int(page) > 0}
    accepted_edges = []
    for decision in merge_decisions or []:
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        try:
            source_page = int(decision.get("source_page", 0) or 0)
            destination_page = int(decision.get("destination_page", 0) or 0)
        except (TypeError, ValueError):
            continue
        if source_page > 0 and destination_page > 0:
            accepted_edges.append((source_page, destination_page))

    changed = True
    while changed:
        changed = False
        for source_page, destination_page in accepted_edges:
            if source_page in pages or destination_page in pages:
                before = len(pages)
                pages.update((source_page, destination_page))
                changed = changed or len(pages) != before
    return sorted(pages)


def _pdf_merge_original_element_text(elem: dict) -> str:
    """Recover immutable extraction text after cross-page content mutation."""
    paragraph_text = [
        str(paragraph.get("plain") or "")
        for paragraph in (elem.get("paragraphs") or [])
        if isinstance(paragraph, dict) and str(paragraph.get("plain") or "").strip()
    ]
    if paragraph_text:
        return "\n\n".join(paragraph_text)
    return str(elem.get("content") or "")


def _pdf_merge_raw_element(
    page_extractions: dict,
    page_number,
    element_index,
) -> dict | None:
    """Resolve a decision reference to the live (post-merge) element dict."""
    if not isinstance(page_extractions, dict):
        return None
    if isinstance(page_number, bool) or isinstance(element_index, bool):
        return None
    try:
        page_index = int(page_number) - 1
        resolved_element_index = int(element_index)
    except (TypeError, ValueError):
        return None
    if page_index < 0 or resolved_element_index < 0:
        return None
    page_info = page_extractions.get(page_index)
    if page_info is None:
        page_info = page_extractions.get(str(page_index))
    if not isinstance(page_info, dict):
        return None
    elements = page_info.get("elements")
    if not isinstance(elements, list) or resolved_element_index >= len(elements):
        return None
    elem = elements[resolved_element_index]
    return elem if isinstance(elem, dict) else None


def _pdf_merge_audit_element(
    page_extractions: dict,
    page_number,
    element_index,
) -> dict | None:
    """Resolve a decision reference to a copy containing original text."""
    elem = _pdf_merge_raw_element(page_extractions, page_number, element_index)
    if elem is None:
        return None
    original = dict(elem)
    original_text = _pdf_merge_original_element_text(elem)
    original["content"] = original_text
    original["rich_content"] = original_text
    return original


def _pdf_merge_start_evidence(
    decision: dict,
    page_extractions: dict,
) -> tuple[dict, dict, str, str | None] | None:
    source_elem = _pdf_merge_audit_element(
        page_extractions,
        decision.get("source_page"),
        decision.get("source_element"),
    )
    destination_elem = _pdf_merge_audit_element(
        page_extractions,
        decision.get("destination_page"),
        decision.get("destination_element"),
    )
    if source_elem is None or destination_elem is None:
        return None
    start_kind, start_exception = _cross_page_destination_start_evidence(
        source_elem,
        destination_elem.get("content", ""),
    )
    return source_elem, destination_elem, start_kind, start_exception


def _normalized_audit_text(text: str) -> str:
    return re.sub(r"\s+", " ", _plain_text(text or "")).strip()


def _pdf_tails_absorbed_as_merge_source(
    decision: dict,
    merge_decisions: list[dict] | None,
) -> list[str]:
    """Tails this decision's destination went on to absorb as a merge source.

    Cross-page merges chain.  The element that receives one page's orphan tail
    is usually the first body block of its page, which is also frequently the
    last body block -- and therefore the *source* of the next page's merge.
    Its live content is then ``original-minus-tail`` plus whatever it later
    absorbed, so comparing it against ``original-minus-tail`` alone reported a
    false ``orphan-tail-destination-mismatch`` on every such chain.
    """
    try:
        endpoint = (
            int(decision.get("destination_page", 0) or 0),
            int(decision.get("destination_element", -1)),
        )
    except (TypeError, ValueError):
        return []
    absorbed = []
    for other in merge_decisions or []:
        if not isinstance(other, dict) or other.get("decision") != "accepted":
            continue
        if other is decision:
            continue
        try:
            source_endpoint = (
                int(other.get("source_page", 0) or 0),
                int(other.get("source_element", -1)),
            )
        except (TypeError, ValueError):
            continue
        if source_endpoint != endpoint:
            continue
        tail = _normalized_audit_text(str(other.get("carried_tail") or ""))
        if tail:
            absorbed.append(tail)
    return absorbed


def _validate_pdf_orphan_tail_decision(
    decision: dict,
    page_extractions: dict | None,
    merge_decisions: list[dict] | None = None,
) -> str | None:
    """Verify one accepted orphan-tail absorption end to end.

    The load-bearing invariant is single ownership of the carried words: the
    source sentence must now end with the tail, and the orphan element must be
    merged away so the tail can neither duplicate nor vanish."""
    tail = _normalized_audit_text(str(decision.get("carried_tail") or ""))
    tail_unterminated = bool(decision.get("tail_unterminated"))
    if (
        not tail
        or len(tail) > CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS
        or (
            not tail_unterminated
            and not _ends_with_sentence_boundary(
                _strip_trailing_footnote_marker(tail)
            )
        )
        or (
            tail_unterminated
            and _ends_with_sentence_boundary(
                _strip_trailing_footnote_marker(tail)
            )
        )
        # Without merge evidence the dangling-source signal is unknown, so
        # apply the permissive form of the shared two-signal policy here;
        # the strict form runs below once the original source is known.
        or not _cross_page_tail_start_acceptable(
            tail,
            source_dangling=True,
            source_ends_capitalized=True,
        )
    ):
        return "orphan-tail-malformed"
    if page_extractions is None:
        return None
    source = _pdf_merge_raw_element(
        page_extractions,
        decision.get("source_page"),
        decision.get("source_element"),
    )
    destination = _pdf_merge_raw_element(
        page_extractions,
        decision.get("destination_page"),
        decision.get("destination_element"),
    )
    if source is None or destination is None:
        return "merge-evidence-missing"
    if (
        source.get("layout_class") != "body"
        or destination.get("layout_class") not in {"body", "scattered"}
        or source.get("footnote_hint")
        or destination.get("footnote_hint")
        or source.get("table_hint")
        or destination.get("table_hint")
        or source.get("skip_translate_reason")
        or destination.get("skip_translate_reason")
    ):
        return "orphan-tail-non-body"
    try:
        destination_page_index = int(decision.get("destination_page")) - 1
        source_page_index = int(decision.get("source_page")) - 1
    except (TypeError, ValueError):
        return "merge-evidence-missing"
    destination_info = page_extractions.get(destination_page_index)
    if destination_info is None:
        destination_info = page_extractions.get(str(destination_page_index), {})
    source_info = page_extractions.get(source_page_index)
    if source_info is None:
        source_info = page_extractions.get(str(source_page_index), {})
    if _is_cross_page_non_body_context(
        _pdf_merge_audit_element(
            page_extractions,
            decision.get("destination_page"),
            decision.get("destination_element"),
        )
        or destination,
        (
            destination_info.get("elements", [])
            if isinstance(destination_info, dict)
            else []
        ),
        _cross_page_dominant_fontsize(page_extractions),
        previous_page_elements=(
            [
                {
                    **element,
                    "content": _pdf_merge_original_element_text(element),
                }
                for element in source_info.get("elements", [])
                if isinstance(element, dict)
            ]
            if isinstance(source_info, dict)
            else []
        ),
    ):
        return "orphan-tail-non-body"
    if not _normalized_audit_text(source.get("content", "")).endswith(tail):
        return "orphan-tail-not-absorbed"
    source_original = _normalized_audit_text(
        _pdf_merge_original_element_text(source)
    )
    if source_original.endswith(tail):
        return "orphan-tail-source-not-mid-sentence"
    continuation_decision = _cross_page_continuation_decision(source_original)
    reported_continuation_reason = decision.get(
        "source_continuation_reason"
    )
    if (
        reported_continuation_reason is not None
        and reported_continuation_reason != continuation_decision.audit_value
    ):
        return "orphan-tail-source-evidence"
    source_dangling = continuation_decision.source_dangling
    source_words = re.findall(r"[A-Za-z][A-Za-z'’-]*", source_original)
    source_ends_capitalized = bool(
        source_dangling and source_words and source_words[-1][:1].isupper()
    )
    if not _cross_page_tail_start_acceptable(
        tail,
        source_dangling=source_dangling,
        source_ends_capitalized=source_ends_capitalized,
    ):
        return "orphan-tail-start-policy"
    destination_original = _normalized_audit_text(
        _pdf_merge_original_element_text(destination)
    )
    if not destination_original.startswith(tail):
        return "orphan-tail-destination-mismatch"
    remainder = destination_original[len(tail):].strip()
    if destination.get("type") == "text_merged_away":
        if remainder:
            return "orphan-tail-destination-mismatch"
        return None
    if not remainder:
        return "orphan-element-not-merged-away"
    live_destination = _normalized_audit_text(destination.get("content", ""))
    for absorbed_tail in _pdf_tails_absorbed_as_merge_source(
        decision,
        merge_decisions,
    ):
        if live_destination.endswith(absorbed_tail):
            live_destination = live_destination[
                : -len(absorbed_tail)
            ].strip()
    if live_destination != remainder:
        return "orphan-tail-destination-mismatch"
    return None


def _validate_pdf_merge_audit(
    merge_decisions: list[dict],
    page_extractions: dict | None = None,
) -> list[dict]:
    warnings = []
    for index, decision in enumerate(merge_decisions or []):
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        kind = str(decision.get("kind") or "cross-page-merge")
        if kind == "cross-page-orphan-tail":
            orphan_invariant = _validate_pdf_orphan_tail_decision(
                decision,
                page_extractions,
                merge_decisions,
            )
            if decision.get("reason") != "accepted":
                orphan_invariant = orphan_invariant or "accepted-reason"
            if orphan_invariant:
                warnings.append({
                    "type": "merge-invariant-violation",
                    "decision_index": index,
                    "source_page": decision.get("source_page"),
                    "destination_page": decision.get("destination_page"),
                    "invariant": orphan_invariant,
                })
            continue
        if kind != "cross-page-merge":
            warnings.append({
                "type": "merge-invariant-violation",
                "decision_index": index,
                "source_page": decision.get("source_page"),
                "destination_page": decision.get("destination_page"),
                "invariant": "unknown-merge-kind",
            })
            continue
        invariant = None
        if decision.get("reason") != "accepted":
            invariant = "accepted-reason"
        elif (
            decision.get("source_layout") != "body"
            or decision.get("destination_layout") != "body"
        ):
            invariant = "body-layout"
        else:
            reported_start = str(decision.get("destination_start") or "none")
            reported_exception = decision.get("destination_start_exception")
            if page_extractions is not None:
                evidence = _pdf_merge_start_evidence(decision, page_extractions)
                if evidence is None:
                    invariant = "merge-evidence-missing"
                else:
                    source_elem, destination_elem, actual_start, actual_exception = (
                        evidence
                    )
                    if (
                        source_elem.get("layout_class") != "body"
                        or destination_elem.get("layout_class") != "body"
                    ):
                        invariant = "body-layout"
                    elif reported_start != actual_start:
                        invariant = "destination-start-metadata-mismatch"
                    elif reported_exception != actual_exception:
                        invariant = "destination-start-exception-mismatch"
                    else:
                        invariant = _cross_page_destination_start_rejection(
                            actual_start,
                            actual_exception,
                        )
            elif reported_exception is not None:
                invariant = "destination-start-evidence-missing"
            else:
                invariant = _cross_page_destination_start_rejection(
                    reported_start,
                    reported_exception,
                )
            if invariant is None and (
                decision.get("source_orientation")
                and decision.get("destination_orientation")
                and decision.get("source_orientation")
                != decision.get("destination_orientation")
            ):
                invariant = "page-orientation"
        if invariant:
            warnings.append({
                "type": "merge-invariant-violation",
                "decision_index": index,
                "source_page": decision.get("source_page"),
                "destination_page": decision.get("destination_page"),
                "invariant": invariant,
            })
    return warnings


def _pdf_formula_raster_difference_summary(source_pix, observed_pix) -> dict:
    """Classify raster-only differences after vector identity is confirmed.

    Exact raster hashes catch overlays and whiteouts, but they also change when
    translated neighboring text used to overlap a protected box by a few
    antialiased edge pixels. Accept only a tiny global delta or a bounded delta
    confined to an outer edge; material changes in the formula core still fail.
    """
    same_shape = (
        source_pix.width == observed_pix.width
        and source_pix.height == observed_pix.height
        and source_pix.n == observed_pix.n
        and len(source_pix.samples) == len(observed_pix.samples)
    )
    if not same_shape or source_pix.n != 1:
        return {
            "tolerable": False,
            "diff_pixels": None,
            "diff_ratio": None,
            "boundary_only": False,
        }

    width = int(source_pix.width)
    height = int(source_pix.height)
    different = [
        index
        for index, (source_value, observed_value) in enumerate(
            zip(source_pix.samples, observed_pix.samples)
        )
        if source_value != observed_value
    ]
    total = max(width * height, 1)
    diff_count = len(different)
    diff_ratio = diff_count / total
    if diff_count == 0:
        return {
            "tolerable": True,
            "diff_pixels": 0,
            "diff_ratio": 0.0,
            "boundary_only": True,
        }

    vertical_band = max(1, min(10, math.ceil(height * 0.30)))
    horizontal_band = max(1, min(10, math.ceil(width * 0.08)))
    boundary_only = all(
        (index // width) < vertical_band
        or (index // width) >= height - vertical_band
        or (index % width) < horizontal_band
        or (index % width) >= width - horizontal_band
        for index in different
    )
    tolerable = diff_ratio <= 0.002 or (boundary_only and diff_ratio <= 0.10)
    return {
        "tolerable": tolerable,
        "diff_pixels": diff_count,
        "diff_ratio": round(diff_ratio, 6),
        "boundary_only": boundary_only,
    }


def _pdf_glyph_substitution_warnings(
    source_text: str,
    output_text: str,
    page_number: int,
) -> list[dict]:
    """Detect renderer-introduced missing-glyph text on one output page.

    A translated page may remove ASCII question marks, but it must not create
    more of them than the source page contained.  PyMuPDF's built-in fonts use
    literal ``?`` for unsupported CJK glyphs, while some renderers expose the
    Unicode replacement character.  Compare against the source so legitimate
    question punctuation or pre-existing damaged glyphs do not become false
    positives.
    """
    source_text = source_text or ""
    output_text = output_text or ""
    warnings = []
    for character, label in (("?", "ascii-question-mark"), ("\ufffd", "unicode-replacement")):
        source_count = source_text.count(character)
        output_count = output_text.count(character)
        if output_count <= source_count:
            continue
        warning = {
            "type": "font-glyph-substitution",
            "page": int(page_number),
            "glyph": label,
            "source_count": source_count,
            "output_count": output_count,
            "introduced_count": output_count - source_count,
        }
        if character == "?":
            warning["longest_output_run"] = max(
                (len(match.group(0)) for match in re.finditer(r"\?+", output_text)),
                default=0,
            )
        warnings.append(warning)
    return warnings


def _pdf_vector_source_glyph_allowance(
    vector_ocr_expectations: list[dict] | None,
) -> dict[int, dict[str, int]]:
    """Count source glyphs intentionally consumed by vector OCR."""
    allowance_by_page: dict[int, dict[str, int]] = {}
    for expectation in vector_ocr_expectations or []:
        page_number = int(expectation.get("page", 0))
        allowance = allowance_by_page.setdefault(
            page_number,
            {"?": 0, "\ufffd": 0},
        )
        allowance["?"] += int(
            expectation.get("source_ascii_question_marks", 0)
        )
        allowance["\ufffd"] += int(
            expectation.get("source_replacement_characters", 0)
        )
    return allowance_by_page


def _open_pdf_audit_source(
    source_path: str | None,
    source_password: str,
):
    """Open and authenticate the optional source document."""
    if not source_path or not os.path.exists(source_path):
        return None
    source_doc = fitz.open(source_path)
    if source_doc.is_encrypted and not source_doc.authenticate(source_password):
        source_doc.close()
        return None
    return source_doc


def _pdf_audit_page_data(
    doc,
    page_number: int,
    page_cache: dict[int, dict],
    *,
    include_dict: bool,
) -> dict:
    """Load each audit page representation at most once."""
    if page_number not in page_cache:
        page = doc[page_number - 1]
        page_cache[page_number] = {
            "text": page.get_text("text"),
            "dict": None,
        }
    if include_dict and page_cache[page_number].get("dict") is None:
        page_cache[page_number]["dict"] = doc[page_number - 1].get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE,
        )
    return page_cache[page_number]


def _audit_pdf_source_structure(
    doc,
    source_doc,
    vector_glyph_allowance: dict[int, dict[str, int]],
    page_cache: dict[int, dict],
    source_page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> None:
    """Compare page counts and high-risk glyph substitutions."""
    if source_doc is None:
        return
    if len(doc) != len(source_doc):
        warnings.append({
            "type": "page-count-mismatch",
            "source_pages": len(source_doc),
            "output_pages": len(doc),
        })
    for page_number in range(1, min(len(doc), len(source_doc)) + 1):
        output_page = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=False,
        )
        source_page = _pdf_audit_page_data(
            source_doc,
            page_number,
            source_page_cache,
            include_dict=False,
        )
        allowance = vector_glyph_allowance.get(page_number, {})
        glyph_warnings = _pdf_glyph_substitution_warnings(
            source_page["text"]
            + "?" * allowance.get("?", 0)
            + "\ufffd" * allowance.get("\ufffd", 0),
            output_page["text"],
            page_number,
        )
        if glyph_warnings:
            pages_checked.add(page_number)
            warnings.extend(glyph_warnings)


def _group_pdf_superscript_expectations(
    superscript_expectations: list[dict],
) -> dict[tuple[int, str], list[dict]]:
    """Group semantic superscript expectations by page and marker."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for expectation in superscript_expectations:
        page_number = int(expectation.get("page", 0))
        for marker in expectation.get("markers") or []:
            marker_text = str(marker.get("text", "")).strip()
            if _pdf_is_semantic_superscript_marker(marker_text):
                grouped.setdefault(
                    (page_number, marker_text),
                    [],
                ).append(marker)
    return grouped


def _pdf_rendered_superscript_scales(
    page_data: dict,
    pattern,
    superscript_flag: int,
) -> tuple[int, list[float]]:
    """Count rendered markers and measure their scale against nearby prose."""
    reference_spans = [
        span
        for block in page_data["dict"].get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if not (int(span.get("flags", 0)) & superscript_flag)
        and _pdf_span_has_lexical_text(span)
    ]
    rendered_scales: list[float] = []
    rendered_count = 0
    for block in page_data["dict"].get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                occurrences = len(
                    list(pattern.finditer(span.get("text", "")))
                )
                if not occurrences:
                    continue
                native_rendered = bool(
                    int(span.get("flags", 0)) & superscript_flag
                )
                reference_span = _pdf_nearby_superscript_reference(
                    span,
                    reference_spans,
                )
                reference_size = (
                    float(reference_span.get("size", 0.0))
                    if reference_span
                    else 0.0
                )
                rendered_scale = (
                    float(span.get("size", 0.0)) / reference_size
                    if reference_size > 0
                    else 0.0
                )
                inferred_rendered = (
                    not native_rendered
                    and reference_span is not None
                    and 0.45 <= rendered_scale <= 0.80
                )
                if not native_rendered and not inferred_rendered:
                    continue
                rendered_count += occurrences
                if reference_size > 0:
                    rendered_scales.extend(
                        [rendered_scale] * occurrences
                    )
    return rendered_count, rendered_scales


def _audit_pdf_superscripts(
    doc,
    source_doc,
    superscript_expectations: list[dict],
    page_cache: dict[int, dict],
    source_page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> tuple[int, int, list[dict]]:
    """Verify semantic superscript presence and relative scale."""
    grouped = _group_pdf_superscript_expectations(
        superscript_expectations
    )
    expected_total = sum(
        len(marker_specs) for marker_specs in grouped.values()
    )
    rendered_total = 0
    exact_source_matches: list[dict] = []
    superscript_flag = int(
        getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1)
    )

    for (page_number, marker_text), marker_specs in sorted(
        grouped.items()
    ):
        expected_count = len(marker_specs)
        if page_number < 1 or page_number > len(doc):
            warnings.append({
                "type": "missing-page",
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
            })
            continue
        pages_checked.add(page_number)
        page_data = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=True,
        )
        pattern = _pdf_marker_pattern(marker_text)
        plain_count = len(list(pattern.finditer(page_data["text"])))
        rendered_count, rendered_scales = (
            _pdf_rendered_superscript_scales(
                page_data,
                pattern,
                superscript_flag,
            )
        )
        rendered_total += min(rendered_count, expected_count)
        if rendered_count < expected_count:
            warnings.append({
                "type": (
                    "flattened-superscript"
                    if plain_count >= expected_count
                    else "missing-superscript"
                ),
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
                "rendered_superscript": rendered_count,
                "plain_occurrences": plain_count,
            })
            continue

        expected_scales = [
            float(marker.get("scale", 0.60))
            for marker in marker_specs
        ]
        if len(rendered_scales) < expected_count:
            source_match_count = 0
            if source_doc is not None and page_number <= len(source_doc):
                source_data = _pdf_audit_page_data(
                    source_doc,
                    page_number,
                    source_page_cache,
                    include_dict=True,
                )
                source_match_count = (
                    _pdf_exact_source_native_superscript_matches(
                        source_data["dict"],
                        page_data["dict"],
                        pattern,
                    )
                )
            if source_match_count >= expected_count:
                exact_source_matches.append({
                    "page": page_number,
                    "marker": marker_text,
                    "expected": expected_count,
                    "matched": source_match_count,
                })
                continue
            warnings.append({
                "type": "superscript-scale-unverifiable",
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
                "measured": len(rendered_scales),
            })
            continue

        for rendered_scale, expected_scale in zip(
            sorted(rendered_scales),
            sorted(expected_scales),
        ):
            if abs(rendered_scale - expected_scale) > 0.14:
                warnings.append({
                    "type": "superscript-size-mismatch",
                    "page": page_number,
                    "marker": marker_text,
                    "expected_scale": round(expected_scale, 3),
                    "rendered_scale": round(rendered_scale, 3),
                })
                break
    return expected_total, rendered_total, exact_source_matches


def _audit_pdf_formula_regions(
    doc,
    source_doc,
    formula_expectations: list[dict],
    page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> tuple[int, list[dict]]:
    """Verify immutable formula signatures, tolerating tiny raster drift."""
    verified_total = 0
    tolerated_raster_differences: list[dict] = []
    compared_fields = (
        "text_sha256",
        "font_sha256",
        "geometry_sha256",
        "raster_sha256",
        "span_count",
        "raster_width",
        "raster_height",
    )
    for expectation in formula_expectations:
        page_number = int(expectation.get("page", 0))
        element_number = int(expectation.get("element", -1))
        expected_signature = expectation.get("signature") or {}
        if page_number < 1 or page_number > len(doc):
            warnings.append({
                "type": "missing-formula-page",
                "page": page_number,
                "element": element_number,
            })
            continue
        if not expected_signature.get("signature_sha256"):
            warnings.append({
                "type": "source-formula-signature-missing",
                "page": page_number,
                "element": element_number,
            })
            continue

        pages_checked.add(page_number)
        page = doc[page_number - 1]
        page_data = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=True,
        )
        try:
            observed_signature = _pdf_formula_region_signature(
                page,
                expectation.get("bbox", (0, 0, 0, 0)),
                page_dict=page_data["dict"],
            )
        except Exception as formula_exc:
            warnings.append({
                "type": "formula-signature-error",
                "page": page_number,
                "element": element_number,
                "error_type": type(formula_exc).__name__,
            })
            continue

        mismatches = [
            field
            for field in compared_fields
            if expected_signature.get(field)
            != observed_signature.get(field)
        ]
        if mismatches == ["raster_sha256"] and source_doc is not None:
            rect = fitz.Rect(
                expectation.get("bbox", (0, 0, 0, 0))
            )
            source_page = source_doc[page_number - 1]
            # ``bbox`` is extraction geometry.  Clipping a rotated page with it
            # unmapped yields two degenerate pixmaps that compare equal, so a
            # genuinely overpainted formula would be tolerated as drift.
            source_clip = _pdf_extraction_rect_to_display(source_page, rect)
            observed_clip = _pdf_extraction_rect_to_display(page, rect)
            if source_clip.is_empty or observed_clip.is_empty:
                warnings.append({
                    "type": "formula-raster-clip-degenerate",
                    "page": page_number,
                    "element": element_number,
                })
                continue
            source_pix = source_page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                colorspace=fitz.csGRAY,
                alpha=False,
                clip=source_clip,
            )
            observed_pix = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                colorspace=fitz.csGRAY,
                alpha=False,
                clip=observed_clip,
            )
            raster_summary = _pdf_formula_raster_difference_summary(
                source_pix,
                observed_pix,
            )
            if raster_summary.get("tolerable"):
                tolerated_raster_differences.append({
                    "page": page_number,
                    "element": element_number,
                    **raster_summary,
                })
                verified_total += 1
                continue
        if mismatches:
            warnings.append({
                "type": "formula-integrity-mismatch",
                "page": page_number,
                "element": element_number,
                "mismatches": mismatches,
            })
            continue
        verified_total += 1
    return verified_total, tolerated_raster_differences


def _audit_pdf_vector_ocr_expectations(
    doc,
    vector_ocr_expectations: list[dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> int:
    """Verify that vector-OCR source phrases no longer remain."""
    expectations_by_page: dict[int, list[dict]] = {}
    for expectation in vector_ocr_expectations:
        expectations_by_page.setdefault(
            int(expectation.get("page", 0)),
            [],
        ).append(expectation)

    verified_elements = 0
    for page_number, page_expectations in sorted(
        expectations_by_page.items()
    ):
        if page_number < 1 or page_number > len(doc):
            for expectation in page_expectations:
                warnings.append({
                    "type": "missing-vector-ocr-page",
                    "page": page_number,
                    "element": int(expectation.get("element", -1)),
                })
            continue
        pages_checked.add(page_number)
        try:
            observed = _pdf_vector_ocr_normalized_match_text(
                _pdf_vector_ocr_page_text(doc[page_number - 1])
            )
        except Exception as vector_audit_exc:
            warnings.append({
                "type": "vector-ocr-audit-error",
                "page": page_number,
                "error_type": type(vector_audit_exc).__name__,
            })
            continue
        for expectation in page_expectations:
            remaining = [
                phrase
                for phrase in expectation.get("phrases") or []
                if _pdf_vector_ocr_normalized_match_text(phrase)
                in observed
            ]
            if remaining:
                warnings.append({
                    "type": "vector-ocr-source-text-remains",
                    "page": page_number,
                    "element": int(expectation.get("element", -1)),
                    "phrases": remaining,
                })
            else:
                verified_elements += 1
    return verified_elements


def _check_pdf_output_structure(
    out_path: str,
    superscript_expectations: list[dict],
    merge_decisions: list[dict] | None = None,
    formula_expectations: list[dict] | None = None,
    vector_ocr_expectations: list[dict] | None = None,
    source_path: str | None = None,
    source_password: str = "",
    page_extractions: dict | None = None,
) -> dict:
    """Check superscripts, accepted merges, and immutable formula regions."""
    warnings = _validate_pdf_merge_audit(
        merge_decisions or [],
        page_extractions,
    )
    pages_checked: set[int] = set()
    page_cache: dict[int, dict] = {}
    source_page_cache: dict[int, dict] = {}
    vector_expectations = vector_ocr_expectations or []
    doc = fitz.open(out_path)
    source_doc = None
    try:
        source_doc = _open_pdf_audit_source(
            source_path,
            source_password,
        )
        _audit_pdf_source_structure(
            doc,
            source_doc,
            _pdf_vector_source_glyph_allowance(vector_expectations),
            page_cache,
            source_page_cache,
            warnings,
            pages_checked,
        )
        (
            expected_superscripts,
            rendered_superscripts,
            exact_source_matches,
        ) = _audit_pdf_superscripts(
            doc,
            source_doc,
            superscript_expectations or [],
            page_cache,
            source_page_cache,
            warnings,
            pages_checked,
        )
        verified_formula_regions, tolerated_raster_differences = (
            _audit_pdf_formula_regions(
                doc,
                source_doc,
                formula_expectations or [],
                page_cache,
                warnings,
                pages_checked,
            )
        )
        verified_vector_elements = (
            _audit_pdf_vector_ocr_expectations(
                doc,
                vector_expectations,
                warnings,
                pages_checked,
            )
        )
    finally:
        doc.close()
        if source_doc is not None:
            source_doc.close()

    return {
        "status": "ok" if not warnings else "warning",
        "warning_count": len(warnings),
        "warnings": warnings,
        "pages_checked": sorted(pages_checked),
        "expected_superscripts": expected_superscripts,
        "rendered_superscripts": rendered_superscripts,
        "expected_formula_regions": len(formula_expectations or []),
        "verified_formula_regions": verified_formula_regions,
        "expected_vector_ocr_elements": len(vector_expectations),
        "verified_vector_ocr_elements": verified_vector_elements,
        "tolerated_formula_raster_differences": tolerated_raster_differences,
        "exact_source_superscript_matches": exact_source_matches,
        "accepted_merges": sum(
            1
            for decision in (merge_decisions or [])
            if decision.get("decision") == "accepted"
        ),
    }


def _check_pdf_output_structure_serialized(
    out_path: str,
    superscript_expectations: list[dict],
    merge_decisions: list[dict] | None = None,
    formula_expectations: list[dict] | None = None,
    vector_ocr_expectations: list[dict] | None = None,
    source_path: str | None = None,
    source_password: str = "",
    page_extractions: dict | None = None,
    *,
    dependencies: PDFAuditDependencies,
) -> dict:
    """Run the final fitz audit outside every concurrent extraction pass.

    PyMuPDF table/text extraction has process-global native state.  A final
    audit racing another task's extraction can therefore report hundreds of
    formula mismatches that disappear on an immediate isolated rerun.  Share
    the extraction slot and confirm any warning once before invalidating page
    caches or spending API budget on a structural rebuild.
    """
    PDF_EXTRACTION_MAX_CONCURRENCY = dependencies.extraction_max_concurrency
    _pdf_extraction_semaphore = dependencies.extraction_semaphore
    _check_pdf_output_structure = dependencies.check_output_structure
    _trim_process_memory = dependencies.trim_process_memory
    log.info(
        "Waiting for serialized PDF structure audit "
        f"(extraction_concurrency={PDF_EXTRACTION_MAX_CONCURRENCY})"
    )
    _pdf_extraction_semaphore.acquire()
    try:
        first = _check_pdf_output_structure(
            out_path,
            superscript_expectations,
            merge_decisions,
            formula_expectations,
            vector_ocr_expectations,
            source_path=source_path,
            source_password=source_password,
            page_extractions=page_extractions,
        )
        if not first.get("warning_count"):
            return first

        log.warning(
            "PDF structure audit returned %s warning(s); confirming once "
            "while the fitz slot remains isolated",
            first.get("warning_count", 0),
        )
        _trim_process_memory()
        confirmed = _check_pdf_output_structure(
            out_path,
            superscript_expectations,
            merge_decisions,
            formula_expectations,
            vector_ocr_expectations,
            source_path=source_path,
            source_password=source_password,
            page_extractions=page_extractions,
        )
        confirmed["initial_warning_count"] = int(first.get("warning_count", 0))
        confirmed["initial_warning_types"] = sorted({
            str(warning.get("type", "unknown"))
            for warning in first.get("warnings", [])
        })
        confirmed["initial_warning_pages"] = sorted({
            int(warning.get("page"))
            for warning in first.get("warnings", [])
            if warning.get("page") is not None
        })
        if not confirmed.get("warning_count"):
            confirmed["transient_initial_warnings_discarded"] = True
            log.warning(
                "Discarded %s transient PDF structure warning(s) after an "
                "isolated confirmation pass returned clean",
                first.get("warning_count", 0),
            )
        return confirmed
    finally:
        _pdf_extraction_semaphore.release()


def _require_clean_pdf_structure_check(check: dict) -> None:
    status = str((check or {}).get("status", "error"))
    warnings = (check or {}).get("warnings") or []
    try:
        warning_count = int((check or {}).get("warning_count", len(warnings)))
    except (TypeError, ValueError):
        warning_count = max(1, len(warnings))
    if status == "ok" and warning_count == 0 and not warnings:
        return

    raise PDFStructureValidationError(check)


def _save_and_validate_pdf_audit(task_id: str, audit: dict) -> str:
    path = _save_pdf_audit(task_id, audit)
    check = audit.get("structure_check") or {}
    log_method = log.info if check.get("warning_count", 0) == 0 else log.warning
    log_method(
        f"[{task_id}] PDF structure audit saved to {path} "
        f"({check.get('warning_count', 0)} warning(s))"
    )
    _require_clean_pdf_structure_check(check)
    return path


def _save_pdf_translation_progress(
    task_id: str,
    completed_indices: set[int],
    total_pages: int,
    current_file: str,
    source_page_fallbacks: list[dict] | None = None,
    *,
    dependencies: PDFAuditDependencies,
):
    tasks = dependencies.tasks
    save_progress = dependencies.save_progress
    payload = {
        "status": "translating",
        "progress": len(completed_indices),
        "total": total_pages,
        "current_file": current_file,
        "completed_files": [str(i) for i in sorted(completed_indices)],
        "filename": tasks[task_id].get("filename", ""),
        "out_filename": tasks[task_id].get("out_filename", ""),
        "type": "pdf",
    }
    if source_page_fallbacks:
        payload["source_page_fallbacks"] = list(source_page_fallbacks)
        payload["completion_status"] = "in_progress_with_source_page_fallbacks"
    save_progress(task_id, payload)
