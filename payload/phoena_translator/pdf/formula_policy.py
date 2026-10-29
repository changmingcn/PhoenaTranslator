"""Source-side formula protection policy: which text keeps its source
because translating it could corrupt nearby mathematics."""

from __future__ import annotations

import re
from phoena_translator.pdf.model import (
    ElementType,
    LayoutClass,
    PageExtraction,
    SkipReason,
)
from phoena_translator.pdf.targets import _pdf_element_requires_translation
from phoena_translator.pdf.semantic_text import (
    _ends_with_sentence_boundary,
    _is_heading_like_elem,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.semantic_cross_page import _cross_page_dominant_fontsize
from phoena_translator.math_text import (
    unicode_math_symbol_count as _unicode_math_symbol_count,
)
from phoena_translator.pdf.math_detection import _looks_like_pdf_identifier_line
from phoena_translator.pdf.geometry import _get_pdf_elem_rect


# A stub is the leftover head of a sentence: one or two lines.  The two
# fixtures this rule exists for are 106 and 44 characters.
PDF_FORMULA_MID_SENTENCE_STUB_MAX_CHARS = 150


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
    plain = _collapsed_plain_text(elem.get("content") or "")
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


def _preserve_pdf_formula_risk_text_elements(page_extractions: dict[int, PageExtraction]) -> list[dict]:
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
            if elem.get("type") != ElementType.TEXT:
                continue
            if not _pdf_element_requires_translation(elem):
                continue
            unprotected = _pdf_text_element_unprotected_math_symbols(elem)
            if not unprotected:
                continue
            if unprotected <= _pdf_prose_inline_math_symbol_tolerance(elem):
                continue
            elem["skip_translate_reason"] = SkipReason.FORMULA_RISK_PRESERVED
            preserved.append({
                "page": int(page_num) + 1,
                "element": elem_index,
                "unprotected_math_symbols": int(unprotected),
                "text": _collapsed_plain_text(elem.get("content") or "")[:120],
            })
    return preserved


def _preserve_pdf_formula_mid_sentence_neighbors(
    page_extractions: dict[int, PageExtraction],
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
            if elem.get("type") == ElementType.FORMULA_IMAGE
            and not elem.get("math_mixed")
        ]
        if not formula_rects:
            continue
        for elem_index, elem in enumerate(elements):
            if elem.get("type") != ElementType.TEXT:
                continue
            if not _pdf_element_requires_translation(elem):
                continue
            # Headings, table cells and rotated labels legitimately end
            # without punctuation; only running prose can continue into a
            # protected formula line.
            if (
                elem.get("table_hint")
                or elem.get("non_horizontal")
                or elem.get("layout_class") == LayoutClass.TABLE
                or _is_heading_like_elem(elem)
                or float(elem.get("fontsize", dominant_fontsize))
                > dominant_fontsize * 1.15
            ):
                continue
            plain = _collapsed_plain_text(elem.get("content") or "")
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
                    and other.get("type") == ElementType.TEXT
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
            elem["skip_translate_reason"] = SkipReason.FORMULA_RISK_PRESERVED
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
    page_extractions: dict[int, PageExtraction] | None = None,
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
                if elem.get("type") == ElementType.FORMULA_IMAGE:
                    queued = int(_pdf_element_requires_translation(elem))
                    formula_queue_entries += queued
                    if page_key in per_page:
                        per_page[page_key]["formula_queue_entries"] += queued
                elif elem.get("type") == ElementType.TEXT:
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
    page_extractions: dict[int, PageExtraction],
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
