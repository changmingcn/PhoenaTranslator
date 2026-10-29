"""The cross-page merge graph and the re-verification of every recorded
merge decision against the final page extractions."""

from __future__ import annotations

from phoena_translator.pdf.model import (
    CrossPageMergeDecision,
    ElementType,
    LayoutClass,
    PageExtraction,
)
from phoena_translator.pdf.semantics.cross_page import (
    CROSS_PAGE_ABSORBED_TAIL_MAX_CHARS,
    _cross_page_continuation_decision,
    _cross_page_destination_has_local_predecessor,
    _cross_page_destination_start_evidence,
    _cross_page_destination_start_rejection,
    _cross_page_dominant_fontsize,
    _cross_page_tail_start_acceptable,
    _is_cross_page_non_body_context,
    _strip_trailing_footnote_marker,
)
from phoena_translator.pdf.text_shape import (
    _collapsed_plain_text,
    _ends_with_sentence_boundary,
)


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
    page_extractions: dict[int, PageExtraction],
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
    page_extractions: dict[int, PageExtraction],
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
    page_extractions: dict[int, PageExtraction],
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
    return _collapsed_plain_text(text or "")


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
    page_extractions: dict[int, PageExtraction] | None,
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
        source.get("layout_class") != LayoutClass.BODY
        or destination.get("layout_class") not in {LayoutClass.BODY, LayoutClass.SCATTERED}
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
    destination_elements = (
        destination_info.get("elements", [])
        if isinstance(destination_info, dict)
        else []
    )
    original_destination_elements = [
        {
            **element,
            "content": _pdf_merge_original_element_text(element),
            "rich_content": _pdf_merge_original_element_text(element),
        }
        for element in destination_elements
        if isinstance(element, dict)
    ]
    try:
        destination_element_index = int(decision.get("destination_element"))
    except (TypeError, ValueError):
        return "merge-evidence-missing"
    if _cross_page_destination_has_local_predecessor(
        destination_element_index,
        original_destination_elements,
        _cross_page_dominant_fontsize(page_extractions),
    ):
        return "orphan-tail-local-predecessor"
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
    if not _cross_page_tail_start_acceptable(
        tail,
        source_dangling=source_dangling,
    ):
        return "orphan-tail-start-policy"
    destination_original = _normalized_audit_text(
        _pdf_merge_original_element_text(destination)
    )
    if not destination_original.startswith(tail):
        return "orphan-tail-destination-mismatch"
    remainder = destination_original[len(tail):].strip()
    if destination.get("type") == ElementType.MERGED_AWAY:
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
    page_extractions: dict[int, PageExtraction] | None = None,
) -> list[dict]:
    warnings = []
    for index, decision in enumerate(merge_decisions or []):
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        kind = str(decision.get("kind") or "cross-page-merge")
        if kind == CrossPageMergeDecision.KIND:
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
                        source_elem.get("layout_class") != LayoutClass.BODY
                        or destination_elem.get("layout_class") != LayoutClass.BODY
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
