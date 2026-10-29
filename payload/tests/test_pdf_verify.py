"""PDF verification: merge-decision audit and output-structure checks.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

from review_support import (
    _two_page_fixture,
)

# ---------------------------------------------------------------------------
# 2026-07-27 — three production defects: merge invariants judged a layout-time
# decision with post-merge state, rotated pages lost every formula signature,
# and doubled source ink reverted whole merge components to untranslated source
# ---------------------------------------------------------------------------


def _accepted_orphan_tail_fixture():
    """Post-merge state for one accepted page1 -> page2 orphan-tail absorption."""
    source_text = "Demand remained stable."
    tail = "the next section discusses a separate result."
    remainder = "Its ownership must stay on page two."
    pages, _page_rects, source, destination = _two_page_fixture(
        source_text,
        f"{tail} {remainder}",
    )
    source["paragraphs"] = [{"plain": source_text, "text_align": "left"}]
    destination["paragraphs"] = [
        {"plain": f"{tail} {remainder}", "text_align": "left"}
    ]
    source["content"] = f"{source_text} {tail}"
    destination["content"] = remainder
    decision = {
        "kind": "cross-page-orphan-tail",
        "decision": "accepted",
        "reason": "accepted",
        "source_continuation_reason": "complete",
        "carried_tail": tail,
        "source_page": 1,
        "destination_page": 2,
        "source_element": 0,
        "destination_element": 0,
    }
    return pages, source, destination, decision, remainder


class TestCrossPageRichDestination:
    SOURCE = (
        "Categorizing the firms requires some judgment, particularly given "
        "that they sometimes share certain characteristics or may act in "
        "multiple capacities. However, the presence of legal"
    )
    DESTINATION = (
        "name identifiers allows for the classification of participants "
        "ex-ante by legal status in combination with existing information "
        "about trading objectives, investment horizon and balance sheet "
        "capacity.11 Appendix A provides more detail on the classification "
        "framework. PTF activity differs across firms.12 Nevertheless, the "
        "sample remains representative."
    )
    DESTINATION_RICH = (
        "name identifiers allows for the classification of participants "
        "ex-ante by legal status in combination with existing information "
        "about trading objectives, investment horizon and balance sheet "
        "capacity.<sup>11</sup> Appendix A provides more detail on the "
        "classification framework. PTF activity differs across firms."
        "<sup>12</sup> Nevertheless, the sample remains representative."
    )

    def test_leading_sentence_with_superscript_splits_losslessly(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )
        from phoena_translator.pdf.translate.inline_markup import (
            _pdf_superscript_signature,
        )
        from phoena_translator.pdf.verify.expectations import (
            _collect_pdf_superscript_expectations,
        )
        from phoena_translator.pdf.verify.merges import (
            _validate_pdf_orphan_tail_decision,
        )

        pages, rects, source, destination = _two_page_fixture(
            self.SOURCE,
            self.DESTINATION,
            dest_rich=self.DESTINATION_RICH,
        )
        source["paragraphs"] = [{"plain": self.SOURCE, "rich": self.SOURCE}]
        destination["paragraphs"] = [{
            "plain": self.DESTINATION,
            "rich": self.DESTINATION_RICH,
        }]
        destination["superscript_runs"] = [
            {"text": "11", "scale": 0.66, "source": "native"},
            {"text": "12", "scale": 0.66, "source": "native"},
        ]
        audit_log = []

        assert _merge_cross_page_sentences(pages, 2, rects, audit_log) == 1
        assert source["content"].endswith("balance sheet capacity.11")
        assert source["rich_content"].endswith(
            "balance sheet capacity.<sup>11</sup>"
        )
        assert destination["content"].startswith("Appendix A provides")
        assert "<sup>11</sup>" not in (destination["rich_content"] or "")
        assert "<sup>12</sup>" in destination["rich_content"]
        assert _pdf_superscript_signature(source["rich_content"]) == ("11",)
        assert _pdf_superscript_signature(destination["rich_content"]) == ("12",)
        assert _validate_pdf_orphan_tail_decision(
            audit_log[0],
            pages,
        ) is None

        expectations = _collect_pdf_superscript_expectations(pages)
        assert [
            (item["page"], [marker["text"] for marker in item["markers"]])
            for item in expectations
        ] == [(1, ["11"]), (2, ["12"])]

    def test_markup_spanning_sentence_boundary_remains_rejected(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        pages, rects, source, destination = _two_page_fixture(
            self.SOURCE,
            "name identifiers complete the sentence. Appendix A follows.",
            dest_rich=(
                "<b>name identifiers complete the sentence. "
                "Appendix A follows.</b>"
            ),
        )

        assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
        assert source["content"] == self.SOURCE
        assert destination["content"].startswith("name identifiers")


class TestOrphanTailAuditPolicy:
    def test_shape_check_accepts_digit_tail_without_evidence(self):
        from phoena_translator.pdf.verify.merges import (
            _validate_pdf_orphan_tail_decision,
        )

        decision = {
            "kind": "cross-page-orphan-tail",
            "decision": "accepted",
            "reason": "accepted",
            "carried_tail": "1-3, respectively.10",
            "source_page": 1,
            "destination_page": 2,
            "source_element": 0,
            "destination_element": 0,
        }
        assert _validate_pdf_orphan_tail_decision(decision, None) is None

    def test_shape_check_without_layout_evidence_allows_signal_one(self):
        from phoena_translator.pdf.verify.merges import (
            _validate_pdf_orphan_tail_decision,
        )

        decision = {
            "kind": "cross-page-orphan-tail",
            "decision": "accepted",
            "reason": "accepted",
            "carried_tail": "4.2 Results.",
            "source_page": 1,
            "destination_page": 2,
            "source_element": 0,
            "destination_element": 0,
        }
        assert _validate_pdf_orphan_tail_decision(decision, None) is None

    def test_audit_rejects_non_body_destination_with_layout_evidence(self):
        from phoena_translator.pdf.verify.merges import (
            _validate_pdf_orphan_tail_decision,
        )

        source_text = "The explanation continues across"
        tail = "Results."
        pages, _rects, source, destination = _two_page_fixture(
            source_text,
            tail,
        )
        source["paragraphs"] = [{"plain": source_text}]
        destination["paragraphs"] = [{"plain": tail}]
        destination["layout_class"] = "heading"
        source["content"] = f"{source_text} {tail}"
        destination["type"] = "text_merged_away"
        decision = {
            "kind": "cross-page-orphan-tail",
            "decision": "accepted",
            "reason": "accepted",
            "source_continuation_reason": "dangling",
            "carried_tail": tail,
            "source_page": 1,
            "destination_page": 2,
            "source_element": 0,
            "destination_element": 0,
        }
        assert (
            _validate_pdf_orphan_tail_decision(decision, pages)
            == "orphan-tail-non-body"
        )


def test_orphan_audit_accepts_complete_source_lowercase_decision():
    from phoena_translator.pdf.verify.merges import _validate_pdf_orphan_tail_decision

    source_text = "Demand remained stable."
    tail = "the next section discusses a separate result."
    remainder = "Its ownership must stay on page two."
    destination_text = f"{tail} {remainder}"
    pages, _page_rects, source, destination = _two_page_fixture(
        source_text,
        destination_text,
    )
    # The live dictionaries model the post-merge state, while paragraph text
    # retains immutable extraction evidence for the audit.
    source["paragraphs"] = [{"plain": source_text, "text_align": "left"}]
    destination["paragraphs"] = [
        {"plain": destination_text, "text_align": "left"}
    ]
    source["content"] = f"{source_text} {tail}"
    destination["content"] = remainder
    decision = {
        "kind": "cross-page-orphan-tail",
        "decision": "accepted",
        "reason": "accepted",
        "source_continuation_reason": "complete",
        "carried_tail": tail,
        "source_page": 1,
        "destination_page": 2,
        "source_element": 0,
        "destination_element": 0,
    }

    assert _validate_pdf_orphan_tail_decision(decision, pages) is None


def test_orphan_audit_rejects_destination_owned_by_local_predecessor():
    from phoena_translator.pdf.verify.merges import _validate_pdf_orphan_tail_decision

    tail = "continue to hold and the models have become extremely capable."
    source = {
        "type": "text",
        "layout_class": "body",
        "content": f"The prior page closes completely. {tail}",
        "paragraphs": [
            {"plain": "The prior page closes completely.", "text_align": "left"}
        ],
        "bbox": [72.0, 645.0, 542.0, 670.0],
        "fontsize": 10.02,
    }
    local_lead = {
        "type": "text",
        "layout_class": "table",
        "table_hint": True,
        "content": "The page-local paragraph starts because scaling laws",
        "paragraphs": [
            {
                "plain": "The page-local paragraph starts because scaling laws",
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 379.92, 542.25, 392.75],
        "fontsize": 10.02,
    }
    destination_original = f"{tail} Regulation is needed and will come."
    destination = {
        "type": "text",
        "layout_class": "body",
        "content": "Regulation is needed and will come.",
        "paragraphs": [
            {"plain": destination_original, "text_align": "left"}
        ],
        "bbox": [72.0, 391.93, 542.27, 428.76],
        "fontsize": 10.02,
    }
    pages = {
        0: {"elements": [source]},
        1: {"elements": [local_lead, destination]},
    }
    decision = {
        "kind": "cross-page-orphan-tail",
        "decision": "accepted",
        "reason": "accepted",
        "source_continuation_reason": "complete",
        "carried_tail": tail,
        "source_page": 1,
        "destination_page": 2,
        "source_element": 0,
        "destination_element": 1,
    }

    assert _validate_pdf_orphan_tail_decision(decision, pages) == (
        "orphan-tail-local-predecessor"
    )


def test_producer_records_pass_evidence_backed_audit_validation():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )
    from phoena_translator.pdf.verify.merges import _validate_pdf_orphan_tail_decision

    cases = [
        (
            "The explanation continues across",
            (
                "pages without changing ownership. A separate sentence "
                "remains on page two."
            ),
            "dangling",
        ),
        (
            "These findings agree with Smith et al.",
            (
                "who document the same effect. A separate sentence remains "
                "on page two."
            ),
            "ambiguous-terminal-abbreviation",
        ),
    ]
    for source_text, destination_text, expected_reason in cases:
        pages, rects, source, destination = _two_page_fixture(
            source_text,
            destination_text,
        )
        source["paragraphs"] = [{"plain": source_text}]
        destination["paragraphs"] = [{"plain": destination_text}]
        audit_log = []

        assert _merge_cross_page_sentences(pages, 2, rects, audit_log) == 1
        assert len(audit_log) == 1
        assert audit_log[0]["source_continuation_reason"] == expected_reason
        assert _validate_pdf_orphan_tail_decision(
            audit_log[0],
            pages,
        ) is None


def test_orphan_audit_still_rejects_a_preserved_endpoint():
    from phoena_translator.pdf.verify.merges import _validate_pdf_orphan_tail_decision

    pages, _source, destination, decision, _remainder = (
        _accepted_orphan_tail_fixture()
    )
    # A preserved endpoint re-renders its ORIGINAL ink, which still holds the
    # carried tail, so the sentence would ship twice.  The auditor must keep
    # treating this as fatal; the fix belongs at the preservation sites.
    destination["skip_translate_reason"] = "formula_risk_preserved"

    assert (
        _validate_pdf_orphan_tail_decision(decision, pages)
        == "orphan-tail-non-body"
    )


def test_orphan_audit_accepts_a_destination_that_later_absorbed_its_own_tail():
    from phoena_translator.pdf.verify.merges import _validate_pdf_orphan_tail_decision

    pages, _source, destination, decision, remainder = (
        _accepted_orphan_tail_fixture()
    )
    chained_tail = "and the discussion continues onto a third page."
    # The destination is also the last body block of its own page, so it
    # becomes the SOURCE of the next page's merge and grows by that tail.
    destination["content"] = f"{remainder} {chained_tail}"
    chained_decision = {
        "kind": "cross-page-orphan-tail",
        "decision": "accepted",
        "reason": "accepted",
        "carried_tail": chained_tail,
        "source_page": 2,
        "destination_page": 3,
        "source_element": 0,
        "destination_element": 0,
    }

    assert _validate_pdf_orphan_tail_decision(
        decision,
        pages,
        [decision, chained_decision],
    ) is None
    # Without the chain evidence the same live state is still unexplained.
    assert (
        _validate_pdf_orphan_tail_decision(decision, pages)
        == "orphan-tail-destination-mismatch"
    )
