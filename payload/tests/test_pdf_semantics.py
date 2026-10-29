"""PDF semantic passes: paragraph, cross-page, reference, formula, table and glossary merges.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import pytest
from review_support import (
    _table_test_elem,
    _two_page_fixture,
)

from phoena_translator.config import AppConfig

# ---------------------------------------------------------------------------
# Cross-page orphan sentence tails (the "交易者。" seam defect)
# ---------------------------------------------------------------------------


def _page_rects_letter():
    import fitz

    return {0: fitz.Rect(0, 0, 612, 792), 1: fitz.Rect(0, 0, 612, 792)}


def _orphan_tail_fixture():
    source_elem = {
        "type": "text",
        "layout_class": "body",
        "fontsize": 11.96,
        "y": 595.0,
        "x": 108.0,
        "bbox": [108.0, 595.0, 504.0, 700.0],
        "content": (
            "Unfortunately, we are unable to estimate the costs directly "
            "because access was cut off by the CFTC.5 Thus, we resort to "
            "documenting empirical regularities, which we believe stem from "
            "the immediacy absorption activity of high frequency"
        ),
        "rich_content": (
            "Unfortunately, we are unable to estimate the costs directly "
            "because access was cut off by the CFTC.<sup>5</sup> Thus, we "
            "resort to documenting empirical regularities, which we believe "
            "stem from the immediacy absorption activity of high frequency"
        ),
        "paragraphs": [
            {
                "plain": (
                    "Unfortunately, we are unable to estimate the costs "
                    "directly because access was cut off by the CFTC.5 Thus, "
                    "we resort to documenting empirical regularities, which "
                    "we believe stem from the immediacy absorption activity "
                    "of high frequency"
                ),
                "text_align": "left",
            }
        ],
        "superscript_runs": [{"text": "5", "scale": 0.6, "source": "font"}],
    }
    orphan_elem = {
        "type": "text",
        "layout_class": "scattered",
        "fontsize": 11.96,
        "y": 104.0,
        "x": 108.0,
        "bbox": [108.0, 104.0, 148.0, 118.0],
        "content": "traders.",
        "rich_content": None,
        "paragraphs": [{"plain": "traders.", "text_align": "left"}],
    }
    body_elem = {
        "type": "text",
        "layout_class": "body",
        "fontsize": 11.96,
        "y": 119.0,
        "x": 108.0,
        "bbox": [108.0, 119.0, 504.0, 230.0],
        "content": (
            "We show that HFTs are much more likely than market makers to "
            "aggressively execute the last contracts before a price move."
        ),
        "rich_content": None,
        "paragraphs": [
            {
                "plain": (
                    "We show that HFTs are much more likely than market "
                    "makers to aggressively execute the last contracts "
                    "before a price move."
                ),
                "text_align": "left",
            }
        ],
    }
    page_extractions = {
        0: {"elements": [source_elem]},
        1: {"elements": [orphan_elem, body_elem]},
    }
    return page_extractions, source_elem, orphan_elem, body_elem


# ---------------------------------------------------------------------------
# Mid-sentence prose coupled to a protected formula line (p15 truncation)
# ---------------------------------------------------------------------------


def _mid_sentence_fixture_text(content, y0, y1):
    return {
        "type": "text",
        "layout_class": "body",
        "fontsize": 11.96,
        "line_height": 14.0,
        "y": y0,
        "x": 108.0,
        "bbox": [108.0, y0, 504.0, y1],
        "content": content,
        "rich_content": None,
        "paragraphs": [{"plain": content, "text_align": "left"}],
    }


# ---------------------------------------------------------------------------
# 2026-07-24 review fixes
# ---------------------------------------------------------------------------


def _cross_page_fixture(first_page_content: str, first_elem_extra: dict | None = None):
    """Two pages: page 1 ends mid-sentence, page 2 opens with its tail."""
    import fitz

    last_elem = {
        "type": "text",
        "layout_class": "body",
        "content": (
            "Transaction prices first stabilized and then rebounded "
            "rapidly, as Fundamental Buyers and Opportunistic Traders"
        ),
        "bbox": [72.0, 659.0, 518.0, 715.0],
        "y": 659.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    first_elem = {
        "type": "text",
        "layout_class": "scattered",
        "content": first_page_content,
        "bbox": [72.0, 104.0, 518.0, 131.0],
        "y": 104.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    if first_elem_extra:
        first_elem.update(first_elem_extra)
    heading_elem = {
        "type": "text",
        "layout_class": "scattered",
        "content": "V. What Do High Frequency Traders Do?",
        "bbox": [72.0, 154.0, 432.0, 172.0],
        "y": 154.0,
        "x": 72.0,
        "fontsize": 17.0,
    }
    page_extractions = {
        0: {"elements": [last_elem]},
        1: {"elements": [first_elem, heading_elem]},
    }
    page_rects = {0: fitz.Rect(0, 0, 612, 792), 1: fitz.Rect(0, 0, 612, 792)}
    return page_extractions, page_rects, last_elem, first_elem


def _mid_sentence_page(with_intervening_prose: bool):
    """A stub ending mid-sentence, a formula below, optionally prose between."""
    def text_elem(content, y0, y1, x0=78.0, x1=440.0):
        return {
            "type": "text",
            "layout_class": "body",
            "content": content,
            "bbox": [x0, y0, x1, y1],
            "y": y0,
            "x": x0,
            "fontsize": 10.0,
        }

    elements = [
        text_elem(
            "to be linearly related to a smaller number K",
            138.8,
            174.8,
        ),
    ]
    if with_intervening_prose:
        elements.append(
            text_elem(
                "such a linear relation implies that the total number of "
                "unknown parameters is bounded.",
                177.8,
                187.8,
            )
        )
    elements.append({
        "type": "formula_image",
        "layout_class": "body",
        "content": "",
        "bbox": [78.0, 190.3, 386.9, 200.8],
        "y": 190.3,
        "x": 78.0,
        "fontsize": 10.0,
    })
    return {0: {"elements": elements}}


class TestCrossPageOrphanTail:
    def test_fragment_detection(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _cross_page_whole_fragment_tail,
        )

        def _cross_page_orphan_tail_fragment(elem):
            fragment = _cross_page_whole_fragment_tail(elem)
            return fragment[0] if fragment else None

        _pages, _source, orphan, body = _orphan_tail_fixture()
        assert _cross_page_orphan_tail_fragment(orphan) == "traders."
        assert _cross_page_orphan_tail_fragment(body) is None  # uppercase start
        unterminated = dict(orphan, content="high frequency")
        assert _cross_page_orphan_tail_fragment(unterminated) is None
        table_cell = dict(orphan, table_hint=True)
        assert _cross_page_orphan_tail_fragment(table_cell) is None

    def test_absorption_moves_tail_and_merges_orphan_away(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        pages, source, orphan, _body = _orphan_tail_fixture()
        audit_log: list = []
        absorbed = _merge_cross_page_sentences(
            pages, 2, _page_rects_letter(), audit_log=audit_log
        )
        assert absorbed == 1
        assert source["content"].endswith("of high frequency traders.")
        # Appending must not disturb existing superscript markup.
        assert "<sup>5</sup>" in source["rich_content"]
        assert source["rich_content"].endswith("of high frequency traders.")
        assert orphan["type"] == "text_merged_away"
        assert audit_log and audit_log[0]["kind"] == "cross-page-orphan-tail"
        assert audit_log[0]["decision"] == "accepted"
        assert audit_log[0]["carried_tail"] == "traders."

    def test_absorption_skips_when_source_sentence_complete(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        pages, source, orphan, _body = _orphan_tail_fixture()
        source["content"] = source["content"] + " traders."
        source["paragraphs"][0]["plain"] += " traders."
        absorbed = _merge_cross_page_sentences(pages, 2, _page_rects_letter())
        assert absorbed == 0
        assert orphan["type"] == "text"

    def test_audit_validator_accepts_and_rejects(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )
        from phoena_translator.pdf.verify.merges import _validate_pdf_merge_audit

        pages, source, _orphan, _body = _orphan_tail_fixture()
        audit_log: list = []
        _merge_cross_page_sentences(
            pages, 2, _page_rects_letter(), audit_log=audit_log
        )
        assert _validate_pdf_merge_audit(audit_log, pages) == []

        # Tamper: strip the tail from the source again -> single-ownership
        # invariant must fire.
        source["content"] = source["content"][: -len(" traders.")]
        violations = _validate_pdf_merge_audit(audit_log, pages)
        assert violations and violations[0]["invariant"] == "orphan-tail-not-absorbed"


class TestFormulaMidSentencePreservation:
    def test_mid_sentence_stub_above_formula_is_preserved(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        stub = _mid_sentence_fixture_text(
            "To test whether HFTs changed their behavior, we interact dummy "
            "variables for the Up phase and the Down phase",
            610.0,
            636.0,
        )
        formula = {
            "type": "formula_image",
            "bbox": [108.0, 638.0, 504.0, 652.0],
        }
        pages = {0: {"elements": [stub, formula]}}
        preserved = _preserve_pdf_formula_mid_sentence_neighbors(pages)
        assert len(preserved) == 1
        assert preserved[0]["reason"] == "mid-sentence-into-formula"
        assert stub["skip_translate_reason"] == "formula_risk_preserved"

    def test_terminated_intro_above_formula_still_translates(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        intro = _mid_sentence_fixture_text(
            "Letting these definitions hold, the regression specification "
            "becomes:",
            610.0,
            636.0,
        )
        formula = {
            "type": "formula_image",
            "bbox": [108.0, 638.0, 504.0, 652.0],
        }
        pages = {0: {"elements": [intro, formula]}}
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        assert "skip_translate_reason" not in intro

    def test_distant_formula_does_not_trigger(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        stub = _mid_sentence_fixture_text(
            "we interact dummy variables for the Up phase and the Down phase",
            300.0,
            326.0,
        )
        formula = {
            "type": "formula_image",
            "bbox": [108.0, 500.0, 504.0, 514.0],
        }
        pages = {0: {"elements": [stub, formula]}}
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        assert "skip_translate_reason" not in stub

    def test_headers_tables_and_rotated_labels_are_not_preserved(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        def formula_at(y0):
            return {"type": "formula_image", "bbox": [108.0, y0, 504.0, y0 + 14.0]}

        anchor = _mid_sentence_fixture_text(
            "Ordinary running body prose that dominates the document size "
            "statistics for this fixture and ends with a full stop.",
            100.0,
            180.0,
        )
        table_header = _mid_sentence_fixture_text(
            "% Volume % Trades Trade-Weighted Vol-Weighted",
            610.0,
            636.0,
        )
        table_header["table_hint"] = True
        rotated = _mid_sentence_fixture_text(
            "Net Position Scaled by Market Trading Volume",
            610.0,
            636.0,
        )
        rotated["non_horizontal"] = True
        display_title = _mid_sentence_fixture_text(
            "The Flash Crash The Impact of High Frequency",
            610.0,
            636.0,
        )
        display_title["fontsize"] = 17.0

        pages = {
            0: {"elements": [anchor, table_header, formula_at(638.0)]},
            1: {"elements": [dict(anchor), rotated, formula_at(638.0)]},
            2: {"elements": [dict(anchor), display_title, formula_at(638.0)]},
        }
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        for elem in (table_header, rotated, display_title):
            assert "skip_translate_reason" not in elem

    # 2026-07-27: three whole paragraphs of a Bank of Japan Review shipped as
    # exact English because they "ran into a formula".  Two of those formulas
    # were prose lines that math detection promoted over one inline variable,
    # and the third paragraph was a finished sentence whose footnote marker
    # hid the full stop.
    def test_mixed_math_line_does_not_freeze_the_paragraph_above(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        paragraph = _mid_sentence_fixture_text(
            "employment report from January 2014 to March 2020 are used to "
            "estimate the above regression analysis. We adopt the",
            610.0,
            636.0,
        )
        promoted_prose_line = {
            "type": "formula_image",
            "bbox": [108.0, 638.0, 504.0, 652.0],
            "math_mixed": True,
        }
        pages = {0: {"elements": [paragraph, promoted_prose_line]}}
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        assert "skip_translate_reason" not in paragraph

    def test_footnote_marker_does_not_disguise_a_finished_sentence(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        finished = _mid_sentence_fixture_text(
            "verify whether the above observations are consistent with the "
            "USD/JPY spot market.18",
            610.0,
            636.0,
        )
        formula = {
            "type": "formula_image",
            "bbox": [108.0, 638.0, 504.0, 652.0],
        }
        pages = {0: {"elements": [finished, formula]}}
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        assert "skip_translate_reason" not in finished

    def test_whole_paragraph_is_too_long_to_be_a_stub(self):
        from phoena_translator.pdf.semantics.formula_policy import (
            _preserve_pdf_formula_mid_sentence_neighbors,
        )

        paragraph = _mid_sentence_fixture_text(
            "that is, the spread between traded price and mid-quote price "
            "(best bid and best ask average price) at the same time. The "
            "following independent variables are used: logarithmic form of "
            "either of the two algorithmic trading proxy indicators and",
            610.0,
            690.0,
        )
        formula = {
            "type": "formula_image",
            "bbox": [108.0, 692.0, 504.0, 706.0],
        }
        pages = {0: {"elements": [paragraph, formula]}}
        assert _preserve_pdf_formula_mid_sentence_neighbors(pages) == []
        assert "skip_translate_reason" not in paragraph

    def test_leading_tail_split_from_first_paragraph(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )
        from phoena_translator.pdf.verify.merges import _validate_pdf_merge_audit

        pages, source, _orphan, body = _orphan_tail_fixture()
        body["content"] = (
            "high frequency traders. Because advanced trading technology can "
            "be deployed across many automated markets, the cost of "
            "intermediation per market has fallen dramatically."
        )
        body["paragraphs"] = [{"plain": body["content"], "text_align": "left"}]
        pages[1]["elements"] = [body]

        audit_log: list = []
        absorbed = _merge_cross_page_sentences(
            pages, 2, _page_rects_letter(), audit_log
        )
        assert absorbed == 1
        assert source["content"].endswith("high frequency traders.")
        assert "Because" not in source["content"]
        assert body["content"].startswith("Because advanced trading")
        assert body["type"] == "text"
        assert audit_log[0]["carried_tail"] == "high frequency traders."
        assert _validate_pdf_merge_audit(audit_log, pages) == []

    def test_footnote_marker_fragment_absorbed_with_markup(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        pages, source, orphan, _body = _orphan_tail_fixture()
        orphan["content"] = "exchange.4"
        orphan["rich_content"] = "exchange.<sup>4</sup>"
        orphan["paragraphs"] = [{"plain": "exchange.4", "text_align": "left"}]
        orphan["superscript_runs"] = [{"text": "4", "scale": 0.6}]

        absorbed = _merge_cross_page_sentences(pages, 2, _page_rects_letter())
        assert absorbed == 1
        assert source["content"].endswith("of high frequency exchange.4")
        assert source["rich_content"].endswith(
            "of high frequency exchange.<sup>4</sup>"
        )
        assert orphan["type"] == "text_merged_away"


class TestCrossPageScatteredDestination:
    """Production miss 2026-07-24: pages 18→19 of the Flash Crash paper.

    The page-top continuation paragraph is classified ``scattered`` (it sits
    alone above the page's first heading), and the split path used to demand
    strict ``body`` — the stranded tail "lifted offers." was then translated
    without context on the next page.
    """

    CONTENT = (
        "lifted offers. By 2:08 p.m. CT, 36 minutes after the Flash Crash "
        "began, prices of E-mini futures had recovered to their levels."
    )

    def test_split_merge_accepts_scattered_destination(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = (
            _cross_page_fixture(self.CONTENT)
        )
        audit_log = []
        absorbed = _merge_cross_page_sentences(
            page_extractions, 2, page_rects, audit_log
        )
        assert absorbed == 1
        assert last_elem["content"].endswith(" lifted offers.")
        assert first_elem["content"].startswith("By 2:08 p.m. CT")
        assert "lifted offers" not in first_elem["content"]
        assert first_elem["type"] == "text"
        assert audit_log and audit_log[0]["decision"] == "accepted"
        assert audit_log[0]["carried_tail"] == "lifted offers."
        assert audit_log[0]["destination_layout"] == "scattered"

    def test_inline_math_records_block_the_merge(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = (
            _cross_page_fixture(
                self.CONTENT,
                {"inline_math_fragments": [{"text": "μ"}]},
            )
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["content"] == self.CONTENT
        assert not last_elem["content"].endswith("lifted offers.")

    def test_non_prose_residual_blocks_the_merge(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        content = (
            "lifted offers. 2008 2009 2010 2011 2012 2013 2014 2015 2016 "
            "2017 2018 2019 2020 2021 2022 2023."
        )
        page_extractions, page_rects, _last_elem, first_elem = (
            _cross_page_fixture(content)
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["content"] == content


class TestWatermarkKeywordCorroboration:
    """A repeated compliance footer in plain black type is content."""

    FOOTER = "For internal use only. Do not distribute."

    @staticmethod
    def _pages(total_pages, **elem_overrides):
        import fitz

        page_extractions = {}
        page_rects = {}
        for page_num in range(total_pages):
            elem = {
                "type": "text",
                "content": TestWatermarkKeywordCorroboration.FOOTER,
                "bbox": [150.0, 770.0, 460.0, 780.0],
                "color": 0,
                "fontsize": 8.0,
            }
            elem.update(elem_overrides)
            page_extractions[page_num] = {"elements": [elem]}
            page_rects[page_num] = fitz.Rect(0, 0, 612, 792)
        return page_extractions, page_rects

    def test_black_footer_keyword_line_is_not_erased(self):
        from phoena_translator.pdf.semantics.watermarks import _mark_watermark_elements

        page_extractions, page_rects = self._pages(3)
        _mark_watermark_elements(page_extractions, page_rects, 3)
        for info in page_extractions.values():
            assert "skip_translate_reason" not in info["elements"][0]

    def test_rotated_stamp_is_still_a_watermark(self):
        from phoena_translator.pdf.semantics.watermarks import _mark_watermark_elements

        page_extractions, page_rects = self._pages(
            3,
            non_horizontal=True,
            bbox=[200.0, 350.0, 420.0, 450.0],
        )
        _mark_watermark_elements(page_extractions, page_rects, 3)
        for info in page_extractions.values():
            assert (
                info["elements"][0].get("skip_translate_reason") == "watermark"
            )

    def test_large_keyword_stamp_is_still_a_watermark(self):
        from phoena_translator.pdf.semantics.watermarks import _mark_watermark_elements

        page_extractions, page_rects = self._pages(
            3,
            fontsize=22.0,
            bbox=[100.0, 60.0, 500.0, 100.0],
        )
        _mark_watermark_elements(page_extractions, page_rects, 3)
        for info in page_extractions.values():
            assert (
                info["elements"][0].get("skip_translate_reason") == "watermark"
            )


class TestTailStartPolicy:
    def test_shapes(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _cross_page_tail_start_acceptable,
        )

        # Lowercase: acceptable regardless of the source signal.
        assert _cross_page_tail_start_acceptable(
            "as liquidity rushed in.", source_dangling=False
        )
        # The first alphabetic character after a digit run is lowercase, so
        # signal 2 applies even when the source itself ended with punctuation.
        assert _cross_page_tail_start_acceptable(
            "1-3, respectively.", source_dangling=True
        )
        assert _cross_page_tail_start_acceptable(
            "1-3, respectively.", source_dangling=False
        )
        # An uppercase opener alone does not trigger signal 2, but signal 1
        # remains authoritative. Non-body headings are filtered by the caller.
        assert _cross_page_tail_start_acceptable(
            "4.2 Results", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
            "1. Introduction", source_dangling=False
        )
        # Uppercase proper-noun continuation: dangling source only.
        assert _cross_page_tail_start_acceptable(
            "High Frequency Traders lifted offers.", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
            "High Frequency Traders lifted offers.", source_dangling=False
        )
        # The lexical helper assumes the caller has already removed captions
        # and headings from the BODY candidate set.
        assert _cross_page_tail_start_acceptable(
            "Table VII presents regression results.", source_dangling=True
        )
        assert _cross_page_tail_start_acceptable(
            "IV. Absorbing a Large Order Flow Imbalance", source_dangling=True
        )


class TestCrossPageDigitTail:
    """Production miss: 2014.pdf pages 24→25, tail '1-3, respectively.10'."""

    SOURCE = (
        "For lagged price changes, coefficient estimates for Passive volume "
        "by High Frequency Traders and Market Makers are negative and "
        "statistically significant at lags 1 and lags"
    )

    def test_digit_tail_with_dangling_source_merges(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = _two_page_fixture(
            self.SOURCE,
            "1-3, respectively.10",
            dest_rich="1-3, respectively.<sup>10</sup>",
            dest_bbox=[72.0, 104.0, 190.0, 117.0],
        )
        audit_log = []
        absorbed = _merge_cross_page_sentences(
            page_extractions, 2, page_rects, audit_log
        )
        assert absorbed == 1
        assert last_elem["content"].endswith("lags 1-3, respectively.10")
        assert last_elem["rich_content"].endswith(
            "1-3, respectively.<sup>10</sup>"
        )
        assert first_elem["type"] == "text_merged_away"
        assert audit_log[0]["carried_tail"] == "1-3, respectively.10"
        assert audit_log[0]["source_continuation_reason"] == "dangling"

    def test_duplicate_digit_tail_is_not_absorbed_twice(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, _last, first_elem = _two_page_fixture(
            self.SOURCE.rstrip() + " 1-3, respectively.",
            "1-3, respectively.10",
            dest_bbox=[72.0, 104.0, 190.0, 117.0],
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["type"] == "text"


class TestCrossPageUppercaseTail:
    def test_proper_noun_continuation_merges(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = _two_page_fixture(
            "In the last minute of the down phase, most contracts were sold by",
            (
                "High Frequency Traders reducing their inventories. Panel B "
                "of the table presents further results for the following day."
            ),
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 1
        assert last_elem["content"].endswith(
            "sold by High Frequency Traders reducing their inventories."
        )
        assert first_elem["content"].startswith("Panel B of the table")

    def test_caption_lead_is_rejected(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, _last, first_elem = _two_page_fixture(
            "In the last minute of the down phase, most contracts were sold by",
            "Table VII presents the regression results of both components.",
        )
        first_elem["layout_class"] = "heading"
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["content"].startswith("Table VII")


class TestCrossPageLowercaseWithApparentBoundary:
    def test_lowercase_start_overrides_source_boundary(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        # The page-end "boundary" is the period of an abbreviation; the
        # lowercase opener on the next page is signal 2 of the OR policy.
        page_extractions, page_rects, last_elem, first_elem = _two_page_fixture(
            "These patterns are consistent with the findings of Smith et al.",
            (
                "who document similar inventory behavior. The remaining "
                "sections examine the aggregate imbalance in more detail."
            ),
        )
        audit_log = []
        absorbed = _merge_cross_page_sentences(
            page_extractions, 2, page_rects, audit_log
        )
        assert absorbed == 1
        assert last_elem["content"].endswith(
            "et al. who document similar inventory behavior."
        )
        assert first_elem["content"].startswith("The remaining sections")
        assert (
            audit_log[0]["source_continuation_reason"]
            == "ambiguous-terminal-abbreviation"
        )


class TestCrossPageCapitalizedNameBreak:
    """2014.pdf pages 13→14: '... half that of High Frequency | Traders.'"""

    SOURCE = (
        "The aggregate inventory response of Market Makers is even "
        "smaller - roughly half that of High Frequency"
    )

    def test_single_capitalized_word_tail_merges(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = _two_page_fixture(
            self.SOURCE,
            (
                "Traders. During the early moments of the Flash Crash, "
                "prices declined further while volume spiked sharply."
            ),
        )
        audit_log = []
        absorbed = _merge_cross_page_sentences(
            page_extractions, 2, page_rects, audit_log
        )
        assert absorbed == 1
        assert last_elem["content"].endswith("of High Frequency Traders.")
        assert first_elem["content"].startswith("During the early moments")
        assert audit_log[0]["carried_tail"] == "Traders."

    def test_source_signal_allows_capitalized_tail(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _cross_page_tail_start_acceptable,
        )

        assert _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True
        )
        assert _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True
        )
        # Non-body captions are rejected before this lexical OR helper.
        assert _cross_page_tail_start_acceptable(
            "Table VII.", source_dangling=True
        )


class TestCrossPageUnterminatedFragment:
    """2014.pdf pages 17→18: the source omits the sentence-final period."""

    SOURCE = (
        "These quantities also exceed the averages by similar multiples. "
        "During the up phase, gross purchases and"
    )
    FRAGMENT = (
        "sales by Opportunistic Traders increased from 39,535 and 37,317 "
        "contracts during May 3-5 to 306,326 and 302,417 contracts"
    )

    @staticmethod
    def _pages(source, fragment, follower):
        import fitz

        last_elem = {
            "type": "text",
            "layout_class": "body",
            "content": source,
            "bbox": [72.0, 620.0, 518.0, 715.0],
            "y": 620.0,
            "x": 72.0,
            "fontsize": 12.0,
        }
        first_elem = {
            "type": "text",
            "layout_class": "scattered",
            "content": fragment,
            "bbox": [72.0, 104.0, 518.0, 131.0],
            "y": 104.0,
            "x": 72.0,
            "fontsize": 12.0,
        }
        follower_elem = {
            "type": "text",
            "layout_class": "body",
            "content": follower,
            "bbox": [72.0, 133.0, 518.0, 175.0],
            "y": 133.0,
            "x": 72.0,
            "fontsize": 12.0,
        }
        page_extractions = {
            0: {"elements": [last_elem]},
            1: {"elements": [first_elem, follower_elem]},
        }
        page_rects = {0: fitz.Rect(0, 0, 612, 792), 1: fitz.Rect(0, 0, 612, 792)}
        return page_extractions, page_rects, last_elem, first_elem

    def test_unterminated_fragment_is_absorbed(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = self._pages(
            self.SOURCE,
            self.FRAGMENT,
            (
                "As the table shows, the process by which the market absorbs "
                "such imbalances is a confluence of different responses."
            ),
        )
        audit_log = []
        absorbed = _merge_cross_page_sentences(
            page_extractions, 2, page_rects, audit_log
        )
        assert absorbed == 1
        assert last_elem["content"].endswith(
            "gross purchases and " + self.FRAGMENT
        )
        assert first_elem["type"] == "text_merged_away"
        assert audit_log[0]["tail_unterminated"] is True

    def test_lowercase_follower_blocks_absorption(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, _last, first_elem = self._pages(
            self.SOURCE,
            self.FRAGMENT,
            "and continued rising through the close of the trading session.",
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["type"] == "text"

    def test_lowercase_destination_signal_overrides_terminated_source(self):
        from phoena_translator.pdf.semantics.cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, last_elem, first_elem = self._pages(
            self.SOURCE.rstrip() + " sales.",
            self.FRAGMENT,
            (
                "As the table shows, the process by which the market absorbs "
                "such imbalances is a confluence of different responses."
            ),
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 1
        assert last_elem["content"].endswith(self.FRAGMENT)
        assert first_elem["type"] == "text_merged_away"


class TestTablePreservation:
    """2026-07-24 policy: tables keep their original English ink verbatim."""

    def test_table_dominated_page_is_preserved(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        elements = [_table_test_elem("table") for _ in range(5)] + [
            _table_test_elem("body"),
            _table_test_elem("scattered"),
            _table_test_elem("body"),
        ]
        pages = {0: {"elements": elements}}
        preserved = _mark_pdf_table_preserved_elements(pages)
        assert len(preserved) == 5
        assert all(
            e.get("skip_translate_reason") == "table_preserved"
            for e in elements[:5]
        )
        assert all("skip_translate_reason" not in e for e in elements[5:])
        assert {entry["trigger"] for entry in preserved} == {"page-density"}

    def test_prose_page_stray_table_elements_keep_translating(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        elements = [_table_test_elem("body") for _ in range(6)] + [
            _table_test_elem("table"),
            _table_test_elem("table"),
        ]
        pages = {0: {"elements": elements}}
        assert _mark_pdf_table_preserved_elements(pages) == []
        assert all("skip_translate_reason" not in e for e in elements)

    def test_region_detected_cells_always_preserved(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        elements = [_table_test_elem("body") for _ in range(6)] + [
            _table_test_elem("table", table_hint=True)
        ]
        pages = {0: {"elements": elements}}
        preserved = _mark_pdf_table_preserved_elements(pages)
        assert len(preserved) == 1
        assert preserved[0]["trigger"] == "region"
        assert elements[6]["skip_translate_reason"] == "table_preserved"

    def test_grid_union_sweeps_misclassified_cells(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        cells = [
            _table_test_elem("table", bbox=[100.0, 200.0 + 20 * row, 500.0, 215.0 + 20 * row])
            for row in range(6)
        ]
        header_cell = _table_test_elem(
            "scattered", content="May 6th", bbox=[380.0, 204.0, 460.0, 214.0]
        )
        caption = _table_test_elem(
            "scattered",
            content="Table I: Market Descriptive Statistics",
            bbox=[180.0, 120.0, 430.0, 135.0],
        )
        notes = _table_test_elem(
            "body",
            content="This table presents summary statistics for the contract.",
            bbox=[100.0, 400.0, 500.0, 440.0],
        )
        first_row_label = _table_test_elem(
            "scattered",
            content="Daily Trading Volume",
            bbox=[110.0, 182.0, 300.0, 196.0],
        )
        page_number = _table_test_elem(
            "table", content="28", bbox=[290.0, 730.0, 310.0, 742.0]
        )
        elements = cells + [
            header_cell,
            first_row_label,
            caption,
            notes,
            page_number,
        ]
        pages = {0: {"elements": elements}}
        preserved = _mark_pdf_table_preserved_elements(pages)
        assert header_cell["skip_translate_reason"] == "table_preserved"
        assert first_row_label["skip_translate_reason"] == "table_preserved"
        assert any(entry["trigger"] == "grid-union" for entry in preserved)
        assert "skip_translate_reason" not in caption
        # The stray table-classified page number must not stretch the grid
        # box down the page and swallow the notes paragraph.
        assert "skip_translate_reason" not in notes

    def test_title_page_author_block_keeps_translating(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        authors = [
            _table_test_elem(
                "table",
                content=name,
                bbox=[130.0, 210.0 + 18 * row, 460.0, 224.0 + 18 * row],
            )
            for row, name in enumerate(
                (
                    "Andrei Kirilenko—MIT Sloan School of Management",
                    "Albert S. Kyle—University of Maryland",
                    "Mehrdad Samadi—University of North Carolina",
                    "Tugkan Tuzun—Board of Governors",
                    "Original Version: October 1, 2010",
                    "This version: May 5, 2014",
                    "ABSTRACT",
                )
            )
        ]
        abstract = _table_test_elem(
            "body",
            content="This study offers an empirical analysis of the events.",
            bbox=[101.0, 360.0, 489.0, 439.0],
        )
        title = _table_test_elem(
            "scattered",
            content="The Flash Crash",
            bbox=[91.0, 145.0, 499.0, 166.0],
        )
        elements = authors + [abstract, title]
        pages = {0: {"elements": elements}}
        assert _mark_pdf_table_preserved_elements(pages) == []
        assert all("skip_translate_reason" not in e for e in elements)

    def test_existing_skip_reason_is_not_overridden(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        elements = [
            _table_test_elem("table", skip_translate_reason="watermark")
            for _ in range(12)
        ]
        pages = {0: {"elements": elements}}
        assert _mark_pdf_table_preserved_elements(pages) == []
        assert all(
            e["skip_translate_reason"] == "watermark" for e in elements
        )

    def test_preserved_elements_leave_translation_set(self):
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        marked = _table_test_elem(
            "table",
            content="Trading Volume 893,262",
            skip_translate_reason="table_preserved",
        )
        assert not _pdf_element_requires_translation(marked)

    # 2026-07-27: page 4 of a Bank of Japan Review shipped 100% English.  A
    # chart page makes the layout classifier stamp "table" on short mid-page
    # paragraphs; one of those seeded the sweep, and the sweep — one box grown
    # from cells in BOTH text columns — then walked up the page.
    def test_body_paragraph_stamped_table_keeps_translating(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        cells = [
            _table_test_elem(
                "table",
                content=f"{row}.75 12.3",
                fontsize=6.8,
                bbox=[64.0, 552.0 + row * 14.0, 278.0, 564.0 + row * 14.0],
            )
            for row in range(12)
        ]
        paragraph = _table_test_elem(
            "table",
            content=(
                "Based on the above understanding, we here try to capture "
                "algorithmic trading developments from late February."
            ),
            fontsize=10.6,
            bbox=[308.0, 477.0, 541.0, 503.0],
        )
        body = _table_test_elem(
            "body",
            content=(
                "Estimation results are as follows. First, estimation results "
                "using fast-paced orders as a proxy indicator of algorithmic "
                "trading show a negative coefficient."
            ),
            fontsize=10.6,
            bbox=[56.0, 153.0, 289.0, 515.0],
        )
        elements = cells + [paragraph, body]
        preserved = _mark_pdf_table_preserved_elements({0: {"elements": elements}})
        assert len(preserved) == len(cells)
        assert "skip_translate_reason" not in paragraph
        assert "skip_translate_reason" not in body

    def test_grid_sweep_does_not_cross_text_columns(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        # A table low in the left column and a chart in the right one, their
        # rows interleaved in y — exactly the shape that made one grown box
        # span the page.
        left = [
            _table_test_elem(
                "table",
                content=f"{row}.75 12.3",
                fontsize=6.8,
                bbox=[64.0, 552.0 + row * 14.0, 278.0, 564.0 + row * 14.0],
            )
            for row in range(4)
        ]
        right = [
            _table_test_elem(
                "table",
                content=f"1{row}.5 4.2",
                fontsize=6.3,
                bbox=[314.0, 558.0 + row * 14.0, 531.0, 570.0 + row * 14.0],
            )
            for row in range(6)
        ]
        # Left column, 2pt under the bottom edge of the two-column box but
        # 36pt clear of the left-hand table's own last row.
        neighbour = _table_test_elem(
            "scattered",
            content="Fast Executions",
            fontsize=6.3,
            bbox=[64.0, 642.0, 278.0, 650.0],
        )
        elements = left + right + [neighbour]
        _mark_pdf_table_preserved_elements({0: {"elements": elements}})
        assert all(
            e["skip_translate_reason"] == "table_preserved"
            for e in left + right
        )
        assert "skip_translate_reason" not in neighbour

    def test_grid_sweep_leaves_stacked_reference_entries_alone(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        # A bibliography page reads to every other signal like a table: short
        # stacked blocks, two columns, a year in every entry.
        stamped = [
            _table_test_elem(
                "table",
                content=f"Author, A. {1990 + row}. “A paper title.”",
                fontsize=8.0,
                bbox=[66.0, 147.0 + row * 20.0, 236.0, 165.0 + row * 20.0],
            )
            for row in range(8)
        ]
        entry = _table_test_elem(
            "body",
            content=(
                "Rothman, Matthew S. 2007c. “Rebalance of Large Cap Quant "
                "Portfolios,” Lehman Brothers Equity Research."
            ),
            fontsize=8.0,
            bbox=[66.0, 308.0, 236.0, 346.0],
        )
        elements = stamped + [entry]
        _mark_pdf_table_preserved_elements({0: {"elements": elements}})
        assert all(
            e["skip_translate_reason"] == "table_preserved" for e in stamped
        )
        assert "skip_translate_reason" not in entry

    def test_grid_sweep_stops_at_the_running_footer(self):
        import fitz

        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        cells = [
            _table_test_elem(
                "table",
                content=f"{row}.75 12.3",
                fontsize=6.5,
                bbox=[314.0, 700.0 + row * 12.0, 531.0, 710.0 + row * 12.0],
            )
            for row in range(6)
        ]
        footer = _table_test_elem(
            "scattered",
            content="Bank of Japan August 2020",
            fontsize=10.6,
            bbox=[402.0, 791.8, 533.9, 804.4],
        )
        elements = cells + [footer]
        _mark_pdf_table_preserved_elements(
            {0: {"elements": elements}},
            None,
            {0: fitz.Rect(0.0, 0.0, 595.32, 841.92)},
        )
        assert "skip_translate_reason" not in footer

    def test_page_prose_freeze_rolls_the_whole_page_back(self):
        from phoena_translator.pdf.semantics.table_preservation import (
            _mark_pdf_table_preserved_elements,
        )

        # A region detector that swallows a whole column: every paragraph
        # arrives already carrying table_hint, which the marker trusts.
        paragraphs = [
            _table_test_elem(
                "table",
                table_hint=True,
                content=(
                    f"Paragraph {index} of running body prose that the region "
                    "detector wrongly claimed as one enormous grid cell "
                    "spanning the entire text column of this page."
                ),
                fontsize=10.6,
                bbox=[56.0, 100.0 + index * 90.0, 289.0, 180.0 + index * 90.0],
            )
            for index in range(5)
        ]
        rollbacks: list[dict] = []
        preserved = _mark_pdf_table_preserved_elements(
            {0: {"elements": paragraphs}},
            rollbacks,
        )
        assert preserved == []
        assert all("skip_translate_reason" not in e for e in paragraphs)
        assert rollbacks and rollbacks[0]["page"] == 1
        assert rollbacks[0]["released_elements"] == 5

    def test_untranslated_delivered_page_is_reported(self):
        from phoena_translator.pdf.verify.leaks import (
            collect_untranslated_delivered_pages,
        )

        frozen = [
            _table_test_elem(
                "body",
                content=(
                    f"Paragraph {index} of running body prose delivered to the "
                    "reader in the source language because every element on "
                    "the page was exempted from translation."
                ),
                fontsize=10.6,
                bbox=[56.0, 100.0 + index * 90.0, 289.0, 180.0 + index * 90.0],
                skip_translate_reason="table_preserved",
            )
            for index in range(4)
        ]
        findings = collect_untranslated_delivered_pages({0: {"elements": frozen}})
        assert len(findings) == 1
        assert findings[0]["page"] == 1
        assert findings[0]["translatable_prose_chars"] == 0
        assert findings[0]["reasons"] == ["table_preserved"]

    def test_all_table_page_is_not_reported_as_untranslated(self):
        from phoena_translator.pdf.verify.leaks import (
            collect_untranslated_delivered_pages,
        )

        cells = [
            _table_test_elem(
                "table",
                content=f"Contract {index} 12.3 units",
                fontsize=6.5,
                bbox=[64.0, 100.0 + index * 14.0, 278.0, 112.0 + index * 14.0],
                skip_translate_reason="table_preserved",
            )
            for index in range(30)
        ]
        assert collect_untranslated_delivered_pages({0: {"elements": cells}}) == []

    def test_config_flag_wiring(self):
        from pathlib import Path

        cfg = AppConfig.from_env({}, home=Path("/tmp"))
        assert cfg.pdf_preserve_tables is False
        cfg_off = AppConfig.from_env(
            {"TRANSLATOR_PDF_PRESERVE_TABLES": "false"},
            home=Path("/tmp"),
        )
        assert cfg_off.pdf_preserve_tables is False

    def test_native_table_cells_prevent_section_header_overmerge(self):
        import fitz

        from phoena_translator.pdf.semantics.continuations import (
            _pdf_internal_paragraph_lead_residuals,
        )
        from phoena_translator.pdf.semantics.tables import (
            _merge_pdf_semantic_table_cells,
        )

        def cell_line(text, y0, y1):
            return _table_test_elem(
                "table",
                content=text,
                table_hint=True,
                preserve_source_style=True,
                fontsize=8.0,
                bbox=[10.0, y0, 190.0, y1],
            )

        elements = [
            cell_line("Jump frequency: indicator for the", 10.0, 19.0),
            cell_line("regularity of price evolution.", 20.0, 29.0),
            cell_line("(ii) Market drivers", 31.0, 39.0),
            cell_line("Flows by location or customer group:", 41.0, 50.0),
            cell_line("informs about market-moving transactions.", 51.0, 60.0),
        ]
        regions = [
            {
                "rect": [0.0, 0.0, 200.0, 70.0],
                "cells": [
                    [0.0, 8.0, 200.0, 30.0],
                    [0.0, 30.0, 200.0, 40.0],
                    [0.0, 40.0, 200.0, 62.0],
                ],
                "fallback_open": False,
                "row_count": 3,
                "col_count": 1,
            }
        ]

        merged, merge_count = _merge_pdf_semantic_table_cells(
            elements,
            [fitz.Rect(regions[0]["rect"])],
            fitz.Rect(0.0, 0.0, 200.0, 100.0),
            table_regions=regions,
        )

        assert merge_count == 2
        assert [element["content"] for element in merged] == [
            "Jump frequency: indicator for the regularity of price evolution.",
            "(ii) Market drivers",
            (
                "Flows by location or customer group: "
                "informs about market-moving transactions."
            ),
        ]
        assert _pdf_internal_paragraph_lead_residuals(merged) == []

    def test_native_table_cell_allows_multiple_bullets(self):
        import fitz

        from phoena_translator.pdf.semantics.continuations import (
            _pdf_internal_paragraph_lead_residuals,
        )
        from phoena_translator.pdf.semantics.tables import (
            _merge_pdf_semantic_table_cells,
        )

        elements = [
            _table_test_elem(
                "table",
                content="• What are current market conditions?",
                table_hint=True,
                preserve_source_style=True,
                fontsize=8.0,
                bbox=[10.0, 10.0, 190.0, 19.0],
            ),
            _table_test_elem(
                "table",
                content="• What is the quality of execution?",
                table_hint=True,
                preserve_source_style=True,
                fontsize=8.0,
                bbox=[10.0, 21.0, 190.0, 30.0],
            ),
        ]
        regions = [
            {
                "rect": [0.0, 0.0, 200.0, 40.0],
                "cells": [[0.0, 8.0, 200.0, 32.0]],
                "fallback_open": False,
                "row_count": 1,
                "col_count": 1,
            }
        ]

        merged, merge_count = _merge_pdf_semantic_table_cells(
            elements,
            [fitz.Rect(regions[0]["rect"])],
            fitz.Rect(0.0, 0.0, 200.0, 100.0),
            table_regions=regions,
        )

        assert merge_count == 1
        assert merged[0]["semantic_native_table_cell"] is True
        assert _pdf_internal_paragraph_lead_residuals(merged) == []

    def test_two_column_study_group_directory_preserves_names(self):
        from phoena_translator.pdf.semantics.entities import (
            _mark_pdf_entity_directory_elements,
        )
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        heading = _table_test_elem(
            "scattered",
            content="Members of the study group",
            bbox=[100.0, 80.0, 300.0, 94.0],
        )
        institutions = [
            "Reserve Bank of Australia",
            "Bank of Canada",
            "European Central Bank",
            "Bank of France",
            "Bank of Japan",
            "Swiss National Bank",
        ]
        names = [
            "Jason Griffin",
            "Rhonda Staskow",
            "Istvan Mak",
            "Alexis Laming",
            "Masao Fujiwara",
            "Thomas Maag",
        ]
        rows = []
        for row, (institution, name) in enumerate(zip(institutions, names)):
            y0 = 120.0 + row * 30.0
            rows.extend(
                [
                    _table_test_elem(
                        "table",
                        content=institution,
                        bbox=[100.0, y0, 230.0, y0 + 12.0],
                    ),
                    _table_test_elem(
                        "table",
                        content=name,
                        bbox=[340.0, y0, 470.0, y0 + 12.0],
                    ),
                ]
            )

        elements = [heading, *rows]
        assert _mark_pdf_entity_directory_elements(elements) == len(rows)
        assert all(
            element.get("entity_directory_hint") for element in rows
        )
        assert all(
            not _pdf_element_requires_translation(element) for element in rows
        )


# ---------------------------------------------------------------------------
# 2026-07-25 strict-audit release blockers C001 / C002 / C004
# ---------------------------------------------------------------------------


def test_complete_source_lowercase_destination_is_absorbed_by_or_policy():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source_text = "Demand remained stable."
    destination_text = (
        "the next section discusses a separate result. "
        "Its ownership must stay on page two."
    )
    pages, page_rects, source, destination = _two_page_fixture(
        source_text,
        destination_text,
    )
    audit_log = []

    absorbed = _merge_cross_page_sentences(pages, 2, page_rects, audit_log)

    assert absorbed == 1
    assert source["content"] == (
        "Demand remained stable. "
        "the next section discusses a separate result."
    )
    assert destination["content"] == "Its ownership must stay on page two."
    assert destination["type"] == "text"
    assert audit_log[0]["source_continuation_reason"] == "complete"


def test_cross_page_first_sentence_boundary_keeps_abbreviation_and_marker():
    from phoena_translator.pdf.semantics.cross_page import (
        _find_leading_sentence_tail,
    )

    assert _find_leading_sentence_tail(
        (
            "trading in the U.S. Treasury market ended calmly. "
            "Another sentence remains."
        ),
        4000,
    ) == (
        "trading in the U.S. Treasury market ended calmly.",
        "Another sentence remains.",
    )
    assert _find_leading_sentence_tail(
        "the first sentence ended.\uf085 Second sentence remains.",
        4000,
    ) == (
        "the first sentence ended.\uf085",
        "Second sentence remains.",
    )
    assert _find_leading_sentence_tail(
        "the first clause; Another sentence remains.",
        4000,
    ) == (
        "the first clause;",
        "Another sentence remains.",
    )


def test_cross_page_both_or_signals_absent_keeps_pages_independent():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    pages, rects, source, destination = _two_page_fixture(
        "Demand remained stable;",
        "The next section begins independently. Another sentence follows.",
    )
    assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
    assert source["content"] == "Demand remained stable;"
    assert destination["content"].startswith("The next section")


def test_cross_page_merge_keeps_midpage_fragment_with_local_predecessor():
    import fitz

    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source = {
        "type": "text",
        "layout_class": "body",
        "content": "The prior page closes with a complete sentence.",
        "paragraphs": [
            {
                "plain": "The prior page closes with a complete sentence.",
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 645.0, 542.0, 670.0],
        "y": 645.0,
        "x": 72.0,
        "fontsize": 10.02,
    }
    local_lead = {
        "type": "text",
        "layout_class": "table",
        "table_hint": True,
        "content": (
            "When I inspect the state today, the picture is critical because "
            "scaling laws"
        ),
        "paragraphs": [
            {
                "plain": (
                    "When I inspect the state today, the picture is critical "
                    "because scaling laws"
                ),
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 379.92, 542.25, 392.75],
        "y": 379.92,
        "x": 72.0,
        "fontsize": 10.02,
    }
    local_continuation = {
        "type": "text",
        "layout_class": "body",
        "content": (
            "continue to hold and the models have become extremely capable. "
            "Regulation is needed and will come."
        ),
        "paragraphs": [
            {
                "plain": (
                    "continue to hold and the models have become extremely "
                    "capable. Regulation is needed and will come."
                ),
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 391.93, 542.27, 428.76],
        "y": 391.93,
        "x": 72.0,
        "fontsize": 10.02,
    }
    pages = {
        0: {"elements": [source]},
        1: {"elements": [local_lead, local_continuation]},
    }
    rect = fitz.Rect(0, 0, 612, 792)
    audit_log = []

    assert _merge_cross_page_sentences(
        pages,
        2,
        {0: rect, 1: rect},
        audit_log,
    ) == 0
    assert source["content"].endswith("complete sentence.")
    assert local_continuation["content"].startswith("continue to hold")
    assert audit_log == []


def test_cross_page_merge_keeps_regular_wrap_after_bold_page_top_line():
    import fitz

    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source = {
        "type": "text",
        "layout_class": "body",
        "content": "The prior page closes with a complete sentence.",
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 9.96,
    }
    bold_lead = {
        "type": "text",
        "layout_class": "scattered",
        "content": (
            "Our view is that the economy will continue to grow robustly, "
            "which will put upward"
        ),
        "paragraphs": [
            {
                "plain": (
                    "Our view is that the economy will continue to grow "
                    "robustly, which will put upward"
                ),
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 72.42, 542.25, 83.97],
        "y": 72.42,
        "x": 72.0,
        "fontsize": 9.96,
        "bold": True,
    }
    regular_wrap = {
        "type": "text",
        "layout_class": "scattered",
        "content": (
            "pressure on inflation and likely necessitate the hikes priced "
            "into the market over the next year."
        ),
        "paragraphs": [
            {
                "plain": (
                    "pressure on inflation and likely necessitate the hikes "
                    "priced into the market over the next year."
                ),
                "text_align": "left",
            }
        ],
        "bbox": [72.0, 84.42, 509.11, 95.97],
        "y": 84.42,
        "x": 72.0,
        "fontsize": 9.96,
        "bold": False,
    }
    pages = {
        0: {"elements": [source]},
        1: {"elements": [bold_lead, regular_wrap]},
    }
    rect = fitz.Rect(0, 0, 612, 792)
    audit_log = []

    assert _merge_cross_page_sentences(
        pages,
        2,
        {0: rect, 1: rect},
        audit_log,
    ) == 0
    assert source["content"].endswith("complete sentence.")
    assert regular_wrap["content"].startswith("pressure on inflation")
    assert audit_log == []


def test_cross_page_merge_rejects_complete_source_midpage_lowercase_opener():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    pages, rects, source, destination = _two_page_fixture(
        "Demand remained stable.",
        (
            "the next section begins independently. "
            "Another sentence remains on page two."
        ),
        dest_bbox=[72.0, 390.0, 518.0, 430.0],
    )
    destination["y"] = 390.0

    assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
    assert source["content"] == "Demand remained stable."
    assert destination["content"].startswith("the next section")


def test_continuation_decision_is_frozen_and_string_compatible():
    from dataclasses import FrozenInstanceError

    from phoena_translator.pdf.semantics.cross_page import (
        ContinuationDecision,
        SourceContinuationEvidence,
        _cross_page_continuation_decision,
    )

    complete = _cross_page_continuation_decision("Demand remained stable.")
    abbreviation = _cross_page_continuation_decision(
        "The findings agree with Smith et al."
    )
    dangling = _cross_page_continuation_decision(
        "The explanation continues across"
    )
    footnoted = _cross_page_continuation_decision(
        "The transactions were recorded.14"
    )

    assert isinstance(complete, ContinuationDecision)
    assert complete.evidence is SourceContinuationEvidence.COMPLETE
    assert complete.audit_value == "complete"
    assert abbreviation.evidence is (
        SourceContinuationEvidence.AMBIGUOUS_TERMINAL_ABBREVIATION
    )
    assert abbreviation.audit_value == "ambiguous-terminal-abbreviation"
    assert dangling.evidence is SourceContinuationEvidence.DANGLING
    assert dangling.source_dangling is True
    assert footnoted.evidence is SourceContinuationEvidence.COMPLETE
    assert _cross_page_continuation_decision(
        "The findings agree with Smith et al."
    ).audit_value == "ambiguous-terminal-abbreviation"
    with pytest.raises(TypeError, match="SourceContinuationEvidence"):
        ContinuationDecision("complete")
    with pytest.raises(FrozenInstanceError):
        complete.evidence = SourceContinuationEvidence.DANGLING


def test_cross_page_merge_allows_page_turn_to_first_destination_column():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    pages, rects, source, destination = _two_page_fixture(
        "The explanation continues across",
        (
            "columns without preserving ownership. A separate sentence "
            "must remain on the next page."
        ),
        dest_bbox=[330.0, 104.0, 540.0, 145.0],
    )
    source["bbox"] = [72.0, 620.0, 288.0, 715.0]
    destination["x"] = 330.0
    audit_log = []

    assert _merge_cross_page_sentences(pages, 2, rects, audit_log) == 1
    assert source["content"].endswith(
        "columns without preserving ownership."
    )
    assert destination["content"] == (
        "A separate sentence must remain on the next page."
    )
    assert destination["type"] == "text"
    assert audit_log[0]["carried_tail"] == (
        "columns without preserving ownership."
    )


def test_cross_page_merge_skips_page_top_heading_and_uses_first_body():
    import fitz

    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source = {
        "type": "text",
        "layout_class": "body",
        "content": "The explanation continues across",
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    heading = {
        "type": "text",
        "layout_class": "heading",
        "content": "Results",
        "bbox": [72.0, 78.0, 250.0, 102.0],
        "y": 78.0,
        "x": 72.0,
        "fontsize": 18.0,
        "bold": True,
    }
    continuation = {
        "type": "text",
        "layout_class": "body",
        "content": "pages without changing ownership.",
        "bbox": [72.0, 120.0, 518.0, 150.0],
        "y": 120.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    pages = {
        0: {"elements": [source]},
        1: {"elements": [heading, continuation]},
    }
    rects = {
        0: fitz.Rect(0, 0, 612, 792),
        1: fitz.Rect(0, 0, 612, 792),
    }

    assert _merge_cross_page_sentences(pages, 2, rects, []) == 1
    assert source["content"] == (
        "The explanation continues across pages without changing ownership."
    )
    assert heading["type"] == "text"
    assert continuation["type"] == "text_merged_away"


def test_cross_page_merge_skips_graph_title_and_table_labels():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    pages, rects, source, _unused = _two_page_fixture(
        "The monitoring discussion continues across",
        "unused.",
    )
    prior_heading = {
        "type": "text",
        "layout_class": "scattered",
        "content": "2.2 Why do central banks monitor fast-paced electronic markets?",
        "bbox": [72.0, 200.0, 500.0, 218.0],
        "y": 200.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    chart_title = {
        "type": "text",
        "layout_class": "scattered",
        "content": "Why do central banks monitor fast-paced electronic markets?",
        "bbox": [72.0, 82.0, 390.0, 98.0],
        "y": 82.0,
        "x": 72.0,
        "fontsize": 11.0,
    }
    chart_unit = {
        "type": "text",
        "layout_class": "scattered",
        "content": "In per cent of respondents",
        "bbox": [72.0, 105.0, 220.0, 118.0],
        "y": 105.0,
        "x": 72.0,
        "fontsize": 10.0,
    }
    chart_label = {
        "type": "text",
        "layout_class": "scattered",
        "content": "Graph 1",
        "bbox": [470.0, 105.0, 520.0, 118.0],
        "y": 105.0,
        "x": 470.0,
        "fontsize": 10.0,
    }
    table_cell = {
        "type": "text",
        "layout_class": "table",
        "table_hint": True,
        "content": "Core functions for FPM monitoring",
        "bbox": [72.0, 128.0, 280.0, 320.0],
        "y": 128.0,
        "x": 72.0,
        "fontsize": 10.0,
    }
    body = {
        "type": "text",
        "layout_class": "body",
        "content": (
            "reserves management and implementation of exchange rate policy. "
            "A complete second sentence remains on the next page."
        ),
        "bbox": [72.0, 360.0, 518.0, 430.0],
        "y": 360.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    pages[0]["elements"] = [prior_heading, source]
    pages[1]["elements"] = [
        chart_title,
        chart_unit,
        chart_label,
        table_cell,
        body,
    ]

    absorbed = _merge_cross_page_sentences(pages, 2, rects, [])

    assert absorbed == 1
    assert source["content"].endswith(
        "reserves management and implementation of exchange rate policy."
    )
    assert chart_title["content"].startswith("Why do central banks")
    assert chart_unit["content"] == "In per cent of respondents"
    assert chart_label["content"] == "Graph 1"
    assert body["content"].startswith("A complete second sentence")


def test_cross_page_merge_does_not_absorb_glossary_entries():
    import fitz

    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source = {
        "type": "text",
        "layout_class": "body",
        "content": "The final bibliography entry has no terminal mark",
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 10.0,
    }
    heading = {
        "type": "text",
        "layout_class": "scattered",
        "content": "Glossary",
        "bbox": [72.0, 82.0, 180.0, 102.0],
        "y": 82.0,
        "x": 72.0,
        "fontsize": 14.0,
    }
    entry = {
        "type": "text",
        "layout_class": "body",
        "content": (
            "Aggregator: Technology that combines prices from several "
            "liquidity providers."
        ),
        "bbox": [72.0, 120.0, 518.0, 160.0],
        "y": 120.0,
        "x": 72.0,
        "fontsize": 10.0,
    }
    pages = {
        0: {"elements": [source]},
        1: {"elements": [heading, entry]},
    }
    rects = {
        0: fitz.Rect(0, 0, 612, 792),
        1: fitz.Rect(0, 0, 612, 792),
    }

    assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
    assert source["content"].endswith("terminal mark")
    assert entry["content"].startswith("Aggregator:")


def test_cross_page_body_sentences_may_start_with_label_words():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    for opener in (
        "Note that markets remained liquid.",
        "Figure 2 shows a decline in spreads.",
        "Section 3 discusses the event window.",
    ):
        pages, rects, source, destination = _two_page_fixture(
            "The explanation continues and readers should",
            f"{opener} A separate sentence remains on the next page.",
        )
        destination["layout_class"] = "body"

        assert _merge_cross_page_sentences(pages, 2, rects, []) == 1
        assert source["content"].endswith(opener)
        assert destination["content"].startswith("A separate sentence")


def test_three_page_merges_keep_two_independent_source_owners():
    import fitz

    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    page_one_source = {
        "type": "text",
        "layout_class": "body",
        "content": "The first explanation continues across",
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    page_two_first = {
        "type": "text",
        "layout_class": "scattered",
        "content": (
            "pages without changing its owner. A complete page-two sentence "
            "remains here."
        ),
        "bbox": [72.0, 104.0, 518.0, 145.0],
        "y": 104.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    page_two_source = {
        "type": "text",
        "layout_class": "body",
        "content": "A different explanation continues across",
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    page_three_first = {
        "type": "text",
        "layout_class": "scattered",
        "content": (
            "three pages without mixing either owner. A final independent "
            "sentence remains here."
        ),
        "bbox": [72.0, 104.0, 518.0, 145.0],
        "y": 104.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    pages = {
        0: {"elements": [page_one_source]},
        1: {"elements": [page_two_first, page_two_source]},
        2: {"elements": [page_three_first]},
    }
    rect = fitz.Rect(0, 0, 612, 792)
    audit_log = []

    assert _merge_cross_page_sentences(
        pages,
        3,
        {0: rect, 1: rect, 2: rect},
        audit_log,
    ) == 2
    assert page_one_source["content"].endswith(
        "across pages without changing its owner."
    )
    assert page_two_first["content"] == "A complete page-two sentence remains here."
    assert page_two_source["content"].endswith(
        "across three pages without mixing either owner."
    )
    assert (
        page_three_first["content"]
        == "A final independent sentence remains here."
    )
    assert [
        (
            entry["source_page"],
            entry["destination_page"],
            entry["source_element"],
            entry["destination_element"],
            entry["carried_tail"],
        )
        for entry in audit_log
    ] == [
        (1, 2, 0, 0, "pages without changing its owner."),
        (2, 3, 1, 0, "three pages without mixing either owner."),
    ]


def test_a_preserved_element_is_refused_as_a_cross_page_merge_endpoint():
    from phoena_translator.pdf.semantics.cross_page import (
        _merge_cross_page_sentences,
    )

    source_text = "Demand remained stable."
    tail = "the next section discusses a separate result."
    remainder = "Its ownership must stay on page two."
    pages, page_rects, _source, destination = _two_page_fixture(
        source_text,
        f"{tail} {remainder}",
    )
    # Stamping first is what ``reconcile_extraction`` now does; the merge must
    # then decline, because a preserved element re-renders its ORIGINAL ink
    # and would show the carried tail on both pages.
    destination["skip_translate_reason"] = "formula_risk_preserved"
    audit_log = []

    absorbed = _merge_cross_page_sentences(pages, 2, page_rects, audit_log)

    assert absorbed == 0
    assert not [d for d in audit_log if d.get("decision") == "accepted"]
    assert destination["content"] == f"{tail} {remainder}"


def test_mid_sentence_stub_running_straight_into_a_formula_is_preserved():
    from phoena_translator.pdf.semantics.formula_policy import (
        _preserve_pdf_formula_mid_sentence_neighbors,
    )

    pages = _mid_sentence_page(with_intervening_prose=False)

    preserved = _preserve_pdf_formula_mid_sentence_neighbors(pages)

    assert [entry["element"] for entry in preserved] == [0]
    assert pages[0]["elements"][0]["skip_translate_reason"] == (
        "formula_risk_preserved"
    )


def test_prose_between_the_stub_and_the_formula_blocks_preservation():
    from phoena_translator.pdf.semantics.formula_policy import (
        _preserve_pdf_formula_mid_sentence_neighbors,
    )

    # An inline math variable splits its own line, so the tail after the
    # variable becomes a separate element and the parent only *looks* like it
    # stops mid-sentence.  Its real continuation is prose, not the formula.
    pages = _mid_sentence_page(with_intervening_prose=True)

    preserved = _preserve_pdf_formula_mid_sentence_neighbors(pages)

    assert preserved == []
    assert "skip_translate_reason" not in pages[0]["elements"][0]
