"""PDF translation: targets, placeholders, validation rules and the translation stage.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging

import pytest
from review_support import (
    _table_test_elem,
)

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# M1 — no silent truncation of oversized candidates
# ---------------------------------------------------------------------------


def test_long_candidate_not_truncated():
    from phoena_translator.pdf.translate.page import _clean_translated_text

    source = "short source"
    candidate = "很长的候选译文段落" * 120
    cleaned, failures = _clean_translated_text(source, candidate, (), (), ())
    assert failures == ()
    assert len(cleaned) >= len(candidate) - 2


class TestRefusalMetaGate:
    SOURCE = "Discussion of policy implications"

    def test_chinese_refusal_needs_retry(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        assert _short_translation_needs_retry(
            self.SOURCE, "抱歉，我无法翻译此内容"
        )
        assert _short_translation_needs_retry(
            self.SOURCE, "我无法翻译此内容"
        )

    def test_meta_preamble_needs_retry(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        assert _short_translation_needs_retry(
            self.SOURCE, "好的，以下是译文：政策含义的讨论"
        )
        assert _short_translation_needs_retry(
            self.SOURCE, "译文：政策含义的讨论"
        )

    def test_ordinary_translation_passes(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        assert not _short_translation_needs_retry(
            self.SOURCE, "政策含义的讨论"
        )
        # 抱歉/无法 appearing as translated content, not as an opener.
        assert not _short_translation_needs_retry(
            "We regret that the data cannot be shared",
            "我们对数据不能共享表示歉意",
        )


class TestZeroProgressCircuitBreaker:
    @staticmethod
    def _context(pdf_audit, page_extractions):
        from phoena_translator.pdf.translate.stage import (
            PDFTranslationStageContext,
        )

        return PDFTranslationStageContext(
            task_id="t",
            total_pages=len(page_extractions),
            source_sha256="sha",
            workers=1,
            fail_open_to_source_page=True,
            logger=log,
            page_extractions=page_extractions,
            pdf_audit=pdf_audit,
            completed_indices=set(),
            save_translation_progress=lambda *args, **kwargs: None,
        )

    def test_all_translation_fallbacks_fail_the_task(self):
        from phoena_translator.pdf.errors import PDFPageTranslationError
        from phoena_translator.pdf.translate.stage import (
            _require_translated_page_progress,
        )

        context = self._context(
            {
                "source_page_fallbacks": [
                    {"page": 1, "stage": "translation"},
                    {"page": 2, "stage": "translation"},
                ]
            },
            {0: {"source_page_fallback": {}}, 1: {"source_page_fallback": {}}},
        )
        with pytest.raises(PDFPageTranslationError):
            _require_translated_page_progress(context)

    def test_any_cached_page_keeps_fail_open(self):
        from phoena_translator.pdf.translate.stage import (
            _require_translated_page_progress,
        )

        context = self._context(
            {
                "source_page_fallbacks": [
                    {"page": 1, "stage": "translation"},
                ]
            },
            {
                0: {"source_page_fallback": {}},
                1: {"elements": [], "cached": {0: "译"}},
            },
        )
        _require_translated_page_progress(context)

    def test_structure_fallbacks_do_not_trip_the_breaker(self):
        from phoena_translator.pdf.translate.stage import (
            _require_translated_page_progress,
        )

        context = self._context(
            {"source_page_fallbacks": [{"page": 1, "stage": "structure"}]},
            {0: {"source_page_fallback": {}}},
        )
        _require_translated_page_progress(context)


def test_pdf_lexical_angle_phrase_survives_translation_protection_and_rendering():
    """BOJ p4: ``<JPY short position>`` is prose, not an HTML tag."""
    from phoena_translator.pdf.render.paragraph_html import _pdf_text_to_html
    from phoena_translator.pdf.translate.inline_markup import _normalize_pdf_translation
    from phoena_translator.pdf.translate.placeholders import (
        _protect_pdf_inline_math_fragments,
        _restore_pdf_inline_math_fragments,
    )

    source = "valuation losses of inventories <JPY short position> caused by appreciation"
    normalized = _normalize_pdf_translation(source)
    assert normalized == source

    protected, records = _protect_pdf_inline_math_fragments(
        normalized,
        [
            {"text": "<", "font_kind": "unicode"},
            {"text": ">", "font_kind": "unicode"},
        ],
    )
    assert [record["original"] for record in records] == ["<", ">"]
    translated = protected.replace(
        "valuation losses of inventories ",
        "库存中",
    ).replace(
        "JPY short position",
        "日元空头头寸",
    ).replace(
        " caused by appreciation",
        "因日元升值而发生估值损失",
    )
    restored = _restore_pdf_inline_math_fragments(translated, records)
    assert restored == "库存中<日元空头头寸>因日元升值而发生估值损失"
    assert "&lt;日元空头头寸&gt;" in _pdf_text_to_html(restored)


def test_pdf_normalizer_still_removes_known_html_wrappers():
    from phoena_translator.pdf.translate.inline_markup import _normalize_pdf_translation

    assert _normalize_pdf_translation("<p>Hello</p>") == "Hello"


class TestCitationApparatusRetryExemption:
    """A bibliography line with no translatable prose must not be retried."""

    APPARATUS_ONLY = [
        "Hendershott, Terrence, Charles M. Jones,",
        "Bertsimas, Dimitris, and Andrew Lo. 1998.",
        "Black, Fischer, and Myron Scholes. 1973. “The",
        "Khandani, Amir E., and Andrew W. Lo. 2007.",
    ]
    MUST_STILL_RETRY = [
        # A quoted title of two or more words is translatable.
        "and Albert J. Menkveld. 2011. “Does Algorithmic Trading Improve "
        "Liquidity?” Journal of Finance 66(1): 1– 33.",
        # Sentence-case prose carries substantive lower-case words.
        "Algorithmic trading is part of a much broader trend in which "
        "computer-based automation has improved efficiency.",
        "The Trading Profits of High Frequency Traders and their market impact",
    ]

    def test_authors_and_year_returned_unchanged_is_accepted(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        for source in self.APPARATUS_ONLY:
            assert not _short_translation_needs_retry(source, source), source

    def test_translatable_content_still_drives_the_retry_ladder(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        for source in self.MUST_STILL_RETRY:
            assert _short_translation_needs_retry(source, source), source

    def test_quoted_title_word_count_gates_the_exemption(self):
        from phoena_translator.pdf.translate.validation import (
            _pdf_citation_apparatus_only,
            _pdf_quoted_title_word_count,
        )

        assert _pdf_quoted_title_word_count("Scholes. 1973. “The") == 1
        assert _pdf_quoted_title_word_count(
            "Lo. 1998. “Optimal Control of Execution Costs.”"
        ) == 5
        assert _pdf_citation_apparatus_only("Scholes, Myron, and F. Black. 1973.")
        assert not _pdf_citation_apparatus_only(
            "Lo, Andrew. 1998. “Optimal Control of Execution Costs.”"
        )


# ---------------------------------------------------------------------------
# 2026-08-31 — deal-table identifier rows must not burn retry ladders
# ---------------------------------------------------------------------------


class TestStructuredIdentifierRowExemption:
    """Rows of dates/codes/numbers are their own translation.

    Measured on a 2026-06-12 credit-research PDF: one deal-table page held
    28 rows such as ``01/16/26 QTSII 2026-1A A2``; every row failed the
    Chinese-output verdict, ran the four-attempt ladder with backoff twice
    (page pass + clean page retry), and still ended as a recorded source
    fallback.
    """

    DEAL_ROWS = [
        "01/16/26 COLO 2026-1A A2",
        "01/16/26 QTSII 2026-1A B",
        "02/02/26 CLGIX 2026-1A C",
        "03/13/26 FLX 2026-2A A2",
        "04/20/26 CLDHQ 2026-1A B1",
        "05/01/26 CONE 2026-DFW3 HRR",
        "AAPL US 4.65 2046",
        "1Q24 GDP",
    ]
    MUST_STILL_TRANSLATE = [
        "12 June 2026",
        "USD Issuance",
        "Deal Type Sub Type 1 Face Bench WAL",
        "FOR ANALYST CERTIFICATION(S) PLEASE SEE PAGE 37.",
        "FIGURE 13",
        "TABLE 2",
        "Executive Summary",
        "BCI, US",
        "Completed: 11-Jun-26, 22:43 GMT Released: 12-Jun-26, 10:30 GMT",
    ]

    def test_identifier_rows_are_recognized(self):
        from phoena_translator.pdf.translate.targets import (
            _looks_like_pdf_structured_identifier_row,
        )

        for row in self.DEAL_ROWS:
            assert _looks_like_pdf_structured_identifier_row(row), row

    def test_prose_and_caption_labels_are_not(self):
        from phoena_translator.pdf.translate.targets import (
            _looks_like_pdf_structured_identifier_row,
        )

        for text in self.MUST_STILL_TRANSLATE:
            assert not _looks_like_pdf_structured_identifier_row(text), text

    def test_echo_of_identifier_row_is_accepted(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
            _translation_part_needs_retry,
        )

        for row in self.DEAL_ROWS:
            assert not _short_translation_needs_retry(row, row), row
            assert not _translation_part_needs_retry(row, row), row

    def test_prose_echo_still_drives_the_retry_ladder(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        assert _short_translation_needs_retry(
            "Discussion of policy implications",
            "Discussion of policy implications",
        )

    def test_admission_gate_skips_identifier_rows(self):
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        # The measured rows carried single_line_heading and, on part of the
        # page, table layout with table_hint; none of those flags may keep
        # the row in the translation queue.
        scattered = _table_test_elem(
            "scattered",
            content="01/16/26 COLO 2026-1A A2",
            single_line_heading=True,
        )
        assert not _pdf_element_requires_translation(scattered)
        table = _table_test_elem(
            "table",
            content="03/13/26 FLX 2026-1A B",
            single_line_heading=True,
            table_hint=True,
        )
        assert not _pdf_element_requires_translation(table)

    def test_bold_or_body_rows_keep_translating(self):
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        bold_header = _table_test_elem(
            "scattered",
            content="01/16/26 COLO 2026-1A A2",
            bold=True,
        )
        assert _pdf_element_requires_translation(bold_header)
        body_row = _table_test_elem(
            "body",
            content="01/16/26 COLO 2026-1A A2",
        )
        assert _pdf_element_requires_translation(body_row)

    def test_header_cells_keep_translating(self):
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        header = _table_test_elem(
            "scattered",
            content="Security",
            bold=True,
            single_line_heading=True,
        )
        assert _pdf_element_requires_translation(header)


class TestSemanticRetrySleeps:
    """Verification failures retry immediately; transport errors keep backoff."""

    @staticmethod
    def _dependencies(create_chat_completion):
        from phoena_translator.pdf.translate.translator import (
            PDFTranslationDependencies,
        )

        return PDFTranslationDependencies(
            create_chat_completion=create_chat_completion,
            strip_think_tags=lambda text: text,
            system_prompt_text="translate to Chinese",
            logger=log,
        )

    @staticmethod
    def _response(content):
        class _Message:
            pass

        class _Choice:
            pass

        class _Response:
            pass

        message = _Message()
        message.content = content
        choice = _Choice()
        choice.message = message
        response = _Response()
        response.choices = [choice]
        return response

    def test_verification_failure_never_sleeps(self, monkeypatch):
        import time as time_module

        from phoena_translator.pdf.translate.translator import translate_text

        sleeps = []
        monkeypatch.setattr(
            time_module, "sleep", lambda seconds: sleeps.append(seconds)
        )
        source = "This is a plain English sentence that must be translated."
        calls = []

        def echo(messages, max_tokens, temperature):
            calls.append(messages)
            return self._response(source)

        result = translate_text(
            source,
            dependencies=self._dependencies(echo),
        )
        assert result == source          # best candidate after all attempts
        assert len(calls) == 4
        assert sleeps == []

    def test_transport_errors_keep_backoff(self, monkeypatch):
        import time as time_module

        import pytest as pytest_module

        from phoena_translator.pdf.translate.translator import translate_text

        sleeps = []
        monkeypatch.setattr(
            time_module, "sleep", lambda seconds: sleeps.append(seconds)
        )

        def broken(messages, max_tokens, temperature):
            raise ConnectionError("socket reset")

        with pytest_module.raises(RuntimeError):
            translate_text(
                "This sentence never gets an answer from the provider.",
                dependencies=self._dependencies(broken),
            )
        assert sleeps == [10, 20, 30]


class TestContactLineAndDateLabelExemption:
    """2026-08-31 round 2: JPM-style running headers must not burn ladders.

    Measured live: the byline ``Nikolaos Panigirtzoglou AC (44-20) 7134-7815
    <e-mail>`` escalated on every page of every document in a six-document
    queue, and the page-header date ``05 August 2026`` joined it.
    """

    CONTACT_LINES = [
        "Nikolaos Panigirtzoglou AC (44-20) 7134-7815 "
        "nikolaos.panigirtzoglou@jpmorgan.com",
        "(44-20) 7742 6565 mika.j.inkinen@jpmorgan.com J.P. Morgan Securities plc",
        "Mika Inkinen (44-20) 7742 6565",
        "Mayur Yeole (91 22) 6157 3872",
    ]
    MUST_STILL_TRANSLATE = [
        "For 2022: Yield change: 240bps",
        "42 day exponential weighted moving average (lambda = 0.98)",
        "Figure 8: Difference in the beta of the average of 10y UST yields",
        "compensation investors require for holding long-term instruments",
        "Krutik P Mehta",  # name alone, no phone: out of this rule's scope
        # A bare digit run is an axis sequence or a year range, not a phone.
        "Private Public 90 94 98 02 06 10 14 18 22 26",
        "Figure 3 Global Bond Supply 2010 2026",
    ]

    def test_contact_lines_are_recognized(self):
        from phoena_translator.pdf.translate.targets import (
            _looks_like_pdf_contact_line,
        )

        for line in self.CONTACT_LINES:
            assert _looks_like_pdf_contact_line(line), line

    def test_prose_with_numbers_is_not(self):
        from phoena_translator.pdf.translate.targets import (
            _looks_like_pdf_contact_line,
        )

        for text in self.MUST_STILL_TRANSLATE:
            assert not _looks_like_pdf_contact_line(text), text

    def test_contact_echo_is_accepted_by_the_verdict(self):
        from phoena_translator.pdf.translate.validation import (
            _short_translation_needs_retry,
        )

        for line in self.CONTACT_LINES:
            assert not _short_translation_needs_retry(line, line), line

    def test_full_date_labels_translate_deterministically(self):
        from phoena_translator.pdf.translate.deterministic_labels import (
            _pdf_full_date_label_translation,
            _translate_pdf_deterministic_labels,
        )

        assert _pdf_full_date_label_translation("05 August 2026") == "2026年8月5日"
        assert _pdf_full_date_label_translation("24 June 2026") == "2026年6月24日"
        assert _pdf_full_date_label_translation("June 24, 2026") == "2026年6月24日"
        assert _pdf_full_date_label_translation("12 June 2026") == "2026年6月12日"
        assert _translate_pdf_deterministic_labels("05 August 2026") == "2026年8月5日"

    def test_partial_or_prose_dates_stay_with_the_model(self):
        from phoena_translator.pdf.translate.deterministic_labels import (
            _pdf_full_date_label_translation,
        )

        assert _pdf_full_date_label_translation("May 2026") is None
        assert _pdf_full_date_label_translation("Released on 05 August 2026") is None
        assert _pdf_full_date_label_translation("48 Item 2026") is None
        assert _pdf_full_date_label_translation("32 March 2026") is None

    def test_admission_gate_skips_bylines_and_dates(self):
        from phoena_translator.pdf.translate.targets import (
            _pdf_element_requires_translation,
        )

        byline = _table_test_elem(
            "scattered",
            content=self.CONTACT_LINES[0],
        )
        assert not _pdf_element_requires_translation(byline)
        cover_contact = _table_test_elem(
            "table",
            content=self.CONTACT_LINES[1],
            table_hint=True,
        )
        assert not _pdf_element_requires_translation(cover_contact)
        date_header = _table_test_elem(
            "scattered",
            content="05 August 2026",
        )
        assert not _pdf_element_requires_translation(date_header)
        prose = _table_test_elem(
            "scattered",
            content="Figure 8: Difference in the beta of the average yields",
        )
        assert _pdf_element_requires_translation(prose)
