"""Regression tests for the 2026-07-22 code-review fixes.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import copy
import logging
import threading

import pytest

from phoena_translator.api_runtime import is_rate_limit_error, strip_think_tags
from phoena_translator.config import AppConfig, get_app_config
from phoena_translator.epub.pipeline import (
    CHUNK_FAILURE_API,
    CHUNK_FAILURE_INTEGRITY,
    EPUBPipeline,
    EPUBPipelineDependencies,
    translate_single_chunk,
)
from phoena_translator.epub.xhtml import is_appendix_xhtml
from phoena_translator.glossary import build_glossary_text
from phoena_translator.llm import DeepSeekTranslationAdapter, LLMSettings
from phoena_translator.logging_setup import configure_logging
from phoena_translator.workers import TaskJob, TaskWorkerPool

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# B1/B2 — EPUB skip-section detection must not fire on substrings
# ---------------------------------------------------------------------------


class TestAppendixDetection:
    def test_calibre_split_content_is_translated(self):
        assert not is_appendix_xhtml(
            "OEBPS/index_split_003.xhtml", "<p>Chapter body text of the book.</p>"
        )

    def test_bare_index_filename_alone_is_translated(self):
        assert not is_appendix_xhtml(
            "OEBPS/index.xhtml", "<p>Main landing content.</p>"
        )

    def test_heading_mentioning_keyword_is_translated(self):
        assert not is_appendix_xhtml(
            "OEBPS/ch06.xhtml", "<h2>Cross-references in law</h2><p>x</p>"
        )
        assert not is_appendix_xhtml(
            "OEBPS/ch07.xhtml", "<h1>European index futures</h1><p>x</p>"
        )

    def test_real_bibliography_filename_is_skipped(self):
        assert is_appendix_xhtml("OEBPS/bibliography.xhtml", "<p>[1] Smith</p>")
        assert is_appendix_xhtml("OEBPS/05_references.xhtml", "<p>[1] Smith</p>")

    def test_real_index_heading_is_skipped(self):
        assert is_appendix_xhtml(
            "OEBPS/backmatter2.xhtml", "<head><title>Index</title></head><p>A, 1</p>"
        )
        assert is_appendix_xhtml(
            "OEBPS/backmatter3.xhtml", "<h1>References and Further Reading</h1>"
        )


# ---------------------------------------------------------------------------
# B3 — a page-level get_text failure must propagate (recovery owns it)
# ---------------------------------------------------------------------------


class _BrokenPage:
    number = 0

    def get_text(self, *args, **kwargs):
        raise RuntimeError("mupdf page decode failed")


def test_pdf_text_extraction_failure_propagates():
    from phoena_translator.pdf.targets import _extract_pdf_text_elements

    with pytest.raises(RuntimeError, match="mupdf page decode failed"):
        _extract_pdf_text_elements(_BrokenPage(), [])


# ---------------------------------------------------------------------------
# B4 — the OpenAI SDK must not retry on its own
# ---------------------------------------------------------------------------


def test_openai_client_created_without_sdk_retries():
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    adapter = DeepSeekTranslationAdapter(
        LLMSettings(api_key="k", base_url="http://localhost", model="m"),
        client_factory=factory,
    )
    adapter._client()
    assert captured["max_retries"] == 0


# ---------------------------------------------------------------------------
# B5 — EPUB integrity failures are retryable once (strict), then fail closed
# ---------------------------------------------------------------------------

_SOURCE_CHUNK = (
    "<html><body><p>This is a long enough paragraph of English prose that "
    "clearly needs translation to Chinese.</p></body></html>"
)


def test_integrity_failure_reported_as_retryable():
    def fake_translate(text, system_prompt=None, **kwargs):
        return "标签全部丢失的输出"

    rel, idx, result, kind = translate_single_chunk(
        "t", "f.xhtml", 0, 1, _SOURCE_CHUNK, "PROMPT",
        translate_text=fake_translate, logger=log,
    )
    assert result is None
    assert kind == CHUNK_FAILURE_INTEGRITY


def test_strict_retry_appends_marker_hint():
    prompts: list[str] = []

    def fake_translate(text, system_prompt=None, **kwargs):
        prompts.append(system_prompt or "")
        return "标签全部丢失的输出"

    translate_single_chunk(
        "t", "f.xhtml", 0, 1, _SOURCE_CHUNK, "PROMPT",
        translate_text=fake_translate, logger=log, strict_retry=True,
    )
    assert "占位符" in prompts[-1] and prompts[-1].startswith("PROMPT")


def test_api_exception_reported_as_api_failure():
    def fake_translate(text, system_prompt=None, **kwargs):
        raise RuntimeError("provider down")

    _rel, _idx, result, kind = translate_single_chunk(
        "t", "f.xhtml", 0, 1, _SOURCE_CHUNK, "PROMPT",
        translate_text=fake_translate, logger=log,
    )
    assert result is None
    assert kind == CHUNK_FAILURE_API


def test_pipeline_fails_closed_after_one_strict_retry(tmp_path):
    calls: list[str] = []

    def fake_translate(text, system_prompt=None, **kwargs):
        calls.append(system_prompt or "")
        return "标签全部丢失的输出"

    book_root = tmp_path / "book"
    book_root.mkdir()
    chapter = book_root / "chapter1.xhtml"
    chapter.write_text(_SOURCE_CHUNK, encoding="utf-8")

    dependencies = EPUBPipelineDependencies(
        translate_text=fake_translate,
        build_glossary=lambda text, task_id="": "",
        make_prompt=lambda prompt, glossary: prompt,
        save_progress=lambda task_id, data: None,
        load_task=lambda task_id: {},
        logger=log,
        system_prompt_xhtml="SYS",
        workers=1,
        api_concurrency=1,
        max_chunk_rounds=4,
        sleep=lambda seconds: None,
    )
    total_files, total_chunks, completed = EPUBPipeline(
        dependencies
    )._translate_extracted("task", book_root)

    assert total_files == 1 and total_chunks == 1
    assert "chapter1.xhtml" in completed
    # One normal attempt plus exactly one strict retry, then fail closed.
    assert len(calls) == 2
    assert "占位符" in calls[1] and "占位符" not in calls[0]
    assert chapter.read_text(encoding="utf-8") == _SOURCE_CHUNK


def test_pipeline_keeps_non_utf8_member_as_source(tmp_path):
    book_root = tmp_path / "book"
    book_root.mkdir()
    good = book_root / "a.xhtml"
    good.write_text(_SOURCE_CHUNK, encoding="utf-8")
    bad = book_root / "b.xhtml"
    bad.write_bytes("<p>UTF-16 content that is long enough here</p>".encode("utf-16"))

    def fake_translate(text, system_prompt=None, **kwargs):
        return text  # echo keeps markup identical -> passes integrity

    dependencies = EPUBPipelineDependencies(
        translate_text=fake_translate,
        build_glossary=lambda text, task_id="": "",
        make_prompt=lambda prompt, glossary: prompt,
        save_progress=lambda task_id, data: None,
        load_task=lambda task_id: {},
        logger=log,
        system_prompt_xhtml="SYS",
        workers=1,
        api_concurrency=1,
        sleep=lambda seconds: None,
    )
    total_files, _chunks, completed = EPUBPipeline(dependencies)._translate_extracted(
        "task", book_root
    )
    assert total_files == 2
    assert completed == {"a.xhtml", "b.xhtml"}
    assert bad.read_bytes().startswith(b"\xff\xfe")  # untouched source bytes


# ---------------------------------------------------------------------------
# B6 — one process-wide configuration snapshot
# ---------------------------------------------------------------------------


def test_get_app_config_is_memoized():
    assert get_app_config() is get_app_config()


def test_pdf_constants_share_the_process_config():
    import phoena_translator.pdf.types as pdf_types

    config = get_app_config()
    assert (
        pdf_types.PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE
        == config.pdf_min_acceptable_htmlbox_scale
    )
    assert pdf_types.PDF_VECTOR_OCR_DPI == config.pdf_vector_ocr_dpi
    assert (
        pdf_types.PDF_VECTOR_OCR_MIN_ALPHA_WORDS
        == config.pdf_vector_ocr_min_alpha_words
    )


def test_font_override_parsed_from_env(tmp_path):
    config = AppConfig.from_env(
        environ={"TRANSLATOR_PDF_FONT_REGULAR": str(tmp_path / "r.otf")},
        home=tmp_path,
    )
    assert config.pdf_font_regular == tmp_path / "r.otf"
    assert config.pdf_font_bold is None


def test_font_override_resolution(tmp_path):
    from phoena_translator.pdf.fonts import _get_chinese_font_path

    font_file = tmp_path / "font.otf"
    font_file.write_bytes(b"stub")
    assert _get_chinese_font_path(override=str(font_file)) == str(font_file)
    with pytest.raises(RuntimeError):
        _get_chinese_font_path(override=str(tmp_path / "missing.otf"))


# ---------------------------------------------------------------------------
# B7 — a crashed job must reach a terminal state
# ---------------------------------------------------------------------------


def test_crashed_job_marks_task_failed():
    failures: list[tuple[str, str]] = []
    recorded = threading.Event()

    def handler(job, worker_id):
        raise RuntimeError("boom")

    def mark_failed(task_id, error):
        failures.append((task_id, error))
        recorded.set()

    pool = TaskWorkerPool(
        max_workers=1,
        max_queue=2,
        max_pending_bytes=10**6,
        handler=handler,
        logger=log,
        mark_failed=mark_failed,
    )
    assert pool.enqueue(
        TaskJob(task_id="tid1", extension="pdf", source_path="s", output_path="o"),
        size_bytes=1,
    )
    assert recorded.wait(timeout=5)
    assert failures[0][0] == "tid1"
    assert "boom" in failures[0][1]


# ---------------------------------------------------------------------------
# B8 — the service log rotates
# ---------------------------------------------------------------------------


def test_log_file_handler_rotates(tmp_path):
    from logging.handlers import RotatingFileHandler

    logger = configure_logging(
        tmp_path / "app.log", logger_name="translator-rotation-test"
    )
    rotating = [
        handler
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert rotating and rotating[0].maxBytes > 0 and rotating[0].backupCount > 0


# ---------------------------------------------------------------------------
# M1 — no silent truncation of oversized candidates
# ---------------------------------------------------------------------------


def test_long_candidate_not_truncated():
    from phoena_translator.pdf.page_translation import _clean_translated_text

    source = "short source"
    candidate = "很长的候选译文段落" * 120
    cleaned, failures = _clean_translated_text(source, candidate, (), (), ())
    assert failures == ()
    assert len(cleaned) >= len(candidate) - 2


# ---------------------------------------------------------------------------
# M4 / M6 — response hygiene and rate-limit classification
# ---------------------------------------------------------------------------


def test_strip_think_tags_handles_unclosed_tag():
    assert strip_think_tags("<think>reasoning</think>你好") == "你好"
    assert strip_think_tags("译文<think>truncated reasoning tail") == "译文"
    assert strip_think_tags("plain 译文") == "plain 译文"


def test_rate_limit_detection_boundaries():
    assert is_rate_limit_error(RuntimeError("Error code: 429 - slow down"))
    assert is_rate_limit_error(RuntimeError("Too Many Requests"))
    assert is_rate_limit_error(RuntimeError("Rate limit reached"))

    class _Coded(Exception):
        status_code = 429

    assert is_rate_limit_error(_Coded("x"))
    assert not is_rate_limit_error(RuntimeError("request id 14290 reset"))
    assert not is_rate_limit_error(RuntimeError("4290 items processed"))
    assert not is_rate_limit_error(RuntimeError("connection refused"))


# ---------------------------------------------------------------------------
# M5 — glossary selects whole terms only
# ---------------------------------------------------------------------------


def test_glossary_whole_word_only():
    inside_only = "The Article discusses many things."
    assert build_glossary_text(inside_only, {"Art": "艺术"}) == ""

    mixed = "Art is long. The Article is different. Art endures."
    assert build_glossary_text(mixed, {"Art": "艺术"}) == "Art = 艺术"


# ---------------------------------------------------------------------------
# B9 — CFF-flavored fonts must not be subset (MuPDF mis-renders the result)
# ---------------------------------------------------------------------------


def test_cff_font_subsetting_is_skipped(tmp_path):
    from phoena_translator.pdf.fonts import _pdf_font_is_cff, _subset_pdf_font

    stub = tmp_path / "font.otf"
    stub.write_bytes(b"OTTO" + b"\x00" * 12)
    assert _pdf_font_is_cff(str(stub))
    assert _subset_pdf_font(str(stub), "测试中文", str(tmp_path), "x") == str(stub)

    truetype_stub = tmp_path / "font.ttf"
    truetype_stub.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 12)
    assert not _pdf_font_is_cff(str(truetype_stub))


# ---------------------------------------------------------------------------
# B10 — unsubsettable large fonts must be rejected before any provider work
# ---------------------------------------------------------------------------


def test_font_embeddability_policy(tmp_path):
    from phoena_translator.pdf.fonts import pdf_font_is_safely_embeddable

    big_cff = tmp_path / "big.otf"
    big_cff.write_bytes(b"OTTO" + b"\x00" * (5 * 1024 * 1024))
    assert not pdf_font_is_safely_embeddable(str(big_cff))

    small_cff = tmp_path / "small.otf"
    small_cff.write_bytes(b"OTTO" + b"\x00" * 128)
    assert pdf_font_is_safely_embeddable(str(small_cff))

    big_ttf = tmp_path / "big.ttf"
    big_ttf.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * (5 * 1024 * 1024))
    assert pdf_font_is_safely_embeddable(str(big_ttf))


def test_resolve_output_fonts_rejects_unsubsettable_large_font(tmp_path):
    from phoena_translator.pdf.output_stage import resolve_output_fonts

    big_cff = tmp_path / "big.otf"
    big_cff.write_bytes(b"OTTO" + b"\x00" * (5 * 1024 * 1024))
    with pytest.raises(RuntimeError, match="TrueType"):
        resolve_output_fonts(str(big_cff), str(big_cff))


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


class TestCrossPageOrphanTail:
    def test_fragment_detection(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _cross_page_orphan_tail_fragment,
        )

        _pages, _source, orphan, body = _orphan_tail_fixture()
        assert _cross_page_orphan_tail_fragment(orphan) == "traders."
        assert _cross_page_orphan_tail_fragment(body) is None  # uppercase start
        unterminated = dict(orphan, content="high frequency")
        assert _cross_page_orphan_tail_fragment(unterminated) is None
        table_cell = dict(orphan, table_hint=True)
        assert _cross_page_orphan_tail_fragment(table_cell) is None

    def test_absorption_moves_tail_and_merges_orphan_away(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _absorb_cross_page_orphan_tails,
        )

        pages, source, orphan, _body = _orphan_tail_fixture()
        audit_log: list = []
        absorbed = _absorb_cross_page_orphan_tails(
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
        from phoena_translator.pdf.semantic_cross_page import (
            _absorb_cross_page_orphan_tails,
        )

        pages, source, orphan, _body = _orphan_tail_fixture()
        source["content"] = source["content"] + " traders."
        source["paragraphs"][0]["plain"] += " traders."
        absorbed = _absorb_cross_page_orphan_tails(pages, 2, _page_rects_letter())
        assert absorbed == 0
        assert orphan["type"] == "text"

    def test_audit_validator_accepts_and_rejects(self):
        from phoena_translator.pdf.audit import _validate_pdf_merge_audit
        from phoena_translator.pdf.semantic_cross_page import (
            _absorb_cross_page_orphan_tails,
        )

        pages, source, _orphan, _body = _orphan_tail_fixture()
        audit_log: list = []
        _absorb_cross_page_orphan_tails(
            pages, 2, _page_rects_letter(), audit_log=audit_log
        )
        assert _validate_pdf_merge_audit(audit_log, pages) == []

        # Tamper: strip the tail from the source again -> single-ownership
        # invariant must fire.
        source["content"] = source["content"][: -len(" traders.")]
        violations = _validate_pdf_merge_audit(audit_log, pages)
        assert violations and violations[0]["invariant"] == "orphan-tail-not-absorbed"


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


class TestFormulaMidSentencePreservation:
    def test_mid_sentence_stub_above_formula_is_preserved(self):
        from phoena_translator.pdf.audit import (
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
        from phoena_translator.pdf.audit import (
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
        from phoena_translator.pdf.audit import (
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
        from phoena_translator.pdf.audit import (
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

    def test_leading_tail_split_from_first_paragraph(self):
        from phoena_translator.pdf.audit import _validate_pdf_merge_audit
        from phoena_translator.pdf.semantic_cross_page import (
            _merge_cross_page_sentences,
        )

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
        from phoena_translator.pdf.semantic_cross_page import (
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


# ---------------------------------------------------------------------------
# Production crash 2026-07-23: Rect(None) raises AssertionError in PyMuPDF
# ---------------------------------------------------------------------------


class TestSourceInkRectRobustness:
    def test_paragraph_without_any_geometry_falls_back_to_elem_rect(self):
        import fitz

        from phoena_translator.pdf.geometry import _get_pdf_source_ink_rects

        elem = {
            "type": "text",
            "bbox": [10.0, 20.0, 200.0, 40.0],
            # The exact shape that crashed assembly on the SEC document:
            # a paragraph carrying no source_lines, no source_line_bboxes
            # and no source_bbox.
            "paragraphs": [{"plain": "cell text", "text_align": "left"}],
        }
        rects = _get_pdf_source_ink_rects(elem)
        assert rects == [fitz.Rect(10.0, 20.0, 200.0, 40.0)]

    def test_malformed_bbox_values_are_skipped(self):
        import fitz

        from phoena_translator.pdf.geometry import (
            _get_pdf_source_ink_rects,
            _safe_bbox_rect,
        )

        assert _safe_bbox_rect(None) is None
        assert _safe_bbox_rect("garbage") is None
        assert _safe_bbox_rect([1, 2]) is None
        assert _safe_bbox_rect([0, 0, 0, 0]) is None  # empty rect
        assert _safe_bbox_rect([1, 2, 3, 4]) == fitz.Rect(1, 2, 3, 4)

        elem = {
            "type": "text",
            "bbox": [0.0, 0.0, 100.0, 10.0],
            "paragraphs": [
                {"plain": "x", "source_bbox": None},
                {"plain": "y", "source_line_bboxes": [None, "bad", [5, 5, 50, 15]]},
            ],
        }
        rects = _get_pdf_source_ink_rects(elem)
        assert rects == [fitz.Rect(5, 5, 50, 15)]


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
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_text import _mark_watermark_elements

        page_extractions, page_rects = self._pages(3)
        _mark_watermark_elements(page_extractions, page_rects, 3)
        for info in page_extractions.values():
            assert "skip_translate_reason" not in info["elements"][0]

    def test_rotated_stamp_is_still_a_watermark(self):
        from phoena_translator.pdf.semantic_text import _mark_watermark_elements

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
        from phoena_translator.pdf.semantic_text import _mark_watermark_elements

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


class TestRefusalMetaGate:
    SOURCE = "Discussion of policy implications"

    def test_chinese_refusal_needs_retry(self):
        from phoena_translator.pdf.translation import _short_translation_needs_retry

        assert _short_translation_needs_retry(
            self.SOURCE, "抱歉，我无法翻译此内容"
        )
        assert _short_translation_needs_retry(
            self.SOURCE, "我无法翻译此内容"
        )

    def test_meta_preamble_needs_retry(self):
        from phoena_translator.pdf.translation import _short_translation_needs_retry

        assert _short_translation_needs_retry(
            self.SOURCE, "好的，以下是译文：政策含义的讨论"
        )
        assert _short_translation_needs_retry(
            self.SOURCE, "译文：政策含义的讨论"
        )

    def test_ordinary_translation_passes(self):
        from phoena_translator.pdf.translation import _short_translation_needs_retry

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
        from phoena_translator.pdf.translation_stage import (
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
        from phoena_translator.pdf.translation_stage import (
            _require_translated_page_progress,
        )
        from phoena_translator.pdf.types import PDFPageTranslationError

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
        from phoena_translator.pdf.translation_stage import (
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
        from phoena_translator.pdf.translation_stage import (
            _require_translated_page_progress,
        )

        context = self._context(
            {"source_page_fallbacks": [{"page": 1, "stage": "structure"}]},
            {0: {"source_page_fallback": {}}},
        )
        _require_translated_page_progress(context)


class TestBuildParagraphsEmptyInput:
    def test_returns_declared_tuple_shape(self):
        import fitz

        from phoena_translator.pdf.semantic_text import _build_pdf_paragraphs

        assert _build_pdf_paragraphs([], fitz.Rect(0, 0, 10, 10), 10.0) == ([], 0.0)


# ---------------------------------------------------------------------------
# 2026-07-24 two-signal OR policy for cross-page sentence absorption
# ---------------------------------------------------------------------------


def _two_page_fixture(
    source_content: str,
    dest_content: str,
    *,
    dest_rich: str | None = None,
    dest_bbox: list | None = None,
):
    import fitz

    last_elem = {
        "type": "text",
        "layout_class": "body",
        "content": source_content,
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    first_elem = {
        "type": "text",
        "layout_class": "scattered",
        "content": dest_content,
        "bbox": dest_bbox or [72.0, 104.0, 518.0, 131.0],
        "y": 104.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    if dest_rich is not None:
        first_elem["rich_content"] = dest_rich
    page_extractions = {
        0: {"elements": [last_elem]},
        1: {"elements": [first_elem]},
    }
    page_rects = {0: fitz.Rect(0, 0, 612, 792), 1: fitz.Rect(0, 0, 612, 792)}
    return page_extractions, page_rects, last_elem, first_elem


class TestTailStartPolicy:
    def test_shapes(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _cross_page_tail_start_acceptable,
        )

        # Lowercase: acceptable regardless of the source signal.
        assert _cross_page_tail_start_acceptable(
            "as liquidity rushed in.", source_dangling=False
        )
        # Digit + lowercase letter: only with a dangling source.
        assert _cross_page_tail_start_acceptable(
            "1-3, respectively.", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
            "1-3, respectively.", source_dangling=False
        )
        # Numbered heading / list item: rejected either way.
        assert not _cross_page_tail_start_acceptable(
            "4.2 Results", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
            "1. Introduction", source_dangling=True
        )
        # Uppercase proper-noun continuation: dangling source only.
        assert _cross_page_tail_start_acceptable(
            "High Frequency Traders lifted offers.", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
            "High Frequency Traders lifted offers.", source_dangling=False
        )
        # Caption/heading leads and roman numerals: rejected.
        assert not _cross_page_tail_start_acceptable(
            "Table VII presents regression results.", source_dangling=True
        )
        assert not _cross_page_tail_start_acceptable(
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
        from phoena_translator.pdf.semantic_cross_page import (
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

    def test_digit_tail_requires_dangling_source(self):
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, _last, first_elem = _two_page_fixture(
            "In the last minute of the down phase, most contracts were sold by",
            "Table VII presents the regression results of both components.",
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["content"].startswith("Table VII")


class TestCrossPageLowercaseWithApparentBoundary:
    def test_lowercase_start_overrides_source_boundary(self):
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
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

    def test_policy_requires_capitalized_source_word(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _cross_page_tail_start_acceptable,
        )

        assert _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True, source_ends_capitalized=True
        )
        assert not _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True, source_ends_capitalized=False
        )
        # Caption leads stay rejected even with a capitalized source word.
        assert not _cross_page_tail_start_acceptable(
            "Table VII.", source_dangling=True, source_ends_capitalized=True
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
        from phoena_translator.pdf.semantic_cross_page import (
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
        from phoena_translator.pdf.semantic_cross_page import (
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

    def test_terminated_source_blocks_absorption(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _merge_cross_page_sentences,
        )

        page_extractions, page_rects, _last, first_elem = self._pages(
            self.SOURCE.rstrip() + " sales.",
            self.FRAGMENT,
            (
                "As the table shows, the process by which the market absorbs "
                "such imbalances is a confluence of different responses."
            ),
        )
        absorbed = _merge_cross_page_sentences(page_extractions, 2, page_rects, [])
        assert absorbed == 0
        assert first_elem["type"] == "text"


def _table_test_elem(layout_class, **overrides):
    base = {"type": "text", "layout_class": layout_class, "content": "cell 12.3"}
    base.update(overrides)
    return base


class TestTablePreservation:
    """2026-07-24 policy: tables keep their original English ink verbatim."""

    def test_table_dominated_page_is_preserved(self):
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.targets import (
            _pdf_element_requires_translation,
        )

        marked = _table_test_elem(
            "table",
            content="Trading Volume 893,262",
            skip_translate_reason="table_preserved",
        )
        assert not _pdf_element_requires_translation(marked)

    def test_config_flag_wiring(self):
        from pathlib import Path

        cfg = AppConfig.from_env({}, home=Path("/tmp"))
        assert cfg.pdf_preserve_tables is True
        cfg_off = AppConfig.from_env(
            {"TRANSLATOR_PDF_PRESERVE_TABLES": "false"},
            home=Path("/tmp"),
        )
        assert cfg_off.pdf_preserve_tables is False


class TestOrphanTailAuditPolicy:
    def test_shape_check_accepts_digit_tail_without_evidence(self):
        from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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

    def test_shape_check_still_rejects_headings(self):
        from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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
        assert (
            _validate_pdf_orphan_tail_decision(decision, None)
            == "orphan-tail-malformed"
        )


# ---------------------------------------------------------------------------
# 2026-07-25 strict-audit release blockers C001 / C002 / C004
# ---------------------------------------------------------------------------


def test_complete_source_lowercase_destination_is_not_absorbed():
    from phoena_translator.pdf.semantic_cross_page import (
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

    assert absorbed == 0
    assert source["content"] == source_text
    assert destination["content"] == destination_text
    assert destination["type"] == "text"
    assert audit_log == []


def test_orphan_audit_rejects_complete_source_lowercase_decision():
    from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

    source_text = "Demand remained stable."
    tail = "the next section discusses a separate result."
    remainder = "Its ownership must stay on page two."
    destination_text = f"{tail} {remainder}"
    pages, _page_rects, source, destination = _two_page_fixture(
        source_text,
        destination_text,
    )
    # The live dictionaries model a forged post-merge state, while paragraph
    # text retains immutable extraction evidence for the audit.
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
        "source_continuation_reason": "ambiguous-terminal-abbreviation",
        "carried_tail": tail,
        "source_page": 1,
        "destination_page": 2,
        "source_element": 0,
        "destination_element": 0,
    }

    assert (
        _validate_pdf_orphan_tail_decision(decision, pages)
        == "orphan-tail-source-evidence"
    )


def _policy_test_elements():
    elements = [
        {
            "type": "text",
            "layout_class": "table",
            "content": (
                f"Contract category {index + 1} reports daily trading "
                "volume of 123 units."
            ),
            "bbox": [100.0, 180.0 + index * 18.0, 480.0, 194.0 + index * 18.0],
            "x": 100.0,
            "y": 180.0 + index * 18.0,
            "fontsize": 10.0,
        }
        for index in range(5)
    ]
    elements.append(
        {
            "type": "text",
            "layout_class": "body",
            "content": "Ordinary prose sentence that needs translation.",
            "bbox": [90.0, 320.0, 500.0, 350.0],
            "x": 90.0,
            "y": 320.0,
            "fontsize": 11.0,
        }
    )
    return elements


def _policy_context(tmp_path, preserve_tables):
    from phoena_translator.pdf.context import ImmutableElementPolicy
    from phoena_translator.pdf.extraction_stage import PDFExtractionStageContext

    kwargs = {
        "task_id": "policy-test",
        "src_path": str(tmp_path / "source.pdf"),
        "pdf_password": "",
        "progress_dir": str(tmp_path),
        "extraction_concurrency": 1,
        "fail_open_to_source_page": False,
        "element_policy": ImmutableElementPolicy(
            preserve_tables=preserve_tables
        ),
        "extraction_semaphore": threading.Semaphore(1),
        "logger": log,
        "pdf_audit": {
            "merge_decisions": [],
            "source_page_fallbacks": [],
        },
        "load_progress": lambda _task_id: None,
        "save_translation_progress": lambda *args, **kwargs: None,
        "extract_page_elements": lambda *args, **kwargs: [],
        "summarize_formula_protection": (
            lambda *args, **kwargs: {"translation_queue_check": "ok"}
        ),
    }
    return PDFExtractionStageContext(**kwargs)


def _policy_result(elements, source_sha256="a" * 64, *, cached=False):
    import fitz

    from phoena_translator.pdf.extraction_stage import PDFExtractionResult

    completed = {0} if cached else set()
    return PDFExtractionResult(
        total_pages=1,
        source_sha256=source_sha256,
        completed_indices=set(completed),
        cached_indices=set(completed),
        page_extractions={0: {"elements": elements}},
        page_rects={0: fitz.Rect(0, 0, 612, 792)},
        forced_fallbacks_by_index={},
    )


def _without_layout_mutation(stage):
    replacements = {
        "_mark_watermark_elements": lambda *args, **kwargs: None,
        "_classify_pdf_page_text_elements": lambda *args, **kwargs: None,
        "_merge_cross_page_sentences": lambda *args, **kwargs: 0,
        "_preserve_pdf_formula_mid_sentence_neighbors": (
            lambda *args, **kwargs: []
        ),
    }
    originals = {name: getattr(stage, name) for name in replacements}
    for name, replacement in replacements.items():
        setattr(stage, name, replacement)
    return originals


def _restore_stage_functions(stage, originals):
    for name, original in originals.items():
        setattr(stage, name, original)


def test_warm_cache_policy_transition_reconciles_before_completion(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf import extraction_stage as stage

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "a" * 64
    off_elements = _policy_test_elements()
    translations = {
        **{
            str(index): (
                f"这是合约类别 {index + 1} 每日交易量为 123 单位的完整中文译文。"
            )
            for index in range(5)
        },
        "5": "这是需要翻译的普通正文句子的完整中文译文。",
    }
    cache._save_pdf_page_translation_cache(
        "policy-test",
        0,
        translations,
        source_sha256,
        elements=off_elements,
    )
    result = _policy_result(
        copy.deepcopy(off_elements),
        source_sha256,
        cached=True,
    )
    context = _policy_context(tmp_path, True)
    originals = _without_layout_mutation(stage)
    try:
        stage.reconcile_extraction(context, result)
        # Retain the historical call so this regression follows both the
        # pre-fix and post-fix production surfaces.
        stage.enforce_table_preservation(context, result)
    finally:
        _restore_stage_functions(stage, originals)

    elements = result.page_extractions[0]["elements"]
    cached = result.page_extractions[0]["cached"]
    assert all(
        element.get("skip_translate_reason") == "table_preserved"
        for element in elements[:5]
    )
    assert all(cached[str(index)] == elements[index]["content"] for index in range(5))
    assert cached["5"] == "这是需要翻译的普通正文句子的完整中文译文。"
    assert result.completed_indices == {0}


def test_table_cache_policy_transition_matrix(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf.extraction_stage import (
        _mark_pdf_table_preserved_elements,
    )

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "b" * 64

    def elements_for(policy):
        elements = _policy_test_elements()
        if policy:
            assert len(_mark_pdf_table_preserved_elements({0: {"elements": elements}})) == 5
        return elements

    def save(policy, task_id):
        elements = elements_for(policy)
        translations = {
            str(index): (
                element["content"]
                if element.get("skip_translate_reason")
                else (
                    "这是需要翻译的普通正文句子的完整中文译文。"
                    if index == 5
                    else (
                        f"这是合约类别 {index + 1} 每日交易量为 123 "
                        "单位的完整中文译文。"
                    )
                )
            )
            for index, element in enumerate(elements)
        }
        path = cache._save_pdf_page_translation_cache(
            task_id,
            0,
            translations,
            source_sha256,
            elements=elements,
        )
        return path, translations

    off_path, off_translations = save(False, "cache-off")
    on_path, on_translations = save(True, "cache-on")

    off_off = cache._load_pdf_page_translation_cache(
        off_path, source_sha256, elements_for(False)
    )
    off_on_elements = elements_for(True)
    off_on = cache._load_pdf_page_translation_cache(
        off_path, source_sha256, off_on_elements
    )
    on_on_first = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(True)
    )
    on_on_second = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(True)
    )
    on_off = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(False)
    )

    assert off_off == off_translations
    assert all(
        off_on[str(index)] == off_on_elements[index]["content"]
        for index in range(5)
    )
    assert off_on["5"] == "这是需要翻译的普通正文句子的完整中文译文。"
    assert on_on_first == on_translations
    assert on_on_second == on_translations
    assert on_off == {"5": "这是需要翻译的普通正文句子的完整中文译文。"}
    assert (
        cache._load_pdf_page_translation_cache(
            off_path, "c" * 64, elements_for(False)
        )
        is None
    )


def test_two_contexts_hold_independent_table_policies(tmp_path):
    from phoena_translator.pdf.extraction_stage import (
        enforce_table_preservation,
    )

    true_result = _policy_result(_policy_test_elements())
    false_result = _policy_result(_policy_test_elements())
    true_context = _policy_context(tmp_path / "true", True)
    false_context = _policy_context(tmp_path / "false", False)

    enforce_table_preservation(true_context, true_result)
    enforce_table_preservation(false_context, false_result)

    assert len(true_context.pdf_audit["table_preserved_elements"]) == 5
    assert all(
        element.get("skip_translate_reason") == "table_preserved"
        for element in true_result.page_extractions[0]["elements"][:5]
    )
    assert false_context.pdf_audit["table_preserved_elements"] == []
    assert all(
        "skip_translate_reason" not in element
        for element in false_result.page_extractions[0]["elements"]
    )


def test_table_formula_overlap_preserves_existing_precedence(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf import extraction_stage as stage

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    elements = _policy_test_elements()
    elements[0]["content"] = (
        "Regression alpha α plus beta β equals gamma γ under limit δ."
    )
    result = _policy_result(elements)
    context = _policy_context(tmp_path, True)
    originals = _without_layout_mutation(stage)
    original_bind = stage._bind_and_reconcile_caches
    stage._bind_and_reconcile_caches = lambda *args, **kwargs: None
    try:
        stage.reconcile_extraction(context, result)
        stage.enforce_formula_protection(context, result)
        stage.enforce_table_preservation(context, result)
    finally:
        stage._bind_and_reconcile_caches = original_bind
        _restore_stage_functions(stage, originals)

    target = result.page_extractions[0]["elements"][0]
    assert target["skip_translate_reason"] == "formula_risk_preserved"
    assert context.pdf_audit["formula_risk_preserved_elements"]
    assert all(
        entry["element"] != 0
        for entry in context.pdf_audit["table_preserved_elements"]
    )


# ---------------------------------------------------------------------------
# 2026-07-25 Advanced-Core: typed contracts and edge ownership
# ---------------------------------------------------------------------------


def _advanced_native_cache_element():
    return {
        "type": "text",
        "content": "Ordinary prose sentence for a stable cache identity.",
        "rich_content": "",
        "paragraphs": [
            {
                "plain": "Ordinary prose sentence for a stable cache identity.",
                "rich": "",
                "text_align": "left",
                "nowrap": False,
                "margin_left": 0.0,
                "text_indent": 0.0,
                "gap_before": 0.0,
                "first_line_indent": False,
            }
        ],
        "bbox": [72.0, 100.0, 510.0, 132.0],
        "layout_class": "body",
        "fontsize": 11.0,
        "bold": False,
        "color": 0,
        "non_horizontal": False,
        "preserve_source_style": False,
    }


def test_cache_identity_golden_is_unchanged_before_typed_hardening():
    from phoena_translator.pdf.cache import _pdf_cache_element_identity

    elem = _advanced_native_cache_element()
    assert (
        _pdf_cache_element_identity(elem)
        == "bae0777a3fafe1f4df4b694efc7a73fe5921c475cf9fb228963689246bf423ea"
    )
    assert (
        _pdf_cache_element_identity(elem, _include_empty_vector_fields=True)
        == "2346d344b4cdf203267bfba8ea611261231238a6db3c867b6db185540a76562a"
    )


def test_cache_identity_typed_core_preserves_legacy_digest():
    from dataclasses import FrozenInstanceError

    from phoena_translator.pdf.cache import (
        CacheIdentity,
        _pdf_cache_element_identity,
        _pdf_cache_identity,
    )

    elem = _advanced_native_cache_element()
    identity = _pdf_cache_identity(elem)
    expected = "bae0777a3fafe1f4df4b694efc7a73fe5921c475cf9fb228963689246bf423ea"

    assert isinstance(identity, CacheIdentity)
    assert identity.digest == expected
    assert _pdf_cache_element_identity(elem) == expected
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        CacheIdentity("not-a-cache-digest")
    with pytest.raises(FrozenInstanceError):
        identity.digest = "0" * 64


def test_continuation_decision_is_frozen_and_string_compatible():
    from dataclasses import FrozenInstanceError

    from phoena_translator.pdf.semantic_cross_page import (
        ContinuationDecision,
        SourceContinuationEvidence,
        _cross_page_continuation_decision,
        _cross_page_source_continuation_reason,
    )

    complete = _cross_page_continuation_decision("Demand remained stable.")
    abbreviation = _cross_page_continuation_decision(
        "The findings agree with Smith et al."
    )
    dangling = _cross_page_continuation_decision(
        "The explanation continues across"
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
    assert _cross_page_source_continuation_reason(
        "The findings agree with Smith et al."
    ) == "ambiguous-terminal-abbreviation"
    with pytest.raises(TypeError, match="SourceContinuationEvidence"):
        ContinuationDecision("complete")
    with pytest.raises(FrozenInstanceError):
        complete.evidence = SourceContinuationEvidence.DANGLING


def test_producer_records_pass_evidence_backed_audit_validation():
    from phoena_translator.pdf.audit import (
        _validate_pdf_orphan_tail_decision,
    )
    from phoena_translator.pdf.semantic_cross_page import (
        _merge_cross_page_sentences,
    )

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


def test_pipeline_dependencies_freeze_one_element_policy(tmp_path):
    from dataclasses import FrozenInstanceError
    from inspect import signature

    from phoena_translator.pdf.context import (
        ImmutableElementPolicy,
        PDFPipelineDependencies,
    )

    policy = ImmutableElementPolicy(preserve_tables=True)
    assert policy.preserve_tables is True
    with pytest.raises(FrozenInstanceError):
        policy.preserve_tables = False

    dependency_field = PDFPipelineDependencies.__dataclass_fields__[
        "element_policy"
    ]
    assert dependency_field.init is False
    constructor_parameters = signature(PDFPipelineDependencies).parameters
    assert "preserve_tables" in constructor_parameters
    assert "element_policy" not in constructor_parameters

    context = _policy_context(tmp_path, True)
    assert context.element_policy == policy
    assert context.preserve_tables is True
    assert "preserve_tables" not in context.__dataclass_fields__


def test_pipeline_dependencies_preserve_legacy_bool_constructor():
    from phoena_translator.pdf import pipeline
    from phoena_translator.pdf.context import (
        ImmutableElementPolicy,
        PDFPipelineDependencies,
    )

    no_op = lambda *args, **kwargs: None
    dependencies = PDFPipelineDependencies(
        api_max_concurrency=1,
        assembly_max_concurrency=1,
        extraction_max_concurrency=1,
        fail_open_to_source_page=False,
        preserve_tables=True,
        minimum_htmlbox_scale=0.5,
        save_clean=False,
        save_garbage=0,
        use_htmlbox=True,
        progress_dir="/tmp/advanced-policy-test",
        system_prompt_text="test",
        translation_workers=1,
        assembly_semaphore=threading.Semaphore(1),
        extraction_semaphore=threading.Semaphore(1),
        logger=log,
        tasks={},
        build_glossary=no_op,
        make_prompt_with_glossary=no_op,
        load_progress=no_op,
        save_progress=no_op,
        translate_text=no_op,
        retry_translate_pdf=no_op,
        trim_process_memory=no_op,
        extract_page_elements=no_op,
        filter_formula_safe_rects=no_op,
        summarize_formula_protection=no_op,
        check_output_structure_serialized=no_op,
        save_translation_progress=no_op,
        font_subsetting_available=False,
    )

    assert dependencies.preserve_tables is True
    assert isinstance(dependencies.element_policy, ImmutableElementPolicy)
    assert dependencies.element_policy.preserve_tables is True

    class _ContextCaptured(RuntimeError):
        pass

    captured = {}
    original_context = pipeline.PDFExtractionStageContext
    original_resolve_fonts = pipeline.resolve_output_fonts

    def capture_context(**kwargs):
        captured.update(kwargs)
        raise _ContextCaptured

    pipeline.PDFExtractionStageContext = capture_context
    pipeline.resolve_output_fonts = lambda *args, **kwargs: (None, None)
    try:
        with pytest.raises(_ContextCaptured):
            pipeline._run_pdf_attempt(
                "policy-test",
                "/tmp/source.pdf",
                "/tmp/output.pdf",
                "",
                {},
                dependencies=dependencies,
            )
    finally:
        pipeline.PDFExtractionStageContext = original_context
        pipeline.resolve_output_fonts = original_resolve_fonts

    assert captured["element_policy"] is dependencies.element_policy
    assert "preserve_tables" not in captured


def test_test_dependency_is_declared_separately():
    from pathlib import Path

    app_root = Path(__file__).resolve().parents[1]
    test_requirements = app_root / "requirements-test.txt"
    assert test_requirements.read_text(encoding="utf-8") == "pytest==8.4.2\n"
    assert "pytest" not in (app_root / "requirements.txt").read_text(
        encoding="utf-8"
    ).lower()


def test_advanced_internal_types_do_not_expand_legacy_exports():
    from phoena_translator import legacy_exports
    from phoena_translator.pdf import context

    assert "ImmutableElementPolicy" not in context.__all__
    assert not hasattr(legacy_exports, "ImmutableElementPolicy")
    assert not hasattr(legacy_exports, "CacheIdentity")
    assert not hasattr(legacy_exports, "ContinuationDecision")


def test_cross_page_merge_rejects_different_destination_column():
    from phoena_translator.pdf.semantic_cross_page import (
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
    original_source = source["content"]
    original_destination = destination["content"]
    audit_log = []

    assert _merge_cross_page_sentences(pages, 2, rects, audit_log) == 0
    assert source["content"] == original_source
    assert destination["content"] == original_destination
    assert destination["type"] == "text"
    assert audit_log == []


def test_cross_page_merge_rejects_page_top_heading_before_continuation():
    import fitz

    from phoena_translator.pdf.semantic_cross_page import (
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

    assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
    assert source["content"] == "The explanation continues across"
    assert heading["type"] == "text"
    assert continuation["type"] == "text"


def test_three_page_merges_keep_two_independent_source_owners():
    import fitz

    from phoena_translator.pdf.semantic_cross_page import (
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
