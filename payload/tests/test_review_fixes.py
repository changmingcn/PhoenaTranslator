"""Regression tests for the 2026-07-22 code-review fixes.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

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
