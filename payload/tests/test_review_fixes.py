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


def test_prepare_fonts_subsets_distinct_regular_and_bold(
    tmp_path,
    monkeypatch,
):
    import shutil
    from types import SimpleNamespace

    from phoena_translator.pdf import output_stage

    regular = tmp_path / "regular.ttf"
    bold = tmp_path / "bold.ttf"
    regular.write_bytes(b"regular")
    bold.write_bytes(b"bold")
    calls = []

    def fake_subset(source, chars, output_dir, role):
        calls.append((source, chars, role))
        subset = tmp_path / f"{role}-subset.ttf"
        subset.write_bytes(role.encode())
        return str(subset)

    class FakeArchive:
        def __init__(self, path):
            self.paths = [path]

        def add(self, path):
            self.paths.append(path)

    monkeypatch.setattr(
        output_stage,
        "_collect_pdf_font_chars",
        lambda *_args: "ABC测试",
    )
    monkeypatch.setattr(output_stage, "_subset_pdf_font", fake_subset)
    monkeypatch.setattr(
        output_stage,
        "_subset_font_renders_like_original",
        lambda *_args: True,
    )
    monkeypatch.setattr(output_stage.fitz, "Archive", FakeArchive)
    context = SimpleNamespace(
        task_id="font-test",
        font_path=str(regular),
        bold_font_path=str(bold),
        font_subsetting_available=True,
        logger=logging.getLogger("font-test"),
    )

    subset_dir = None
    try:
        subset_regular, subset_bold, distinct, _archive, subset_dir = (
            output_stage._prepare_fonts(context, {})
        )

        assert distinct is True
        assert subset_regular.endswith("regular-subset.ttf")
        assert subset_bold.endswith("bold-subset.ttf")
        assert [call[2] for call in calls] == ["regular", "bold"]
        assert subset_dir is not None
    finally:
        if subset_dir:
            shutil.rmtree(subset_dir, ignore_errors=True)


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

    # 2026-07-27: three whole paragraphs of a Bank of Japan Review shipped as
    # exact English because they "ran into a formula".  Two of those formulas
    # were prose lines that math detection promoted over one inline variable,
    # and the third paragraph was a finished sentence whose footnote marker
    # hid the full stop.
    def test_mixed_math_line_does_not_freeze_the_paragraph_above(self):
        from phoena_translator.pdf.audit import (
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
        from phoena_translator.pdf.audit import (
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
        from phoena_translator.pdf.audit import (
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

    def test_duplicate_digit_tail_is_not_absorbed_twice(self):
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
        from phoena_translator.pdf.audit import (
            _collect_pdf_superscript_expectations,
            _validate_pdf_orphan_tail_decision,
        )
        from phoena_translator.pdf.math_detection import (
            _pdf_superscript_signature,
        )
        from phoena_translator.pdf.semantic_cross_page import (
            _merge_cross_page_sentences,
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
        from phoena_translator.pdf.semantic_cross_page import (
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
        first_elem["layout_class"] = "heading"
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

    def test_source_signal_allows_capitalized_tail(self):
        from phoena_translator.pdf.semantic_cross_page import (
            _cross_page_tail_start_acceptable,
        )

        assert _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True, source_ends_capitalized=True
        )
        assert _cross_page_tail_start_acceptable(
            "Traders.", source_dangling=True, source_ends_capitalized=False
        )
        # Non-body captions are rejected before this lexical OR helper.
        assert _cross_page_tail_start_acceptable(
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

    def test_lowercase_destination_signal_overrides_terminated_source(self):
        from phoena_translator.pdf.semantic_cross_page import (
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

    # 2026-07-27: page 4 of a Bank of Japan Review shipped 100% English.  A
    # chart page makes the layout classifier stamp "table" on short mid-page
    # paragraphs; one of those seeded the sweep, and the sweep — one box grown
    # from cells in BOTH text columns — then walked up the page.
    def test_body_paragraph_stamped_table_keeps_translating(self):
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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

        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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
        from phoena_translator.pdf.extraction_stage import (
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

        from phoena_translator.pdf.semantic_continuations import (
            _pdf_internal_paragraph_lead_residuals,
        )
        from phoena_translator.pdf.semantic_tables import (
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

        from phoena_translator.pdf.semantic_continuations import (
            _pdf_internal_paragraph_lead_residuals,
        )
        from phoena_translator.pdf.semantic_tables import (
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
        from phoena_translator.pdf.targets import (
            _mark_pdf_entity_directory_elements,
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

    def test_shape_check_without_layout_evidence_allows_signal_one(self):
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
        assert _validate_pdf_orphan_tail_decision(decision, None) is None

    def test_audit_rejects_non_body_destination_with_layout_evidence(self):
        from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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


# ---------------------------------------------------------------------------
# 2026-07-25 strict-audit release blockers C001 / C002 / C004
# ---------------------------------------------------------------------------


def test_complete_source_lowercase_destination_is_absorbed_by_or_policy():
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

    assert absorbed == 1
    assert source["content"] == (
        "Demand remained stable. "
        "the next section discusses a separate result."
    )
    assert destination["content"] == "Its ownership must stay on page two."
    assert destination["type"] == "text"
    assert audit_log[0]["source_continuation_reason"] == "complete"


def test_orphan_audit_accepts_complete_source_lowercase_decision():
    from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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


def test_cross_page_first_sentence_boundary_keeps_abbreviation_and_marker():
    from phoena_translator.pdf.semantic_cross_page import (
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
    from phoena_translator.pdf.semantic_cross_page import (
        _merge_cross_page_sentences,
    )

    pages, rects, source, destination = _two_page_fixture(
        "Demand remained stable;",
        "The next section begins independently. Another sentence follows.",
    )
    assert _merge_cross_page_sentences(pages, 2, rects, []) == 0
    assert source["content"] == "Demand remained stable;"
    assert destination["content"].startswith("The next section")


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


def test_native_superscript_flag_does_not_turn_same_baseline_prose_into_markup():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    # This mirrors the corrupt MuPDF span flags observed after a split T_3 1/2
    # expression: the following ordinary 11.96pt prose is flagged as native
    # superscript even though it is neither smaller nor raised.
    spans = [
        {
            "text": "2",
            "size": 5.98,
            "flags": 4,
            "origin": [217.87, 187.79],
            "x0": 217.87,
            "x1": 221.52,
            "y0": 182.0,
            "y1": 188.5,
        },
        {
            "text": "-space.",
            "size": 11.96,
            "flags": 5,
            "origin": [223.22, 181.69],
            "x0": 223.22,
            "x1": 259.81,
            "y0": 170.0,
            "y1": 183.0,
        },
        {
            "text": "Consequently",
            "size": 11.96,
            "flags": 5,
            "origin": [266.02, 181.69],
            "x0": 266.02,
            "x1": 336.40,
            "y0": 170.0,
            "y1": 183.0,
        },
        {
            "text": "it",
            "size": 11.96,
            "flags": 5,
            "origin": [340.82, 181.69],
            "x0": 340.82,
            "x1": 348.44,
            "y0": 170.0,
            "y1": 183.0,
        },
    ]

    _mark_pdf_superscript_spans(spans)

    assert not any(span["superscript"] for span in spans)
    assert "".join(span["rich"] for span in spans) == "2-space.Consequentlyit"


def test_native_superscript_flag_keeps_a_smaller_raised_marker():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    spans = [
        {
            "text": "capacity.",
            "size": 12.0,
            "flags": 4,
            "origin": [72.0, 100.0],
            "x0": 72.0,
            "x1": 124.0,
            "y0": 88.0,
            "y1": 102.0,
        },
        {
            "text": "11",
            "size": 7.0,
            "flags": 5,
            "origin": [124.0, 96.0],
            "x0": 124.0,
            "x1": 133.0,
            "y0": 89.0,
            "y1": 98.0,
        },
    ]

    _mark_pdf_superscript_spans(spans)

    assert spans[0]["superscript"] is False
    assert spans[1]["superscript"] is True
    assert spans[1]["rich"] == "<sup>11</sup>"


def test_native_superscript_flag_keeps_a_same_size_raised_marker():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    spans = [
        {
            "text": "x",
            "size": 12.0,
            "flags": 4,
            "origin": [72.0, 100.0],
            "x0": 72.0,
            "x1": 80.0,
            "y0": 88.0,
            "y1": 102.0,
        },
        {
            "text": "2",
            "size": 12.0,
            "flags": 5,
            "origin": [80.0, 96.0],
            "x0": 80.0,
            "x1": 87.0,
            "y0": 87.0,
            "y1": 99.0,
        },
    ]

    _mark_pdf_superscript_spans(spans)

    assert spans[1]["superscript"] is True
    assert spans[1]["superscript_source"] == "native"


def test_native_superscript_flag_keeps_a_marker_only_line():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    spans = [{
        "text": "\u2020",
        "size": 8.0,
        "flags": 5,
        "origin": [72.0, 96.0],
        "x0": 72.0,
        "x1": 77.0,
        "y0": 88.0,
        "y1": 98.0,
    }]

    _mark_pdf_superscript_spans(spans)

    assert spans[0]["superscript"] is True
    assert spans[0]["superscript_source"] == "native"


def test_native_superscript_flag_rejects_a_remote_unrelated_reference():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    spans = [
        {
            "text": "base",
            "size": 12.0,
            "flags": 4,
            "origin": [72.0, 100.0],
            "x0": 72.0,
            "x1": 96.0,
            "y0": 88.0,
            "y1": 102.0,
        },
        {
            "text": "2",
            "size": 7.0,
            "flags": 5,
            "origin": [140.0, 96.0],
            "x0": 140.0,
            "x1": 146.0,
            "y0": 89.0,
            "y1": 98.0,
        },
    ]

    _mark_pdf_superscript_spans(spans)

    assert spans[1]["superscript"] is False


def test_native_superscript_geometry_accepts_the_inclusive_raise_boundary():
    from phoena_translator.pdf.math_detection import _mark_pdf_superscript_spans

    spans = [
        {
            "text": "x",
            "size": 12.0,
            "flags": 4,
            "origin": [72.0, 100.0],
            "x0": 72.0,
            "x1": 80.0,
            "y0": 88.0,
            "y1": 102.0,
        },
        {
            "text": "3",
            "size": 9.6,
            "flags": 5,
            "origin": [80.0, 98.56],
            "x0": 80.0,
            "x1": 86.0,
            "y0": 89.0,
            "y1": 100.0,
        },
    ]

    _mark_pdf_superscript_spans(spans)

    assert spans[1]["superscript"] is True


def test_translation_integrity_source_fallback_cache_rebinds_once(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "d" * 64
    source_element = _advanced_native_cache_element()
    fallback_element = copy.deepcopy(source_element)
    fallback_element["skip_translate_reason"] = "translation_integrity_fallback"
    path = cache._save_pdf_page_translation_cache(
        "fallback-cache",
        0,
        {"0": source_element["content"]},
        source_sha256,
        elements=[fallback_element],
    )

    resumed_element = copy.deepcopy(source_element)
    loaded = cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    )

    assert loaded == {"0": source_element["content"]}
    assert resumed_element["skip_translate_reason"] == "translation_integrity_fallback"


def test_unmarked_source_echo_cache_is_still_rejected(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "e" * 64
    source_element = _advanced_native_cache_element()
    path = cache._save_pdf_page_translation_cache(
        "source-echo-cache",
        0,
        {"0": source_element["content"]},
        source_sha256,
        elements=[source_element],
    )

    resumed_element = copy.deepcopy(source_element)
    assert cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    ) == {}
    assert "skip_translate_reason" not in resumed_element


def test_fallback_cache_is_checked_after_a_normal_source_echo_is_rejected(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "f" * 64
    source_element = _advanced_native_cache_element()
    fallback_element = copy.deepcopy(source_element)
    fallback_element["skip_translate_reason"] = "translation_integrity_fallback"
    path = cache._save_pdf_page_translation_cache(
        "mixed-source-echo-cache",
        0,
        {
            "0": source_element["content"],
            "1": source_element["content"],
        },
        source_sha256,
        elements=[source_element, fallback_element],
    )

    resumed_element = copy.deepcopy(source_element)
    loaded = cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    )

    assert loaded == {"0": source_element["content"]}
    assert resumed_element["skip_translate_reason"] == "translation_integrity_fallback"


def test_clean_page_retries_use_the_configured_worker_pool(monkeypatch):
    import time
    from types import SimpleNamespace

    from phoena_translator.pdf import translation_stage as stage

    lock = threading.Lock()
    active = 0
    max_active = 0
    committed: list[int] = []

    def fake_translate_page(_page_context, page_num, _page_info):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return page_num, {0: "译文"}

    monkeypatch.setattr(
        stage,
        "_run_first_pass",
        lambda *_args, **_kwargs: {page: RuntimeError("retry") for page in range(3)},
    )
    monkeypatch.setattr(stage, "translate_page", fake_translate_page)
    monkeypatch.setattr(stage, "_validate_pdf_page_translations", lambda *_args: None)
    monkeypatch.setattr(
        stage,
        "_commit_translated_page",
        lambda _context, page_num, _translations, _lock: committed.append(page_num),
    )
    monkeypatch.setattr(stage, "_apply_page_source_fallbacks", lambda *_args: None)
    monkeypatch.setattr(stage, "_require_translated_page_progress", lambda *_args: None)

    context = SimpleNamespace(
        workers=3,
        page_extractions={page: {"elements": [{}]} for page in range(3)},
        logger=log,
        task_id="parallel-clean-retry",
    )
    stage.translate_pages(context, SimpleNamespace(), [0, 1, 2])

    assert max_active == 3
    assert sorted(committed) == [0, 1, 2]


def test_clean_page_retries_keep_worker_errors_bound_to_scheduled_pages(monkeypatch):
    from types import SimpleNamespace

    from phoena_translator.pdf import translation_stage as stage

    captured_errors = {}
    committed = []

    def fake_translate_page(_page_context, page_num, _page_info):
        if page_num == 0:
            return 2, {0: "wrong page"}
        raise TimeoutError("provider retry timed out")

    monkeypatch.setattr(
        stage,
        "_run_first_pass",
        lambda *_args, **_kwargs: {0: RuntimeError("retry"), 1: RuntimeError("retry")},
    )
    monkeypatch.setattr(stage, "translate_page", fake_translate_page)
    monkeypatch.setattr(stage, "_validate_pdf_page_translations", lambda *_args: None)
    monkeypatch.setattr(
        stage,
        "_commit_translated_page",
        lambda _context, page_num, _translations, _lock: committed.append(page_num),
    )
    monkeypatch.setattr(
        stage,
        "_apply_page_source_fallbacks",
        lambda _context, errors: captured_errors.update(errors),
    )
    monkeypatch.setattr(stage, "_require_translated_page_progress", lambda *_args: None)

    context = SimpleNamespace(
        workers=2,
        page_extractions={page: {"elements": [{}]} for page in range(3)},
        logger=log,
        task_id="retry-page-binding",
    )
    stage.translate_pages(context, SimpleNamespace(), [0, 1])

    assert committed == []
    assert sorted(captured_errors) == [1, 2]
    assert "returned page 3 for scheduled page 1" in str(captured_errors[1])
    assert isinstance(captured_errors[2], TimeoutError)


def test_render_failure_restores_source_pages_without_restarting_pipeline(
    tmp_path,
    monkeypatch,
):
    import shutil
    from types import SimpleNamespace

    import fitz

    from phoena_translator.pdf import assembly
    from phoena_translator.pdf.types import PDFPageRenderError

    source_path = tmp_path / "source.pdf"
    work_path = tmp_path / "assembly-work.pdf"
    source_document = fitz.open()
    for page_number in range(1, 4):
        page = source_document.new_page()
        page.insert_text((72, 72), f"SOURCE PAGE {page_number}")
    source_document.save(source_path)
    source_document.close()
    shutil.copy2(source_path, work_path)

    work_document = fitz.open(work_path)
    work_document[0].insert_text((72, 100), "CHECKPOINTED TRANSLATION")
    work_document.saveIncr()
    # Simulate the dirty current page at the instant its renderer raises. This
    # mutation must disappear when the document closes without another save.
    work_document[1].insert_text((72, 100), "UNSAFE PARTIAL RENDER")

    audit = {
        "source_page_fallbacks": [],
        "merge_decisions": [],
        "superscript_expectations": [{"page": 1}, {"page": 2}, {"page": 3}],
        "formula_expectations": [],
        "vector_ocr_expectations": [],
    }
    context = SimpleNamespace(
        task_id="inline-render-recovery",
        total_pages=3,
        pdf_password="",
        fail_open_to_source_page=True,
        assembly_work_path=str(work_path),
        src_path=str(source_path),
        out_doc=work_document,
        page_extractions={0: {}, 1: {}, 2: {}},
        pdf_audit=audit,
        logger=log,
    )
    monkeypatch.setattr(
        assembly,
        "_expand_pdf_fallback_pages_for_accepted_merges",
        lambda *_args, **_kwargs: [1, 2],
    )

    recovered = assembly._continue_assembly_with_source_pages(
        context,
        1,
        PDFPageRenderError(2, 7, "no safe rectangle"),
    )
    try:
        assert recovered is True
        assert "CHECKPOINTED TRANSLATION" not in context.out_doc[0].get_text()
        assert "UNSAFE PARTIAL RENDER" not in context.out_doc[1].get_text()
        assert [
            entry["page"] for entry in audit["source_page_fallbacks"]
        ] == [1, 2]
        assert context.page_extractions[0]["source_page_fallback"]["page"] == 1
        assert context.page_extractions[1]["source_page_fallback"]["page"] == 2
        assert audit["superscript_expectations"] == [{"page": 3}]
    finally:
        context.out_doc.close()


def test_assembly_continues_after_an_inline_render_source_fallback(monkeypatch):
    from types import SimpleNamespace

    from phoena_translator.pdf import assembly
    from phoena_translator.pdf.types import PDFPageRenderError

    attempted = []

    def fake_assemble_page(_context, page_num):
        attempted.append(page_num)
        if page_num == 0:
            raise PDFPageRenderError(1, 2, "render failure")

    monkeypatch.setattr(assembly, "_assemble_page", fake_assemble_page)
    monkeypatch.setattr(
        assembly,
        "_continue_assembly_with_source_pages",
        lambda *_args, **_kwargs: True,
    )

    assembly.assemble_document_pages(SimpleNamespace(total_pages=3))

    assert attempted == [0, 1, 2]


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


def test_cross_page_merge_allows_page_turn_to_first_destination_column():
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

    assert _merge_cross_page_sentences(pages, 2, rects, []) == 1
    assert source["content"] == (
        "The explanation continues across pages without changing ownership."
    )
    assert heading["type"] == "text"
    assert continuation["type"] == "text_merged_away"


def test_cross_page_merge_skips_graph_title_and_table_labels():
    from phoena_translator.pdf.semantic_cross_page import (
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

    from phoena_translator.pdf.semantic_cross_page import (
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
    from phoena_translator.pdf.semantic_cross_page import (
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


def test_pdf_lexical_angle_phrase_survives_translation_protection_and_rendering():
    """BOJ p4: ``<JPY short position>`` is prose, not an HTML tag."""
    from phoena_translator.pdf.math_detection import (
        _normalize_pdf_translation,
        _protect_pdf_inline_math_fragments,
        _restore_pdf_inline_math_fragments,
    )
    from phoena_translator.pdf.rendering import _pdf_text_to_html

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
    from phoena_translator.pdf.math_detection import _normalize_pdf_translation

    assert _normalize_pdf_translation("<p>Hello</p>") == "Hello"


class _FakePDFTable:
    def __init__(self, bbox, rows, columns, extracted):
        self.bbox = bbox
        self.row_count = rows
        self.col_count = columns
        self._extracted = extracted

    def extract(self):
        return self._extracted


def test_pdf_sparse_two_column_wrapper_yields_to_nested_real_table():
    """BOJ p1: body prose beside Chart 1 is not one giant 1x2 table."""
    from phoena_translator.pdf.extraction import (
        _prune_sparse_pdf_table_wrappers,
    )

    prose = (
        "decision to execution are conducted automatically based on "
        "pre-determined programs and algorithmic trading has been on an "
        "upward trend because it enables high-speed and high-frequency trading"
    )
    chart = (
        "Major types of algorithms Strategy Contents Trading Algorithms "
        "Market make Directional Arbitrage Execution Algorithms"
    )
    wrapper = _FakePDFTable(
        (57.0, 585.5, 539.4, 789.1),
        1,
        2,
        [[prose, chart]],
    )
    nested = _FakePDFTable(
        (316.1, 606.6, 532.7, 745.0),
        6,
        3,
        [["Strategy", None, "Contents"]] * 6,
    )

    assert _prune_sparse_pdf_table_wrappers([wrapper, nested]) == [nested]
    assert _prune_sparse_pdf_table_wrappers([wrapper]) == [wrapper]


def test_pdf_five_justified_fragments_recombine_on_one_baseline():
    """BOJ p2: ``traditional market making function (liquidity`` is prose."""
    import fitz

    from phoena_translator.pdf.semantic_baselines import (
        _normalize_pdf_same_baseline_prose_fragments,
    )

    def fragment(text, x0, x1, y0=100.0, y1=112.0):
        return {
            "plain": text,
            "rich": text,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "fontsize": 10.0,
            "line_height": 12.0,
            "color": 0,
            "bold": False,
            "table_hint": False,
            "math_protected": False,
        }

    fragments = [
        fragment("traditional ", 50.0, 95.0),
        fragment("market ", 108.0, 143.0),
        fragment("making ", 156.0, 191.0),
        fragment("function ", 204.0, 244.0),
        fragment("(liquidity ", 257.0, 290.0),
        fragment("provision).", 50.0, 105.0, 114.0, 126.0),
    ]
    normalized = _normalize_pdf_same_baseline_prose_fragments(
        fragments,
        fitz.Rect(50.0, 95.0, 290.0, 126.0),
    )

    assert len(normalized) == 2
    assert normalized[0]["plain"] == (
        "traditional market making function (liquidity"
    )
    assert normalized[0]["same_baseline_coalesced"] is True
    assert normalized[1]["plain"] == "provision)."


def test_pdf_justified_discourse_connector_stays_in_its_paragraph():
    """BOJ p1: a justified ``functioning. Thus,`` row is not a new block."""
    import fitz

    from phoena_translator.pdf.semantic_baselines import (
        _normalize_pdf_same_baseline_prose_fragments,
    )

    def fragment(text, x0, x1, y0=100.0, y1=112.0):
        return {
            "plain": text,
            "rich": text,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "fontsize": 10.0,
            "line_height": 12.0,
            "color": 0,
            "bold": False,
            "table_hint": False,
            "math_protected": False,
        }

    fragments = [
        fragment("mechanism ", 50.0, 98.0),
        fragment("and ", 111.0, 132.0),
        fragment("market ", 145.0, 180.0),
        fragment("functioning. ", 193.0, 250.0),
        fragment("Thus, ", 263.0, 290.0),
        fragment(
            "understanding characteristics of algorithmic trading is ",
            50.0,
            290.0,
            114.0,
            126.0,
        ),
    ]
    normalized = _normalize_pdf_same_baseline_prose_fragments(
        fragments,
        fitz.Rect(50.0, 95.0, 290.0, 126.0),
    )

    assert normalized[0]["plain"] == "mechanism and market functioning. Thus,"
    assert not normalized[0].get("same_baseline_paragraph_start")


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


def test_a_preserved_element_is_refused_as_a_cross_page_merge_endpoint():
    from phoena_translator.pdf.semantic_cross_page import (
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


def test_preservation_is_decided_before_cross_page_merging():
    import inspect

    from phoena_translator.pdf import extraction_stage

    body = inspect.getsource(extraction_stage.reconcile_extraction)
    merge_at = body.index("_merge_cross_page_sentences")
    for stamper in (
        "_preserve_pdf_formula_mid_sentence_neighbors",
        "_preserve_formula_risk_before_cache_binding",
        "enforce_table_preservation",
    ):
        assert body.index(stamper) < merge_at, stamper
    # ...and every stamp still lands before cache binding.
    assert merge_at < body.index("_bind_and_reconcile_caches")


def test_orphan_audit_still_rejects_a_preserved_endpoint():
    from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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
    from phoena_translator.pdf.audit import _validate_pdf_orphan_tail_decision

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


def _single_formula_page(rotation: int):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((300.0, 700.0), "f(x) = 1", fontsize=12)
    if rotation:
        page.set_rotation(rotation)
    return doc, page


def test_formula_signature_survives_a_rotated_page():
    from phoena_translator.pdf.math_detection import (
        _pdf_formula_region_signature,
    )

    bbox = [295.0, 686.0, 390.0, 706.0]
    upright_doc, upright_page = _single_formula_page(0)
    rotated_doc, rotated_page = _single_formula_page(90)
    # The rotated page reports ink at y=686..706 while ``page.rect`` is only
    # 612 tall, so clipping against ``page.rect`` used to collapse the region.
    assert rotated_page.rect.height < bbox[1]

    upright = _pdf_formula_region_signature(upright_page, bbox)
    rotated = _pdf_formula_region_signature(rotated_page, bbox)

    assert upright["span_count"] == rotated["span_count"] == 1
    # Extraction coordinates are rotation independent, so the glyph signature
    # must match; only the raster is sampled from the rotated view.
    assert rotated["signature_sha256"] == upright["signature_sha256"]
    assert rotated["raster_width"] > 0 and rotated["raster_height"] > 0
    upright_doc.close()
    rotated_doc.close()


def test_one_unsignable_formula_does_not_strip_its_page_siblings(monkeypatch):
    from phoena_translator.pdf import targets

    attempted = []

    def flaky_signature(page, bbox, page_dict=None):
        attempted.append(list(bbox))
        if len(attempted) == 1:
            raise ValueError("empty formula region")
        return {"signature_sha256": "b" * 64, "span_count": 1}

    monkeypatch.setattr(
        targets,
        "_pdf_formula_region_signature",
        flaky_signature,
    )
    doc, page = _single_formula_page(0)
    elements = [
        {"type": "formula_image", "bbox": [10.0, 10.0, 40.0, 30.0]},
        {"type": "formula_image", "bbox": [10.0, 50.0, 40.0, 70.0]},
        {"type": "formula_image", "bbox": [10.0, 90.0, 40.0, 110.0]},
    ]

    enriched = targets._enrich_pdf_page_elements(page, elements, [])

    assert len(attempted) == 3, "every formula must be attempted"
    assert "math_signature" not in enriched[0]
    assert enriched[1]["math_signature"]["signature_sha256"] == "b" * 64
    assert enriched[2]["math_signature"]["signature_sha256"] == "b" * 64
    doc.close()


def _glyph_span(text: str, x0: float, x1: float, y0: float = 86.83, color: int = 0x231F20):
    return {
        "text": text,
        "font": "NewBaskervilleStd-Roman",
        "size": 10.0,
        "color": color,
        "bbox": (x0, y0, x1, y0 + 10.0),
    }


def test_ink_stroked_twice_in_place_is_collapsed_to_one_span():
    from phoena_translator.pdf.targets import _drop_pdf_duplicate_glyph_spans

    line = "that this is equivalent to maximizing the expected value "
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [_glyph_span(line, 66.0, 428.64)]},
                    {"spans": [_glyph_span(line, 66.0, 428.60)]},
                ],
            }
        ]
    }

    cleaned, dropped = _drop_pdf_duplicate_glyph_spans(page_dict)

    assert dropped == 1
    assert len(cleaned["blocks"][0]["lines"]) == 1
    assert cleaned["blocks"][0]["lines"][0]["spans"][0]["text"] == line


def test_repeated_text_at_a_different_position_is_kept():
    from phoena_translator.pdf.targets import _drop_pdf_duplicate_glyph_spans

    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _glyph_span("0.00%", 200.0, 240.0, y0=300.0),
                            _glyph_span("0.00%", 400.0, 440.0, y0=300.0),
                            _glyph_span("0.00%", 200.0, 240.0, y0=400.0),
                        ]
                    }
                ],
            }
        ]
    }

    cleaned, dropped = _drop_pdf_duplicate_glyph_spans(page_dict)

    assert dropped == 0
    assert cleaned is page_dict
    assert len(cleaned["blocks"][0]["lines"][0]["spans"]) == 3


def test_duplicate_detection_leaves_image_blocks_and_whitespace_alone():
    from phoena_translator.pdf.targets import _drop_pdf_duplicate_glyph_spans

    page_dict = {
        "blocks": [
            {"type": 1, "bbox": (0.0, 0.0, 10.0, 10.0)},
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _glyph_span("  ", 66.0, 70.0),
                            _glyph_span("  ", 66.0, 70.0),
                        ]
                    }
                ],
            },
        ]
    }

    cleaned, dropped = _drop_pdf_duplicate_glyph_spans(page_dict)

    assert dropped == 0
    assert cleaned["blocks"][0]["type"] == 1
    assert len(cleaned["blocks"][1]["lines"][0]["spans"]) == 2


def test_a_bare_doi_line_is_not_protected_as_formula_geometry():
    from phoena_translator.pdf.math_detection import (
        _looks_like_pdf_identifier_line,
    )

    # The ``=`` alone used to promote this line to a protected formula region,
    # whose raster expectation then failed on font-subsetting drift.
    assert _looks_like_pdf_identifier_line("doi=10.1257/jep.27.2.51")
    assert _looks_like_pdf_identifier_line("doi: 10.1257/jep.27.2.51")
    assert _looks_like_pdf_identifier_line("10.1093/rfs/hhab001")
    # Real notation with decimals and a division must still be protected.
    assert not _looks_like_pdf_identifier_line("Σ = 10.5 / 2.25 + ε")
    assert not _looks_like_pdf_identifier_line("x = a/b where a ≥ 0")


def test_duplicate_collapse_keeps_the_last_drawn_copy_not_the_white_knockout():
    from phoena_translator.pdf.targets import _drop_pdf_duplicate_glyph_spans

    line = "differs from the semiconductor industry in at least "
    # Apache FOP emits a white knockout copy FIRST and the readable copy
    # second.  PDF paints in stream order, so the survivor must be the last
    # one; keeping the first redrew every translated paragraph in white.
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [_glyph_span(line, 66.0, 428.61, color=0xFFFFFF)]},
                    {"spans": [_glyph_span(line, 66.0, 428.60, color=0x231F20)]},
                ],
            }
        ]
    }

    cleaned, dropped = _drop_pdf_duplicate_glyph_spans(page_dict)

    assert dropped == 1
    survivors = [
        span
        for block in cleaned["blocks"]
        for cleaned_line in block["lines"]
        for span in cleaned_line["spans"]
    ]
    assert len(survivors) == 1
    assert survivors[0]["color"] == 0x231F20, "the white knockout must not win"


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


def test_mid_sentence_stub_running_straight_into_a_formula_is_preserved():
    from phoena_translator.pdf.audit import (
        _preserve_pdf_formula_mid_sentence_neighbors,
    )

    pages = _mid_sentence_page(with_intervening_prose=False)

    preserved = _preserve_pdf_formula_mid_sentence_neighbors(pages)

    assert [entry["element"] for entry in preserved] == [0]
    assert pages[0]["elements"][0]["skip_translate_reason"] == (
        "formula_risk_preserved"
    )


def test_prose_between_the_stub_and_the_formula_blocks_preservation():
    from phoena_translator.pdf.audit import (
        _preserve_pdf_formula_mid_sentence_neighbors,
    )

    # An inline math variable splits its own line, so the tail after the
    # variable becomes a separate element and the parent only *looks* like it
    # stops mid-sentence.  Its real continuation is prose, not the formula.
    pages = _mid_sentence_page(with_intervening_prose=True)

    preserved = _preserve_pdf_formula_mid_sentence_neighbors(pages)

    assert preserved == []
    assert "skip_translate_reason" not in pages[0]["elements"][0]


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
        from phoena_translator.pdf.translation import (
            _short_translation_needs_retry,
        )

        for source in self.APPARATUS_ONLY:
            assert not _short_translation_needs_retry(source, source), source

    def test_translatable_content_still_drives_the_retry_ladder(self):
        from phoena_translator.pdf.translation import (
            _short_translation_needs_retry,
        )

        for source in self.MUST_STILL_RETRY:
            assert _short_translation_needs_retry(source, source), source

    def test_quoted_title_word_count_gates_the_exemption(self):
        from phoena_translator.pdf.translation import (
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


class TestRotatedPageCoordinateSpace:
    """Extraction geometry lives in the unrotated box, not ``page.rect``."""

    @staticmethod
    def _rotated_page(rotation):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(
            fitz.Rect(72, 640, 540, 760),
            "Regulators responded with circuit breakers and minimum quoting "
            "obligations, but the evidence on their effectiveness is mixed.",
            fontsize=11,
            fontname="tiro",
        )
        if rotation:
            page.set_rotation(rotation)
        return doc, page

    def test_extraction_rect_is_a_noop_without_rotation(self):
        import fitz

        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect

        doc, page = self._rotated_page(0)
        assert tuple(_pdf_page_extraction_rect(page)) == tuple(page.rect)
        doc.close()

    def test_extraction_rect_derotates_the_display_box(self):
        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect

        doc, page = self._rotated_page(90)
        assert tuple(page.rect) == (0.0, 0.0, 792.0, 612.0)
        assert tuple(_pdf_page_extraction_rect(page)) == (0.0, 0.0, 612.0, 792.0)
        doc.close()

    def test_clamping_against_the_display_box_destroys_the_rect(self):
        import fitz

        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect
        from phoena_translator.pdf.rendering import _sanitize_pdf_text_rect

        doc, page = self._rotated_page(90)
        elem = fitz.Rect(100.0, 700.0, 300.0, 714.0)

        crushed = _sanitize_pdf_text_rect(fitz.Rect(elem), fitz.Rect(page.rect))
        intact = _sanitize_pdf_text_rect(
            fitz.Rect(elem), _pdf_page_extraction_rect(page)
        )

        assert crushed is not None and crushed.height <= 2.0
        assert intact is not None and abs(intact.height - elem.height) < 0.01
        doc.close()

    def test_body_text_in_the_far_band_is_not_demoted_to_scattered(self):
        import fitz

        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect
        from phoena_translator.pdf.targets import (
            _classify_pdf_page_text_elements, _extract_page_elements)

        doc, page = self._rotated_page(90)
        elements = _extract_page_elements(page, doc)
        assert elements, "probe page produced no elements"

        display = [dict(e) for e in elements]
        _classify_pdf_page_text_elements(display, fitz.Rect(page.rect))
        extraction = [dict(e) for e in elements]
        _classify_pdf_page_text_elements(
            extraction, _pdf_page_extraction_rect(page)
        )

        assert any(e.get("layout_class") == "scattered" for e in display)
        assert all(
            e.get("layout_class") == "body"
            for e in extraction
            if e.get("type") == "text"
        )
        doc.close()
