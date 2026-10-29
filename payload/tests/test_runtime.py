"""Service runtime: configuration, provider gateway, workers, logging and fonts.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging
import threading

from phoena_translator.api_runtime import is_rate_limit_error, strip_think_tags
from phoena_translator.config import AppConfig, get_app_config
from phoena_translator.glossary import build_glossary_text
from phoena_translator.llm import DeepSeekTranslationAdapter, LLMSettings
from phoena_translator.logging_setup import configure_logging
from phoena_translator.workers import TaskJob, TaskWorkerPool

log = logging.getLogger("test")


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
# B6 — one process-wide configuration snapshot
# ---------------------------------------------------------------------------


def test_get_app_config_is_memoized():
    assert get_app_config() is get_app_config()


def test_pdf_constants_share_the_process_config():
    import phoena_translator.pdf.settings as pdf_settings

    config = get_app_config()
    assert (
        pdf_settings.PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE
        == config.pdf_min_acceptable_htmlbox_scale
    )
    assert pdf_settings.PDF_VECTOR_OCR_DPI == config.pdf_vector_ocr_dpi
    assert (
        pdf_settings.PDF_VECTOR_OCR_MIN_ALPHA_WORDS
        == config.pdf_vector_ocr_min_alpha_words
    )


def test_font_override_parsed_from_env(tmp_path):
    config = AppConfig.from_env(
        environ={"TRANSLATOR_PDF_FONT_REGULAR": str(tmp_path / "r.otf")},
        home=tmp_path,
    )
    assert config.pdf_font_regular == tmp_path / "r.otf"
    assert config.pdf_font_bold is None


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
