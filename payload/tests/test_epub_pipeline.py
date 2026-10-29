"""EPUB pipeline: segment validation, retries and archive safety.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging

import pytest

from phoena_translator.epub.pipeline import (
    EPUBPipeline,
    EPUBPipelineDependencies,
)

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# B5 — the pipeline rejects bad segments
# ---------------------------------------------------------------------------

_SOURCE_CHUNK = (
    "<html><body><p>This is a long enough paragraph of English prose that "
    "clearly needs translation to Chinese.</p></body></html>"
)


def test_pipeline_fails_without_source_fallback_after_bounded_retries(tmp_path):
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
        workers=1,
        api_concurrency=1,
        max_chunk_rounds=4,
        sleep=lambda seconds: None,
        max_fallback_segments=0,
        max_fallback_ratio=0,
    )
    with pytest.raises(ValueError, match="no output published"):
        EPUBPipeline(dependencies)._translate_extracted("task", book_root)

    assert len(calls) == 4
    assert all("JSON" in prompt for prompt in calls)
    assert "上次响应未通过校验" in calls[1]
    assert chapter.read_text(encoding="utf-8") == _SOURCE_CHUNK


def test_pipeline_rejects_non_utf8_member_before_translation(tmp_path):
    book_root = tmp_path / "book"
    book_root.mkdir()
    good = book_root / "a.xhtml"
    good.write_text(_SOURCE_CHUNK, encoding="utf-8")
    bad = book_root / "b.xhtml"
    bad.write_bytes("<p>UTF-16 content that is long enough here</p>".encode("utf-16"))

    def fake_translate(text, system_prompt=None, **kwargs):
        pytest.fail("non-UTF-8 input must fail before any translation call")

    dependencies = EPUBPipelineDependencies(
        translate_text=fake_translate,
        build_glossary=lambda text, task_id="": "",
        make_prompt=lambda prompt, glossary: prompt,
        save_progress=lambda task_id, data: None,
        load_task=lambda task_id: {},
        logger=log,
        workers=1,
        api_concurrency=1,
        sleep=lambda seconds: None,
    )
    with pytest.raises(ValueError, match="b.xhtml"):
        EPUBPipeline(dependencies)._translate_extracted("task", book_root)
    assert bad.read_bytes().startswith(b"\xff\xfe")  # untouched source bytes
