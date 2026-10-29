"""Textbox fitting and source-text restoration on real PyMuPDF pages.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import fitz
import pytest
from golden_support import build_test_font

from phoena_translator.pdf.render.primitives import (
    _insert_pdf_textbox_with_fit,
    _restore_pdf_source_text_element,
)

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可"


@pytest.fixture(scope="module")
def font_path(tmp_path_factory):
    built = build_test_font(tmp_path_factory.mktemp("font"))
    if built is None:
        pytest.skip("no CJK TrueType font available to build the test font")
    return str(built[0])


@pytest.fixture
def page():
    document = fitz.open()
    yield document.new_page(width=612, height=792)
    document.close()


def _fit(page, rect, text, font_path, **options):
    options.setdefault("fontsize", 10.0)
    options.setdefault("lineheight_points", 12.0)
    return _insert_pdf_textbox_with_fit(
        page,
        fitz.Rect(rect),
        text,
        align=0,
        color=(0, 0, 0),
        fontfile=font_path,
        **options,
    )


def _raise(*_args, **_kwargs):
    raise RuntimeError("engine refused")


# ---------------------------------------------------------------------------
# _insert_pdf_textbox_with_fit
# ---------------------------------------------------------------------------


def test_fit_rejects_empty_text_and_invalid_rects(page, font_path):
    assert _fit(page, (72, 100, 300, 120), "  ", font_path) == (False, "empty text", None)
    assert _fit(page, (float("nan"), 100, 300, 120), "x", font_path) == (
        False,
        "invalid text rect",
        None,
    )


def test_fit_refuses_rects_that_enter_a_formula(page, font_path):
    assert _fit(
        page,
        (72, 100, 300, 120),
        ZH[:4],
        font_path,
        protected_rects=[fitz.Rect(0, 0, 612, 792)],
    ) == (False, "no formula-safe text rect", None)


def test_fit_reports_a_font_that_cannot_be_registered(page, tmp_path):
    ok, message, fontsize = _fit(page, (72, 100, 300, 120), ZH[:4], str(tmp_path / "x.ttf"))
    assert not ok and fontsize is None
    assert message.startswith("textbox font registration failed")


def test_fit_draws_at_full_size_when_the_text_fits(page, font_path):
    ok, message, fontsize = _fit(page, (72, 100, 300, 130), ZH[:8], font_path)
    assert ok and message.startswith("spare=") and fontsize == 10.0
    assert ZH[:8] in page.get_text().replace("\n", "")


def test_fit_shrinks_or_extends_before_giving_up(page, font_path):
    ok, message, fontsize = _fit(page, (72, 100, 160, 112), ZH[:20], font_path)
    assert ok and message.startswith("fit after resize")
    assert fontsize <= 10.0


def test_fit_stays_inside_vetted_rects(page, font_path):
    vetted = [fitz.Rect(72, 100, 160, 112)]
    # Too long for one 6pt line of the box, short enough for one 8pt page line.
    ok, message, _ = _fit(page, vetted[0], ZH[:20], font_path, fit_rects=vetted)
    assert not ok
    assert "single-line fallback would leave vetted text rects" in message


def test_fit_uses_one_line_when_no_box_is_tall_enough(page, font_path):
    # Every enlarged box runs into the formula below-right; one 8pt line fits.
    formula = fitz.Rect(173, 104, 260, 140)
    ok, message, fontsize = _fit(
        page, (72, 100, 172, 103), "abc def", font_path, protected_rects=[formula]
    )
    assert ok and fontsize == 8.0
    assert message.startswith("used single-line fallback after textbox deficit")
    assert "abc def" in page.get_text()


def test_fit_single_line_must_not_cross_a_formula(page, font_path):
    formula = fitz.Rect(60, 104, 300, 140)
    ok, message, _ = _fit(
        page, (72, 100, 172, 103), "abc def", font_path, protected_rects=[formula]
    )
    assert not ok
    assert message == "single-line fallback would cross a protected formula region"


def test_fit_single_line_must_not_overflow_the_page(page, font_path):
    formula = fitz.Rect(173, 104, 260, 140)
    ok, message, _ = _fit(
        page, (72, 100, 172, 103), ZH * 3, font_path, protected_rects=[formula]
    )
    assert not ok
    assert message == "single-line fallback would overflow the page width"


def test_fit_single_line_must_respect_the_minimum_scale(page, font_path):
    formula = fitz.Rect(173, 110, 260, 140)
    ok, message, _ = _fit(
        page,
        (72, 100, 172, 109),
        "abc def",
        font_path,
        fontsize=20.0,
        lineheight_points=24.0,
        protected_rects=[formula],
    )
    assert not ok
    assert message == "single-line fallback would violate minimum render scale"


def test_fit_measures_the_line_conservatively_when_the_font_cannot(
    page, font_path, monkeypatch
):
    monkeypatch.setattr(fitz, "Font", _raise)
    formula = fitz.Rect(173, 104, 260, 140)
    ok, message, _ = _fit(
        page, (72, 100, 172, 103), "abc def", font_path, protected_rects=[formula]
    )
    assert not ok
    assert message == "single-line fallback would overflow the page width"


def test_fit_survives_textbox_errors_and_reports_the_last_one(page, font_path, monkeypatch):
    monkeypatch.setattr(fitz.Page, "insert_textbox", _raise)
    ok, message, fontsize = _fit(page, (72, 100, 300, 130), "abc def", font_path)
    assert ok and fontsize == 8.0
    assert message == "used single-line fallback after engine refused"


def test_fit_reports_a_failing_single_line_fallback(page, font_path, monkeypatch):
    monkeypatch.setattr(fitz.Page, "insert_textbox", _raise)
    monkeypatch.setattr(fitz.Page, "insert_text", _raise)
    ok, message, fontsize = _fit(page, (72, 100, 300, 130), "abc def", font_path)
    assert not ok and fontsize is None
    assert message == "engine refused; single-line fallback failed: engine refused"


# ---------------------------------------------------------------------------
# _restore_pdf_source_text_element
# ---------------------------------------------------------------------------


def _restore(page, rect, content, font_path, **options):
    return _restore_pdf_source_text_element(
        page,
        {"content": content},
        fitz.Rect(rect),
        color=(0, 0, 0),
        fontfile=font_path,
        fontsize=options.pop("fontsize", 10.0),
        lineheight_points=12.0,
        **options,
    )


def test_restore_needs_visible_source_text(page, font_path):
    assert _restore(page, (72, 100, 300, 130), " <b></b> ", font_path) == (
        False,
        "empty source text",
    )


def test_restore_uses_the_formula_safe_textbox_first(page, font_path):
    ok, message = _restore(page, (72, 100, 300, 130), "Source words", font_path)
    assert ok and message.startswith("source text restored via textbox (spare=")
    assert "Source words" in page.get_text()


def test_restore_writes_directly_into_the_original_region_as_a_last_resort(
    page, font_path
):
    # The source occupied this box originally, so the direct walk may ignore
    # the formula guard that rejected every fitted candidate.
    ok, message = _restore(
        page,
        (72, 100, 300, 130),
        "Source words",
        font_path,
        protected_rects=[fitz.Rect(0, 0, 612, 792)],
    )
    assert ok and message == "source text restored directly at fontsize 10.0"


def test_restore_reports_why_nothing_was_written(page, font_path, tmp_path):
    ok, message = _restore(page, (float("nan"), 100, 300, 130), "Source", font_path)
    assert not ok and message == "invalid text rect; invalid source rect"

    ok, message = _restore(page, (72, 100, 300, 130), "Source", str(tmp_path / "x.ttf"))
    assert not ok and "source restore font registration failed" in message

    ok, message = _restore(
        page,
        (72, 100, 80, 104),
        "Source words " * 20,
        font_path,
        protected_rects=[fitz.Rect(0, 0, 612, 792)],
    )
    assert not ok
    assert message.startswith("no formula-safe text rect; source restore no-fit (textbox deficit")


def test_restore_reports_textbox_errors(page, font_path, monkeypatch):
    monkeypatch.setattr(fitz.Page, "insert_textbox", _raise)
    monkeypatch.setattr(fitz.Page, "insert_text", _raise)
    ok, message = _restore(page, (72, 100, 300, 130), "Source", font_path)
    assert not ok and message.endswith("source restore no-fit (engine refused)")
