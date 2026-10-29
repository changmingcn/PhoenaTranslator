"""PDF rendering: assembly, source-page fallbacks and output fonts.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import pytest


def test_font_override_resolution(tmp_path):
    from phoena_translator.pdf.render.fonts import _get_chinese_font_path

    font_file = tmp_path / "font.otf"
    font_file.write_bytes(b"stub")
    assert _get_chinese_font_path(override=str(font_file)) == str(font_file)
    with pytest.raises(RuntimeError):
        _get_chinese_font_path(override=str(tmp_path / "missing.otf"))


# ---------------------------------------------------------------------------
# B9 — CFF-flavored fonts must not be subset (MuPDF mis-renders the result)
# ---------------------------------------------------------------------------


def test_cff_font_subsetting_is_skipped(tmp_path):
    from phoena_translator.pdf.render.fonts import _pdf_font_is_cff, _subset_pdf_font

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
    from phoena_translator.pdf.render.fonts import pdf_font_is_safely_embeddable

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
    from phoena_translator.pdf.render.fonts import resolve_output_fonts

    big_cff = tmp_path / "big.otf"
    big_cff.write_bytes(b"OTTO" + b"\x00" * (5 * 1024 * 1024))
    with pytest.raises(RuntimeError, match="TrueType"):
        resolve_output_fonts(str(big_cff), str(big_cff))
