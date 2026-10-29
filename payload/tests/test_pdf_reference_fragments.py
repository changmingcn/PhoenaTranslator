"""Reference-entry fragments cloned from a block holding several bibliography entries.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import fitz

from phoena_translator.pdf.model import (
    PARAGRAPH_MAX_HANGING_SHARE,
    PARAGRAPH_MAX_MARGIN_SHARE,
    ElementType,
)
from phoena_translator.pdf.semantics.references import (
    _make_pdf_reference_fragment_from_source_lines,
)

PAGE = fitz.Rect(0, 0, 612, 792)


def _parent(**extra):
    return {
        "type": ElementType.TEXT,
        "content": "several entries",
        "bbox": [72.0, 100.0, 540.0, 200.0],
        "fontsize": 10.0,
        "layout_class": "body",
        "strong_semantic_continuation_merged": True,
        "reference_entry_merged": 3,
        "skip_translate_reason": "stale",
        "table_hint": True,
        "preserve_source_style": True,
        **extra,
    }


def _line(plain, x0, y0, x1, y1, rich=None):
    line = {"plain": plain, "bbox": [x0, y0, x1, y1]}
    if rich is not None:
        line["rich"] = rich
    return line


def test_hanging_indent_entry_becomes_its_own_paragraph():
    parent = _parent(
        inline_math_fragments=[{"text": "β"}, {"text": "β"}, {"text": "γ"}],
        superscript_scale=0.55,
    )
    lines = [
        _line("Smith, J. (2020). Dealer β", 72, 100, 540, 112, rich="<i>Smith, J.</i> (2020). Dealer β"),
        _line(" ", 72, 112, 540, 113),
        _line("balance sheets.<sup>3</sup>", 90, 113, 400, 125),
    ]
    fragment = _make_pdf_reference_fragment_from_source_lines(parent, lines, PAGE)

    assert fragment["content"] == "Smith, J. (2020). Dealer β balance sheets.3"
    assert fragment["rich_content"] == (
        "<i>Smith, J.</i> (2020). Dealer β balance sheets.<sup>3</sup>"
    )
    assert fragment["bbox"] == [72.0, 100.0, 540.0, 125.0]
    assert (fragment["x"], fragment["y"]) == (72.0, 100.0)
    assert fragment["superscript_runs"] == [
        {"text": "3", "scale": 0.55, "source": "reference-entry-split"}
    ]
    assert fragment["superscript_scale"] == 0.55
    assert fragment["inline_math_fragments"] == [{"text": "β"}]
    assert fragment["merged_visual_line_count"] == 2
    (paragraph,) = fragment["paragraphs"]
    assert paragraph["margin_left"] == 18.0 and paragraph["text_indent"] == -18.0
    assert paragraph["source_line_bboxes"] == [
        [72.0, 100.0, 540.0, 112.0],
        [90.0, 113.0, 400.0, 125.0],
    ]
    assert fragment["source_line_bboxes"] == paragraph["source_line_bboxes"]
    assert paragraph["text_align"] == "left" and paragraph["toc_leader"] is None
    for stale in ("strong_semantic_continuation_merged", "reference_entry_merged", "skip_translate_reason"):
        assert stale not in fragment
    assert fragment["reference_entry_split"] is True
    assert fragment["table_hint"] is False and fragment["preserve_source_style"] is False
    assert fragment["layout_class"] == "body"
    # The parent is not modified.
    assert parent["reference_entry_merged"] == 3


def test_flush_entry_keeps_zero_indents_and_drops_plain_rich_text():
    lines = [
        _line("Jones, K. (2019). Market depth", 72, 100, 540, 112),
        _line("and liquidity provision.", 73, 112, 300, 124),
    ]
    fragment = _make_pdf_reference_fragment_from_source_lines(_parent(), lines, PAGE)
    (paragraph,) = fragment["paragraphs"]
    assert paragraph["margin_left"] == 0.0 and paragraph["text_indent"] == 0.0
    assert fragment["rich_content"] is None
    assert fragment["superscript_runs"] == [] and fragment["superscript_scale"] is None
    assert fragment["inline_math_fragments"] == []


def test_single_line_entry_and_indent_limits():
    (single,) = _make_pdf_reference_fragment_from_source_lines(
        _parent(fontsize=0.2), [_line("Lee, A. (2018).", 72, 100, 200, 112)], PAGE
    )["paragraphs"]
    assert single["margin_left"] == 0.0 and single["text_indent"] == 0.0

    narrow = [
        _line("Wu, B. (2017).", 72, 100, 120, 112),
        _line("Title", 110, 112, 130, 124),
    ]
    (paragraph,) = _make_pdf_reference_fragment_from_source_lines(
        _parent(), narrow, PAGE
    )["paragraphs"]
    width = 130 - 72
    assert paragraph["margin_left"] == width * PARAGRAPH_MAX_MARGIN_SHARE
    assert paragraph["text_indent"] == -width * PARAGRAPH_MAX_HANGING_SHARE


def test_superscript_markup_variants_and_default_scale():
    lines = [
        _line("Kim, C. (2016).", 72, 100, 300, 112,
              rich='Kim, C. (2016).<SUP class="n">4,&nbsp;5</SUP><sup> </sup>'),
    ]
    fragment = _make_pdf_reference_fragment_from_source_lines(_parent(), lines, PAGE)
    assert fragment["superscript_runs"] == [
        {"text": "4,\xa05", "scale": 0.60, "source": "reference-entry-split"}
    ]
    assert fragment["rich_content"].startswith("Kim, C. (2016).<SUP")
