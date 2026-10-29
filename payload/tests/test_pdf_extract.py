"""PDF extraction: native lines, superscripts, signatures, OCR gates and the extraction stage.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging

import pytest
from review_support import (
    _policy_context,
    _policy_result,
    _policy_test_elements,
    _single_formula_page,
)

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# B3 — a page-level get_text failure must propagate (recovery owns it)
# ---------------------------------------------------------------------------


class _BrokenPage:
    number = 0

    def get_text(self, *args, **kwargs):
        raise RuntimeError("mupdf page decode failed")


class _FakePDFTable:
    def __init__(self, bbox, rows, columns, extracted):
        self.bbox = bbox
        self.row_count = rows
        self.col_count = columns
        self._extracted = extracted

    def extract(self):
        return self._extracted


def _glyph_span(text: str, x0: float, x1: float, y0: float = 86.83, color: int = 0x231F20):
    return {
        "text": text,
        "font": "NewBaskervilleStd-Roman",
        "size": 10.0,
        "color": color,
        "bbox": (x0, y0, x1, y0 + 10.0),
    }


def test_pdf_text_extraction_failure_propagates():
    from phoena_translator.pdf.extract.page import _extract_pdf_text_elements

    with pytest.raises(RuntimeError, match="mupdf page decode failed"):
        _extract_pdf_text_elements(_BrokenPage(), [])


class TestBuildParagraphsEmptyInput:
    def test_returns_declared_tuple_shape(self):
        import fitz

        from phoena_translator.pdf.extract.paragraphs import _build_pdf_paragraphs

        assert _build_pdf_paragraphs([], fitz.Rect(0, 0, 10, 10), 10.0) == ([], 0.0)


def test_pdf_bold_run_in_sentence_wrapping_into_mixed_line_stays_one_segment():
    from phoena_translator.pdf.extract.page import _split_pdf_heading_segments

    lines = [
        {
            "plain": (
                "The continued growth in China's surplus and how it is recycled "
                "matters for competing export"
            ),
            "rich": (
                "<b>The continued growth in China's surplus and how it is recycled "
                "matters for competing export</b>"
            ),
            "bold": True,
            "fontsize": 9.96,
            "x0": 72.0,
            "x1": 542.1,
            "y0": 71.85,
            "y1": 84.74,
        },
        {
            "plain": (
                "economies and the assets China saves in. Market-share gains "
                "continue to pressure trading partners."
            ),
            "rich": (
                "<b>economies and the assets China saves in.</b> Market-share "
                "gains continue to pressure trading partners."
            ),
            "bold": False,
            "fontsize": 9.96,
            "x0": 72.0,
            "x1": 542.2,
            "y0": 83.86,
            "y1": 96.75,
        },
    ]

    assert _split_pdf_heading_segments(lines) == [lines]


def test_pdf_standalone_bold_heading_still_splits_from_regular_body():
    from phoena_translator.pdf.extract.page import _split_pdf_heading_segments

    heading = {
        "plain": "Economic Outlook",
        "rich": "<b>Economic Outlook</b>",
        "bold": True,
        "fontsize": 10.0,
        "x0": 72.0,
        "x1": 190.0,
        "y0": 72.0,
        "y1": 84.0,
    }
    body = {
        "plain": "Growth is expected to remain resilient over the coming year.",
        "rich": "Growth is expected to remain resilient over the coming year.",
        "bold": False,
        "fontsize": 10.0,
        "x0": 72.0,
        "x1": 490.0,
        "y0": 84.0,
        "y1": 96.0,
    }

    assert _split_pdf_heading_segments([heading, body]) == [[heading], [body]]


def test_pdf_mixed_visual_line_keeps_inline_bold_markup():
    import fitz

    from phoena_translator.pdf.extract.paragraphs import (
        _make_pdf_text_element_from_lines,
    )

    line = {
        "plain": "Bold opening sentence. Regular body follows.",
        "rich": "<b>Bold opening sentence.</b> Regular body follows.",
        "x0": 72.0,
        "x1": 330.0,
        "y0": 72.0,
        "y1": 84.0,
        "fontsize": 10.0,
        "line_height": 12.0,
        "color": 0,
        "bold": False,
        "non_horizontal": False,
        "rotation": 0,
        "superscript_runs": [],
        "inline_math_fragments": [],
    }

    element = _make_pdf_text_element_from_lines(
        [line],
        fitz.Rect(0.0, 0.0, 612.0, 792.0),
    )

    assert element is not None
    assert element["rich_content"] == line["rich"]


def test_two_contexts_hold_independent_table_policies(tmp_path):
    from phoena_translator.pdf.extract.stage import (
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


def test_native_superscript_flag_does_not_turn_same_baseline_prose_into_markup():
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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
    from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans

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


def test_pdf_sparse_two_column_wrapper_yields_to_nested_real_table():
    """BOJ p1: body prose beside Chart 1 is not one giant 1x2 table."""
    from phoena_translator.pdf.extract.tables import _prune_sparse_pdf_table_wrappers

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

    from phoena_translator.pdf.extract.baselines import (
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

    from phoena_translator.pdf.extract.baselines import (
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


def test_formula_signature_survives_a_rotated_page():
    from phoena_translator.pdf.extract.signatures import _pdf_formula_region_signature

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


def test_ink_stroked_twice_in_place_is_collapsed_to_one_span():
    from phoena_translator.pdf.extract.page import _drop_pdf_duplicate_glyph_spans

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
    from phoena_translator.pdf.extract.page import _drop_pdf_duplicate_glyph_spans

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
    from phoena_translator.pdf.extract.page import _drop_pdf_duplicate_glyph_spans

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
    from phoena_translator.pdf.extract.math_detection import (
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
    from phoena_translator.pdf.extract.page import _drop_pdf_duplicate_glyph_spans

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
