"""Split-layout cells: one element per visually separated cell of a PDF line.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import fitz

from phoena_translator.pdf.extract.page import _make_pdf_split_layout_element
from phoena_translator.pdf.model import ElementType

PAGE = fitz.Rect(0, 0, 612, 792)


def _span(text, x0, x1, *, y0=100.0, y1=110.0, size=9.0, color=0, bold=False, rich=None, **extra):
    return {
        "text": text,
        "rich": rich if rich is not None else text,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "size": size,
        "color": color,
        "bold": bold,
        **extra,
    }


def _cell(cluster, x0, x1, align="left"):
    return {"cluster": cluster, "x0": x0, "x1": x1, "align": align}


def _build(cluster, *, align="left", cell_x=(300.0, 360.0), line_bbox=(72, 98, 540, 112),
           direction=(1, 0), in_table=False, non_horizontal=False, math=None, fontsize=9.5):
    return _make_pdf_split_layout_element(
        page_rect=PAGE,
        line={"bbox": list(line_bbox), "dir": direction},
        cell=_cell(cluster, *cell_x, align=align),
        line_fontsize=fontsize,
        line_in_table=in_table,
        line_non_horizontal=non_horizontal,
        cluster_math=math or {},
    )


def test_blank_cell_produces_no_element():
    assert _build([_span("  ", 300, 310), _span("\n", 310, 320)]) is None


def test_cell_geometry_style_and_flags():
    cluster = [
        _span("Total ", 300, 330, size=9.0, color=0x112233),
        _span("12.3", 330, 360, y0=99.0, y1=111.0, size=9.0, color=0x112233),
        _span(" %", 360, 365, size=8.0, color=0),
    ]
    elem = _build(
        cluster,
        align="right",
        cell_x=(300.0, 365.0),
        direction=(0, -1),
        in_table=True,
        math={"inline_math_fragments": [{"text": "%"}]},
    )
    assert elem["type"] == ElementType.TEXT
    assert elem["content"] == "Total 12.3 %"
    assert elem["bbox"] == [300.0, 99.0, 365.0, 111.0]
    assert elem["rect"] == fitz.Rect(300, 99, 365, 111)
    assert (elem["x"], elem["y"]) == (300.0, 99.0)
    # A short right-aligned nowrap cell may extend left into free space.
    assert elem["render_bbox"][0] < 300.0 and elem["render_bbox"][2] == 365.0
    assert elem["rich_content"] is None and elem["bold"] is False
    assert elem["superscript_runs"] == [] and elem["superscript_scale"] is None
    assert elem["fontsize"] == 9.0 and elem["color"] == 0x112233
    assert elem["top_padding"] == 1.0
    assert elem["line_height"] == 12.0
    assert elem["rotation"] == 90 and elem["non_horizontal"] is False
    assert elem["preserve_source_style"] is True and elem["table_hint"] is True
    assert elem["inline_math_fragments"] == [{"text": "%"}]
    (paragraph,) = elem["paragraphs"]
    assert paragraph == {
        "plain": "Total 12.3 %",
        "rich": "Total 12.3 %",
        "margin_left": 0.0,
        "text_indent": 0.0,
        "gap_before": 0.0,
        "text_align": "right",
        "nowrap": True,
    }


def test_line_height_never_drops_below_the_line_font():
    elem = _build([_span("Rate", 300, 330, y0=100.0, y1=104.0)], fontsize=10.0)
    assert elem["line_height"] == 10.5
    assert elem["top_padding"] == 2.0


def test_superscripts_keep_rich_text_and_their_dominant_scale():
    cluster = [
        _span("Yield", 300, 330, rich="Yield"),
        _span("12", 330, 336, rich="<sup>12</sup>", superscript=True, superscript_scale=0.55,
              superscript_source="font"),
        _span("a", 336, 340, rich="<sup>a</sup>", superscript=True),
        _span(" ", 340, 342, superscript=True),
    ]
    elem = _build(cluster, non_horizontal=True)
    assert elem["content"] == "Yield12a"
    assert elem["rich_content"] == "Yield<sup>12</sup><sup>a</sup>"
    assert elem["superscript_runs"] == [
        {"text": "12", "scale": 0.55, "source": "font"},
        {"text": "a", "scale": 0.60, "source": None},
    ]
    assert elem["superscript_scale"] == 0.55
    assert elem["non_horizontal"] is True and elem["rotation"] == 0


def test_partly_bold_cells_keep_markup_and_mostly_bold_cells_are_bold():
    partly = _build([
        _span("Net ", 300, 320, bold=True, rich="<b>Net </b>"),
        _span("flow of funds", 320, 360),
    ])
    assert partly["rich_content"] == "<b>Net </b>flow of funds" and partly["bold"] is False
    mostly = _build([
        _span("Net flow of ", 300, 340, bold=True, rich="<b>Net flow of </b>"),
        _span("funds", 340, 360),
    ])
    assert mostly["rich_content"] == "<b>Net flow of </b>funds" and mostly["bold"] is True
    entirely = _build([_span("Net flow", 300, 340, bold=True, rich="<b>Net flow</b>")])
    assert entirely["rich_content"] is None and entirely["bold"] is True
