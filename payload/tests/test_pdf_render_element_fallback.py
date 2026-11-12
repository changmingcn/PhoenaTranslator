"""Render failures confined to the failing element: a page failure that no
figure or table takes, attributed to one element, keeps that element alone in
its exact source ink and translates the rest of the page.

Synthetic fixtures only.  Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import fitz
import pytest

from test_pdf_figure_table_fallback import (
    LONG_CELL,
    ZH,
    Document,
    _body,
    _cell,
    _formula_over_right_edge,
    _inject_render_failures,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)

from phoena_translator.pdf.errors import PDFPageRenderError
from phoena_translator.pdf.extract.signatures import _pdf_formula_region_signature
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.render import assembly


def _cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _output_chars(work_path, page_index=0):
    """(char, centre) for every visible character of an output page."""
    chars = []
    with fitz.open(str(work_path)) as output:
        for block in output[page_index].get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    for char in span["chars"]:
                        if char["c"].strip():
                            box = fitz.Rect(char["bbox"])
                            centre = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
                            chars.append((char["c"], box, centre))
    return chars


def _assert_kept_alone(document, elem):
    """The kept element's ink area holds its own source text and no drawn translation."""
    rect = fitz.Rect(elem["bbox"])
    inside = [char for char, box, centre in _output_chars(document.work_path) if box.intersects(rect)]
    assert not any(_cjk(char) for char in inside)
    assert "".join(inside).replace(" ", "").find(elem["content"].replace(" ", "")[:8]) >= 0


def _kept_records(document):
    return [
        entry
        for entry in document.audit.get("element_source_fallbacks", [])
        if entry.get("source_ink_kept")
    ]


def _preserved_subtitle_page():
    """Cause B: a translated chart title whose ink box clips (by under a point)
    the first line of a ``formula_risk_preserved`` subtitle below it."""
    return [
        _body(60),
        _text([72, 200, 300, 211.2], "Exhibit 9: Gross issuance by month", fontsize=9.0),
        _text(
            [72, 210.5, 300, 221.0],
            "Historical issuance: same period to date",
            fontsize=8.5,
            skip_translate_reason=SkipReason.FORMULA_RISK_PRESERVED,
        ),
        _body(400),
    ]


# ---------------------------------------------------------------------------
# Cause B: a preserved element a neighbour's redaction would erase
# ---------------------------------------------------------------------------


def test_preserved_element_clipped_by_a_neighbour_is_protected_not_the_page(tmp_path, font_path):
    elements = _preserved_subtitle_page()
    translations = {0: ZH[:30], 1: "图表9：按月总发行量", 3: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    # The subtitle keeps its exact source ink; title and body are translated.
    assert text.count("Historical issuance") == 1
    assert "图表9" in _squash(text) and "Exhibit 9" not in text
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    assert "Below we show" not in text
    _assert_kept_alone(document, elements[2])
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 2, "render")
    assert "would need a redraw after neighboring redaction" in entry["reason"]
    assert "protected from neighbouring redaction" in entry["reason"]
    # Its own reason is untouched: it was preserved before rendering.
    assert elements[2]["skip_translate_reason"] == SkipReason.FORMULA_RISK_PRESERVED


def test_protection_keeps_a_deeply_clipped_preserved_line(tmp_path, font_path):
    # The title's ink box reaches 3 pt into the preserved line (under half a
    # line): unprotected, its redaction would take the line's glyphs too.
    elements = _preserved_subtitle_page()
    elements[1]["bbox"] = [72, 200, 300, 212.0]
    elements[2]["bbox"] = [72, 209.0, 300, 220.0]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:30], 1: "图表9", 3: ZH[5:30]})])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert text.count("Historical issuance: same period to date") == 1
    assert "Exhibit 9" not in text and "图表9" in _squash(text)
    _assert_kept_alone(document, elements[2])


def test_two_preserved_subtitles_each_get_one_retry(tmp_path, font_path):
    left = _preserved_subtitle_page()[1:3]
    right = []
    for elem in left:
        x0, y0, x1, y1 = elem["bbox"]
        right.append(dict(elem, bbox=[x0 + 260, y0, x1 + 260, y1]))
    elements = [_body(60), *left, *right]
    translations = {0: ZH[:30], 1: "图表9", 3: "图表10"}
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert [entry["element"] for entry in _kept_records(document)] == [2, 4]
    assert "图表9" in _squash(text) and "图表10" in _squash(text)


# ---------------------------------------------------------------------------
# Cause C: a text box that fits at no scale and whose source restore fails
# ---------------------------------------------------------------------------


def _unplaceable_cell_page():
    """A small cell outside any figure or table (no title, no table region):
    too narrow for its translation and, with a protected formula over its
    right edge, for its own re-typeset source text."""
    elements = [
        _body(100),
        _cell(330, 210, LONG_CELL, width=24.0, table_hint=True),
        _cell(72, 210, "Ticker"),
        _body(400),
    ]
    elements.append(_formula_over_right_edge(elements[1]))
    return elements


def test_cell_that_fits_nowhere_is_kept_alone_in_source_ink(tmp_path, font_path):
    elements = _unplaceable_cell_page()
    translations = {0: ZH[:30], 1: ZH * 4, 2: "代码", 3: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])
    with fitz.open(str(document.source_path)) as source:
        formula_before = _pdf_formula_region_signature(source[0], elements[4]["bbox"])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert document.figure_tables() == []
    assert "代码" in text and ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    assert ZH * 4 not in _squash(text)
    assert _squash(text).count(_squash(LONG_CELL)[:10]) == 1
    _assert_kept_alone(document, elements[1])
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert "textbox fallback failed" in entry["reason"]
    assert "translation withdrawn" in entry["reason"]
    assert elements[1]["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert document.audit["untranslated_english_leaks"] == []
    # The protected formula region is untouched (text, fonts, geometry).
    with fitz.open(str(document.work_path)) as output:
        formula_after = _pdf_formula_region_signature(output[0], elements[4]["bbox"])
    for key in ("text_sha256", "font_sha256", "geometry_sha256", "span_count"):
        assert formula_after[key] == formula_before[key]


def test_reassembled_page_has_no_stale_attempt_events(tmp_path, font_path, monkeypatch):
    elements = [_body(100), _text([72, 200, 300, 210], "Analyst line"), _body(400)]
    translations = {0: ZH[:30], 1: "分析师", 2: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])
    _inject_render_failures(monkeypatch, {(0, 2)})

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert _squash(text).count(ZH[:30]) == 1 and "分析师" in text
    placements = [event["element"] for event in document.audit["htmlbox_placement_events"]]
    assert placements.count(0) == 1 and placements.count(1) == 1 and 2 not in placements
    assert [entry["element"] for entry in document.audit["element_source_fallbacks"]] == [2]


# ---------------------------------------------------------------------------
# Failures that are not confined: the page fallback stays
# ---------------------------------------------------------------------------


def test_element_cap_ends_in_the_page_fallback_with_clean_state(tmp_path, font_path, monkeypatch):
    elements = [_body(100), _body(200), _body(300)]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 1: ZH[3:20], 2: ZH[6:20]})])
    monkeypatch.setattr(assembly, "ELEMENT_SOURCE_MAX_PER_PAGE", 1)
    _inject_render_failures(monkeypatch, {(0, 1), (0, 2)})

    (text,) = document.assemble()

    assert document.fallback_pages() == [1]
    assert text.count("Below we show") == 3
    assert document.audit.get("element_source_fallbacks", []) == []
    assert all("skip_translate_reason" not in elem for elem in elements)


def test_a_kept_element_that_still_fails_ends_the_loop(tmp_path, font_path, monkeypatch):
    elements = _preserved_subtitle_page()
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 1: "图表9"})])
    attempts = []

    def overlap_redraw(page_num, elements, removal):
        attempts.append(page_num)
        raise PDFPageRenderError(page_num + 1, 2, "injected redraw failure")

    monkeypatch.setattr(assembly, "_overlap_redraw_indices", overlap_redraw)

    document.assemble()

    assert attempts == [0, 0]
    assert document.fallback_pages() == [1]
    assert _kept_records(document) == []


def test_watermark_is_never_kept_alone(tmp_path, font_path, monkeypatch):
    mark = _text([72, 300, 300, 312], "DRAFT", skip_translate_reason=SkipReason.WATERMARK)
    elements = [_body(100), mark]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20]})])
    real = assembly._render_element

    def render(context, plan, css, index):
        if index == 0:
            raise PDFPageRenderError(1, 1, "injected watermark failure")
        return real(context, plan, css, index)

    monkeypatch.setattr(assembly, "_render_element", render)

    document.assemble()

    assert document.fallback_pages() == [1]


def test_confined_page_pulled_back_by_a_later_merge_keeps_no_record(
    tmp_path, font_path, monkeypatch
):
    page_one = [_body(100), _text([72, 200, 300, 210], "Analyst line"), _body(400)]
    page_two = [_body(100), _body(200)]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 2,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [(page_one, {0: ZH[:20], 1: "分析师", 2: ZH[3:20]}), (page_two, {0: ZH[6:20], 1: ZH[9:20]})],
        merge_decisions=[merge],
    )
    # Page 1 keeps element 1 alone; page 2 then fails on its merge endpoint.
    _inject_render_failures(monkeypatch, {(0, 1), (1, 0)})

    first, _second = document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert "分析师" not in first
    assert _kept_records(document) == []
    assert page_one[1].get("skip_translate_reason") != SkipReason.RENDER_ELEMENT_SOURCE


def test_invariant_refuses_a_kept_element_that_would_still_be_redrawn(
    tmp_path, font_path, monkeypatch
):
    # Safety net: were the kept element still a redraw candidate (its view
    # skipped), the attempt is refused and the page falls back.
    elements = [
        _body(60),
        _text([72, 200, 300, 211.2], "First line of a note"),
        _text([72, 210.5, 300, 221.0], "Second line of a note"),
    ]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 1: "注一", 2: "注二"})])
    monkeypatch.setattr(assembly, "_without_kept_source_elements", lambda elements, kept: elements)
    _inject_render_failures(monkeypatch, {(0, 2)})

    document.assemble()

    assert document.fallback_pages() == [1]
    assert "would still be redacted or redrawn" in document.audit["source_page_fallbacks"][0]["reason"]
    assert _kept_records(document) == []


def test_whiteout_after_failed_redaction_does_not_confine(tmp_path, font_path, monkeypatch):
    # With the whiteout fallback the neighbour's source text stays in the
    # text layer beside the kept ink, and whiteout paints whole ink boxes:
    # an attempt that needed it is refused, the page falls back.
    elements = _preserved_subtitle_page()
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 1: "图表9", 3: ZH[5:20]})])

    def broken(*_args, **_kwargs):
        raise RuntimeError("injected redaction failure")

    monkeypatch.setattr(fitz.Page, "apply_redactions", broken)

    document.assemble()

    assert document.fallback_pages() == [1]
    assert "whiteout path was taken" in document.audit["source_page_fallbacks"][0]["reason"]
    assert _kept_records(document) == []


# ---------------------------------------------------------------------------
# The neighbour-ink guard after redaction
# ---------------------------------------------------------------------------


def _plan(document, kept, render, protected=()):
    elements = document.pages[0][0]
    out_page = document.context.out_doc[0]
    return assembly.PDFPageAssemblyPlan(
        page_num=0,
        out_page=out_page,
        elements=elements,
        translations={},
        layout_styles={},
        watermark_indices=set(),
        protected_formula_rects=[fitz.Rect(rect) for rect in protected],
        translated_indices=set(render),
        redraw_source_indices=set(),
        render_indices=set(render),
        source_elements=frozenset(kept),
    )


def test_guard_refuses_a_neighbour_glyph_left_beside_the_kept_ink(tmp_path, font_path):
    elements = _preserved_subtitle_page()
    document = Document(tmp_path, font_path, [(elements, {})])
    # Nothing redacted: the title's glyphs are all still there.
    plan = _plan(document, kept={2}, render={1})
    with pytest.raises(PDFPageRenderError, match="source glyph of neighbour elem 1") as raised:
        assembly._check_kept_source_neighbours(document.context, plan)
    assert raised.value.element_source_refused is True


def test_guard_refuses_a_neighbour_sharing_a_line_with_the_kept_ink(tmp_path, font_path):
    elements = _preserved_subtitle_page()
    elements[2]["bbox"] = [72, 204, 300, 214.5]  # overlaps the title by 7 of 11 pt
    document = Document(tmp_path, font_path, [(elements, {})])
    plan = _plan(document, kept={2}, render={1})
    with pytest.raises(PDFPageRenderError, match="shares a line with the kept ink"):
        assembly._check_kept_source_neighbours(document.context, plan)


def test_guard_accepts_the_protected_redaction(tmp_path, font_path):
    elements = _preserved_subtitle_page()
    document = Document(tmp_path, font_path, [(elements, {1: "图表9"})])
    plan = _plan(
        document,
        kept={2},
        render={1},
        protected=assembly._kept_source_ink_rects(elements[2]),
    )
    assembly._redact_page(document.context, plan)
    assembly._check_kept_source_neighbours(document.context, plan)  # no raise
    page_text = plan.out_page.get_text()
    assert "Historical issuance" in page_text and "Exhibit 9" not in page_text



# --- review round 1 ----------------------------------------------------------


def _page_pixels(path, clip=None):
    with fitz.open(str(path)) as doc:
        return doc[0].get_pixmap(dpi=72, clip=clip, alpha=False).samples


def test_a_page_with_every_translation_withdrawn_is_a_source_page(tmp_path, font_path, monkeypatch):
    # R4: confinement keeps both text elements; nothing translated is drawn.
    document = Document(tmp_path, font_path, [([_body(100), _body(300)], {0: ZH[:30], 1: ZH[5:30]})])
    _inject_render_failures(monkeypatch, {(0, 0), (0, 1)})

    (text,) = document.assemble()

    assert document.fallback_pages() == [1]
    assert _kept_records(document) == []
    assert document.audit.get("element_source_fallbacks", []) == []
    assert not _cjk(text)
    assert _page_pixels(document.work_path) == _page_pixels(document.source_path)


def test_a_page_whose_only_translation_is_withdrawn_at_assembly_is_a_source_page(tmp_path, font_path):
    # R4 through the safety net: the page's one translation holds a raw token.
    document = Document(
        tmp_path, font_path, [([_body(100)], {0: f"{ZH[:20]}PHOENA_IMATH_0000_0123456789ABCDEF"})]
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == [1]
    assert _kept_records(document) == []
    assert "PHOENA" not in text
    assert _page_pixels(document.work_path) == _page_pixels(document.source_path)


# --- review round 2 (T5) -----------------------------------------------------
# The class: a page that had translation candidates and draws none of them
# after retention of any kind (a retained group, single kept elements, no
# formula-safe place, or several together) is its source page.


def _withdrawn_table_page(first_translation=None):
    from test_pdf_figure_table_fallback import _region

    elements = [
        _body(80),
        _region([60, 200, 560, 260]),
        _cell(72, 210, "Ticker", table_hint=True),
        _cell(200, 210, "GA AUM", table_hint=True),
        # Too small for its translation and for its own source text: the
        # whole table stays in source.
        _cell(330, 210, LONG_CELL, width=24.0, table_hint=True),
        _cell(72, 230, "MET", table_hint=True),
        _body(400),
    ]
    elements.append(_formula_over_right_edge(elements[4]))
    translations = {
        0: first_translation or elements[0]["content"],
        2: "代码",
        3: "资产",
        4: ZH * 4,
        5: "大都会",
        6: elements[6]["content"],
    }
    return elements, translations


@pytest.mark.parametrize(
    "first_translation",
    [None, f"{ZH[:20]}PHOENA_IMATH_0000_0123456789ABCDEF"],
    ids=["group-only", "group-and-element"],
)
def test_a_page_withdrawn_by_group_retention_is_a_source_page(
    tmp_path, font_path, first_translation
):
    elements, translations = _withdrawn_table_page(first_translation)
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == [1]
    assert document.figure_tables() == []
    assert document.audit.get("element_source_fallbacks", []) == []
    assert not _cjk(text) and "PHOENA" not in text
    assert _page_pixels(document.work_path) == _page_pixels(document.source_path)


def test_a_page_withdrawn_by_preflight_retention_is_a_source_page(font_path):
    from test_pdf_assembly import Page

    elem = _text([72, 100, 300, 114], "Unplaceable source text")
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [40, 60, 580, 300]}
    page = Page(font_path, [elem, formula], {0: ZH[:8]})

    with pytest.raises(PDFPageRenderError, match="every translation on the page was withdrawn"):
        assembly._prepare_page(page.context, 0)
    assert "skip_translate_reason" not in elem
    assert page.audit.get("element_source_fallbacks", []) == []


def test_a_confined_page_that_needed_whiteout_is_restored_and_keeps_its_graphic(
    tmp_path, font_path, monkeypatch
):
    # R3: redaction fails, so the whiteout path paints the translated element's
    # ink box, erasing a background graphic away from the kept element.
    import shutil

    document = Document(tmp_path, font_path, [([_body(100), _body(300)], {0: ZH[:30], 1: ZH[5:30]})])
    document.context.out_doc.close()
    with fitz.open(str(document.source_path)) as source:
        source[0].draw_rect(fitz.Rect(65, 95, 550, 150), color=None, fill=(0.5, 0.7, 0.9), overlay=False)
        source.saveIncr()
    shutil.copy2(document.source_path, document.work_path)
    document.context.out_doc = fitz.open(str(document.work_path))

    def failing_redaction(*_args, **_kwargs):
        raise RuntimeError("injected redaction failure")

    monkeypatch.setattr(fitz.Page, "apply_redactions", failing_redaction)
    _inject_render_failures(monkeypatch, {(0, 1)})
    graphic = fitz.Rect(500, 102, 510, 108)

    document.assemble()

    assert document.fallback_pages() == [1]
    assert _kept_records(document) == []
    assert _page_pixels(document.work_path, graphic) == _page_pixels(document.source_path, graphic)
    assert min(_page_pixels(document.work_path, graphic)) < 250  # the graphic is there
