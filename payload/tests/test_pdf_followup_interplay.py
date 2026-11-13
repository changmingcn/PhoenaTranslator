"""Interplay of the sign check, the review leftovers and the stacked-cell /
invisible-text handling on one page.

Each test puts two of the new behaviours next to each other: an element kept
for a sign that lost its amount beside a stacked table cell, the by-page
cleanup of a source page meeting the records both add, an invisible element
beside a kept-source element, and the per-page limit the two nets share.
All fixtures are synthetic.
"""

from __future__ import annotations

import fitz

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_invisible_source_text import (
    LOGO,
    LOGO_COLOUR,
    _logo_pixel,
    _visible_letter_spans,
)
from test_pdf_placeholder_token_leak import _leaking_cached_page, _minted_tokens
from test_pdf_prose_number_guard import FAITHFUL, LOSSY, PROSE, _signs
from test_pdf_render_element_fallback import _kept_records
from test_pdf_stacked_cells import STACKED_BLOCK, STACKED_LINES, _stacked

from phoena_translator.pdf.model import SkipReason
from phoena_translator.pdf.render import assembly
from phoena_translator.pdf.translate.numeric_guard import (
    PDF_LABEL_NUMBERS_REASON,
    PDF_NUMERIC_ROW_REASON,
    PDF_PROSE_NUMBERS_REASON,
)

NUMERIC_REASONS = (PDF_NUMERIC_ROW_REASON, PDF_LABEL_NUMBERS_REASON, PDF_PROSE_NUMBERS_REASON)


def _records_by_element(document, page=1):
    entries = [
        entry
        for entry in document.audit.get("element_source_fallbacks", [])
        if int(entry["page"]) == page
    ]
    elements = [int(entry["element"]) for entry in entries]
    assert len(elements) == len(set(elements)), entries  # nobody recorded twice
    return {int(entry["element"]): entry for entry in entries}


def _prose_and_stacked_page(stacked_translation=STACKED_BLOCK):
    prose = _text([72, 200, 540, 240], PROSE, layout_class="body", inline_math_fragments=_signs("+", "+"))
    stacked = _stacked(STACKED_LINES, bands=2)
    neighbour = _text([300, 646, 400, 654], "Rising star", fontsize=8.0, layout_class="table")
    elements = [_body(100), prose, _body(400), stacked, neighbour]
    translations = {0: ZH[:30], 1: LOSSY, 2: ZH[5:30], 3: stacked_translation, 4: "新晋升级"}
    return elements, translations, prose, stacked


# --- 1. a prose-number element and a stacked cell on one page --------------


def test_a_prose_number_element_and_a_stacked_cell_each_keep_their_ink_once(
    tmp_path, font_path
):
    elements, translations, prose, stacked = _prose_and_stacked_page()
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    squashed = _squash(text)
    assert "机床订单" not in squashed and "评级票据" not in squashed
    assert ZH[:30] in squashed and "新晋升级" in squashed
    records = _records_by_element(document)
    assert sorted(records) == [1, 3]
    assert records[1]["reason"].startswith(PDF_PROSE_NUMBERS_REASON)
    assert records[3]["reason"].startswith(assembly.PDF_STACKED_TABLE_CELLS_REASON)
    assert all(entry["source_ink_kept"] for entry in records.values())
    assert prose["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert stacked["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert "stacked_table_cell_waivers" not in document.audit


def test_a_stacked_cell_that_also_lost_its_numbers_is_recorded_once_by_the_numeric_net(
    tmp_path, font_path
):
    # The numeric net runs first; the stacked net only sees what is still drawn.
    elements, translations, _prose, stacked = _prose_and_stacked_page("流入 评级票据")
    document = Document(tmp_path, font_path, [(elements, translations)])

    document.assemble()

    assert document.fallback_pages() == []
    records = _records_by_element(document)
    assert sorted(records) == [1, 3]
    assert records[3]["reason"].startswith(NUMERIC_REASONS)
    assert stacked["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE


# --- 2. a page restored whole keeps none of the new records ----------------


def test_a_page_pulled_back_by_a_merge_closure_keeps_no_prose_or_stacked_record(
    tmp_path, font_path
):
    elements, translations, _prose, _stacked_elem = _prose_and_stacked_page()
    leaking, _kept = _leaking_cached_page()
    token = _minted_tokens()["SUP"]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 2,
        "destination_page": 2,
        "destination_element": 1,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (elements, translations),
            (leaking, {0: ZH[:30], 1: f"进一步{token}表明", 2: ZH[5:30]}),
            ([_body(100), _body(300)], {0: ZH[3:30], 1: ZH[7:30]}),
        ],
        merge_decisions=[merge],
    )

    document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert _records_by_element(document, page=1) == {}
    assert _records_by_element(document, page=2) == {}
    assert "figure_table_source_fallbacks" not in document.audit


def test_a_page_whose_only_translation_lost_its_numbers_is_delivered_with_a_warning(
    tmp_path, font_path
):
    """R-19: the sign check never restores a page.  The page's only drawn
    translation cannot keep its source alone: it is drawn as the live
    release draws it, and its loss is one delivered warning."""
    prose = _text([72, 200, 540, 240], PROSE, layout_class="body", inline_math_fragments=_signs("+", "+"))
    document = Document(
        tmp_path,
        font_path,
        [([prose], {0: LOSSY}), ([_body(100)], {0: ZH[:30]})],
    )

    (text, _second) = document.assemble()

    assert document.fallback_pages() == []
    assert "机床订单" in _squash(text)
    assert _records_by_element(document, page=1) == {}
    (warning,) = document.audit["delivered_element_warnings"]
    assert (warning["page"], warning["element"], warning["stage"]) == (1, 0, "render")
    assert warning["reason"] == PDF_PROSE_NUMBERS_REASON


# --- 4. an invisible element next to a kept-source element ------------------


def _logo_document_with_a_kept_neighbour(tmp_path, font_path):
    letter = _text([42.0, -13.3, 287.8, 94.0], "M", fontsize=107.0, layout_class="scattered")
    letter["skip_translate_reason"] = SkipReason.INVISIBLE_SOURCE
    # The date line touches the letter's box and is translated (the redraw
    # trigger); the note beside it loses its amount: kept in its source ink.
    date_line = _text([200, 84, 290, 93], "9 April 2030", fontsize=9.0, layout_class="scattered")
    note = _text(
        [300, 84, 420, 93],
        "Issued at ~3.5% by the desk",
        fontsize=9.0,
        layout_class="scattered",
        inline_math_fragments=_signs("~"),
    )
    title = _text([72, 120, 540, 135], "A translated page title", fontsize=12.0, layout_class="heading")
    document = Document(
        tmp_path,
        font_path,
        [
            (
                [letter, date_line, note, title],
                {0: "M", 1: "2030年4月9日", 2: "由交易台以~发布", 3: ZH[:12]},
            )
        ],
    )
    document.context.out_doc.close()
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    shape = page.new_shape()
    shape.draw_rect(LOGO)
    shape.finish(fill=LOGO_COLOUR, color=None)
    shape.commit()
    page.insert_text((60, 92), "M", fontsize=100, fill_opacity=0, color=(0.4, 0.45, 0.7))
    page.insert_text((200, 92), "9 April 2030", fontsize=9)
    page.insert_text((300, 92), "Issued at ~3.5% by the desk", fontsize=9)
    page.insert_text((72, 134), "A translated page title", fontsize=12)
    source.save(str(document.source_path))
    source.save(str(document.work_path))
    source.close()
    document.context.out_doc = fitz.open(str(document.work_path))
    return document, letter, note


def test_an_invisible_element_beside_a_kept_source_element_is_neither_redrawn_nor_painted(
    tmp_path, font_path
):
    document, letter, note = _logo_document_with_a_kept_neighbour(tmp_path, font_path)

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert ZH[:12] in _squash(text) and "2030年4月9日" in _squash(text)
    records = _records_by_element(document)
    assert sorted(records) == [2]
    assert records[2]["reason"].startswith(PDF_PROSE_NUMBERS_REASON)
    assert note["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert letter["skip_translate_reason"] == SkipReason.INVISIBLE_SOURCE
    assert _visible_letter_spans(document.work_path) == []
    assert [
        event
        for event in document.audit.get("render_scale_events", [])
        if event.get("element") == 0
    ] == []
    red, _green, blue = _logo_pixel(document.work_path, fitz.Point(60, 50))
    assert blue > 120 and red < 100
    with fitz.open(str(document.work_path)) as output:
        assert "Issued at ~3.5%" in output[0].get_text()


# --- 5. the per-page limit of kept elements ---------------------------------


def test_the_two_nets_share_the_per_page_limit(tmp_path, font_path):
    """R-19: sign-check keeps take the stacked-cell path, so they share its
    ``ELEMENT_SOURCE_MAX_PER_PAGE`` room in element order; beyond it each net
    leaves its element to today's path with its own record."""
    limit = assembly.ELEMENT_SOURCE_MAX_PER_PAGE
    lossy = [
        _text(
            [72, 40 + 28 * n, 540, 60 + 28 * n],
            f"Sales rose +${n + 3}.5bn in the quarter.",
            layout_class="body",
            inline_math_fragments=_signs("+"),
        )
        for n in range(limit + 1)
    ]
    stacked = _stacked(STACKED_LINES, bands=2)
    tail = _text([72, 720, 540, 740], "A closing remark without figures.", layout_class="body")
    elements = [*lossy, stacked, tail]
    translations = {n: "本季度销售上升+。" for n in range(limit + 1)}
    translations.update({limit + 1: STACKED_BLOCK, limit + 2: "结束语。"})
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    records = _records_by_element(document)
    assert sorted(records) == list(range(limit))
    assert all(records[n]["reason"].startswith(PDF_PROSE_NUMBERS_REASON) for n in range(limit))
    assert [w["element"] for w in document.audit["delivered_element_warnings"]] == [limit]
    assert [w["element"] for w in document.audit["stacked_table_cell_waivers"]] == [limit + 1]
    assert "结束语" in _squash(text)


def test_faithful_prose_beside_a_stacked_cell_is_drawn(tmp_path, font_path):
    elements, translations, _prose, _stacked_elem = _prose_and_stacked_page()
    translations[1] = FAITHFUL
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert "123亿美元" in _squash(text)
    assert sorted(_records_by_element(document)) == [3]
