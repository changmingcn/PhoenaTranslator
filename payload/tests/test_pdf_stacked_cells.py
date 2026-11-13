"""Stacked table cells keep their source ink (acceptance item B2).

A table merger can coalesce the cells of one column over several table rows
into one element.  Drawn as one paragraph from the top of its box, its values
land in the first row and the rows below are left empty.  Such an element is
kept in its exact source ink (reason ``stacked-table-cells``) unless keeping
it would cost more than the element.  All fixtures are synthetic.
"""

from __future__ import annotations

import fitz
import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_render_element_fallback import _kept_records

from phoena_translator.pdf.cache import _pdf_cache_identity
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.render import assembly
from phoena_translator.pdf.semantics.tables import (
    _merge_pdf_semantic_table_cells,
    _pdf_semantic_cell_shared_row_bands,
    pdf_stacked_table_cell,
)


def _stacked(lines, *, bands=None, x0=135.0, y0=646.0, pitch=10.0, width=40.0, **extra):
    """A merged table cell whose source lines sit one under another."""
    source_lines = [
        {
            "plain": plain,
            "rich": plain,
            "bbox": [x0, y0 + pitch * row, x0 + width, y0 + pitch * row + 8.0],
        }
        for row, plain in enumerate(lines)
    ]
    bottom = y0 + pitch * (len(lines) - 1) + 8.0
    elem = _text(
        [x0, y0, x0 + width, bottom],
        " ".join(lines),
        fontsize=8.0,
        layout_class="table",
        table_hint=True,
        semantic_table_cell=len(lines) > 1,
        paragraphs=[
            {
                "plain": " ".join(lines),
                "rich": " ".join(lines),
                "source_bbox": [x0, y0, x0 + width, bottom],
                "source_line_bboxes": [line["bbox"] for line in source_lines],
                "source_lines": source_lines,
            }
        ],
        **extra,
    )
    if bands is not None:
        elem["semantic_cell_row_bands"] = bands
    return elem


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "bands", "expected"),
    [
        # label plus values: a group header, a column label and its data rows,
        # backed by separate cells of another column beside two of its lines
        (["Inflow", "Rated notes", "0.4", "0.7", "0.1", "52%"], 2, True),
        (["Total", "Inflow less outflow", "100%"], 2, True),
        # one value line under labels without that backing is one cell (R-6):
        # a tile holding a label and its value, a header with a date line
        (["Inflow", "Rated notes", "0.4"], 1, False),
        (["Loan growth", "12.5%"], 0, False),
        (["Close", "10/11/2025"], 0, False),
        (["Move", "-142 unch"], None, False),
        # two or more lines of values only are rows on their own (R-6b)
        (["Inflow", "Rated notes", "0.4", "0.7", "0.1", "52%"], 1, True),
        (["Move", "-142", "19", "unch"], None, True),
        (["0.4", "0.8", "52%"], 0, True),
        (["0.4", "0.8", "52%", "1.9"], 4, True),
        (["Spread", "+25bp", "n.a.", "-"], 0, True),
        (["1.2x", "$3.4bn"], 0, True),
        # a line with a value and a word is not values only
        (["24 months", "Median 80", "6 quarters"], 0, False),
        # a line of four-digit years alone is a header, not a value
        (["2019", "Actual"], 2, False),
        (["2019", "2020", "Actual"], 0, False),
        (["2019", "0.4"], 0, False),
        (["2019", "Actual", "2,416", "583"], 2, True),
        # word-only stacked column: the merger's record decides
        (["Stance", "Long", "Long", "Short", "Flat"], 5, True),
        (["Stance", "Long", "Long", "Short", "Flat"], 1, False),
    ],
)
def test_the_stacked_cell_predicate(lines, bands, expected):
    assert pdf_stacked_table_cell(_stacked(lines, bands=bands)) is expected


def test_a_single_line_or_an_unmerged_element_is_never_stacked():
    assert not pdf_stacked_table_cell(_stacked(["Inflow 0.4"], bands=6))
    elem = _stacked(["Inflow", "0.4", "0.8"], bands=6)
    elem["semantic_table_cell"] = False
    assert not pdf_stacked_table_cell(elem)


# ---------------------------------------------------------------------------
# The merger's record
# ---------------------------------------------------------------------------


def _cell_fragment(x, y, content):
    return {
        "type": ElementType.TEXT,
        "bbox": [x, y, x + 6.0 * len(content), y + 8.0],
        "y": y,
        "x": x,
        "content": content,
        "fontsize": 8.0,
        "table_hint": True,
        "layout_class": "table",
        "paragraphs": [
            {
                "plain": content,
                "rich": content,
                "source_lines": [
                    {"plain": content, "rich": content, "bbox": [x, y, x + 6.0 * len(content), y + 8.0]}
                ],
            }
        ],
    }


def _stacked_column_table(*, row_label_header=False):
    """A title, one large gap, then a header and data rows at one pitch: the
    inferred row plan takes header and data as one logical row, so the
    middle column's group header, its label and its values are merged.

    Without a header over the row labels they lie far from every column
    origin and stay separate cells, one per table row; with one, the row
    labels are merged into one cell as well."""
    elements = [_cell_fragment(30, 618, "Table title")]
    if row_label_header:
        elements.append(_cell_fragment(30, 635, "Sales"))
    elements += [_cell_fragment(142, 635, "Inflow"), _cell_fragment(250, 635, "Outflow")]
    rows = [("Rated notes", "Redemptions"), ("0.4", "17.6"), ("0.7", "17.6"), ("0.1", "0.0")]
    for row, (middle, right) in enumerate(rows, start=1):
        y = 635 + 10.0 * row
        elements += [
            _cell_fragment(30, y, f"Item {row}"),
            _cell_fragment(142 + (0 if row == 1 else 12), y, middle),
            _cell_fragment(250, y, right),
        ]
    return elements


def test_the_merger_records_the_bands_a_cell_shares_with_other_columns():
    page_rect = fitz.Rect(0, 0, 612, 792)
    merged, count = _merge_pdf_semantic_table_cells(
        _stacked_column_table(), [fitz.Rect(20, 610, 400, 690)], page_rect
    )
    assert count >= 1
    (middle,) = [elem for elem in merged if elem.get("content", "").startswith("Inflow")]
    assert middle["semantic_table_cell"]
    # The four data rows hold a separate row-label cell each; the header band
    # holds only the right column's cell, which is one cell over all bands.
    assert middle["semantic_cell_row_bands"] == 4
    assert pdf_stacked_table_cell(middle)


def test_with_every_column_merged_the_figure_columns_are_stacked_and_the_label_column_is_not():
    """Every column merged into one cell each: no other column shows separate
    cells beside any of them (R-6 records no band).  The columns of figures
    merged with their headings are stacks by their own value lines (R-6b);
    the column of row labels has no value line and no other-column evidence,
    so it is drawn as before."""
    page_rect = fitz.Rect(0, 0, 612, 792)
    merged, _count = _merge_pdf_semantic_table_cells(
        _stacked_column_table(row_label_header=True),
        [fitz.Rect(20, 610, 400, 690)],
        page_rect,
    )
    cells = {
        elem["content"].split()[0]: elem for elem in merged if elem.get("semantic_table_cell")
    }
    assert set(cells) == {"Sales", "Inflow", "Outflow"}
    assert all(elem["semantic_cell_row_bands"] == 0 for elem in cells.values())
    assert pdf_stacked_table_cell(cells["Inflow"])
    assert pdf_stacked_table_cell(cells["Outflow"])
    assert not pdf_stacked_table_cell(cells["Sales"])


def test_a_wrapped_header_row_records_no_row_evidence_and_is_not_stacked():
    elements = [
        _cell_fragment(30, 600, "Table title"),
        _cell_fragment(30, 620, "Yield"),
        _cell_fragment(142, 620, "Total"),
        _cell_fragment(30, 630, "move"),
        _cell_fragment(142, 630, "return"),
    ]
    merged, _count = _merge_pdf_semantic_table_cells(
        elements, [fitz.Rect(20, 595, 400, 640)], fitz.Rect(0, 0, 612, 792)
    )
    cells = [elem for elem in merged if elem.get("semantic_table_cell")]
    assert {elem["content"] for elem in cells} == {"Yield move", "Total return"}
    # Each header's neighbour wraps in parallel into one cell: no rows (R-6).
    assert all(elem["semantic_cell_row_bands"] == 0 for elem in cells)
    assert not any(pdf_stacked_table_cell(elem) for elem in cells)


def test_the_recorded_field_stays_out_of_the_cache_identity():
    elem = _stacked(["Inflow", "0.4", "0.8"])
    before = _pdf_cache_identity(elem)
    elem["semantic_cell_row_bands"] = 7
    assert _pdf_cache_identity(elem) == before


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

STACKED_LINES = ["Inflow", "Rated notes", "0.4", "0.7", "0.1", "52%"]
STACKED_BLOCK = "流入 评级票据 0.4 0.7 0.1 52%"


def _stacked_page():
    stacked = _stacked(STACKED_LINES, bands=2)
    neighbour = _text([300, 646, 400, 654], "Rising star", fontsize=8.0, layout_class="table")
    return [_body(100), stacked, neighbour], stacked


def _stacked_records(document):
    return [
        entry
        for entry in _kept_records(document)
        if entry["reason"].startswith(assembly.PDF_STACKED_TABLE_CELLS_REASON)
    ]


def test_a_stacked_cell_keeps_its_source_ink_and_the_neighbours_are_translated(
    tmp_path, font_path
):
    elements, stacked = _stacked_page()
    document = Document(
        tmp_path, font_path, [(elements, {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级"})]
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "评级票据" not in _squash(text)
    assert ZH[:30] in _squash(text) and "新晋升级" in _squash(text)
    (entry,) = _stacked_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert "translation withdrawn" in entry["reason"]
    assert stacked["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert "stacked_table_cell_waivers" not in document.audit


def test_a_stacked_cell_alone_on_its_page_is_drawn_as_today(tmp_path, font_path):
    stacked = _stacked(STACKED_LINES, bands=2)
    document = Document(tmp_path, font_path, [([stacked], {0: STACKED_BLOCK})])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "评级票据" in _squash(text)
    assert _stacked_records(document) == []
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert (waiver["page"], waiver["element"]) == (1, 0)
    assert "no other translation" in waiver["reason"]


def test_a_stacked_cell_whose_ink_a_translated_neighbour_touches_is_drawn_as_today(
    tmp_path, font_path
):
    stacked = _stacked(STACKED_LINES, bands=2)
    # The neighbour's ink starts a quarter point right of the cell's ink.
    neighbour = _text([175.2, 646, 260, 654], "Rising star", fontsize=8.0, layout_class="table")
    document = Document(
        tmp_path,
        font_path,
        [([_body(100), stacked, neighbour], {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级"})],
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert _stacked_records(document) == []
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert waiver["element"] == 1 and "neighbour elem 2" in waiver["reason"]
    assert "新晋升级" in _squash(text)


def test_a_refusal_after_planning_puts_a_stacked_cell_back_on_todays_path(
    tmp_path, font_path, monkeypatch
):
    # R-19, the path the sign check shares: text redaction fails, so keeping
    # the cell in source is refused after planning (the whiteout path).  The
    # page is redone once with the cell drawn as today, not restored whole.
    monkeypatch.setattr(assembly, "_remove_source_ink", lambda *args, **kwargs: False)
    elements, stacked = _stacked_page()
    document = Document(
        tmp_path, font_path, [(elements, {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级"})]
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "评级票据" in _squash(text) and "新晋升级" in _squash(text)
    assert _stacked_records(document) == []
    assert "skip_translate_reason" not in stacked
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert waiver["element"] == 1 and "refused" in waiver["reason"]


def test_stacked_cells_beyond_the_per_page_limit_are_drawn_as_today(tmp_path, font_path):
    limit = assembly.ELEMENT_SOURCE_MAX_PER_PAGE
    cells = [
        _stacked(STACKED_LINES, bands=2, x0=40.0 + 60.0 * (n % 9), y0=300.0 + 80.0 * (n // 9))
        for n in range(limit + 1)
    ]
    elements = [_body(100), *cells]
    translations = {0: ZH[:30], **{n + 1: STACKED_BLOCK for n in range(limit + 1)}}
    document = Document(tmp_path, font_path, [(elements, translations)])

    document.assemble()

    assert document.fallback_pages() == []
    assert len(_stacked_records(document)) == limit
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert waiver["element"] == limit + 1 and "kept in source" in waiver["reason"]


def test_an_untranslated_stacked_cell_is_not_a_candidate(tmp_path, font_path):
    elements, _stacked_elem = _stacked_page()
    document = Document(
        tmp_path,
        font_path,
        [(elements, {0: ZH[:30], 1: " ".join(STACKED_LINES), 2: "新晋升级"})],
    )

    document.assemble()

    assert _stacked_records(document) == []
    assert "stacked_table_cell_waivers" not in document.audit


def test_a_band_counts_only_with_a_candidate_wholly_beside_the_group():
    members = [(n, _cell_fragment(142, 640 + 10 * n, f"v{n}")) for n in range(4)]
    spanning = [(10 + n, _cell_fragment(100, 640 + 10 * n, "spanning note text")) for n in range(4)]
    beside = [(20 + n, _cell_fragment(30, 640 + 10 * n, "lbl")) for n in range(2)]
    bands = [
        [members[n], spanning[n]] + ([beside[n]] if n < 2 else []) for n in range(4)
    ]
    assert _pdf_semantic_cell_shared_row_bands(members, bands) == 2


def test_a_neighbour_left_without_a_place_over_the_kept_ink_waives_the_cell():
    stacked = _stacked(STACKED_LINES, bands=2)
    over = _text([150, 660, 300, 668], "Rising star", fontsize=8.0)
    away = _text([400, 100, 500, 108], "Far away", fontsize=8.0)
    elements = [_body(100), stacked, over, away]
    page_trans = {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级", 3: "远处"}
    conflicts = assembly._stacked_table_cell_placement_conflicts(
        elements, {1: "x"}, [2], {0, 1, 2}, {0, 2}, page_trans
    )
    assert list(conflicts) == [1] and "neighbour elem 2" in conflicts[1]
    assert assembly._stacked_table_cell_placement_conflicts(
        elements, {1: "x"}, [3], {0, 1, 3}, {0, 3}, page_trans
    ) == {}


# Ruling R-21: another element counts as a translation left on the page only
# when its translation differs from its source as plain text.
SOURCE_ECHOES = pytest.mark.parametrize(
    ("content", "rich", "stored"),
    [
        ("2025E", "2025<sup>E</sup>", "2025<sup>E</sup>"),
        ("Copyright 2026 Sample Analytics.", None, "Copyright <b>2026</b> Sample Analytics."),
        ("Rates & credit", "Rates &amp; credit", "Rates &amp; credit"),
        ("Rates and credit", None, "  Rates   and\ncredit "),
    ],
    ids=["markup", "markup-in-translation", "entity", "white-space"],
)


@SOURCE_ECHOES
def test_a_source_echo_is_no_translation_left_on_the_page(content, rich, stored):
    stacked = _stacked(STACKED_LINES, bands=2)
    echo = _text([300, 646, 400, 654], content, fontsize=8.0, layout_class="table")
    if rich is not None:
        echo["rich_content"] = rich
    elements = [stacked, echo]
    conflicts = assembly._stacked_table_cell_placement_conflicts(
        elements, {0: "x"}, [], {0, 1}, {1}, {0: STACKED_BLOCK, 1: stored}
    )
    assert list(conflicts) == [0] and "no other translation" in conflicts[0]
    # The mirror: a real translation of the same element is one.
    assert assembly._stacked_table_cell_placement_conflicts(
        elements, {0: "x"}, [], {0, 1}, {1}, {0: STACKED_BLOCK, 1: "利率与信用"}
    ) == {}


def _echo_label(y=300.0):
    """A short label whose stored translation is its source with markup."""
    return _text(
        [300, y, 400, y + 8], "2025E", fontsize=8.0, layout_class="table",
        rich_content="2025<sup>E</sup>",
    )


def test_a_stacked_cell_beside_only_a_markup_echo_is_drawn_as_today(tmp_path, font_path):
    stacked = _stacked(STACKED_LINES, bands=2)
    document = Document(
        tmp_path, font_path, [([stacked, _echo_label()], {0: STACKED_BLOCK, 1: "2025<sup>E</sup>"})]
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "评级票据" in _squash(text)
    assert _stacked_records(document) == []
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert (waiver["page"], waiver["element"]) == (1, 0)
    assert "no other translation" in waiver["reason"]


def test_a_stacked_cell_beside_a_markup_echo_and_a_translation_keeps_its_source_ink(
    tmp_path, font_path
):
    stacked = _stacked(STACKED_LINES, bands=2)
    document = Document(
        tmp_path,
        font_path,
        [(
            [_body(100), stacked, _echo_label()],
            {0: ZH[:30], 1: STACKED_BLOCK, 2: "2025<sup>E</sup>"},
        )],
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "评级票据" not in _squash(text)
    assert ZH[:30] in _squash(text)
    (entry,) = _stacked_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert "stacked_table_cell_waivers" not in document.audit


def test_a_stacked_merge_endpoint_is_drawn_as_today(tmp_path, font_path):
    elements, _stacked_elem = _stacked_page()
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (elements, {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级"}),
            ([_body(100)], {0: ZH[6:30]}),
        ],
        merge_decisions=[merge],
    )

    document.assemble()

    assert _stacked_records(document) == []
    (waiver,) = document.audit["stacked_table_cell_waivers"]
    assert waiver["element"] == 1 and "merge endpoint" in waiver["reason"]


# ---------------------------------------------------------------------------
# R-6: a single cell that spans several lines is not a stack of cells
# ---------------------------------------------------------------------------

RISK_LINES = [
    "Funding costs could rise if",
    "deposit competition stays",
    "intense through the rest of",
    "the year and margins narrow",
    "further than we assume",
]
MITIGANT_LINES = [
    "The bank holds a large stock",
    "of liquid assets and has",
    "termed out much of its",
    "wholesale funding over the",
    "last two years already",
]


def _risk_table_fragments():
    """Key risk | Mitigant | Score: two text cells per row wrapping in
    parallel over five lines, a one-line score, then a row of a tile holding
    a label and its value and a header cell with a date line."""
    elements = [_cell_fragment(76, 80, "Risk table")]
    elements += [
        _cell_fragment(76, 100, "Key risk"),
        _cell_fragment(256, 100, "Mitigant"),
        _cell_fragment(436, 100, "Score"),
    ]
    for row, top in enumerate((120, 190)):
        for line in range(5):
            elements.append(_cell_fragment(76, top + 11 * line, RISK_LINES[line]))
            elements.append(_cell_fragment(256, top + 11 * line, MITIGANT_LINES[line]))
        elements.append(_cell_fragment(436, top, str(3 + row)))
    elements += [
        _cell_fragment(76, 260, "Loan growth"),
        _cell_fragment(76, 271, "12.5%"),
        _cell_fragment(256, 260, "Close"),
        _cell_fragment(256, 271, "10/11/2025"),
        _cell_fragment(436, 265, "7"),
    ]
    return elements


def test_parallel_wrapped_text_cells_a_tile_and_a_dated_header_are_not_stacked():
    merged, _count = _merge_pdf_semantic_table_cells(
        _risk_table_fragments(), [fitz.Rect(70, 75, 560, 290)], fitz.Rect(0, 0, 612, 792)
    )
    cells = [elem for elem in merged if elem.get("semantic_table_cell")]
    contents = [elem["content"] for elem in cells]
    wrapped = [content for content in contents if content.startswith(("Funding", "The bank"))]
    assert len(wrapped) == 4
    assert all(len(content.split()) < 30 for content in wrapped)  # one row each
    assert {"Loan growth 12.5%", "Close 10/11/2025"} <= set(contents)
    for elem in cells:
        assert elem["semantic_cell_row_bands"] == 0, elem["content"]
        assert not pdf_stacked_table_cell(elem), elem["content"]


def test_lines_of_one_parallel_wrapped_neighbour_cell_are_one_cell_not_rows():
    members = [(n, _cell_fragment(76, 120 + 11 * n, RISK_LINES[n])) for n in range(5)]
    neighbour = [(10 + n, _cell_fragment(256, 120 + 11 * n, MITIGANT_LINES[n])) for n in range(5)]
    bands = [[members[n], neighbour[n]] for n in range(5)]
    # Each line its own cell (the reading before R-6): five bands counted.
    assert _pdf_semantic_cell_shared_row_bands(members, bands) == 5
    # After merging the neighbour is one cell: no evidence of rows.
    one_cell = {index: 10 for index, _ in neighbour}
    assert _pdf_semantic_cell_shared_row_bands(members, bands, one_cell) == 0


def test_only_the_bands_beside_separate_cells_of_another_column_count():
    members = [(n, _cell_fragment(142, 120 + 11 * n, f"line {n}")) for n in range(5)]
    wrapped = [(10 + n, _cell_fragment(256, 120 + 11 * n, MITIGANT_LINES[n])) for n in range(5)]
    labels = {1: (21, _cell_fragment(30, 131, "Row A")), 3: (23, _cell_fragment(30, 153, "Row B"))}
    bands = [
        [members[n], wrapped[n]] + ([labels[n]] if n in labels else []) for n in range(5)
    ]
    cell_of = {index: 10 for index, _ in wrapped}
    assert _pdf_semantic_cell_shared_row_bands(members, bands, cell_of) == 2
    # One row label alone has no other cell of its column to differ from.
    del labels[3]
    bands = [
        [members[n], wrapped[n]] + ([labels[n]] if n in labels else []) for n in range(5)
    ]
    assert _pdf_semantic_cell_shared_row_bands(members, bands, cell_of) == 0


def test_separate_cells_of_different_columns_are_no_evidence_of_rows():
    members = [(n, _cell_fragment(142, 120 + 11 * n, f"line {n}")) for n in range(5)]
    left = (21, _cell_fragment(30, 131, "Row A"))  # left column, band 1
    right = (23, _cell_fragment(400, 153, "Note"))  # a far-right column, band 3
    bands = [[members[n]] for n in range(5)]
    bands[1].append(left)
    bands[3].append(right)
    assert _pdf_semantic_cell_shared_row_bands(members, bands) == 0


def test_a_fragment_beside_the_last_line_of_a_wrapped_neighbour_is_no_row():
    """A neighbour cell wraps over the cell's five bands, its first line wide
    and its last line short; a separate fragment sits beside that last line
    within the neighbour's extent.  The two cells share a band, so they are
    not cells of two rows."""
    members = [(n, _cell_fragment(76, 120 + 11 * n, RISK_LINES[n])) for n in range(5)]
    widths = ["a wide first line of the note", "next line", "next line", "next line", "end"]
    neighbour = [(10 + n, _cell_fragment(256, 120 + 11 * n, widths[n])) for n in range(5)]
    beside = (30, _cell_fragment(300, 164, "x9"))
    bands = [[members[n], neighbour[n]] for n in range(5)]
    bands[4].append(beside)
    cell_of = {index: 10 for index, _ in neighbour}
    assert _pdf_semantic_cell_shared_row_bands(members, bands, cell_of) == 0


@pytest.mark.parametrize(
    ("lines", "bands"),
    [
        (["Plan for the next", "4", "quarters"], 0),
        (["Inflow", "Rated notes", "0.4"], 7),
        (["Stance", "Long", "Long", "Short", "Flat"], 5),
    ],
)
def test_a_native_cell_with_fewer_than_two_value_lines_is_never_stacked(lines, bands):
    elem = _stacked(lines, bands=bands, semantic_native_table_cell=True)
    assert not pdf_stacked_table_cell(elem)
    elem["semantic_native_table_cell"] = False
    assert pdf_stacked_table_cell(elem) is (bands >= 2)


# Column headings over a column of figures, and chart axis titles with their
# tick values (all synthetic).
VALUE_STACKS = [
    ["Change (%)", "-3.1%", "0.4%", "-12%"],
    ["Issued ($bn)", "312", "1,045"],
    ["Spread (bp)", "300", "200", "100", "0"],
    ["140", "Index level (RHS)", "120", "100", "80"],
    ["2.10", "1.80", "1.50", "Ratio (x)"],
]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bands", [None, 0, 1, 2, 4])
@pytest.mark.parametrize("lines", VALUE_STACKS)
def test_two_value_lines_are_a_stack_whatever_the_grid_and_the_other_columns(
    lines, bands, native
):
    """R-6b: a heading or axis title with two or more value-only lines is a
    stack, native cell or not, with or without other-column evidence."""
    elem = _stacked(lines, bands=bands, semantic_native_table_cell=native)
    assert pdf_stacked_table_cell(elem)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bands", [None, 0, 1])
@pytest.mark.parametrize(
    "lines",
    [
        # side-by-side wrapped text cells: no value line
        ["Funding risk is", "limited because the", "bank has termed out"],
        # a tile holding a label and its value, a header with a date line
        ["Loan growth", "12.5%"],
        ["Close", "10/11/2025"],
        # a native single cell holding one number between words
        ["Plan for the next", "4", "quarters"],
        # one value line under an axis title
        ["-250", "Weekly fund flows (USD mn)"],
    ],
)
def test_the_reviewers_single_cells_hold_fewer_than_two_value_lines_and_are_not_stacked(
    lines, bands, native
):
    elem = _stacked(lines, bands=bands, semantic_native_table_cell=native)
    assert not pdf_stacked_table_cell(elem)


def _large_merged_region(ticks):
    """Four exhibits merged into one element: titles, legends, a month row and
    a source line each; ``ticks`` value-only lines in the first exhibit."""
    lines = []
    for exhibit in range(4):
        lines.append(f"Exhibit {exhibit + 1}: spreads of new issues by month (bp)")
        if exhibit == 0:
            lines += [str(250 - 50 * tick) for tick in range(ticks)]
        lines += ["New issues Secondary", "Jan Apr Jul Oct", "Source: a synthetic data provider"]
    return lines


def test_a_large_merged_region_with_one_value_line_is_not_stacked():
    """The region has 17 lines: 16 label lines and one value-only line."""
    lines = _large_merged_region(ticks=1)
    assert len(lines) == 17
    elem = _stacked(lines, bands=0)
    assert not pdf_stacked_table_cell(elem)
    elem["semantic_native_table_cell"] = True
    elem["semantic_cell_row_bands"] = 9
    assert not pdf_stacked_table_cell(elem)


def test_a_large_merged_region_with_two_value_lines_is_stacked():
    """R-6b holds for any element of the table path: with two tick values the
    same region is a stack (keeping it is then left to the waiver paths)."""
    elem = _stacked(_large_merged_region(ticks=2), bands=0, semantic_native_table_cell=True)
    assert pdf_stacked_table_cell(elem)


def _generated_page(writer):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    writer(page)
    doc = fitz.open("pdf", doc.tobytes())
    return doc, doc[0]


def _ruled_risk_table(page):
    columns = [72, 252, 432, 540]
    rows = [100, 120, 190, 260, 300]
    for x in columns:
        page.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page.draw_line((columns[0], y), (columns[-1], y))
    page.insert_text((76, 114), "Key risk", fontsize=9)
    page.insert_text((256, 114), "Mitigant", fontsize=9)
    page.insert_text((436, 114), "Score", fontsize=9)
    for row in range(2):
        top = rows[1 + row] + 12
        for line in range(5):
            page.insert_text((76, top + line * 11), RISK_LINES[line], fontsize=9)
            page.insert_text((256, top + line * 11), MITIGANT_LINES[line], fontsize=9)
        page.insert_text((436, top), str(3 + row), fontsize=9)
    page.insert_text((76, 280), "Loan growth", fontsize=9)
    page.insert_text((76, 291), "12.5%", fontsize=9)
    page.insert_text((256, 280), "Close", fontsize=9)
    page.insert_text((256, 291), "10/11/2025", fontsize=9)
    page.insert_text((436, 285), "7", fontsize=9)


def _native_two_by_two(page):
    for x in (72, 242, 412):
        page.draw_line((x, 280), (x, 400))
    for y in (280, 370, 400):
        page.draw_line((72, y), (412, y))
    page.insert_text((82, 305), "Plan for the next", fontsize=10)
    page.insert_text((82, 325), "4", fontsize=10)
    page.insert_text((82, 345), "quarters", fontsize=10)
    page.insert_text((252, 305), "Stable outlook", fontsize=10)
    page.insert_text((82, 390), "Revenue", fontsize=10)
    page.insert_text((252, 390), "17", fontsize=10)


@pytest.mark.parametrize("writer", [_ruled_risk_table, _native_two_by_two])
def test_the_reviewers_ruled_tables_keep_no_cell(writer):
    from phoena_translator.pdf.extract.page import _extract_page_elements

    doc, page = _generated_page(writer)
    cells = [
        elem
        for elem in _extract_page_elements(page, doc)
        if elem.get("semantic_table_cell")
    ]
    assert cells
    assert not any(pdf_stacked_table_cell(elem) for elem in cells)
    if writer is _native_two_by_two:
        (cell,) = cells
        assert cell["content"] == "Plan for the next 4 quarters"
        assert cell["semantic_native_table_cell"]


def _ruled_table_without_row_rules(page):
    """A header row, then one ruled body cell per column holding four lines:
    a column heading over three figures beside a wrapped note."""
    for x in (72, 242, 412):
        page.draw_line((x, 280), (x, 400))
    for y in (280, 300, 400):
        page.draw_line((72, y), (412, y))
    page.insert_text((82, 294), "Region", fontsize=10)
    page.insert_text((252, 294), "Note", fontsize=10)
    body = [
        ("Change (%)", "Steady demand"),
        ("-3.1%", "from new"),
        ("0.4%", "issuers"),
        ("-12%", "this year"),
    ]
    for row, (left, right) in enumerate(body):
        page.insert_text((82, 320 + 20 * row), left, fontsize=10)
        page.insert_text((252, 320 + 20 * row), right, fontsize=10)


def test_a_native_cell_holding_a_heading_over_figures_is_stacked():
    """R-6b against the native grid: the grid confirms each body cell as one
    cell, and no other column shows separate cells; the heading over three
    figures is still a stack, the wrapped note beside it is not."""
    from phoena_translator.pdf.extract.page import _extract_page_elements

    doc, page = _generated_page(_ruled_table_without_row_rules)
    cells = {
        elem["content"]: elem
        for elem in _extract_page_elements(page, doc)
        if elem.get("semantic_table_cell")
    }
    figures = cells["Change (%) -3.1% 0.4% -12%"]
    note = cells["Steady demand from new issuers this year"]
    assert figures["semantic_native_table_cell"] and note["semantic_native_table_cell"]
    assert figures["semantic_cell_row_bands"] == 0
    assert pdf_stacked_table_cell(figures)
    assert not pdf_stacked_table_cell(note)


def test_a_label_and_value_cell_without_row_evidence_is_drawn_translated(
    tmp_path, font_path
):
    tile = _stacked(["Loan growth", "12.5%"], bands=0)
    neighbour = _text([300, 646, 400, 654], "Rising star", fontsize=8.0, layout_class="table")
    document = Document(
        tmp_path,
        font_path,
        [([_body(100), tile, neighbour], {0: ZH[:30], 1: "贷款增长 12.5%", 2: "新晋升级"})],
    )

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "贷款增长" in _squash(text)
    assert _stacked_records(document) == []
    assert "stacked_table_cell_waivers" not in document.audit
