"""Figure/table-scoped source fallback: a figure or table whose text cannot be
typeset is delivered in its exact source ink, the rest of its page translated.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from types import SimpleNamespace

import fitz
import pytest
from golden_support import build_test_font

from phoena_translator.pdf.errors import PDFPageRenderError
from phoena_translator.pdf.geometry import _filter_pdf_formula_safe_rects
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.render import assembly
from phoena_translator.pdf.render import figure_table_groups as groups
from phoena_translator.pdf.render import stage

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可"
PAGE = fitz.Rect(0, 0, 612, 792)
BODY = (
    "Below we show the estimated exposure for each insurer and explain how "
    "the general account figures were assembled from the company filings "
    "and the investor presentations published this quarter."
)
LONG_CELL = "Asset-base definition / source Estimated AI exposure"


@pytest.fixture(scope="module")
def font_path(tmp_path_factory):
    built = build_test_font(tmp_path_factory.mktemp("font"))
    if built is None:
        pytest.skip("no CJK TrueType font available to build the test font")
    return built[0]


def _text(bbox, content, fontsize=10.0, **extra):
    return {
        "type": ElementType.TEXT,
        "bbox": list(bbox),
        "content": content,
        "fontsize": fontsize,
        **extra,
    }


def _region(bbox):
    return {"type": ElementType.TABLE_IMAGE, "bbox": list(bbox), "rect": list(bbox)}


def _formula_over_right_edge(cell):
    """A protected formula region over a cell's right edge: neither the
    translation nor the restored source text can be placed (case 1, page 8)."""
    x0, y0, x1, y1 = cell["bbox"]
    return {"type": ElementType.FORMULA_IMAGE, "bbox": [x1 - 6, y0 - 2, x1 + 34, y1 + 2]}


def _body(y, content=BODY):
    return _text([72, y, 540, y + 40], content, layout_class="body")


def _cell(x, y, content, width=40.0, **extra):
    extra.setdefault("layout_class", "table")
    return _text([x, y, x + width, y + 6], content, fontsize=5.0, **extra)


def _figure(y, title="Figure 3: A detailed snapshot of AI exposures", cells=4):
    """A captioned small-type table the table finder did not recognise."""
    elements = [_text([72, y, 400, y + 9], title, fontsize=9.0, layout_class="scattered")]
    for row in range(cells):
        elements.append(_cell(72 + 120 * (row % 3), y + 14 + 10 * (row // 3), f"Cell {row} AUM"))
    elements.append(
        _text(
            [72, y + 60, 300, y + 67],
            "Source: Company reports.",
            fontsize=7.0,
            layout_class="scattered",
        )
    )
    return elements


class Document:
    """A source PDF, its HTML-box work copy and the assembly context."""

    def __init__(self, tmp_path, font_path, pages, *, merge_decisions=()):
        self.source_path = tmp_path / "source.pdf"
        self.work_path = tmp_path / "work.pdf"
        source = fitz.open()
        for elements, _translations in pages:
            page = source.new_page(width=612, height=792)
            for elem in elements:
                if elem["type"] == ElementType.TEXT:
                    x0, _, _, y1 = elem["bbox"]
                    page.insert_text(
                        (x0, y1 - 1), elem["content"], fontsize=min(elem["fontsize"], 9.0)
                    )
        source.save(str(self.source_path))
        source.close()
        shutil.copy2(self.source_path, self.work_path)
        self.pages = pages
        self.audit = _new_pdf_audit("t", 0.6, [])
        self.audit["merge_decisions"] = list(merge_decisions)
        self.context = assembly.PDFAssemblyContext(
            task_id="t",
            total_pages=len(pages),
            pdf_password="",
            use_htmlbox=True,
            fail_open_to_source_page=True,
            minimum_htmlbox_scale=0.6,
            assembly_work_path=str(self.work_path),
            src_path=str(self.source_path),
            out_doc=fitz.open(str(self.work_path)),
            page_extractions={
                number: {
                    "elements": elements,
                    "cached": {str(k): v for k, v in translations.items()},
                }
                for number, (elements, translations) in enumerate(pages)
            },
            pdf_audit=self.audit,
            font_path=str(font_path),
            bold_font_path=str(font_path),
            has_distinct_bold_font=False,
            font_archive=fitz.Archive(str(font_path.parent)),
            font_basename=font_path.name,
            logger=logging.getLogger("test-figure-table"),
            filter_formula_safe_rects=_filter_pdf_formula_safe_rects,
            trim_process_memory=lambda: None,
        )

    def assemble(self) -> list[str]:
        assembly.assemble_document_pages(self.context)
        if self.context.out_doc is not None:
            self.context.out_doc.close()
            self.context.out_doc = None
        with fitz.open(str(self.work_path)) as output:
            return [page.get_text().replace("\n", "") for page in output]

    def fallback_pages(self) -> list[int]:
        return [entry["page"] for entry in self.audit["source_page_fallbacks"]]

    def figure_tables(self) -> list[dict]:
        return self.audit.get("figure_table_source_fallbacks", [])


def _inject_render_failures(monkeypatch, failing):
    """Raise a render failure for each (page index, element index) in ``failing``
    when that element is drawn, after the page was redacted and earlier
    elements were drawn."""
    real = assembly._render_element

    def render(context, plan, css, index):
        if (plan.page_num, index) in failing:
            raise PDFPageRenderError(plan.page_num + 1, index, "injected no-fit")
        return real(context, plan, css, index)

    monkeypatch.setattr(assembly, "_render_element", render)


def _squash(text: str) -> str:
    return text.replace(" ", "")


# ---------------------------------------------------------------------------
# Real no-fit failures
# ---------------------------------------------------------------------------


def test_failing_cell_in_a_recognised_table_keeps_only_that_table(tmp_path, font_path):
    elements = [
        _body(100),
        _region([60, 200, 560, 260]),
        _cell(72, 210, "Ticker", table_hint=True),
        _cell(200, 210, "GA AUM", table_hint=True),
        # Too small for its translation and for its own source text.
        _cell(330, 210, LONG_CELL, width=24.0, table_hint=True),
        _cell(72, 230, "MET", table_hint=True),
        _body(400),
    ]
    elements.append(_formula_over_right_edge(elements[4]))
    translations = {0: ZH[:30], 2: "代码", 3: "资产", 4: ZH * 4, 5: "大都会", 6: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    (record,) = document.figure_tables()
    assert record["page"] == 1 and record["element"] == 4
    assert record["group"] == groups.TABLE_REGION
    assert record["elements"] == [2, 3, 4, 5]
    assert record["narrowed_elements"] == [2, 3, 4, 5]
    assert "Page 1 elem 4" in record["reason"]
    # The table keeps its exact source ink; the page around it is translated.
    for source in ("Ticker", "GA AUM", "MET"):
        assert source in text
    for translation in ("代码", "资产", "大都会"):
        assert translation not in text
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    assert "Below we show" not in text
    for index in (2, 3, 4, 5):
        assert elements[index]["skip_translate_reason"] == SkipReason.RENDER_FIGURE_TABLE_SOURCE


def test_failing_cell_in_an_unrecognised_scattered_table_keeps_its_caption_band(
    tmp_path, font_path
):
    figure = _figure(200)
    # The header cell: classed scattered, no table hint, too small to fit.
    figure[1] = _cell(72, 214, LONG_CELL, width=24.0, layout_class="scattered")
    elements = [_body(100), *figure, _body(400)]
    elements.append(_formula_over_right_edge(elements[2]))
    translations = {
        0: ZH[:30],
        1: "图3：保险公司人工智能敞口详表",
        2: ZH * 4,
        3: "单元一",
        4: "单元二",
        5: "单元三",
        6: "资料来源：公司报告。",
        7: ZH[5:30],
    }
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    (record,) = document.figure_tables()
    assert record["group"] == groups.CAPTION_BAND
    assert record["element"] == 2
    assert record["elements"] == [2, 3, 4, 5]
    # Title, source line and body stay translated.
    assert "图3" in _squash(text) and "资料来源" in _squash(text)
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    for source in ("Cell 1 AUM", "Cell 2 AUM", "Cell 3 AUM"):
        assert source in text
    for translation in ("单元一", "单元二", "单元三"):
        assert translation not in text


# ---------------------------------------------------------------------------
# Injected failures: page state, several groups, body text, merges
# ---------------------------------------------------------------------------


def test_reassembled_page_has_no_duplicate_ink_or_audit_events(
    tmp_path, font_path, monkeypatch
):
    elements = [_body(100), *_figure(200), _body(400)]
    translations = {0: ZH[:30], 2: "单元零", 3: "单元一", 4: "单元二", 5: "单元三", 7: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])
    # Element 0 is drawn before the failing cell 4, so the failed attempt
    # leaves translated ink and audit events behind.
    _inject_render_failures(monkeypatch, {(0, 4)})

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert _squash(text).count(ZH[:30]) == 1
    assert "Below we show" not in text
    # Cell 2 was redacted and drawn translated before cell 4 failed; the
    # retry starts from the checkpoint, so its source ink is back.
    assert "Cell 0 AUM" in text and "单元零" not in text
    placements = [event["element"] for event in document.audit["htmlbox_placement_events"]]
    assert placements.count(0) == 1 and placements.count(7) == 1
    assert not {2, 3, 4, 5} & set(placements)
    (record,) = document.figure_tables()
    assert record["narrowed_elements"] == [2, 3, 4, 5]
    fallbacks = document.audit["element_source_fallbacks"]
    assert sorted(entry["element"] for entry in fallbacks) == [2, 3, 4, 5]
    assert {entry["stage"] for entry in fallbacks} == {"render-figure-table"}


def test_two_failing_figures_on_one_page_are_each_narrowed(tmp_path, font_path, monkeypatch):
    first = _figure(150, title="Figure 3: Insurer exposure")
    second = _figure(300, title="Figure 4: Hyperscaler holders")
    elements = [_body(80), *first, *second, _body(450)]
    translations = {
        index: f"译文{index}"
        for index, elem in enumerate(elements)
        if elem["type"] == ElementType.TEXT
    }
    document = Document(tmp_path, font_path, [(elements, translations)])
    _inject_render_failures(monkeypatch, {(0, 3), (0, 10)})

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    records = document.figure_tables()
    assert [record["element"] for record in records] == [3, 10]
    assert [record["elements"] for record in records] == [[2, 3, 4, 5], [8, 9, 10, 11]]
    for index in (1, 6, 7, 12, 0, 13):  # titles, source lines, body
        assert f"译文{index}" in text


def test_failing_body_element_still_falls_back_to_the_merged_source_pages(
    tmp_path, font_path, monkeypatch
):
    page_one = [_body(100), *_figure(200), _body(600)]
    page_two = [_body(100), _body(200)]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 7,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (page_one, {0: ZH[:20], 7: ZH[3:20]}),
            (page_two, {0: ZH[6:20], 1: ZH[9:20]}),
        ],
        merge_decisions=[merge],
    )
    _inject_render_failures(monkeypatch, {(0, 0)})

    first, second = document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert document.figure_tables() == []
    assert "Below we show" in first and "Below we show" in second
    assert ZH[6:20] not in _squash(second)


def test_narrowed_figure_no_longer_expands_across_accepted_merges(
    tmp_path, font_path, monkeypatch
):
    page_one = [_body(100), *_figure(200), _body(600)]
    page_two = [_body(100), _body(200)]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 7,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (page_one, {0: ZH[:20], 3: "单元一", 7: ZH[3:20]}),
            (page_two, {0: ZH[6:20], 1: ZH[9:20]}),
        ],
        merge_decisions=[merge],
    )
    _inject_render_failures(monkeypatch, {(0, 3)})

    first, second = document.assemble()

    assert document.fallback_pages() == []
    assert [record["page"] for record in document.figure_tables()] == [1]
    assert ZH[:20] in _squash(first) and ZH[3:20] in _squash(first)
    assert ZH[6:20] in _squash(second)
    assert "inline_render_source_page_fallbacks" not in document.audit


def test_figure_holding_a_merge_endpoint_keeps_the_page_fallback(
    tmp_path, font_path, monkeypatch
):
    elements = [_body(100), *_figure(200)]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 4,
        "destination_page": 1,
        "destination_element": 0,
    }
    document = Document(
        tmp_path, font_path, [(elements, {0: ZH[:20], 3: "单元一"})], merge_decisions=[merge]
    )
    _inject_render_failures(monkeypatch, {(0, 3)})

    document.assemble()

    assert document.fallback_pages() == [1]
    assert document.figure_tables() == []


def test_cross_page_closure_failure_is_never_narrowed(tmp_path, font_path, monkeypatch):
    elements = [_body(100), *_figure(200)]
    document = Document(
        tmp_path,
        font_path,
        [(elements, {0: ZH[:20], 3: "单元一"}), ([_body(100)], {0: ZH[:9]})],
    )
    real = assembly._render_element

    def render(context, plan, css, index):
        if (plan.page_num, index) == (0, 3):
            error = PDFPageRenderError(1, 3, "closure")
            error.fallback_pages = [1, 2]
            raise error
        return real(context, plan, css, index)

    monkeypatch.setattr(assembly, "_render_element", render)

    document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert document.figure_tables() == []


def test_failure_that_survives_narrowing_restores_the_page_state_and_falls_back(
    tmp_path, font_path, monkeypatch
):
    elements = [_body(100), *_figure(200), _body(400)]
    translations = {0: ZH[:20], 3: "单元一", 7: ZH[3:20]}
    document = Document(tmp_path, font_path, [(elements, translations)])
    attempts = []
    real = assembly._render_element

    def render(context, plan, css, index):
        if index == 0:
            attempts.append(plan.page_num)
        if (plan.page_num, index) == (0, 3):
            raise PDFPageRenderError(1, 3, "injected no-fit")
        if (plan.page_num, index) == (0, 7):
            raise PDFPageRenderError(1, 7, "body no-fit")
        return real(context, plan, css, index)

    monkeypatch.setattr(assembly, "_render_element", render)

    (text,) = document.assemble()

    assert attempts == [0, 0]
    assert document.fallback_pages() == [1]
    assert "elem 7: body no-fit" in document.audit["source_page_fallbacks"][0]["reason"]
    assert document.figure_tables() == []
    assert document.audit.get("element_source_fallbacks", []) == []
    assert document.audit["htmlbox_placement_events"] == []
    assert "skip_translate_reason" not in elements[3]
    assert "Below we show" in text


def test_a_kept_element_that_still_fails_ends_the_loop(tmp_path, font_path, monkeypatch):
    elements = [_body(100), *_figure(200)]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 3: "单元一"})])
    attempts = []

    def redact(context, plan):
        attempts.append(plan.page_num)
        raise PDFPageRenderError(1, 3, "injected redaction failure")

    monkeypatch.setattr(assembly, "_redact_page", redact)

    document.assemble()

    assert attempts == [0, 0]
    assert document.fallback_pages() == [1]
    assert document.figure_tables() == []


def test_groups_per_page_are_capped(tmp_path, font_path, monkeypatch):
    first = _figure(150, title="Figure 3: Insurer exposure")
    second = _figure(300, title="Figure 4: Hyperscaler holders")
    elements = [_body(80), *first, *second]
    translations = {3: "单元一", 10: "单元二"}
    document = Document(tmp_path, font_path, [(elements, translations)])
    monkeypatch.setattr(assembly, "FIGURE_TABLE_RULES", replace(groups.RULES, max_groups_per_page=1))
    _inject_render_failures(monkeypatch, {(0, 3), (0, 10)})

    document.assemble()

    assert document.fallback_pages() == [1]


def test_narrowed_figure_is_excluded_from_leaks_and_expectations(
    tmp_path, font_path, monkeypatch
):
    elements = [_body(100), *_figure(200)]
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:20], 3: "单元一", 4: "单元二"})])
    document.audit["superscript_expectations"] = [
        {"page": 1, "element": 3},
        {"page": 1, "element": 0},
    ]
    document.audit["vector_ocr_expectations"] = [{"page": 1, "element": 4}]
    _inject_render_failures(monkeypatch, {(0, 3)})

    document.assemble()

    assert document.audit["untranslated_english_leaks"] == []
    assert document.audit["superscript_expectations"] == [{"page": 1, "element": 0}]
    assert document.audit["vector_ocr_expectations"] == []
    exclusions = document.audit["element_source_fallback_expectation_exclusions"]
    assert exclusions == {"superscript_expectations": 1, "vector_ocr_expectations": 1}


def test_without_a_page_checkpoint_the_page_fallback_is_unchanged(font_path, monkeypatch):
    attempted = []

    def fake_assemble_page(_context, page_num):
        attempted.append(page_num)

    monkeypatch.setattr(assembly, "_assemble_page", fake_assemble_page)
    assembly.assemble_document_pages(
        SimpleNamespace(
            total_pages=2,
            fail_open_to_source_page=True,
            use_htmlbox=False,
            assembly_work_path="work.pdf",
        )
    )
    assert attempted == [0, 1]


# ---------------------------------------------------------------------------
# Grouping rule
# ---------------------------------------------------------------------------


def _group(elements, index, **kwargs):
    return groups.find_figure_table_group(
        elements, index, page_rect=PAGE, dominant_fontsize=10.0, **kwargs
    )


def test_body_prose_is_never_a_figure_or_table():
    elements = [_body(100), *_figure(200)]
    assert _group(elements, 0) is None


def test_a_region_holding_body_prose_is_not_trusted():
    elements = [_region([60, 90, 560, 300]), _body(100), *_figure(200)]
    group = _group(elements, 4)
    assert group is not None and group.kind == groups.CAPTION_BAND
    assert 1 not in group.members


def test_small_type_without_a_caption_is_not_bounded():
    elements = [_body(100), *_figure(200)[1:5]]
    assert _group(elements, 2) is None


def test_band_mostly_unclassified_as_table_is_not_trusted():
    elements = [_body(100), *_figure(200)]
    for elem in elements[2:6]:
        elem["layout_class"] = "scattered"
    assert _group(elements, 3) is None


def test_band_stops_at_the_source_line_and_body_text():
    elements = [*_figure(200), _cell(72, 275, "Footnote-sized text"), _body(300)]
    group = _group(elements, 2)
    assert group.members == frozenset({1, 2, 3, 4})


def test_a_caption_below_frames_the_band_upward():
    cells = [_cell(72 + 120 * column, 214, f"Cell {column} AUM") for column in range(3)]
    elements = [
        _body(150),
        *cells,
        _text([72, 230, 400, 239], "Table 2. Exposure by insurer", fontsize=9.0),
        _body(300),
    ]
    group = _group(elements, 2)
    assert group is not None and group.members == frozenset({1, 2, 3})


def test_side_by_side_titles_split_the_page_width():
    left = [_text([72, 200, 290, 209], "Figure 1: Insurers", fontsize=9.0)]
    right = [_text([320, 200, 540, 209], "Figure 2: Holders", fontsize=9.0)]
    cells = [
        _cell(80, 215, "8%"),
        _cell(150, 225, "6%"),
        _cell(330, 215, "100%"),
        _cell(420, 225, "80%"),
    ]
    elements = [*left, *right, *cells, _body(300)]
    assert _group(elements, 2).members == frozenset({2, 3})
    assert _group(elements, 4).members == frozenset({4, 5})


def test_kept_and_protected_elements_end_the_narrowing():
    elements = [_body(100), *_figure(200)]
    assert _group(elements, 3, kept_indices={3}) is None
    assert _group(elements, 3, protected_indices={4}) is None
    assert _group(elements, 3) is not None


def test_merged_away_text_inside_a_band_is_not_narrowed():
    elements = [_body(100), *_figure(200)]
    elements.append({"type": ElementType.MERGED_AWAY, "bbox": [200, 224, 230, 230], "content": "x"})
    # A merged-away element closes the band; the cells below it are cut off.
    group = _group(elements, 3)
    assert group is None or 8 not in group.members


# ---------------------------------------------------------------------------
# Task record
# ---------------------------------------------------------------------------


def _finalize(audit):
    saved = {}
    context = SimpleNamespace(
        task_id="t",
        out_path="out.pdf",
        src_path="src.pdf",
        pdf_password="",
        total_pages=1,
        pdf_audit=audit,
        page_extractions={},
        logger=logging.getLogger("test-figure-table"),
        trim_process_memory=lambda: None,
        check_output_structure_serialized=lambda *a, **k: {
            "status": "ok",
            "warning_count": 0,
            "warnings": [],
        },
        save_progress=lambda task_id, record: saved.update(record),
    )
    stage.finalize_output(context, fail_open_to_source_page=False)
    return saved


def test_task_record_lists_narrowed_figures_only_when_there_are_some(monkeypatch):
    monkeypatch.setattr(stage, "_save_and_validate_pdf_audit", lambda *_a, **_k: "audit.json")
    entry = {"page": 3, "element": 6, "group": groups.CAPTION_BAND, "elements": [6, 7]}

    audit = _new_pdf_audit("t", 0.6, [])
    audit["element_source_fallbacks"] = [{"page": 3, "element": 6, "stage": "render-figure-table"}]
    audit["figure_table_source_fallbacks"] = [entry]
    record = _finalize(audit)
    assert record["completion_status"] == "completed"
    assert record["figure_table_source_fallbacks"] == [entry]
    assert record["element_source_fallbacks"][0]["stage"] == "render-figure-table"

    record = _finalize(_new_pdf_audit("t", 0.6, []))
    assert "figure_table_source_fallbacks" not in record
