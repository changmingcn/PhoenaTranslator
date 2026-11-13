"""Review leftover F4 (B5): a page delivered whole as its source page keeps no
record of a partial delivery.

The class: once a page is delivered as its exact source page, every record
that describes a partial delivery on that page is stale, whichever stage
wrote it.  The cleanup is by page (``_drop_pdf_partial_delivery_records``):
element-level records of every stage (translation, render,
render-preflight, render-figure-table) and group-level records go; a page
that still carries translated text keeps its records.  Each source-page
conversion path is covered: every translation withdrawn at assembly, the
accepted-merge closure, the final structure audit, and the whole-page
fallback at the translation stage.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import fitz
import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_placeholder_token_leak import (
    BULLET_B,
    ZH_B,
    _leaking_cached_page,
    _minted_tokens,
)

from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.render import stage
from phoena_translator.pdf.render.source_fallbacks import (
    _drop_pdf_partial_delivery_records,
    _record_pdf_element_source_fallback,
)

log = logging.getLogger("test-source-page-record-cleanup")

# Every stage that writes an element-level record today.
ELEMENT_STAGES = ["translation", "render", "render-preflight", "render-figure-table"]


def _seed(audit, page, element, stage_name, **extra):
    entry = _record_pdf_element_source_fallback(audit, page, element, stage_name, "seeded record")
    entry.update(extra)
    return entry


def _pages_of(audit, key="element_source_fallbacks"):
    return sorted({int(entry["page"]) for entry in audit.get(key) or []})


# --- the function, by page -------------------------------------------------


def test_the_cleanup_is_by_page_whatever_stage_wrote_the_record():
    audit = _new_pdf_audit("t", 0.6, [])
    for page in (1, 2):
        for element, stage_name in enumerate(ELEMENT_STAGES):
            _seed(audit, page, element, stage_name)
        _seed(audit, page, 9, "render", source_ink_kept=True)
    audit["figure_table_source_fallbacks"] = [
        {"page": 1, "element": 0, "elements": [0]},
        {"page": 2, "element": 0, "elements": [0]},
    ]

    _drop_pdf_partial_delivery_records(audit, [1])

    assert _pages_of(audit) == [2]
    assert len(audit["element_source_fallbacks"]) == len(ELEMENT_STAGES) + 1
    assert _pages_of(audit, "figure_table_source_fallbacks") == [2]

    _drop_pdf_partial_delivery_records(audit, {2})

    assert audit["element_source_fallbacks"] == []
    assert "figure_table_source_fallbacks" not in audit


def test_the_cleanup_drops_the_stacked_cell_waivers_of_source_pages():
    audit = _new_pdf_audit("t", 0.6, [])
    audit["stacked_table_cell_waivers"] = [
        {"page": 1, "element": 1, "reason": "accepted cross-page merge endpoint"},
        {"page": 2, "element": 4, "reason": "no other translation on its page"},
    ]

    _drop_pdf_partial_delivery_records(audit, [1])

    assert _pages_of(audit, "stacked_table_cell_waivers") == [2]

    _drop_pdf_partial_delivery_records(audit, {2})

    assert "stacked_table_cell_waivers" not in audit


def test_no_page_leaves_the_audit_as_it_is():
    audit = _new_pdf_audit("t", 0.6, [])
    _seed(audit, 1, 0, "translation")
    before = dict(audit)

    _drop_pdf_partial_delivery_records(audit, [])
    _drop_pdf_partial_delivery_records(audit, [0, -1])

    assert audit == before


# --- path 1: every translation of the page withdrawn at assembly -----------


def _withdrawn_page(tmp_path, font_path):
    """Element 0 was kept in source at translation; element 1 has no
    formula-safe place, so no translation is left and the page is source."""
    kept = _text([72, 100, 540, 120], "Earlier retained body text", layout_class="body")
    kept["skip_translate_reason"] = SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    unplaceable = _text([72, 400, 300, 414], "Unplaceable source text", layout_class="body")
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [40, 360, 580, 600]}
    return Document(
        tmp_path,
        font_path,
        [([kept, unplaceable, formula], {0: kept["content"], 1: ZH[:8]})],
    )


@pytest.mark.parametrize("stage_name", ELEMENT_STAGES)
def test_a_withdrawn_page_keeps_no_element_record_of_any_stage(tmp_path, font_path, stage_name):
    document = _withdrawn_page(tmp_path, font_path)
    _seed(document.audit, 1, 0, stage_name)

    document.assemble()

    assert document.fallback_pages() == [1]
    assert document.audit.get("element_source_fallbacks", []) == []
    assert document.figure_tables() == []


def test_a_page_that_still_draws_a_translation_keeps_its_records(tmp_path, font_path):
    # Control for path 1: the same retained element next to a placeable
    # translation; the page is translated and the record stays.
    kept = _text([72, 100, 540, 120], "Earlier retained body text", layout_class="body")
    kept["skip_translate_reason"] = SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    document = Document(
        tmp_path, font_path, [([kept, _body(300)], {0: kept["content"], 1: ZH[:30]})]
    )
    entry = _seed(document.audit, 1, 0, "translation")

    document.assemble()

    assert document.fallback_pages() == []
    assert document.audit["element_source_fallbacks"] == [entry]


# --- path 2: the accepted-merge closure at assembly ------------------------


@pytest.mark.parametrize("stage_name", ELEMENT_STAGES)
def test_a_merge_closure_keeps_no_record_on_its_pages_and_all_on_others(
    tmp_path, font_path, stage_name
):
    elements, _kept = _leaking_cached_page()
    token = _minted_tokens()["SUP"]
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
            (elements, {0: ZH[:30], 1: f"进一步{token}表明", 2: ZH[5:30]}),
            ([_body(100), _body(300)], {0: ZH[6:30], 1: ZH[9:30]}),
            ([_body(100), _body(300)], {0: ZH[3:30], 1: ZH[7:30]}),
        ],
        merge_decisions=[merge],
    )
    _seed(document.audit, 1, 0, stage_name)
    _seed(document.audit, 2, 1, stage_name)
    third = _seed(document.audit, 3, 1, stage_name)

    document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert document.audit["element_source_fallbacks"] == [third]


def test_a_merge_closure_drops_the_stacked_cell_waiver_of_its_endpoint(tmp_path, font_path):
    """The reviewers' reproduction: page 1's stacked cell is a merge endpoint
    and is waived (drawn as one block); page 2 leaks a placeholder and falls
    back, and the closure restores page 1 whole too.  No waiver remains."""
    from test_pdf_stacked_cells import STACKED_BLOCK, STACKED_LINES, _stacked

    stacked = _stacked(STACKED_LINES, bands=2)
    neighbour = _text([300, 646, 400, 654], "Rising star", fontsize=8.0, layout_class="table")
    leaking, _kept = _leaking_cached_page()
    token = _minted_tokens()["SUP"]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 1,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            ([_body(100), stacked, neighbour], {0: ZH[:30], 1: STACKED_BLOCK, 2: "新晋升级"}),
            (leaking, {0: ZH[:30], 1: f"进一步{token}表明", 2: ZH[5:30]}),
            ([_body(100), _body(300)], {0: ZH[3:30], 1: ZH[7:30]}),
        ],
        merge_decisions=[merge],
    )

    document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert "stacked_table_cell_waivers" not in document.audit


# --- path 3: the final structure audit -------------------------------------


def _two_page_pdf(path, text):
    document = fitz.open()
    for page in (1, 2):
        document.new_page().insert_text((72, 72), f"{text} {page}")
    document.save(str(path))
    document.close()


@pytest.mark.parametrize("stage_name", ELEMENT_STAGES)
def test_a_structure_fallback_keeps_no_record_on_its_page(tmp_path, stage_name):
    source_path, out_path = tmp_path / "source.pdf", tmp_path / "out.pdf"
    _two_page_pdf(source_path, "SOURCE")
    _two_page_pdf(out_path, "TRANSLATED")
    audit = _new_pdf_audit("t", 0.6, [])
    _seed(audit, 1, 0, stage_name)
    _seed(audit, 1, 3, "render", source_ink_kept=True)
    kept = _seed(audit, 2, 0, stage_name)
    audit["figure_table_source_fallbacks"] = [{"page": 1, "element": 0, "elements": [0]}]
    waiver = {"page": 2, "element": 5, "reason": "no other translation on its page"}
    audit["stacked_table_cell_waivers"] = [
        {"page": 1, "element": 4, "reason": "accepted cross-page merge endpoint"},
        waiver,
    ]
    audit["structure_check"] = {
        "status": "warning", "warning_count": 1, "warnings": [{"page": 1, "type": "x"}]
    }
    context = SimpleNamespace(
        task_id="t",
        out_path=str(out_path),
        src_path=str(source_path),
        pdf_password="",
        pdf_audit=audit,
        page_extractions={0: {"elements": []}, 1: {"elements": []}},
        logger=log,
        check_output_structure_serialized=lambda *a, **k: {
            "status": "ok", "warning_count": 0, "warnings": []
        },
    )

    stage._apply_structure_fallbacks(context)

    assert [entry["page"] for entry in audit["source_page_fallbacks"]] == [1]
    assert audit["element_source_fallbacks"] == [kept]
    assert "figure_table_source_fallbacks" not in audit
    assert audit["stacked_table_cell_waivers"] == [waiver]
    with fitz.open(str(out_path)) as output:
        assert "SOURCE 1" in output[0].get_text()
        assert "TRANSLATED 2" in output[1].get_text()


# --- path 4: the whole-page fallback at the translation stage --------------


def _translation_context(merge_decisions):
    from phoena_translator.pdf.translate import stage as translate_stage

    def body(y):
        return {
            "type": "text",
            "layout_class": "body",
            "content": BULLET_B,
            "bbox": [72.0, y, 540.0, y + 12.0],
            "fontsize": 10.0,
        }

    audit = _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = list(merge_decisions)
    return translate_stage.PDFTranslationStageContext(
        task_id="t",
        total_pages=3,
        source_sha256="0" * 64,
        workers=1,
        fail_open_to_source_page=True,
        logger=log,
        page_extractions={
            0: {"elements": [body(100), body(200)]},
            1: {"elements": [body(100)]},
            2: {"elements": [body(100)]},
        },
        pdf_audit=audit,
        completed_indices=set(),
        save_translation_progress=lambda *args, **kwargs: None,
    )


def _run_translation_stage(monkeypatch, merge_decisions):
    """Page 1 keeps element 1 in source after both tries (an element-level
    record at the translation stage); page 2 fails both tries; page 3 is
    translated, so the document is not refused as an all-English copy."""
    from phoena_translator.pdf.context import PDFPageTranslationContext
    from phoena_translator.pdf.translate import stage as translate_stage

    token = _minted_tokens()["IMATH"]

    def fake_translate_page(_page_context, page_num, _info):
        if page_num == 1:
            raise RuntimeError("provider failure")
        if page_num == 2:
            return page_num, {0: ZH_B}
        return page_num, {0: ZH_B, 1: f"{ZH_B}{token}。"}

    monkeypatch.setattr(translate_stage, "translate_page", fake_translate_page)
    monkeypatch.setattr(
        translate_stage, "_save_pdf_page_translation_cache", lambda *a, **k: None
    )
    context = _translation_context(merge_decisions)
    page_context = PDFPageTranslationContext(
        task_id="t", prompt="翻译", logger=log, translate_text=lambda *a, **k: ""
    )
    translate_stage.translate_pages(context, page_context, [0, 1, 2])
    return context


def test_translation_stage_without_a_merge_keeps_the_translated_pages_records(monkeypatch):
    # Control for path 4: page 1 is translated, so its record stays.
    context = _run_translation_stage(monkeypatch, [])

    assert [entry["page"] for entry in context.pdf_audit["source_page_fallbacks"]] == [2]
    assert [
        (entry["page"], entry["element"], entry["stage"])
        for entry in context.pdf_audit["element_source_fallbacks"]
    ] == [(1, 1, "translation")]


def test_translation_stage_merge_expansion_keeps_no_record_on_a_source_page(monkeypatch):
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 0,
        "destination_page": 2,
        "destination_element": 0,
    }
    context = _run_translation_stage(monkeypatch, [merge])

    assert [entry["page"] for entry in context.pdf_audit["source_page_fallbacks"]] == [1, 2]
    assert context.pdf_audit.get("element_source_fallbacks", []) == []
