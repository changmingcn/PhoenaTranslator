"""Page assembly: preflight, redaction, HTML-box placement, textbox fallback
and source restoration on real PyMuPDF pages.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging

import fitz
import pytest
from golden_support import build_test_font

from phoena_translator.pdf.errors import PDFPageRenderError
from phoena_translator.pdf.geometry import _filter_pdf_formula_safe_rects
from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.render import assembly

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可"
SOURCE = "Source English text here"


@pytest.fixture(scope="module")
def font_path(tmp_path_factory):
    built = build_test_font(tmp_path_factory.mktemp("font"))
    if built is None:
        pytest.skip("no CJK TrueType font available to build the test font")
    return built[0]


def _text(bbox, content=SOURCE, **extra):
    return {
        "type": ElementType.TEXT,
        "bbox": list(bbox),
        "content": content,
        "fontsize": 10.0,
        **extra,
    }


class Page:
    """One-page document plus the assembly context that renders it."""

    def __init__(
        self,
        font_path,
        elements,
        translations=None,
        *,
        use_htmlbox=True,
        minimum_scale=0.6,
        total_pages=1,
        merge_decisions=(),
        info=None,
    ):
        self.doc = fitz.open()
        for _ in range(total_pages):
            self.doc.new_page(width=612, height=792)
        for elem in elements:
            if elem.get("type") in {ElementType.TEXT, ElementType.MERGED_AWAY}:
                x0, _, _, y1 = elem["bbox"]
                self.doc[0].insert_text((x0, y1 - 3), elem.get("content", ""), fontsize=9)
        self.elements = elements
        if info is None:
            info = {"elements": elements}
            if translations is not None:
                info["cached"] = {str(k): v for k, v in translations.items()}
        self.audit = _new_pdf_audit("t", minimum_scale, [])
        self.audit["merge_decisions"] = list(merge_decisions)
        self.context = assembly.PDFAssemblyContext(
            task_id="t",
            total_pages=total_pages,
            pdf_password="",
            use_htmlbox=use_htmlbox,
            fail_open_to_source_page=False,
            minimum_htmlbox_scale=minimum_scale,
            assembly_work_path=None,
            src_path="",
            out_doc=self.doc,
            page_extractions={0: info},
            pdf_audit=self.audit,
            font_path=str(font_path),
            bold_font_path=str(font_path),
            has_distinct_bold_font=False,
            font_archive=fitz.Archive(str(font_path.parent)),
            font_basename=font_path.name,
            logger=logging.getLogger("test-assembly"),
            filter_formula_safe_rects=_filter_pdf_formula_safe_rects,
            trim_process_memory=lambda: None,
        )

    def use_work_file(self, path, password=""):
        """Back the document by ``path`` the way HTML-box assembly does."""
        self.doc.save(str(path))
        self.doc.close()
        self.doc = fitz.open(str(path))
        if password:
            assert self.doc.authenticate(password)
        self.context.out_doc = self.doc
        self.context.assembly_work_path = str(path)

    def assemble(self):
        assembly._assemble_page(self.context, 0)
        return self.doc[0].get_text().replace("\n", "")

    def events(self, name):
        return self.audit[name]


# ---------------------------------------------------------------------------
# _prepare_page: which pages and elements are rendered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info",
    [
        {"source_page_fallback": {"page": 1}, "elements": [], "cached": {}},
        {"appendix": True, "elements": [], "cached": {}},
        {},
        {"elements": [_text([72, 100, 300, 114])]},
    ],
    ids=["source-page-fallback", "appendix", "no-layout", "no-translations"],
)
def test_prepare_page_skips_pages_without_work(font_path, info):
    page = Page(font_path, [], info=info)
    assert assembly._prepare_page(page.context, 0) is None


def test_prepare_page_skips_a_page_whose_translations_change_nothing(font_path):
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: SOURCE})
    assert assembly._prepare_page(page.context, 0) is None
    assert SOURCE in page.assemble()


def test_merged_away_text_is_redacted_without_a_translation(font_path):
    merged_away = _text([72, 100, 300, 114], type=ElementType.MERGED_AWAY)
    page = Page(font_path, [merged_away])
    plan = assembly._prepare_page(page.context, 0)
    assert plan is not None and plan.render_indices == set()
    assert SOURCE not in page.assemble()


def test_translated_text_replaces_its_source_through_the_base_htmlbox(font_path):
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]})
    text = page.assemble()
    assert ZH[:8] in text and SOURCE not in text
    (event,) = page.events("htmlbox_placement_events")
    assert event["step"] == "base" and event["scale"] == 1.0
    assert event["rect"] == [72.0, 100.0, 300.0, 114.0]
    assert page.events("render_scale_events") == []


def test_htmlbox_extends_the_rect_before_scaling(font_path):
    page = Page(font_path, [_text([72, 100, 160, 114])], {0: ZH})
    assert ZH in page.assemble()
    (event,) = page.events("htmlbox_placement_events")
    assert event["step"] != "base" and event["scale"] == 1.0


def test_table_cell_scales_inside_its_own_geometry(font_path):
    cell = _text([72, 100, 200, 114], table_hint=True)
    page = Page(font_path, [cell], {0: ZH[:16]})
    assert ZH[:16] in page.assemble()
    (scale_event,) = page.events("render_scale_events")
    assert 0.6 <= scale_event["scale"] < 1.0 and scale_event["table_cell"] is True
    (placement,) = page.events("htmlbox_placement_events")
    assert placement["step"] == "base" and placement["scale"] == scale_event["scale"]


def test_unreadable_table_cell_falls_back_to_the_textbox_and_then_source(font_path):
    cell = _text([72, 100, 110, 112], table_hint=True)
    page = Page(font_path, [cell], {0: ZH * 2})
    text = page.assemble()
    (rejected,) = page.events("rejected_render_scale_events")
    # PyMuPDF reports the floor it gave up at; nothing was drawn.
    assert rejected["scale"] <= 0.6 and rejected["inserted"] is False
    assert cell["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    assert SOURCE.replace(" ", "") in text.replace(" ", "")


def test_textbox_mode_records_the_fallback_fit(font_path):
    page = Page(font_path, [_text([72, 100, 300, 130])], {0: ZH[:8]}, use_htmlbox=False)
    assert ZH[:8] in page.assemble()
    (event,) = page.events("textbox_fallback_events")
    assert event["scale"] == 1.0 and event["rendered_fontsize"] == 10.0
    assert page.events("htmlbox_placement_events") == []


def test_textbox_mode_restores_source_when_the_translation_cannot_fit(font_path):
    elem = _text([72, 100, 300, 112])
    page = Page(font_path, [elem], {0: ZH * 30}, use_htmlbox=False)
    page.assemble()
    assert elem["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    (fallback,) = page.audit["element_source_fallbacks"]
    assert fallback["stage"] == "render" and "textbox fallback failed" in fallback["reason"]


def test_semantic_superscript_that_cannot_render_restores_source(font_path):
    cell = _text([72, 100, 110, 112], table_hint=True)
    page = Page(font_path, [cell], {0: ZH * 2 + "<sup>1</sup>"}, use_htmlbox=False)
    page.assemble()
    assert cell["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    (fallback,) = page.audit["element_source_fallbacks"]
    assert "semantic superscript could not be rendered" in fallback["reason"]


def test_preflight_keeps_source_ink_when_no_formula_safe_rect_exists(font_path):
    elem = _text([72, 100, 300, 114])
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [40, 60, 580, 300]}
    page = Page(font_path, [elem, formula], {0: ZH[:8]})
    plan = assembly._prepare_page(page.context, 0)
    assert plan is None
    assert elem["skip_translate_reason"] == SkipReason.RENDER_NO_SAFE_RECT_FALLBACK
    (fallback,) = page.audit["element_source_fallbacks"]
    assert fallback["stage"] == "render-preflight"


def test_preflight_rejects_an_unplaceable_cross_page_merge_endpoint(font_path):
    elem = _text([72, 100, 300, 114])
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [40, 60, 580, 300]}
    decision = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 0,
        "destination_page": 2,
        "destination_element": 3,
    }
    page = Page(
        font_path,
        [elem, formula],
        {0: ZH[:8]},
        total_pages=2,
        merge_decisions=[
            {"decision": "rejected"},
            "junk",
            {"decision": "accepted", "source_page": "x", "source_element": 0},
            decision,
        ],
    )
    with pytest.raises(PDFPageRenderError, match="source-page closure") as raised:
        assembly._prepare_page(page.context, 0)
    assert raised.value.fallback_pages == [1, 2]


def test_rotated_text_that_cannot_be_placed_keeps_source(font_path):
    elem = _text([300, 100, 314, 300], non_horizontal=True, rotation=90)
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [280, 80, 340, 320]}
    page = Page(font_path, [elem, formula], {0: ZH[:6]})
    assert assembly._prepare_page(page.context, 0) is None
    assert elem["skip_translate_reason"] == SkipReason.RENDER_NO_SAFE_RECT_FALLBACK


def test_rotated_text_renders_through_the_rotated_textbox(font_path):
    elem = _text([300, 100, 314, 300], non_horizontal=True, rotation=90)
    page = Page(font_path, [elem], {0: ZH[:6]})
    assert ZH[:6] in page.assemble()
    assert "skip_translate_reason" not in elem


def test_overlapping_unchanged_neighbor_is_redrawn(font_path):
    first = _text([72, 100, 300, 114])
    neighbor = _text([72, 112, 300, 126], content="Neighbor line stays English")
    page = Page(font_path, [first, neighbor], {0: ZH[:8], 1: neighbor["content"]})
    plan = assembly._prepare_page(page.context, 0)
    assert plan.redraw_source_indices == {1}
    text = page.assemble()
    assert ZH[:8] in text and "Neighbor" in text


def test_overlapping_preserved_table_forces_a_source_page(font_path):
    first = _text([72, 100, 300, 114])
    table = _text(
        [72, 112, 300, 126],
        content="Preserved table row",
        skip_translate_reason=SkipReason.TABLE_PRESERVED,
    )
    page = Page(font_path, [first, table], {0: ZH[:8]})
    with pytest.raises(PDFPageRenderError, match="table_preserved"):
        assembly._prepare_page(page.context, 0)


def test_redrawn_neighbor_inside_a_formula_has_no_htmlbox_rect(font_path):
    first = _text([72, 100, 300, 114])
    neighbor = _text([72, 112, 300, 126], content="Neighbor line stays English")
    # The formula starts just below the neighbor's top edge: every ladder
    # candidate collides and no trimmed piece is tall enough to use.
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [0, 112.2, 612, 400]}
    page = Page(font_path, [first, neighbor, formula], {0: ZH[:8]})
    with pytest.raises(PDFPageRenderError, match="no formula-safe htmlbox rectangle"):
        page.assemble()


def test_textbox_mode_without_a_safe_rect_raises(font_path):
    elem = _text([72, 100, 300, 114])
    formula = {"type": ElementType.FORMULA_IMAGE, "bbox": [40, 60, 580, 300]}
    page = Page(font_path, [elem, formula], {0: ZH[:8]}, use_htmlbox=False)
    with pytest.raises(PDFPageRenderError, match="no neighbor-safe textbox rectangle"):
        page.assemble()


def test_untranslated_english_is_recorded_as_a_leak(font_path):
    elem = _text([72, 100, 300, 114])
    page = Page(font_path, [elem], {0: "Other English words entirely"})
    page.assemble()
    (leak,) = page.events("untranslated_english_leaks")
    assert leak["element"] == 0 and leak["text"] == "Other English words entirely"


def test_leak_audit_failure_does_not_stop_assembly(font_path, monkeypatch):
    def broken(*_args):
        raise ValueError("boom")

    monkeypatch.setattr(assembly, "_pdf_collect_untranslated_english_leaks", broken)
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]})
    assert ZH[:8] in page.assemble()
    assert page.events("untranslated_english_leaks") == []


def test_watermarks_are_redacted_but_never_redrawn(font_path):
    mark = _text(
        [72, 100, 300, 114],
        content="CONFIDENTIAL DRAFT",
        skip_translate_reason=SkipReason.WATERMARK,
    )
    image_mark = {
        "type": ElementType.IMAGE,
        "bbox": [72, 200, 300, 214],
        "skip_translate_reason": SkipReason.WATERMARK,
    }
    page = Page(font_path, [mark, image_mark], {0: ZH[:4]})
    plan = assembly._prepare_page(page.context, 0)
    assert plan.translated_indices == set() and plan.watermark_indices == {0, 1}
    assert plan.render_indices == set()
    assert "CONFIDENTIAL" not in page.assemble()


@pytest.mark.parametrize("annotation_cleanup_fails", [False, True])
def test_failed_redaction_whites_out_the_exact_source_ink(
    font_path, monkeypatch, annotation_cleanup_fails
):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    monkeypatch.setattr(fitz.Page, "apply_redactions", refuse)
    if annotation_cleanup_fails:
        monkeypatch.setattr(fitz.Page, "delete_annot", refuse)
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]})
    assert ZH[:8] in page.assemble()
    white = [
        fitz.Rect(drawing["rect"])
        for drawing in page.doc[0].get_drawings()
        if drawing.get("fill") == (1.0, 1.0, 1.0)
    ]
    assert fitz.Rect(72, 100, 300, 114) in white


def test_render_failure_whose_source_cannot_be_restored_raises(font_path):
    elem = _text([72, 100, 100, 108], content=SOURCE * 20)
    page = Page(font_path, [elem], {0: ZH * 30}, use_htmlbox=False)
    with pytest.raises(PDFPageRenderError, match="textbox fallback failed"):
        page.assemble()


def test_render_failure_on_a_merge_endpoint_needs_the_whole_merge(font_path):
    elem = _text([72, 100, 300, 112])
    decision = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 0,
        "destination_page": 2,
        "destination_element": 0,
    }
    page = Page(
        font_path,
        [elem],
        {0: ZH * 30},
        use_htmlbox=False,
        total_pages=2,
        merge_decisions=[decision],
    )
    with pytest.raises(PDFPageRenderError, match="source-page closure") as raised:
        page.assemble()
    assert raised.value.fallback_pages == [1, 2]


def test_rotated_text_that_does_not_fit_falls_back_to_source(font_path):
    elem = _text([300, 100, 314, 140], content="Axis", non_horizontal=True, rotation=90)
    page = Page(font_path, [elem], {0: ZH * 10})
    page.assemble()
    assert elem["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    (fallback,) = page.audit["element_source_fallbacks"]
    assert "rotated textbox failed" in fallback["reason"]


def test_rotated_text_whose_source_cannot_be_restored_raises(font_path):
    elem = _text([300, 100, 304, 104], content=SOURCE * 10, non_horizontal=True, rotation=90)
    page = Page(font_path, [elem], {0: ZH * 10})
    with pytest.raises(PDFPageRenderError, match="rotated textbox failed"):
        page.assemble()


def test_translation_without_visible_text_restores_source(font_path):
    elem = _text([72, 100, 300, 114])
    page = Page(font_path, [elem], {0: "<br>"})
    page.assemble()
    assert elem["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    (fallback,) = page.audit["element_source_fallbacks"]
    assert "rendered no visible text" in fallback["reason"]


def test_invisible_translation_of_invisible_source_is_left_blank(font_path):
    elem = _text([72, 100, 300, 114], content=" ")
    page = Page(font_path, [elem], {0: "<br>"})
    page.assemble()
    assert "skip_translate_reason" not in elem
    assert page.audit.get("element_source_fallbacks", []) == []


def test_invisible_translation_whose_source_cannot_be_restored_raises(font_path):
    elem = _text([72, 100, 76, 103], content=SOURCE * 20)
    page = Page(font_path, [elem], {0: "<br>"})
    with pytest.raises(PDFPageRenderError, match="rendered no visible text"):
        page.assemble()


def test_textbox_mode_restores_source_for_a_markup_only_translation(font_path):
    elem = _text([72, 100, 300, 114])
    page = Page(font_path, [elem], {0: "<b> </b>"}, use_htmlbox=False)
    page.assemble()
    assert elem["skip_translate_reason"] == SkipReason.RENDER_SOURCE_RESTORE
    (fallback,) = page.audit["element_source_fallbacks"]
    assert "empty textbox fallback" in fallback["reason"]


def test_textbox_mode_leaves_a_markup_only_translation_of_blank_source(font_path):
    elem = _text([72, 100, 300, 114], content=" ")
    page = Page(font_path, [elem], {0: "<b> </b>"}, use_htmlbox=False)
    page.assemble()
    assert "skip_translate_reason" not in elem
    assert page.events("textbox_fallback_events") == []


def test_textbox_mode_markup_only_translation_unrestorable_source_raises(font_path):
    elem = _text([72, 100, 100, 108], content=SOURCE * 20)
    page = Page(font_path, [elem], {0: "<b> </b>"}, use_htmlbox=False)
    with pytest.raises(PDFPageRenderError, match="empty textbox fallback"):
        page.assemble()


def test_textbox_mode_shrinks_a_long_translation(font_path):
    page = Page(font_path, [_text([72, 100, 200, 112])], {0: ZH[:14]}, use_htmlbox=False)
    assert ZH[:14] in page.assemble()
    (event,) = page.events("textbox_fallback_events")
    assert event["scale"] < 1.0 and "fit after resize" in event["reason"]


def test_compact_fixed_label_may_shrink_below_the_textbox_floor(font_path):
    # "other" has a fixed table translation; only that exact label may drop
    # below the 5.5pt textbox floor to stay inside its cell.
    cell = _text([72, 100, 82, 106], content="other", table_hint=True, fontsize=7.0)
    page = Page(font_path, [cell], {0: "其他"}, use_htmlbox=False)
    page.assemble()
    (event,) = page.events("textbox_fallback_events")
    assert event["table_cell"] is True and event["rendered_fontsize"] < 5.5


def test_superscript_whose_source_cannot_be_restored_raises(font_path):
    cell = _text([72, 100, 76, 103], content=SOURCE * 20, table_hint=True)
    page = Page(font_path, [cell], {0: ZH * 2 + "<sup>1</sup>"}, use_htmlbox=False)
    with pytest.raises(PDFPageRenderError, match="semantic superscript could not be rendered"):
        page.assemble()


def test_htmlbox_exception_falls_back_to_the_textbox(font_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("htmlbox engine failure")

    monkeypatch.setattr(fitz.Page, "insert_htmlbox", explode)
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]})
    assert ZH[:8] in page.assemble()
    assert page.events("htmlbox_placement_events") == []
    (event,) = page.events("textbox_fallback_events")
    assert 0.6 <= event["scale"] <= 1.0


def test_htmlbox_scale_below_the_floor_is_a_hard_failure(font_path, monkeypatch):
    monkeypatch.setattr(
        assembly, "_insert_pdf_htmlbox_at_readable_scale", lambda *_a, **_k: (1.0, 0.3)
    )
    cell = _text([72, 100, 110, 112], table_hint=True)
    page = Page(font_path, [cell], {0: ZH * 2})
    with pytest.raises(PDFPageRenderError, match="unreadable htmlbox scale 0.30") as raised:
        page.assemble()
    assert raised.value.retryable is False


def test_readable_scale_exception_falls_back_to_the_textbox(font_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise ValueError("scale failure")

    monkeypatch.setattr(assembly, "_insert_pdf_htmlbox_at_readable_scale", explode)
    cell = _text([72, 100, 200, 114], table_hint=True)
    page = Page(font_path, [cell], {0: ZH[:16]})
    assert ZH[:16] in page.assemble()
    (event,) = page.events("textbox_fallback_events")
    assert event["table_cell"] is True


def test_textbox_scale_below_the_floor_is_a_hard_failure(font_path, monkeypatch):
    monkeypatch.setattr(
        assembly, "_insert_pdf_textbox_with_fit", lambda *_a, **_k: (True, "spare=0", 3.0)
    )
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]}, use_htmlbox=False)
    with pytest.raises(PDFPageRenderError, match="unreadable textbox scale 0.30") as raised:
        page.assemble()
    assert raised.value.retryable is False


def test_htmlbox_assembly_checkpoints_and_reopens_the_work_file(font_path, tmp_path):
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]}, total_pages=2)
    page.use_work_file(tmp_path / "work.pdf")
    first = page.context.out_doc
    assembly._assemble_page(page.context, 0)
    reopened = page.context.out_doc
    assert reopened is not None and reopened is not first
    assert ZH[:8] in reopened[0].get_text().replace("\n", "")
    reopened.close()
    with fitz.open(str(tmp_path / "work.pdf")) as saved:
        assert ZH[:8] in saved[0].get_text().replace("\n", "")


def test_last_page_checkpoint_leaves_no_open_document(font_path, tmp_path):
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]})
    page.use_work_file(tmp_path / "work.pdf")
    assembly._assemble_page(page.context, 0)
    assert page.context.out_doc is None


def test_reopening_an_encrypted_work_file_with_a_wrong_password_fails(font_path, tmp_path):
    page = Page(font_path, [_text([72, 100, 300, 114])], {0: ZH[:8]}, total_pages=2)
    encrypted = tmp_path / "encrypted.pdf"
    page.doc.save(
        str(encrypted),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    page.doc.close()
    page.doc = fitz.open(str(encrypted))
    assert page.doc.authenticate("secret")
    page.context.out_doc = page.doc
    page.context.assembly_work_path = str(encrypted)
    page.context.pdf_password = "wrong"
    with pytest.raises(RuntimeError, match="PDF密码错误"):
        assembly._assemble_page(page.context, 0)
    assert page.context.out_doc is None
