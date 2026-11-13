"""Ruling R-19: the sign check never costs more than its element.

An element whose translation fails the sign check (``prose-number-missing``)
keeps its source through the path the stacked-cell rule uses, and is left to
today's path (drawn as the live release draws it) whenever keeping it would
cost more than the element: the page left with nothing translated, the
per-page limit, a
neighbour's removal touching its ink, any refusal of the confinement.  Such
a waiver is recorded once in ``delivered_element_warnings`` by the assembly
stage only, and only while the translation is in the delivered page: when
the element or its page is put back to source, the warning goes.
All data is synthetic.
"""

from __future__ import annotations

import fitz

from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.render import assembly
from phoena_translator.pdf.render.source_fallbacks import (
    _drop_pdf_partial_delivery_records,
    _record_pdf_element_source_fallback,
)
from phoena_translator.pdf.translate.numeric_guard import (
    PDF_DELIVERED_ELEMENT_WARNINGS,
    PDF_PROSE_NUMBERS_REASON,
    record_pdf_delivered_prose_loss,
)
from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_local_structure_warnings import RESTORING, Checker, _run, _translated_pages
from test_pdf_prose_number_guard import (
    LOSSY,
    LOSSY_FORM,
    PROSE,
    _prose,
    _provider,
    _run_translation,
    _signs,
)
from test_pdf_render_element_fallback import _kept_records


def _audit_with_warnings(*pages):
    audit = _new_pdf_audit("t", 0.6, [])
    for page in pages:
        record_pdf_delivered_prose_loss(audit, page, 4, "render", "Fees rose +$7.5bn")
    return audit


def test_the_key_is_the_one_the_record_and_the_cleanup_use():
    assert PDF_DELIVERED_ELEMENT_WARNINGS == "delivered_element_warnings"


def test_the_cleanup_drops_the_warnings_of_source_pages_only():
    audit = _audit_with_warnings(1, 2)

    _drop_pdf_partial_delivery_records(audit, [1])

    assert [w["page"] for w in audit[PDF_DELIVERED_ELEMENT_WARNINGS]] == [2]

    _drop_pdf_partial_delivery_records(audit, {2})

    assert PDF_DELIVERED_ELEMENT_WARNINGS not in audit


def test_the_record_shows_the_warning_of_a_delivered_page(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, Checker([]), audit=_audit_with_warnings(2))

    assert _translated_pages(run.out) == [1, 2, 3]
    assert run.record["completion_status"] == "completed"
    (warning,) = run.record[PDF_DELIVERED_ELEMENT_WARNINGS]
    assert (warning["page"], warning["element"], warning["reason"]) == (
        2, 4, PDF_PROSE_NUMBERS_REASON
    )
    assert run.saved[-1][PDF_DELIVERED_ELEMENT_WARNINGS] == [warning]


def test_a_record_without_a_warning_keeps_its_shape(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, Checker([]))

    assert PDF_DELIVERED_ELEMENT_WARNINGS not in run.record


def test_a_page_later_restored_whole_keeps_no_warning(tmp_path, monkeypatch):
    # Page 2 carries a delivered prose-number warning; the final structure
    # audit then restores page 2 whole for another reason.  Page 3 is still
    # delivered as translated and keeps its own warning.
    flaw = {"type": "font-glyph-substitution", "page": 2, **RESTORING["font-glyph-substitution"]}
    run = _run(tmp_path, monkeypatch, Checker([flaw]), audit=_audit_with_warnings(2, 3))

    assert _translated_pages(run.out) == [1, 3]
    assert [entry["page"] for entry in run.record["source_page_fallbacks"]] == [2]
    assert [w["page"] for w in run.record[PDF_DELIVERED_ELEMENT_WARNINGS]] == [3]
    assert [w["page"] for w in run.audit[PDF_DELIVERED_ELEMENT_WARNINGS]] == [3]


def test_a_document_whose_warned_page_is_restored_records_no_warning(tmp_path, monkeypatch):
    flaw = {"type": "font-glyph-substitution", "page": 2, **RESTORING["font-glyph-substitution"]}
    run = _run(tmp_path, monkeypatch, Checker([flaw]), audit=_audit_with_warnings(2))

    assert PDF_DELIVERED_ELEMENT_WARNINGS not in run.record
    assert PDF_DELIVERED_ELEMENT_WARNINGS not in run.audit


def test_a_warning_is_recorded_once_and_goes_with_an_element_put_back_to_source():
    audit = {}
    for _ in range(2):
        record_pdf_delivered_prose_loss(audit, 1, 2, "render", "Up +$1.5bn", why="limit")
    record_pdf_delivered_prose_loss(audit, 1, 3, "render")
    assert [(w["element"], w.get("waived")) for w in audit[PDF_DELIVERED_ELEMENT_WARNINGS]] == [
        (2, "limit"),
        (3, None),
    ]
    _record_pdf_element_source_fallback(audit, 1, 2, "render", "restored")
    assert [w["element"] for w in audit[PDF_DELIVERED_ELEMENT_WARNINGS]] == [3]
    _record_pdf_element_source_fallback(audit, 1, 3, "render", "restored")
    assert PDF_DELIVERED_ELEMENT_WARNINGS not in audit


# --- the translation stage does not check (R-20) ----------------------------


def test_translation_keeps_the_only_translation_of_a_page_and_writes_no_warning(monkeypatch):
    context, elements, saved, calls = _run_translation(
        monkeypatch, _provider(LOSSY_FORM), elements=[_prose()]
    )

    assert saved[0] == {0: LOSSY}  # as the live release keeps it
    assert "skip_translate_reason" not in elements[0]
    assert context.pdf_audit["source_page_fallbacks"] == []
    assert context.pdf_audit.get("element_source_fallbacks", []) == []
    assert PDF_DELIVERED_ELEMENT_WARNINGS not in context.pdf_audit
    assert len(calls) == 1  # as the live release: no retry


# --- the assembly net --------------------------------------------------------


def _lossy_element(bbox, content=PROSE, translation=LOSSY):
    signs = "+" * content.count("+$")
    return _text(bbox, content, layout_class="body", inline_math_fragments=_signs(*signs)), translation


def _only_warning(document, element):
    (warning,) = document.audit[PDF_DELIVERED_ELEMENT_WARNINGS]
    assert (warning["page"], warning["element"], warning["stage"]) == (1, element, "render")
    assert warning["reason"] == PDF_PROSE_NUMBERS_REASON
    return warning


def test_the_only_translation_of_a_page_is_drawn_with_one_warning(tmp_path, font_path):
    prose, lossy = _lossy_element([72, 200, 540, 240])
    document = Document(tmp_path, font_path, [([prose], {0: lossy}), ([_body(100)], {0: ZH[:30]})])

    (text, _second) = document.assemble()

    assert document.fallback_pages() == []
    assert "机床订单" in _squash(text)
    assert _kept_records(document) == []
    assert "no other translation" in _only_warning(document, 0)["waived"]


def test_beyond_the_per_page_limit_the_translation_is_drawn_with_one_warning(tmp_path, font_path):
    limit = assembly.ELEMENT_SOURCE_MAX_PER_PAGE
    rows = [
        _lossy_element([72, 60 + 30 * n, 540, 80 + 30 * n], f"Orders rose +${n + 3}.5bn this week.", "本周订单增加+。")
        for n in range(limit + 1)
    ]
    tail = _text([72, 720, 540, 740], "A closing remark without figures.", layout_class="body")
    translations = {n: translation for n, (_elem, translation) in enumerate(rows)}
    translations[limit + 1] = "结束语。"
    document = Document(tmp_path, font_path, [([*(e for e, _t in rows), tail], translations)])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    kept = {entry["element"] for entry in _kept_records(document)}
    assert kept == set(range(limit))
    assert "kept in source on the page" in _only_warning(document, limit)["waived"]
    assert "结束语" in _squash(text) and "本周订单增加" in _squash(text)


# F2 (second reader, round 2): the kept element's neighbour shares its line.
FEE_BOX = [72.0, 300.32501220703125, 148.54498291015625, 312.6910095214844]


def _fee_document(tmp_path, font_path, fee, fee_box, translation, signs=""):
    note_x = fee_box[2] - 0.2
    elements = [
        _body(100),
        _text(fee_box, fee, fontsize=9.0, layout_class="body", inline_math_fragments=_signs(*signs)),
        _text([note_x, fee_box[1], note_x + 60, fee_box[3]], "Closing note.", fontsize=9.0, layout_class="body"),
        _body(500),
    ]
    translations = {0: "开篇介绍当地前景。", 1: translation, 2: "结束语。", 3: "最后的评论结束了这份简报。"}
    return Document(tmp_path, font_path, [(elements, translations)])


def test_f2_as_written_is_delivered(tmp_path, font_path):
    document = _fee_document(tmp_path, font_path, "Required fee: $1.2.", FEE_BOX, "需要缴费。")

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "需要缴费" in _squash(text) and "结束语" in _squash(text)
    assert PDF_DELIVERED_ELEMENT_WARNINGS not in document.audit


def test_f2_with_a_protected_sign_is_delivered_with_one_warning(tmp_path, font_path):
    box = [72.0, FEE_BOX[1], 153.5, FEE_BOX[3]]
    document = _fee_document(tmp_path, font_path, "Required fee: +$1.2.", box, "需要缴费+。", "+")

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "需要缴费" in _squash(text) and "结束语" in _squash(text)
    assert _kept_records(document) == []
    assert "neighbour elem 2" in _only_warning(document, 1)["waived"]


def test_a_refusal_after_planning_puts_the_element_back_on_todays_path(tmp_path, font_path, monkeypatch):
    # Text redaction fails: keeping an element in source is refused after
    # planning (the whiteout path); the page is redone without the keep.
    monkeypatch.setattr(assembly, "_remove_source_ink", lambda *args, **kwargs: False)
    prose, lossy = _lossy_element([72, 200, 540, 240])
    document = Document(tmp_path, font_path, [([_body(100), prose, _body(400)], {0: ZH[:30], 1: lossy, 2: ZH[5:30]})])

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "机床订单" in _squash(text)
    assert _kept_records(document) == []
    assert "refused" in _only_warning(document, 1)["waived"]


# F6 (second reader, round 2): the drawn translation is later restored to
# source.  The test font lets the long translation fit, so the no-fit is
# injected; the restore itself is the real one.
def _f6_document(tmp_path, font_path, monkeypatch, content, signs=""):
    element = _text([72, 300, 350, 324], content, fontsize=10.0, layout_class="body", inline_math_fragments=_signs(*signs))

    def render(context, plan, css, index):
        element_plan = assembly._prepare_element_render(context, plan, index)
        assembly._restore_planned_source_or_raise(context, plan, element_plan, "injected no-fit")

    monkeypatch.setattr(assembly, "_render_element", render)
    return Document(tmp_path, font_path, [([element], {0: "报告期内收入增长+。" * 600})])


def _assert_restored_without_warning(document, content):
    document.assemble()
    with fitz.open(str(document.work_path)) as output:
        assert content in output[0].get_text().replace("\n", " ")
    (entry,) = document.audit["element_source_fallbacks"]
    assert "injected no-fit" in entry["reason"]
    assert PDF_DELIVERED_ELEMENT_WARNINGS not in document.audit


def test_f6_as_written_records_no_warning(tmp_path, font_path, monkeypatch):
    content = "Revenue was $43m in the reporting period."
    _assert_restored_without_warning(_f6_document(tmp_path, font_path, monkeypatch, content), content)


def test_f6_a_waived_element_restored_to_source_keeps_no_warning(tmp_path, font_path, monkeypatch):
    content = "Revenue was +$43m in the reporting period."
    document = _f6_document(tmp_path, font_path, monkeypatch, content, "+")
    recorded = []
    real = assembly._record_stacked_table_cell_waivers

    def spy(context, page_num, elements, waivers):
        real(context, page_num, elements, waivers)
        recorded.append(len(context.pdf_audit.get(PDF_DELIVERED_ELEMENT_WARNINGS, [])))

    monkeypatch.setattr(assembly, "_record_stacked_table_cell_waivers", spy)
    _assert_restored_without_warning(document, content)
    assert recorded[-1] == 1  # waived and warned at planning, then withdrawn
