"""Local final-structure warnings deliver the translated page.

A page whose confirmed final structure warnings are all on the explicit
element-level allowlist is delivered as translated and its warnings are
recorded.  Any other type, an unknown type, or a page mixing both restores the
exact source page as before; page-less warnings still fail the task.  Every
document here is synthetic.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import copy
import logging
import shutil
import threading
from functools import partial
from types import SimpleNamespace

import fitz
import pytest
from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _cell,
    _formula_over_right_edge,
    _squash,
    font_path,  # noqa: F401  (module fixture)
)

from phoena_translator.pdf.errors import PDFStructureValidationError
from phoena_translator.pdf.extract.signatures import _pdf_formula_region_signature
from phoena_translator.pdf.model import SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.render import primitives, stage
from phoena_translator.pdf.verify import output
from phoena_translator.pdf.verify.output import PDF_LOCAL_STRUCTURE_WARNING_TYPES

# One synthetic warning per allowlisted type, shaped like the checker's.
LOCAL = {
    "formula-integrity-mismatch": {
        "element": 4,
        "mismatches": ["text_sha256", "font_sha256", "raster_sha256"],
        "max_delta": 120,
        "diff_ratio": 0.25,
    },
    "superscript-size-mismatch": {"marker": "3", "expected_scale": 0.8, "rendered_scale": 0.6},
    "flattened-superscript": {
        "marker": "2",
        "expected": 1,
        "rendered_superscript": 0,
        "plain_occurrences": 1,
    },
    "missing-superscript": {
        "marker": "2",
        "expected": 1,
        "rendered_superscript": 0,
        "plain_occurrences": 0,
    },
    "superscript-scale-unverifiable": {"marker": "4", "expected": 1, "measured": 0},
    "vector-ocr-source-text-remains": {"element": 7, "phrases": ["Axis label"]},
    "formula-signature-error": {"element": 5, "error_type": "ValueError"},
    "formula-raster-clip-degenerate": {"element": 6},
    "source-formula-signature-missing": {"element": 8},
}
RESTORING = {
    "font-glyph-substitution": {"source_count": 0, "output_count": 2},
    "vector-ocr-audit-error": {"error_type": "RuntimeError"},
    "a-type-added-later": {"element": 1},
}


def _local(kind, page):
    return {"type": kind, "page": page, **LOCAL[kind]}


def _pdfs(tmp_path, pages):
    paths = []
    for name, label in (("source", "SOURCE"), ("out", "TRANSLATED")):
        document = fitz.open()
        for number in range(1, pages + 1):
            document.new_page().insert_text((72, 72), f"{label} PAGE {number}")
        path = tmp_path / f"{name}.pdf"
        document.save(str(path))
        document.close()
        paths.append(str(path))
    return paths


def _translated_pages(out_path):
    with fitz.open(out_path) as document:
        return [n for n, page in enumerate(document, 1) if "TRANSLATED" in page.get_text()]


class Checker:
    """The final audit of a fixed set of flaws: a flaw is reported while its
    page is still the translated page and is gone once the page is restored."""

    def __init__(self, flaws, extra=(), status=None, count_bias=0, after_restore=None):
        self.flaws, self.extra = list(flaws), list(extra)
        self.status, self.count_bias, self.after_restore = status, count_bias, after_restore
        self.calls = 0

    def __call__(self, out_path, *_args, **_kwargs):
        self.calls += 1
        live = set(_translated_pages(out_path))
        if self.after_restore is not None and self.calls > 1:
            warnings = [dict(w) for w in self.after_restore]
        else:
            warnings = [dict(w) for w in self.flaws if w.get("page") in live]
        warnings += [dict(w) for w in self.extra]
        status = self.status or ("warning" if warnings else "ok")
        if not warnings:
            status = "ok"
        return {
            "status": status,
            "warning_count": len(warnings) + (self.count_bias if warnings else 0),
            "warnings": warnings,
        }


def _run(
    tmp_path, monkeypatch, checker, *, pages=3, merges=(), audit=None, fail_open=True,
    page_extractions=None,
):
    source, out = _pdfs(tmp_path, pages)
    audit = audit if audit is not None else _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = [
        {
            "decision": "accepted",
            "source_page": a,
            "source_element": 0,
            "destination_page": b,
            "destination_element": 0,
        }
        for a, b in merges
    ]
    run = SimpleNamespace(record=None, audit=audit, out=out, checker=checker, saved=[])

    def save_audit(_task_id, completed):
        run.saved.append(copy.deepcopy(completed))
        output._require_clean_pdf_structure_check(completed.get("structure_check"))
        return "audit.json"

    def save_progress(_task_id, record):
        run.record = record

    monkeypatch.setattr(stage, "_save_and_validate_pdf_audit", save_audit)
    context = SimpleNamespace(
        task_id="t",
        out_path=out,
        src_path=source,
        pdf_password="",
        total_pages=pages,
        pdf_audit=audit,
        page_extractions=page_extractions if page_extractions is not None else {},
        logger=logging.getLogger("test-local-structure-warnings"),
        trim_process_memory=lambda: None,
        check_output_structure_serialized=checker,
        save_progress=save_progress,
    )
    run.context = context
    stage.finalize_output(context, fail_open_to_source_page=fail_open)
    return run


def _record_of(warning):
    return {key: value for key, value in warning.items() if key != "phrases"}


def _fallback_pages(run):
    return [entry["page"] for entry in run.audit["source_page_fallbacks"]]


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------


def test_the_allowlist_is_exactly_the_named_element_level_types():
    assert PDF_LOCAL_STRUCTURE_WARNING_TYPES == frozenset(LOCAL)
    assert not PDF_LOCAL_STRUCTURE_WARNING_TYPES & set(RESTORING)


@pytest.mark.parametrize("kind", sorted(LOCAL))
def test_a_local_warning_delivers_the_translated_page_and_is_recorded(
    tmp_path, monkeypatch, kind
):
    flaw = _local(kind, 2)
    run = _run(tmp_path, monkeypatch, Checker([flaw]))

    assert _translated_pages(run.out) == [1, 2, 3]
    assert run.record["completion_status"] == "completed"
    assert run.record["source_page_fallbacks"] == []
    assert run.record["delivered_local_warnings"] == [_record_of(flaw)]
    assert run.audit["delivered_local_warnings"] == [_record_of(flaw)]
    assert run.audit.get("quality_status") != "completed_with_source_page_fallbacks"
    # Moved, not dropped: the check keeps the verbatim warning.
    check = run.audit["structure_check"]
    assert (check["status"], check["warning_count"], check["warnings"]) == ("ok", 0, [])
    assert check["delivered_local_warnings"] == [flaw]
    assert run.saved[-1]["delivered_local_warnings"] == [_record_of(flaw)]
    assert run.checker.calls == 1


def test_a_formula_record_names_the_differing_fingerprints(tmp_path, monkeypatch):
    flaw = _local("formula-integrity-mismatch", 1)
    run = _run(tmp_path, monkeypatch, Checker([flaw]))
    (record,) = run.record["delivered_local_warnings"]
    assert (record["page"], record["element"]) == (1, 4)
    assert record["mismatches"] == ["text_sha256", "font_sha256", "raster_sha256"]


@pytest.mark.parametrize("kind", [*sorted(RESTORING), None])
def test_any_other_or_unknown_type_restores_the_page(tmp_path, monkeypatch, kind):
    flaw = {"page": 2, **RESTORING.get(kind, {})}
    if kind:
        flaw["type"] = kind
    run = _run(tmp_path, monkeypatch, Checker([flaw]))

    assert _translated_pages(run.out) == [1, 3]
    (entry,) = run.record["source_page_fallbacks"]
    assert (entry["page"], entry["stage"]) == (2, "structure")
    assert entry["reason"] == "confirmed final structure audit: " + (kind or "unknown")
    assert run.record["completion_status"] == "completed_with_source_page_fallbacks"
    assert "delivered_local_warnings" not in run.record
    assert "delivered_local_warnings" not in run.audit


def test_a_page_with_one_local_and_one_other_warning_is_restored(tmp_path, monkeypatch):
    flaws = [_local("formula-integrity-mismatch", 2), {"type": "font-glyph-substitution", "page": 2}]
    run = _run(tmp_path, monkeypatch, Checker(flaws))

    assert _translated_pages(run.out) == [1, 3]
    (entry,) = run.record["source_page_fallbacks"]
    assert entry["reason"] == (
        "confirmed final structure audit: font-glyph-substitution, formula-integrity-mismatch"
    )
    assert "delivered_local_warnings" not in run.record


def test_restoring_one_page_still_delivers_a_local_page(tmp_path, monkeypatch):
    flaws = [_local("superscript-size-mismatch", 1), {"type": "a-type-added-later", "page": 3}]
    run = _run(tmp_path, monkeypatch, Checker(flaws))

    assert _translated_pages(run.out) == [1, 2]
    assert _fallback_pages(run) == [3]
    assert run.record["completion_status"] == "completed_with_source_page_fallbacks"
    assert run.record["delivered_local_warnings"] == [_record_of(flaws[0])]
    (before,) = run.audit["structure_checks_before_source_fallback"]
    assert before["warnings"] == flaws
    assert run.checker.calls == 2


# ---------------------------------------------------------------------------
# Cross-page merges follow the page
# ---------------------------------------------------------------------------


def test_a_delivered_page_drags_no_merge_partner_back(tmp_path, monkeypatch):
    run = _run(
        tmp_path, monkeypatch, Checker([_local("formula-integrity-mismatch", 2)]), merges=[(2, 3)]
    )
    assert _translated_pages(run.out) == [1, 2, 3]
    assert _fallback_pages(run) == []
    assert [w["page"] for w in run.record["delivered_local_warnings"]] == [2]


def test_a_restored_page_still_drags_its_merge_partner_back(tmp_path, monkeypatch):
    flaws = [_local("formula-integrity-mismatch", 2), {"type": "font-glyph-substitution", "page": 3}]
    run = _run(tmp_path, monkeypatch, Checker(flaws), merges=[(2, 3)])

    assert _translated_pages(run.out) == [1]
    assert _fallback_pages(run) == [2, 3]
    reasons = {entry["page"]: entry["reason"] for entry in run.audit["source_page_fallbacks"]}
    assert reasons[2] == (
        "confirmed final structure audit expanded through accepted cross-page merge "
        "linked to warning page(s) 3"
    )
    assert reasons[3] == "confirmed final structure audit: font-glyph-substitution"
    assert "delivered_local_warnings" not in run.audit


def test_expectations_stay_for_delivered_pages_and_leave_restored_ones(tmp_path, monkeypatch):
    audit = _new_pdf_audit("t", 0.6, [])
    audit["formula_expectations"] = [{"page": n, "element": 0, "signature": {}} for n in (1, 2, 3)]
    flaws = [_local("formula-integrity-mismatch", 1), {"type": "font-glyph-substitution", "page": 3}]
    run = _run(tmp_path, monkeypatch, Checker(flaws), audit=audit)

    assert [e["page"] for e in run.audit["formula_expectations"]] == [1, 2]
    assert run.audit["source_page_fallback_expectation_exclusions"] == {"formula_expectations": 1}
    assert [w["page"] for w in run.audit["delivered_local_warnings"]] == [1]


# ---------------------------------------------------------------------------
# Everything else keeps today's behaviour
# ---------------------------------------------------------------------------


def _watermarked(pages):
    """A removed distribution code (element 1) on every page; page 1 also
    carries an older-detector watermark (element 0) that has no record."""
    audit = _new_pdf_audit("t", 0.6, [])
    audit["removed_watermarks"] = [
        {"page": page, "element": 1, "bbox": [-3.0, 783.0, 70.0, 795.5],
         "text": "Zq4Wm8Xr2Lc7Hn", "kind": "distribution_code"}
        for page in range(1, pages + 1)
    ]
    extractions = {
        n: {"elements": [
            {"type": "text", "content": "DRAFT",
             **({"skip_translate_reason": SkipReason.WATERMARK} if n == 0 else {})},
            {"type": "text", "content": "Zq4Wm8Xr2Lc7Hn", "skip_translate_reason": SkipReason.WATERMARK},
        ]}
        for n in range(pages)
    }
    return audit, extractions


def test_a_page_restored_to_source_has_no_watermark_removal_record(tmp_path, monkeypatch):
    # R5: page 1 is restored by the final audit, so its code is visible again.
    audit, extractions = _watermarked(2)
    checker = Checker([{"type": "font-glyph-substitution", "page": 1}])

    run = _run(tmp_path, monkeypatch, checker, pages=2, audit=audit, page_extractions=extractions)

    assert _fallback_pages(run) == [1]
    assert [r["page"] for r in run.saved[-1]["removed_watermarks"]] == [2]
    assert [r["page"] for r in run.record["removed_watermarks"]] == [2]
    assert "skip_translate_reason" not in extractions[0]["elements"][1]
    assert extractions[0]["elements"][0]["skip_translate_reason"] == SkipReason.WATERMARK
    assert extractions[1]["elements"][1]["skip_translate_reason"] == SkipReason.WATERMARK


def test_no_watermark_removal_is_saved_when_every_page_is_restored(tmp_path, monkeypatch):
    audit, extractions = _watermarked(2)
    checker = Checker([{"type": "font-glyph-substitution", "page": n} for n in (1, 2)])

    run = _run(tmp_path, monkeypatch, checker, pages=2, audit=audit, page_extractions=extractions)

    assert _fallback_pages(run) == [1, 2]
    assert "removed_watermarks" not in run.saved[-1]
    assert "removed_watermarks" not in run.record


def _raises(tmp_path, monkeypatch, checker, **options):
    audit = _new_pdf_audit("t", 0.6, [])
    with pytest.raises(PDFStructureValidationError):
        _run(tmp_path, monkeypatch, checker, audit=audit, **options)
    assert "delivered_local_warnings" not in audit
    return audit


def test_a_page_less_warning_still_fails_the_task(tmp_path, monkeypatch):
    checker = Checker(
        [_local("formula-integrity-mismatch", 1)],
        extra=[{"type": "page-count-mismatch", "page": 0}],
    )
    audit = _raises(tmp_path, monkeypatch, checker)
    assert audit["source_page_fallbacks"] == []
    assert audit["structure_check"]["warning_count"] == 2


def test_strict_mode_still_fails_on_a_local_warning(tmp_path, monkeypatch):
    _raises(
        tmp_path, monkeypatch, Checker([_local("superscript-size-mismatch", 1)]), fail_open=False
    )


def test_a_local_warning_left_on_a_restored_page_is_never_delivered(tmp_path, monkeypatch):
    checker = Checker(
        [{"type": "font-glyph-substitution", "page": 2}],
        after_restore=[_local("formula-integrity-mismatch", 2)],
    )
    audit = _raises(tmp_path, monkeypatch, checker)
    assert [entry["page"] for entry in audit["source_page_fallbacks"]] == [2]


def test_a_checker_that_did_not_finish_restores_as_before(tmp_path, monkeypatch):
    checker = Checker([_local("formula-signature-error", 1)], status="error")
    run = _run(tmp_path, monkeypatch, checker)
    assert _fallback_pages(run) == [1]
    assert "delivered_local_warnings" not in run.audit


def test_an_inconsistent_warning_count_restores_as_before(tmp_path, monkeypatch):
    checker = Checker([_local("formula-integrity-mismatch", 1)], count_bias=1)
    run = _run(tmp_path, monkeypatch, checker)
    assert _fallback_pages(run) == [1]
    assert "delivered_local_warnings" not in run.audit


def test_a_stale_delivered_list_from_an_earlier_attempt_is_dropped(tmp_path, monkeypatch):
    audit = _new_pdf_audit("t", 0.6, [])
    audit["delivered_local_warnings"] = [_record_of(_local("formula-integrity-mismatch", 1))]
    run = _run(tmp_path, monkeypatch, Checker([]), audit=audit)
    assert "delivered_local_warnings" not in run.audit
    assert "delivered_local_warnings" not in run.record


def _confirming(first, second):
    results = iter([first, second])
    dependencies = SimpleNamespace(
        extraction_max_concurrency=1,
        extraction_semaphore=threading.Semaphore(1),
        trim_process_memory=lambda: None,
        check_output_structure=lambda *_a, **_k: copy.deepcopy(next(results)),
    )
    return partial(output._check_pdf_output_structure_serialized, dependencies=dependencies)


def test_the_two_pass_confirmation_still_discards_a_transient_warning(tmp_path, monkeypatch):
    flaw = _local("formula-integrity-mismatch", 1)
    warned = {"status": "warning", "warning_count": 1, "warnings": [flaw]}
    clean = {"status": "ok", "warning_count": 0, "warnings": []}

    run = _run(tmp_path, monkeypatch, _confirming(warned, clean))
    assert run.audit["structure_check"]["transient_initial_warnings_discarded"] is True
    assert "delivered_local_warnings" not in run.audit

    (tmp_path / "again").mkdir()
    run = _run(tmp_path / "again", monkeypatch, _confirming(warned, warned))
    check = run.audit["structure_check"]
    assert check["initial_warning_count"] == 1
    assert check["delivered_local_warnings"] == [flaw]
    assert run.audit["delivered_local_warnings"] == [_record_of(flaw)]


# ---------------------------------------------------------------------------
# The source restore never writes into a protected formula region
# ---------------------------------------------------------------------------


def test_a_restore_that_would_cross_a_formula_keeps_the_element_in_source_ink(
    tmp_path, font_path, monkeypatch, caplog  # noqa: F811
):
    # A cell whose right edge is a protected formula region with visible
    # formula ink.  Its translation does not fit and every formula-safe
    # textbox is refused, so only the direct re-typeset of the source text is
    # left; the cell is tall enough for that walk to fit its own box, which
    # crosses the formula's unredacted ink.  Without the guard the element is
    # re-typeset instead of kept in its exact source ink.
    monkeypatch.setattr(
        primitives,
        "_insert_pdf_textbox_with_fit",
        lambda *_a, **_k: (False, "no formula-safe text rect", None),
    )
    cell = _cell(330, 210, "Coupon", width=40.0, table_hint=True)
    cell["bbox"] = cell["rect"] = [330.0, 204.0, 370.0, 216.0]
    formula = _formula_over_right_edge(cell)
    elements = [_body(100), cell, _cell(72, 210, "Ticker"), _body(400), formula]
    translations = {0: ZH[:30], 1: ZH * 4, 2: "代码", 3: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])
    document.context.out_doc.close()
    with fitz.open(str(document.source_path)) as source:
        source[0].insert_text((366, 214), "x+y=2", fontsize=7)
        source.saveIncr()
    shutil.copy2(document.source_path, document.work_path)
    document.context.out_doc = fitz.open(str(document.work_path))
    with fitz.open(str(document.source_path)) as source:
        formula_before = _pdf_formula_region_signature(source[0], formula["bbox"])
    assert formula_before["span_count"] >= 1  # the protected region holds ink

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert ZH[:30] in _squash(text) and "代码" in text
    (entry,) = [
        e for e in document.audit["element_source_fallbacks"] if e.get("source_ink_kept")
    ]
    assert (entry["page"], entry["element"]) == (1, 1)
    assert "exact source ink kept in place" in entry["reason"]
    assert "source restore would cross a protected formula region" in caplog.text
    assert elements[1]["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    with fitz.open(str(document.work_path)) as out:
        formula_after = _pdf_formula_region_signature(out[0], formula["bbox"])
    for key in ("text_sha256", "font_sha256", "geometry_sha256", "span_count"):
        assert formula_after[key] == formula_before[key]
