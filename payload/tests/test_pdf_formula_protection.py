"""Formula-protection preflight: fail closed, or preserve warning pages as
exact source until the formula audit is clean.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import json
import logging
import threading

import fitz
import pytest

from phoena_translator.pdf import cache
from phoena_translator.pdf.context import ImmutableElementPolicy
from phoena_translator.pdf.extract import stage
from phoena_translator.pdf.model import ElementType


def _page(page_number):
    return {
        "elements": [
            {
                "type": ElementType.TEXT,
                "content": f"Ordinary prose on page {page_number} that needs translation.",
                "bbox": [72.0, 100.0, 500.0, 120.0],
                "fontsize": 10.0,
            },
            {
                "type": ElementType.FORMULA_IMAGE,
                "bbox": [72.0, 140.0, 300.0, 170.0],
                "math_reasons": ["unicode_math"],
                "math_signature": {"text_sha256": "a" * 64, "ignored": "x"},
            },
        ]
    }


class Harness:
    def __init__(self, tmp_path, summaries, *, pages=3, fail_open=True, merges=(), fallbacks=()):
        cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
        self.summaries = list(summaries)
        self.summary_calls = []
        self.progress = []
        self.audit = {
            "merge_decisions": list(merges),
            "source_page_fallbacks": [dict(entry) for entry in fallbacks],
        }
        self.context = stage.PDFExtractionStageContext(
            task_id="formula-test",
            src_path=str(tmp_path / "source.pdf"),
            pdf_password="",
            progress_dir=str(tmp_path),
            extraction_concurrency=1,
            fail_open_to_source_page=fail_open,
            element_policy=ImmutableElementPolicy(preserve_tables=False),
            extraction_semaphore=threading.Semaphore(1),
            logger=logging.getLogger("test-formula-protection"),
            pdf_audit=self.audit,
            load_progress=lambda _task_id: None,
            save_translation_progress=lambda *args: self.progress.append(args),
            extract_page_elements=lambda *args, **kwargs: [],
            summarize_formula_protection=self.summarize,
        )
        self.result = stage.PDFExtractionResult(
            total_pages=pages,
            source_sha256="a" * 64,
            completed_indices=set(),
            cached_indices=set(),
            page_extractions={index: _page(index + 1) for index in range(pages)},
            page_rects={index: fitz.Rect(0, 0, 612, 792) for index in range(pages)},
            forced_fallbacks_by_index={},
        )
        self.audit_path = tmp_path / "formula-test_audit.json"

    def summarize(self, expectations, total_pages, page_extractions):
        self.summary_calls.append(
            (len(expectations), total_pages, sorted(page_extractions))
        )
        return self.summaries.pop(0)

    def run(self):
        stage.enforce_formula_protection(self.context, self.result)


OK = {"translation_queue_check": "ok"}


def _warning(**per_page):
    return {"translation_queue_check": "warning", "per_page": per_page}


def test_clean_audit_records_expectations_and_checkpoints(tmp_path):
    harness = Harness(tmp_path, [OK])
    harness.run()
    audit = harness.audit
    assert audit["formula_protection"] == OK
    assert [entry["page"] for entry in audit["formula_expectations"]] == [1, 2, 3]
    assert audit["formula_expectations"][0]["signature"] == {"text_sha256": "a" * 64}
    assert audit["superscript_expectations"] == [] and audit["vector_ocr_expectations"] == []
    assert audit["total_pages"] == 3 and audit["phase1_checkpointed_at"] > 0
    assert harness.summary_calls == [(3, 3, [0, 1, 2])]
    assert harness.progress == [] and audit["source_page_fallbacks"] == []
    saved = json.loads(harness.audit_path.read_text(encoding="utf-8"))
    assert saved["total_pages"] == 3


def test_warning_fails_closed_without_source_page_fallback(tmp_path):
    harness = Harness(tmp_path, [_warning(**{"2": {"formula_queue_entries": 1}})], fail_open=False)
    with pytest.raises(RuntimeError, match="unprotected math symbol"):
        harness.run()


def test_warning_page_is_preserved_from_source_and_audit_rerun(tmp_path):
    warning = _warning(
        **{"2": {"formula_queue_entries": 1, "unprotected_math_symbols_in_text": 3}}
    )
    harness = Harness(tmp_path, [warning, OK])
    harness.run()
    audit = harness.audit
    (entry,) = audit["source_page_fallbacks"]
    assert entry["page"] == 2 and entry["stage"] == "formula-protection"
    assert entry["reason"] == (
        "formula protection preflight: 3 unprotected math symbol(s) in "
        "translatable text; 1 protected region(s) eligible for translation"
    )
    assert harness.result.page_extractions[1] == {"source_page_fallback": entry}
    assert harness.result.completed_indices == {1}
    assert audit["formula_protection_before_source_fallback"] == warning
    assert audit["formula_protection_source_fallback_rounds"] == [
        {"direct_pages": [2], "expanded_pages": [], "source_pages": [2]}
    ]
    assert [item["page"] for item in audit["formula_expectations"]] == [1, 3]
    assert audit["source_page_fallback_expectation_exclusions"] == {"formula_expectations": 1}
    assert audit["formula_protection"] == OK
    ((task_id, completed, total, message, fallbacks),) = harness.progress
    assert (task_id, completed, total) == ("formula-test", {1}, 3)
    assert message == "公式风险页已保留为英文原页，继续翻译和组装..."
    assert fallbacks is audit["source_page_fallbacks"]


def test_accepted_cross_page_merges_pull_their_partner_page_along(tmp_path):
    merge = {
        "decision": "accepted",
        "source_page": 2,
        "source_element": 0,
        "destination_page": 3,
        "destination_element": 0,
    }
    harness = Harness(
        tmp_path,
        [_warning(**{"2": {"unprotected_math_symbols_in_text": 1}}), OK],
        merges=[merge],
    )
    harness.run()
    second, third = harness.audit["source_page_fallbacks"]
    assert (second["page"], third["page"]) == (2, 3)
    assert third["reason"] == (
        "formula protection preflight expanded to preserve an accepted "
        "cross-page merge linked to warning page(s) 2"
    )
    assert harness.audit["formula_protection_source_fallback_rounds"] == [
        {"direct_pages": [2], "expanded_pages": [3], "source_pages": [2, 3]}
    ]


def test_each_round_preserves_only_the_new_warning_pages(tmp_path):
    first = _warning(**{"1": {"formula_queue_entries": 2}})
    second = _warning(**{"1": {"formula_queue_entries": 2}, "3": {"formula_queue_entries": 1}})
    harness = Harness(tmp_path, [first, second, OK])
    harness.run()
    assert [entry["page"] for entry in harness.audit["source_page_fallbacks"]] == [1, 3]
    rounds = harness.audit["formula_protection_source_fallback_rounds"]
    assert [item["source_pages"] for item in rounds] == [[1], [3]]
    # The pre-fallback snapshot keeps the first warning only.
    assert harness.audit["formula_protection_before_source_fallback"] == first
    assert len(harness.progress) == 2


@pytest.mark.parametrize(
    "fallbacks, pages",
    [([{"page": 2, "stage": "render", "reason": "earlier"}], 3), ([], 1)],
    ids=["already-source", "no-layout-for-page"],
)
def test_warning_that_no_new_source_page_can_absorb_is_fatal(tmp_path, fallbacks, pages):
    harness = Harness(
        tmp_path,
        [_warning(**{"2": {"formula_queue_entries": 1}})],
        pages=pages,
        fallbacks=fallbacks,
    )
    with pytest.raises(RuntimeError, match="remained non-clean"):
        harness.run()


def test_warning_without_page_details_preserves_every_extracted_page(tmp_path):
    harness = Harness(tmp_path, [{"translation_queue_check": "warning"}, OK], pages=2)
    harness.run()
    assert [entry["page"] for entry in harness.audit["source_page_fallbacks"]] == [1, 2]


def test_math_dense_prose_is_frozen_as_source_before_the_audit(tmp_path):
    harness = Harness(tmp_path, [OK], pages=1)
    dense = harness.result.page_extractions[0]["elements"][0]
    dense["content"] = "Regression alpha α plus beta β equals gamma γ under limit δ."
    harness.run()
    assert dense["skip_translate_reason"] == "formula_risk_preserved"
    (preserved,) = harness.audit["formula_risk_preserved_elements"]
    assert preserved["page"] == 1 and preserved["element"] == 0
