"""Helpers shared by several PDF regression test modules."""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# 2026-07-24 two-signal OR policy for cross-page sentence absorption
# ---------------------------------------------------------------------------


def _two_page_fixture(
    source_content: str,
    dest_content: str,
    *,
    dest_rich: str | None = None,
    dest_bbox: list | None = None,
):
    import fitz

    last_elem = {
        "type": "text",
        "layout_class": "body",
        "content": source_content,
        "bbox": [72.0, 620.0, 518.0, 715.0],
        "y": 620.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    first_elem = {
        "type": "text",
        "layout_class": "scattered",
        "content": dest_content,
        "bbox": dest_bbox or [72.0, 104.0, 518.0, 131.0],
        "y": 104.0,
        "x": 72.0,
        "fontsize": 12.0,
    }
    if dest_rich is not None:
        first_elem["rich_content"] = dest_rich
    page_extractions = {
        0: {"elements": [last_elem]},
        1: {"elements": [first_elem]},
    }
    page_rects = {0: fitz.Rect(0, 0, 612, 792), 1: fitz.Rect(0, 0, 612, 792)}
    return page_extractions, page_rects, last_elem, first_elem


def _table_test_elem(layout_class, **overrides):
    base = {"type": "text", "layout_class": layout_class, "content": "cell 12.3"}
    base.update(overrides)
    return base


def _policy_test_elements():
    elements = [
        {
            "type": "text",
            "layout_class": "table",
            "content": (
                f"Contract category {index + 1} reports daily trading "
                "volume of 123 units."
            ),
            "bbox": [100.0, 180.0 + index * 18.0, 480.0, 194.0 + index * 18.0],
            "x": 100.0,
            "y": 180.0 + index * 18.0,
            "fontsize": 10.0,
        }
        for index in range(5)
    ]
    elements.append(
        {
            "type": "text",
            "layout_class": "body",
            "content": "Ordinary prose sentence that needs translation.",
            "bbox": [90.0, 320.0, 500.0, 350.0],
            "x": 90.0,
            "y": 320.0,
            "fontsize": 11.0,
        }
    )
    return elements


def _policy_context(tmp_path, preserve_tables):
    from phoena_translator.pdf.context import ImmutableElementPolicy
    from phoena_translator.pdf.extract.stage import PDFExtractionStageContext

    kwargs = {
        "task_id": "policy-test",
        "src_path": str(tmp_path / "source.pdf"),
        "pdf_password": "",
        "progress_dir": str(tmp_path),
        "extraction_concurrency": 1,
        "fail_open_to_source_page": False,
        "element_policy": ImmutableElementPolicy(
            preserve_tables=preserve_tables
        ),
        "extraction_semaphore": threading.Semaphore(1),
        "logger": log,
        "pdf_audit": {
            "merge_decisions": [],
            "source_page_fallbacks": [],
        },
        "load_progress": lambda _task_id: None,
        "save_translation_progress": lambda *args, **kwargs: None,
        "extract_page_elements": lambda *args, **kwargs: [],
        "summarize_formula_protection": (
            lambda *args, **kwargs: {"translation_queue_check": "ok"}
        ),
    }
    return PDFExtractionStageContext(**kwargs)


def _policy_result(elements, source_sha256="a" * 64, *, cached=False):
    import fitz

    from phoena_translator.pdf.extract.stage import PDFExtractionResult

    completed = {0} if cached else set()
    return PDFExtractionResult(
        total_pages=1,
        source_sha256=source_sha256,
        completed_indices=set(completed),
        cached_indices=set(completed),
        page_extractions={0: {"elements": elements}},
        page_rects={0: fitz.Rect(0, 0, 612, 792)},
        forced_fallbacks_by_index={},
    )


def _single_formula_page(rotation: int):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((300.0, 700.0), "f(x) = 1", fontsize=12)
    if rotation:
        page.set_rotation(rotation)
    return doc, page
