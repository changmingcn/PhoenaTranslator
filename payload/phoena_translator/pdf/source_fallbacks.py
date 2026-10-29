"""Source fallbacks: record which pages or elements keep their source text,
drop the audit expectations they can no longer meet, and splice exact
source pages into the output."""

from __future__ import annotations

import logging
import os
import stat
import fitz
import tempfile
from pathlib import Path
from phoena_translator.pdf.text_shape import _collapsed_plain_text
from phoena_translator.pdf.model import PdfAudit


log = logging.getLogger("translator")

def _pdf_source_page_fallback_entry(
    page: int,
    stage: str,
    reason: str,
    *,
    error_type: str = "",
) -> dict:
    """Return the public audit record for one exact source-page fallback."""
    entry = {
        "page": int(page),
        "stage": str(stage or "unknown"),
        "reason": str(reason or "unspecified page failure")[:2000],
    }
    if error_type:
        entry["error_type"] = str(error_type)
    return entry


def _append_pdf_source_page_fallback(entries: list[dict], entry: dict) -> dict:
    """Append a fallback once per page while retaining the first root cause."""
    page = int(entry.get("page", 0) or 0)
    for existing in entries:
        if int(existing.get("page", 0) or 0) == page:
            return existing
    entries.append(dict(entry))
    entries.sort(key=lambda item: int(item.get("page", 0) or 0))
    return entry


def _exclude_pdf_audit_expectations_for_source_pages(
    audit: PdfAudit,
    page_numbers: set[int] | list[int],
) -> None:
    """Exclude knowingly untranslated source pages from translation-only checks."""
    pages = {int(page) for page in page_numbers if int(page) > 0}
    if not pages:
        return

    excluded = audit.setdefault("source_page_fallback_expectation_exclusions", {})
    for key in (
        "superscript_expectations",
        "formula_expectations",
        "vector_ocr_expectations",
    ):
        values = list(audit.get(key) or [])
        retained = [
            value
            for value in values
            if int(value.get("page", 0) or 0) not in pages
        ]
        removed = len(values) - len(retained)
        if removed:
            excluded[key] = int(excluded.get(key, 0) or 0) + removed
            audit[key] = retained


def _record_pdf_element_source_fallback(
    audit: PdfAudit,
    page_number: int,
    element_index: int,
    stage: str,
    reason: str,
    source_text: str = "",
) -> dict:
    """Record that one element keeps its exact source text inside a translated page.

    This is the element-granular sibling of ``source_page_fallbacks``: the page
    itself still translates and assembles normally, only this element's original
    ink (or restored source text) is delivered untranslated.
    """
    entry = {
        "page": int(page_number),
        "element": int(element_index),
        "stage": str(stage or "unknown"),
        "reason": str(reason or "unspecified element failure")[:500],
    }
    if source_text:
        entry["source_text"] = _collapsed_plain_text(source_text)[:160]
    audit.setdefault("element_source_fallbacks", []).append(entry)
    return entry


def _exclude_pdf_audit_expectations_for_element(
    audit: PdfAudit,
    page_number: int,
    element_index: int,
) -> None:
    """Exclude one knowingly untranslated element from translation-only checks."""
    page_number = int(page_number)
    element_index = int(element_index)
    excluded = audit.setdefault("element_source_fallback_expectation_exclusions", {})
    for key in (
        "superscript_expectations",
        "vector_ocr_expectations",
    ):
        values = list(audit.get(key) or [])
        retained = [
            value
            for value in values
            if not (
                int(value.get("page", 0) or 0) == page_number
                and int(value.get("element", -1)) == element_index
            )
        ]
        removed = len(values) - len(retained)
        if removed:
            excluded[key] = int(excluded.get(key, 0) or 0) + removed
            audit[key] = retained


def _replace_pdf_pages_with_source(
    output_path: str,
    source_path: str,
    page_numbers: set[int] | list[int],
    *,
    source_password: str = "",
    task_id: str = "",
) -> list[int]:
    """Atomically replace selected one-based output pages with exact source pages."""
    pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not pages:
        return []

    output_doc = None
    source_doc = None
    tmp_path = None
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    original_mode = stat.S_IMODE(os.stat(output_path).st_mode)
    try:
        output_doc = fitz.open(output_path)
        source_doc = fitz.open(source_path)
        if output_doc.is_encrypted and not output_doc.authenticate(source_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法修复输出PDF")
        if source_doc.is_encrypted and not source_doc.authenticate(source_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法读取源PDF回退页")
        if len(output_doc) != len(source_doc):
            raise RuntimeError(
                "source-page fallback refused because page counts differ: "
                f"output={len(output_doc)}, source={len(source_doc)}"
            )
        invalid = [page for page in pages if page > len(source_doc)]
        if invalid:
            raise RuntimeError(f"source-page fallback pages out of range: {invalid}")

        for page in pages:
            page_index = page - 1
            output_doc.delete_page(page_index)
            output_doc.insert_pdf(
                source_doc,
                from_page=page_index,
                to_page=page_index,
                start_at=page_index,
                links=1,
                annots=1,
                widgets=1,
                final=1,
            )

        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{Path(output_path).stem}_source_fallback_",
            suffix=".pdf",
            dir=directory,
        )
        os.close(fd)
        output_doc.save(tmp_path, deflate=True, garbage=4, clean=True)
        output_doc.close()
        output_doc = None
        source_doc.close()
        source_doc = None
        os.chmod(tmp_path, original_mode or 0o644)
        os.replace(tmp_path, output_path)
        tmp_path = None
        log.warning(
            f"[{task_id}] Restored exact source page(s) {pages} into the output PDF"
        )
        return pages
    finally:
        if output_doc is not None:
            output_doc.close()
        if source_doc is not None:
            source_doc.close()
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
