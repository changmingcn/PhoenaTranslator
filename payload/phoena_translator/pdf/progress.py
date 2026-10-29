"""Task-progress records written while a PDF is being translated."""

from __future__ import annotations

from phoena_translator.pdf.verify.output import PDFAuditDependencies


def _save_pdf_translation_progress(
    task_id: str,
    completed_indices: set[int],
    total_pages: int,
    current_file: str,
    source_page_fallbacks: list[dict] | None = None,
    *,
    dependencies: PDFAuditDependencies,
):
    save_progress = dependencies.save_progress
    # filename/out_filename/type are carried over by the task store.
    payload = {
        "status": "translating",
        "progress": len(completed_indices),
        "total": total_pages,
        "current_file": current_file,
        "completed_files": [str(i) for i in sorted(completed_indices)],
        "type": "pdf",
    }
    if source_page_fallbacks:
        payload["source_page_fallbacks"] = list(source_page_fallbacks)
        payload["completion_status"] = "in_progress_with_source_page_fallbacks"
    save_progress(task_id, payload)
