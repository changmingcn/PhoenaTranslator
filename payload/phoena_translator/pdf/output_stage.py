"""Output-document assembly, structural validation, and completion records."""

from __future__ import annotations

import copy
import gc
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

import fitz

from phoena_translator.pdf.assembly import (
    PDFAssemblyContext,
    assemble_document_pages,
)
from phoena_translator.pdf.audit import (
    _expand_pdf_fallback_pages_for_accepted_merges,
    _require_clean_pdf_structure_check,
    _save_and_validate_pdf_audit,
)
from phoena_translator.pdf.cache import (
    _append_pdf_source_page_fallback,
    _exclude_pdf_audit_expectations_for_source_pages,
    _load_pdf_layout_cache,
    _pdf_source_page_fallback_entry,
    _replace_pdf_pages_with_source,
    _save_pdf_layout_cache,
)
from phoena_translator.pdf.fonts import (
    _collect_pdf_font_chars,
    _get_chinese_font_path,
    _subset_pdf_font,
)
from phoena_translator.pdf.rendering import _compact_pdf_file_in_place


@dataclass(frozen=True)
class PDFOutputStageContext:
    task_id: str
    src_path: str
    out_path: str
    pdf_password: str
    total_pages: int
    assembly_concurrency: int
    use_htmlbox: bool
    minimum_htmlbox_scale: float
    save_clean: bool
    save_garbage: int
    font_subsetting_available: bool
    font_path: str
    bold_font_path: str
    assembly_semaphore: Any
    logger: logging.Logger
    tasks: dict[str, dict]
    pdf_audit: dict
    page_extractions: dict[int, dict]
    page_rects: dict[int, Any]
    completed_indices: set[int]
    save_progress: Callable[[str, dict], None]
    save_translation_progress: Callable[..., None]
    trim_process_memory: Callable[[], None]
    filter_formula_safe_rects: Callable[..., list[Any]]
    check_output_structure_serialized: Callable[..., dict]


def _open_output_document(
    context: PDFOutputStageContext,
) -> tuple[fitz.Document, str | None]:
    assembly_work_path = None
    document = None
    try:
        if context.use_htmlbox:
            output_dir = os.path.dirname(os.path.abspath(context.out_path)) or "."
            descriptor, assembly_work_path = tempfile.mkstemp(
                prefix=f"{context.task_id}_htmlbox_",
                suffix=".pdf",
                dir=output_dir,
            )
            os.close(descriptor)
            shutil.copy2(context.src_path, assembly_work_path)
            document = fitz.open(assembly_work_path)
        else:
            document = fitz.open(context.src_path)
        if document.is_encrypted and not document.authenticate(context.pdf_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法写出加密PDF")
        return document, assembly_work_path
    except Exception:
        if document is not None:
            try:
                document.close()
            except Exception as cleanup_error:
                context.logger.warning(
                    "[%s] Failed to close rejected output document: %s",
                    context.task_id,
                    cleanup_error,
                )
        if assembly_work_path:
            try:
                os.unlink(assembly_work_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                context.logger.warning(
                    "[%s] Failed to remove rejected HTML-box work file: %s",
                    context.task_id,
                    cleanup_error,
                )
        raise


def resolve_output_fonts() -> tuple[str, str]:
    """Fail fast on missing render prerequisites before provider work begins."""

    return _get_chinese_font_path(), _get_chinese_font_path(bold=True)


def _prepare_fonts(
    context: PDFOutputStageContext,
    page_extractions: dict[int, dict],
) -> tuple[str, str, bool, fitz.Archive, str | None]:
    font_path = context.font_path
    bold_font_path = context.bold_font_path
    has_distinct_bold = os.path.abspath(bold_font_path) != os.path.abspath(font_path)
    subset_font_dir = None
    # Every current/resumed translation is rebound to ``info['cached']`` before
    # assembly.  The old parallel translated_texts channel was always empty here.
    used_chars = _collect_pdf_font_chars(page_extractions, {})
    if context.font_subsetting_available and used_chars:
        subset_font_dir = tempfile.mkdtemp(prefix="pdf_font_subset_")
        try:
            font_path = _subset_pdf_font(
                font_path,
                used_chars,
                subset_font_dir,
                "regular",
            )
            if has_distinct_bold:
                context.logger.info(
                    f"[{context.task_id}] Created subset regular font for "
                    f"{len(used_chars)} chars; keeping full bold font"
                )
            else:
                context.logger.info(
                    f"[{context.task_id}] Created subset fonts for "
                    f"{len(used_chars)} chars"
                )
        except Exception as error:
            context.logger.warning(
                f"[{context.task_id}] Font subsetting failed, using full font: {error}"
            )
            shutil.rmtree(subset_font_dir, ignore_errors=True)
            subset_font_dir = None
    if not has_distinct_bold:
        bold_font_path = font_path
    try:
        font_dir = os.path.dirname(font_path)
        font_archive = fitz.Archive(font_dir)
        if os.path.dirname(bold_font_path) != font_dir:
            font_archive.add(os.path.dirname(bold_font_path))
        return (
            font_path,
            bold_font_path,
            has_distinct_bold,
            font_archive,
            subset_font_dir,
        )
    except Exception:
        # Until this function returns, it owns the subset directory.  Archive
        # construction can fail after a successful subset operation, before
        # the caller has any path it could clean up.
        if subset_font_dir:
            shutil.rmtree(subset_font_dir, ignore_errors=True)
        raise


def _save_output_document(
    context: PDFOutputStageContext,
    document: fitz.Document,
    assembly_work_path: str | None,
) -> tuple[None, None]:
    gc.collect()
    context.logger.info(
        f"[{context.task_id}] Saving PDF "
        f"(assembly_concurrency={context.assembly_concurrency}, "
        f"garbage={context.save_garbage}, clean={context.save_clean})"
    )
    if context.use_htmlbox and assembly_work_path:
        if document is not None:
            document.close()
        document = None
        os.replace(assembly_work_path, context.out_path)
        assembly_work_path = None
        _compact_pdf_file_in_place(context.out_path, context.task_id)
    else:
        document.save(
            context.out_path,
            deflate=True,
            garbage=context.save_garbage,
            clean=context.save_clean,
        )
        document.close()
        document = None
    os.chmod(context.out_path, 0o644)
    return document, assembly_work_path


def assemble_output(context: PDFOutputStageContext) -> None:
    """Assemble and persist the output while owning all temporary resources."""

    layout_cache_path = _save_pdf_layout_cache(
        context.task_id,
        context.page_extractions,
        context.page_rects,
    )
    context.logger.info(
        f"[{context.task_id}] Saved PDF layout cache before assembly wait: "
        f"{layout_cache_path}"
    )
    # The cache is the hand-off boundary to the serialized assembly stage.
    # Empty the large in-memory containers before waiting for a scarce slot;
    # otherwise the subsequent cache load temporarily retains two layouts.
    context.page_extractions.clear()
    context.page_rects.clear()
    context.trim_process_memory()
    context.save_translation_progress(
        context.task_id,
        context.completed_indices,
        context.total_pages,
        "等待PDF组装...",
        context.pdf_audit["source_page_fallbacks"],
    )
    context.logger.info(
        f"[{context.task_id}] Waiting for PDF assembly slot "
        f"(assembly_concurrency={context.assembly_concurrency})"
    )
    acquired = False
    document = None
    assembly_context = None
    assembly_work_path = None
    subset_font_dir = None
    try:
        context.assembly_semaphore.acquire()
        acquired = True
        context.logger.info(
            f"[{context.task_id}] Phase 3: Assembling output PDF with original layout"
        )
        page_extractions, page_rects = _load_pdf_layout_cache(context.task_id)
        # Rebind the serialized layout to the long-lived pipeline context.
        # ``dict.update`` shares the loaded element objects; it does not copy
        # the multi-megabyte layout.  Final audit and failure recovery must not
        # observe the intentionally cleared pre-wait containers above.
        context.page_extractions.update(page_extractions)
        context.page_rects.update(page_rects)
        document, assembly_work_path = _open_output_document(context)
        (
            font_path,
            bold_font_path,
            has_distinct_bold,
            font_archive,
            subset_font_dir,
        ) = _prepare_fonts(context, page_extractions)
        assembly_context = PDFAssemblyContext(
            task_id=context.task_id,
            total_pages=context.total_pages,
            pdf_password=context.pdf_password,
            use_htmlbox=context.use_htmlbox,
            minimum_htmlbox_scale=context.minimum_htmlbox_scale,
            assembly_work_path=assembly_work_path,
            out_doc=document,
            page_extractions=page_extractions,
            pdf_audit=context.pdf_audit,
            font_path=font_path,
            bold_font_path=bold_font_path,
            has_distinct_bold_font=has_distinct_bold,
            font_archive=font_archive,
            font_basename=os.path.basename(font_path),
            logger=context.logger,
            filter_formula_safe_rects=context.filter_formula_safe_rects,
            trim_process_memory=context.trim_process_memory,
        )
        assemble_document_pages(assembly_context)
        document = assembly_context.out_doc
        document, assembly_work_path = _save_output_document(
            context,
            document,
            assembly_work_path,
        )
        assembly_context.out_doc = document
    finally:
        current_documents = []
        if assembly_context is not None and assembly_context.out_doc is not None:
            current_documents.append(assembly_context.out_doc)
        if document is not None and all(
            document is not current for current in current_documents
        ):
            current_documents.append(document)
        for current_document in current_documents:
            try:
                current_document.close()
            except Exception:
                pass
        if assembly_work_path:
            try:
                os.unlink(assembly_work_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                context.logger.warning(
                    "[%s] Failed to remove HTML-box work file during cleanup: %s",
                    context.task_id,
                    cleanup_error,
                )
        if subset_font_dir:
            shutil.rmtree(subset_font_dir, ignore_errors=True)
        if acquired:
            try:
                context.assembly_semaphore.release()
            except ValueError:
                pass


def _run_structure_check(context: PDFOutputStageContext) -> dict:
    try:
        return context.check_output_structure_serialized(
            context.out_path,
            context.pdf_audit["superscript_expectations"],
            context.pdf_audit["merge_decisions"],
            context.pdf_audit["formula_expectations"],
            context.pdf_audit["vector_ocr_expectations"],
            source_path=context.src_path,
            source_password=context.pdf_password,
            page_extractions=context.page_extractions,
        )
    except Exception as error:
        return {
            "status": "error",
            "warning_count": 1,
            "warnings": [
                {
                    "type": "checker-error",
                    "error_type": type(error).__name__,
                }
            ],
        }


def _apply_structure_fallbacks(context: PDFOutputStageContext) -> None:
    while context.pdf_audit["structure_check"].get("status") != "ok" or (
        int(context.pdf_audit["structure_check"].get("warning_count", 0) or 0)
        or context.pdf_audit["structure_check"].get("warnings")
    ):
        warnings = list(context.pdf_audit["structure_check"].get("warnings") or [])
        warning_pages = set()
        has_unscoped_warning = False
        for warning in warnings:
            if not isinstance(warning, dict):
                has_unscoped_warning = True
                continue
            try:
                warning_page = int(warning.get("page", 0) or 0)
            except (TypeError, ValueError):
                warning_page = 0
            if warning_page < 1:
                has_unscoped_warning = True
            else:
                warning_pages.add(warning_page)
        if not warnings or has_unscoped_warning:
            break
        direct_warning_pages = set(warning_pages)
        warning_pages = set(
            _expand_pdf_fallback_pages_for_accepted_merges(
                warning_pages,
                context.pdf_audit["merge_decisions"],
            )
        )
        existing = {
            int(entry.get("page", 0) or 0)
            for entry in context.pdf_audit["source_page_fallbacks"]
        }
        new_pages = sorted(warning_pages - existing)
        if not new_pages:
            break
        candidate_audit = copy.deepcopy(context.pdf_audit)
        candidate_audit.setdefault(
            "structure_checks_before_source_fallback",
            [],
        ).append(candidate_audit["structure_check"])
        for page_number in new_pages:
            warning_types = sorted(
                {
                    str(warning.get("type", "unknown"))
                    for warning in warnings
                    if int(warning.get("page", 0) or 0) == page_number
                }
            )
            if page_number in direct_warning_pages:
                reason = "confirmed final structure audit: " + ", ".join(warning_types)
            else:
                reason = (
                    "confirmed final structure audit expanded through accepted "
                    "cross-page merge linked to warning page(s) "
                    + ", ".join(str(page) for page in sorted(direct_warning_pages))
                )
            _append_pdf_source_page_fallback(
                candidate_audit["source_page_fallbacks"],
                _pdf_source_page_fallback_entry(
                    page_number,
                    "structure",
                    reason,
                    error_type="PDFStructureValidationError",
                ),
            )
        _exclude_pdf_audit_expectations_for_source_pages(
            candidate_audit,
            new_pages,
        )
        # The output file is the transaction boundary: do not claim a source
        # fallback in the audit until the atomic page replacement succeeds.
        _replace_pdf_pages_with_source(
            context.out_path,
            context.src_path,
            new_pages,
            source_password=context.pdf_password,
            task_id=context.task_id,
        )
        context.pdf_audit.clear()
        context.pdf_audit.update(candidate_audit)
        context.pdf_audit["structure_check"] = _run_structure_check(context)


def finalize_output(
    context: PDFOutputStageContext,
    *,
    fail_open_to_source_page: bool,
) -> None:
    """Validate output, persist audit, and mark the task complete."""

    context.pdf_audit["structure_check"] = _run_structure_check(context)
    if fail_open_to_source_page:
        _apply_structure_fallbacks(context)
    _require_clean_pdf_structure_check(context.pdf_audit["structure_check"])
    context.trim_process_memory()
    completed_audit = copy.deepcopy(context.pdf_audit)
    completed_audit["completed_at"] = time.time()
    if completed_audit["source_page_fallbacks"]:
        completed_audit["quality_status"] = "completed_with_source_page_fallbacks"
    _save_and_validate_pdf_audit(context.task_id, completed_audit)
    completion_status = (
        "completed_with_source_page_fallbacks"
        if completed_audit["source_page_fallbacks"]
        else "completed"
    )
    task = context.tasks[context.task_id]
    context.save_progress(
        context.task_id,
        {
            "status": "completed",
            "completion_status": completion_status,
            "progress": context.total_pages,
            "total": context.total_pages,
            "current_file": "",
            "completed_files": [str(index) for index in range(context.total_pages)],
            "filename": task.get("filename", ""),
            "out_filename": task.get("out_filename", ""),
            "type": "pdf",
            "output": context.out_path,
            "source_page_fallbacks": completed_audit["source_page_fallbacks"],
            "element_source_fallbacks": completed_audit.get(
                "element_source_fallbacks", []
            ),
            "untranslated_english_leaks": completed_audit.get(
                "untranslated_english_leaks", []
            ),
        },
    )
    # Publish completion metadata to shared in-memory state only after both
    # durable audit and task progress writes have succeeded.  On failure, the
    # recovery path therefore records failed_at without a contradictory
    # completed_at in the same audit object.
    context.pdf_audit.clear()
    context.pdf_audit.update(completed_audit)
    # Completion is already durable and must be an irreversible terminal
    # state.  A broken logging handler/filter is observational only and may
    # not send the top-level recovery path from completed back to failed.
    try:
        context.logger.info(
            f"[{context.task_id}] PDF translation completed "
            f"({len(context.pdf_audit['source_page_fallbacks'])} source-page "
            "fallback(s), "
            f"{len(context.pdf_audit.get('element_source_fallbacks') or [])} "
            "element-level source fallback(s))!"
        )
    except Exception:
        pass


__all__ = [
    "PDFOutputStageContext",
    "assemble_output",
    "finalize_output",
    "resolve_output_fonts",
]
