"""Failure classification and bounded retry policy for PDF translation."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Callable

from phoena_translator.pdf.audit import _expand_pdf_fallback_pages_for_accepted_merges
from phoena_translator.pdf.cache import (
    _append_pdf_source_page_fallback,
    _pdf_layout_cache_path,
    _pdf_page_cache_path,
    _pdf_source_page_fallback_entry,
    _save_pdf_audit,
)
from phoena_translator.pdf.types import (
    PDFPageExtractionError,
    PDFPageRenderError,
    PDFStructureValidationError,
)


@dataclass(frozen=True)
class PDFRecoveryContext:
    task_id: str
    structure_retry: int
    fail_open_to_source_page: bool
    pdf_audit: dict
    tasks: dict[str, dict]
    logger: logging.Logger
    save_progress: Callable[[str, dict], None]
    trim_process_memory: Callable[[], None]


@dataclass(frozen=True)
class PDFRecoveryDecision:
    """A non-recursive instruction returned to the top-level pipeline loop."""

    should_retry: bool
    structure_retry: int
    source_page_fallbacks: tuple[dict, ...]
    reason: str
    new_fallback_pages: tuple[int, ...] = ()


def _recovery_decision(
    context: PDFRecoveryContext,
    *,
    should_retry: bool,
    structure_retry: int | None = None,
    reason: str,
    new_fallback_pages: list[int] | tuple[int, ...] = (),
) -> PDFRecoveryDecision:
    return PDFRecoveryDecision(
        should_retry=should_retry,
        structure_retry=(
            context.structure_retry
            if structure_retry is None
            else int(structure_retry)
        ),
        source_page_fallbacks=tuple(
            dict(entry)
            for entry in context.pdf_audit.get("source_page_fallbacks", [])
        ),
        reason=reason,
        new_fallback_pages=tuple(int(page) for page in new_fallback_pages),
    )


def _retry_with_source_pages(
    context: PDFRecoveryContext,
    error: Exception,
) -> PDFRecoveryDecision | None:
    fallback_pages = _expand_pdf_fallback_pages_for_accepted_merges(
        list(getattr(error, "fallback_pages", []) or []),
        context.pdf_audit.get("merge_decisions", []),
    )
    if not (
        context.fail_open_to_source_page
        and isinstance(error, (PDFPageRenderError, PDFPageExtractionError))
        and fallback_pages
    ):
        return None
    existing = {
        int(entry.get("page", 0) or 0)
        for entry in context.pdf_audit["source_page_fallbacks"]
    }
    new_pages = sorted(
        {
            int(page)
            for page in fallback_pages
            if int(page) > 0 and int(page) not in existing
        }
    )
    if not new_pages:
        return None
    stage = "render" if isinstance(error, PDFPageRenderError) else "extraction"
    for page_number in new_pages:
        _append_pdf_source_page_fallback(
            context.pdf_audit["source_page_fallbacks"],
            _pdf_source_page_fallback_entry(
                page_number,
                stage,
                str(error),
                error_type=type(error).__name__,
            ),
        )
    context.pdf_audit["source_page_fallback_reassembly"] = {
        "pages": new_pages,
        "reason": str(error),
    }
    try:
        _save_pdf_audit(context.task_id, context.pdf_audit)
    except Exception:
        pass
    context.logger.warning(
        f"[{context.task_id}] Reassembling with exact source page(s) {new_pages} "
        f"after {stage} failure; translation caches stay intact"
    )
    return _recovery_decision(
        context,
        should_retry=True,
        reason=f"new-{stage}-source-page-fallback",
        new_fallback_pages=new_pages,
    )


def _record_repair_cleanup_failure(
    context: PDFRecoveryContext,
    error: OSError,
    *,
    target: str,
    page_number: int | None = None,
) -> None:
    entry = {
        "target": target,
        "error_type": type(error).__name__,
        "reason": str(error)[:1000],
    }
    if page_number is not None:
        entry["page"] = int(page_number)
    context.pdf_audit["auto_repair_cleanup_error"] = entry
    try:
        _save_pdf_audit(context.task_id, context.pdf_audit)
    except Exception:
        pass
    page_suffix = f" for page {page_number}" if page_number is not None else ""
    context.logger.error(
        f"[{context.task_id}] Structural repair cannot remove {target}"
        f"{page_suffix}: {error}; recording a terminal task failure"
    )


def _retry_structural_repair(
    context: PDFRecoveryContext,
    error: Exception,
) -> PDFRecoveryDecision | None:
    repair_pages = _expand_pdf_fallback_pages_for_accepted_merges(
        list(getattr(error, "repair_pages", []) or []),
        context.pdf_audit.get("merge_decisions", []),
    )
    repairable = isinstance(
        error,
        (PDFStructureValidationError, PDFPageRenderError, PDFPageExtractionError),
    )
    if (
        context.fail_open_to_source_page
        or not repairable
        or context.structure_retry >= 1
        or not repair_pages
    ):
        return None
    for page_number in repair_pages:
        try:
            os.unlink(_pdf_page_cache_path(context.task_id, page_number - 1))
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            _record_repair_cleanup_failure(
                context,
                cleanup_error,
                target="page translation cache",
                page_number=page_number,
            )
            return None
    try:
        os.unlink(_pdf_layout_cache_path(context.task_id))
    except FileNotFoundError:
        pass
    except OSError as cleanup_error:
        _record_repair_cleanup_failure(
            context,
            cleanup_error,
            target="layout cache",
        )
        return None
    context.pdf_audit["auto_repair"] = {
        "attempt": context.structure_retry + 1,
        "pages": repair_pages,
        "reason": str(error),
    }
    try:
        _save_pdf_audit(context.task_id, context.pdf_audit)
    except Exception:
        pass
    context.logger.warning(
        f"[{context.task_id}] Rebuilding structurally mismatched page(s) "
        f"{repair_pages} once before failing the task"
    )
    return _recovery_decision(
        context,
        should_retry=True,
        structure_retry=context.structure_retry + 1,
        reason="single-structural-repair",
    )


def _task_record(context: PDFRecoveryContext) -> dict:
    return context.tasks.get(context.task_id, {})


def _trim_recovery_memory(context: PDFRecoveryContext) -> None:
    """Run optional memory cleanup without replacing the triggering failure."""

    try:
        context.trim_process_memory()
    except Exception as cleanup_error:
        context.pdf_audit.setdefault("recovery_cleanup_warnings", []).append(
            {
                "action": "trim_process_memory",
                "error_type": type(cleanup_error).__name__,
                "reason": str(cleanup_error)[:1000],
            }
        )
        try:
            _save_pdf_audit(context.task_id, context.pdf_audit)
        except Exception:
            pass
        context.logger.warning(
            f"[{context.task_id}] Recovery memory cleanup failed without "
            f"masking {context.pdf_audit.get('failure_type', 'the primary error')}: "
            f"{cleanup_error}"
        )


def recover_pdf_failure(
    context: PDFRecoveryContext,
    error: Exception,
) -> PDFRecoveryDecision:
    """Persist failure evidence and return a non-recursive recovery decision."""

    context.pdf_audit["failed_at"] = time.time()
    context.pdf_audit["failure_type"] = type(error).__name__
    try:
        _save_pdf_audit(context.task_id, context.pdf_audit)
    except Exception:
        pass
    _trim_recovery_memory(context)

    decision = _retry_with_source_pages(context, error)
    if decision is not None:
        return decision
    decision = _retry_structural_repair(context, error)
    if decision is not None:
        return decision

    task = _task_record(context)
    if re.search(
        r"cannot schedule new futures after (?:interpreter )?shutdown",
        str(error),
        flags=re.IGNORECASE,
    ):
        context.logger.warning(
            f"[{context.task_id}] Interpreter shutdown interrupted PDF work; "
            "preserving caches and returning task to the resume queue"
        )
        context.save_progress(
            context.task_id,
            {
                "status": "queued",
                "progress": task.get("progress", 0),
                "total": task.get("total", 0),
                "current_file": "等待服务恢复...",
                "completed_files": task.get("completed_files", []),
                "filename": task.get("filename", ""),
                "out_filename": task.get("out_filename", ""),
                "type": "pdf",
                "error": None,
                "source_page_fallbacks": context.pdf_audit["source_page_fallbacks"],
            },
        )
        return _recovery_decision(
            context,
            should_retry=False,
            reason="queued-for-interpreter-restart",
        )

    context.logger.error(
        f"[{context.task_id}] PDF translation error: {traceback.format_exc()}"
    )
    context.save_progress(
        context.task_id,
        {
            "status": "failed",
            "progress": task.get("progress", 0),
            "total": task.get("total", 0),
            "current_file": "",
            "completed_files": task.get("completed_files", []),
            "filename": task.get("filename", ""),
            "out_filename": task.get("out_filename", ""),
            "type": "pdf",
            "error": str(error),
            "source_page_fallbacks": context.pdf_audit["source_page_fallbacks"],
        },
    )
    return _recovery_decision(
        context,
        should_retry=False,
        reason="terminal-failure",
    )


__all__ = [
    "PDFRecoveryContext",
    "PDFRecoveryDecision",
    "recover_pdf_failure",
]
