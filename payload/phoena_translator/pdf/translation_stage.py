"""Concurrent translation, commit, retry, and source-fallback stage."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from phoena_translator.pdf.audit import (
    _expand_pdf_fallback_pages_for_accepted_merges,
)
from phoena_translator.pdf.cache import (
    _append_pdf_source_page_fallback,
    _exclude_pdf_audit_expectations_for_element,
    _exclude_pdf_audit_expectations_for_source_pages,
    _pdf_source_page_fallback_entry,
    _record_pdf_element_source_fallback,
    _save_pdf_page_translation_cache,
)
from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.page_translation import translate_page
from phoena_translator.pdf.translation import _validate_pdf_page_translations
from phoena_translator.pdf.types import (
    PDFPageTranslationError,
    PDFTranslationIntegrityError,
)


@dataclass
class PDFTranslationStageContext:
    """Mutable state required by the concurrent translation stage."""

    task_id: str
    total_pages: int
    source_sha256: str
    workers: int
    fail_open_to_source_page: bool
    logger: logging.Logger
    page_extractions: dict[int, dict]
    pdf_audit: dict
    completed_indices: set[int]
    save_translation_progress: Callable[..., None]


def _is_accepted_merge_endpoint(
    pdf_audit: dict,
    page_number: int,
    element_index: int,
) -> bool:
    """Return whether an element was mutated by an accepted cross-page merge."""

    for decision in pdf_audit.get("merge_decisions") or []:
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        for page_key, element_key in (
            ("source_page", "source_element"),
            ("destination_page", "destination_element"),
        ):
            try:
                endpoint = (
                    int(decision.get(page_key, 0) or 0),
                    int(decision.get(element_key, -1)),
                )
            except (TypeError, ValueError):
                continue
            if endpoint == (page_number, element_index):
                return True
    return False


def _commit_translated_page(
    context: PDFTranslationStageContext,
    page_num: int,
    translations: dict,
    progress_lock: threading.Lock,
) -> None:
    """Validate and persist one translated page without repeating provider work."""

    elements = context.page_extractions[page_num]["elements"]
    _validate_pdf_page_translations(page_num + 1, elements, translations)
    for commit_attempt in range(2):
        try:
            _save_pdf_page_translation_cache(
                context.task_id,
                page_num,
                translations,
                context.source_sha256,
                elements=elements,
            )
            context.page_extractions[page_num]["cached"] = translations
            with progress_lock:
                context.completed_indices.add(page_num)
                context.save_translation_progress(
                    context.task_id,
                    context.completed_indices,
                    context.total_pages,
                    (
                        "并发翻译中 "
                        f"({len(context.completed_indices)}/{context.total_pages})"
                    ),
                    context.pdf_audit["source_page_fallbacks"],
                )
            return
        except Exception as commit_error:
            if commit_attempt:
                raise
            context.logger.warning(
                "[%s] Page %s: local cache/progress commit failed; retrying "
                "the same validated translation without another provider call: %s",
                context.task_id,
                page_num + 1,
                commit_error,
            )


def _commit_page_with_element_fallbacks(
    context: PDFTranslationStageContext,
    page_num: int,
    integrity_error: PDFTranslationIntegrityError,
    progress_lock: threading.Lock,
) -> bool:
    """Keep only unsafe elements as source text after both translation tries."""

    if not integrity_error.issues:
        return False
    elements = context.page_extractions[page_num]["elements"]
    translations = dict(integrity_error.translations)
    element_entries = []
    for issue in integrity_error.issues:
        try:
            element_index = int(issue.get("element", 0)) - 1
        except (TypeError, ValueError):
            return False
        if not 0 <= element_index < len(elements):
            return False
        element_entries.append((element_index, str(issue.get("reason", "integrity"))))
    merge_endpoints = [
        element_index
        for element_index, _reason in element_entries
        if _is_accepted_merge_endpoint(
            context.pdf_audit,
            page_num + 1,
            element_index,
        )
    ]
    if merge_endpoints:
        # Both endpoint elements were mutated during semantic merging.  Keeping
        # just one mutated element as source text can duplicate or drop the
        # carried fragment, so let the page-level policy preserve the complete
        # accepted-merge component instead.
        context.logger.warning(
            f"[{context.task_id}] Page {page_num + 1}: element-level source "
            f"fallback touches accepted cross-page merge endpoint(s) "
            f"{merge_endpoints}; escalating to source-page closure"
        )
        return False
    for element_index, _reason in element_entries:
        element = elements[element_index]
        element["skip_translate_reason"] = "translation_integrity_fallback"
        translations[element_index] = element.get("content", "")
    _commit_translated_page(context, page_num, translations, progress_lock)
    for element_index, reason in element_entries:
        element = elements[element_index]
        _record_pdf_element_source_fallback(
            context.pdf_audit,
            page_num + 1,
            element_index,
            "translation",
            reason,
            source_text=element.get("content", ""),
        )
        _exclude_pdf_audit_expectations_for_element(
            context.pdf_audit,
            page_num + 1,
            element_index,
        )
    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1}: kept {len(element_entries)} "
        "element(s) as exact source text after translation retries; the rest "
        "of the page is translated"
    )
    return True


def _run_first_pass(
    context: PDFTranslationStageContext,
    page_context: PDFPageTranslationContext,
    pages_to_translate: list[int],
    progress_lock: threading.Lock,
) -> dict[int, Exception]:
    errors = {}
    with ThreadPoolExecutor(max_workers=context.workers) as executor:
        futures = {
            executor.submit(
                translate_page,
                page_context,
                page_num,
                context.page_extractions[page_num],
            ): page_num
            for page_num in pages_to_translate
        }
        for future in as_completed(futures):
            scheduled_page_num = futures[future]
            try:
                page_num, translations = future.result()
                _validate_pdf_page_translations(
                    page_num + 1,
                    context.page_extractions[page_num]["elements"],
                    translations,
                )
            except Exception as error:
                errors[scheduled_page_num] = error
                context.logger.warning(
                    f"[{context.task_id}] Page {scheduled_page_num + 1} failed translation "
                    f"integrity; queuing one clean page retry: {error}"
                )
                continue
            # Cache and progress failures are local failures.  They may retry
            # the same validated value inside the commit helper, but must never
            # be classified as translation failures or trigger another LLM call.
            _commit_translated_page(
                context,
                page_num,
                translations,
                progress_lock,
            )
    return errors


def _apply_page_source_fallbacks(
    context: PDFTranslationStageContext,
    final_errors: dict[int, Exception],
) -> None:
    if not final_errors:
        return
    if not context.fail_open_to_source_page:
        raise PDFPageTranslationError(final_errors)
    direct_pages = set(final_errors)
    fallback_pages = [
        page_number
        for page_number in _expand_pdf_fallback_pages_for_accepted_merges(
            direct_pages,
            context.pdf_audit.get("merge_decisions") or [],
        )
        if 1 <= page_number <= context.total_pages
    ]
    for page_number in fallback_pages:
        page_error = final_errors.get(page_number)
        if page_error is not None:
            reason = str(page_error)
            error_type = type(page_error).__name__
        else:
            reason = (
                "translation source fallback expanded through an accepted "
                "cross-page merge linked to failed page(s) "
                + ", ".join(str(page) for page in sorted(direct_pages))
            )
            error_type = "PDFCrossPageMergeFallback"
        entry = _append_pdf_source_page_fallback(
            context.pdf_audit["source_page_fallbacks"],
            _pdf_source_page_fallback_entry(
                page_number,
                "translation",
                reason,
                error_type=error_type,
            ),
        )
        page_index = page_number - 1
        page_info = context.page_extractions.setdefault(page_index, {})
        page_info["source_page_fallback"] = entry
        page_info.pop("cached", None)
        page_info.pop("cached_seed", None)
        context.completed_indices.add(page_index)
        if page_error is not None:
            context.logger.error(
                f"[{context.task_id}] Page {page_number}: translation failed "
                "twice; preserving the exact source page and continuing the PDF"
            )
        else:
            context.logger.warning(
                f"[{context.task_id}] Page {page_number}: preserving the exact "
                "source page because it belongs to an accepted cross-page merge "
                "component containing a translation failure"
            )
    _exclude_pdf_audit_expectations_for_source_pages(
        context.pdf_audit,
        fallback_pages,
    )
    context.save_translation_progress(
        context.task_id,
        context.completed_indices,
        context.total_pages,
        "翻译失败页已回退为英文原页，继续组装...",
        context.pdf_audit["source_page_fallbacks"],
    )


def translate_pages(
    context: PDFTranslationStageContext,
    page_context: PDFPageTranslationContext,
    pages_to_translate: list[int],
) -> None:
    """Translate pages concurrently, retry failures once, then apply policy."""

    progress_lock = threading.Lock()
    first_pass_errors = _run_first_pass(
        context,
        page_context,
        pages_to_translate,
        progress_lock,
    )
    final_errors = {}
    for page_num in sorted(first_pass_errors):
        try:
            retry_page_num, retry_translations = translate_page(
                page_context,
                page_num,
                context.page_extractions[page_num],
            )
            _validate_pdf_page_translations(
                retry_page_num + 1,
                context.page_extractions[retry_page_num]["elements"],
                retry_translations,
            )
        except Exception as retry_error:
            if isinstance(retry_error, PDFTranslationIntegrityError) and (
                _commit_page_with_element_fallbacks(
                    context,
                    page_num,
                    retry_error,
                    progress_lock,
                )
            ):
                continue
            final_errors[page_num + 1] = retry_error
            context.logger.error(
                f"[{context.task_id}] Page {page_num + 1} failed clean "
                f"translation retry: {retry_error}"
            )
            continue
        # As above, persistence is a deterministic local boundary and stays
        # outside the provider/integrity retry branch.
        _commit_translated_page(
            context,
            retry_page_num,
            retry_translations,
            progress_lock,
        )
        context.logger.info(
            f"[{context.task_id}] Page {page_num + 1} recovered on clean "
            "translation retry"
        )
    _apply_page_source_fallbacks(context, final_errors)


__all__ = ["PDFTranslationStageContext", "translate_pages"]
