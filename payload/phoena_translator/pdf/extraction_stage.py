"""Source extraction, cache reconciliation, and formula-preflight stages."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import fitz

from phoena_translator.pdf.audit import (
    _collect_pdf_formula_expectations,
    _collect_pdf_superscript_expectations,
    _collect_pdf_vector_ocr_expectations,
    _expand_pdf_fallback_pages_for_accepted_merges,
    _pdf_formula_protection_fallback_pages,
    _preserve_pdf_formula_mid_sentence_neighbors,
    _preserve_pdf_formula_risk_text_elements,
)
from phoena_translator.pdf.cache import (
    _append_pdf_source_page_fallback,
    _exclude_pdf_audit_expectations_for_source_pages,
    _load_pdf_cached_page_indices,
    _load_pdf_page_translation_cache,
    _pdf_page_cache_path,
    _pdf_source_page_fallback_entry,
    _pdf_source_sha256,
    _reconcile_pdf_page_translation_cache,
    _save_pdf_audit,
    _save_pdf_page_translation_cache,
)
from phoena_translator.pdf.context import ImmutableElementPolicy
from phoena_translator.pdf.extraction import _is_skip_page
from phoena_translator.pdf.semantics import (
    _derive_pdf_page_layout_styles,
    _mark_watermark_elements,
    _merge_cross_page_sentences,
)
from phoena_translator.pdf.targets import (
    _classify_pdf_page_text_elements,
    _pdf_element_requires_translation,
)
from phoena_translator.pdf.translation import _validate_pdf_page_translations
from phoena_translator.pdf.types import (
    PDFPageExtractionError,
    PDFTranslationIntegrityError,
)


@dataclass(frozen=True)
class PDFExtractionStageContext:
    task_id: str
    src_path: str
    pdf_password: str
    progress_dir: str
    extraction_concurrency: int
    fail_open_to_source_page: bool
    element_policy: ImmutableElementPolicy
    extraction_semaphore: Any
    logger: logging.Logger
    pdf_audit: dict
    load_progress: Callable[[str], dict | None]
    save_translation_progress: Callable[..., None]
    extract_page_elements: Callable[..., list[dict]]
    summarize_formula_protection: Callable[..., dict]

    @property
    def preserve_tables(self) -> bool:
        """Compatibility view; policy ownership stays with ``element_policy``."""

        return self.element_policy.preserve_tables


@dataclass
class PDFExtractionResult:
    total_pages: int
    source_sha256: str
    completed_indices: set[int]
    cached_indices: set[int]
    page_extractions: dict[int, dict]
    page_rects: dict[int, fitz.Rect]
    forced_fallbacks_by_index: dict[int, dict]


@dataclass(frozen=True)
class PDFTranslationPlan:
    pages_to_translate: list[int]
    prompt: str


def _normalize_source_fallbacks(pdf_audit: dict, total_pages: int) -> dict[int, dict]:
    normalized = []
    for raw_entry in pdf_audit["source_page_fallbacks"]:
        page_number = int(raw_entry.get("page", 0) or 0)
        if page_number < 1 or page_number > total_pages:
            raise RuntimeError(f"source-page fallback page out of range: {page_number}")
        _append_pdf_source_page_fallback(
            normalized,
            _pdf_source_page_fallback_entry(
                page_number,
                raw_entry.get("stage", "recovery"),
                raw_entry.get("reason", "prior page fallback"),
                error_type=raw_entry.get("error_type", ""),
            ),
        )
    pdf_audit["source_page_fallbacks"] = normalized
    return {int(entry["page"]) - 1: entry for entry in normalized}


def _load_resume_state(
    context: PDFExtractionStageContext,
    source_sha256: str,
    total_pages: int,
) -> tuple[set[int], set[int]]:
    progress_data = context.load_progress(context.task_id)
    os.makedirs(
        os.path.join(context.progress_dir, f"{context.task_id}_pages"),
        exist_ok=True,
    )
    cached_indices = set(_load_pdf_cached_page_indices(context.task_id, source_sha256))
    completed_indices = set(cached_indices)
    if not cached_indices and progress_data and progress_data.get("completed_files"):
        context.logger.warning(
            f"[{context.task_id}] Progress lists completed PDF pages but "
            "translation cache is missing; retranslating from scratch"
        )
    status = (
        f"准备续跑 ({len(completed_indices)}/{total_pages})"
        if completed_indices
        else ""
    )
    context.save_translation_progress(
        context.task_id,
        completed_indices,
        total_pages,
        status,
        context.pdf_audit["source_page_fallbacks"],
    )
    return completed_indices, cached_indices


def _extract_pages(
    context: PDFExtractionStageContext,
    document: fitz.Document,
    forced_fallbacks: dict[int, dict],
) -> tuple[dict[int, dict], dict[int, fitz.Rect]]:
    page_extractions = {}
    page_rects = {}
    skip_remaining = False
    for page_num in range(len(document)):
        page = document[page_num]
        page_rects[page_num] = fitz.Rect(page.rect)
        if page_num in forced_fallbacks:
            page_extractions[page_num] = {
                "source_page_fallback": forced_fallbacks[page_num]
            }
            context.logger.warning(
                f"[{context.task_id}] Page {page_num + 1}: preserving the exact "
                f"source page after {forced_fallbacks[page_num].get('stage')} failure"
            )
            continue
        try:
            page_text = page.get_text("text").strip()
        except Exception as error:
            context.logger.error(
                f"[{context.task_id}] Page {page_num + 1} text probe failed: {error}"
            )
            raise PDFPageExtractionError(page_num + 1, str(error)) from error
        if _is_skip_page(page_text):
            if not skip_remaining:
                context.logger.info(
                    f"[{context.task_id}] Skip-section detected at page {page_num + 1}"
                )
                skip_remaining = True
            page_extractions[page_num] = {"appendix": True}
            continue
        if skip_remaining:
            context.logger.info(
                f"[{context.task_id}] Resuming translation at page {page_num + 1}"
            )
            skip_remaining = False
        try:
            page_extractions[page_num] = {
                "elements": context.extract_page_elements(page, document)
            }
        except Exception as error:
            context.logger.error(
                f"[{context.task_id}] Page {page_num + 1} extraction failed: {error}"
            )
            raise PDFPageExtractionError(page_num + 1, str(error)) from error
    return page_extractions, page_rects


def extract_document(context: PDFExtractionStageContext) -> PDFExtractionResult:
    """Open and extract the document while owning the extraction semaphore."""

    context.logger.info(
        f"[{context.task_id}] Waiting for PDF extraction slot "
        f"(extraction_concurrency={context.extraction_concurrency})"
    )
    acquired = False
    document = None
    primary_error: BaseException | None = None
    try:
        context.extraction_semaphore.acquire()
        acquired = True
        document = fitz.open(context.src_path)
        if document.is_encrypted and not document.authenticate(context.pdf_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法打开加密PDF")
        total_pages = len(document)
        source_sha256 = _pdf_source_sha256(context.src_path)
        forced_fallbacks = _normalize_source_fallbacks(
            context.pdf_audit,
            total_pages,
        )
        context.logger.info(
            f"[{context.task_id}] Starting PDF translation: {total_pages} pages"
        )
        completed_indices, cached_indices = _load_resume_state(
            context,
            source_sha256,
            total_pages,
        )
        context.logger.info(f"[{context.task_id}] Phase 1: Extracting page elements")
        page_extractions, page_rects = _extract_pages(
            context,
            document,
            forced_fallbacks,
        )
        pages_with_elements = sum(
            1 for value in page_extractions.values() if "elements" in value
        )
        context.logger.info(
            f"[{context.task_id}] Phase 1 done. Pages with extracted elements: "
            f"{pages_with_elements} ({len(cached_indices)} cache candidate(s))"
        )
        return PDFExtractionResult(
            total_pages=total_pages,
            source_sha256=source_sha256,
            completed_indices=completed_indices,
            cached_indices=cached_indices,
            page_extractions=page_extractions,
            page_rects=page_rects,
            forced_fallbacks_by_index=forced_fallbacks,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = None
        try:
            if document is not None:
                try:
                    document.close()
                except Exception as error:
                    if primary_error is None:
                        close_error = error
                    else:
                        context.logger.warning(
                            "[%s] PDF close also failed while preserving the "
                            "primary extraction error: %s",
                            context.task_id,
                            error,
                        )
        finally:
            try:
                if acquired:
                    context.extraction_semaphore.release()
            except Exception as release_error:
                if primary_error is None and close_error is None:
                    raise
                context.logger.warning(
                    "[%s] Failed to release PDF extraction slot while "
                    "preserving an earlier error: %s",
                    context.task_id,
                    release_error,
                )
            if close_error is not None:
                # A close failure after an otherwise successful extraction is
                # still surfaced, but it can no longer mask a page-level root
                # cause that the recovery policy knows how to handle.
                raise close_error


def _bind_and_reconcile_caches(
    context: PDFExtractionStageContext,
    result: PDFExtractionResult,
) -> None:
    cache_repairs = repaired_pages = incremental_pages = incremental_elements = 0
    for page_num in sorted(list(result.completed_indices)):
        info = result.page_extractions.get(page_num, {})
        cache_file = _pdf_page_cache_path(context.task_id, page_num)
        if "elements" not in info or not os.path.exists(cache_file):
            result.completed_indices.discard(page_num)
            continue
        cached = _load_pdf_page_translation_cache(
            cache_file,
            result.source_sha256,
            elements=info["elements"],
        )
        if cached is None:
            result.completed_indices.discard(page_num)
            context.logger.warning(
                f"[{context.task_id}] Page {page_num + 1} cache could not be rebound "
                "to the final current layout; scheduling a clean translation"
            )
            continue
        info["cached"] = cached
    if result.cached_indices:
        active = sum(1 for info in result.page_extractions.values() if "cached" in info)
        context.logger.info(
            f"[{context.task_id}] Identity-rebound {active}/"
            f"{len(result.cached_indices)} candidate page cache(s) to the final "
            "current layout (page count only; integrity is checked next)"
        )

    for page_num in range(result.total_pages):
        info = result.page_extractions.get(page_num, {})
        if "elements" not in info or "cached" not in info:
            continue
        reconciled, repaired = _reconcile_pdf_page_translation_cache(
            info["elements"],
            info["cached"],
        )
        missing = [
            index
            for index, element in enumerate(info["elements"])
            if _pdf_element_requires_translation(element)
            and str(index) not in reconciled
        ]
        try:
            _validate_pdf_page_translations(
                page_num + 1,
                info["elements"],
                reconciled,
                allow_missing=bool(missing),
            )
        except PDFTranslationIntegrityError as error:
            context.logger.warning(
                f"[{context.task_id}] Page {page_num + 1} cached translation "
                f"failed active integrity validation; discarding cache: {error}"
            )
            info.pop("cached", None)
            info.pop("cached_seed", None)
            result.completed_indices.discard(page_num)
            try:
                os.unlink(_pdf_page_cache_path(context.task_id, page_num))
            except FileNotFoundError:
                pass
            continue
        info["cached"] = reconciled
        if repaired:
            _save_pdf_page_translation_cache(
                context.task_id,
                page_num,
                reconciled,
                result.source_sha256,
                elements=info["elements"],
            )
            cache_repairs += repaired
            repaired_pages += 1
        if missing:
            info["cached_seed"] = reconciled
            info["missing_translation_indices"] = missing
            result.completed_indices.discard(page_num)
            incremental_pages += 1
            incremental_elements += len(missing)
    if cache_repairs:
        context.logger.info(
            f"[{context.task_id}] Restored {cache_repairs} preserved formula/cache "
            f"element(s) across {repaired_pages} page(s)"
        )
    if incremental_elements:
        context.logger.info(
            f"[{context.task_id}] Found {incremental_elements} newly translatable "
            f"cached element(s) across {incremental_pages} page(s)"
        )


def reconcile_extraction(
    context: PDFExtractionStageContext,
    result: PDFExtractionResult,
) -> None:
    """Run deterministic layout/merge passes, then bind valid page caches."""

    _mark_watermark_elements(
        result.page_extractions,
        result.page_rects,
        result.total_pages,
    )
    for page_num in range(result.total_pages):
        info = result.page_extractions.get(page_num, {})
        if "elements" in info:
            _classify_pdf_page_text_elements(
                info["elements"],
                result.page_rects[page_num],
            )
    absorbed_tails = _merge_cross_page_sentences(
        result.page_extractions,
        result.total_pages,
        result.page_rects,
        audit_log=context.pdf_audit["merge_decisions"],
    )
    if absorbed_tails:
        context.logger.info(
            f"[{context.task_id}] Completed {absorbed_tails} page-broken "
            "sentence(s) by absorbing next-page tails"
        )
    # Mid-sentence stubs must be marked BEFORE cache binding: the mark is part
    # of the element identity, so a stale cached translation for the stub can
    # never rebind and override the preservation.
    mid_sentence_preserved = _preserve_pdf_formula_mid_sentence_neighbors(
        result.page_extractions
    )
    if mid_sentence_preserved:
        context.pdf_audit["formula_mid_sentence_preserved_elements"] = (
            mid_sentence_preserved
        )
        context.logger.warning(
            f"[{context.task_id}] Kept {len(mid_sentence_preserved)} "
            "mid-sentence stub(s) coupled to protected formula lines as "
            "exact source"
        )
    _preserve_formula_risk_before_cache_binding(context, result)
    enforce_table_preservation(context, result)
    _bind_and_reconcile_caches(context, result)
    result.completed_indices.update(result.forced_fallbacks_by_index)


def _preserve_formula_risk_before_cache_binding(
    context: PDFExtractionStageContext,
    result: PDFExtractionResult,
) -> list[dict]:
    """Freeze formula-risk source ownership before identity-based cache reuse."""

    preserved = _preserve_pdf_formula_risk_text_elements(
        result.page_extractions
    )
    if preserved:
        context.pdf_audit["formula_risk_preserved_elements"] = preserved
        context.logger.warning(
            f"[{context.task_id}] Formula-risk preflight kept {len(preserved)} "
            "math-dense text element(s) as exact source"
        )
    return preserved


def enforce_formula_protection(
    context: PDFExtractionStageContext,
    result: PDFExtractionResult,
) -> None:
    """Fail closed or preserve exact source pages until formula audit is clean."""

    # ``reconcile_extraction`` normally performs this identity-bearing mark
    # before cache binding. Keep the idempotent call for direct stage users.
    _preserve_formula_risk_before_cache_binding(context, result)
    context.pdf_audit["superscript_expectations"] = (
        _collect_pdf_superscript_expectations(result.page_extractions)
    )
    context.pdf_audit["formula_expectations"] = _collect_pdf_formula_expectations(
        result.page_extractions
    )
    context.pdf_audit["vector_ocr_expectations"] = _collect_pdf_vector_ocr_expectations(
        result.page_extractions
    )
    summarize = context.summarize_formula_protection
    context.pdf_audit["formula_protection"] = summarize(
        context.pdf_audit["formula_expectations"],
        result.total_pages,
        result.page_extractions,
    )
    while (
        context.pdf_audit["formula_protection"].get("translation_queue_check") != "ok"
    ):
        if not context.fail_open_to_source_page:
            raise RuntimeError(
                "PDF formula protection audit found an unprotected math symbol "
                "or a protected region eligible for translation"
            )
        current = context.pdf_audit["formula_protection"]
        direct_pages = _pdf_formula_protection_fallback_pages(
            current,
            result.page_extractions,
        )
        fallback_pages = _expand_pdf_fallback_pages_for_accepted_merges(
            direct_pages,
            context.pdf_audit["merge_decisions"],
        )
        existing = {
            int(entry.get("page", 0) or 0)
            for entry in context.pdf_audit["source_page_fallbacks"]
        }
        new_pages = [
            page
            for page in fallback_pages
            if page not in existing
            and "elements" in result.page_extractions.get(page - 1, {})
        ]
        if not new_pages:
            raise RuntimeError(
                "PDF formula protection audit remained non-clean after all "
                "localizable pages were preserved from source"
            )
        context.pdf_audit.setdefault(
            "formula_protection_before_source_fallback",
            current,
        )
        context.pdf_audit.setdefault(
            "formula_protection_source_fallback_rounds",
            [],
        ).append(
            {
                "direct_pages": direct_pages,
                "expanded_pages": sorted(set(fallback_pages) - set(direct_pages)),
                "source_pages": new_pages,
            }
        )
        for page_number in new_pages:
            details = (current.get("per_page") or {}).get(str(page_number), {})
            if page_number in direct_pages:
                reason = (
                    "formula protection preflight: "
                    f"{int(details.get('unprotected_math_symbols_in_text', 0) or 0)} "
                    "unprotected math symbol(s) in translatable text; "
                    f"{int(details.get('formula_queue_entries', 0) or 0)} "
                    "protected region(s) eligible for translation"
                )
            else:
                reason = (
                    "formula protection preflight expanded to preserve an accepted "
                    "cross-page merge linked to warning page(s) "
                    + ", ".join(str(page) for page in direct_pages)
                )
            entry = _pdf_source_page_fallback_entry(
                page_number,
                "formula-protection",
                reason,
                error_type="PDFFormulaProtectionWarning",
            )
            _append_pdf_source_page_fallback(
                context.pdf_audit["source_page_fallbacks"],
                entry,
            )
            result.page_extractions[page_number - 1] = {"source_page_fallback": entry}
            result.completed_indices.add(page_number - 1)
        _exclude_pdf_audit_expectations_for_source_pages(
            context.pdf_audit,
            new_pages,
        )
        context.pdf_audit["formula_protection"] = summarize(
            context.pdf_audit["formula_expectations"],
            result.total_pages,
            result.page_extractions,
        )
        context.save_translation_progress(
            context.task_id,
            result.completed_indices,
            result.total_pages,
            "公式风险页已保留为英文原页，继续翻译和组装...",
            context.pdf_audit["source_page_fallbacks"],
        )
        context.logger.warning(
            f"[{context.task_id}] Formula protection preflight preserved exact "
            f"source page(s) {new_pages}; continuing document delivery"
        )
    context.pdf_audit["total_pages"] = result.total_pages
    context.pdf_audit["phase1_checkpointed_at"] = time.time()
    _save_pdf_audit(context.task_id, context.pdf_audit)


def _mark_pdf_table_preserved_elements(page_extractions: dict) -> list[dict]:
    """Mark table elements to keep their original ink, like formula risks.

    Tables rarely survive translation with row/column alignment intact
    (2026-07-24 policy: preserve them verbatim, exactly like dense math).
    Region-detected cells (``table_hint``) are always preserved; classifier
    ``layout_class == "table"`` elements are preserved only on
    table-dominated pages, because the layout classifier also stamps a few
    stray short fragments per ordinary prose page and those must keep
    translating."""
    preserved = []
    for page_num, info in page_extractions.items():
        elements = info.get("elements") or []
        text_indices = [
            index
            for index, elem in enumerate(elements)
            if elem.get("type") == "text"
        ]
        table_like = [
            index
            for index in text_indices
            if elements[index].get("table_hint")
            or elements[index].get("layout_class") == "table"
        ]
        count = len(table_like)
        # Real data tables carry numbers; a title page's centered
        # author/date block also gets classified "table" but is mostly
        # digit-free prose and must keep translating.
        digit_bearing = sum(
            1
            for index in table_like
            if any(
                char.isdigit()
                for char in str(elements[index].get("content", ""))
            )
        )
        numeric_enough = digit_bearing * 5 >= count * 2
        page_is_tabular = numeric_enough and (
            count >= 10 or (count >= 4 and count * 2 >= len(text_indices))
        )
        marked_rects = []
        for index in table_like:
            elem = elements[index]
            if elem.get("skip_translate_reason"):
                continue
            if not (elem.get("table_hint") or page_is_tabular):
                continue
            elem["skip_translate_reason"] = "table_preserved"
            preserved.append(
                {
                    "page": int(page_num) + 1,
                    "element": index,
                    "trigger": (
                        "region" if elem.get("table_hint") else "page-density"
                    ),
                }
            )
            try:
                rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
            except Exception:
                continue
            if not rect.is_empty:
                marked_rects.append(rect)
        # Grid-union sweep: the classifier occasionally stamps individual
        # cells (a column header, a first data row) as scattered/body.
        # Build the grid box from the LARGEST vertical cluster of preserved
        # cells (a stray table-classified page number must not stretch it
        # across the whole page), then grow it row by row over adjacent
        # column-aligned elements.  Captions and notes sit a paragraph gap
        # away and outside the growth limit, so they keep translating.
        if len(marked_rects) >= 4:
            marked_rects.sort(key=lambda rect: rect.y0)
            clusters = [[marked_rects[0]]]
            for rect in marked_rects[1:]:
                if rect.y0 - clusters[-1][-1].y1 > 30.0:
                    clusters.append([rect])
                else:
                    clusters[-1].append(rect)
            cluster = max(clusters, key=len)
            if len(cluster) >= 4:
                union = fitz.Rect(cluster[0])
                for rect in cluster[1:]:
                    union |= rect
                changed = True
                while changed:
                    changed = False
                    for index in text_indices:
                        elem = elements[index]
                        if elem.get("skip_translate_reason"):
                            continue
                        try:
                            rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
                        except Exception:
                            continue
                        if rect.is_empty:
                            continue
                        intersection = fitz.Rect(rect) & union
                        contained = (
                            not intersection.is_empty
                            and intersection.get_area()
                            >= rect.get_area() * 0.6
                        )
                        x_aligned = (
                            rect.x0 >= union.x0 - 4.0
                            and rect.x1 <= union.x1 + 4.0
                        )
                        gap = max(union.y0 - rect.y1, rect.y0 - union.y1)
                        # Intra-table row gaps run a few points; paragraph
                        # spacing (an abstract under an author block) starts
                        # around 9pt and must stay outside the grid.
                        gap_limit = max(
                            4.0, 0.45 * float(elem.get("fontsize", 12.0))
                        )
                        adjacent = x_aligned and (
                            not intersection.is_empty or gap <= gap_limit
                        )
                        if not (contained or adjacent):
                            continue
                        elem["skip_translate_reason"] = "table_preserved"
                        union |= rect
                        changed = True
                        preserved.append(
                            {
                                "page": int(page_num) + 1,
                                "element": index,
                                "trigger": "grid-union",
                            }
                        )
    return preserved


def enforce_table_preservation(
    context: PDFExtractionStageContext,
    extraction_result: PDFExtractionResult,
) -> None:
    """Apply the preserve-tables policy and record it in the audit."""

    if not context.element_policy.preserve_tables:
        context.pdf_audit["table_preserved_elements"] = []
        return
    preserved = _mark_pdf_table_preserved_elements(
        extraction_result.page_extractions
    )
    if preserved or "table_preserved_elements" not in context.pdf_audit:
        context.pdf_audit["table_preserved_elements"] = preserved
    if preserved:
        pages = sorted({entry["page"] for entry in preserved})
        context.logger.info(
            f"[{context.task_id}] Preserving {len(preserved)} table "
            f"element(s) verbatim on page(s) {pages}"
        )


def build_translation_plan(
    context: PDFExtractionStageContext,
    result: PDFExtractionResult,
    system_prompt: str,
    build_glossary: Callable[[str, str], str],
    make_prompt_with_glossary: Callable[[str, str], str],
) -> PDFTranslationPlan:
    """Derive render styles and build the glossary only when it is needed."""

    for page_num in range(result.total_pages):
        info = result.page_extractions.get(page_num, {})
        if "elements" in info:
            info["layout_styles"] = _derive_pdf_page_layout_styles(
                info["elements"],
                result.page_rects[page_num],
            )
    pages_to_translate = [
        page_num
        for page_num in range(result.total_pages)
        if "elements" in result.page_extractions.get(page_num, {})
        and page_num not in result.completed_indices
    ]
    all_source_text = "".join(
        element["content"] + "\n"
        for page_num in range(result.total_pages)
        for element in result.page_extractions.get(page_num, {}).get("elements", [])
        if element["type"] == "text"
    )
    incremental_fill = bool(pages_to_translate) and all(
        "cached_seed" in result.page_extractions.get(page_num, {})
        for page_num in pages_to_translate
    )
    if pages_to_translate and not incremental_fill:
        context.save_translation_progress(
            context.task_id,
            result.completed_indices,
            result.total_pages,
            "正在构建术语表...",
            context.pdf_audit["source_page_fallbacks"],
        )
        glossary = build_glossary(all_source_text, context.task_id)
    elif incremental_fill:
        glossary = ""
        context.logger.info(
            f"[{context.task_id}] Reusing cached translations and skipping "
            f"glossary/API rebuild for {len(pages_to_translate)} incremental page(s)"
        )
    else:
        glossary = ""
        context.logger.info(
            f"[{context.task_id}] All PDF pages are cached; skipping glossary/API "
            "translation and reassembling output"
        )
    context.logger.info(f"[{context.task_id}] Glossary ready ({len(glossary)} chars)")
    return PDFTranslationPlan(
        pages_to_translate=pages_to_translate,
        prompt=make_prompt_with_glossary(system_prompt, glossary),
    )


__all__ = [
    "PDFExtractionResult",
    "PDFExtractionStageContext",
    "PDFTranslationPlan",
    "build_translation_plan",
    "enforce_formula_protection",
    "extract_document",
    "reconcile_extraction",
]
