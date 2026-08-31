"""Source extraction, cache reconciliation, and formula-preflight stages."""

from __future__ import annotations

import logging
import os
import re
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
from phoena_translator.pdf.geometry import _pdf_page_extraction_rect
from phoena_translator.pdf.math_detection import _plain_text
from phoena_translator.pdf.semantic_cross_page import (
    _cross_page_dominant_fontsize,
)
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
        # Every consumer of this rect compares it against EXTRACTION geometry
        # -- header/footer bands, column widths, cross-page body eligibility.
        # ``page.rect`` is the rotated display box, so on a ``/Rotate 90`` page
        # the ratios were taken against a 792x612 box while the ink lived in a
        # 612x792 one, and body text in the bottom extraction band came out
        # classified ``scattered`` instead of ``body`` -- silently disqualifying
        # it as a cross-page merge source.  For an unrotated page the helper
        # returns ``page.rect`` unchanged, so this is a no-op there.
        page_rects[page_num] = _pdf_page_extraction_rect(page)
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
    # Preservation must be decided BEFORE cross-page merging, not after.
    # ``_is_cross_page_body_elem`` already refuses any element carrying a
    # ``skip_translate_reason`` as a merge endpoint, but running the stampers
    # afterwards defeated that guard: a preserved element re-renders its
    # ORIGINAL ink, so a preserved destination kept the carried tail visible
    # on page N+1 while page N shipped it translated, and a preserved source
    # dropped the tail entirely.  Deciding first also keeps both operands of
    # the formula-risk test on the same side of the merge.  The stamps still
    # precede cache binding below, which is the constraint the identity
    # reconciliation actually requires.
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
    report_untranslated_delivered_pages(context, result)
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


def _pdf_element_is_body_prose(elem: dict, dominant_fontsize: float) -> bool:
    """Running prose set in the document's body type — never a table cell.

    A preserved element keeps its English ink, which is right for grid cells
    and chart labels and never right for a paragraph.  Three signals have to
    agree, because each one alone has a real counter-example:

    * body type — data tables and chart annotations are set smaller than the
      running text, but a term/definition table is not;
    * a sentence's worth of words (the threshold matches
      ``_looks_like_pdf_body_text_false_table`` so both prose/table
      discriminators read the same), but a cell can hold a whole sentence;
    * and either enough words that no cell would hold them, or a line wrap —
      a cell is laid out to fit its row, a paragraph wraps.
    """
    if elem.get("type") != "text":
        return False
    plain = re.sub(r"\s+", " ", _plain_text(elem.get("content") or "")).strip()
    words = len(re.findall(r"[A-Za-z][A-Za-z'’-]*", plain))
    if words < 8:
        return False
    fontsize = float(
        elem.get("fontsize", dominant_fontsize) or dominant_fontsize
    )
    if fontsize < dominant_fontsize * 0.92:
        return False
    if words >= 12:
        return True
    line_height = max(
        float(elem.get("line_height", 0.0) or 0.0),
        fontsize * 1.2,
        8.0,
    )
    try:
        rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
    except Exception:
        return False
    return rect.height > line_height * 1.6


def _pdf_table_marked_rect_clusters(
    rects: list[fitz.Rect],
) -> list[list[fitz.Rect]]:
    """Group marked cell rects into one cluster per table.

    Clustering on vertical adjacency alone merges a table in the left text
    column with a chart in the right one.  The merged union then spans the
    page, every later ``x_aligned`` test passes vacuously, and the sweep walks
    up the text columns paragraph by paragraph (2026-07-27: this delivered a
    whole page of a two-column paper in English).  Cells of one table always
    share horizontal extent, so require that too.
    """
    clusters: list[list[fitz.Rect]] = []
    for rect in sorted(rects, key=lambda item: (item.y0, item.x0)):
        touching = [
            cluster
            for cluster in clusters
            if any(
                min(rect.x1, member.x1) - max(rect.x0, member.x0) > -12.0
                and max(rect.y0 - member.y1, member.y0 - rect.y1) <= 30.0
                for member in cluster
            )
        ]
        if not touching:
            clusters.append([rect])
            continue
        merged = [rect]
        for cluster in touching:
            merged.extend(cluster)
            clusters.remove(cluster)
        clusters.append(merged)
    return clusters


def _pdf_element_in_page_footer_band(elem: dict, page_rect) -> bool:
    """True for the running footer band under the last body/table line.

    Only the foot of the page is fenced off, not the classifier's symmetric
    header band: a page can legitimately open with a chart box, and its title
    sits inside the top band.
    """
    if page_rect is None:
        return False
    height = float(getattr(page_rect, "height", 0.0) or 0.0)
    if height <= 0.0:
        return False
    try:
        rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
    except Exception:
        return False
    if rect.is_empty:
        return False
    center_y = ((rect.y0 + rect.y1) / 2.0 - float(page_rect.y0)) / height
    return center_y >= 0.92


def _mark_pdf_table_preserved_elements(
    page_extractions: dict,
    rollbacks: list[dict] | None = None,
    page_rects: dict | None = None,
) -> list[dict]:
    """Mark table elements to keep their original ink, like formula risks.

    Tables rarely survive translation with row/column alignment intact
    (2026-07-24 policy: preserve them verbatim, exactly like dense math).
    Region-detected cells (``table_hint``) are always preserved; classifier
    ``layout_class == "table"`` elements are preserved only on
    table-dominated pages, because the layout classifier also stamps a few
    stray short fragments per ordinary prose page and those must keep
    translating.  Body prose is never eligible for either heuristic trigger,
    and a page whose prose would end up frozen anyway rolls the whole page's
    marks back."""
    preserved = []
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions or {})
    for page_num, info in page_extractions.items():
        elements = info.get("elements") or []
        text_indices = [
            index
            for index, elem in enumerate(elements)
            if elem.get("type") == "text"
        ]
        prose_indices = {
            index
            for index in text_indices
            if _pdf_element_is_body_prose(elements[index], dominant_fontsize)
        }
        # Region geometry stays authoritative: a cell inside a detected grid
        # is preserved whatever it reads like.  The classifier is the noisy
        # signal, so it may not speak for prose.
        table_like = [
            index
            for index in text_indices
            if elements[index].get("table_hint")
            or (
                elements[index].get("layout_class") == "table"
                and index not in prose_indices
            )
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
        # Grid sweep: the classifier occasionally stamps individual cells (a
        # column header, a first data row, the second line of a table note)
        # as scattered/body.  Each unmarked element is compared against the
        # marked elements themselves rather than against one box grown from
        # all of them: a single box merges a table in the left text column
        # with a chart in the right one, spans the page, and then walks up
        # both text columns row by row (2026-07-27: a whole page of a
        # two-column paper delivered in English).  Chaining locally cannot
        # jump columns, because every step needs horizontal overlap with the
        # element it grows from, and cannot cross a paragraph, because prose
        # is not eligible.
        page_rect = (page_rects or {}).get(page_num)
        if len(marked_rects) >= 4:
            changed = True
            while changed:
                changed = False
                anchors = [
                    (index, fitz.Rect(elements[index].get("bbox", elements[index].get("rect"))))
                    for index in text_indices
                    if elements[index].get("skip_translate_reason")
                    == "table_preserved"
                ]
                for index in text_indices:
                    elem = elements[index]
                    if elem.get("skip_translate_reason"):
                        continue
                    if index in prose_indices:
                        continue
                    # The sweep exists to recover cells the classifier stamped
                    # inconsistently.  ``body`` is not an inconsistent stamp,
                    # it is the classifier saying "running text" — and a page
                    # of stacked reference entries reads to every other signal
                    # exactly like a page of table rows.
                    if elem.get("layout_class") == "body":
                        continue
                    # The running footer sits a few points under the last
                    # note line of a chart box in the same column; it belongs
                    # to the page, not to the table.
                    if _pdf_element_in_page_footer_band(elem, page_rect):
                        continue
                    try:
                        rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
                    except Exception:
                        continue
                    if rect.is_empty:
                        continue
                    # Intra-table row gaps run a few points; a paragraph gap is
                    # wider and stays outside.  Deliberately the same limit the
                    # box-growing sweep used: this rewrite is here to stop the
                    # sweep reaching ACROSS a page, not to let it reach further.
                    gap_limit = max(
                        4.0, 0.45 * float(elem.get("fontsize", 12.0) or 12.0)
                    )
                    attached = False
                    for anchor_index, anchor in anchors:
                        if anchor.is_empty:
                            continue
                        overlap = min(rect.x1, anchor.x1) - max(
                            rect.x0, anchor.x0
                        )
                        if overlap < min(rect.width, anchor.width) * 0.5:
                            continue
                        gap = max(anchor.y0 - rect.y1, rect.y0 - anchor.y1)
                        if gap > gap_limit:
                            continue
                        attached = True
                        break
                    if not attached:
                        continue
                    elem["skip_translate_reason"] = "table_preserved"
                    changed = True
                    preserved.append(
                        {
                            "page": int(page_num) + 1,
                            "element": index,
                            "trigger": "grid-union",
                        }
                    )
        # Circuit breaker.  Verbatim preservation exists so a table keeps its
        # geometry; it must never be the reason a reader gets a page of
        # untranslated prose.  If this page's marks would freeze most of its
        # running text anyway — a table region mis-detected over a whole
        # column, some layout nobody has seen yet — drop every mark the policy
        # made here and let the page translate.  Delivering a translated page
        # with a reflowed table beats delivering the source page.
        prose_total = sum(
            len(str(elements[index].get("content", "")))
            for index in prose_indices
        )
        prose_frozen = sum(
            len(str(elements[index].get("content", "")))
            for index in prose_indices
            if elements[index].get("skip_translate_reason") == "table_preserved"
        )
        if prose_total >= 200 and prose_frozen * 2 > prose_total:
            page_marks = [
                entry
                for entry in preserved
                if entry["page"] == int(page_num) + 1
            ]
            for entry in page_marks:
                elem = elements[entry["element"]]
                if elem.get("skip_translate_reason") == "table_preserved":
                    elem.pop("skip_translate_reason", None)
                preserved.remove(entry)
            if rollbacks is not None:
                rollbacks.append({
                    "page": int(page_num) + 1,
                    "reason": "table-preservation-would-freeze-page-prose",
                    "released_elements": len(page_marks),
                    "prose_chars": int(prose_total),
                    "frozen_prose_chars": int(prose_frozen),
                })
    return preserved


def enforce_table_preservation(
    context: PDFExtractionStageContext,
    extraction_result: PDFExtractionResult,
) -> None:
    """Apply the preserve-tables policy and record it in the audit."""

    if not context.element_policy.preserve_tables:
        context.pdf_audit["table_preserved_elements"] = []
        return
    rollbacks: list[dict] = []
    preserved = _mark_pdf_table_preserved_elements(
        extraction_result.page_extractions,
        rollbacks,
        extraction_result.page_rects,
    )
    if preserved or "table_preserved_elements" not in context.pdf_audit:
        context.pdf_audit["table_preserved_elements"] = preserved
    if rollbacks or "table_preservation_rollbacks" not in context.pdf_audit:
        context.pdf_audit["table_preservation_rollbacks"] = rollbacks
    if preserved:
        pages = sorted({entry["page"] for entry in preserved})
        context.logger.info(
            f"[{context.task_id}] Preserving {len(preserved)} table "
            f"element(s) verbatim on page(s) {pages}"
        )
    for entry in rollbacks:
        context.logger.warning(
            f"[{context.task_id}] Page {entry['page']}: released "
            f"{entry['released_elements']} table-preserved element(s) — "
            "keeping them verbatim would have left "
            f"{entry['frozen_prose_chars']}/{entry['prose_chars']} body-prose "
            "characters untranslated"
        )


# A page with less running prose than this is a plate: a full-page chart, a
# table sheet, a cover.  Above it, delivering almost none of that prose in the
# target language is a defect however the elements got exempted.
PDF_UNTRANSLATED_PAGE_MIN_PROSE_CHARS = 400
PDF_UNTRANSLATED_PAGE_MIN_TRANSLATED_SHARE = 0.25


def collect_untranslated_delivered_pages(
    page_extractions: dict,
    page_rects: dict | None = None,
) -> list[dict]:
    """Report pages whose body prose will ship almost entirely as source.

    Every existing delivery gate keys off ``skip_translate_reason`` or
    ``_pdf_element_requires_translation``, so the one decision that exempts an
    element also excuses it from every later check.  Page 4 of a 2020 Bank of
    Japan Review shipped 100% English with an empty leak list, no element
    fallbacks and ``structure_check: ok``, because "everything was exempted"
    is indistinguishable from "nothing leaked".  This asks the one question
    none of them ask: did the reader get this page in the target language?

    Reports only.  Nothing here may block or fall back — a page that is
    legitimately all table still delivers.
    """
    del page_rects  # reserved; prose detection is typographic, not positional
    findings: list[dict] = []
    dominant_fontsize = _cross_page_dominant_fontsize(page_extractions or {})
    for page_num in sorted(page_extractions or {}):
        info = page_extractions.get(page_num) or {}
        elements = info.get("elements") or []
        if not elements:
            continue  # appendix, skip page, or an already-declared fallback
        prose = [
            elem
            for elem in elements
            if _pdf_element_is_body_prose(elem, dominant_fontsize)
        ]
        total = sum(len(str(elem.get("content", ""))) for elem in prose)
        if total < PDF_UNTRANSLATED_PAGE_MIN_PROSE_CHARS:
            continue
        translatable = sum(
            len(str(elem.get("content", "")))
            for elem in prose
            if _pdf_element_requires_translation(elem)
        )
        if translatable >= total * PDF_UNTRANSLATED_PAGE_MIN_TRANSLATED_SHARE:
            continue
        reasons = sorted(
            {
                str(elem.get("skip_translate_reason"))
                for elem in prose
                if elem.get("skip_translate_reason")
            }
        )
        findings.append({
            "page": int(page_num) + 1,
            "prose_chars": int(total),
            "translatable_prose_chars": int(translatable),
            "reasons": reasons,
        })
    return findings


def report_untranslated_delivered_pages(
    context: PDFExtractionStageContext,
    extraction_result: PDFExtractionResult,
) -> None:
    """Record the untranslated-page audit; never blocks delivery."""

    findings = collect_untranslated_delivered_pages(
        extraction_result.page_extractions,
        extraction_result.page_rects,
    )
    context.pdf_audit["untranslated_delivered_pages"] = findings
    for entry in findings:
        context.logger.warning(
            f"[{context.task_id}] Page {entry['page']} will ship "
            f"{entry['translatable_prose_chars']}/{entry['prose_chars']} "
            "body-prose characters translated; the rest is kept as source "
            f"({', '.join(entry['reasons']) or 'no recorded reason'})"
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
