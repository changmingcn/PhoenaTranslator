"""Pipeline for deterministic PDF processing."""

from __future__ import annotations

import time

from phoena_translator.pdf.context import (
    PDFPageTranslationContext,
    PDFPipelineDependencies,
)
from phoena_translator.pdf.extraction_stage import (
    PDFExtractionStageContext,
    build_translation_plan,
    enforce_formula_protection,
    extract_document,
    reconcile_extraction,
)
from phoena_translator.pdf.output_stage import (
    PDFOutputStageContext,
    assemble_output,
    finalize_output,
    resolve_output_fonts,
)
from phoena_translator.pdf.recovery import (
    PDFRecoveryContext,
    PDFRecoveryDecision,
    recover_pdf_failure,
)
from phoena_translator.pdf.translation_stage import (
    PDFTranslationStageContext,
    translate_pages,
)


# A source page can advance recovery only once, and structural repair advances
# once. The actual bound tightens to the reported source-page count; 512 is a
# defensive ceiling when an extraction failure occurs before that count can be
# reported, and still leaves deliberate headroom above the 333-page regression.
PDF_MAX_RECOVERY_ROUNDS = 512


def _new_pdf_audit(
    task_id: str,
    minimum_htmlbox_scale: float,
    source_page_fallbacks: list[dict],
) -> dict:
    return {
        "schema_version": 2,
        "task_id": task_id,
        "created_at": time.time(),
        "merge_decisions": [],
        "superscript_expectations": [],
        "formula_expectations": [],
        "vector_ocr_expectations": [],
        "formula_protection": {},
        "htmlbox_placement_events": [],
        "render_scale_events": [],
        "rejected_render_scale_events": [],
        "textbox_fallback_events": [],
        "untranslated_english_leaks": [],
        "source_page_fallbacks": [
            dict(entry) for entry in source_page_fallbacks
        ],
        "minimum_acceptable_render_scale": minimum_htmlbox_scale,
    }


def _fallback_page_numbers(entries: list[dict] | tuple[dict, ...]) -> tuple[int, ...]:
    pages = set()
    for entry in entries:
        try:
            page_number = int(entry.get("page", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if page_number > 0:
            pages.add(page_number)
    return tuple(sorted(pages))


def _recovery_state_signature(
    structure_retry: int,
    source_page_fallbacks: list[dict] | tuple[dict, ...],
) -> tuple[int, tuple[int, ...]]:
    return int(structure_retry), _fallback_page_numbers(source_page_fallbacks)


def _retry_makes_progress(
    current_state: tuple[int, tuple[int, ...]],
    decision: PDFRecoveryDecision,
) -> bool:
    current_structure_retry, current_page_tuple = current_state
    next_state = _recovery_state_signature(
        decision.structure_retry,
        decision.source_page_fallbacks,
    )
    next_structure_retry, next_page_tuple = next_state
    current_pages = set(current_page_tuple)
    next_pages = set(next_page_tuple)
    source_page_progress = (
        next_structure_retry == current_structure_retry
        and next_pages > current_pages
    )
    structural_progress = (
        current_structure_retry == 0
        and next_structure_retry == 1
        and next_pages == current_pages
    )
    return source_page_progress or structural_progress


def _reported_total_pages(
    task_id: str,
    pdf_audit: dict,
    dependencies: PDFPipelineDependencies,
) -> int | None:
    candidates = [
        pdf_audit.get("total_pages"),
        dependencies.tasks.get(task_id, {}).get("total"),
    ]
    for candidate in candidates:
        try:
            total_pages = int(candidate)
        except (TypeError, ValueError):
            continue
        if total_pages > 0:
            return total_pages
    return None


def _tighten_recovery_round_limit(
    current_limit: int,
    *,
    task_id: str,
    pdf_audit: dict,
    dependencies: PDFPipelineDependencies,
    initial_fallback_pages: tuple[int, ...],
    initial_structure_retry: int,
) -> int:
    if not dependencies.fail_open_to_source_page:
        structural_rounds = 1 if initial_structure_retry == 0 else 0
        return min(current_limit, structural_rounds)
    total_pages = _reported_total_pages(task_id, pdf_audit, dependencies)
    if total_pages is None:
        return current_limit
    existing_valid_pages = sum(
        1 for page in initial_fallback_pages if 1 <= page <= total_pages
    )
    return min(current_limit, max(total_pages - existing_valid_pages, 0))


def _run_pdf_attempt(
    task_id: str,
    src_path: str,
    out_path: str,
    pdf_password: str,
    pdf_audit: dict,
    *,
    dependencies: PDFPipelineDependencies,
) -> None:
    """Run one PDF attempt; retry control belongs only to ``translate_pdf``."""

    font_path, bold_font_path = resolve_output_fonts(
        dependencies.font_regular_override,
        dependencies.font_bold_override,
    )
    extraction_context = PDFExtractionStageContext(
        task_id=task_id,
        src_path=src_path,
        pdf_password=pdf_password,
        progress_dir=dependencies.progress_dir,
        extraction_concurrency=dependencies.extraction_max_concurrency,
        fail_open_to_source_page=dependencies.fail_open_to_source_page,
        element_policy=dependencies.element_policy,
        extraction_semaphore=dependencies.extraction_semaphore,
        logger=dependencies.logger,
        pdf_audit=pdf_audit,
        load_progress=dependencies.load_progress,
        save_translation_progress=dependencies.save_translation_progress,
        extract_page_elements=dependencies.extract_page_elements,
        summarize_formula_protection=dependencies.summarize_formula_protection,
    )
    extraction_result = extract_document(extraction_context)
    total_pages = extraction_result.total_pages
    pdf_audit["total_pages"] = total_pages
    source_sha256 = extraction_result.source_sha256
    completed_indices = extraction_result.completed_indices
    page_extractions = extraction_result.page_extractions
    page_rects = extraction_result.page_rects

    reconcile_extraction(extraction_context, extraction_result)
    enforce_formula_protection(extraction_context, extraction_result)
    translation_plan = build_translation_plan(
        extraction_context,
        extraction_result,
        dependencies.system_prompt_text,
        dependencies.build_glossary,
        dependencies.make_prompt_with_glossary,
    )

    dependencies.logger.info(
        f"[{task_id}] Phase 2: Concurrent translation "
        f"(workers={dependencies.translation_workers}, "
        f"api_concurrency={dependencies.api_max_concurrency})"
    )
    dependencies.save_translation_progress(
        task_id,
        completed_indices,
        total_pages,
        (
            "并发翻译中 "
            f"(线程 {dependencies.translation_workers}/"
            f"API {dependencies.api_max_concurrency})"
        ),
        pdf_audit["source_page_fallbacks"],
    )
    page_translation_context = PDFPageTranslationContext(
        task_id=task_id,
        prompt=translation_plan.prompt,
        logger=dependencies.logger,
        translate_text=dependencies.translate_text,
    )
    translation_stage_context = PDFTranslationStageContext(
        task_id=task_id,
        total_pages=total_pages,
        source_sha256=source_sha256,
        workers=dependencies.translation_workers,
        fail_open_to_source_page=dependencies.fail_open_to_source_page,
        logger=dependencies.logger,
        page_extractions=page_extractions,
        pdf_audit=pdf_audit,
        completed_indices=completed_indices,
        save_translation_progress=dependencies.save_translation_progress,
    )
    translate_pages(
        translation_stage_context,
        page_translation_context,
        translation_plan.pages_to_translate,
    )

    output_context = PDFOutputStageContext(
        task_id=task_id,
        src_path=src_path,
        out_path=out_path,
        pdf_password=pdf_password,
        total_pages=total_pages,
        assembly_concurrency=dependencies.assembly_max_concurrency,
        use_htmlbox=dependencies.use_htmlbox,
        minimum_htmlbox_scale=dependencies.minimum_htmlbox_scale,
        save_clean=dependencies.save_clean,
        save_garbage=dependencies.save_garbage,
        font_subsetting_available=dependencies.font_subsetting_available,
        font_path=font_path,
        bold_font_path=bold_font_path,
        assembly_semaphore=dependencies.assembly_semaphore,
        logger=dependencies.logger,
        tasks=dependencies.tasks,
        pdf_audit=pdf_audit,
        page_extractions=page_extractions,
        page_rects=page_rects,
        completed_indices=completed_indices,
        save_progress=dependencies.save_progress,
        save_translation_progress=dependencies.save_translation_progress,
        trim_process_memory=dependencies.trim_process_memory,
        filter_formula_safe_rects=dependencies.filter_formula_safe_rects,
        check_output_structure_serialized=(
            dependencies.check_output_structure_serialized
        ),
    )
    assemble_output(output_context)
    finalize_output(
        output_context,
        fail_open_to_source_page=dependencies.fail_open_to_source_page,
    )


def _recovery_context(
    task_id: str,
    structure_retry: int,
    pdf_audit: dict,
    dependencies: PDFPipelineDependencies,
) -> PDFRecoveryContext:
    return PDFRecoveryContext(
        task_id=task_id,
        structure_retry=structure_retry,
        fail_open_to_source_page=dependencies.fail_open_to_source_page,
        pdf_audit=pdf_audit,
        tasks=dependencies.tasks,
        logger=dependencies.logger,
        save_progress=dependencies.save_progress,
        trim_process_memory=dependencies.trim_process_memory,
    )


def translate_pdf(
    task_id: str,
    src_path: str,
    out_path: str,
    pdf_password: str = "",
    _structure_retry: int = 0,
    _forced_source_page_fallbacks: list[dict] | None = None,
    *,
    dependencies: PDFPipelineDependencies,
):
    """Translate a PDF with bounded, constant-stack recovery."""

    structure_retry = 1 if int(_structure_retry) >= 1 else 0
    source_page_fallbacks = [
        dict(entry) for entry in (_forced_source_page_fallbacks or [])
    ]
    current_state = _recovery_state_signature(
        structure_retry,
        source_page_fallbacks,
    )
    initial_structure_retry = structure_retry
    initial_fallback_pages = current_state[1]
    seen_states = {current_state}
    recovery_round_limit = PDF_MAX_RECOVERY_ROUNDS

    for recovery_round in range(PDF_MAX_RECOVERY_ROUNDS + 1):
        pdf_audit = _new_pdf_audit(
            task_id,
            dependencies.minimum_htmlbox_scale,
            source_page_fallbacks,
        )
        try:
            _run_pdf_attempt(
                task_id,
                src_path,
                out_path,
                pdf_password,
                pdf_audit,
                dependencies=dependencies,
            )
            return None
        except Exception as error:
            context = _recovery_context(
                task_id,
                structure_retry,
                pdf_audit,
                dependencies,
            )
            decision = recover_pdf_failure(context, error)
            if not decision.should_retry:
                return None

            recovery_round_limit = _tighten_recovery_round_limit(
                recovery_round_limit,
                task_id=task_id,
                pdf_audit=pdf_audit,
                dependencies=dependencies,
                initial_fallback_pages=initial_fallback_pages,
                initial_structure_retry=initial_structure_retry,
            )

            next_state = _recovery_state_signature(
                decision.structure_retry,
                decision.source_page_fallbacks,
            )
            if not _retry_makes_progress(current_state, decision):
                stalled = RuntimeError(
                    "PDF recovery refused a retry without a new source page or "
                    "the single structural-repair transition"
                )
                recover_pdf_failure(context, stalled)
                return None
            if next_state in seen_states:
                stalled = RuntimeError(
                    f"PDF recovery state repeated without progress: {next_state}"
                )
                recover_pdf_failure(context, stalled)
                return None
            if recovery_round >= recovery_round_limit:
                exhausted = RuntimeError(
                    "PDF recovery exceeded the explicit limit of "
                    f"{recovery_round_limit} retry round(s)"
                )
                recover_pdf_failure(context, exhausted)
                return None

            seen_states.add(next_state)
            current_state = next_state
            structure_retry = decision.structure_retry
            source_page_fallbacks = [
                dict(entry) for entry in decision.source_page_fallbacks
            ]
            dependencies.logger.warning(
                f"[{task_id}] Advancing bounded PDF recovery round "
                f"{recovery_round + 1}/{recovery_round_limit}: "
                f"{decision.reason}"
            )


__all__ = ["translate_pdf"]
