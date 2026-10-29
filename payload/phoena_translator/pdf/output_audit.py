"""Verify the assembled output PDF against the source and the extraction
expectations (structure, superscripts, formula regions, recovered text)."""

from __future__ import annotations

import math
import shutil
import re
import os
import fitz
import logging
from phoena_translator.pdf.superscripts import (
    _pdf_exact_source_native_superscript_matches,
    _pdf_is_semantic_superscript_marker,
    _pdf_nearby_superscript_reference,
    _pdf_span_has_lexical_text,
)
from phoena_translator.pdf.model import PageExtraction, PdfAudit
from phoena_translator.pdf.vector_ocr import (
    _pdf_vector_ocr_normalized_match_text,
    _render_pdf_page_png_for_ocr,
    _run_pdf_tesseract,
)
from phoena_translator.pdf.errors import PDFStructureValidationError
from phoena_translator.pdf.cache import _save_pdf_audit
from phoena_translator.pdf.merge_audit import _validate_pdf_merge_audit
from phoena_translator.pdf.geometry import _pdf_extraction_rect_to_display
from phoena_translator.pdf.signatures import _pdf_formula_region_signature
from phoena_translator.pdf.inline_markup import _pdf_marker_pattern
from typing import Any, Callable
from dataclasses import dataclass


log = logging.getLogger("translator")


@dataclass(frozen=True)
class PDFAuditDependencies:
    extraction_max_concurrency: int
    extraction_semaphore: Any
    check_output_structure: Callable[..., dict]
    trim_process_memory: Callable[[], None]
    save_progress: Callable[[str, dict], None]


def _pdf_vector_ocr_page_text(page) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract is unavailable for vector OCR output audit")
    png_bytes, _scale = _render_pdf_page_png_for_ocr(page)
    return _run_pdf_tesseract(png_bytes, "txt")


def _pdf_formula_raster_difference_summary(source_pix, observed_pix) -> dict:
    """Classify raster-only differences after vector identity is confirmed.

    Exact raster hashes catch overlays and whiteouts, but they also change when
    translated neighboring text used to overlap a protected box by a few
    antialiased edge pixels. Accept only a tiny global delta or a bounded delta
    confined to an outer edge; material changes in the formula core still fail.
    """
    same_shape = (
        source_pix.width == observed_pix.width
        and source_pix.height == observed_pix.height
        and source_pix.n == observed_pix.n
        and len(source_pix.samples) == len(observed_pix.samples)
    )
    if not same_shape or source_pix.n != 1:
        return {
            "tolerable": False,
            "diff_pixels": None,
            "diff_ratio": None,
            "boundary_only": False,
        }

    width = int(source_pix.width)
    height = int(source_pix.height)
    different = [
        index
        for index, (source_value, observed_value) in enumerate(
            zip(source_pix.samples, observed_pix.samples)
        )
        if source_value != observed_value
    ]
    total = max(width * height, 1)
    diff_count = len(different)
    diff_ratio = diff_count / total
    if diff_count == 0:
        return {
            "tolerable": True,
            "diff_pixels": 0,
            "diff_ratio": 0.0,
            "boundary_only": True,
        }

    vertical_band = max(1, min(10, math.ceil(height * 0.30)))
    horizontal_band = max(1, min(10, math.ceil(width * 0.08)))
    boundary_only = all(
        (index // width) < vertical_band
        or (index // width) >= height - vertical_band
        or (index % width) < horizontal_band
        or (index % width) >= width - horizontal_band
        for index in different
    )
    tolerable = diff_ratio <= 0.002 or (boundary_only and diff_ratio <= 0.10)
    return {
        "tolerable": tolerable,
        "diff_pixels": diff_count,
        "diff_ratio": round(diff_ratio, 6),
        "boundary_only": boundary_only,
    }


def _pdf_glyph_substitution_warnings(
    source_text: str,
    output_text: str,
    page_number: int,
) -> list[dict]:
    """Detect renderer-introduced missing-glyph text on one output page.

    A translated page may remove ASCII question marks, but it must not create
    more of them than the source page contained.  PyMuPDF's built-in fonts use
    literal ``?`` for unsupported CJK glyphs, while some renderers expose the
    Unicode replacement character.  Compare against the source so legitimate
    question punctuation or pre-existing damaged glyphs do not become false
    positives.
    """
    source_text = source_text or ""
    output_text = output_text or ""
    warnings = []
    for character, label in (("?", "ascii-question-mark"), ("\ufffd", "unicode-replacement")):
        source_count = source_text.count(character)
        output_count = output_text.count(character)
        if output_count <= source_count:
            continue
        warning = {
            "type": "font-glyph-substitution",
            "page": int(page_number),
            "glyph": label,
            "source_count": source_count,
            "output_count": output_count,
            "introduced_count": output_count - source_count,
        }
        if character == "?":
            warning["longest_output_run"] = max(
                (len(match.group(0)) for match in re.finditer(r"\?+", output_text)),
                default=0,
            )
        warnings.append(warning)
    return warnings


def _pdf_vector_source_glyph_allowance(
    vector_ocr_expectations: list[dict] | None,
) -> dict[int, dict[str, int]]:
    """Count source glyphs intentionally consumed by vector OCR."""
    allowance_by_page: dict[int, dict[str, int]] = {}
    for expectation in vector_ocr_expectations or []:
        page_number = int(expectation.get("page", 0))
        allowance = allowance_by_page.setdefault(
            page_number,
            {"?": 0, "\ufffd": 0},
        )
        allowance["?"] += int(
            expectation.get("source_ascii_question_marks", 0)
        )
        allowance["\ufffd"] += int(
            expectation.get("source_replacement_characters", 0)
        )
    return allowance_by_page


def _open_pdf_audit_source(
    source_path: str | None,
    source_password: str,
):
    """Open and authenticate the optional source document."""
    if not source_path or not os.path.exists(source_path):
        return None
    source_doc = fitz.open(source_path)
    if source_doc.is_encrypted and not source_doc.authenticate(source_password):
        source_doc.close()
        return None
    return source_doc


def _pdf_audit_page_data(
    doc,
    page_number: int,
    page_cache: dict[int, dict],
    *,
    include_dict: bool,
) -> dict:
    """Load each audit page representation at most once."""
    if page_number not in page_cache:
        page = doc[page_number - 1]
        page_cache[page_number] = {
            "text": page.get_text("text"),
            "dict": None,
        }
    if include_dict and page_cache[page_number].get("dict") is None:
        page_cache[page_number]["dict"] = doc[page_number - 1].get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE,
        )
    return page_cache[page_number]


def _audit_pdf_source_structure(
    doc,
    source_doc,
    vector_glyph_allowance: dict[int, dict[str, int]],
    page_cache: dict[int, dict],
    source_page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> None:
    """Compare page counts and high-risk glyph substitutions."""
    if source_doc is None:
        return
    if len(doc) != len(source_doc):
        warnings.append({
            "type": "page-count-mismatch",
            "source_pages": len(source_doc),
            "output_pages": len(doc),
        })
    for page_number in range(1, min(len(doc), len(source_doc)) + 1):
        output_page = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=False,
        )
        source_page = _pdf_audit_page_data(
            source_doc,
            page_number,
            source_page_cache,
            include_dict=False,
        )
        allowance = vector_glyph_allowance.get(page_number, {})
        glyph_warnings = _pdf_glyph_substitution_warnings(
            source_page["text"]
            + "?" * allowance.get("?", 0)
            + "\ufffd" * allowance.get("\ufffd", 0),
            output_page["text"],
            page_number,
        )
        if glyph_warnings:
            pages_checked.add(page_number)
            warnings.extend(glyph_warnings)


def _group_pdf_superscript_expectations(
    superscript_expectations: list[dict],
) -> dict[tuple[int, str], list[dict]]:
    """Group semantic superscript expectations by page and marker."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for expectation in superscript_expectations:
        page_number = int(expectation.get("page", 0))
        for marker in expectation.get("markers") or []:
            marker_text = str(marker.get("text", "")).strip()
            if _pdf_is_semantic_superscript_marker(marker_text):
                grouped.setdefault(
                    (page_number, marker_text),
                    [],
                ).append(marker)
    return grouped


def _pdf_rendered_superscript_scales(
    page_data: dict,
    pattern,
    superscript_flag: int,
) -> tuple[int, list[float]]:
    """Count rendered markers and measure their scale against nearby prose."""
    reference_spans = [
        span
        for block in page_data["dict"].get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if not (int(span.get("flags", 0)) & superscript_flag)
        and _pdf_span_has_lexical_text(span)
    ]
    rendered_scales: list[float] = []
    rendered_count = 0
    for block in page_data["dict"].get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                occurrences = len(
                    list(pattern.finditer(span.get("text", "")))
                )
                if not occurrences:
                    continue
                native_rendered = bool(
                    int(span.get("flags", 0)) & superscript_flag
                )
                reference_span = _pdf_nearby_superscript_reference(
                    span,
                    reference_spans,
                )
                reference_size = (
                    float(reference_span.get("size", 0.0))
                    if reference_span
                    else 0.0
                )
                rendered_scale = (
                    float(span.get("size", 0.0)) / reference_size
                    if reference_size > 0
                    else 0.0
                )
                inferred_rendered = (
                    not native_rendered
                    and reference_span is not None
                    and 0.45 <= rendered_scale <= 0.80
                )
                if not native_rendered and not inferred_rendered:
                    continue
                rendered_count += occurrences
                if reference_size > 0:
                    rendered_scales.extend(
                        [rendered_scale] * occurrences
                    )
    return rendered_count, rendered_scales


def _audit_pdf_superscripts(
    doc,
    source_doc,
    superscript_expectations: list[dict],
    page_cache: dict[int, dict],
    source_page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> tuple[int, int, list[dict]]:
    """Verify semantic superscript presence and relative scale."""
    grouped = _group_pdf_superscript_expectations(
        superscript_expectations
    )
    expected_total = sum(
        len(marker_specs) for marker_specs in grouped.values()
    )
    rendered_total = 0
    exact_source_matches: list[dict] = []
    superscript_flag = int(
        getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1)
    )

    for (page_number, marker_text), marker_specs in sorted(
        grouped.items()
    ):
        expected_count = len(marker_specs)
        if page_number < 1 or page_number > len(doc):
            warnings.append({
                "type": "missing-page",
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
            })
            continue
        pages_checked.add(page_number)
        page_data = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=True,
        )
        pattern = _pdf_marker_pattern(marker_text)
        plain_count = len(list(pattern.finditer(page_data["text"])))
        rendered_count, rendered_scales = (
            _pdf_rendered_superscript_scales(
                page_data,
                pattern,
                superscript_flag,
            )
        )
        rendered_total += min(rendered_count, expected_count)
        if rendered_count < expected_count:
            warnings.append({
                "type": (
                    "flattened-superscript"
                    if plain_count >= expected_count
                    else "missing-superscript"
                ),
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
                "rendered_superscript": rendered_count,
                "plain_occurrences": plain_count,
            })
            continue

        expected_scales = [
            float(marker.get("scale", 0.60))
            for marker in marker_specs
        ]
        if len(rendered_scales) < expected_count:
            source_match_count = 0
            if source_doc is not None and page_number <= len(source_doc):
                source_data = _pdf_audit_page_data(
                    source_doc,
                    page_number,
                    source_page_cache,
                    include_dict=True,
                )
                source_match_count = (
                    _pdf_exact_source_native_superscript_matches(
                        source_data["dict"],
                        page_data["dict"],
                        pattern,
                    )
                )
            if source_match_count >= expected_count:
                exact_source_matches.append({
                    "page": page_number,
                    "marker": marker_text,
                    "expected": expected_count,
                    "matched": source_match_count,
                })
                continue
            warnings.append({
                "type": "superscript-scale-unverifiable",
                "page": page_number,
                "marker": marker_text,
                "expected": expected_count,
                "measured": len(rendered_scales),
            })
            continue

        for rendered_scale, expected_scale in zip(
            sorted(rendered_scales),
            sorted(expected_scales),
        ):
            if abs(rendered_scale - expected_scale) > 0.14:
                warnings.append({
                    "type": "superscript-size-mismatch",
                    "page": page_number,
                    "marker": marker_text,
                    "expected_scale": round(expected_scale, 3),
                    "rendered_scale": round(rendered_scale, 3),
                })
                break
    return expected_total, rendered_total, exact_source_matches


def _audit_pdf_formula_regions(
    doc,
    source_doc,
    formula_expectations: list[dict],
    page_cache: dict[int, dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> tuple[int, list[dict]]:
    """Verify immutable formula signatures, tolerating tiny raster drift."""
    verified_total = 0
    tolerated_raster_differences: list[dict] = []
    compared_fields = (
        "text_sha256",
        "font_sha256",
        "geometry_sha256",
        "raster_sha256",
        "span_count",
        "raster_width",
        "raster_height",
    )
    for expectation in formula_expectations:
        page_number = int(expectation.get("page", 0))
        element_number = int(expectation.get("element", -1))
        expected_signature = expectation.get("signature") or {}
        if page_number < 1 or page_number > len(doc):
            warnings.append({
                "type": "missing-formula-page",
                "page": page_number,
                "element": element_number,
            })
            continue
        if not expected_signature.get("signature_sha256"):
            warnings.append({
                "type": "source-formula-signature-missing",
                "page": page_number,
                "element": element_number,
            })
            continue

        pages_checked.add(page_number)
        page = doc[page_number - 1]
        page_data = _pdf_audit_page_data(
            doc,
            page_number,
            page_cache,
            include_dict=True,
        )
        try:
            observed_signature = _pdf_formula_region_signature(
                page,
                expectation.get("bbox", (0, 0, 0, 0)),
                page_dict=page_data["dict"],
            )
        except Exception as formula_exc:
            warnings.append({
                "type": "formula-signature-error",
                "page": page_number,
                "element": element_number,
                "error_type": type(formula_exc).__name__,
            })
            continue

        mismatches = [
            field
            for field in compared_fields
            if expected_signature.get(field)
            != observed_signature.get(field)
        ]
        if mismatches == ["raster_sha256"] and source_doc is not None:
            rect = fitz.Rect(
                expectation.get("bbox", (0, 0, 0, 0))
            )
            source_page = source_doc[page_number - 1]
            # ``bbox`` is extraction geometry.  Clipping a rotated page with it
            # unmapped yields two degenerate pixmaps that compare equal, so a
            # genuinely overpainted formula would be tolerated as drift.
            source_clip = _pdf_extraction_rect_to_display(source_page, rect)
            observed_clip = _pdf_extraction_rect_to_display(page, rect)
            if source_clip.is_empty or observed_clip.is_empty:
                warnings.append({
                    "type": "formula-raster-clip-degenerate",
                    "page": page_number,
                    "element": element_number,
                })
                continue
            source_pix = source_page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                colorspace=fitz.csGRAY,
                alpha=False,
                clip=source_clip,
            )
            observed_pix = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                colorspace=fitz.csGRAY,
                alpha=False,
                clip=observed_clip,
            )
            raster_summary = _pdf_formula_raster_difference_summary(
                source_pix,
                observed_pix,
            )
            if raster_summary.get("tolerable"):
                tolerated_raster_differences.append({
                    "page": page_number,
                    "element": element_number,
                    **raster_summary,
                })
                verified_total += 1
                continue
        if mismatches:
            warnings.append({
                "type": "formula-integrity-mismatch",
                "page": page_number,
                "element": element_number,
                "mismatches": mismatches,
            })
            continue
        verified_total += 1
    return verified_total, tolerated_raster_differences


def _audit_pdf_vector_ocr_expectations(
    doc,
    vector_ocr_expectations: list[dict],
    warnings: list[dict],
    pages_checked: set[int],
) -> int:
    """Verify that vector-OCR source phrases no longer remain."""
    expectations_by_page: dict[int, list[dict]] = {}
    for expectation in vector_ocr_expectations:
        expectations_by_page.setdefault(
            int(expectation.get("page", 0)),
            [],
        ).append(expectation)

    verified_elements = 0
    for page_number, page_expectations in sorted(
        expectations_by_page.items()
    ):
        if page_number < 1 or page_number > len(doc):
            for expectation in page_expectations:
                warnings.append({
                    "type": "missing-vector-ocr-page",
                    "page": page_number,
                    "element": int(expectation.get("element", -1)),
                })
            continue
        pages_checked.add(page_number)
        try:
            observed = _pdf_vector_ocr_normalized_match_text(
                _pdf_vector_ocr_page_text(doc[page_number - 1])
            )
        except Exception as vector_audit_exc:
            warnings.append({
                "type": "vector-ocr-audit-error",
                "page": page_number,
                "error_type": type(vector_audit_exc).__name__,
            })
            continue
        for expectation in page_expectations:
            remaining = [
                phrase
                for phrase in expectation.get("phrases") or []
                if _pdf_vector_ocr_normalized_match_text(phrase)
                in observed
            ]
            if remaining:
                warnings.append({
                    "type": "vector-ocr-source-text-remains",
                    "page": page_number,
                    "element": int(expectation.get("element", -1)),
                    "phrases": remaining,
                })
            else:
                verified_elements += 1
    return verified_elements


def _check_pdf_output_structure(
    out_path: str,
    superscript_expectations: list[dict],
    merge_decisions: list[dict] | None = None,
    formula_expectations: list[dict] | None = None,
    vector_ocr_expectations: list[dict] | None = None,
    source_path: str | None = None,
    source_password: str = "",
    page_extractions: dict[int, PageExtraction] | None = None,
) -> dict:
    """Check superscripts, accepted merges, and immutable formula regions."""
    warnings = _validate_pdf_merge_audit(
        merge_decisions or [],
        page_extractions,
    )
    pages_checked: set[int] = set()
    page_cache: dict[int, dict] = {}
    source_page_cache: dict[int, dict] = {}
    vector_expectations = vector_ocr_expectations or []
    doc = fitz.open(out_path)
    source_doc = None
    try:
        source_doc = _open_pdf_audit_source(
            source_path,
            source_password,
        )
        _audit_pdf_source_structure(
            doc,
            source_doc,
            _pdf_vector_source_glyph_allowance(vector_expectations),
            page_cache,
            source_page_cache,
            warnings,
            pages_checked,
        )
        (
            expected_superscripts,
            rendered_superscripts,
            exact_source_matches,
        ) = _audit_pdf_superscripts(
            doc,
            source_doc,
            superscript_expectations or [],
            page_cache,
            source_page_cache,
            warnings,
            pages_checked,
        )
        verified_formula_regions, tolerated_raster_differences = (
            _audit_pdf_formula_regions(
                doc,
                source_doc,
                formula_expectations or [],
                page_cache,
                warnings,
                pages_checked,
            )
        )
        verified_vector_elements = (
            _audit_pdf_vector_ocr_expectations(
                doc,
                vector_expectations,
                warnings,
                pages_checked,
            )
        )
    finally:
        doc.close()
        if source_doc is not None:
            source_doc.close()

    return {
        "status": "ok" if not warnings else "warning",
        "warning_count": len(warnings),
        "warnings": warnings,
        "pages_checked": sorted(pages_checked),
        "expected_superscripts": expected_superscripts,
        "rendered_superscripts": rendered_superscripts,
        "expected_formula_regions": len(formula_expectations or []),
        "verified_formula_regions": verified_formula_regions,
        "expected_vector_ocr_elements": len(vector_expectations),
        "verified_vector_ocr_elements": verified_vector_elements,
        "tolerated_formula_raster_differences": tolerated_raster_differences,
        "exact_source_superscript_matches": exact_source_matches,
        "accepted_merges": sum(
            1
            for decision in (merge_decisions or [])
            if decision.get("decision") == "accepted"
        ),
    }


def _check_pdf_output_structure_serialized(
    out_path: str,
    superscript_expectations: list[dict],
    merge_decisions: list[dict] | None = None,
    formula_expectations: list[dict] | None = None,
    vector_ocr_expectations: list[dict] | None = None,
    source_path: str | None = None,
    source_password: str = "",
    page_extractions: dict[int, PageExtraction] | None = None,
    *,
    dependencies: PDFAuditDependencies,
) -> dict:
    """Run the final fitz audit outside every concurrent extraction pass.

    PyMuPDF table/text extraction has process-global native state.  A final
    audit racing another task's extraction can therefore report hundreds of
    formula mismatches that disappear on an immediate isolated rerun.  Share
    the extraction slot and confirm any warning once before invalidating page
    caches or spending API budget on a structural rebuild.
    """
    log.info(
        "Waiting for serialized PDF structure audit "
        f"(extraction_concurrency={dependencies.extraction_max_concurrency})"
    )
    dependencies.extraction_semaphore.acquire()
    try:
        first = dependencies.check_output_structure(
            out_path,
            superscript_expectations,
            merge_decisions,
            formula_expectations,
            vector_ocr_expectations,
            source_path=source_path,
            source_password=source_password,
            page_extractions=page_extractions,
        )
        if not first.get("warning_count"):
            return first

        log.warning(
            "PDF structure audit returned %s warning(s); confirming once "
            "while the fitz slot remains isolated",
            first.get("warning_count", 0),
        )
        dependencies.trim_process_memory()
        confirmed = dependencies.check_output_structure(
            out_path,
            superscript_expectations,
            merge_decisions,
            formula_expectations,
            vector_ocr_expectations,
            source_path=source_path,
            source_password=source_password,
            page_extractions=page_extractions,
        )
        confirmed["initial_warning_count"] = int(first.get("warning_count", 0))
        confirmed["initial_warning_types"] = sorted({
            str(warning.get("type", "unknown"))
            for warning in first.get("warnings", [])
        })
        confirmed["initial_warning_pages"] = sorted({
            int(warning.get("page"))
            for warning in first.get("warnings", [])
            if warning.get("page") is not None
        })
        if not confirmed.get("warning_count"):
            confirmed["transient_initial_warnings_discarded"] = True
            log.warning(
                "Discarded %s transient PDF structure warning(s) after an "
                "isolated confirmation pass returned clean",
                first.get("warning_count", 0),
            )
        return confirmed
    finally:
        dependencies.extraction_semaphore.release()


def _require_clean_pdf_structure_check(check: dict) -> None:
    status = str((check or {}).get("status", "error"))
    warnings = (check or {}).get("warnings") or []
    try:
        warning_count = int((check or {}).get("warning_count", len(warnings)))
    except (TypeError, ValueError):
        warning_count = max(1, len(warnings))
    if status == "ok" and warning_count == 0 and not warnings:
        return

    raise PDFStructureValidationError(check)


def _save_and_validate_pdf_audit(task_id: str, audit: PdfAudit) -> str:
    path = _save_pdf_audit(task_id, audit)
    check = audit.get("structure_check") or {}
    log_method = log.info if check.get("warning_count", 0) == 0 else log.warning
    log_method(
        f"[{task_id}] PDF structure audit saved to {path} "
        f"({check.get('warning_count', 0)} warning(s))"
    )
    _require_clean_pdf_structure_check(check)
    return path
