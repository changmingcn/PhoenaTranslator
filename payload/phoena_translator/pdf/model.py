"""Vocabulary of the PDF element model.

Elements stay plain dicts at runtime: they are JSON-serialised into the
layout and page caches and hashed into cache identities.  The TypedDicts
below document that schema (every key is optional because passes add keys
as they go) and the StrEnums name the values that used to be bare strings.
A StrEnum member compares, hashes and serialises exactly like its string,
so cached data and identities are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict


class ElementType(StrEnum):
    TEXT = "text"
    # A text element absorbed by a cross-page merge; kept so indices stay stable.
    MERGED_AWAY = "text_merged_away"
    FORMULA_IMAGE = "formula_image"
    TABLE_IMAGE = "table_image"
    IMAGE = "image"


class LayoutClass(StrEnum):
    BODY = "body"
    HEADING = "heading"
    FOOTER = "footer"
    TABLE = "table"
    SCATTERED = "scattered"


class SkipReason(StrEnum):
    """Why a text element keeps its source text instead of being translated."""

    WATERMARK = "watermark"
    EMBEDDED_THUMBNAIL_TEXT = "embedded_thumbnail_text"
    TABLE_PRESERVED = "table_preserved"
    FORMULA_RISK_PRESERVED = "formula_risk_preserved"
    FORMULA_GEOMETRY_OVERLAP = "formula_geometry_overlap"
    TRANSLATION_INTEGRITY_FALLBACK = "translation_integrity_fallback"
    RENDER_SOURCE_RESTORE = "render_source_restore"
    RENDER_NO_SAFE_RECT_FALLBACK = "render_no_safe_rect_fallback"


Rect = list[float]  # [x0, y0, x1, y1] in page points

# Font size assumed for an element or line that carries none.
DEFAULT_FONTSIZE = 11.0

# A paragraph's rendered indents are clamped to shares of its width:
# margin_left <= 0.35 w and -0.18 w <= text_indent <= 0.25 w.
PARAGRAPH_MAX_MARGIN_SHARE = 0.35
PARAGRAPH_MAX_HANGING_SHARE = 0.18
PARAGRAPH_MAX_FIRST_LINE_SHARE = 0.25


def indent_step_threshold(fontsize: float) -> float:
    """Smallest x offset that counts as a deliberate first-line or hanging indent."""
    return max(fontsize * 0.50, 3.0)


class SuperscriptRun(TypedDict, total=False):
    text: str
    source: str
    scale: float


class InlineMathFragment(TypedDict, total=False):
    text: str
    font_kind: str


class SourceLine(TypedDict, total=False):
    bbox: Rect
    plain: str
    rich: str
    same_baseline_fragments: list[dict]


class Paragraph(TypedDict, total=False):
    plain: str
    rich: str
    source_bbox: Rect
    source_line_bboxes: list[Rect]
    source_lines: list[SourceLine]
    margin_left: float
    text_indent: float
    gap_before: float
    text_align: str
    nowrap: bool | None
    first_line_indent: bool
    toc_leader: dict | None
    same_baseline_paragraph_start: bool


class TextElement(TypedDict, total=False):
    type: ElementType
    # Geometry: ``bbox`` and ``rect`` hold the same box; ``x``/``y`` its origin.
    bbox: Rect
    rect: Rect
    x: float
    y: float
    render_bbox: Rect
    source_line_bboxes: list[Rect]
    rotation: int
    non_horizontal: bool
    # Content and style.
    content: str
    rich_content: str | None
    paragraphs: list[Paragraph]
    fontsize: float
    line_height: float
    top_padding: float
    bold: bool
    color: int
    superscript_runs: list[SuperscriptRun]
    superscript_scale: float | None
    inline_math_fragments: list[InlineMathFragment]
    # Classification.
    layout_class: LayoutClass
    skip_translate_reason: SkipReason
    single_line_heading: bool
    table_hint: bool
    preserve_source_style: bool
    semantic_table_cell: bool
    semantic_native_table_cell: bool
    glossary_cell_hint: bool
    glossary_term_hint: bool
    glossary_definition_hint: bool
    entity_directory_hint: bool
    reference_entry_hint: bool
    formula_adjacent_text: bool
    footnote_hint: bool
    watermark: bool
    vector_ocr: bool
    # Provenance counters written by the merge passes.
    merged_visual_line_count: int
    strong_semantic_continuation_merged: int
    reference_entry_merged: int


class FormulaImageElement(TypedDict, total=False):
    type: ElementType
    bbox: Rect
    rect: Rect
    x: float
    y: float
    math_line_count: int
    math_candidate_symbol_count: int
    math_mixed: bool
    math_reasons: list[str]
    math_signature: dict[str, Any]


class TableImageElement(TypedDict, total=False):
    type: ElementType
    bbox: Rect
    rect: Rect
    y: float


PdfElement = TextElement | FormulaImageElement | TableImageElement


class PageExtraction(TypedDict, total=False):
    """One page of ``page_extractions``; the present key says what it holds."""

    elements: list[PdfElement]
    layout_styles: dict[str, Any]
    cached: dict[str, str]
    appendix: bool
    source_page_fallback: dict[str, Any]


class PdfAudit(TypedDict, total=False):
    """The per-task audit document written to ``<task>_audit.json``."""

    # Created by pipeline._new_pdf_audit.
    schema_version: int
    task_id: str
    created_at: float
    minimum_acceptable_render_scale: float
    total_pages: int
    # Extraction: expectations the output audit verifies.
    merge_decisions: list[dict[str, Any]]
    superscript_expectations: list[dict[str, Any]]
    formula_expectations: list[dict[str, Any]]
    vector_ocr_expectations: list[dict[str, Any]]
    formula_protection: dict[str, Any]
    formula_protection_before_source_fallback: dict[str, Any]
    formula_protection_source_fallback_rounds: int
    formula_risk_preserved_elements: list[dict[str, Any]]
    formula_mid_sentence_preserved_elements: list[dict[str, Any]]
    table_preserved_elements: list[dict[str, Any]]
    table_preservation_rollbacks: list[dict[str, Any]]
    phase1_checkpointed_at: float
    # Translation and assembly.
    source_page_fallbacks: list[dict[str, Any]]
    element_source_fallbacks: list[dict[str, Any]]
    inline_render_source_page_fallbacks: list[dict[str, Any]]
    untranslated_english_leaks: list[dict[str, Any]]
    untranslated_delivered_pages: list[int]
    htmlbox_placement_events: list[dict[str, Any]]
    render_scale_events: list[dict[str, Any]]
    rejected_render_scale_events: list[dict[str, Any]]
    textbox_fallback_events: list[dict[str, Any]]
    # Recovery.
    auto_repair: dict[str, Any]
    auto_repair_cleanup_error: dict[str, Any]
    source_page_fallback_reassembly: dict[str, Any]
    recovery_cleanup_warnings: list[dict[str, Any]]
    failure_type: str
    failed_at: float
    # Completion.
    structure_check: dict[str, Any]
    quality_status: str
    completed_at: float


@dataclass(frozen=True)
class CrossPageMergeDecision:
    """An accepted cross-page sentence merge, as recorded in the audit."""

    KIND = "cross-page-orphan-tail"

    source_page: int
    source_element: int
    destination_element: int
    source_continuation_reason: str
    carried_tail: str
    source_layout: str | None
    destination_layout: str | None
    tail_unterminated: bool = False

    def to_audit(self) -> dict[str, Any]:
        """Serialise in the audit's historical key order (pages are 1-based)."""
        entry: dict[str, Any] = {
            "kind": self.KIND,
            "source_page": self.source_page + 1,
            "destination_page": self.source_page + 2,
            "source_element": self.source_element,
            "destination_element": self.destination_element,
            "decision": "accepted",
            "reason": "accepted",
            "source_continuation_reason": self.source_continuation_reason,
            "carried_tail": self.carried_tail,
            "source_layout": self.source_layout,
            "destination_layout": self.destination_layout,
        }
        if self.tail_unterminated:
            entry["tail_unterminated"] = True
        return entry


__all__ = [
    "CrossPageMergeDecision",
    "DEFAULT_FONTSIZE",
    "PARAGRAPH_MAX_FIRST_LINE_SHARE",
    "PARAGRAPH_MAX_HANGING_SHARE",
    "PARAGRAPH_MAX_MARGIN_SHARE",
    "ElementType",
    "FormulaImageElement",
    "InlineMathFragment",
    "LayoutClass",
    "PageExtraction",
    "Paragraph",
    "PdfAudit",
    "PdfElement",
    "SkipReason",
    "SourceLine",
    "SuperscriptRun",
    "TableImageElement",
    "TextElement",
    "indent_step_threshold",
]
