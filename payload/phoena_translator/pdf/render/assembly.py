"""Position-preserving PDF page assembly stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import fitz

from phoena_translator.pdf.errors import PDFPageRenderError
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_source_ink_rects,
    _pdf_page_extraction_rect,
    _subtract_pdf_protected_rects,
)
from phoena_translator.pdf.model import (
    ElementType,
    PageExtraction,
    PdfAudit,
    SkipReason,
    DEFAULT_FONTSIZE,
)
from phoena_translator.pdf.render.figure_table_groups import (
    ELEMENT_RESTORE as FIGURE_TABLE_ELEMENT_RESTORE,
    PAGE_FAILURE as FIGURE_TABLE_PAGE_FAILURE,
    RULES as FIGURE_TABLE_RULES,
    FigureTableGroup,
    find_figure_table_group,
)
from phoena_translator.pdf.patterns import PDF_FORMULA_GUARD_PADDING
from phoena_translator.pdf.render.page_styles import _derive_pdf_page_layout_styles
from phoena_translator.pdf.render.paragraph_html import (
    PDF_HTMLBOX_SUPERSCRIPT_CSS,
    _build_paragraph_html,
    _pdf_plain_drawn_text,
    _split_pdf_translated_paragraphs,
)
from phoena_translator.pdf.render.primitives import (
    _build_htmlbox_rect_ladder,
    _build_pdf_rotated_text_rects,
    _collect_pdf_overlap_redraw_indices,
    _constrain_pdf_htmlbox_rect_ladder_to_neighbors,
    _insert_pdf_htmlbox_at_readable_scale,
    _insert_pdf_rotated_textbox_with_fit,
    _insert_pdf_textbox_with_fit,
    _paint_pdf_vector_ocr_element_background,
    _resolve_pdf_render_style,
    _restore_pdf_source_text_element,
)
from phoena_translator.pdf.render.source_fallbacks import (
    _append_pdf_source_page_fallback,
    _clear_pdf_figure_table_skip_reasons,
    _drop_pdf_figure_table_audit_records,
    _exclude_pdf_audit_expectations_for_element,
    _exclude_pdf_audit_expectations_for_source_pages,
    _pdf_source_page_fallback_entry,
    _record_pdf_element_source_fallback,
    _replace_pdf_pages_with_source,
)
from phoena_translator.pdf.semantics.cross_page import _cross_page_dominant_fontsize
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.translate.deterministic_labels import (
    _translate_pdf_fixed_short_label,
)
from phoena_translator.pdf.translate.numeric_guard import (
    PDF_LABEL_NUMBERS_REASON,
    PDF_NUMERIC_ROW_REASON,
    pdf_numeric_guard_failure,
)
from phoena_translator.pdf.translate.placeholders import (
    PDF_PLACEHOLDER_LEAK_REASON,
    pdf_internal_placeholder_leaks,
)
from phoena_translator.pdf.translate.inline_markup import (
    _normalize_pdf_translation,
    _pdf_superscript_signature,
)
from phoena_translator.pdf.verify.leaks import _pdf_collect_untranslated_english_leaks
from phoena_translator.pdf.verify.merges import (
    _expand_pdf_fallback_pages_for_accepted_merges,
)


@dataclass
class PDFAssemblyContext:
    """Mutable document and immutable rendering settings for page assembly."""

    task_id: str
    total_pages: int
    pdf_password: str
    use_htmlbox: bool
    fail_open_to_source_page: bool
    minimum_htmlbox_scale: float
    assembly_work_path: str | None
    src_path: str
    out_doc: Any
    page_extractions: dict[int, PageExtraction]
    pdf_audit: PdfAudit
    font_path: str
    bold_font_path: str
    has_distinct_bold_font: bool
    font_archive: Any
    font_basename: str
    logger: logging.Logger
    filter_formula_safe_rects: Callable[..., list[Any]]
    trim_process_memory: Callable[[], None]
    # Per-assembly values the figure/table narrowing computes once.
    figure_table_cache: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PDFPageAssemblyPlan:
    """Prepared page state shared by redaction and element rendering."""

    page_num: int
    out_page: fitz.Page
    elements: list[dict]
    translations: dict[int, str]
    layout_styles: dict
    watermark_indices: set[int]
    protected_formula_rects: list[fitz.Rect]
    translated_indices: set[int]
    redraw_source_indices: set[int]
    render_indices: set[int]
    # Figures/tables this attempt keeps in source ink, and failing elements
    # whose group was refused (see ``_assemble_page_with_figure_table_fallbacks``).
    source_groups: tuple[FigureTableGroup, ...] = ()
    refused_seeds: set[int] = field(default_factory=set)
    # Single elements an earlier attempt could not render and no figure or
    # table took: kept in exact source ink, their ink protected from the
    # neighbours' redaction (``_keep_failed_element_source_ink``).
    source_elements: frozenset[int] = frozenset()


@dataclass(frozen=True)
class PDFElementRenderPlan:
    """Geometry, text, and style prepared for one renderable element."""

    index: int
    element: dict
    source_paragraphs: list[dict]
    source_rect: fitz.Rect
    render_rect: fitz.Rect
    fontsize: float
    color: tuple[float, float, float]
    text: str
    line_height: float
    table_cell: bool
    box_html: str
    has_semantic_superscript: bool


@dataclass(frozen=True)
class PDFHTMLBoxResult:
    inserted: bool
    error: Exception
    render_rect: fitz.Rect
    textbox_fit_rects: list[fitz.Rect] | None


def _accepted_merge_fallback_pages_for_element(
    context: PDFAssemblyContext,
    page_num: int,
    element_index: int,
) -> list[int]:
    """Return the source-page closure for an accepted merge endpoint."""

    page_number = page_num + 1
    merge_decisions = context.pdf_audit.get("merge_decisions") or []
    is_endpoint = False
    for decision in merge_decisions:
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
                is_endpoint = True
                break
        if is_endpoint:
            break
    if not is_endpoint:
        return []
    return [
        page
        for page in _expand_pdf_fallback_pages_for_accepted_merges(
            {page_number},
            merge_decisions,
        )
        if 1 <= page <= context.total_pages
    ]


def _cross_page_merge_fallback_error(
    context: PDFAssemblyContext,
    page_num: int,
    element_index: int,
    reason: str,
) -> PDFPageRenderError | None:
    """Build a whole-merge source fallback instead of restoring one endpoint."""

    fallback_pages = _accepted_merge_fallback_pages_for_element(
        context,
        page_num,
        element_index,
    )
    if not fallback_pages:
        return None
    error = PDFPageRenderError(
        page_num + 1,
        element_index,
        f"{reason}; accepted cross-page merge requires source-page closure",
    )
    error.fallback_pages = fallback_pages
    error.repair_pages = fallback_pages
    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1} elem {element_index}: "
        f"render fallback touches an accepted cross-page merge endpoint; "
        f"preserving source-page closure {fallback_pages}"
    )
    return error


def _try_restore_source_element(
    context: PDFAssemblyContext,
    out_page: fitz.Page,
    page_num: int,
    protected_formula_rects: list[fitz.Rect],
    element_index: int,
    element: dict,
    restore_rect: fitz.Rect,
    restore_fontsize: float,
    restore_color: tuple[float, float, float],
    restore_line_height: float,
    failure_reason: str,
) -> bool:
    """Restore one source element after its translated rendering fails."""

    merge_fallback_error = _cross_page_merge_fallback_error(
        context,
        page_num,
        element_index,
        failure_reason,
    )
    if merge_fallback_error is not None:
        raise merge_fallback_error

    restored, restore_message = _restore_pdf_source_text_element(
        out_page,
        element,
        restore_rect,
        color=restore_color,
        fontfile=context.font_path,
        fontsize=restore_fontsize,
        lineheight_points=restore_line_height,
        protected_rects=protected_formula_rects,
    )
    if not restored:
        context.logger.warning(
            f"[{context.task_id}] Page {page_num + 1} elem {element_index}: "
            f"source restore also failed ({restore_message})"
        )
        return False
    element["skip_translate_reason"] = SkipReason.RENDER_SOURCE_RESTORE
    _record_pdf_element_source_fallback(
        context.pdf_audit,
        page_num + 1,
        element_index,
        "render",
        f"{failure_reason}; {restore_message}",
        source_text=element.get("content", ""),
    )
    _exclude_pdf_audit_expectations_for_element(
        context.pdf_audit,
        page_num + 1,
        element_index,
    )
    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1} elem {element_index}: "
        f"{failure_reason}; restored exact source text in place and translated "
        "the rest of the page"
    )
    return True


@dataclass(frozen=True)
class _PageLayout:
    """Page geometry the preflight needs before the assembly plan exists."""

    page_num: int
    out_page: fitz.Page
    elements: list[dict]
    layout_styles: dict
    watermark_indices: set[int]
    protected_formula_rects: list[fitz.Rect]


def _prepare_page(
    context: PDFAssemblyContext,
    page_num: int,
    source_groups: tuple[FigureTableGroup, ...] = (),
    refused_seeds: set[int] | None = None,
    source_elements: dict[int, str] | None = None,
) -> PDFPageAssemblyPlan | None:
    """Resolve translations and validate formula-safe placement before redaction.

    ``source_groups`` are figures/tables an earlier attempt at this page could
    not typeset; their text keeps its exact source ink.  ``refused_seeds`` is
    the page's set of failing elements whose group was refused; decisions
    made in this attempt add to it.  ``source_elements`` maps each single
    element an earlier attempt could not render (and no group took) to the
    failure: it keeps its exact source ink, and its ink is protected like a
    formula region so that no neighbour's redaction, re-typeset or drawn
    translation touches it.
    """

    if refused_seeds is None:
        refused_seeds = set()
    if source_elements is None:
        source_elements = {}
    out_page = context.out_doc[page_num]
    info = context.page_extractions.get(page_num, {})
    page_trans = _page_translations(info)
    if page_trans is None:
        return None
    elements = info.get("elements", [])
    merged_away_indices = {
        i for i, elem in enumerate(elements) if elem.get("type") == ElementType.MERGED_AWAY
    }
    leaked = _placeholder_leak_indices(elements, page_trans, source_elements, source_groups)
    if leaked:
        # Safety net for a translation that reached assembly with a raw
        # internal token (a page cache written before translation validation
        # rejected it): never draw it, keep the element's exact source ink.
        source_elements = dict(source_elements)
        for index in leaked:
            merge_fallback_error = _cross_page_merge_fallback_error(
                context, page_num, index, PDF_PLACEHOLDER_LEAK_RENDER_FAILURE
            )
            if merge_fallback_error is not None:
                raise merge_fallback_error
            source_elements[index] = PDF_PLACEHOLDER_LEAK_RENDER_FAILURE
    numeric_flaws = _numeric_flaw_indices(elements, page_trans, source_elements, source_groups)
    if numeric_flaws:
        # Safety net for a cached translation that lost or moved a numeric
        # row's numbers, or put numbers into a digit-free label: never draw
        # it, keep the element's exact source ink.
        source_elements = dict(source_elements)
        merge_endpoints = _assembly_accepted_merge_endpoints(context)
        for index, failure in numeric_flaws.items():
            if (page_num + 1, index) in merge_endpoints:
                merge_fallback_error = _cross_page_merge_fallback_error(
                    context, page_num, index, failure
                )
                if merge_fallback_error is not None:
                    raise merge_fallback_error
            source_elements[index] = failure

    layout = _PageLayout(
        page_num=page_num,
        out_page=out_page,
        elements=elements,
        layout_styles=info.get("layout_styles") or _derive_pdf_page_layout_styles(
            elements, _pdf_page_extraction_rect(out_page)
        ),
        watermark_indices={
            i
            for i, elem in enumerate(elements)
            if elem.get("skip_translate_reason") == SkipReason.WATERMARK
        },
        protected_formula_rects=[
            _get_pdf_elem_rect(elem)
            for elem in elements
            if elem.get("type") == ElementType.FORMULA_IMAGE
        ]
        + [
            rect
            for index in sorted(source_elements)
            if 0 <= index < len(elements)
            for rect in _kept_source_ink_rects(elements[index])
        ],
    )
    translated_indices = {
        i
        for i, elem in enumerate(elements)
        if i not in layout.watermark_indices
        and elem["type"] == ElementType.TEXT
        and i in page_trans
        and page_trans[i] != elem.get("content", "")
    }
    candidates = set(translated_indices)  # before any retention
    for group in source_groups:
        _keep_figure_table_source_ink(context, layout, group, page_trans, translated_indices)
    for index in sorted(source_elements):
        _keep_failed_element_source_ink(
            context, layout, index, source_elements[index], page_trans, translated_indices
        )
    kept_in_source = set().union(
        *(group.members | group.extra_members for group in source_groups)
    )

    # Validate formula-safe placement before redacting any source text.
    # Previously this check happened only during insertion, after the
    # original glyphs had already been removed; a no-safe-rectangle
    # result could therefore create a silent blank. Nothing has been
    # redacted yet, so an unplaceable element can simply keep its
    # original source ink while the rest of the page still translates.
    unplaceable = sorted(_unplaceable_indices(context, layout, page_trans, translated_indices))
    if candidates and not translated_indices.difference(unplaceable):
        # Every translation candidate of the page is withdrawn, whatever
        # withdrew it (a retained figure/table group, single elements kept in
        # source, no formula-safe place, or several together): the page is its
        # source page, delivered and recorded as such, not a "translated"
        # page.  An unplaceable merge endpoint keeps its merge closure.  No
        # preflight record is written; the group and element records of this
        # attempt go with the source-page cleanup.
        for i in unplaceable:
            merge_fallback_error = _cross_page_merge_fallback_error(
                context, page_num, i, "no formula-safe rectangle for translated merge endpoint"
            )
            if merge_fallback_error is not None:
                raise merge_fallback_error
        raise _element_source_refusal(
            page_num, min(candidates), "every translation on the page was withdrawn"
        )
    for i in unplaceable:
        _keep_source_ink(context, layout, i, page_trans, translated_indices)
    _audit_untranslated_english(context, page_num, elements, page_trans)

    if not translated_indices and not layout.watermark_indices and not merged_away_indices:
        return None  # nothing to change on this page

    try:
        redraw_source_indices = _overlap_redraw_indices(
            page_num,
            _without_kept_source_elements(elements, source_elements),
            set(translated_indices) | set(layout.watermark_indices) | merged_away_indices,
        )
    except PDFPageRenderError as error:
        _attach_figure_table_decision(
            context,
            error,
            page_num,
            elements,
            groups=tuple(source_groups),
            refused=refused_seeds,
            translated=translated_indices,
            watermark=set(layout.watermark_indices),
            merged_away=merged_away_indices,
        )
        raise
    if kept_in_source & (redraw_source_indices | translated_indices):
        # Invariant: nothing inside a narrowed figure/table is redacted,
        # redrawn or re-typeset.  The group's extra members were chosen for
        # exactly this removal set, so this is a safety net only.
        raise _figure_table_refusal(
            page_num,
            source_groups[-1],
            "a kept element would still be redrawn",
        )
    touched = set(source_elements) & (
        redraw_source_indices
        | translated_indices
        | set(layout.watermark_indices)
        | merged_away_indices
    )
    if touched:
        # Invariant: an element kept in source after a render failure is
        # never redacted, redrawn or re-typeset.  Safety net only.
        raise _element_source_refusal(
            page_num, min(touched), "a kept element would still be redacted or redrawn"
        )
    if redraw_source_indices:
        context.logger.info(
            f"[{context.task_id}] Page {page_num + 1}: redrawing {len(redraw_source_indices)} "
            "unchanged overlapping text element(s) after redaction"
        )
    return PDFPageAssemblyPlan(
        page_num=page_num,
        out_page=out_page,
        elements=elements,
        translations=page_trans,
        layout_styles=layout.layout_styles,
        watermark_indices=layout.watermark_indices,
        protected_formula_rects=layout.protected_formula_rects,
        translated_indices=translated_indices,
        redraw_source_indices=redraw_source_indices,
        render_indices=set(translated_indices) | redraw_source_indices,
        source_groups=tuple(source_groups),
        refused_seeds=refused_seeds,
        source_elements=frozenset(source_elements),
    )


PDF_PLACEHOLDER_LEAK_RENDER_FAILURE = (
    f"{PDF_PLACEHOLDER_LEAK_REASON}: translation still holds a raw internal "
    "placeholder token"
)


def _assembly_net_candidates(
    elements: list[dict],
    page_trans: dict[int, str],
    source_elements: dict[int, str],
    source_groups: tuple[FigureTableGroup, ...] = (),
) -> list[int]:
    """The elements whose translation would be drawn: the only candidates of
    either assembly net (internal tokens, numeric rows).

    An element that will not be drawn is a candidate of neither: one kept in
    source (singly, or as a member or extra member of a retained figure/table
    group), a watermark, merged-away or other non-text element, or one whose
    translation is its source text (not translated, or kept as source at
    translation).  Its existing record and protection stand, so no element
    is recorded twice.
    """
    kept_in_groups = set().union(
        *(group.members | group.extra_members for group in source_groups)
    )
    candidates = []
    for index in sorted(page_trans):
        if index in source_elements or index in kept_in_groups:
            continue
        if not 0 <= index < len(elements):
            continue
        elem = elements[index]
        if elem.get("type") != ElementType.TEXT:
            continue
        if elem.get("skip_translate_reason") == SkipReason.WATERMARK:
            continue
        translation = page_trans[index]
        if not isinstance(translation, str) or translation == elem.get("content", ""):
            continue
        candidates.append(index)
    return candidates


def _placeholder_leak_indices(
    elements: list[dict],
    page_trans: dict[int, str],
    source_elements: dict[int, str],
    source_groups: tuple[FigureTableGroup, ...] = (),
) -> list[int]:
    """Candidates (``_assembly_net_candidates``) whose translation holds an
    internal token."""
    leaked = []
    for index in _assembly_net_candidates(elements, page_trans, source_elements, source_groups):
        elem = elements[index]
        source = elem.get("rich_content") or elem.get("content", "")
        if pdf_internal_placeholder_leaks(source, page_trans[index]):
            leaked.append(index)
    return leaked


# --- numeric row guard: assembly net (begin) ---
def _numeric_flaw_indices(
    elements: list[dict],
    page_trans: dict[int, str],
    source_elements: dict[int, str],
    source_groups: tuple[FigureTableGroup, ...] = (),
) -> dict[int, str]:
    """Candidates (``_assembly_net_candidates``) whose translation fails
    condition N or M.  The guard itself also treats a translation whose plain
    text is the plain source text as the element kept in source."""
    flawed = {}
    for index in _assembly_net_candidates(elements, page_trans, source_elements, source_groups):
        elem = elements[index]
        source = elem.get("rich_content") or elem.get("content", "")
        failure = pdf_numeric_guard_failure(source, page_trans[index])
        if failure:
            flawed[index] = f"{failure}: {PDF_NUMERIC_FLAW_RENDER_FAILURES[failure]}"
    return flawed


PDF_NUMERIC_FLAW_RENDER_FAILURES = {
    PDF_NUMERIC_ROW_REASON: "translation does not hold the numeric row's numbers in order",
    PDF_LABEL_NUMBERS_REASON: "translation puts numbers into a label without any digit",
}


def _assembly_accepted_merge_endpoints(context: PDFAssemblyContext) -> frozenset:
    """(page number, element index) of every accepted merge endpoint, computed
    once per assembly; only these need the merge-closure lookup."""
    cache = getattr(context, "figure_table_cache", None)
    if cache is not None and "accepted_merge_endpoints" in cache:
        return cache["accepted_merge_endpoints"]
    endpoints = set()
    for decision in context.pdf_audit.get("merge_decisions") or []:
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        for page_key, element_key in (
            ("source_page", "source_element"),
            ("destination_page", "destination_element"),
        ):
            try:
                endpoints.add(
                    (int(decision.get(page_key, 0) or 0), int(decision.get(element_key, -1)))
                )
            except (TypeError, ValueError):
                continue
    endpoints = frozenset(endpoints)
    if cache is not None:
        cache["accepted_merge_endpoints"] = endpoints
    return endpoints
# --- numeric row guard: assembly net (end) ---


def _page_translations(info: PageExtraction) -> dict[int, str] | None:
    """The page's translations by element index, or ``None`` to leave it as is."""
    if info.get("source_page_fallback"):
        return None
    if "appendix" in info or ("elements" not in info and "cached" not in info):
        return None
    if "cached" in info:
        return {int(k): v for k, v in info["cached"].items()}
    if any(
        elem.get("type") == ElementType.MERGED_AWAY for elem in info.get("elements", [])
    ):
        # Cross-page semantic merging may consume the only text element on a
        # page.  Such a page still needs an assembly plan so its stale source
        # ink is redacted even though it has no translation payload.
        return {}
    return None


def _formula_safe_htmlbox_ladder(
    context: PDFAssemblyContext,
    page: _PageLayout | PDFPageAssemblyPlan,
    index: int,
    render_rect: fitz.Rect,
    fontsize: float,
) -> list[tuple[str, fitz.Rect]]:
    """Formula-safe HTML-box rectangles for one element, preferred first."""
    element = page.elements[index]
    if element.get("table_hint") or element.get("glossary_cell_hint"):
        # Table rules and neighboring cells remain fixed. Let a translated
        # cell scale inside its original geometry; a downward extension
        # would turn the row into overlapping text while the table lines
        # stay in place.
        ladder = [("base", fitz.Rect(render_rect))]
    else:
        ladder = _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
            _build_htmlbox_rect_ladder(
                render_rect,
                _pdf_page_extraction_rect(page.out_page),
                fontsize,
                prefer_widen=bool(element.get("single_line_heading")),
            ),
            render_rect,
            page.elements,
            index,
            fontsize,
            ignored_indices=page.watermark_indices,
        )
    return context.filter_formula_safe_rects(
        ladder,
        page.protected_formula_rects,
        anchor_rect=render_rect,
    )


def _unplaceable_indices(
    context: PDFAssemblyContext,
    layout: _PageLayout,
    page_trans: dict[int, str],
    translated_indices: set[int],
) -> set[int]:
    """Translated elements that have no formula-safe place to be drawn."""
    unplaceable = set()
    for i in sorted(translated_indices):
        elem = layout.elements[i]
        render_style = _resolve_pdf_render_style(elem, layout.layout_styles)
        render_rect = _get_pdf_elem_rect(elem)
        fontsize = float(render_style.get("fontsize", elem.get("fontsize", DEFAULT_FONTSIZE)))
        if elem.get("non_horizontal"):
            if not _build_pdf_rotated_text_rects(
                render_rect,
                _pdf_page_extraction_rect(layout.out_page),
                fontsize,
                int(elem.get("rotation", 90) or 90),
                layout.protected_formula_rects,
            ):
                unplaceable.add(i)
        elif (
            context.use_htmlbox or _pdf_superscript_signature(page_trans[i])
        ) and not _formula_safe_htmlbox_ladder(context, layout, i, render_rect, fontsize):
            unplaceable.add(i)
    return unplaceable


def _keep_source_ink(
    context: PDFAssemblyContext,
    layout: _PageLayout,
    index: int,
    page_trans: dict[int, str],
    translated_indices: set[int],
) -> None:
    """Leave an unplaceable element's source ink untouched and record why."""
    elem = layout.elements[index]
    page_number = layout.page_num + 1
    merge_fallback_error = _cross_page_merge_fallback_error(
        context,
        layout.page_num,
        index,
        "no formula-safe rectangle for translated merge endpoint",
    )
    if merge_fallback_error is not None:
        raise merge_fallback_error
    elem["skip_translate_reason"] = SkipReason.RENDER_NO_SAFE_RECT_FALLBACK
    translated_indices.discard(index)
    page_trans[index] = elem.get("content", "")
    _record_pdf_element_source_fallback(
        context.pdf_audit,
        page_number,
        index,
        "render-preflight",
        "no formula-safe rectangle; source ink kept in place",
        source_text=elem.get("content", ""),
    )
    _exclude_pdf_audit_expectations_for_element(
        context.pdf_audit,
        page_number,
        index,
    )
    context.logger.warning(
        f"[{context.task_id}] Page {page_number} elem {index}: no formula-safe "
        "rectangle; keeping the element's exact source ink and "
        "translating the rest of the page"
    )


def _keep_figure_table_source_ink(
    context: PDFAssemblyContext,
    layout: _PageLayout,
    group: FigureTableGroup,
    page_trans: dict[int, str],
    translated_indices: set[int],
) -> dict:
    """Leave a whole figure or table untouched in its source ink and record it."""
    page_number = layout.page_num + 1
    reason = (
        f"kept with {group.kind} figure/table of elem {group.trigger_element}: "
        f"{group.reason}"
    )
    narrowed = []
    withdrawn = []
    for index in sorted(group.members | group.extra_members):
        elem = layout.elements[index]
        if index in translated_indices:
            translated_indices.discard(index)
            withdrawn.append(index)
            if index in group.members:
                narrowed.append(index)
        if index in page_trans:
            page_trans[index] = elem.get("content", "")
        if not elem.get("skip_translate_reason"):
            elem["skip_translate_reason"] = SkipReason.RENDER_FIGURE_TABLE_SOURCE
        _exclude_pdf_audit_expectations_for_element(
            context.pdf_audit,
            page_number,
            index,
        )
    for index in withdrawn:
        _record_pdf_element_source_fallback(
            context.pdf_audit,
            page_number,
            index,
            "render-figure-table",
            reason
            if index in group.members
            else (
                f"kept with {group.kind} figure/table of elem {group.trigger_element}: "
                "redacting it would force kept element(s) to be redrawn"
            ),
            source_text=layout.elements[index].get("content", ""),
        )
    entry = {
        "page": page_number,
        "element": group.trigger_element,
        "stage": "render",
        "reason": group.reason,
        "group": group.kind,
        "trigger": group.trigger,
        "elements": sorted(group.members),
        "narrowed_elements": narrowed,
        # Outside elements kept too, so that no member has to be redrawn.
        "extra_kept_elements": sorted(group.extra_members),
        "bbox": list(group.bbox),
    }
    context.pdf_audit.setdefault("figure_table_source_fallbacks", []).append(entry)
    context.logger.warning(
        f"[{context.task_id}] Page {page_number} elem {group.trigger_element}: "
        f"keeping the whole figure/table ({group.kind}, {len(group.members)} "
        f"element(s), {len(narrowed)} translation(s) withdrawn) in its exact "
        "source ink and translating the rest of the page"
    )
    return entry


def _kept_source_ink_rects(elem: dict) -> list[fitz.Rect]:
    """The source glyph rectangles of an element kept in source ink."""
    try:
        rects = [rect for rect in _get_pdf_source_ink_rects(elem) if not rect.is_empty]
    except Exception:
        rects = []
    if not rects:
        rect = _get_pdf_elem_rect(elem)
        rects = [] if rect.is_empty or rect.is_infinite else [rect]
    return rects


def _without_kept_source_elements(
    elements: list[dict],
    source_elements: dict[int, str],
) -> list[dict]:
    """The page's elements as the overlap-redraw search must see them.

    An element kept in source after a render failure has its ink protected
    from every redaction rectangle, so a neighbour's redaction cannot erase
    it: it is never a redraw candidate, and no redraw chains through it.
    """
    if not source_elements:
        return elements
    return [
        dict(elem, type="kept_source") if index in source_elements else elem
        for index, elem in enumerate(elements)
    ]


def _keep_failed_element_source_ink(
    context: PDFAssemblyContext,
    layout: _PageLayout,
    index: int,
    failure: str,
    page_trans: dict[int, str],
    translated_indices: set[int],
) -> dict:
    """Keep one element that failed the page in its exact source ink; record why."""
    elem = layout.elements[index]
    page_number = layout.page_num + 1
    withdrawn = index in translated_indices
    translated_indices.discard(index)
    if withdrawn and index in page_trans:
        page_trans[index] = elem.get("content", "")
    if not elem.get("skip_translate_reason"):
        elem["skip_translate_reason"] = SkipReason.RENDER_ELEMENT_SOURCE
    detail = (
        "translation withdrawn; exact source ink kept in place"
        if withdrawn
        else "exact source ink protected from neighbouring redaction"
    )
    entry = _record_pdf_element_source_fallback(
        context.pdf_audit,
        page_number,
        index,
        "render",
        f"{failure}; {detail}",
        source_text=elem.get("content", ""),
    )
    # Marks the records of this confinement, which are dropped if the page is
    # later delivered as its source page.
    entry["source_ink_kept"] = True
    _exclude_pdf_audit_expectations_for_element(context.pdf_audit, page_number, index)
    context.logger.warning(
        f"[{context.task_id}] Page {page_number} elem {index}: {detail} after a "
        "render failure; translating the rest of the page"
    )
    return entry


def _element_source_refusal(page_num: int, index: int, why: str) -> PDFPageRenderError:
    """A failure that keeping single elements in source cannot cure."""
    error = PDFPageRenderError(page_num + 1, index, f"element kept in source refused: {why}")
    error.element_source_refused = True
    error.figure_table_decided = True
    return error


def _check_kept_source_neighbours(context: PDFAssemblyContext, plan: PDFPageAssemblyPlan) -> None:
    """After redaction: no neighbour of a kept element left a source glyph.

    The kept element's ink (padded like a formula guard) is cut out of the
    neighbours' redaction rectangles.  A neighbour glyph lying wholly inside
    that cut-out would survive under the neighbour's translation (double
    ink), so the page is not confined when:

    - a neighbour's line overlaps a kept line by half a line or more (its
      glyphs could sit inside the kept ink and be mistaken for it), or
    - after redaction a glyph is still centred in a neighbour's ink, outside
      the kept ink and outside the formula regions (whose glyphs every
      redaction keeps by design).
    """
    padding = PDF_FORMULA_GUARD_PADDING

    def padded(rect: fitz.Rect) -> fitz.Rect:
        return fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)

    kept_rects = [
        rect
        for index in sorted(plan.source_elements)
        for rect in _kept_source_ink_rects(plan.elements[index])
    ]
    guards = [padded(rect) for rect in kept_rects]
    kept = min(plan.source_elements)
    touching: dict[int, list[fitz.Rect]] = {}
    for index, elem in enumerate(plan.elements):
        removed = (
            elem.get("type") == ElementType.MERGED_AWAY
            or index in plan.watermark_indices
            or (elem.get("type") == ElementType.TEXT and index in plan.render_indices)
        )
        if not removed or index in plan.source_elements:
            continue
        rects = _kept_source_ink_rects(elem)
        if not any(rect.intersects(guard) for rect in rects for guard in guards):
            continue
        for rect in rects:
            for kept_rect in kept_rects:
                overlap = rect & kept_rect
                if not overlap.is_empty and overlap.height >= 0.5 * min(
                    rect.height, kept_rect.height
                ):
                    raise _element_source_refusal(
                        plan.page_num,
                        kept,
                        f"neighbour elem {index} shares a line with the kept ink",
                    )
        touching[index] = rects
    if not touching:
        return
    formula_guards = [
        padded(_get_pdf_elem_rect(elem))
        for elem in plan.elements
        if elem.get("type") == ElementType.FORMULA_IMAGE
    ]
    for block in plan.out_page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    if not str(char.get("c", "")).strip():
                        continue
                    box = fitz.Rect(char["bbox"])
                    centre = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
                    if any(centre in rect for rect in kept_rects) or any(
                        centre in guard for guard in formula_guards
                    ):
                        continue
                    for index, rects in touching.items():
                        if any(centre in rect for rect in rects):
                            raise _element_source_refusal(
                                plan.page_num,
                                kept,
                                f"source glyph of neighbour elem {index} would survive "
                                "beside the kept ink",
                            )


def _figure_table_refusal(
    page_num: int,
    group: FigureTableGroup,
    why: str,
) -> PDFPageRenderError:
    """The error that withdraws ``group``: its failure takes today's path."""
    error = PDFPageRenderError(
        page_num + 1,
        group.trigger_element,
        f"{group.reason}; figure/table group refused: {why}",
    )
    error.figure_table_refused = group
    return error


def _figure_table_closure(
    elements: list[dict],
    members: frozenset[int],
    kept: set[int],
    translated: set[int],
    watermark: set[int],
    merged_away: set[int],
    endpoints: set[int],
) -> tuple[frozenset[int], str]:
    """Outside elements that must stay in source with ``members``, or a refusal.

    Redacting a translated element erases the glyphs of any text whose box it
    clips, and that text is then re-typeset.  For kept text that would break
    the promise of exact source ink, so every outside element whose
    redaction would force a kept element to be redrawn is kept too (typically
    the figure's own title or an adjacent label), repeated to a fixed point
    bounded by the page's element count.  It is computed for the removal set
    of the next attempt (this attempt's translated text less what the group
    keeps), so it is settled before any attempt is discarded.  A watermark
    (always removed), a merged-away element or a cross-page merge endpoint
    cannot be kept: refusal.  Every element added here is checked by
    ``validate_figure_table_group`` with the members.
    """
    own = set(kept) | set(members)
    remaining = set(translated) - own
    always_removed = set(watermark) | set(merged_away)
    extras: set[int] = set()
    for _ in range(len(elements) + 1):
        removal = remaining | always_removed
        if not _collect_pdf_overlap_redraw_indices(elements, removal) & (own | extras):
            return frozenset(extras), ""
        for culprit in sorted(removal):
            reached = _collect_pdf_overlap_redraw_indices(elements, {culprit}) & (own | extras)
            if not reached:
                continue
            if culprit in watermark:
                return frozenset(), f"watermark elem {culprit} overlaps kept element(s) {sorted(reached)}"
            if culprit in merged_away or culprit in endpoints:
                return frozenset(), (
                    f"cross-page merge elem {culprit} overlaps kept element(s) {sorted(reached)}"
                )
            # Whether a kept outside element is acceptable (prose, body type,
            # titles) is decided with the members by the group validator.
            extras.add(culprit)
            remaining.discard(culprit)
    return frozenset(), "no fixed point for the elements to keep"


def _decide_figure_table_group(
    context: PDFAssemblyContext,
    page_num: int,
    elements: list[dict],
    seed: int,
    reason: str,
    trigger: str,
    *,
    groups: tuple[FigureTableGroup, ...],
    refused: set[int],
    translated: set[int],
    watermark: set[int],
    merged_away: set[int],
) -> FigureTableGroup | None:
    """The group to narrow for this failure, decided inside the attempt.

    Everything that decides whether a group is usable (its members, the
    outside elements it must keep, and the checks on both) is settled here,
    before any page attempt is discarded.  ``None`` leaves the failure on
    today's path in the same attempt: the element restore completes, or the
    page failure is raised.  A seed decided once on a page is never proposed
    again there, and nothing is proposed once the page keeps the maximum
    number of groups.
    """
    if not _figure_table_fallback_enabled(context):
        return None
    if seed in refused or len(groups) >= FIGURE_TABLE_RULES.max_groups_per_page:
        return None
    page_number = page_num + 1
    kept = set().union(*(group.members | group.extra_members for group in groups))
    endpoints = _merge_endpoint_indices(context, page_number)
    dominant_fontsize = _assembly_dominant_fontsize(context)
    page_rect = (
        _pdf_page_extraction_rect(context.out_doc[page_num])
        if context.out_doc is not None
        else None
    )
    refusals: list[str] = []
    group = find_figure_table_group(
        elements,
        seed,
        page_rect=page_rect,
        dominant_fontsize=dominant_fontsize,
        reason=reason,
        protected_indices=endpoints,
        kept_indices=kept,
        graphics=_page_graphic_rects(context, page_num),
        trigger=trigger,
        extend=lambda members: _figure_table_closure(
            elements,
            members,
            kept,
            translated,
            watermark,
            merged_away,
            endpoints,
        ),
        refusals=refusals,
    )
    if group is None:
        refused.add(seed)
        if refusals:
            context.logger.warning(
                f"[{context.task_id}] Page {page_number} elem {seed}: figure/table "
                f"group refused ({refusals[0]}); the failure takes today's path"
            )
    return group


def _attach_figure_table_decision(
    context: PDFAssemblyContext,
    error: PDFPageRenderError,
    page_num: int,
    elements: list[dict],
    *,
    groups: tuple[FigureTableGroup, ...],
    refused: set[int],
    translated: set[int],
    watermark: set[int],
    merged_away: set[int],
) -> None:
    """Decide a page-level render failure's group and attach it to the error."""
    if getattr(error, "figure_table_decided", False):
        return
    error.figure_table_decided = True
    page_number = page_num + 1
    if int(getattr(error, "page", 0) or 0) != page_number:
        return
    if list(getattr(error, "fallback_pages", []) or []) != [page_number]:
        return  # an accepted cross-page merge closure spans pages
    group = _decide_figure_table_group(
        context,
        page_num,
        elements,
        int(getattr(error, "element", -1)),
        str(error),
        FIGURE_TABLE_PAGE_FAILURE,
        groups=groups,
        refused=refused,
        translated=translated,
        watermark=watermark,
        merged_away=merged_away,
    )
    if group is not None:
        error.figure_table_group = group


def _audit_untranslated_english(
    context: PDFAssemblyContext,
    page_num: int,
    elements: list[dict],
    page_trans: dict[int, str],
) -> None:
    """Record translatable English that would survive into the output."""
    log = context.logger
    try:
        leak_entries = _pdf_collect_untranslated_english_leaks(
            page_num + 1,
            elements,
            page_trans,
        )
        if leak_entries:
            context.pdf_audit.setdefault("untranslated_english_leaks", []).extend(leak_entries)
            for leak_entry in leak_entries:
                log.warning(
                    f"[{context.task_id}] Page {page_num + 1} elem "
                    f"{leak_entry['element']}: translatable English "
                    "prose survived untranslated in final output: "
                    f"{leak_entry['text']!r}"
                )
    except Exception as leak_exc:
        log.warning(
            f"[{context.task_id}] Page {page_num + 1}: untranslated-English "
            f"leak audit skipped: {type(leak_exc).__name__}"
        )


def _overlap_redraw_indices(
    page_num: int,
    elements: list[dict],
    removal_seed_indices: set[int],
) -> set[int]:
    """Unchanged text that redaction would erase as collateral and must redraw."""
    redraw_source_indices = _collect_pdf_overlap_redraw_indices(
        elements,
        removal_seed_indices,
    )
    for i in sorted(redraw_source_indices):
        if elements[i].get("skip_translate_reason") in {
            SkipReason.FORMULA_RISK_PRESERVED,
            SkipReason.TABLE_PRESERVED,
        }:
            # A math-dense or table-preserved element cannot be redrawn
            # from extracted glyph order without risking mangled notation
            # or broken row/column alignment.  Its ink would be collateral
            # of a neighbor's redaction, so this page must fall back to
            # the exact source copy.
            raise PDFPageRenderError(
                page_num + 1,
                i,
                f"{elements[i].get('skip_translate_reason')} element would "
                "need a redraw after neighboring redaction",
            )
    return redraw_source_indices


def _redact_page(context: PDFAssemblyContext, plan: PDFPageAssemblyPlan) -> str:
    """Remove source ink safely and return the page HTML-box stylesheet."""

    removal_rects = _source_removal_rects(plan)
    if removal_rects and not _remove_source_ink(context, plan, removal_rects):
        if plan.source_elements:
            # Whiteout paints whole ink boxes, graphics under them included;
            # an attempt that needed it is not eligible for confinement.
            raise _element_source_refusal(
                plan.page_num,
                min(plan.source_elements),
                "text redaction failed and the whiteout path was taken",
            )

    # Text converted to vector outlines is invisible to PDF text
    # redaction.  Paint only the detected glyph area (or the interior
    # of a detected diagram cell), preserving surrounding borders and
    # connector line art, before writing the translated semantic unit.
    for i in sorted(plan.translated_indices):
        _paint_pdf_vector_ocr_element_background(
            plan.out_page,
            plan.elements[i],
            plan.protected_formula_rects,
        )
    return _page_htmlbox_css(context.font_basename)


def _source_removal_rects(plan: PDFPageAssemblyPlan) -> list[fitz.Rect]:
    """Source-ink rectangles to remove, minus the immutable formula regions."""
    removal_rects = []
    for i, elem in enumerate(plan.elements):
        if not (
            elem["type"] == ElementType.MERGED_AWAY
            or i in plan.watermark_indices
            or (elem["type"] == ElementType.TEXT and i in plan.render_indices)
        ):
            continue
        if elem.get("type") in {ElementType.TEXT, ElementType.MERGED_AWAY}:
            source_rects = _get_pdf_source_ink_rects(elem)
        else:
            source_rects = [_get_pdf_elem_rect(elem)]
        for source_rect in source_rects:
            removal_rects.extend(
                _subtract_pdf_protected_rects(
                    source_rect,
                    plan.protected_formula_rects,
                )
            )
    return removal_rects


def _remove_source_ink(
    context: PDFAssemblyContext,
    plan: PDFPageAssemblyPlan,
    removal_rects: list[fitz.Rect],
) -> bool:
    """Redact the text under ``removal_rects``, or white it out if redaction
    fails; ``True`` when the redaction succeeded."""
    out_page = plan.out_page
    try:
        for rect in removal_rects:
            out_page.add_redact_annot(rect, fill=False)
        out_page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )
        return True
    except Exception as exc:
        context.logger.warning(
            f"[{context.task_id}] Page {plan.page_num + 1}: text redaction failed ({exc}); falling back to whiteout"
        )
        # Pending redact annotations would otherwise be saved into the
        # delivered file and render as marked boxes in viewers.
        try:
            for annot in list(
                out_page.annots(types=(fitz.PDF_ANNOT_REDACT,))
            ):
                out_page.delete_annot(annot)
        except Exception:
            pass
    # Paint exactly the recorded ink rects: an expanded margin would
    # cover neighboring glyph edges and the line art the redaction
    # path deliberately preserves, and nothing redraws that sliver.
    for rect in removal_rects:
        for safe_rect in _subtract_pdf_protected_rects(
            fitz.Rect(rect), plan.protected_formula_rects
        ):
            shape = out_page.new_shape()
            shape.draw_rect(safe_rect)
            shape.finish(fill=(1, 1, 1), color=(1, 1, 1))
            shape.commit()
    return False


def _page_htmlbox_css(font_basename: str) -> str:
    return "".join(
        [
            f'@font-face{{font-family:zh;src:url("{font_basename}");}}',
            "*{font-family:zh;margin:0;padding:0;letter-spacing:0px;word-spacing:0px;font-weight:normal;}",
            ".en{letter-spacing:0px;word-spacing:normal;}",
            # A compound citation such as ``<sup>41, 42</sup>`` is one
            # semantic marker.  MuPDF otherwise treats the ordinary space
            # after the comma as a line-break opportunity and can strand
            # ``42`` at the left edge of the next line.  Keep the marker
            # indivisible while still allowing the whole superscript to move
            # to the next line when the preceding line is full.
            PDF_HTMLBOX_SUPERSCRIPT_CSS,
        ]
    )


def _restore_source_or_raise(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    index: int,
    element: dict,
    *,
    fontsize: float,
    color: tuple[float, float, float],
    line_height: float,
    reason: str,
) -> None:
    """Restore the element's source text in place, or fail the page.

    Inside a figure or table that can be bounded, the restored source is not
    kept: the error raised here makes the page be reassembled with that
    figure or table whole in its exact source ink (the restore's ink and
    record go with the discarded attempt).  When no group is usable the
    restore stands, or the page failure is raised, exactly as before.
    """
    restored = _try_restore_source_element(
        context,
        page_plan.out_page,
        page_plan.page_num,
        page_plan.protected_formula_rects,
        index,
        element,
        _get_pdf_elem_rect(element),
        fontsize,
        color,
        line_height,
        reason,
    )
    page_number = page_plan.page_num + 1
    if restored:
        trigger = FIGURE_TABLE_ELEMENT_RESTORE
        detail = "source text would be re-typeset in place"
    else:
        trigger = FIGURE_TABLE_PAGE_FAILURE
        detail = "source restore also failed"
    group = _decide_figure_table_group(
        context,
        page_plan.page_num,
        page_plan.elements,
        index,
        f"Page {page_number} elem {index}: {reason}; {detail}",
        trigger,
        groups=tuple(page_plan.source_groups),
        refused=page_plan.refused_seeds,
        translated=set(page_plan.translated_indices),
        watermark=set(page_plan.watermark_indices),
        merged_away={
            i
            for i, elem in enumerate(page_plan.elements)
            if elem.get("type") == ElementType.MERGED_AWAY
        },
    )
    if group is not None:
        error = PDFPageRenderError(page_number, index, f"{reason}; {detail}")
        error.figure_table_decided = True
        error.figure_table_group = group
        raise error
    if not restored:
        error = PDFPageRenderError(page_number, index, reason)
        error.figure_table_decided = True
        raise error


def _restore_planned_source_or_raise(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    reason: str,
) -> None:
    _restore_source_or_raise(
        context,
        page_plan,
        element_plan.index,
        element_plan.element,
        fontsize=element_plan.fontsize,
        color=element_plan.color,
        line_height=element_plan.line_height,
        reason=reason,
    )


def _prepare_element_render(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    index: int,
) -> PDFElementRenderPlan | None:
    """Resolve one element's style and handle rotated text immediately."""

    element = page_plan.elements[index]
    style = _resolve_pdf_render_style(element, page_plan.layout_styles)
    source_rect = _get_pdf_elem_rect(element)
    fontsize = float(style.get("fontsize", element.get("fontsize", DEFAULT_FONTSIZE)))
    color_int = int(style.get("color", element.get("color", 0)))
    red, green, blue = (
        (color_int >> 16) & 0xFF,
        (color_int >> 8) & 0xFF,
        color_int & 0xFF,
    )
    color = (red / 255.0, green / 255.0, blue / 255.0)
    color_hex = style.get("color_hex", f"#{red:02x}{green:02x}{blue:02x}")
    text = (
        element.get("rich_content") or element.get("content", "")
        if index in page_plan.redraw_source_indices
        else page_plan.translations[index]
    )
    line_height = float(style.get("line_height", fontsize * 1.35))
    if element.get("non_horizontal"):
        _render_rotated_element(
            context,
            page_plan,
            index,
            text,
            fontsize=fontsize,
            color=color,
            line_height=line_height,
        )
        return None
    source_paragraphs = element.get("paragraphs") or [
        {
            "margin_left": 0.0,
            "text_indent": 0.0,
            "gap_before": 0.0,
            "text_align": "left",
        }
    ]
    paragraph_html = _build_paragraph_html(
        element,
        source_paragraphs,
        _split_pdf_translated_paragraphs(text, len(source_paragraphs)),
        fontsize,
        color_hex,
        line_height,
    )
    if not paragraph_html:
        # This element's source ink is already redacted; returning silently
        # would leave a blank region with no record.  Restore the exact
        # source text unless the source itself has no visible text.
        source_plain = _plain_text(
            element.get("rich_content") or element.get("content", "")
        ).strip()
        if source_plain:
            _restore_source_or_raise(
                context,
                page_plan,
                index,
                element,
                fontsize=fontsize,
                color=color,
                line_height=line_height,
                reason="translated payload rendered no visible text",
            )
        return None
    table_cell = bool(element.get("table_hint") or element.get("glossary_cell_hint"))
    top_padding = max(
        0.0,
        element.get("top_padding", 0.0) + (0.0 if table_cell else fontsize * 0.12),
    )
    box_html = (
        '<div style="box-sizing:border-box;width:100%;overflow:visible;'
        f'padding-top:{top_padding:.1f}px;">{"".join(paragraph_html)}</div>'
    )
    return PDFElementRenderPlan(
        index=index,
        element=element,
        source_paragraphs=source_paragraphs,
        source_rect=source_rect,
        render_rect=fitz.Rect(source_rect),
        fontsize=fontsize,
        color=color,
        text=text,
        line_height=line_height,
        table_cell=table_cell,
        box_html=box_html,
        has_semantic_superscript=bool(_pdf_superscript_signature(text)),
    )


def _render_rotated_element(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    index: int,
    text: str,
    *,
    fontsize: float,
    color: tuple[float, float, float],
    line_height: float,
) -> None:
    """Draw a rotated chart or table label at its native rotation."""
    element = page_plan.elements[index]
    font_path = (
        context.bold_font_path
        if bool(element.get("bold")) and context.has_distinct_bold_font
        else context.font_path
    )
    inserted, message = _insert_pdf_rotated_textbox_with_fit(
        page_plan.out_page,
        _get_pdf_elem_rect(element),
        text,
        rotation=int(element.get("rotation", 90) or 90),
        color=color,
        fontfile=font_path,
        fontsize=fontsize,
        protected_rects=page_plan.protected_formula_rects,
    )
    if not inserted:
        _restore_source_or_raise(
            context,
            page_plan,
            index,
            element,
            fontsize=fontsize,
            color=color,
            line_height=line_height,
            reason=f"rotated textbox failed: {message}",
        )
        return
    context.logger.info(
        f"[{context.task_id}] Page {page_plan.page_num + 1} elem "
        f"{index}: rotated textbox {message}"
    )


def _htmlbox_placement_event(
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    step_name: str,
    rect: fitz.Rect,
    render_rect: fitz.Rect,
    scale: float,
) -> dict:
    return {
        "page": page_plan.page_num + 1,
        "element": element_plan.index,
        "step": step_name,
        "rect": [round(float(value), 3) for value in rect],
        "source_rect": [round(float(value), 3) for value in render_rect],
        "scale": scale,
        "neighbor_constrained": "neighbor-trim" in step_name,
        "formula_trimmed": "formula-trim" in step_name,
        "table_cell": element_plan.table_cell,
        "text": _plain_text(element_plan.text)[:160],
    }


def _try_htmlbox(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    css: str,
) -> PDFHTMLBoxResult:
    """Try the no-shrink ladder and one readable-scale HTML-box fallback."""

    render_rect = fitz.Rect(element_plan.render_rect)
    disabled = RuntimeError("PDF htmlbox rendering disabled")
    if not (context.use_htmlbox or element_plan.has_semantic_superscript):
        return PDFHTMLBoxResult(False, disabled, render_rect, None)
    ladder = _formula_safe_htmlbox_ladder(
        context,
        page_plan,
        element_plan.index,
        render_rect,
        element_plan.fontsize,
    )
    if not ladder:
        raise PDFPageRenderError(
            page_plan.page_num + 1,
            element_plan.index,
            "no formula-safe htmlbox rectangle",
        )
    textbox_fit_rects = [fitz.Rect(candidate_rect) for _, candidate_rect in ladder]
    inserted, error = _insert_htmlbox_without_shrinking(
        context, page_plan, element_plan, css, ladder, render_rect
    )
    if not inserted and error is None:
        inserted, error = _insert_htmlbox_at_readable_scale(
            context, page_plan, element_plan, css, ladder[-1], render_rect
        )
    if inserted:
        return PDFHTMLBoxResult(True, disabled, render_rect, textbox_fit_rects)
    return PDFHTMLBoxResult(False, error, ladder[-1][1], textbox_fit_rects)


def _insert_htmlbox_without_shrinking(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    css: str,
    ladder: list[tuple[str, fitz.Rect]],
    render_rect: fitz.Rect,
) -> tuple[bool, Exception | None]:
    """Walk the ladder at full size; return (inserted, the error that stopped it).

    No-shrink policy: insert_htmlbox with scale_low near 1 writes nothing
    when the text cannot fit (returns spare=-1 instead of raising), so the
    ladder can be walked on the real page.
    """
    for step_name, candidate_rect in ladder:
        try:
            spare, _scale = page_plan.out_page.insert_htmlbox(
                candidate_rect,
                element_plan.box_html,
                css=css,
                archive=context.font_archive,
                scale_low=0.98,
            )
        except Exception as exc:
            return False, exc
        if spare >= 0:
            context.pdf_audit["htmlbox_placement_events"].append(
                _htmlbox_placement_event(
                    page_plan, element_plan, step_name, candidate_rect, render_rect, 1.0
                )
            )
            if step_name != "base":
                context.logger.info(
                    f"[{context.task_id}] Page {page_plan.page_num + 1} elem "
                    f"{element_plan.index}: htmlbox kept fontsize via {step_name}"
                )
            return True, None
    return False, None


def _insert_htmlbox_at_readable_scale(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    css: str,
    last_step: tuple[str, fitz.Rect],
    render_rect: fitz.Rect,
) -> tuple[bool, Exception | None]:
    """Fit the last ladder rectangle at a readable scale, or say why not."""
    try:
        return _insert_scaled_htmlbox(
            context, page_plan, element_plan, css, last_step, render_rect
        )
    except PDFPageRenderError:
        raise
    except Exception as exc:
        return False, exc


def _insert_scaled_htmlbox(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    css: str,
    last_step: tuple[str, fitz.Rect],
    render_rect: fitz.Rect,
) -> tuple[bool, Exception | None]:
    pdf_audit = context.pdf_audit
    page_num = page_plan.page_num
    i = element_plan.index
    scale_step_name, scale_rect = last_step
    spare, scale = _insert_pdf_htmlbox_at_readable_scale(
        page_plan.out_page,
        scale_rect,
        element_plan.box_html,
        css=css,
        archive=context.font_archive,
        minimum_scale=context.minimum_htmlbox_scale,
    )
    if spare >= 0:
        scale = float(scale)
        if scale < context.minimum_htmlbox_scale:
            # ``scale_low`` promises that a lower scale is not drawn. If
            # PyMuPDF ever violates that contract, abort the assembly
            # because a fallback would otherwise double-write.
            raise PDFPageRenderError(
                page_num + 1,
                i,
                (
                    f"unreadable htmlbox scale {scale:.2f} below "
                    f"minimum {context.minimum_htmlbox_scale:.2f}"
                ),
                retryable=False,
            )
        pdf_audit["render_scale_events"].append(
            _scale_event(page_plan, element_plan, scale)
        )
        pdf_audit["htmlbox_placement_events"].append(
            _htmlbox_placement_event(
                page_plan,
                element_plan,
                scale_step_name,
                scale_rect,
                render_rect,
                round(scale, 4),
            )
        )
        context.logger.warning(
            f"[{context.task_id}] Page {page_num + 1} elem {i}: htmlbox scaled to {scale:.2f} "
            "after extension ladder exhausted"
        )
        return True, None
    rejected_scale = float(scale)
    if rejected_scale > 0:
        pdf_audit["rejected_render_scale_events"].append(
            _scale_event(page_plan, element_plan, rejected_scale, inserted=False)
        )
        context.logger.info(
            f"[{context.task_id}] Page {page_num + 1} elem {i}: "
            f"htmlbox would require scale {rejected_scale:.2f}; "
            "no HTML was inserted, using textbox fallback"
        )
    return False, RuntimeError(f"htmlbox no-fit even at free scale (scale={scale:.2f})")


def _scale_event(
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    scale: float,
    *,
    inserted: bool | None = None,
) -> dict:
    """An audit render-scale event, keys in their established order."""
    fontsize = element_plan.fontsize
    event = {
        "page": page_plan.page_num + 1,
        "element": element_plan.index,
        "scale": round(scale, 4),
        "source_fontsize": round(fontsize, 3),
        "rendered_fontsize": round(fontsize * scale, 3),
        "table_cell": element_plan.table_cell,
        "redraw_unchanged_source": element_plan.index in page_plan.redraw_source_indices,
    }
    if inserted is not None:
        event["inserted"] = inserted
    event["text"] = _plain_text(element_plan.text)[:160]
    return event


def _render_element(
    context: PDFAssemblyContext,
    plan: PDFPageAssemblyPlan,
    css: str,
    i: int,
) -> None:
    """Render one translated or overlap-redrawn element."""

    element_plan = _prepare_element_render(context, plan, i)
    if element_plan is None:
        return
    htmlbox_result = _try_htmlbox(context, plan, element_plan, css)
    if htmlbox_result.inserted:
        return
    if element_plan.has_semantic_superscript:
        _restore_planned_source_or_raise(
            context,
            plan,
            element_plan,
            f"semantic superscript could not be rendered: {htmlbox_result.error}",
        )
        return
    render_rect = htmlbox_result.render_rect
    textbox_fit_rects = htmlbox_result.textbox_fit_rects
    if not context.use_htmlbox:
        render_rect, textbox_fit_rects = _textbox_rects(
            context, plan, element_plan, render_rect
        )
    _render_textbox_fallback(
        context,
        plan,
        element_plan,
        render_rect,
        textbox_fit_rects,
        htmlbox_result.error,
    )


def _textbox_rects(
    context: PDFAssemblyContext,
    plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    render_rect: fitz.Rect,
) -> tuple[fitz.Rect, list[fitz.Rect]]:
    """Neighbor- and formula-safe textbox rectangles when HTML boxes are off."""
    textbox_ladder = [("base", fitz.Rect(render_rect))]
    if not element_plan.table_cell:
        textbox_ladder = _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
            textbox_ladder,
            render_rect,
            plan.elements,
            element_plan.index,
            element_plan.fontsize,
            ignored_indices=plan.watermark_indices,
        )
    textbox_ladder = context.filter_formula_safe_rects(
        textbox_ladder,
        plan.protected_formula_rects,
        anchor_rect=render_rect,
    )
    if not textbox_ladder:
        raise PDFPageRenderError(
            plan.page_num + 1,
            element_plan.index,
            "no neighbor-safe textbox rectangle",
        )
    return textbox_ladder[-1][1], [
        fitz.Rect(candidate_rect) for _, candidate_rect in textbox_ladder
    ]


def _render_textbox_fallback(
    context: PDFAssemblyContext,
    plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    render_rect: fitz.Rect,
    textbox_fit_rects: list[fitz.Rect] | None,
    htmlbox_err: Exception,
) -> None:
    """Draw the translation as plain text after the HTML box was not used."""
    elem = element_plan.element
    log = context.logger
    page_num = plan.page_num
    i = element_plan.index
    align_name = element_plan.source_paragraphs[0].get("text_align", "left")
    align_code = {"left": 0, "center": 1, "right": 2, "justify": 3}.get(align_name, 0)
    fallback_text = _pdf_plain_drawn_text(element_plan.text).strip()
    if not fallback_text:
        # The ink is already redacted; restore source instead of leaving a
        # silent blank when the source itself had visible text.
        source_plain = _plain_text(
            elem.get("rich_content") or elem.get("content", "")
        ).strip()
        if source_plain:
            _restore_planned_source_or_raise(
                context,
                plan,
                element_plan,
                f"empty textbox fallback after htmlbox failure: {htmlbox_err}",
            )
            return
        log.warning(f"[{context.task_id}] Page {page_num + 1} elem {i}: empty textbox fallback")
        return

    ok, textbox_msg, textbox_fontsize = _insert_pdf_textbox_with_fit(
        plan.out_page,
        render_rect,
        fallback_text,
        align=align_code,
        color=element_plan.color,
        fontfile=context.font_path,
        fontsize=element_plan.fontsize,
        lineheight_points=element_plan.line_height,
        protected_rects=plan.protected_formula_rects,
        allow_compact_fontsize=_is_compact_fixed_table_label(
            elem, element_plan.table_cell, fallback_text
        ),
        fit_rects=textbox_fit_rects,
        minimum_scale=context.minimum_htmlbox_scale,
    )
    if not ok:
        _restore_planned_source_or_raise(
            context,
            plan,
            element_plan,
            f"textbox fallback failed: {htmlbox_err}; {textbox_msg}",
        )
        return
    if textbox_fontsize is not None:
        _record_textbox_fallback(
            context, plan, element_plan, render_rect, textbox_fontsize, textbox_msg
        )
    if "fallback" in textbox_msg or "fit after resize" in textbox_msg:
        log.info(
            f"[{context.task_id}] Page {page_num + 1} elem {i}: textbox fallback adjusted: {textbox_msg}"
        )


def _is_compact_fixed_table_label(elem: dict, table_cell: bool, fallback_text: str) -> bool:
    """A fixed table label may shrink below the textbox floor to stay in its cell."""
    source_plain = _collapsed_plain_text(elem.get("rich_content") or elem.get("content", ""))
    fixed_short_label = _plain_text(
        _translate_pdf_fixed_short_label(source_plain)
    ).strip()
    return bool(
        table_cell
        and fixed_short_label != source_plain
        and fallback_text == fixed_short_label
        and "\n" not in fallback_text
    )


def _record_textbox_fallback(
    context: PDFAssemblyContext,
    plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    render_rect: fitz.Rect,
    textbox_fontsize: float,
    textbox_msg: str,
) -> None:
    fontsize = element_plan.fontsize
    textbox_scale = textbox_fontsize / max(fontsize, 1.0)
    if textbox_scale < context.minimum_htmlbox_scale:
        raise PDFPageRenderError(
            plan.page_num + 1,
            element_plan.index,
            (
                f"unreadable textbox scale {textbox_scale:.2f} below "
                f"minimum {context.minimum_htmlbox_scale:.2f}"
            ),
            retryable=False,
        )
    context.pdf_audit["textbox_fallback_events"].append(
        {
            "page": plan.page_num + 1,
            "element": element_plan.index,
            "rect": [round(float(value), 3) for value in render_rect],
            "scale": round(textbox_scale, 4),
            "source_fontsize": round(fontsize, 3),
            "rendered_fontsize": round(textbox_fontsize, 3),
            "table_cell": element_plan.table_cell,
            "redraw_unchanged_source": element_plan.index in plan.redraw_source_indices,
            "reason": textbox_msg,
            "text": _plain_text(element_plan.text)[:160],
        }
    )


def _assemble_page(
    context: PDFAssemblyContext,
    page_num: int,
    source_groups: tuple[FigureTableGroup, ...] = (),
    refused_seeds: set[int] | None = None,
    source_elements: dict[int, str] | None = None,
) -> None:
    """Assemble one output page and checkpoint HTML-box work when configured."""

    total_pages = context.total_pages
    pdf_password = context.pdf_password
    assembly_work_path = context.assembly_work_path
    out_doc = context.out_doc
    plan = _prepare_page(context, page_num, source_groups, refused_seeds, source_elements)
    if plan is None:
        return
    render_indices = plan.render_indices

    # Remove original text of translated/merged-away/watermark blocks.
    # True redaction (text-only, keeping images and line art such as
    # footnote separator rules) beats painting white boxes: nothing is
    # covered, and copy/search on the output no longer hits the hidden
    # source-language layer. Whiteout remains as the fallback.
    css = _redact_page(context, plan)
    if plan.source_elements:
        _check_kept_source_neighbours(context, plan)

    for i in sorted(render_indices):
        try:
            _render_element(context, plan, css, i)
        except PDFPageRenderError as error:
            _attach_figure_table_decision(
                context,
                error,
                page_num,
                plan.elements,
                groups=tuple(plan.source_groups),
                refused=plan.refused_seeds,
                translated=set(plan.translated_indices),
                watermark=set(plan.watermark_indices),
                merged_away={
                    j
                    for j, elem in enumerate(plan.elements)
                    if elem.get("type") == ElementType.MERGED_AWAY
                },
            )
            raise

    if context.use_htmlbox and assembly_work_path:
        out_doc.saveIncr()
        out_doc.close()
        out_doc = None
        context.out_doc = None
        context.trim_process_memory()
        if page_num < total_pages - 1:
            reopened_document = fitz.open(assembly_work_path)
            try:
                if reopened_document.is_encrypted and not reopened_document.authenticate(
                    pdf_password
                ):
                    raise RuntimeError("PDF密码错误或未提供密码，无法写出加密PDF")
            except BaseException:
                try:
                    reopened_document.close()
                except Exception as cleanup_error:
                    context.logger.warning(
                        "[%s] Failed to close rejected reopened PDF: %s",
                        context.task_id,
                        cleanup_error,
                    )
                raise
            out_doc = reopened_document
    context.out_doc = out_doc


_MISSING = object()


def _shallow_audit_copy(audit: PdfAudit) -> dict:
    """Copy the audit's top-level lists and dicts; page attempts only append,
    filter and count at that level."""
    return {
        key: (
            list(value)
            if isinstance(value, list)
            else dict(value) if isinstance(value, dict) else value
        )
        for key, value in audit.items()
    }


@dataclass(frozen=True)
class _PageAttemptSnapshot:
    """Audit and element state before a page attempt, for a clean retry."""

    audit: dict
    skip_reasons: tuple

    @classmethod
    def capture(cls, context: PDFAssemblyContext, page_num: int) -> "_PageAttemptSnapshot":
        elements = (context.page_extractions.get(page_num) or {}).get("elements") or []
        return cls(
            audit=_shallow_audit_copy(context.pdf_audit),
            skip_reasons=tuple(
                elem.get("skip_translate_reason", _MISSING) for elem in elements
            ),
        )

    def restore(self, context: PDFAssemblyContext, page_num: int) -> None:
        # The audit object is shared with the output stage: mutate in place.
        context.pdf_audit.clear()
        context.pdf_audit.update(_shallow_audit_copy(self.audit))
        elements = (context.page_extractions.get(page_num) or {}).get("elements") or []
        for elem, reason in zip(elements, self.skip_reasons):
            if reason is _MISSING:
                elem.pop("skip_translate_reason", None)
            else:
                elem["skip_translate_reason"] = reason


def _figure_table_fallback_enabled(context: PDFAssemblyContext) -> bool:
    """Narrowing needs the per-page checkpoint that lets a page be redone."""
    return bool(
        getattr(context, "fail_open_to_source_page", False)
        and getattr(context, "use_htmlbox", False)
        and getattr(context, "assembly_work_path", None)
    )


def _merge_endpoint_indices(context: PDFAssemblyContext, page_number: int) -> set[int]:
    """Elements of this page that are endpoints of an accepted cross-page merge."""
    endpoints = set()
    for decision in context.pdf_audit.get("merge_decisions") or []:
        if not isinstance(decision, dict) or decision.get("decision") != "accepted":
            continue
        for page_key, element_key in (
            ("source_page", "source_element"),
            ("destination_page", "destination_element"),
        ):
            try:
                if int(decision.get(page_key, 0) or 0) == page_number:
                    endpoints.add(int(decision.get(element_key, -1)))
            except (TypeError, ValueError):
                continue
    return endpoints


def _assembly_dominant_fontsize(context: PDFAssemblyContext) -> float:
    """The document's dominant type size, computed once per assembly.

    Assembly never changes element type sizes, so the whole-document scan
    is done on first use and reused by every grouping lookup.
    """
    cache = getattr(context, "figure_table_cache", None)
    if cache is None:
        return _cross_page_dominant_fontsize(context.page_extractions)
    if "dominant_fontsize" not in cache:
        cache["dominant_fontsize"] = _cross_page_dominant_fontsize(context.page_extractions)
    return cache["dominant_fontsize"]


def _page_graphic_rects(context: PDFAssemblyContext, page_num: int) -> list[fitz.Rect]:
    """Visible vector drawings of the source page: evidence of a figure's extent.

    Read from the untouched source PDF (the output page may already carry
    this attempt's whiteouts) and cached per page for the assembly.
    """
    cache = getattr(context, "figure_table_cache", None)
    if cache is None:
        return []
    by_page = cache.setdefault("graphics", {})
    if page_num in by_page:
        return by_page[page_num]
    rects: list[fitz.Rect] = []
    src_path = getattr(context, "src_path", "") or ""
    if src_path:
        try:
            with fitz.open(src_path) as source:
                if source.is_encrypted and not source.authenticate(context.pdf_password):
                    raise RuntimeError("source PDF could not be authenticated")
                for drawing in source[page_num].get_drawings():
                    fill, stroke = drawing.get("fill"), drawing.get("color")
                    visible = any(
                        colour is not None and tuple(colour) != (1.0, 1.0, 1.0)
                        for colour in (fill, stroke)
                    )
                    rect = fitz.Rect(drawing.get("rect"))
                    if visible and not rect.is_infinite:
                        rects.append(rect)
        except Exception as exc:
            context.logger.warning(
                f"[{context.task_id}] Page {page_num + 1}: figure/table graphics "
                f"unavailable ({type(exc).__name__}); extents use text only"
            )
            rects = []
    by_page[page_num] = rects
    return rects


def _discard_page_attempt(context: PDFAssemblyContext) -> None:
    """Drop the current page's uncommitted ink by reopening the last checkpoint."""
    document = context.out_doc
    context.out_doc = None
    if document is not None:
        document.close()
    reopened_document = fitz.open(context.assembly_work_path)
    try:
        if reopened_document.is_encrypted and not reopened_document.authenticate(
            context.pdf_password
        ):
            raise RuntimeError("PDF密码错误或未提供密码，无法重新组装当前页")
    except BaseException:
        reopened_document.close()
        raise
    context.out_doc = reopened_document


# Hard ceiling on single elements kept in source ink on one page.
ELEMENT_SOURCE_MAX_PER_PAGE = 8


def _confinable_element(
    context: PDFAssemblyContext,
    page_num: int,
    error: PDFPageRenderError,
    groups: list[FigureTableGroup],
    confined: dict[int, str],
) -> int | None:
    """The element a page failure can be confined to, or ``None``.

    Only a failure of this page alone, attributed to one text element that is
    not yet kept (by a group or alone), is confined.  An accepted cross-page
    merge endpoint never is: its sentence would be half in each language, so
    the merge closure fallback stays.  Watermarks and merged-away text are
    always removed, so they cannot be kept.
    """
    if getattr(error, "element_source_refused", False):
        return None
    if getattr(error, "figure_table_refused", None) is not None:
        return None
    page_number = page_num + 1
    if int(getattr(error, "page", 0) or 0) != page_number:
        return None
    if list(getattr(error, "fallback_pages", []) or []) != [page_number]:
        return None
    try:
        index = int(getattr(error, "element", -1))
    except (TypeError, ValueError):
        return None
    elements = (context.page_extractions.get(page_num) or {}).get("elements") or []
    if not 0 <= index < len(elements):
        return None
    elem = elements[index]
    if elem.get("type") != ElementType.TEXT:
        return None
    if elem.get("skip_translate_reason") == SkipReason.WATERMARK:
        return None
    kept = set().union(*(group.members | group.extra_members for group in groups))
    if index in confined or index in kept:
        return None
    if index in _merge_endpoint_indices(context, page_number):
        return None
    if len(confined) >= ELEMENT_SOURCE_MAX_PER_PAGE:
        context.logger.warning(
            f"[{context.task_id}] Page {page_number} elem {index}: "
            f"{ELEMENT_SOURCE_MAX_PER_PAGE} element(s) already kept in source; "
            "the page-level fallback applies"
        )
        return None
    return index


def _assemble_page_with_figure_table_fallbacks(
    context: PDFAssemblyContext,
    page_num: int,
) -> None:
    """Assemble one page; a failing figure, table or element falls back alone.

    Whether a failure narrows a figure or table is decided inside the
    attempt (``_decide_figure_table_group``).  A refused group costs nothing:
    the failure takes today's path in the same attempt.  An accepted group
    discards the attempt: the page's uncommitted ink is dropped (the page was
    checkpointed before it), its audit and element state are restored, and
    the page is assembled again with the group kept in its exact source ink.
    A page failure no group takes, attributed to one element
    (``_confinable_element``), is handled the same way with that element
    alone kept in its exact source ink and protected from its neighbours.

    Bound: group attempts are limited as before (``2 * max_groups_per_page``
    retries: each accepted group has a seed never grouped before on the
    page, at most ``max_groups_per_page`` held at once, and a group dropped
    by the safety net bars its seed), and each element retry keeps one more
    element, at most ``ELEMENT_SOURCE_MAX_PER_PAGE``; so at most
    ``2 * max_groups_per_page + ELEMENT_SOURCE_MAX_PER_PAGE + 1`` attempts.
    On exhaustion, and for any failure that is not narrowed, the page's
    state is restored to before the first attempt (when an attempt was
    discarded) and the page error is raised, so the page-level fallback
    applies.
    """
    if not _figure_table_fallback_enabled(context):
        _assemble_page(context, page_num)
        return
    snapshot = _PageAttemptSnapshot.capture(context, page_num)
    groups: list[FigureTableGroup] = []
    refused: set[int] = set()
    confined: dict[int, str] = {}
    discarded = False  # whether an attempt was thrown away (state to restore)
    max_group_retries = 2 * FIGURE_TABLE_RULES.max_groups_per_page
    group_retries = 0
    last_error: PDFPageRenderError | None = None
    # Defence in depth: every retry above grows a bounded set, so this hard
    # ceiling is never reached by a correct page.
    for _attempt in range(max_group_retries + ELEMENT_SOURCE_MAX_PER_PAGE + 1):
        try:
            if confined:
                _assemble_page(context, page_num, tuple(groups), refused, dict(confined))
            else:
                _assemble_page(context, page_num, tuple(groups), refused)
            return
        except PDFPageRenderError as error:
            last_error = error
            dropped = getattr(error, "figure_table_refused", None)
            group = getattr(error, "figure_table_group", None)
            if dropped is not None and dropped in groups:
                if group_retries >= max_group_retries:
                    break
                group_retries += 1
                groups.remove(dropped)
                refused.add(dropped.trigger_element)
                message = (
                    f"figure/table group of elem {dropped.trigger_element} dropped "
                    f"({error}); reassembling the page without it"
                )
            elif group is not None:
                if group_retries >= max_group_retries:
                    break
                group_retries += 1
                groups.append(group)
                message = (
                    f"elem {group.trigger_element}: {group.trigger} inside a "
                    f"figure/table ({group.kind}); reassembling the page with its "
                    f"{len(group.members)} element(s) kept in exact source ink"
                )
            else:
                index = _confinable_element(context, page_num, error, groups, confined)
                if index is None:
                    if discarded:
                        snapshot.restore(context, page_num)
                        context.logger.warning(
                            f"[{context.task_id}] Page {page_num + 1}: keeping "
                            f"{len(groups)} figure/table group(s) and "
                            f"{len(confined)} element(s) in source did not make "
                            "the page renderable; the page-level fallback applies"
                        )
                    raise
                confined[index] = str(error)
                message = (
                    f"elem {index}: render failure outside any figure/table "
                    f"({error}); reassembling the page with that element alone "
                    "kept in exact source ink"
                )
            _discard_page_attempt(context)
            discarded = True
            snapshot.restore(context, page_num)
            context.logger.warning(f"[{context.task_id}] Page {page_num + 1} {message}")
    snapshot.restore(context, page_num)
    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1}: figure/table and element "
        "narrowing used all their attempts; the page-level fallback applies"
    )
    raise last_error


def _continue_assembly_with_source_pages(
    context: PDFAssemblyContext,
    page_num: int,
    error: PDFPageRenderError,
) -> bool:
    """Discard the failed page mutation and continue in the same assembly pass.

    HTML-box assembly checkpoints each completed page with ``saveIncr``. A
    render exception therefore leaves only the current page dirty in memory.
    Closing without another save restores that page from the last checkpoint;
    any earlier page pulled into the fallback through an accepted cross-page
    merge is atomically replaced from the source PDF before reopening.
    """
    if not (
        context.fail_open_to_source_page
        and context.assembly_work_path
        and context.src_path
    ):
        return False

    current_page = page_num + 1
    if int(getattr(error, "page", 0) or 0) != current_page:
        return False
    fallback_pages = sorted({
        int(page)
        for page in _expand_pdf_fallback_pages_for_accepted_merges(
            list(getattr(error, "fallback_pages", []) or []),
            context.pdf_audit.get("merge_decisions", []),
        )
        if 1 <= int(page) <= context.total_pages
    })
    existing_pages = {
        int(entry.get("page", 0) or 0)
        for entry in context.pdf_audit.get("source_page_fallbacks", [])
    }
    new_pages = [page for page in fallback_pages if page not in existing_pages]
    if not new_pages:
        return False

    document = context.out_doc
    context.out_doc = None
    if document is not None:
        document.close()

    # The current and future pages were never checkpointed after this failure,
    # so reopening is sufficient for them. Only already-committed pages need
    # an explicit source replacement.
    previous_pages = [page for page in new_pages if page < current_page]
    if previous_pages:
        _replace_pdf_pages_with_source(
            context.assembly_work_path,
            context.src_path,
            previous_pages,
            source_password=context.pdf_password,
            task_id=context.task_id,
        )

    reopened_document = fitz.open(context.assembly_work_path)
    try:
        if reopened_document.is_encrypted and not reopened_document.authenticate(
            context.pdf_password
        ):
            raise RuntimeError("PDF密码错误或未提供密码，无法恢复源页组装")
    except BaseException:
        reopened_document.close()
        raise

    for fallback_page in new_pages:
        entry = _append_pdf_source_page_fallback(
            context.pdf_audit.setdefault("source_page_fallbacks", []),
            _pdf_source_page_fallback_entry(
                fallback_page,
                "render",
                str(error),
                error_type=type(error).__name__,
            ),
        )
        context.page_extractions.setdefault(
            fallback_page - 1,
            {},
        )["source_page_fallback"] = entry
    _exclude_pdf_audit_expectations_for_source_pages(
        context.pdf_audit,
        new_pages,
    )
    # A page narrowed earlier in this pass and now pulled back to source
    # through an accepted merge keeps no narrowing record or state.
    _drop_pdf_figure_table_audit_records(context.pdf_audit, new_pages)
    _clear_pdf_figure_table_skip_reasons(context.page_extractions, new_pages)
    context.pdf_audit.setdefault(
        "inline_render_source_page_fallbacks",
        [],
    ).append({
        "failed_page": current_page,
        "pages": new_pages,
        "reason": str(error),
    })
    context.out_doc = reopened_document
    context.logger.warning(
        f"[{context.task_id}] Page {current_page} render failed; restored exact "
        f"source page(s) {new_pages} and continuing the current assembly pass"
    )
    return True


def assemble_document_pages(context: PDFAssemblyContext) -> None:
    """Render every translated page and retain the current output document."""

    if getattr(context, "figure_table_cache", None) is not None:
        # Values cached for one assembly only (a reused context starts clean).
        context.figure_table_cache = {}
    for page_num in range(context.total_pages):
        try:
            _assemble_page_with_figure_table_fallbacks(context, page_num)
        except PDFPageRenderError as error:
            if _continue_assembly_with_source_pages(
                context,
                page_num,
                error,
            ):
                continue
            raise


__all__ = ["PDFAssemblyContext", "assemble_document_pages"]


