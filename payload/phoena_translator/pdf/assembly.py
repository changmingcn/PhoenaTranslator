"""Position-preserving PDF page assembly stage."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import fitz

from phoena_translator.pdf.audit import (
    _expand_pdf_fallback_pages_for_accepted_merges,
)
from phoena_translator.pdf.cache import (
    _exclude_pdf_audit_expectations_for_element,
    _record_pdf_element_source_fallback,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_source_ink_rects,
    _subtract_pdf_protected_rects,
)
from phoena_translator.pdf.math_detection import (
    _normalize_pdf_translation,
    _pdf_superscript_signature,
    _plain_text,
)
from phoena_translator.pdf.rendering import (
    _build_htmlbox_rect_ladder,
    _build_pdf_rotated_text_rects,
    _collect_pdf_overlap_redraw_indices,
    _constrain_pdf_htmlbox_rect_ladder_to_neighbors,
    _expand_pdf_htmlbox_rect,
    _insert_pdf_htmlbox_at_readable_scale,
    _insert_pdf_rotated_textbox_with_fit,
    _insert_pdf_textbox_with_fit,
    _paint_pdf_vector_ocr_element_background,
    _pdf_plain_text_to_html,
    _pdf_text_to_html,
    _resolve_pdf_render_style,
    _restore_pdf_source_text_element,
)
from phoena_translator.pdf.semantics import (
    _derive_pdf_page_layout_styles,
    _strip_pdf_toc_label,
)
from phoena_translator.pdf.targets import _translate_pdf_fixed_short_label
from phoena_translator.pdf.translation import (
    _pdf_collect_untranslated_english_leaks,
    _pdf_toc_entries,
    _split_pdf_translated_paragraphs,
)
from phoena_translator.pdf.types import (
    PDF_HTMLBOX_SUPERSCRIPT_CSS,
    PDFPageRenderError,
)


@dataclass
class PDFAssemblyContext:
    """Mutable document and immutable rendering settings for page assembly."""

    task_id: str
    total_pages: int
    pdf_password: str
    use_htmlbox: bool
    minimum_htmlbox_scale: float
    assembly_work_path: str | None
    out_doc: Any
    page_extractions: dict[int, dict]
    pdf_audit: dict
    font_path: str
    bold_font_path: str
    has_distinct_bold_font: bool
    font_archive: Any
    font_basename: str
    logger: logging.Logger
    filter_formula_safe_rects: Callable[..., list[Any]]
    trim_process_memory: Callable[[], None]


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


@dataclass(frozen=True)
class PDFElementRenderPlan:
    """Geometry, text, and style prepared for one renderable element."""

    index: int
    element: dict
    render_style: dict
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
    element["skip_translate_reason"] = "render_source_restore"
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


def _prepare_page(
    context: PDFAssemblyContext,
    page_num: int,
) -> PDFPageAssemblyPlan | None:
    """Resolve translations and validate formula-safe placement before redaction."""

    task_id = context.task_id
    PDF_USE_HTMLBOX = context.use_htmlbox
    out_doc = context.out_doc
    page_extractions = context.page_extractions
    pdf_audit = context.pdf_audit
    log = context.logger
    _filter_pdf_formula_safe_rects = context.filter_formula_safe_rects
    out_page = out_doc[page_num]

    info = page_extractions.get(page_num, {})

    if info.get("source_page_fallback"):
        return
    if "appendix" in info or ("elements" not in info and "cached" not in info):
        return

    elements = info.get("elements", [])
    merged_away_indices = {
        i for i, elem in enumerate(elements) if elem.get("type") == "text_merged_away"
    }
    if "cached" in info:
        page_trans = {int(k): v for k, v in info["cached"].items()}
    elif merged_away_indices:
        # Cross-page semantic merging may consume the only text element on a
        # page.  Such a page still needs an assembly plan so its stale source
        # ink is redacted even though it has no translation payload.
        page_trans = {}
    else:
        return

    layout_styles = info.get("layout_styles") or _derive_pdf_page_layout_styles(
        elements, out_page.rect
    )
    watermark_indices = {
        i
        for i, elem in enumerate(elements)
        if elem.get("skip_translate_reason") == "watermark"
    }
    protected_formula_rects = [
        fitz.Rect(elem.get("bbox", elem.get("rect")))
        for elem in elements
        if elem.get("type") == "formula_image"
    ]

    # Find which elements were actually translated
    translated_indices = set()
    for i, elem in enumerate(elements):
        if i in watermark_indices:
            continue
        if elem["type"] == "text" and i in page_trans:
            if page_trans[i] != elem.get("content", ""):
                translated_indices.add(i)

    # Validate formula-safe placement before redacting any source text.
    # Previously this check happened only during insertion, after the
    # original glyphs had already been removed; a no-safe-rectangle
    # result could therefore create a silent blank. Nothing has been
    # redacted yet, so an unplaceable element can simply keep its
    # original source ink while the rest of the page still translates.
    unplaceable_indices = set()
    for i in sorted(translated_indices):
        elem = elements[i]
        text_to_insert = page_trans[i]
        render_style = _resolve_pdf_render_style(elem, layout_styles)
        bbox = elem.get("bbox", elem.get("rect"))
        render_rect = fitz.Rect(bbox) if isinstance(bbox, list) else fitz.Rect(bbox)
        fontsize = float(render_style.get("fontsize", elem.get("fontsize", 11.0)))
        if elem.get("non_horizontal"):
            if not _build_pdf_rotated_text_rects(
                render_rect,
                out_page.rect,
                fontsize,
                int(elem.get("rotation", 90) or 90),
                protected_formula_rects,
            ):
                unplaceable_indices.add(i)
            continue
        if PDF_USE_HTMLBOX or _pdf_superscript_signature(text_to_insert):
            is_table_cell = bool(
                elem.get("table_hint") or elem.get("glossary_cell_hint")
            )
            ladder = (
                [("base", fitz.Rect(render_rect))]
                if is_table_cell
                else _build_htmlbox_rect_ladder(
                    render_rect,
                    out_page.rect,
                    fontsize,
                    prefer_widen=bool(elem.get("single_line_heading")),
                )
            )
            if not is_table_cell:
                ladder = _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
                    ladder,
                    render_rect,
                    elements,
                    i,
                    fontsize,
                    ignored_indices=watermark_indices,
                )
            if not _filter_pdf_formula_safe_rects(
                ladder,
                protected_formula_rects,
                anchor_rect=render_rect,
            ):
                unplaceable_indices.add(i)

    for i in sorted(unplaceable_indices):
        elem = elements[i]
        merge_fallback_error = _cross_page_merge_fallback_error(
            context,
            page_num,
            i,
            "no formula-safe rectangle for translated merge endpoint",
        )
        if merge_fallback_error is not None:
            raise merge_fallback_error
        elem["skip_translate_reason"] = "render_no_safe_rect_fallback"
        translated_indices.discard(i)
        page_trans[i] = elem.get("content", "")
        _record_pdf_element_source_fallback(
            pdf_audit,
            page_num + 1,
            i,
            "render-preflight",
            "no formula-safe rectangle; source ink kept in place",
            source_text=elem.get("content", ""),
        )
        _exclude_pdf_audit_expectations_for_element(
            pdf_audit,
            page_num + 1,
            i,
        )
        log.warning(
            f"[{task_id}] Page {page_num + 1} elem {i}: no formula-safe "
            "rectangle; keeping the element's exact source ink and "
            "translating the rest of the page"
        )

    try:
        leak_entries = _pdf_collect_untranslated_english_leaks(
            page_num + 1,
            elements,
            page_trans,
        )
        if leak_entries:
            pdf_audit.setdefault("untranslated_english_leaks", []).extend(leak_entries)
            for leak_entry in leak_entries:
                log.warning(
                    f"[{task_id}] Page {page_num + 1} elem "
                    f"{leak_entry['element']}: translatable English "
                    "prose survived untranslated in final output: "
                    f"{leak_entry['text']!r}"
                )
    except Exception as leak_exc:
        log.warning(
            f"[{task_id}] Page {page_num + 1}: untranslated-English "
            f"leak audit skipped: {type(leak_exc).__name__}"
        )

    if not translated_indices and not watermark_indices and not merged_away_indices:
        return  # nothing to change on this page

    removal_seed_indices = (
        set(translated_indices) | set(watermark_indices) | merged_away_indices
    )
    redraw_source_indices = _collect_pdf_overlap_redraw_indices(
        elements,
        removal_seed_indices,
    )
    for i in sorted(redraw_source_indices):
        if elements[i].get("skip_translate_reason") in {
            "formula_risk_preserved",
            "table_preserved",
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
    render_indices = set(translated_indices) | redraw_source_indices
    if redraw_source_indices:
        log.info(
            f"[{task_id}] Page {page_num + 1}: redrawing {len(redraw_source_indices)} "
            "unchanged overlapping text element(s) after redaction"
        )
    return PDFPageAssemblyPlan(
        page_num=page_num,
        out_page=out_page,
        elements=elements,
        translations=page_trans,
        layout_styles=layout_styles,
        watermark_indices=watermark_indices,
        protected_formula_rects=protected_formula_rects,
        translated_indices=translated_indices,
        redraw_source_indices=redraw_source_indices,
        render_indices=render_indices,
    )


def _redact_page(context: PDFAssemblyContext, plan: PDFPageAssemblyPlan) -> str:
    """Remove source ink safely and return the page HTML-box stylesheet."""

    task_id = context.task_id
    page_num = plan.page_num
    out_page = plan.out_page
    elements = plan.elements
    watermark_indices = plan.watermark_indices
    protected_formula_rects = plan.protected_formula_rects
    translated_indices = plan.translated_indices
    render_indices = plan.render_indices
    font_basename = context.font_basename
    log = context.logger
    removal_rects = []
    for i, elem in enumerate(elements):
        if (
            elem["type"] == "text_merged_away"
            or i in watermark_indices
            or (elem["type"] == "text" and i in render_indices)
        ):
            if elem.get("type") in {"text", "text_merged_away"}:
                source_rects = _get_pdf_source_ink_rects(elem)
            else:
                bbox = elem.get("bbox", elem.get("rect"))
                source_rects = [
                    fitz.Rect(bbox) if isinstance(bbox, list) else fitz.Rect(bbox)
                ]
            for source_rect in source_rects:
                removal_rects.extend(
                    _subtract_pdf_protected_rects(
                        source_rect,
                        protected_formula_rects,
                    )
                )
    if removal_rects:
        redacted = False
        try:
            for rect in removal_rects:
                out_page.add_redact_annot(rect, fill=False)
            out_page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )
            redacted = True
        except Exception as exc:
            log.warning(
                f"[{task_id}] Page {page_num + 1}: text redaction failed ({exc}); falling back to whiteout"
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
        if not redacted:
            # Paint exactly the recorded ink rects: an expanded margin would
            # cover neighboring glyph edges and the line art the redaction
            # path deliberately preserves, and nothing redraws that sliver.
            for rect in removal_rects:
                for safe_rect in _subtract_pdf_protected_rects(
                    fitz.Rect(rect), protected_formula_rects
                ):
                    shape = out_page.new_shape()
                    shape.draw_rect(safe_rect)
                    shape.finish(fill=(1, 1, 1), color=(1, 1, 1))
                    shape.commit()

    # Text converted to vector outlines is invisible to PDF text
    # redaction.  Paint only the detected glyph area (or the interior
    # of a detected diagram cell), preserving surrounding borders and
    # connector line art, before writing the translated semantic unit.
    for i in sorted(translated_indices):
        elem = elements[i]
        _paint_pdf_vector_ocr_element_background(
            out_page,
            elem,
            protected_formula_rects,
        )

    css_parts = [
        f'@font-face{{font-family:zh;src:url("{font_basename}");}}',
    ]
    css_parts.append(
        "*{font-family:zh;margin:0;padding:0;letter-spacing:0px;word-spacing:0px;font-weight:normal;}"
    )
    css_parts.append(".en{letter-spacing:0px;word-spacing:normal;}")
    # A compound citation such as ``<sup>41, 42</sup>`` is one
    # semantic marker.  MuPDF otherwise treats the ordinary space
    # after the comma as a line-break opportunity and can strand
    # ``42`` at the left edge of the next line.  Keep the marker
    # indivisible while still allowing the whole superscript to move
    # to the next line when the preceding line is full.
    css_parts.append(PDF_HTMLBOX_SUPERSCRIPT_CSS)
    css = "".join(css_parts)
    return css


def _build_paragraph_html(
    element: dict,
    source_paragraphs: list[dict],
    translated_paragraphs: list[str],
    fontsize: float,
    color_hex: str,
    line_height: float,
) -> list[str]:
    """Render paragraph metadata into deterministic HTML fragments."""

    fragments = []
    for metadata, translated_text in zip(source_paragraphs, translated_paragraphs):
        if not translated_text.strip():
            continue
        margin_top = max(0.0, metadata.get("gap_before", 0.0))
        margin_left = max(0.0, metadata.get("margin_left", 0.0))
        text_indent = metadata.get("text_indent", 0.0)
        text_align = metadata.get("text_align", "left")
        nowrap = metadata.get("nowrap", False)
        toc_metadata = metadata.get("toc_leader")
        source_toc_entries = _pdf_toc_entries(metadata.get("plain", ""))
        translated_toc_entries = _pdf_toc_entries(translated_text)
        entries_match = (
            len(source_toc_entries) > 1
            and len(source_toc_entries) == len(translated_toc_entries)
            and [page for _, page in source_toc_entries]
            == [page for _, page in translated_toc_entries]
        )
        if entries_match:
            rows = []
            for (_, page_reference), (translated_label, _) in zip(
                source_toc_entries,
                translated_toc_entries,
            ):
                safe_label = _pdf_plain_text_to_html(translated_label)
                safe_page = html.escape(page_reference)
                rows.append(
                    "<tr>"
                    f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{safe_label}</td>'
                    '<td style="padding:0 4px;vertical-align:baseline;white-space:nowrap;'
                    'width:3.2em;text-align:center;">......</td>'
                    f'<td style="padding:0;vertical-align:baseline;white-space:nowrap;'
                    f'width:3.2em;text-align:right;">{safe_page}</td>'
                    "</tr>"
                )
            fragments.append(
                f'<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;padding:0;"
                f"font-size:{fontsize:.1f}px;color:{color_hex};"
                f'line-height:{line_height:.1f}px;">{"".join(rows)}</table>'
            )
        elif toc_metadata:
            label = (
                _strip_pdf_toc_label(translated_text, toc_metadata["page"])
                or toc_metadata["label"]
            )
            safe_label = _pdf_plain_text_to_html(label)
            safe_page = html.escape(toc_metadata["page"])
            fragments.append(
                f'<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;padding:0;"
                f"font-size:{fontsize:.1f}px;color:{color_hex};"
                f'line-height:{line_height:.1f}px;"><tr>'
                f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{safe_label}</td>'
                '<td style="padding:0 4px;vertical-align:baseline;white-space:nowrap;'
                'width:3.2em;text-align:center;">......</td>'
                '<td style="padding:0;vertical-align:baseline;white-space:nowrap;'
                f'width:3.2em;text-align:right;">{safe_page}</td></tr></table>'
            )
        else:
            safe_text = _pdf_text_to_html(
                translated_text,
                superscript_scale=float(element.get("superscript_scale") or 0.60),
            )
            single_line_heading = bool(element.get("single_line_heading"))
            white_space = (
                "white-space:nowrap;"
                if single_line_heading
                or (element.get("layout_class", "body") != "body" and nowrap)
                else ""
            )
            fragments.append(
                f'<p style="font-size:{fontsize:.1f}px;color:{color_hex};'
                f"line-height:{line_height:.1f}px;"
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;"
                f"padding:0;text-indent:{text_indent:.1f}px;"
                f'text-align:{text_align};{white_space}">{safe_text}</p>'
            )
    return fragments


def _prepare_element_render(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    index: int,
) -> PDFElementRenderPlan | None:
    """Resolve one element's style and handle rotated text immediately."""

    element = page_plan.elements[index]
    style = _resolve_pdf_render_style(element, page_plan.layout_styles)
    bbox = element.get("bbox", element.get("rect"))
    source_rect = fitz.Rect(bbox) if isinstance(bbox, list) else fitz.Rect(bbox)
    fontsize = float(style.get("fontsize", element.get("fontsize", 11.0)))
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
        font_path = (
            context.bold_font_path
            if bool(element.get("bold")) and context.has_distinct_bold_font
            else context.font_path
        )
        inserted, message = _insert_pdf_rotated_textbox_with_fit(
            page_plan.out_page,
            source_rect,
            text,
            rotation=int(element.get("rotation", 90) or 90),
            color=color,
            fontfile=font_path,
            fontsize=fontsize,
            protected_rects=page_plan.protected_formula_rects,
        )
        if not inserted:
            reason = f"rotated textbox failed: {message}"
            if not _try_restore_source_element(
                context,
                page_plan.out_page,
                page_plan.page_num,
                page_plan.protected_formula_rects,
                index,
                element,
                source_rect,
                fontsize,
                color,
                line_height,
                reason,
            ):
                raise PDFPageRenderError(page_plan.page_num + 1, index, reason)
        else:
            context.logger.info(
                f"[{context.task_id}] Page {page_plan.page_num + 1} elem "
                f"{index}: rotated textbox {message}"
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
    translated_paragraphs = _split_pdf_translated_paragraphs(
        text,
        len(source_paragraphs),
    )
    paragraph_html = _build_paragraph_html(
        element,
        source_paragraphs,
        translated_paragraphs,
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
            reason = "translated payload rendered no visible text"
            if not _try_restore_source_element(
                context,
                page_plan.out_page,
                page_plan.page_num,
                page_plan.protected_formula_rects,
                index,
                element,
                source_rect,
                fontsize,
                color,
                line_height,
                reason,
            ):
                raise PDFPageRenderError(page_plan.page_num + 1, index, reason)
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
        render_style=style,
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


def _try_htmlbox(
    context: PDFAssemblyContext,
    page_plan: PDFPageAssemblyPlan,
    element_plan: PDFElementRenderPlan,
    css: str,
) -> PDFHTMLBoxResult:
    """Try the no-shrink ladder and one readable-scale HTML-box fallback."""

    task_id = context.task_id
    page_num = page_plan.page_num
    i = element_plan.index
    PDF_USE_HTMLBOX = context.use_htmlbox
    PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE = context.minimum_htmlbox_scale
    out_page = page_plan.out_page
    elements = page_plan.elements
    watermark_indices = page_plan.watermark_indices
    protected_formula_rects = page_plan.protected_formula_rects
    redraw_source_indices = page_plan.redraw_source_indices
    pdf_audit = context.pdf_audit
    font_archive = context.font_archive
    log = context.logger
    _filter_pdf_formula_safe_rects = context.filter_formula_safe_rects
    elem = element_plan.element
    render_rect = fitz.Rect(element_plan.render_rect)
    fontsize = element_plan.fontsize
    text_to_insert = element_plan.text
    is_table_cell = element_plan.table_cell
    box_html = element_plan.box_html
    has_semantic_superscript = element_plan.has_semantic_superscript
    htmlbox_err = RuntimeError("PDF htmlbox rendering disabled")
    inserted = False
    textbox_fit_rects = None
    if PDF_USE_HTMLBOX or has_semantic_superscript:
        # No-shrink policy: insert_htmlbox with scale_low near 1 writes
        # nothing when the text cannot fit (returns spare=-1 instead of
        # raising), so we can safely walk the rect ladder on the real
        # page and only fall back to scaling when it is exhausted.
        ladder_exc = None
        # Table rules and neighboring cells remain fixed. Let a
        # translated cell scale inside its original geometry; a
        # downward extension would turn the row into overlapping
        # text while the table lines stay in place.
        ladder = (
            [("base", fitz.Rect(render_rect))]
            if is_table_cell
            else _build_htmlbox_rect_ladder(
                render_rect,
                out_page.rect,
                fontsize,
                prefer_widen=bool(elem.get("single_line_heading")),
            )
        )
        if not is_table_cell:
            ladder = _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
                ladder,
                render_rect,
                elements,
                i,
                fontsize,
                ignored_indices=watermark_indices,
            )
        ladder = _filter_pdf_formula_safe_rects(
            ladder,
            protected_formula_rects,
            anchor_rect=render_rect,
        )
        if not ladder:
            raise PDFPageRenderError(
                page_num + 1,
                i,
                "no formula-safe htmlbox rectangle",
            )
        textbox_fit_rects = [fitz.Rect(candidate_rect) for _, candidate_rect in ladder]
        for step_name, candidate_rect in ladder:
            try:
                spare, scale = out_page.insert_htmlbox(
                    candidate_rect,
                    box_html,
                    css=css,
                    archive=font_archive,
                    scale_low=0.98,
                )
            except Exception as exc:
                ladder_exc = exc
                break
            if spare >= 0:
                inserted = True
                pdf_audit["htmlbox_placement_events"].append(
                    {
                        "page": page_num + 1,
                        "element": i,
                        "step": step_name,
                        "rect": [round(float(value), 3) for value in candidate_rect],
                        "source_rect": [
                            round(float(value), 3) for value in render_rect
                        ],
                        "scale": 1.0,
                        "neighbor_constrained": "neighbor-trim" in step_name,
                        "formula_trimmed": "formula-trim" in step_name,
                        "table_cell": is_table_cell,
                        "text": _plain_text(text_to_insert)[:160],
                    }
                )
                if step_name != "base":
                    log.info(
                        f"[{task_id}] Page {page_num + 1} elem {i}: htmlbox kept fontsize via {step_name}"
                    )
                break
        if inserted:
            return PDFHTMLBoxResult(True, htmlbox_err, render_rect, textbox_fit_rects)
        if ladder_exc is None:
            try:
                spare, scale = _insert_pdf_htmlbox_at_readable_scale(
                    out_page,
                    ladder[-1][1],
                    box_html,
                    css=css,
                    archive=font_archive,
                )
                if spare >= 0:
                    scale = float(scale)
                    scale_step_name, scale_rect = ladder[-1]
                    scale_event = {
                        "page": page_num + 1,
                        "element": i,
                        "scale": round(scale, 4),
                        "source_fontsize": round(fontsize, 3),
                        "rendered_fontsize": round(fontsize * scale, 3),
                        "table_cell": is_table_cell,
                        "redraw_unchanged_source": i in redraw_source_indices,
                        "text": _plain_text(text_to_insert)[:160],
                    }
                    if scale < PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE:
                        # ``scale_low`` promises that a lower scale
                        # is not drawn. If PyMuPDF ever violates
                        # that contract, abort the assembly because
                        # a fallback would otherwise double-write.
                        raise PDFPageRenderError(
                            page_num + 1,
                            i,
                            (
                                f"unreadable htmlbox scale {scale:.2f} below "
                                f"minimum {PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE:.2f}"
                            ),
                            retryable=False,
                        )
                    pdf_audit["render_scale_events"].append(scale_event)
                    pdf_audit["htmlbox_placement_events"].append(
                        {
                            "page": page_num + 1,
                            "element": i,
                            "step": scale_step_name,
                            "rect": [round(float(value), 3) for value in scale_rect],
                            "source_rect": [
                                round(float(value), 3) for value in render_rect
                            ],
                            "scale": round(scale, 4),
                            "neighbor_constrained": "neighbor-trim" in scale_step_name,
                            "formula_trimmed": "formula-trim" in scale_step_name,
                            "table_cell": is_table_cell,
                            "text": _plain_text(text_to_insert)[:160],
                        }
                    )
                    log.warning(
                        f"[{task_id}] Page {page_num + 1} elem {i}: htmlbox scaled to {scale:.2f} "
                        "after extension ladder exhausted"
                    )
                    return PDFHTMLBoxResult(
                        True, htmlbox_err, render_rect, textbox_fit_rects
                    )
                rejected_scale = float(scale)
                if rejected_scale > 0:
                    pdf_audit["rejected_render_scale_events"].append(
                        {
                            "page": page_num + 1,
                            "element": i,
                            "scale": round(rejected_scale, 4),
                            "source_fontsize": round(fontsize, 3),
                            "rendered_fontsize": round(
                                fontsize * rejected_scale,
                                3,
                            ),
                            "table_cell": is_table_cell,
                            "redraw_unchanged_source": i in redraw_source_indices,
                            "inserted": False,
                            "text": _plain_text(text_to_insert)[:160],
                        }
                    )
                    log.info(
                        f"[{task_id}] Page {page_num + 1} elem {i}: "
                        f"htmlbox would require scale {rejected_scale:.2f}; "
                        "no HTML was inserted, using textbox fallback"
                    )
                htmlbox_err = RuntimeError(
                    f"htmlbox no-fit even at free scale (scale={scale:.2f})"
                )
            except PDFPageRenderError:
                raise
            except Exception as exc:
                htmlbox_err = exc
        else:
            htmlbox_err = ladder_exc
        render_rect = ladder[-1][1]
    return PDFHTMLBoxResult(False, htmlbox_err, render_rect, textbox_fit_rects)


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
    task_id = context.task_id
    page_num = plan.page_num
    PDF_USE_HTMLBOX = context.use_htmlbox
    PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE = context.minimum_htmlbox_scale
    out_page = plan.out_page
    elements = plan.elements
    watermark_indices = plan.watermark_indices
    protected_formula_rects = plan.protected_formula_rects
    redraw_source_indices = plan.redraw_source_indices
    pdf_audit = context.pdf_audit
    font_path = context.font_path
    log = context.logger
    _filter_pdf_formula_safe_rects = context.filter_formula_safe_rects
    elem = element_plan.element
    render_style = element_plan.render_style
    source_paragraphs = element_plan.source_paragraphs
    render_rect = element_plan.render_rect
    fontsize = element_plan.fontsize
    r_val = round(element_plan.color[0] * 255)
    g_val = round(element_plan.color[1] * 255)
    b_val = round(element_plan.color[2] * 255)
    text_to_insert = element_plan.text
    lh = element_plan.line_height
    is_table_cell = element_plan.table_cell
    has_semantic_superscript = element_plan.has_semantic_superscript
    htmlbox_result = _try_htmlbox(context, plan, element_plan, css)
    if htmlbox_result.inserted:
        return
    htmlbox_err = htmlbox_result.error
    render_rect = htmlbox_result.render_rect
    textbox_fit_rects = htmlbox_result.textbox_fit_rects

    if has_semantic_superscript:
        if _try_restore_source_element(
            context,
            out_page,
            page_num,
            protected_formula_rects,
            i,
            elem,
            fitz.Rect(elem.get("bbox", elem.get("rect"))),
            fontsize,
            (r_val / 255.0, g_val / 255.0, b_val / 255.0),
            lh,
            f"semantic superscript could not be rendered: {htmlbox_err}",
        ):
            return
        raise PDFPageRenderError(
            page_num + 1,
            i,
            f"semantic superscript could not be rendered: {htmlbox_err}",
        )

    if not PDF_USE_HTMLBOX:
        textbox_ladder = [("base", fitz.Rect(render_rect))]
        if not is_table_cell and not render_style.get("preserve_source_style"):
            expanded_render_rect = _expand_pdf_htmlbox_rect(
                render_rect,
                out_page.rect,
                text_to_insert,
                fontsize,
                source_paragraphs,
                aggressive=True,
            )
            textbox_ladder.append(("legacy-expand", expanded_render_rect))
        if not is_table_cell:
            textbox_ladder = _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
                textbox_ladder,
                render_rect,
                elements,
                i,
                fontsize,
                ignored_indices=watermark_indices,
            )
        textbox_ladder = _filter_pdf_formula_safe_rects(
            textbox_ladder,
            protected_formula_rects,
            anchor_rect=render_rect,
        )
        if not textbox_ladder:
            raise PDFPageRenderError(
                page_num + 1,
                i,
                "no neighbor-safe textbox rectangle",
            )
        render_rect = textbox_ladder[-1][1]
        textbox_fit_rects = [
            fitz.Rect(candidate_rect) for _, candidate_rect in textbox_ladder
        ]

    align_name = source_paragraphs[0].get("text_align", "left")
    align_code = {"left": 0, "center": 1, "right": 2, "justify": 3}.get(align_name, 0)
    fallback_text = _plain_text(_normalize_pdf_translation(text_to_insert)).strip()
    if not fallback_text:
        # The ink is already redacted; restore source instead of leaving a
        # silent blank when the source itself had visible text.
        source_plain = _plain_text(
            elem.get("rich_content") or elem.get("content", "")
        ).strip()
        if source_plain:
            reason = f"empty textbox fallback after htmlbox failure: {htmlbox_err}"
            if not _try_restore_source_element(
                context,
                out_page,
                page_num,
                protected_formula_rects,
                i,
                elem,
                fitz.Rect(elem.get("bbox", elem.get("rect"))),
                fontsize,
                (r_val / 255.0, g_val / 255.0, b_val / 255.0),
                lh,
                reason,
            ):
                raise PDFPageRenderError(page_num + 1, i, reason)
            return
        log.warning(f"[{task_id}] Page {page_num + 1} elem {i}: empty textbox fallback")
        return

    source_plain = re.sub(
        r"\s+",
        " ",
        _plain_text(elem.get("rich_content") or elem.get("content", "")),
    ).strip()
    fixed_short_label = _plain_text(
        _translate_pdf_fixed_short_label(source_plain)
    ).strip()
    compact_fixed_table_label = bool(
        is_table_cell
        and fixed_short_label != source_plain
        and fallback_text == fixed_short_label
        and "\n" not in fallback_text
    )

    ok, textbox_msg, textbox_fontsize = _insert_pdf_textbox_with_fit(
        out_page,
        render_rect,
        fallback_text,
        align=align_code,
        color=(r_val / 255.0, g_val / 255.0, b_val / 255.0),
        fontfile=font_path,
        fontsize=fontsize,
        lineheight_points=lh,
        protected_rects=protected_formula_rects,
        allow_compact_fontsize=compact_fixed_table_label,
        fit_rects=textbox_fit_rects,
    )
    if not ok:
        if _try_restore_source_element(
            context,
            out_page,
            page_num,
            protected_formula_rects,
            i,
            elem,
            fitz.Rect(elem.get("bbox", elem.get("rect"))),
            fontsize,
            (r_val / 255.0, g_val / 255.0, b_val / 255.0),
            lh,
            f"textbox fallback failed: {htmlbox_err}; {textbox_msg}",
        ):
            return
        raise PDFPageRenderError(
            page_num + 1,
            i,
            f"textbox fallback failed: {htmlbox_err}; {textbox_msg}",
        )
    if textbox_fontsize is not None:
        textbox_scale = textbox_fontsize / max(fontsize, 1.0)
        if textbox_scale < PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE:
            raise PDFPageRenderError(
                page_num + 1,
                i,
                (
                    f"unreadable textbox scale {textbox_scale:.2f} below "
                    f"minimum {PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE:.2f}"
                ),
                retryable=False,
            )
        pdf_audit["textbox_fallback_events"].append(
            {
                "page": page_num + 1,
                "element": i,
                "rect": [round(float(value), 3) for value in render_rect],
                "scale": round(textbox_scale, 4),
                "source_fontsize": round(fontsize, 3),
                "rendered_fontsize": round(textbox_fontsize, 3),
                "table_cell": is_table_cell,
                "redraw_unchanged_source": i in redraw_source_indices,
                "reason": textbox_msg,
                "text": _plain_text(text_to_insert)[:160],
            }
        )
    if "fallback" in textbox_msg or "fit after resize" in textbox_msg:
        log.info(
            f"[{task_id}] Page {page_num + 1} elem {i}: textbox fallback adjusted: {textbox_msg}"
        )


def _assemble_page(context: PDFAssemblyContext, page_num: int) -> None:
    """Assemble one output page and checkpoint HTML-box work when configured."""

    total_pages = context.total_pages
    pdf_password = context.pdf_password
    PDF_USE_HTMLBOX = context.use_htmlbox
    assembly_work_path = context.assembly_work_path
    out_doc = context.out_doc
    _trim_process_memory = context.trim_process_memory
    plan = _prepare_page(context, page_num)
    if plan is None:
        return
    render_indices = plan.render_indices

    # Remove original text of translated/merged-away/watermark blocks.
    # True redaction (text-only, keeping images and line art such as
    # footnote separator rules) beats painting white boxes: nothing is
    # covered, and copy/search on the output no longer hits the hidden
    # source-language layer. Whiteout remains as the fallback.
    css = _redact_page(context, plan)

    for i in sorted(render_indices):
        _render_element(context, plan, css, i)

    if PDF_USE_HTMLBOX and assembly_work_path:
        out_doc.saveIncr()
        out_doc.close()
        out_doc = None
        context.out_doc = None
        _trim_process_memory()
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


def assemble_document_pages(context: PDFAssemblyContext) -> None:
    """Render every translated page and retain the current output document."""

    for page_num in range(context.total_pages):
        _assemble_page(context, page_num)


__all__ = ["PDFAssemblyContext", "assemble_document_pages"]
