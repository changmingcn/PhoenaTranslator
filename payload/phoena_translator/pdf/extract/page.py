"""Extract one page into ordered, semantically merged elements: native text
lines, table and image regions, then the page merge passes and their
invariants."""

from __future__ import annotations

import logging
import re

import fitz

from phoena_translator.pdf.extract.baselines import (
    _normalize_pdf_same_baseline_prose_fragments,
)
from phoena_translator.pdf.extract.distribution_codes import _token_conditions
from phoena_translator.pdf.extract.filters import (
    _mark_pdf_embedded_thumbnail_text_elements,
    _sanitize_pdf_hidden_text_artifact_spans,
)
from phoena_translator.pdf.extract.math_detection import (
    _demote_pdf_wrapped_identifier_math_lines,
    _make_pdf_formula_element_from_lines,
    _pdf_line_math_evidence,
    _propagate_pdf_math_line_context,
)
from phoena_translator.pdf.extract.paragraphs import (
    _detect_pdf_drop_cap_span,
    _looks_like_split_layout_line,
    _make_pdf_text_element_from_lines,
    _merge_adjacent_heading_elements,
    _merge_pdf_list_marker_clusters,
    _normalize_pdf_drop_cap_lines,
    _split_pdf_disjoint_line_segments,
    _split_pdf_text_elements_into_semantic_fragments,
)
from phoena_translator.pdf.extract.signatures import _pdf_formula_region_signature
from phoena_translator.pdf.extract.superscripts import _mark_pdf_superscript_spans
from phoena_translator.pdf.extract.tables import _find_pdf_table_regions
from phoena_translator.pdf.extract.vector_ocr import _extract_pdf_vector_ocr_elements
from phoena_translator.pdf.geometry import (
    _cluster_pdf_line_spans,
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
    _line_overlaps_pdf_table_rects,
    _pdf_rotation_from_direction,
    _split_pdf_line_layout_cells,
)
from phoena_translator.pdf.model import DEFAULT_FONTSIZE, ElementType, PdfElement, SkipReason
from phoena_translator.pdf.semantics.continuations import (
    _merge_pdf_detached_list_marker_elements,
    _merge_pdf_semantic_continuation_elements,
    _merge_pdf_strong_continuation_elements,
    _merge_pdf_wrapped_line_elements,
    _pdf_internal_paragraph_lead_residuals,
    _pdf_strong_continuation_residual_pairs,
)
from phoena_translator.pdf.semantics.entities import _mark_pdf_entity_directory_elements
from phoena_translator.pdf.semantics.formulae import (
    _merge_pdf_wrapped_inline_math_continuation_fragments,
    _promote_pdf_formula_overlaps,
    _split_pdf_formula_adjacent_paragraph_elements,
)
from phoena_translator.pdf.semantics.layout_classifier import (
    _classify_pdf_page_text_elements,
)
from phoena_translator.pdf.semantics.references import (
    _complete_pdf_citation_separator_inline_math_fragments,
    _mark_pdf_reference_entry_elements,
    _merge_pdf_reference_entry_elements,
    _pdf_reference_continuation_residual_pairs,
    _pdf_reference_entry_structure_residuals,
    _split_pdf_reference_entry_elements,
)
from phoena_translator.pdf.semantics.tables import _merge_pdf_semantic_table_cells
from phoena_translator.pdf.text_shape import (
    _collapsed_plain_text,
    _ends_with_sentence_boundary,
    _looks_like_heading_text,
    _pick_pdf_dominant_value,
)

# Some generators (observed: Apache FOP 2.7) stroke every line of body text
# twice at the same coordinates.  The two copies are never bit-identical --
# measured drift is under a twentieth of a point -- so duplicate detection
# needs a tolerance rather than an equality test on the bbox.
PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE = 0.25
log = logging.getLogger("translator")
# Postconditions of the page merge passes: (residual finder, failure text).
_PAGE_SEMANTIC_INVARIANTS = (
    (
        _pdf_strong_continuation_residual_pairs,
        "semantic paragraph coalescing invariant failed "
        "({count} unmerged strong continuation pair(s))",
    ),
    (
        lambda elements, _page_rect: _pdf_internal_paragraph_lead_residuals(elements),
        "semantic paragraph boundary invariant failed "
        "({count} embedded successor lead(s))",
    ),
    (
        _pdf_reference_continuation_residual_pairs,
        "reference-entry coalescing invariant failed "
        "({count} unmerged hanging continuation pair(s))",
    ),
    (
        lambda elements, _page_rect: _pdf_reference_entry_structure_residuals(elements),
        "reference-entry semantic invariant failed "
        "({count} element(s) still contain multiple bibliography entries)",
    ),
)


def _extract_pdf_table_elements(
    page,
) -> tuple[list[dict], list[fitz.Rect], list[dict]]:
    """Return table placeholders and their geometry for one page."""
    elements: list[dict] = []
    table_rects: list[fitz.Rect] = []
    table_regions: list[dict] = []
    try:
        table_regions = _find_pdf_table_regions(page)
        for region in table_regions:
            rect = fitz.Rect(region["rect"])
            table_rects.append(rect)
            elements.append({
                "type": ElementType.TABLE_IMAGE,
                "y": rect.y0,
                "rect": rect,
                "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
            })
    except Exception as exc:
        log.warning(f"Table detection failed: {exc}")
    return elements, table_rects, table_regions


def _extract_pdf_image_elements(
    page,
    table_rects: list[fitz.Rect],
) -> tuple[list[dict], list[fitz.Rect]]:
    """Return image placeholders, excluding images substantially inside tables."""
    elements: list[dict] = []
    image_rects: list[fitz.Rect] = []
    try:
        for image_info in page.get_images(full=True):
            image_rects_for_xref = page.get_image_rects(image_info[0])
            if not image_rects_for_xref:
                continue
            try:
                for image_rect in image_rects_for_xref:
                    rect = fitz.Rect(image_rect)
                    image_rects.append(fitz.Rect(rect))
                    overlaps_table = any(
                        not (intersection := rect & table_rect).is_empty
                        and (intersection.width * intersection.height)
                        / max(rect.width * rect.height, 1)
                        > 0.3
                        for table_rect in table_rects
                    )
                    if not overlaps_table:
                        elements.append({
                            "type": ElementType.IMAGE,
                            "y": rect.y0,
                            "rect": rect,
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        })
            except Exception as image_exc:
                log.warning(
                    "Page %s: skipped unreadable image placement: %s",
                    page.number + 1,
                    image_exc,
                )
                continue
    except Exception as exc:
        log.warning(
            "Page %s: image enumeration failed; continuing without image "
            "placeholders: %s",
            page.number + 1,
            exc,
        )
    return elements, image_rects


def _pdf_span_entry(span: dict) -> dict | None:
    """Normalize a PyMuPDF span into the internal rich-text representation."""
    text = span["text"]
    if not text:
        return None
    span_bold = bool(span.get("flags", 0) & (1 << 4))
    font_name = span.get("font", "").lower()
    is_bold = (
        span_bold
        or "bold" in font_name
        or "heavy" in font_name
        or "black" in font_name
    )
    return {
        "text": text,
        "font": str(span.get("font", "")),
        "x0": float(span["bbox"][0]),
        "x1": float(span["bbox"][2]),
        "y0": float(span["bbox"][1]),
        "y1": float(span["bbox"][3]),
        "size": float(span["size"]),
        "color": span["color"],
        "flags": int(span.get("flags", 0)),
        "origin": tuple(span.get("origin", (span["bbox"][0], span["bbox"][3]))),
        "bold": is_bold,
        "invisible": _pdf_span_invisible(span),
    }


# A page whose text is mostly invisible is a scan with a text layer: that layer
# is what gets translated, so invisible text is marked only below this share
# of the page's non-blank span characters (counted before any merge).
PDF_INVISIBLE_TEXT_SCAN_SHARE = 0.5
# Text trace occurrence types (``page.get_texttrace()``): a fill, a stroke,
# and the invisible render mode.  Clipping is not reported by the trace: a
# text clip without fill or stroke (render mode 7) leaves no occurrence, and
# fill or stroke plus clip (modes 4-6) reads as the fill or stroke alone.
_PDF_TRACE_FILL = 0
_PDF_TRACE_INVISIBLE = 3
# The lexemes of a content stream (ISO 32000-1, 7.2): white space, a comment,
# the ``(`` that opens a literal string (read to its end by
# ``_pdf_literal_string_end``), a dictionary delimiter, a hexadecimal string,
# an array or brace delimiter, a name, and a run of regular characters
# (anything but white space and the delimiters ``()<>[]{}/%``).  What none
# of them matches -- ``)`` or ``>`` alone, a ``<`` that opens no valid
# hexadecimal string -- is a state the lexer cannot follow.
_PDF_REGULAR_CHAR = rb"[^\x00\t\n\x0c\r ()<>\[\]{}/%]"
_PDF_LEXEME_RE = re.compile(
    rb"(?P<space>[\x00\t\n\x0c\r ]+)"
    rb"|(?P<comment>%[^\r\n]*)"
    rb"|(?P<string>\()"
    rb"|(?P<dictionary><<|>>)"
    rb"|(?P<hex><[0-9A-Fa-f\x00\t\n\x0c\r ]*>)"
    rb"|(?P<bracket>[\[\]{}])"
    rb"|(?P<name>/" + _PDF_REGULAR_CHAR + rb"*)"
    rb"|(?P<regular>" + _PDF_REGULAR_CHAR + rb"+)"
)
# Within a literal string, what changes the lexer's state: an escape, and
# the parentheses that nest.
_PDF_LITERAL_STRING_STOP_RE = re.compile(rb"[\\()]")
_PDF_NUMBER_RE = re.compile(rb"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)")
_PDF_INTEGER_RE = re.compile(rb"[+-]?[0-9]+")
# A regular token beginning with one of these and not a number is one a
# reader may split (MuPDF reads ``4Tr`` as an unknown keyword, a reader whose
# number lexer stops at the first letter reads ``4`` and ``Tr``).
_PDF_NUMBER_START = b"+-.0123456789"
_PDF_OPERAND_KEYWORDS = frozenset({b"true", b"false", b"null"})
# The non-clipping text render modes: fill, stroke, fill and stroke, invisible.
_PDF_NON_CLIPPING_RENDER_MODES = range(4)
# Grid (points) on which glyph origins of the trace and of the extraction
# are matched; a neighbouring grid cell matches too.
_PDF_GLYPH_ORIGIN_GRID = 0.05


def _pdf_trace_occurrence_invisible(occurrence: dict) -> bool:
    """Whether one paint occurrence of the text trace shows nothing: the
    invisible render mode, or a fill with fill opacity exactly 0.  A stroke
    is visible whatever the span flags say."""
    try:
        kind = int(occurrence.get("type", _PDF_TRACE_FILL))
        opacity = float(occurrence.get("opacity", 1.0))
    except (TypeError, ValueError):
        return False
    return kind == _PDF_TRACE_INVISIBLE or (kind == _PDF_TRACE_FILL and opacity == 0.0)


def _pdf_glyph_cell(x: float, y: float) -> tuple[int, int]:
    return (round(x / _PDF_GLYPH_ORIGIN_GRID), round(y / _PDF_GLYPH_ORIGIN_GRID))


def _pdf_text_paint_record(page) -> dict | None:
    """Per glyph place (character, origin) of the text trace, whether one of
    its paint occurrences is visible and whether one is invisible.  ``None``
    when the trace has no invisible occurrence (nothing to mark) or cannot be
    read (no evidence: nothing is marked)."""
    try:
        trace = page.get_texttrace()
    except Exception as exc:  # the trace is evidence only, never required
        log.debug("Page %s: text trace unavailable: %s", page.number + 1, exc)
        return None
    record: dict[tuple, list[bool]] = {}
    any_invisible = False
    for occurrence in trace or ():
        invisible = _pdf_trace_occurrence_invisible(occurrence)
        any_invisible = any_invisible or invisible
        for char in occurrence.get("chars") or ():
            try:
                key = (int(char[0]), *_pdf_glyph_cell(float(char[2][0]), float(char[2][1])))
            except (TypeError, ValueError, IndexError):
                continue
            seen = record.setdefault(key, [False, False])
            seen[1 if invisible else 0] = True
    return record if any_invisible else None


def _pdf_glyph_painted_invisibly(record: dict, char: dict) -> bool:
    """A glyph is invisible when the trace paints it at its place, every
    such occurrence is invisible, and no visible occurrence shares the place."""
    text = char.get("c") or ""
    origin = char.get("origin")
    if len(text) != 1 or not origin:
        return False
    column, row = _pdf_glyph_cell(float(origin[0]), float(origin[1]))
    found = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            seen = record.get((ord(text), column + dx, row + dy))
            if seen is None:
                continue
            if seen[0]:
                return False
            found = found or seen[1]
    return found


def _pdf_literal_string_end(stream: bytes, position: int) -> int | None:
    """The position after the literal string whose ``(`` ends at ``position``:
    balanced parentheses nest inside it, a backslash escapes the byte after
    it.  ``None`` when the stream ends first (an unbalanced string)."""
    depth = 1
    while depth:
        stop = _PDF_LITERAL_STRING_STOP_RE.search(stream, position)
        if stop is None:
            return None
        position = stop.end()
        char = stream[stop.start()]
        if char == 0x5C:  # backslash
            position += 1
        elif char == 0x28:  # (
            depth += 1
        else:
            depth -= 1
    return position


def _pdf_render_mode_operand_is_non_clipping(
    operand: tuple[str, bytes] | None, before: tuple[str, bytes] | None
) -> bool:
    """Whether ``Tr`` with the token ``operand`` before it, and ``before``
    before that, selects a non-clipping mode with certainty: exactly one
    operand (``before`` is an operator, or the stream begins; with two,
    MuPDF takes the first), and it is an integer whose value is 0-3 (a real
    is not: readers truncate it)."""
    return (
        operand is not None
        and operand[0] == "number"
        and _PDF_INTEGER_RE.fullmatch(operand[1]) is not None
        and int(operand[1]) in _PDF_NON_CLIPPING_RENDER_MODES
        and (before is None or before[0] == "operator")
    )


def _pdf_stream_text_modes_are_non_clipping(stream: bytes) -> bool:
    """Whether every text render mode operator in a content stream is read
    with certainty as selecting a non-clipping mode (0-3).

    A small lexer (``_PDF_LEXEME_RE``) reads the stream and skips white
    space, comments, literal strings and hexadecimal strings, so letters
    inside them are never taken for the operator; ``Tr`` counts only as an
    operator token, with its operand token before it
    (``_pdf_render_mode_operand_is_non_clipping``).  Uncertain, and so
    ``False``: an operand that does not read as a non-clipping mode; a token
    that begins like a number but is not one (``4Tr``); an inline image
    (``BI``: its data cannot be skipped with certainty); an unbalanced
    string or anything else the lexer cannot follow.  The lexer stops after
    the stream's last occurrence of the bytes ``Tr``: nothing beyond it can
    be the operator, so what follows there (an inline image at the end of
    the stream, say) cannot hide one; a stream without them has none."""
    final = stream.rfind(b"Tr")
    position = 0
    last = before = None  # (kind, token) of the two latest tokens
    while position <= final:
        found = _PDF_LEXEME_RE.match(stream, position)
        if found is None:
            return False
        kind, token = found.lastgroup, found.group()
        position = found.end()
        if kind in ("space", "comment"):
            continue
        if kind == "string":
            position = _pdf_literal_string_end(stream, position)
            if position is None:
                return False
        elif kind == "regular":
            if _PDF_NUMBER_RE.fullmatch(token):
                kind = "number"
            elif token[0] in _PDF_NUMBER_START:
                return False
            elif token not in _PDF_OPERAND_KEYWORDS:
                kind = "operator"
                if token == b"BI":
                    return False
                if token == b"Tr" and not _pdf_render_mode_operand_is_non_clipping(last, before):
                    return False
        last, before = (kind, token), last
    return True


def _pdf_page_clips_with_text(page) -> bool:
    """Whether the page may clip with text: true unless every text render mode
    operator in the page's content (its streams read as one) and in each of
    its form XObjects is read with certainty as carrying a non-clipping mode
    (0-3; ``_pdf_stream_text_modes_are_non_clipping``).  Uncertain evidence
    -- an operand that does not read as one of those modes, an inline image,
    a state the lexer cannot follow, content that cannot be read -- counts
    as clipping, so nothing on the page is marked invisible."""
    try:
        doc = page.parent
        streams = [b"\n".join(doc.xref_stream(xref) or b"" for xref in page.get_contents())]
        # ``get_xobjects`` lists the page's form XObjects, nested ones
        # included (never images).
        streams += [doc.xref_stream(item[0]) or b"" for item in page.get_xobjects()]
        return not all(_pdf_stream_text_modes_are_non_clipping(stream) for stream in streams)
    except Exception:
        return True


def _annotate_pdf_span_paint_visibility(page, page_dict: dict) -> None:
    """Record on every span of ``page_dict`` whether the page paints none of
    its glyphs visibly (``paint_invisible``), read per paint occurrence from
    the text trace.  The extraction's own span flags lose evidence: a stroke
    over a transparent fill, and a transparent and an opaque copy of a glyph
    at one place.  Nothing is recorded invisible on a page that clips with
    text (the trace does not report the clip)."""
    record = _pdf_text_paint_record(page)
    if record is None:
        return
    raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    candidates = []
    for block, raw_block in zip(page_dict.get("blocks", []), raw.get("blocks", [])):
        if block.get("type") != 0 or raw_block.get("type") != 0:
            continue
        for line, raw_line in zip(block.get("lines", []), raw_block.get("lines", [])):
            for span, raw_span in zip(line.get("spans", []), raw_line.get("spans", [])):
                chars = raw_span.get("chars") or []
                if "".join(char.get("c", "") for char in chars) != span.get("text", ""):
                    continue
                shown = [char for char in chars if (char.get("c") or "").strip()]
                if shown and all(_pdf_glyph_painted_invisibly(record, char) for char in shown):
                    candidates.append(span)
    if candidates and not _pdf_page_clips_with_text(page):
        for span in candidates:
            span["paint_invisible"] = True


def _pdf_span_invisible(span: dict) -> bool:
    """Whether the source page shows none of the span's glyphs, as recorded
    from the text trace (``_annotate_pdf_span_paint_visibility``).  Without
    that record a span is visible."""
    return bool(span.get("paint_invisible"))


def _pdf_page_is_scan_layer(page_dict: dict) -> bool:
    """Whether invisible text is ``PDF_INVISIBLE_TEXT_SCAN_SHARE`` or more of
    the page's non-blank span characters, counted on the spans as extracted
    (before any merge): a scan with a text layer."""
    total = invisible = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                count = len("".join(str(span.get("text", "")).split()))
                total += count
                if _pdf_span_invisible(span):
                    invisible += count
    return bool(total) and invisible >= PDF_INVISIBLE_TEXT_SCAN_SHARE * total


def _pdf_spans_invisible(spans: list[dict]) -> bool:
    """Every non-blank span is invisible (and there is one)."""
    shown = [span for span in spans if str(span.get("text", "")).strip()]
    return bool(shown) and all(span.get("invisible") for span in shown)


def _make_pdf_split_layout_element(
    *,
    page_rect: fitz.Rect,
    line: dict,
    cell: dict,
    line_fontsize: float,
    line_in_table: bool,
    line_non_horizontal: bool,
    cluster_math: dict,
) -> dict | None:
    """Build one semantic cell from a visually split PDF line."""
    cluster = cell["cluster"]
    cluster_plain = re.sub(
        r"\s+",
        " ",
        "".join(span["text"] for span in cluster),
    ).strip()
    if not cluster_plain:
        return None
    cluster_rich = "".join(span["rich"] for span in cluster).strip()
    cluster_x0 = float(cell["x0"])
    cluster_x1 = float(cell["x1"])
    cluster_y0 = min(span["y0"] for span in cluster)
    cluster_y1 = max(span["y1"] for span in cluster)
    bold_chars = sum(len(span["text"].strip()) for span in cluster if span["bold"])
    total_chars = sum(len(span["text"].strip()) for span in cluster)
    superscript_runs = _pdf_cluster_superscript_runs(cluster)
    rect = fitz.Rect(cluster_x0, cluster_y0, cluster_x1, cluster_y1)
    return {
        "type": ElementType.TEXT,
        "y": cluster_y0,
        "x": cluster_x0,
        "rect": rect,
        "bbox": [cluster_x0, cluster_y0, cluster_x1, cluster_y1],
        "render_bbox": _get_pdf_render_bbox(
            rect,
            page_rect,
            [{"plain": cluster_plain, "text_align": cell["align"], "nowrap": True}],
            line_fontsize,
        ),
        "content": cluster_plain,
        "rich_content": (
            cluster_rich
            if superscript_runs or 0 < bold_chars < total_chars * 0.9
            else None
        ),
        "superscript_runs": superscript_runs,
        "superscript_scale": _pdf_cluster_superscript_scale(superscript_runs),
        "paragraphs": [{
            "plain": cluster_plain,
            "rich": cluster_rich,
            "margin_left": 0.0,
            "text_indent": 0.0,
            "gap_before": 0.0,
            "text_align": cell["align"],
            "nowrap": True,
        }],
        "top_padding": max(0.0, cluster_y0 - float(line["bbox"][1])),
        "fontsize": _most_common_pdf_value([span["size"] for span in cluster], line_fontsize),
        "line_height": max(cluster_y1 - cluster_y0, line_fontsize * 1.05),
        "color": _most_common_pdf_value([span["color"] for span in cluster], 0),
        "bold": bold_chars > total_chars * 0.5 if total_chars > 0 else False,
        "non_horizontal": line_non_horizontal,
        "rotation": _pdf_rotation_from_direction(line.get("dir")),
        "inline_math_fragments": [
            dict(fragment)
            for fragment in (cluster_math.get("inline_math_fragments") or [])
        ],
        "preserve_source_style": bool(line_in_table),
        "table_hint": bool(line_in_table),
    }


def _pdf_cluster_superscript_runs(cluster: list[dict]) -> list[dict]:
    return [
        {
            "text": span["text"].strip(),
            "scale": float(span.get("superscript_scale") or 0.60),
            "source": span.get("superscript_source"),
        }
        for span in cluster
        if span.get("superscript") and span.get("text", "").strip()
    ]


def _pdf_cluster_superscript_scale(runs: list[dict]) -> float | None:
    """The text-weighted dominant superscript scale, or ``None`` without runs."""
    if not runs:
        return None
    return float(
        _pick_pdf_dominant_value(
            [(round(run["scale"], 3), max(len(run["text"]), 1)) for run in runs],
            0.60,
        )
    )


def _most_common_pdf_value(values: list, default):
    return max(set(values), key=values.count) if values else default


def _extract_pdf_line_record(
    line: dict,
    block_rect: fitz.Rect,
    table_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
) -> tuple[list[dict], dict | None]:
    """Extract either split-layout elements or one normal line record."""
    span_entries = [
        entry
        for span in line.get("spans", [])
        if (entry := _pdf_span_entry(span)) is not None
    ]
    if not span_entries:
        return [], None

    _mark_pdf_superscript_spans(span_entries)
    line_rect = fitz.Rect(line["bbox"])
    line_in_table = _line_overlaps_pdf_table_rects(line_rect, table_rects)
    line_math = _pdf_line_math_evidence(
        span_entries,
        table_hint=line_in_table,
    )
    line_superscript_runs = [
        {
            "text": span["text"].strip(),
            "scale": float(span.get("superscript_scale") or 0.60),
            "source": span.get("superscript_source"),
        }
        for span in span_entries
        if span.get("superscript") and span.get("text", "").strip()
    ]
    line_fontsize = float(_pick_pdf_dominant_value(
        [
            (float(span["size"]), max(len(span["text"].strip()), 1))
            for span in span_entries
        ],
        float(span_entries[0]["size"]),
    ))
    line_color = max(
        set(span["color"] for span in span_entries),
        key=[span["color"] for span in span_entries].count,
    )
    line_plain_chars = sum(len(span["text"].strip()) for span in span_entries)
    line_bold_chars = sum(
        len(span["text"].strip()) for span in span_entries if span["bold"]
    )
    line_rotation = _pdf_rotation_from_direction(line.get("dir"))
    line_non_horizontal = line_rotation in {90, 270}
    clusters = _merge_pdf_list_marker_clusters(
        _cluster_pdf_line_spans(span_entries, line_fontsize)
    )
    if _looks_like_split_layout_line(
        clusters,
        line["bbox"],
        block_rect,
        line_fontsize,
    ):
        split_elements: list[dict] = []
        for cell in _split_pdf_line_layout_cells(clusters, line["bbox"]):
            cluster_math = _pdf_line_math_evidence(
                cell["cluster"],
                table_hint=line_in_table,
            )
            if cluster_math["protected"]:
                cluster = cell["cluster"]
                formula_element = _make_pdf_formula_element_from_lines([{
                    "x0": min(span["x0"] for span in cluster),
                    "x1": max(span["x1"] for span in cluster),
                    "y0": min(span["y0"] for span in cluster),
                    "y1": max(span["y1"] for span in cluster),
                    "math_reasons": cluster_math["reasons"],
                    "math_mixed": cluster_math["mixed"],
                    "math_symbol_count": cluster_math["math_symbol_count"],
                }])
                if formula_element:
                    split_elements.append(formula_element)
                continue
            element = _make_pdf_split_layout_element(
                page_rect=page_rect,
                line=line,
                cell=cell,
                line_fontsize=line_fontsize,
                line_in_table=line_in_table,
                line_non_horizontal=line_non_horizontal,
                cluster_math=cluster_math,
            )
            if element:
                if _pdf_spans_invisible(cell["cluster"]):
                    element["source_invisible_text"] = element.get("content", "")
                split_elements.append(element)
        return split_elements, None

    return [], {
        "plain": "".join(span["text"] for span in span_entries),
        "rich": "".join(span["rich"] for span in span_entries),
        "x0": float(line["bbox"][0]),
        "x1": float(line["bbox"][2]),
        "y0": float(line["bbox"][1]),
        "y1": float(line["bbox"][3]),
        "fontsize": line_fontsize,
        "line_height": float(line["bbox"][3]) - float(line["bbox"][1]),
        "color": line_color,
        "bold": (
            line_bold_chars > line_plain_chars * 0.5
            if line_plain_chars > 0
            else False
        ),
        "non_horizontal": line_non_horizontal,
        "rotation": line_rotation,
        "table_hint": bool(line_in_table),
        "drop_cap": _detect_pdf_drop_cap_span(span_entries),
        "superscript_runs": line_superscript_runs,
        "inline_math_fragments": [
            dict(fragment)
            for fragment in (line_math.get("inline_math_fragments") or [])
        ],
        "math_protected": bool(line_math["protected"]),
        "math_reasons": list(line_math["reasons"]),
        "math_mixed": bool(line_math["mixed"]),
        "math_symbol_count": int(line_math["math_symbol_count"]),
        "invisible_plain": (
            "".join(span["text"] for span in span_entries)
            if _pdf_spans_invisible(span_entries)
            else None
        ),
    }


def _group_pdf_block_lines(
    line_infos: list[dict],
) -> list[tuple[tuple[bool, bool], list[dict]]]:
    """Group consecutive lines by table and formula protection state."""
    line_groups: list[tuple[tuple[bool, bool], list[dict]]] = []
    current_group: list[dict] = []
    current_key: tuple[bool, bool] | None = None
    for line_info in line_infos:
        key = (
            bool(line_info.get("table_hint")),
            bool(line_info.get("math_protected")),
        )
        if current_group and key != current_key:
            line_groups.append((current_key, current_group))
            current_group = [line_info]
            current_key = key
            continue
        if not current_group:
            current_key = key
        current_group.append(line_info)
    if current_group and current_key is not None:
        line_groups.append((current_key, current_group))
    return line_groups


def _pdf_bold_run_in_wraps_into_mixed_line(
    heading_lines: list[dict],
    transition_line: dict,
) -> bool:
    """Return whether a bold run-in sentence crosses a visual line boundary.

    A common report layout bolds the opening sentence of a paragraph.  When
    that sentence wraps, the first visual line is entirely bold while the next
    line starts bold and changes to regular text after the sentence-ending
    punctuation.  The generic heading splitter used to treat the fully bold
    first line as a standalone heading, so the two halves of one sentence were
    translated independently.

    Require the mixed line to contain both a leading bold run and a regular
    suffix, and require that the leading bold run completes a sentence.  Those
    signals distinguish a wrapped run-in sentence from an ordinary standalone
    heading followed by body prose.
    """
    if not heading_lines or not transition_line:
        return False
    previous = heading_lines[-1]
    if (
        not previous.get("bold")
        or _ends_with_sentence_boundary(previous.get("plain", ""))
    ):
        return False

    rich_remainder = str(transition_line.get("rich") or "").lstrip()
    bold_fragments: list[str] = []
    while rich_remainder:
        match = re.match(r"(?is)^<b\b[^>]*>(.*?)</b>", rich_remainder)
        if match is None:
            break
        fragment = _collapsed_plain_text(match.group(1))
        if fragment:
            bold_fragments.append(fragment)
        rich_remainder = rich_remainder[match.end():].lstrip()

    bold_prefix = " ".join(bold_fragments).strip()
    regular_suffix = _collapsed_plain_text(rich_remainder)
    if (
        not bold_prefix
        or not regular_suffix
        or not _ends_with_sentence_boundary(bold_prefix)
    ):
        return False

    previous_size = max(float(previous.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    transition_size = max(float(transition_line.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    if abs(previous_size - transition_size) > max(previous_size * 0.08, 0.7):
        return False

    previous_height = max(
        float(previous.get("y1", 0.0)) - float(previous.get("y0", 0.0)),
        1.0,
    )
    transition_height = max(
        float(transition_line.get("y1", 0.0))
        - float(transition_line.get("y0", 0.0)),
        1.0,
    )
    vertical_gap = float(transition_line.get("y0", 0.0)) - float(
        previous.get("y1", 0.0)
    )
    if vertical_gap < -max(previous_height, transition_height) * 0.35:
        return False
    if vertical_gap > max(previous_size * 0.95, previous_height, transition_height):
        return False

    return abs(
        float(transition_line.get("x0", 0.0)) - float(previous.get("x0", 0.0))
    ) <= max(previous_size * 0.60, 6.0)


def _split_pdf_heading_segments(group_lines: list[dict]) -> list[list[dict]]:
    """Separate a short bold heading glued to following body text."""
    dominant_size = float(_pick_pdf_dominant_value(
        [
            (
                float(line.get("fontsize", DEFAULT_FONTSIZE)),
                max(len(line.get("plain", "").strip()), 1),
            )
            for line in group_lines
        ],
        DEFAULT_FONTSIZE,
    ))
    segments = _split_pdf_disjoint_line_segments(group_lines, dominant_size)
    split_segments: list[list[dict]] = []
    for candidate_lines in segments:
        if len(candidate_lines) < 2:
            split_segments.append(candidate_lines)
            continue
        heading_count = 0
        while (
            heading_count < len(candidate_lines)
            and heading_count < 2
            and not candidate_lines[heading_count].get("drop_cap")
            and (
                candidate_lines[heading_count].get("bold")
                or _looks_like_heading_text(
                    candidate_lines[heading_count].get("plain", "")
                )
            )
        ):
            heading_count += 1
        next_line_is_body = (
            0 < heading_count < len(candidate_lines)
            and not (
                candidate_lines[heading_count].get("bold")
                or _looks_like_heading_text(
                    candidate_lines[heading_count].get("plain", "")
                )
            )
        )
        if next_line_is_body and _pdf_bold_run_in_wraps_into_mixed_line(
            candidate_lines[:heading_count],
            candidate_lines[heading_count],
        ):
            next_line_is_body = False
        if next_line_is_body:
            split_segments.extend([
                candidate_lines[:heading_count],
                candidate_lines[heading_count:],
            ])
        else:
            split_segments.append(candidate_lines)
    return split_segments


def _append_pdf_grouped_line_elements(
    elements: list[dict],
    line_infos: list[dict],
    page_rect: fitz.Rect,
) -> None:
    """Convert normalized line groups into formula or text elements."""
    for (table_hint, math_protected), group_lines in _group_pdf_block_lines(
        line_infos
    ):
        if math_protected:
            formula_element = _make_pdf_formula_element_from_lines(group_lines)
            if formula_element:
                elements.append(formula_element)
            continue
        if table_hint:
            for group_line in group_lines:
                element = _make_pdf_text_element_from_lines(
                    [group_line],
                    page_rect,
                    preserve_source_style=True,
                    table_hint=True,
                )
                if element:
                    _note_pdf_invisible_lines(element, [group_line])
                    elements.append(element)
            continue
        for segment_lines in _split_pdf_heading_segments(group_lines):
            element = _make_pdf_text_element_from_lines(
                segment_lines,
                page_rect,
            )
            if element:
                _note_pdf_invisible_lines(element, segment_lines)
                elements.append(element)


def _note_pdf_invisible_lines(element: dict, lines: list[dict]) -> None:
    """Note an element built only from invisible lines (each line's text is
    still the text its spans had: no normalisation merged a visible line in).

    The note is the element's text at creation; ``_mark_pdf_invisible_source``
    honours it only while the element still holds exactly that text, so a
    later merge with visible text cancels it.
    """
    if lines and all(
        line.get("invisible_plain") is not None
        and _collapsed_plain_text(line["invisible_plain"])
        == _collapsed_plain_text(line.get("plain", ""))
        for line in lines
    ):
        element["source_invisible_text"] = element.get("content", "")


def _mark_pdf_invisible_source(elements: list[dict], *, scan_layer: bool) -> int:
    """Give every element the source page does not show ``INVISIBLE_SOURCE``.

    An element is invisible when it still holds exactly the text noted at its
    creation from invisible spans only.  Not marked: an element that already
    has a skip reason, one shaped like a distribution code (its own detector
    decides, and removes it), and every element of a page that is a scan with
    a text layer (``scan_layer``, decided on the spans before any merge by
    ``_pdf_page_is_scan_layer``), where the layer is translated as before.
    The notes are removed from every element.
    """
    noted = [
        elem
        for elem in elements
        if elem.get("type") == ElementType.TEXT
        and elem.get("source_invisible_text") is not None
        and _collapsed_plain_text(elem.get("content", ""))
        == _collapsed_plain_text(elem["source_invisible_text"])
    ]
    marked = 0
    if not scan_layer:
        for elem in noted:
            if elem.get("skip_translate_reason") or all(_token_conditions(elem).values()):
                continue
            elem["skip_translate_reason"] = SkipReason.INVISIBLE_SOURCE
            marked += 1
    for elem in elements:
        elem.pop("source_invisible_text", None)
    return marked


def _append_pdf_text_block_elements(
    elements: list[dict],
    block: dict,
    page,
    table_rects: list[fitz.Rect],
) -> None:
    """Append all semantic elements represented by one PDF text block."""
    block, removed_hidden_spans = _sanitize_pdf_hidden_text_artifact_spans(
        block,
        fitz.Rect(page.rect),
    )
    if removed_hidden_spans:
        log.info(
            "Page %s: ignored %s oversized hidden text artifact span(s)",
            page.number + 1,
            removed_hidden_spans,
        )
    if not block:
        return
    block_rect = fitz.Rect(block["bbox"])
    line_infos: list[dict] = []
    for line in block.get("lines", []):
        split_elements, line_info = _extract_pdf_line_record(
            line,
            block_rect,
            table_rects,
            fitz.Rect(page.rect),
        )
        elements.extend(split_elements)
        if line_info is not None:
            line_infos.append(line_info)

    line_infos = _normalize_pdf_drop_cap_lines(line_infos, block_rect)
    line_infos = _normalize_pdf_same_baseline_prose_fragments(
        line_infos,
        block_rect,
    )
    _demote_pdf_wrapped_identifier_math_lines(line_infos)
    _propagate_pdf_math_line_context(line_infos)
    if line_infos:
        _append_pdf_grouped_line_elements(
            elements,
            line_infos,
            fitz.Rect(page.rect),
        )


def _drop_pdf_duplicate_glyph_spans(page_dict: dict) -> tuple[dict, int]:
    """Keep one copy of text that the generator stroked twice in place.

    Apache FOP 2.7 emits every body line as two identical spans at the same
    coordinates.  Extraction then produced two overlapping elements per line
    plus stray half-line fragments, and because redaction erases a glyph whose
    box is merely clipped, the overlap-redraw closure in ``rendering`` grew
    until it reached a ``formula_risk_preserved`` element and failed the whole
    page.  One such page then dragged its entire accepted cross-page merge
    component back to untranslated source copies -- 19 pages from a single
    duplicated line.

    Only redundant ink is removed: same text, same font, same size, and the
    same position within ``PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE``.  Text
    repeated elsewhere on the page -- table cells, running heads, a column of
    ``0.00%`` -- sits at a different bbox and survives.

    Colour is deliberately NOT part of the identity, because the observed
    duplication is a white knockout copy at ``0xffffff`` followed by the real
    ``0x231f20`` copy.  The survivor must therefore be the LAST occurrence:
    PDF paints in stream order, so the final copy is the one a reader sees.
    Keeping the first copy instead inherited the white colour and redrew every
    translated paragraph in white on white -- a fully blank page that still
    extracted perfect Chinese text.
    """
    spans_in_order: list[dict] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans_in_order.append(span)

    winner_by_key: dict[tuple, list[list]] = {}
    for position, span in enumerate(spans_in_order):
        text = span.get("text", "")
        if not text.strip():
            continue
        try:
            bbox = tuple(float(value) for value in span.get("bbox", ()))
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        key = (
            text,
            span.get("font", ""),
            round(float(span.get("size", 0.0)), 2),
        )
        for entry in winner_by_key.setdefault(key, []):
            if all(
                abs(bbox[axis] - entry[0][axis])
                <= PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE
                for axis in range(4)
            ):
                entry[1].append(position)
                break
        else:
            winner_by_key[key].append([bbox, [position]])

    discarded: set[int] = set()
    for entries in winner_by_key.values():
        for _bbox, positions in entries:
            discarded.update(positions[:-1])
    if not discarded:
        return page_dict, 0

    # Rebuild with a counter that mirrors the walk above, so a span is matched
    # by its position in stream order rather than by identity or value.
    position = 0
    blocks = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            blocks.append(block)
            continue
        lines = []
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                if position not in discarded:
                    spans.append(span)
                position += 1
            if spans:
                lines.append({**line, "spans": spans})
        if lines:
            blocks.append({**block, "lines": lines})
    return {**page_dict, "blocks": blocks}, len(discarded)


def _extract_pdf_text_elements(
    page,
    table_rects: list[fitz.Rect],
    visibility: dict | None = None,
) -> list[dict]:
    """Extract rich text and formula elements from every text block.

    A ``get_text`` failure must propagate: the extraction stage wraps it into
    ``PDFPageExtractionError`` so the recovery policy preserves the exact
    source page with an audit record, instead of silently emitting a page
    with no text elements.

    ``visibility``, when given, receives ``scan_layer``: the page-level
    decision of ``_pdf_page_is_scan_layer`` on the spans as extracted, before
    any merge, for ``_mark_pdf_invisible_source``.
    """
    elements: list[dict] = []
    page_dict = page.get_text(
        "dict",
        flags=fitz.TEXT_PRESERVE_WHITESPACE,
    )
    _annotate_pdf_span_paint_visibility(page, page_dict)
    page_dict, duplicate_spans = _drop_pdf_duplicate_glyph_spans(page_dict)
    if visibility is not None:
        visibility["scan_layer"] = _pdf_page_is_scan_layer(page_dict)
    if duplicate_spans:
        log.info(
            "Page %s: dropped %s duplicate glyph span(s) stroked in place",
            page.number + 1,
            duplicate_spans,
        )
    dropped_blocks = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        try:
            _append_pdf_text_block_elements(
                elements,
                block,
                page,
                table_rects,
            )
        except Exception as block_exc:
            dropped_blocks += 1
            log.warning(
                "PDF text block extraction failed at "
                f"{block.get('bbox')}: {block_exc}"
            )
    if dropped_blocks:
        log.warning(
            "Page %s: dropped %s unextractable text block(s)",
            page.number + 1,
            dropped_blocks,
        )
    return elements


def _pdf_element_sort_key(element: dict) -> tuple[float, float]:
    bbox = element.get("bbox")
    fallback_x = bbox[0] if isinstance(bbox, list) else 0
    return element["y"], element.get("x", fallback_x)


def _log_page_pass(page, count: int, message: str) -> None:
    if count:
        log.info("Page %s: " + message, page.number + 1, count)


def _apply_reclassifying_pass(page, elements, page_rect, merge, message):
    """Run a ``(elements, count)`` pass; reclassify the page if it changed."""
    elements, count = merge(elements, page_rect)
    if count:
        _classify_pdf_page_text_elements(elements, page_rect)
    _log_page_pass(page, count, message)
    return elements


def _merge_and_validate_pdf_page_semantics(
    page,
    elements: list[dict],
    table_rects: list[fitz.Rect],
    table_regions: list[dict],
) -> list[dict]:
    """Run ordered semantic merges and enforce their postconditions."""
    page_rect = fitz.Rect(page.rect)
    elements.sort(key=_pdf_element_sort_key)
    elements, semantic_table_cell_count = _merge_pdf_semantic_table_cells(
        elements,
        table_rects,
        page_rect,
        table_regions=table_regions,
    )
    _log_page_pass(page, semantic_table_cell_count, "merged %s wrapped semantic table cell(s)")

    # Line-level coalescing before classification.
    elements = _split_pdf_text_elements_into_semantic_fragments(elements, page_rect)
    elements.sort(key=_pdf_element_sort_key)
    elements = _merge_pdf_detached_list_marker_elements(elements, page_rect)
    elements = _merge_pdf_wrapped_line_elements(elements, page_rect)
    elements = _merge_pdf_semantic_continuation_elements(elements, page_rect)
    elements = _merge_adjacent_heading_elements(elements)
    _log_page_pass(
        page,
        _merge_pdf_wrapped_inline_math_continuation_fragments(elements, page_rect),
        "merged %s wrapped inline-math continuation fragment(s)",
    )

    # Post-classification paragraph and bibliography passes.
    _classify_pdf_page_text_elements(elements, page_rect)
    for merge, message in (
        (
            _merge_pdf_strong_continuation_elements,
            "merged %s strong post-classification paragraph continuation(s)",
        ),
        (
            _split_pdf_reference_entry_elements,
            "split %s coupled bibliography entry boundary/boundaries",
        ),
        (
            _merge_pdf_reference_entry_elements,
            "merged %s hanging reference continuation(s)",
        ),
    ):
        elements = _apply_reclassifying_pass(page, elements, page_rect, merge, message)

    for find_residuals, failure in _PAGE_SEMANTIC_INVARIANTS:
        residuals = find_residuals(elements, page_rect)
        if residuals:
            raise RuntimeError(
                f"Page {page.number + 1}: " + failure.format(count=len(residuals))
            )

    # Preservation marks that depend on the final paragraph structure.
    _log_page_pass(
        page,
        _mark_pdf_entity_directory_elements(elements),
        "marked %s aligned entity-directory row(s)",
    )
    _log_page_pass(
        page,
        _split_pdf_formula_adjacent_paragraph_elements(elements, page_rect),
        "isolated %s formula-adjacent paragraph fragment(s)",
    )
    _log_page_pass(
        page,
        _promote_pdf_formula_overlaps(elements),
        "conservatively preserved %s text element(s) touching formula geometry",
    )
    _mark_pdf_reference_entry_elements(elements)
    return elements


def _enrich_pdf_page_elements(
    page,
    elements: list[dict],
    image_rects: list[fitz.Rect],
) -> list[dict]:
    """Add image OCR, citation, and immutable-formula audit metadata."""
    page_rect = fitz.Rect(page.rect)
    embedded_count = _mark_pdf_embedded_thumbnail_text_elements(
        elements,
        image_rects,
        page_rect,
    )
    if embedded_count:
        log.info(
            "Page %s: preserved %s tiny text element(s) inside embedded image "
            "thumbnail(s)",
            page.number + 1,
            embedded_count,
        )

    vector_elements = _extract_pdf_vector_ocr_elements(
        page,
        elements,
        image_rects,
    )
    if vector_elements:
        elements.extend(vector_elements)
        elements.sort(key=lambda element: (
            float(element.get("y", _get_pdf_elem_rect(element).y0)),
            float(element.get("x", _get_pdf_elem_rect(element).x0)),
        ))

    completed_separators = (
        _complete_pdf_citation_separator_inline_math_fragments(elements)
    )
    if completed_separators:
        log.info(
            "Page %s: restored %s merged citation-separator protection "
            "record(s)",
            page.number + 1,
            completed_separators,
        )

    try:
        page_dict = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE,
        )
    except Exception as page_dict_exc:
        log.warning(
            f"Page {page.number + 1}: formula signature extraction failed: "
            f"{type(page_dict_exc).__name__}"
        )
        return elements

    # Signatures are per-element evidence, so they must fail per element.  A
    # shared ``try`` around the loop let one unsignable region silently strip
    # the signature from every later formula on the page, and the audit then
    # rejected the finished document with one
    # ``source-formula-signature-missing`` warning per stripped element.
    for index, element in enumerate(elements):
        if element.get("type") != ElementType.FORMULA_IMAGE:
            continue
        try:
            element["math_signature"] = _pdf_formula_region_signature(
                page,
                element.get("bbox", element.get("rect")),
                page_dict=page_dict,
            )
        except Exception as signature_exc:
            log.warning(
                f"Page {page.number + 1} elem {index}: formula signature "
                f"extraction failed: {type(signature_exc).__name__}: "
                f"{signature_exc}"
            )
    return elements


def _extract_page_elements(page, doc) -> list[PdfElement]:
    """Extract and semantically normalize ordered elements from one PDF page."""
    del doc  # retained for the legacy callback contract
    table_elements, table_rects, table_regions = _extract_pdf_table_elements(
        page
    )
    image_elements, image_rects = _extract_pdf_image_elements(
        page,
        table_rects,
    )
    visibility: dict = {}
    elements = [
        *table_elements,
        *image_elements,
        *_extract_pdf_text_elements(page, table_rects, visibility),
    ]
    elements = _merge_and_validate_pdf_page_semantics(
        page,
        elements,
        table_rects,
        table_regions,
    )
    elements = _enrich_pdf_page_elements(page, elements, image_rects)
    _log_page_pass(
        page,
        _mark_pdf_invisible_source(
            elements, scan_layer=bool(visibility.get("scan_layer"))
        ),
        "kept %s text element(s) the source page does not show out of translation",
    )
    return elements
