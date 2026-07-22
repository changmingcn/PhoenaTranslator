"""Semantic formulae helpers for deterministic PDF processing."""

from __future__ import annotations

import logging
import re
from collections import deque

import fitz
from phoena_translator.math_text import (
    MATH_FUNCTION_WORDS,
    is_math_block as _is_math_block,
    is_unicode_math_symbol as _is_unicode_math_symbol,
    unicode_math_symbol_count as _unicode_math_symbol_count,
)

from phoena_translator.pdf.types import (
    PDF_FORMULA_GUARD_PADDING,
    _PDF_BARE_FOOTNOTE_LEAD_RE,
    _PDF_WRAPPED_MATH_DANGLING_TAIL_RE,
    _PDF_WRAPPED_MATH_HYPHEN_TAIL_RE,
    _PDF_WRAPPED_MATH_PROSE_WORD_RE,
    _PDF_WRAPPED_MATH_SCRIPT_LEAD_RE,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
    _get_pdf_source_ink_rect,
    _get_pdf_source_ink_rects,
    _pdf_elem_last_source_line_rect,
    _pdf_rect_intersects_protected,
)
from phoena_translator.pdf.math_detection import (
    _looks_like_pdf_list_marker,
    _pdf_superscript_signature,
    _plain_text,
)

log = logging.getLogger("translator")

from phoena_translator.pdf.semantic_references import _translate_pdf_citation_labels
from phoena_translator.pdf.semantic_text import (
    _join_pdf_line_fragments,
    _looks_like_heading_text,
    _looks_like_pdf_paragraph_lead,
    _pick_pdf_dominant_value,
)


def _split_pdf_formula_adjacent_paragraph_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> int:
    """Isolate paragraph or visual-line runs that touch immutable formulae.

    A single native PDF block can contain three independent paragraphs.  If a
    formula grazes the first paragraph, promoting the union would preserve all
    three in English.  Paragraph source boxes recorded during extraction let
    us isolate the genuinely affected paragraph while retaining paragraph-
    level translation and redraw for its safe siblings.  A formula can also
    occupy only the first or last visual line of one long paragraph; recorded
    source lines then keep the remaining contiguous prose run translatable as
    one semantic unit instead of promoting the whole paragraph.
    """
    formula_elements = [
        elem for elem in elements or [] if elem.get("type") == "formula_image"
    ]
    protected_rects = [
        fitz.Rect(elem.get("bbox", elem.get("rect"))) for elem in formula_elements
    ]
    if not protected_rects:
        return 0

    split_count = 0
    rebuilt = []
    for elem in elements or []:
        paragraphs = [
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if isinstance(paragraph, dict) and (paragraph.get("plain") or "").strip()
        ]
        # Keep a complete semantic paragraph whole when only the native font
        # box of its final line grazes the next equation.  Probe the whole
        # element with the same conservative rule used by final formula
        # promotion.  A mixed formula/prose block fails this probe and still
        # goes through the line-run splitter below.
        keep_whole_safe_paragraph = False
        if elem.get("type") == "text" and len(paragraphs) == 1:
            probe = dict(elem)
            _promote_pdf_formula_overlaps([probe, *formula_elements])
            keep_whole_safe_paragraph = bool(
                probe.get("type") == "text" and probe.get("formula_adjacent_text")
            )
        expanded_paragraphs = []
        for paragraph in paragraphs:
            if keep_whole_safe_paragraph:
                expanded_paragraphs.append(paragraph)
                continue
            source_lines = [
                line
                for line in (paragraph.get("source_lines") or [])
                if isinstance(line, dict)
                and (line.get("plain") or "").strip()
                and line.get("bbox")
            ]
            if len(source_lines) < 2:
                expanded_paragraphs.append(paragraph)
                continue

            touches_formula = [
                _pdf_rect_intersects_protected(
                    fitz.Rect(line["bbox"]),
                    protected_rects,
                    padding=0.0,
                )
                for line in source_lines
            ]
            if not any(touches_formula) or all(touches_formula):
                expanded_paragraphs.append(paragraph)
                continue

            line_runs = []
            current_lines = [source_lines[0]]
            current_touch = touches_formula[0]
            for line, touching in zip(source_lines[1:], touches_formula[1:]):
                if touching != current_touch:
                    line_runs.append(current_lines)
                    current_lines = [line]
                    current_touch = touching
                else:
                    current_lines.append(line)
            line_runs.append(current_lines)

            for run_index, run_lines in enumerate(line_runs):
                paragraph_copy = dict(paragraph)
                paragraph_copy["plain"] = _join_pdf_line_fragments(
                    [line.get("plain", "") for line in run_lines]
                )
                paragraph_copy["rich"] = _join_pdf_line_fragments(
                    [line.get("rich", line.get("plain", "")) for line in run_lines]
                )
                paragraph_copy["source_bbox"] = [
                    min(float(line["bbox"][0]) for line in run_lines),
                    min(float(line["bbox"][1]) for line in run_lines),
                    max(float(line["bbox"][2]) for line in run_lines),
                    max(float(line["bbox"][3]) for line in run_lines),
                ]
                paragraph_copy["source_line_bboxes"] = [
                    [float(value) for value in line["bbox"]] for line in run_lines
                ]
                paragraph_copy["source_lines"] = [dict(line) for line in run_lines]
                if run_index:
                    paragraph_copy["gap_before"] = 0.0
                expanded_paragraphs.append(paragraph_copy)
        paragraphs = expanded_paragraphs
        if (
            elem.get("type") != "text"
            or len(paragraphs) < 2
            or not all(paragraph.get("source_bbox") for paragraph in paragraphs)
            or not _pdf_rect_intersects_protected(
                _get_pdf_elem_rect(elem),
                protected_rects,
            )
        ):
            rebuilt.append(elem)
            continue

        original_rect = _get_pdf_elem_rect(elem)
        superscript_queues: dict[str, deque] = {}
        for run in elem.get("superscript_runs") or []:
            marker = str(run.get("text", ""))
            if marker:
                superscript_queues.setdefault(marker, deque()).append(dict(run))
        inline_math_queues: dict[str, deque] = {}
        for fragment in elem.get("inline_math_fragments") or []:
            marker = str(fragment.get("text", ""))
            if marker:
                inline_math_queues.setdefault(marker, deque()).append(dict(fragment))

        fragments = []
        for paragraph in paragraphs:
            try:
                paragraph_rect = fitz.Rect(paragraph["source_bbox"])
            except (TypeError, ValueError):
                fragments = []
                break
            paragraph_rect = paragraph_rect & fitz.Rect(page_rect)
            if paragraph_rect.is_empty:
                fragments = []
                break

            plain = (paragraph.get("plain") or "").strip()
            rich = (paragraph.get("rich") or plain).strip()
            fragment = dict(elem)
            fragment["rect"] = paragraph_rect
            fragment["bbox"] = [float(value) for value in paragraph_rect]
            fragment["render_bbox"] = _get_pdf_render_bbox(
                paragraph_rect,
                page_rect,
                [paragraph],
                float(elem.get("fontsize", 11.0)),
            )
            fragment["x"] = float(paragraph_rect.x0)
            fragment["y"] = float(paragraph_rect.y0)
            fragment["content"] = plain
            fragment["rich_content"] = rich if rich != plain else None
            fragment["top_padding"] = 0.0

            paragraph_copy = dict(paragraph)
            absolute_margin = original_rect.x0 + float(
                paragraph_copy.get("margin_left", 0.0)
            )
            paragraph_copy["margin_left"] = max(
                0.0,
                absolute_margin - paragraph_rect.x0,
            )
            paragraph_copy["gap_before"] = 0.0
            fragment["paragraphs"] = [paragraph_copy]

            selected_runs = []
            for marker in _pdf_superscript_signature(rich):
                queue = superscript_queues.get(marker)
                if queue:
                    selected_runs.append(queue.popleft())
            fragment["superscript_runs"] = selected_runs
            fragment["superscript_scale"] = (
                float(
                    _pick_pdf_dominant_value(
                        [
                            (
                                round(float(run.get("scale", 0.60)), 3),
                                max(len(run.get("text", "")), 1),
                            )
                            for run in selected_runs
                        ],
                        0.60,
                    )
                )
                if selected_runs
                else None
            )

            selected_math = []
            math_cursor = 0
            while True:
                next_match = None
                for marker, queue in inline_math_queues.items():
                    if not queue:
                        continue
                    position = plain.find(marker, math_cursor)
                    if position < 0:
                        continue
                    candidate = (position, -len(marker), marker, queue)
                    if next_match is None or candidate[:2] < next_match[:2]:
                        next_match = candidate
                if next_match is None:
                    break
                position, _, marker, queue = next_match
                selected_math.append(queue.popleft())
                math_cursor = position + len(marker)
            fragment["inline_math_fragments"] = selected_math

            bold_text = "".join(
                _plain_text(match.group(1))
                for match in re.finditer(r"(?is)<b>(.*?)</b>", rich)
            )
            fragment["bold"] = bool(
                plain and len(bold_text.strip()) >= max(1, len(plain) * 0.5)
            )
            fragments.append(fragment)

        if not fragments:
            rebuilt.append(elem)
            continue
        rebuilt.extend(fragments)
        split_count += len(fragments) - 1

    if split_count:
        rebuilt.sort(
            key=lambda element: (
                float(element.get("y", _get_pdf_elem_rect(element).y0)),
                float(element.get("x", _get_pdf_elem_rect(element).x0)),
            )
        )
        elements[:] = rebuilt
    return split_count


def _pdf_wrapped_math_fragment_prose_words(plain: str) -> list[str]:
    """Real dictionary-style prose words (>=3 letters, not math functions)."""
    return [
        word
        for word in _PDF_WRAPPED_MATH_PROSE_WORD_RE.findall(plain or "")
        if word.lower() not in MATH_FUNCTION_WORDS
    ]


def _pdf_wrapped_math_continuation_pair_can_merge(
    previous: dict,
    current: dict,
    protected_rects: list,
    blocker_rects: list,
) -> bool:
    """Recognize a mid-sentence fragment split off by a wrapped script variable.

    An inline variable such as ``q^E_ls`` carries stacked sub/superscripts.
    PyMuPDF splits the physical row at those baseline changes, so one sentence
    becomes several line objects and blocks: the paragraph ends at ``... net
    flow (qE`` and the continuation ``ls), other traders' E-`` is emitted as a
    standalone short element (same row, abutting horizontally) or as a
    next-row wrap.  Translated in isolation those fragments come back with
    prose words untranslated.  Merge only when the previous element is genuine
    prose that ends at a dangling opening bracket + short variable stem (or a
    hyphen wrap) and the fragment starts like the continuation (script tail
    such as ``ls),`` or a lowercase word) and still contains real prose words.
    Script-scale INLINE formula rects may sit inside the reunited sentence;
    display formulas, tables and images are never crossed.
    """
    for elem in (previous, current):
        if (
            elem.get("type") != "text"
            or elem.get("skip_translate_reason")
            or elem.get("watermark")
            or elem.get("non_horizontal")
            or elem.get("vector_ocr")
            or elem.get("formula_adjacent_text")
        ):
            return False
    if previous.get("table_hint"):
        return False
    if bool(previous.get("bold")) != bool(current.get("bold")):
        return False

    p_plain = re.sub(r"\s+", " ", _plain_text(previous.get("content", ""))).strip()
    f_plain = re.sub(r"\s+", " ", _plain_text(current.get("content", ""))).strip()
    if len(p_plain) < 40 or not f_plain or len(f_plain) > 90:
        return False
    f_paragraphs = [
        paragraph
        for paragraph in (current.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(f_paragraphs) > 1:
        return False
    if (
        _looks_like_pdf_paragraph_lead(f_plain)
        or _looks_like_pdf_list_marker(f_plain)
        or _PDF_BARE_FOOTNOTE_LEAD_RE.match(f_plain)
    ):
        return False
    if not _pdf_wrapped_math_fragment_prose_words(f_plain):
        return False

    script_lead = bool(_PDF_WRAPPED_MATH_SCRIPT_LEAD_RE.match(f_plain))
    lower_lead = bool(f_plain[:1].islower())
    dangling_math = bool(_PDF_WRAPPED_MATH_DANGLING_TAIL_RE.search(p_plain))
    hyphen_wrap = bool(_PDF_WRAPPED_MATH_HYPHEN_TAIL_RE.search(p_plain))
    if not (
        (dangling_math and (script_lead or lower_lead)) or (hyphen_wrap and lower_lead)
    ):
        return False

    p_rect = _get_pdf_elem_rect(previous)
    f_rect = _get_pdf_elem_rect(current)
    p_fontsize = max(float(previous.get("fontsize", 11.0)), 1.0)
    f_fontsize = max(float(current.get("fontsize", 11.0)), 1.0)
    if f_fontsize > p_fontsize * 1.30 or f_fontsize < p_fontsize * 0.50:
        return False

    last_rect = _pdf_elem_last_source_line_rect(previous) or fitz.Rect(p_rect)

    # Same-row continuation: a stacked-script split leaves the fragment on the
    # same visual row, starting where (or slightly before, under the raised
    # script) the previous line's glyphs end and extending to the right.
    y_overlap = min(float(f_rect.y1), float(last_rect.y1)) - max(
        float(f_rect.y0), float(last_rect.y0)
    )
    min_row_height = max(min(float(f_rect.height), float(last_rect.height)), 1.0)
    lead_gap = float(f_rect.x0) - float(last_rect.x1)
    same_row = (
        y_overlap >= min_row_height * 0.35
        and -p_fontsize * 2.0 <= lead_gap <= p_fontsize * 2.5
        and float(f_rect.x1) > float(last_rect.x1) + 1.0
    )
    if same_row:
        if not (dangling_math and (script_lead or lower_lead)):
            return False
    else:
        vertical_gap = float(f_rect.y0) - float(last_rect.y1)
        if vertical_gap < -max(float(f_rect.height), p_fontsize) * 0.60:
            return False
        if vertical_gap > max(p_fontsize * 1.90, 12.0):
            return False
        overlap = min(float(p_rect.x1), float(f_rect.x1)) - max(
            float(p_rect.x0), float(f_rect.x0)
        )
        if overlap < min(float(p_rect.width), float(f_rect.width)) * 0.20:
            return False

    # A script-scale formula rect is an INLINE variable (for example a
    # promoted wrapped ``q^E_ls`` run) and may live inside the reunited
    # sentence; its immutable geometry stays in place and the renderer's
    # formula-safe trimming flows the translation around it.  Display-scale
    # formulas and any table/image geometry are never crossed.
    display_rects = [
        rect
        for rect in (protected_rects or [])
        if not (
            float(rect.height) <= p_fontsize * 1.60
            and float(rect.width) <= p_fontsize * 12.0
        )
    ]
    union = fitz.Rect(p_rect) | fitz.Rect(f_rect)
    if _pdf_rect_intersects_protected(union, display_rects, padding=0.0):
        return False
    if _pdf_rect_intersects_protected(union, blocker_rects, padding=0.0):
        return False
    return True


def _merge_pdf_wrapped_math_continuation_pair(
    previous: dict,
    current: dict,
    page_rect: fitz.Rect,
) -> dict:
    """Fold a wrapped-script continuation fragment back into its paragraph."""
    p_rect = _get_pdf_elem_rect(previous)
    f_rect = _get_pdf_elem_rect(current)
    rect = fitz.Rect(p_rect) | fitz.Rect(f_rect)

    plain = _join_pdf_line_fragments(
        [
            previous.get("content", ""),
            current.get("content", ""),
        ]
    )
    has_rich = bool(previous.get("rich_content") or current.get("rich_content"))
    rich = (
        _join_pdf_line_fragments(
            [
                previous.get("rich_content") or previous.get("content", ""),
                current.get("rich_content") or current.get("content", ""),
            ]
        )
        if has_rich
        else None
    )

    f_paragraph = next(
        (
            paragraph
            for paragraph in (current.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ),
        None,
    )
    f_lines = [
        dict(line)
        for line in ((f_paragraph or {}).get("source_lines") or [])
        if (line.get("plain") or "").strip()
    ]
    if not f_lines:
        f_lines = [
            {
                "plain": current.get("content", ""),
                "rich": current.get("rich_content") or current.get("content", ""),
                "bbox": [float(value) for value in f_rect],
            }
        ]

    paragraphs = [
        dict(paragraph)
        for paragraph in (previous.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if paragraphs:
        last = paragraphs[-1]
        last["plain"] = _join_pdf_line_fragments(
            [
                last.get("plain", ""),
                (f_paragraph or {}).get("plain") or current.get("content", ""),
            ]
        )
        last["rich"] = _join_pdf_line_fragments(
            [
                last.get("rich") or last.get("plain", ""),
                (f_paragraph or {}).get("rich")
                or current.get("rich_content")
                or current.get("content", ""),
            ]
        )
        last_lines = [dict(line) for line in (last.get("source_lines") or [])]
        last_lines.extend(f_lines)
        last["source_lines"] = last_lines
        last_bboxes = [
            [float(value) for value in line["bbox"]]
            for line in last_lines
            if line.get("bbox")
        ]
        if last_bboxes:
            last["source_line_bboxes"] = last_bboxes
            last["source_bbox"] = [
                min(box[0] for box in last_bboxes),
                min(box[1] for box in last_bboxes),
                max(box[2] for box in last_bboxes),
                max(box[3] for box in last_bboxes),
            ]
        last["nowrap"] = None
    else:
        paragraphs = [
            {
                "plain": plain,
                "rich": rich or plain,
                "source_bbox": [float(value) for value in rect],
                "source_lines": f_lines,
                "margin_left": 0.0,
                "text_indent": 0.0,
                "gap_before": 0.0,
                "text_align": "left",
                "nowrap": None,
            }
        ]

    fontsize = float(previous.get("fontsize", 11.0))
    merged = dict(previous)
    merged.update(
        {
            "y": float(rect.y0),
            "x": float(rect.x0),
            "rect": fitz.Rect(rect),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "render_bbox": _get_pdf_render_bbox(rect, page_rect, paragraphs, fontsize),
            "content": plain,
            "rich_content": rich,
            "paragraphs": paragraphs,
            "fontsize": fontsize,
            "line_height": max(
                float(previous.get("line_height", fontsize * 1.1)),
                float(current.get("line_height", fontsize * 1.1)),
            ),
            "bold": bool(previous.get("bold")) and bool(current.get("bold")),
            "superscript_runs": [
                dict(run)
                for elem in (previous, current)
                for run in (elem.get("superscript_runs") or [])
            ],
            "superscript_scale": (
                previous.get("superscript_scale") or current.get("superscript_scale")
            ),
            "inline_math_fragments": [
                dict(fragment)
                for elem in (previous, current)
                for fragment in (elem.get("inline_math_fragments") or [])
            ],
            "table_hint": bool(previous.get("table_hint")),
            "preserve_source_style": bool(previous.get("preserve_source_style")),
            "single_line_heading": False,
            "wrapped_math_continuation_merged": (
                int(previous.get("wrapped_math_continuation_merged") or 1) + 1
            ),
        }
    )
    merged.pop("layout_class", None)
    return merged


def _merge_pdf_wrapped_inline_math_continuation_fragments(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> int:
    """Merge sentence fragments broken by a wrapped inline math variable.

    Runs before layout classification so the rejoined sentence classifies as
    ordinary body prose and translates as one unit instead of leaving short
    incoherent fragments (which the model then returns untranslated and the
    renderer shrinks into their narrow source strips).  Mutates ``elements``
    in place; returns the number of merges.
    """
    if not elements:
        return 0
    protected_rects = [
        fitz.Rect(elem.get("bbox", elem.get("rect")))
        for elem in elements
        if elem.get("type") == "formula_image"
    ]
    blocker_rects = [
        fitz.Rect(elem.get("bbox", elem.get("rect")))
        for elem in elements
        if elem.get("type") in {"table_image", "image"}
    ]
    merge_count = 0
    index = 0
    while index < len(elements):
        current = elements[index]
        if current.get("type") != "text":
            index += 1
            continue
        cursor = index + 1
        blocked = False
        while cursor < len(elements) and elements[cursor].get("type") != "text":
            if elements[cursor].get("type") != "formula_image":
                blocked = True
            cursor += 1
        if cursor >= len(elements):
            break
        if blocked or not _pdf_wrapped_math_continuation_pair_can_merge(
            current,
            elements[cursor],
            protected_rects,
            blocker_rects,
        ):
            index += 1
            continue
        elements[index] = _merge_pdf_wrapped_math_continuation_pair(
            current,
            elements[cursor],
            page_rect,
        )
        del elements[cursor]
        merge_count += 1
        # Stay on the merged element: the same sentence can wrap repeatedly
        # (qE_ls, qE_ot, qS_ot) and each tail merges in turn.
    return merge_count


def _promote_pdf_formula_overlaps(
    elements: list[dict],
    padding: float = PDF_FORMULA_GUARD_PADDING,
) -> int:
    """Fail closed when source text geometry touches a protected formula.

    TeX radicals and scripts often extend into the previous line's glyph box.
    PDF redaction is glyph-based, so subtracting only the intersection rectangle
    can still delete the whole neighboring glyph. Promote that source text to
    an immutable element before translation instead.
    """
    formulas = [elem for elem in elements or [] if elem.get("type") == "formula_image"]
    protected_rects = [
        fitz.Rect(elem.get("bbox", elem.get("rect"))) for elem in formulas
    ]
    promoted = 0
    for elem in elements or []:
        if elem.get("type") != "text":
            continue
        layout_rect = fitz.Rect(elem.get("bbox", elem.get("rect")))
        # Recovered table cells intentionally carry a larger render rectangle
        # so Chinese can wrap within the original cell.  Formula protection,
        # however, must classify the source glyphs rather than that available
        # whitespace: a unit row immediately below a header is not part of the
        # header merely because the future render rectangle reaches it.
        rect = (
            _get_pdf_source_ink_rect(elem)
            if elem.get("semantic_table_cell")
            else layout_rect
        )
        touching = [
            formula
            for formula, protected in zip(formulas, protected_rects)
            if _pdf_rect_intersects_protected(rect, [protected], padding)
        ]
        if not touching:
            continue

        has_recorded_source_lines = any(
            paragraph.get("source_lines") or paragraph.get("source_line_bboxes")
            for paragraph in (elem.get("paragraphs") or [])
        )
        if has_recorded_source_lines and not any(
            _pdf_rect_intersects_protected(
                source_rect,
                [fitz.Rect(formula.get("bbox", formula.get("rect")))],
                padding,
            )
            for source_rect in _get_pdf_source_ink_rects(elem)
            for formula in touching
        ):
            # The element's outer bbox is diagonal across disjoint visual
            # lines, but no recorded source glyph touches the formula.  Keep
            # the prose translatable; assembly redacts the same precise source
            # rectangles and trims its render ladder around immutable math.
            elem["formula_adjacent_text"] = True
            elem["formula_overlap_ratio"] = 0.0
            continue

        # A formula's font box can graze the top or bottom of an adjacent
        # prose line even though no prose glyph belongs to the formula.  The
        # former all-or-nothing rule promoted the whole paragraph in that
        # situation, leaving acknowledgements, appendix prose and table notes
        # untranslated.  Keep linguistically substantial text translatable
        # when every *actual* overlap is only a shallow edge sliver.  Assembly
        # still subtracts the immutable formula rectangles from redaction and
        # trims the insertion rectangle before drawing, so formula pixels stay
        # protected.
        plain = re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
        prose_words = re.findall(r"[A-Za-z]{3,}", plain)
        rect_area = max(rect.get_area(), 1.0)
        actual_intersections = []
        shallow_edge_only = True
        shallow_limit = max(float(elem.get("fontsize", 0.0)) * 0.35, 3.0)
        for formula in touching:
            protected = fitz.Rect(formula.get("bbox", formula.get("rect")))
            intersection = rect & protected
            if intersection.is_empty or intersection.get_area() <= 0:
                continue
            actual_intersections.append(intersection)
            edge_sliver = (
                (
                    abs(intersection.y0 - rect.y0) <= 0.1
                    or abs(intersection.y1 - rect.y1) <= 0.1
                )
                and intersection.height <= shallow_limit
            ) or (
                (
                    abs(intersection.x0 - rect.x0) <= 0.1
                    or abs(intersection.x1 - rect.x1) <= 0.1
                )
                and intersection.width <= shallow_limit
            )
            if not edge_sliver:
                shallow_edge_only = False
                break
        overlap_ratio = (
            sum(intersection.get_area() for intersection in actual_intersections)
            / rect_area
        )
        if (
            elem.get("semantic_table_cell")
            and not actual_intersections
            and not elem.get("inline_math_fragments")
        ):
            elem["formula_adjacent_text"] = True
            elem["formula_overlap_ratio"] = 0.0
            continue
        safe_long_prose = (
            len(plain) >= 160
            and len(prose_words) >= 12
            and rect.height >= max(float(elem.get("fontsize", 0.0)) * 1.8, 20.0)
        )
        safe_short_heading = (
            len(plain) <= 100
            and len(prose_words) >= 2
            and (bool(elem.get("bold")) or _looks_like_heading_text(plain))
        )
        safe_prose_line = (
            len(plain) >= 60
            and len(prose_words) >= 8
            and rect.height <= max(float(elem.get("fontsize", 0.0)) * 1.8, 22.0)
        )
        # A footnote sentence can be one short visual line immediately above
        # or below a displayed equation.  Native font boxes on consecutive
        # nine-point lines commonly overlap by two or three points even when
        # their painted glyphs do not.  Requiring 60 characters / eight long
        # words promoted those complete prose sentences to immutable formula
        # images (ASIC 452 pp. 65 and 69), leaving ordinary English in an
        # otherwise translated footnote.  Admit only a sentence-terminated,
        # symbol-free prose line through the wider edge-overlap allowance;
        # mixed formula/prose lines and equation fragments remain protected.
        safe_short_sentence_line = (
            len(prose_words) >= 3
            and not elem.get("inline_math_fragments")
            and not any(_is_unicode_math_symbol(char) for char in plain)
            and bool(re.search(r"[.!?][\"'’”)]?\s*$", plain))
            and rect.height <= max(float(elem.get("fontsize", 0.0)) * 1.8, 22.0)
        )
        safe_citation_label_line = (
            not actual_intersections
            and not elem.get("inline_math_fragments")
            and _translate_pdf_citation_labels(plain) != plain
        )
        ordinary_safe_overlap = (
            safe_long_prose or safe_short_heading or safe_prose_line
        ) and overlap_ratio <= 0.08
        short_sentence_safe_overlap = safe_short_sentence_line and overlap_ratio <= 0.28
        if (
            (
                ordinary_safe_overlap
                or short_sentence_safe_overlap
                or safe_citation_label_line
            )
            and not elem.get("table_hint")
            and (not elem.get("inline_math_fragments") or not actual_intersections)
            and not _looks_like_pdf_paragraph_lead(plain)
            and not _looks_like_pdf_list_marker(plain)
            and not _is_math_block(plain)
            and shallow_edge_only
        ):
            elem["formula_adjacent_text"] = True
            elem["formula_overlap_ratio"] = round(overlap_ratio, 6)
            continue

        reasons = {"formula_geometry_overlap"}
        for formula in touching:
            reasons.update(formula.get("math_reasons") or [])
        elem["type"] = "formula_image"
        elem["math_reasons"] = sorted(reasons)
        elem["math_mixed"] = True
        elem["math_line_count"] = max(1, len(elem.get("paragraphs") or []))
        elem["math_candidate_symbol_count"] = _unicode_math_symbol_count(
            _plain_text(elem.get("content", ""))
        )
        elem["skip_translate_reason"] = "formula_geometry_overlap"
        promoted += 1
    return promoted
