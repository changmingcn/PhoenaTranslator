"""Formula-adjacent prose: merge wrapped inline-math lines, split paragraphs
at formula contact and preserve text that overlaps formula geometry."""

from __future__ import annotations

import logging
import re
from collections import deque

import fitz

from phoena_translator.math_text import (
    MATH_FUNCTION_WORDS,
)
from phoena_translator.math_text import (
    is_math_block as _is_math_block,
)
from phoena_translator.math_text import (
    is_unicode_math_symbol as _is_unicode_math_symbol,
)
from phoena_translator.math_text import (
    unicode_math_symbol_count as _unicode_math_symbol_count,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
    _get_pdf_source_ink_rect,
    _get_pdf_source_ink_rects,
    _pdf_elem_last_source_line_rect,
    _pdf_rect_intersects_protected,
)
from phoena_translator.pdf.patterns import (
    _PDF_BARE_FOOTNOTE_LEAD_RE,
    PDF_FORMULA_GUARD_PADDING,
)
from phoena_translator.pdf.text_shape import (
    _collapsed_plain_text,
    _looks_like_pdf_list_marker,
    _plain_text,
)
from phoena_translator.pdf.translate.inline_markup import _pdf_superscript_signature

log = logging.getLogger("translator")
from phoena_translator.pdf.model import DEFAULT_FONTSIZE, ElementType, SkipReason
from phoena_translator.pdf.semantics.references import _translate_pdf_citation_labels
from phoena_translator.pdf.text_shape import (
    _join_pdf_line_fragments,
    _looks_like_heading_text,
    _looks_like_pdf_paragraph_lead,
    _pick_pdf_dominant_value,
)

_PDF_WRAPPED_MATH_DANGLING_TAIL_RE = re.compile(r"[(\[][A-Za-z]{0,3}$")
_PDF_WRAPPED_MATH_HYPHEN_TAIL_RE = re.compile(r"[A-Za-z]-$")
_PDF_WRAPPED_MATH_SCRIPT_LEAD_RE = re.compile(r"^[a-z]{1,3}[)\],]")
_PDF_WRAPPED_MATH_PROSE_WORD_RE = re.compile(r"[A-Za-z][a-z]{2,}")


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
        elem for elem in elements or [] if elem.get("type") == ElementType.FORMULA_IMAGE
    ]
    protected_rects = [
        _get_pdf_elem_rect(elem) for elem in formula_elements
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
        if not _keeps_whole_formula_adjacent_paragraph(elem, paragraphs, formula_elements):
            paragraphs = [
                run
                for paragraph in paragraphs
                for run in _split_pdf_paragraph_at_formula_lines(paragraph, protected_rects)
            ]
        if (
            elem.get("type") != ElementType.TEXT
            or len(paragraphs) < 2
            or not all(paragraph.get("source_bbox") for paragraph in paragraphs)
            or not _pdf_rect_intersects_protected(
                _get_pdf_elem_rect(elem),
                protected_rects,
            )
        ):
            rebuilt.append(elem)
            continue

        fragments = _pdf_paragraph_fragments(elem, paragraphs, page_rect)
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


def _keeps_whole_formula_adjacent_paragraph(
    elem: dict,
    paragraphs: list[dict],
    formula_elements: list[dict],
) -> bool:
    """Keep one complete paragraph whole when only its line edge grazes math.

    The native font box of a paragraph's final line can graze the next
    equation.  Probe the element with the same conservative rule used by
    final formula promotion; a mixed formula/prose block fails the probe and
    still goes through the line-run splitter.
    """
    if elem.get("type") != ElementType.TEXT or len(paragraphs) != 1:
        return False
    probe = dict(elem)
    _promote_pdf_formula_overlaps([probe, *formula_elements])
    return bool(
        probe.get("type") == ElementType.TEXT and probe.get("formula_adjacent_text")
    )


def _split_pdf_paragraph_at_formula_lines(
    paragraph: dict,
    protected_rects: list[fitz.Rect],
) -> list[dict]:
    """Split a paragraph into runs of consecutive lines that do or do not touch math."""
    source_lines = [
        line
        for line in (paragraph.get("source_lines") or [])
        if isinstance(line, dict)
        and (line.get("plain") or "").strip()
        and line.get("bbox")
    ]
    if len(source_lines) < 2:
        return [paragraph]

    touches_formula = [
        _pdf_rect_intersects_protected(
            fitz.Rect(line["bbox"]),
            protected_rects,
            padding=0.0,
        )
        for line in source_lines
    ]
    if not any(touches_formula) or all(touches_formula):
        return [paragraph]

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

    runs = []
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
        runs.append(paragraph_copy)
    return runs


def _pdf_marker_queues(items: list[dict] | None) -> dict[str, deque]:
    """Group superscript runs or inline-math fragments by marker text, in order."""
    queues: dict[str, deque] = {}
    for item in items or []:
        marker = str(item.get("text", ""))
        if marker:
            queues.setdefault(marker, deque()).append(dict(item))
    return queues


def _pdf_paragraph_fragments(
    elem: dict,
    paragraphs: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """One text element per paragraph, or ``[]`` when any paragraph box is unusable."""
    original_rect = _get_pdf_elem_rect(elem)
    superscript_queues = _pdf_marker_queues(elem.get("superscript_runs"))
    inline_math_queues = _pdf_marker_queues(elem.get("inline_math_fragments"))
    fragments = []
    for paragraph in paragraphs:
        try:
            paragraph_rect = fitz.Rect(paragraph["source_bbox"])
        except (TypeError, ValueError):
            return []
        paragraph_rect = paragraph_rect & fitz.Rect(page_rect)
        if paragraph_rect.is_empty:
            return []
        fragments.append(
            _pdf_paragraph_fragment(
                elem,
                paragraph,
                paragraph_rect,
                page_rect,
                original_rect,
                superscript_queues,
                inline_math_queues,
            )
        )
    return fragments


def _pdf_paragraph_fragment(
    elem: dict,
    paragraph: dict,
    paragraph_rect: fitz.Rect,
    page_rect: fitz.Rect,
    original_rect: fitz.Rect,
    superscript_queues: dict[str, deque],
    inline_math_queues: dict[str, deque],
) -> dict:
    plain = (paragraph.get("plain") or "").strip()
    rich = (paragraph.get("rich") or plain).strip()
    fragment = dict(elem)
    fragment["rect"] = paragraph_rect
    fragment["bbox"] = [float(value) for value in paragraph_rect]
    fragment["render_bbox"] = _get_pdf_render_bbox(
        paragraph_rect,
        page_rect,
        [paragraph],
        float(elem.get("fontsize", DEFAULT_FONTSIZE)),
    )
    fragment["x"] = float(paragraph_rect.x0)
    fragment["y"] = float(paragraph_rect.y0)
    fragment["content"] = plain
    fragment["rich_content"] = rich if rich != plain else None
    fragment["top_padding"] = 0.0

    paragraph_copy = dict(paragraph)
    absolute_margin = original_rect.x0 + float(paragraph_copy.get("margin_left", 0.0))
    paragraph_copy["margin_left"] = max(0.0, absolute_margin - paragraph_rect.x0)
    paragraph_copy["gap_before"] = 0.0
    fragment["paragraphs"] = [paragraph_copy]

    selected_runs = _take_pdf_superscript_runs(rich, superscript_queues)
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
    fragment["inline_math_fragments"] = _take_pdf_inline_math_fragments(
        plain, inline_math_queues
    )

    bold_text = "".join(
        _plain_text(match.group(1))
        for match in re.finditer(r"(?is)<b>(.*?)</b>", rich)
    )
    fragment["bold"] = bool(
        plain and len(bold_text.strip()) >= max(1, len(plain) * 0.5)
    )
    return fragment


def _take_pdf_superscript_runs(rich: str, queues: dict[str, deque]) -> list[dict]:
    """Claim the next queued run for each superscript marker in ``rich``."""
    selected_runs = []
    for marker in _pdf_superscript_signature(rich):
        queue = queues.get(marker)
        if queue:
            selected_runs.append(queue.popleft())
    return selected_runs


def _take_pdf_inline_math_fragments(plain: str, queues: dict[str, deque]) -> list[dict]:
    """Claim queued inline-math fragments in the order their text occurs in ``plain``.

    At each position the earliest match wins, and among matches at the same
    position the longest marker wins.
    """
    selected_math = []
    math_cursor = 0
    while True:
        next_match = None
        for marker, queue in queues.items():
            if not queue:
                continue
            position = plain.find(marker, math_cursor)
            if position < 0:
                continue
            candidate = (position, -len(marker), marker, queue)
            if next_match is None or candidate[:2] < next_match[:2]:
                next_match = candidate
        if next_match is None:
            return selected_math
        position, _, marker, queue = next_match
        selected_math.append(queue.popleft())
        math_cursor = position + len(marker)


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
    if not (_pdf_wrappable_prose(previous) and _pdf_wrappable_prose(current)):
        return False
    if previous.get("table_hint"):
        return False
    if bool(previous.get("bold")) != bool(current.get("bold")):
        return False
    dangling_continuation = _pdf_wrapped_math_text_continues(previous, current)
    if dangling_continuation is None:
        return False

    p_rect = _get_pdf_elem_rect(previous)
    f_rect = _get_pdf_elem_rect(current)
    p_fontsize = max(float(previous.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    f_fontsize = max(float(current.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    if f_fontsize > p_fontsize * 1.30 or f_fontsize < p_fontsize * 0.50:
        return False
    last_rect = _pdf_elem_last_source_line_rect(previous) or fitz.Rect(p_rect)
    if not _pdf_wrapped_math_rows_continue(
        p_rect, last_rect, f_rect, p_fontsize, dangling_continuation
    ):
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


def _pdf_wrappable_prose(elem: dict) -> bool:
    return not (
        elem.get("type") != ElementType.TEXT
        or elem.get("skip_translate_reason")
        or elem.get("watermark")
        or elem.get("non_horizontal")
        or elem.get("vector_ocr")
        or elem.get("formula_adjacent_text")
    )


def _pdf_wrapped_math_text_continues(previous: dict, current: dict) -> bool | None:
    """Classify the textual seam between ``previous`` and fragment ``current``.

    Returns ``True`` when the previous line ends at a dangling script variable
    that the fragment continues, ``False`` for a hyphen wrap continued in
    lowercase, and ``None`` when the text does not read as a wrapped
    continuation at all.
    """
    p_plain = _collapsed_plain_text(previous.get("content", ""))
    f_plain = _collapsed_plain_text(current.get("content", ""))
    if len(p_plain) < 40 or not f_plain or len(f_plain) > 90:
        return None
    f_paragraphs = [
        paragraph
        for paragraph in (current.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(f_paragraphs) > 1:
        return None
    if (
        _looks_like_pdf_paragraph_lead(f_plain)
        or _looks_like_pdf_list_marker(f_plain)
        or _PDF_BARE_FOOTNOTE_LEAD_RE.match(f_plain)
    ):
        return None
    if not _pdf_wrapped_math_fragment_prose_words(f_plain):
        return None

    script_lead = bool(_PDF_WRAPPED_MATH_SCRIPT_LEAD_RE.match(f_plain))
    lower_lead = bool(f_plain[:1].islower())
    dangling_math = bool(_PDF_WRAPPED_MATH_DANGLING_TAIL_RE.search(p_plain))
    hyphen_wrap = bool(_PDF_WRAPPED_MATH_HYPHEN_TAIL_RE.search(p_plain))
    dangling_continuation = dangling_math and (script_lead or lower_lead)
    if not (dangling_continuation or (hyphen_wrap and lower_lead)):
        return None
    return dangling_continuation


def _pdf_wrapped_math_rows_continue(
    p_rect: fitz.Rect,
    last_rect: fitz.Rect,
    f_rect: fitz.Rect,
    p_fontsize: float,
    dangling_continuation: bool,
) -> bool:
    """Whether the fragment sits where a wrapped continuation of the line would."""
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
        return dangling_continuation
    vertical_gap = float(f_rect.y0) - float(last_rect.y1)
    if vertical_gap < -max(float(f_rect.height), p_fontsize) * 0.60:
        return False
    if vertical_gap > max(p_fontsize * 1.90, 12.0):
        return False
    overlap = min(float(p_rect.x1), float(f_rect.x1)) - max(
        float(p_rect.x0), float(f_rect.x0)
    )
    return overlap >= min(float(p_rect.width), float(f_rect.width)) * 0.20


def _merge_pdf_wrapped_math_continuation_pair(
    previous: dict,
    current: dict,
    page_rect: fitz.Rect,
) -> dict:
    """Fold a wrapped-script continuation fragment back into its paragraph."""
    f_rect = _get_pdf_elem_rect(current)
    rect = fitz.Rect(_get_pdf_elem_rect(previous)) | fitz.Rect(f_rect)

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
    paragraphs = _pdf_wrapped_math_merged_paragraphs(previous, current, f_rect)
    if not paragraphs:
        paragraphs = [
            {
                "plain": plain,
                "rich": rich or plain,
                "source_bbox": [float(value) for value in rect],
                "source_lines": _pdf_wrapped_fragment_source_lines(current, f_rect),
                "margin_left": 0.0,
                "text_indent": 0.0,
                "gap_before": 0.0,
                "text_align": "left",
                "nowrap": None,
            }
        ]

    fontsize = float(previous.get("fontsize", DEFAULT_FONTSIZE))
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


def _first_pdf_text_paragraph(elem: dict) -> dict | None:
    return next(
        (
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ),
        None,
    )


def _pdf_wrapped_fragment_source_lines(current: dict, f_rect: fitz.Rect) -> list[dict]:
    """The fragment's recorded source lines, or one line spanning its box."""
    f_paragraph = _first_pdf_text_paragraph(current)
    f_lines = [
        dict(line)
        for line in ((f_paragraph or {}).get("source_lines") or [])
        if (line.get("plain") or "").strip()
    ]
    return f_lines or [
        {
            "plain": current.get("content", ""),
            "rich": current.get("rich_content") or current.get("content", ""),
            "bbox": [float(value) for value in f_rect],
        }
    ]


def _pdf_wrapped_math_merged_paragraphs(
    previous: dict,
    current: dict,
    f_rect: fitz.Rect,
) -> list[dict]:
    """Copies of the previous paragraphs with the fragment appended to the last.

    Returns ``[]`` when the previous element records no text paragraph.
    """
    paragraphs = [
        dict(paragraph)
        for paragraph in (previous.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if not paragraphs:
        return []
    f_paragraph = _first_pdf_text_paragraph(current) or {}
    last = paragraphs[-1]
    last["plain"] = _join_pdf_line_fragments(
        [
            last.get("plain", ""),
            f_paragraph.get("plain") or current.get("content", ""),
        ]
    )
    last["rich"] = _join_pdf_line_fragments(
        [
            last.get("rich") or last.get("plain", ""),
            f_paragraph.get("rich")
            or current.get("rich_content")
            or current.get("content", ""),
        ]
    )
    last_lines = [dict(line) for line in (last.get("source_lines") or [])]
    last_lines.extend(_pdf_wrapped_fragment_source_lines(current, f_rect))
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
    return paragraphs


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
        _get_pdf_elem_rect(elem)
        for elem in elements
        if elem.get("type") == ElementType.FORMULA_IMAGE
    ]
    blocker_rects = [
        _get_pdf_elem_rect(elem)
        for elem in elements
        if elem.get("type") in {ElementType.TABLE_IMAGE, ElementType.IMAGE}
    ]
    merge_count = 0
    index = 0
    while index < len(elements):
        current = elements[index]
        if current.get("type") != ElementType.TEXT:
            index += 1
            continue
        cursor = index + 1
        blocked = False
        while cursor < len(elements) and elements[cursor].get("type") != ElementType.TEXT:
            if elements[cursor].get("type") != ElementType.FORMULA_IMAGE:
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
    formulas = [elem for elem in elements or [] if elem.get("type") == ElementType.FORMULA_IMAGE]
    protected_rects = [
        _get_pdf_elem_rect(elem) for elem in formulas
    ]
    promoted = 0
    for elem in elements or []:
        if elem.get("type") != ElementType.TEXT:
            continue
        # Recovered table cells intentionally carry a larger render rectangle
        # so Chinese can wrap within the original cell.  Formula protection,
        # however, must classify the source glyphs rather than that available
        # whitespace: a unit row immediately below a header is not part of the
        # header merely because the future render rectangle reaches it.
        rect = (
            _get_pdf_source_ink_rect(elem)
            if elem.get("semantic_table_cell")
            else _get_pdf_elem_rect(elem)
        )
        touching = [
            formula
            for formula, protected in zip(formulas, protected_rects)
            if _pdf_rect_intersects_protected(rect, [protected], padding)
        ]
        if not touching:
            continue

        if _recorded_source_ink_clears_formulas(elem, touching, padding):
            # The element's outer bbox is diagonal across disjoint visual
            # lines, but no recorded source glyph touches the formula.  Keep
            # the prose translatable; assembly redacts the same precise source
            # rectangles and trims its render ladder around immutable math.
            _mark_pdf_formula_adjacent_text(elem, 0.0)
            continue

        intersections, shallow_edge_only = _pdf_formula_intersections(
            rect,
            touching,
            shallow_limit=max(float(elem.get("fontsize", 0.0)) * 0.35, 3.0),
        )
        overlap_ratio = (
            sum(intersection.get_area() for intersection in intersections)
            / max(rect.get_area(), 1.0)
        )
        if (
            elem.get("semantic_table_cell")
            and not intersections
            and not elem.get("inline_math_fragments")
        ):
            _mark_pdf_formula_adjacent_text(elem, 0.0)
            continue
        if shallow_edge_only and _formula_adjacent_prose_stays_translatable(
            elem, rect, overlap_ratio, bool(intersections)
        ):
            _mark_pdf_formula_adjacent_text(elem, round(overlap_ratio, 6))
            continue

        _promote_pdf_text_to_formula(elem, touching)
        promoted += 1
    return promoted


def _mark_pdf_formula_adjacent_text(elem: dict, overlap_ratio: float) -> None:
    elem["formula_adjacent_text"] = True
    elem["formula_overlap_ratio"] = overlap_ratio


def _recorded_source_ink_clears_formulas(
    elem: dict,
    touching: list[dict],
    padding: float,
) -> bool:
    """True when the element records source lines and none of them touch the formulas."""
    has_recorded_source_lines = any(
        paragraph.get("source_lines") or paragraph.get("source_line_bboxes")
        for paragraph in (elem.get("paragraphs") or [])
    )
    return has_recorded_source_lines and not any(
        _pdf_rect_intersects_protected(
            source_rect,
            [_get_pdf_elem_rect(formula)],
            padding,
        )
        for source_rect in _get_pdf_source_ink_rects(elem)
        for formula in touching
    )


def _pdf_formula_intersections(
    rect: fitz.Rect,
    touching: list[dict],
    *,
    shallow_limit: float,
) -> tuple[list[fitz.Rect], bool]:
    """Actual overlaps with ``rect`` and whether each is a shallow edge sliver.

    Stops at the first overlap that is not a sliver; that overlap is still
    included in the returned list.
    """
    intersections = []
    for formula in touching:
        intersection = rect & _get_pdf_elem_rect(formula)
        if intersection.is_empty or intersection.get_area() <= 0:
            continue
        intersections.append(intersection)
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
            return intersections, False
    return intersections, True


def _formula_adjacent_prose_stays_translatable(
    elem: dict,
    rect: fitz.Rect,
    overlap_ratio: float,
    overlaps: bool,
) -> bool:
    """Whether linguistically substantial text may stay translatable near math.

    A formula's font box can graze the top or bottom of an adjacent prose line
    even though no prose glyph belongs to the formula.  The former
    all-or-nothing rule promoted the whole paragraph in that situation,
    leaving acknowledgements, appendix prose and table notes untranslated.
    The caller admits text only when every actual overlap is a shallow edge
    sliver; assembly still subtracts the immutable formula rectangles from
    redaction and trims the insertion rectangle before drawing, so formula
    pixels stay protected.
    """
    plain = _collapsed_plain_text(elem.get("content", ""))
    prose_words = re.findall(r"[A-Za-z]{3,}", plain)
    fontsize = float(elem.get("fontsize", 0.0))
    inline_math = elem.get("inline_math_fragments")
    safe_long_prose = (
        len(plain) >= 160
        and len(prose_words) >= 12
        and rect.height >= max(fontsize * 1.8, 20.0)
    )
    safe_short_heading = (
        len(plain) <= 100
        and len(prose_words) >= 2
        and (bool(elem.get("bold")) or _looks_like_heading_text(plain))
    )
    safe_prose_line = (
        len(plain) >= 60
        and len(prose_words) >= 8
        and rect.height <= max(fontsize * 1.8, 22.0)
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
        and not inline_math
        and not any(_is_unicode_math_symbol(char) for char in plain)
        and bool(re.search(r"[.!?][\"'’”)]?\s*$", plain))
        and rect.height <= max(fontsize * 1.8, 22.0)
    )
    safe_citation_label_line = (
        not overlaps
        and not inline_math
        and _translate_pdf_citation_labels(plain) != plain
    )
    ordinary_safe_overlap = (
        safe_long_prose or safe_short_heading or safe_prose_line
    ) and overlap_ratio <= 0.08
    short_sentence_safe_overlap = safe_short_sentence_line and overlap_ratio <= 0.28
    return (
        (ordinary_safe_overlap or short_sentence_safe_overlap or safe_citation_label_line)
        and not elem.get("table_hint")
        and (not inline_math or not overlaps)
        and not _looks_like_pdf_paragraph_lead(plain)
        and not _looks_like_pdf_list_marker(plain)
        and not _is_math_block(plain)
    )


def _promote_pdf_text_to_formula(elem: dict, touching: list[dict]) -> None:
    reasons = {"formula_geometry_overlap"}
    for formula in touching:
        reasons.update(formula.get("math_reasons") or [])
    elem["type"] = ElementType.FORMULA_IMAGE
    elem["math_reasons"] = sorted(reasons)
    elem["math_mixed"] = True
    elem["math_line_count"] = max(1, len(elem.get("paragraphs") or []))
    elem["math_candidate_symbol_count"] = _unicode_math_symbol_count(
        _plain_text(elem.get("content", ""))
    )
    elem["skip_translate_reason"] = SkipReason.FORMULA_GEOMETRY_OVERLAP
