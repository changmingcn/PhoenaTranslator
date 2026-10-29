"""Merge passes that join paragraph fragments on one page: detached list
markers, wrapped visual lines, numbered-item and strong continuations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import fitz

from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_primary_paragraph_rect,
    _get_pdf_render_bbox,
)
from phoena_translator.pdf.patterns import _PDF_BARE_FOOTNOTE_LEAD_RE
from phoena_translator.pdf.text_shape import (
    _collapsed_plain_text,
    _looks_like_pdf_list_marker,
)

log = logging.getLogger("translator")
from phoena_translator.pdf.extract.paragraphs import (
    _is_pdf_single_visual_line_element,
    _make_pdf_text_element_from_lines,
    _pdf_visual_line_info_from_element,
)
from phoena_translator.pdf.extract.superscripts import (
    _pdf_element_footnote_definition_marker,
    _pdf_source_line_footnote_definition_marker,
)
from phoena_translator.pdf.model import (
    DEFAULT_FONTSIZE,
    PARAGRAPH_MAX_FIRST_LINE_SHARE,
    PARAGRAPH_MAX_HANGING_SHARE,
    PARAGRAPH_MAX_MARGIN_SHARE,
    ElementType,
    LayoutClass,
    indent_step_threshold,
)
from phoena_translator.pdf.semantics.grouping import (
    ParagraphIndent,
    build_merged_text_element,
    dominant_group_style,
    first_and_body_x0,
    group_adjacent,
    group_plain_and_rich,
    group_rect,
    group_source_lines,
    merge_groups,
)
from phoena_translator.pdf.semantics.references import (
    _looks_like_pdf_reference_entry_lead,
    _pdf_reference_entry_lead_counts,
)
from phoena_translator.pdf.semantics.toc import _parse_pdf_toc_leader
from phoena_translator.pdf.text_shape import (
    _is_heading_like_elem,
    _join_pdf_line_fragments,
    _looks_like_pdf_paragraph_lead,
    _pdf_text_colors_semantically_compatible,
)

_PDF_SEMANTIC_TERMINAL_RE = re.compile(
    r"[.!?。！？:：;；][\"'’”)]*$"
)


def _pdf_detached_list_marker_target_score(
    marker: dict,
    candidate: dict,
    page_rect: fitz.Rect,
) -> float | None:
    """Score prose whose first visual line belongs to a detached list marker."""
    if marker is candidate:
        return None
    for elem in (marker, candidate):
        if (
            elem.get("type") != ElementType.TEXT
            or elem.get("table_hint")
            or elem.get("preserve_source_style")
            or elem.get("non_horizontal")
        ):
            return None

    marker_plain = _collapsed_plain_text(marker.get("content", ""))
    candidate_plain = _collapsed_plain_text(candidate.get("content", ""))
    if (
        not _looks_like_pdf_list_marker(marker_plain)
        or not candidate_plain
        or _looks_like_pdf_list_marker(candidate_plain)
        or _looks_like_pdf_paragraph_lead(candidate_plain)
        or len(candidate_plain) < 8
    ):
        return None

    candidate_paragraphs = [
        paragraph
        for paragraph in (candidate.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(candidate_paragraphs) != 1:
        return None

    marker_rect = _get_pdf_primary_paragraph_rect(marker)
    candidate_rect = _get_pdf_primary_paragraph_rect(candidate)
    source_lines = candidate_paragraphs[0].get("source_lines") or []
    first_line_rect = (
        fitz.Rect(source_lines[0]["bbox"])
        if source_lines and source_lines[0].get("bbox")
        else fitz.Rect(candidate_rect)
    )
    if first_line_rect.x0 <= marker_rect.x0:
        return None

    fontsize = max(float(candidate.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    horizontal_gap = first_line_rect.x0 - marker_rect.x1
    if horizontal_gap < -max(fontsize * 0.35, 3.0):
        return None
    if horizontal_gap > max(fontsize * 5.0, page_rect.width * 0.12, 42.0):
        return None

    vertical_overlap = max(
        0.0,
        min(marker_rect.y1, first_line_rect.y1)
        - max(marker_rect.y0, first_line_rect.y0),
    )
    overlap_ratio = vertical_overlap / max(
        min(marker_rect.height, first_line_rect.height),
        1.0,
    )
    center_delta = abs(
        (marker_rect.y0 + marker_rect.y1) / 2.0
        - (first_line_rect.y0 + first_line_rect.y1) / 2.0
    )
    if overlap_ratio < 0.35 and center_delta > max(fontsize * 0.65, 6.0):
        return None

    return center_delta + max(horizontal_gap, 0.0) * 0.10


def _merge_pdf_detached_list_marker_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """Attach standalone ``(b)``/bullet objects to their prose paragraph.

    A number of official reports put the list marker in a separate PDF block
    while the italic lead and roman body occupy one neighboring block.  The
    marker and body still form one semantic paragraph and must be submitted to
    the model and redrawn as a single unit.
    """
    merged_elements = list(elements)
    removed_indices: set[int] = set()
    for marker_index, marker in enumerate(merged_elements):
        marker_plain = _collapsed_plain_text(marker.get("content", ""))
        if (
            marker.get("type") != ElementType.TEXT
            or not _looks_like_pdf_list_marker(marker_plain)
            or marker.get("table_hint")
            or marker.get("preserve_source_style")
            or marker.get("non_horizontal")
        ):
            continue

        scored_targets = []
        for candidate_index, candidate in enumerate(merged_elements):
            if candidate_index in removed_indices:
                continue
            score = _pdf_detached_list_marker_target_score(
                marker,
                candidate,
                page_rect,
            )
            if score is not None:
                scored_targets.append((score, candidate_index))
        if not scored_targets:
            continue

        _, target_index = min(scored_targets)
        target = merged_elements[target_index]
        target_paragraph = next(
            paragraph
            for paragraph in (target.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        )
        marker_rect = _get_pdf_primary_paragraph_rect(marker)
        target_rect = _get_pdf_primary_paragraph_rect(target)
        merged_rect = marker_rect | target_rect

        target_plain = _collapsed_plain_text(target.get("content", ""))
        target_rich = (target.get("rich_content") or target.get("content", "")).strip()
        marker_rich = (marker.get("rich_content") or marker.get("content", "")).strip()
        merged_plain = f"{marker_plain} {target_plain}".strip()
        merged_rich = f"{marker_rich} {target_rich}".strip()

        source_lines = [
            dict(line)
            for line in (target_paragraph.get("source_lines") or [])
            if (line.get("plain") or "").strip()
        ]
        if source_lines:
            first_line = dict(source_lines[0])
            first_line_rect = (
                fitz.Rect(first_line.get("bbox", target_rect)) | marker_rect
            )
            first_line.update(
                {
                    "plain": f"{marker_plain} {first_line.get('plain', '').strip()}".strip(),
                    "rich": f"{marker_rich} {(first_line.get('rich') or first_line.get('plain', '')).strip()}".strip(),
                    "bbox": [float(value) for value in first_line_rect],
                }
            )
            source_lines[0] = first_line
        else:
            source_lines = [
                {
                    "plain": merged_plain,
                    "rich": merged_rich,
                    "bbox": [float(value) for value in merged_rect],
                }
            ]

        body_x0 = float(target_rect.x0)
        paragraph = dict(target_paragraph)
        paragraph.update(
            {
                "plain": merged_plain,
                "rich": merged_rich,
                "source_bbox": [float(value) for value in merged_rect],
                "source_line_bboxes": [
                    [float(value) for value in line["bbox"]] for line in source_lines
                ],
                "source_lines": source_lines,
                "margin_left": min(
                    max(body_x0 - merged_rect.x0, 0.0),
                    merged_rect.width * PARAGRAPH_MAX_MARGIN_SHARE,
                ),
                "text_indent": max(
                    -merged_rect.width * PARAGRAPH_MAX_HANGING_SHARE,
                    float(marker_rect.x0) - body_x0,
                ),
                "gap_before": 0.0,
                "text_align": "left",
                "nowrap": None,
                "first_line_indent": False,
                "toc_leader": _parse_pdf_toc_leader(merged_plain),
            }
        )

        merged = dict(target)
        merged.update(
            {
                "y": float(merged_rect.y0),
                "x": float(merged_rect.x0),
                "rect": fitz.Rect(merged_rect),
                "bbox": [float(value) for value in merged_rect],
                "render_bbox": _get_pdf_render_bbox(
                    merged_rect,
                    page_rect,
                    [paragraph],
                    float(target.get("fontsize", DEFAULT_FONTSIZE)),
                ),
                "content": merged_plain,
                "rich_content": merged_rich
                if (target.get("rich_content") or marker.get("rich_content"))
                else None,
                "paragraphs": [paragraph],
                "superscript_runs": [
                    dict(run)
                    for elem in (marker, target)
                    for run in (elem.get("superscript_runs") or [])
                ],
                "inline_math_fragments": [
                    dict(fragment)
                    for elem in (marker, target)
                    for fragment in (elem.get("inline_math_fragments") or [])
                ],
                "merged_visual_line_count": max(
                    int(target.get("merged_visual_line_count") or 1),
                    len(source_lines),
                ),
                "source_line_bboxes": [
                    [float(value) for value in line["bbox"]] for line in source_lines
                ],
                "detached_list_marker_merged": True,
            }
        )
        merged_elements[target_index] = merged
        removed_indices.add(marker_index)

    result = [
        elem
        for index, elem in enumerate(merged_elements)
        if index not in removed_indices
    ]
    result.sort(
        key=lambda elem: (
            elem["y"],
            elem.get(
                "x",
                elem.get("bbox", [0])[0] if isinstance(elem.get("bbox"), list) else 0,
            ),
        )
    )
    return result


def _pdf_semantic_continuation_elements_can_merge(
    group: list[dict],
    current: dict,
    page_rect: fitz.Rect,
) -> bool:
    """Return whether ``current`` continues a split numbered/bulleted item."""
    if not group:
        return False
    first = group[0]
    previous = group[-1]
    for elem in (first, previous, current):
        if (
            elem.get("type") != ElementType.TEXT
            or elem.get("table_hint")
            or elem.get("preserve_source_style")
            or elem.get("non_horizontal")
            or elem.get("skip_translate_reason")
            or len(
                [
                    paragraph
                    for paragraph in (elem.get("paragraphs") or [])
                    if (paragraph.get("plain") or "").strip()
                ]
            )
            != 1
        ):
            return False

    first_plain = _collapsed_plain_text(first.get("content", ""))
    current_plain = _collapsed_plain_text(current.get("content", ""))
    if (
        not _looks_like_pdf_paragraph_lead(first_plain)
        or not current_plain
        or _looks_like_pdf_paragraph_lead(current_plain)
        or _looks_like_pdf_list_marker(current_plain)
        or _pdf_element_footnote_definition_marker(current)
        or _is_heading_like_elem(current)
    ):
        return False

    previous_rect = _get_pdf_primary_paragraph_rect(previous)
    current_rect = _get_pdf_primary_paragraph_rect(current)
    first_rect = _get_pdf_primary_paragraph_rect(first)
    fontsize = max(float(previous.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    current_fontsize = max(float(current.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    if abs(fontsize - current_fontsize) > max(fontsize * 0.10, 0.8):
        return False
    if bool(previous.get("bold")) != bool(current.get("bold")):
        return False
    if not _pdf_text_colors_semantically_compatible(
        previous.get("color", 0),
        current.get("color", 0),
    ):
        return False

    vertical_gap = current_rect.y0 - previous_rect.y1
    if vertical_gap < -min(previous_rect.height, current_rect.height) * 0.25:
        return False
    if vertical_gap > max(fontsize * 1.15, 9.0):
        return False

    # A continuation is indented past the bullet/number marker.  Requiring
    # that geometry prevents a following ordinary paragraph from being
    # swallowed into the final list item.
    expected_body_x0 = first_rect.x0 + max(fontsize * 0.45, 3.0)
    if current_rect.x0 < expected_body_x0:
        return False
    overlap = max(
        0.0,
        min(previous_rect.x1, current_rect.x1) - max(previous_rect.x0, current_rect.x0),
    )
    if overlap / max(min(previous_rect.width, current_rect.width), 1.0) < 0.30:
        return False
    return current_rect.x0 <= first_rect.x0 + max(
        fontsize * 5.5,
        page_rect.width * 0.16,
        48.0,
    )


def _merge_pdf_semantic_continuation_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """Join multi-line fragments that belong to one list paragraph."""

    def merge(group: list[dict]) -> dict:
        rect = group_rect(group)
        plain, rich = group_plain_and_rich(group)
        source_lines = group_source_lines(group)
        first_x0, body_x0 = first_and_body_x0(source_lines)
        return build_merged_text_element(
            group,
            page_rect,
            rect=rect,
            plain=plain,
            rich=rich,
            source_lines=source_lines,
            indent=ParagraphIndent(
                margin_left=min(max(body_x0 - rect.x0, 0.0), rect.width * PARAGRAPH_MAX_MARGIN_SHARE),
                text_indent=max(-rect.width * PARAGRAPH_MAX_HANGING_SHARE, first_x0 - body_x0),
            ),
            toc_leader=_parse_pdf_toc_leader(plain),
            render_fontsize=float(group[0].get("fontsize", DEFAULT_FONTSIZE)),
        )

    merged_elements, _ = merge_groups(
        group_adjacent(
            elements,
            lambda elem: _looks_like_pdf_paragraph_lead(
                _collapsed_plain_text(elem.get("content", ""))
            ),
            lambda group, elem: _pdf_semantic_continuation_elements_can_merge(
                group, elem, page_rect
            ),
        ),
        merge,
    )
    return merged_elements


def _pdf_wrapped_line_group_can_start(elem: dict, page_rect: fitz.Rect) -> bool:
    if not _is_pdf_single_visual_line_element(elem) or _is_heading_like_elem(elem):
        return False
    plain = _collapsed_plain_text(elem.get("content", ""))
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", plain)
    rect = _get_pdf_elem_rect(elem)
    numbered_body = _looks_like_pdf_paragraph_lead(plain) and len(words) >= 6
    prose_width = rect.width >= max(page_rect.width * 0.50, 220.0)
    return numbered_body or len(plain) >= 58 or (len(words) >= 9 and prose_width)


def _pdf_wrapped_line_elements_can_merge(
    group: list[dict],
    current: dict,
    page_rect: fitz.Rect,
) -> bool:
    if not group or not _is_pdf_single_visual_line_element(current):
        return False
    previous = group[-1]
    if not _is_pdf_single_visual_line_element(previous):
        return False

    current_plain = _collapsed_plain_text(current.get("content", ""))
    if (
        not current_plain
        or _looks_like_pdf_paragraph_lead(current_plain)
        or _looks_like_pdf_list_marker(current_plain)
        or _pdf_element_footnote_definition_marker(current)
        or _is_heading_like_elem(current)
    ):
        return False

    first_plain = _collapsed_plain_text(group[0].get("content", ""))
    if not _looks_like_pdf_paragraph_lead(first_plain) and len(first_plain) < 45:
        return False

    previous_rect = _get_pdf_elem_rect(previous)
    current_rect = _get_pdf_elem_rect(current)
    previous_fontsize = max(float(previous.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    current_fontsize = max(float(current.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    if abs(previous_fontsize - current_fontsize) > max(previous_fontsize * 0.08, 0.7):
        return False
    if bool(previous.get("bold")) != bool(current.get("bold")):
        return False
    if not _pdf_text_colors_semantically_compatible(
        previous.get("color", 0),
        current.get("color", 0),
    ):
        return False

    if len(group) >= 2:
        body_x0 = min(_get_pdf_elem_rect(elem).x0 for elem in group[1:])
        first_line_indent = current_rect.x0 - body_x0
        if first_line_indent >= max(previous_fontsize * 1.5, 18.0):
            return False

    previous_height = max(previous_rect.height, 1.0)
    current_height = max(current_rect.height, 1.0)
    vertical_gap = current_rect.y0 - previous_rect.y1
    if vertical_gap < -min(previous_height, current_height) * 0.35:
        return False
    if vertical_gap > max(
        previous_fontsize * 1.45,
        previous_height * 1.30,
        current_height * 1.30,
        8.0,
    ):
        return False

    first_rect = _get_pdf_elem_rect(group[0])
    returns_to_reference_lead = abs(current_rect.x0 - first_rect.x0) <= max(
        previous_fontsize * 0.50, 4.0
    ) and _looks_like_pdf_reference_entry_lead(current_plain)
    if returns_to_reference_lead:
        # A hanging bibliography entry returns from its indented body to the
        # left margin for the next author.  Without this boundary, one source
        # element can contain one complete citation plus the beginning of the
        # next, while the latter's final line is translated separately.
        return False

    overlap = max(
        0.0,
        min(previous_rect.x1, current_rect.x1) - max(previous_rect.x0, current_rect.x0),
    )
    if overlap / max(min(previous_rect.width, current_rect.width), 1.0) < 0.35:
        return False
    if abs(current_rect.x0 - previous_rect.x0) > max(
        previous_fontsize * 5.0,
        page_rect.width * 0.14,
        42.0,
    ):
        return False
    return True


def _merge_pdf_wrapped_line_group(group: list[dict], page_rect: fitz.Rect) -> dict | None:
    """Rebuild one visual-line run as a paragraph element (``None`` if unsafe)."""
    line_infos = [_pdf_visual_line_info_from_element(elem) for elem in group]
    merged = _make_pdf_text_element_from_lines(line_infos, page_rect)
    if merged is None:
        return None

    plain_text = _join_pdf_line_fragments([line["plain"] for line in line_infos])
    rich_text = _join_pdf_line_fragments([line["rich"] for line in line_infos])
    merged_rect = _get_pdf_elem_rect(merged)
    first_x0 = float(line_infos[0]["x0"])
    body_x0 = min(
        (float(line["x0"]) for line in line_infos[1:]),
        default=first_x0,
    )
    hanging_lead = _looks_like_pdf_paragraph_lead(line_infos[0]["plain"])
    if hanging_lead:
        margin_left = max(0.0, body_x0 - merged_rect.x0)
        text_indent = max(-merged_rect.width * PARAGRAPH_MAX_HANGING_SHARE, first_x0 - body_x0)
    else:
        margin_left = 0.0
        text_indent = max(0.0, first_x0 - body_x0)

    paragraph = {
        "plain": plain_text,
        "rich": rich_text,
        "source_bbox": [float(value) for value in merged_rect],
        "source_line_bboxes": [
            [
                float(line["x0"]),
                float(line["y0"]),
                float(line["x1"]),
                float(line["y1"]),
            ]
            for line in line_infos
        ],
        "source_lines": [
            {
                "plain": line.get("plain", ""),
                "rich": line.get("rich", line.get("plain", "")),
                "bbox": [
                    float(line["x0"]),
                    float(line["y0"]),
                    float(line["x1"]),
                    float(line["y1"]),
                ],
            }
            for line in line_infos
        ],
        "margin_left": min(margin_left, merged_rect.width * PARAGRAPH_MAX_MARGIN_SHARE),
        "text_indent": min(text_indent, merged_rect.width * PARAGRAPH_MAX_FIRST_LINE_SHARE),
        "gap_before": 0.0,
        "text_align": "left",
        "nowrap": None,
        "first_line_indent": False,
        "toc_leader": _parse_pdf_toc_leader(plain_text),
    }
    merged["content"] = plain_text
    if any(elem.get("rich_content") for elem in group) or any(
        elem.get("superscript_runs") for elem in group
    ):
        merged["rich_content"] = rich_text
    else:
        merged["rich_content"] = None
    merged["paragraphs"] = [paragraph]
    merged["render_bbox"] = _get_pdf_render_bbox(
        merged_rect,
        page_rect,
        [paragraph],
        float(merged.get("fontsize", DEFAULT_FONTSIZE)),
    )
    pitches = sorted(
        float(line_infos[pos + 1]["y0"]) - float(line_infos[pos]["y0"])
        for pos in range(len(line_infos) - 1)
        if float(line_infos[pos + 1]["y0"]) > float(line_infos[pos]["y0"])
    )
    if pitches:
        merged["line_height"] = max(
            float(merged.get("line_height", 0.0)),
            pitches[len(pitches) // 2],
        )
    merged["merged_visual_line_count"] = len(group)
    merged["source_line_bboxes"] = [
        [float(value) for value in _get_pdf_elem_rect(elem)] for elem in group
    ]
    return merged


def _merge_pdf_wrapped_line_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """Coalesce visual-line PDF blocks into semantic paragraph elements.

    Some official PDFs contain one text object per justified visual line.
    Translation and redraw must operate on the union paragraph rectangle, not
    on each source line independently.  Geometry/style gates keep tables,
    columns, headings, formulas, and numbered successor paragraphs separate.
    """
    merged_elements = []
    for group in group_adjacent(
        elements,
        lambda elem: _pdf_wrapped_line_group_can_start(elem, page_rect),
        lambda group, elem: _pdf_wrapped_line_elements_can_merge(group, elem, page_rect),
    ):
        merged = _merge_pdf_wrapped_line_group(group, page_rect) if len(group) > 1 else None
        if merged is None:
            merged_elements.extend(group)
        else:
            merged_elements.append(merged)
    return merged_elements


def _pdf_strong_continuation_element_is_eligible(elem: dict) -> bool:
    if (
        elem.get("type") != ElementType.TEXT
        or elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
        or elem.get("watermark")
        or elem.get("layout_class") in {LayoutClass.HEADING, LayoutClass.FOOTER, LayoutClass.TABLE}
        or elem.get("glossary_cell_hint")
        or elem.get("glossary_term_hint")
        or elem.get("glossary_definition_hint")
        or _is_heading_like_elem(elem)
    ):
        return False
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    return bool(len(paragraphs) == 1 and not paragraphs[0].get("toc_leader"))


@dataclass(frozen=True)
class StrongContinuationThresholds:
    """Gates for joining a post-classification paragraph split (``em`` = font size)."""

    min_first_chars: int = 35
    min_previous_chars: int = 8
    min_current_chars: int = 12
    fontsize_tolerance_em: float = 0.15
    fontsize_tolerance_min: float = 1.0
    max_vertical_overlap_share: float = 0.40
    max_gap_em: float = 0.55
    max_gap_min: float = 4.5
    max_outdent_em: float = 0.35
    max_outdent_min: float = 3.0
    max_indent_em: float = 4.0
    max_indent_page_share: float = 0.08
    max_indent_min: float = 32.0
    min_horizontal_overlap: float = 0.35
    # A capitalised continuation needs at least this hanging indent.
    capitalized_min_indent_em: float = 0.80
    capitalized_min_indent: float = 6.0


STRONG = StrongContinuationThresholds()


def _pdf_strong_continuation_elements_can_merge(
    group: list[dict],
    current: dict,
    page_rect: fitz.Rect,
) -> bool:
    """Recognize a split block that is unambiguously the same paragraph.

    This is intentionally a post-classification rule.  It handles footnotes,
    superscript/style boundaries and hanging reference continuations while
    excluding headings, TOC rows, glossary cells, a new footnote number and a
    new left-indented reference entry.
    """
    if not group or not _pdf_strong_continuation_element_is_eligible(current):
        return False
    first = group[0]
    previous = group[-1]
    if not all(
        _pdf_strong_continuation_element_is_eligible(elem) for elem in (first, previous)
    ):
        return False

    first_plain = _collapsed_plain_text(first.get("content", ""))
    previous_plain = _collapsed_plain_text(previous.get("content", ""))
    current_plain = _collapsed_plain_text(current.get("content", ""))
    if (
        len(first_plain) < STRONG.min_first_chars
        or len(previous_plain) < STRONG.min_previous_chars
        or len(current_plain) < STRONG.min_current_chars
        or _PDF_SEMANTIC_TERMINAL_RE.search(previous_plain)
        or re.search(
            r"[.\u2024\u2025\u2026\u00b7·]{6,}", previous_plain + current_plain
        )
        or _looks_like_pdf_paragraph_lead(current_plain)
        or _looks_like_pdf_list_marker(current_plain)
        or _PDF_BARE_FOOTNOTE_LEAD_RE.match(current_plain)
        or _pdf_element_footnote_definition_marker(current)
    ):
        return False

    previous_rect = _get_pdf_primary_paragraph_rect(previous)
    current_rect = _get_pdf_primary_paragraph_rect(current)
    previous_fontsize = max(float(previous.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    current_fontsize = max(float(current.get("fontsize", DEFAULT_FONTSIZE)), 1.0)
    if abs(previous_fontsize - current_fontsize) > max(
        previous_fontsize * STRONG.fontsize_tolerance_em,
        STRONG.fontsize_tolerance_min,
    ):
        return False

    vertical_gap = current_rect.y0 - previous_rect.y1
    if vertical_gap < -min(previous_rect.height, current_rect.height) * STRONG.max_vertical_overlap_share:
        return False
    if vertical_gap > max(previous_fontsize * STRONG.max_gap_em, STRONG.max_gap_min):
        return False

    x_shift = current_rect.x0 - previous_rect.x0
    if x_shift < -max(previous_fontsize * STRONG.max_outdent_em, STRONG.max_outdent_min):
        # Hanging references return left for the next entry.  That is a new
        # semantic paragraph, not a continuation of the prior entry.
        return False
    if x_shift > max(
        previous_fontsize * STRONG.max_indent_em,
        page_rect.width * STRONG.max_indent_page_share,
        STRONG.max_indent_min,
    ):
        return False

    overlap = max(
        0.0,
        min(previous_rect.x1, current_rect.x1) - max(previous_rect.x0, current_rect.x0),
    )
    if overlap / max(min(previous_rect.width, current_rect.width), 1.0) < STRONG.min_horizontal_overlap:
        return False

    # A same-column capitalized line is commonly a new participant, label or
    # reference entry.  Capitalized continuations remain safe when the source
    # uses an explicit hanging indent.
    lexical_probe = re.sub(r"^[\s\"'“”‘’(\[]+", "", current_plain)
    if (
        lexical_probe
        and (lexical_probe[0].isupper() or lexical_probe[0].isdigit())
        and x_shift < max(
            previous_fontsize * STRONG.capitalized_min_indent_em,
            STRONG.capitalized_min_indent,
        )
    ):
        return False
    return True


def _merge_pdf_strong_continuation_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> tuple[list[dict], int]:
    """Merge post-classification strong semantic continuations."""

    def merge(group: list[dict]) -> dict:
        rect = group_rect(group)
        plain, rich = group_plain_and_rich(group)
        source_lines = group_source_lines(group)
        first_x0, body_x0 = first_and_body_x0(source_lines)
        style = dominant_group_style(group)
        fontsize = style["fontsize"]
        hanging_indent = body_x0 - first_x0 >= indent_step_threshold(fontsize)
        first_line_indent = first_x0 - body_x0 >= indent_step_threshold(fontsize)
        if hanging_indent:
            margin_left = max(0.0, body_x0 - rect.x0)
            text_indent = first_x0 - body_x0
        else:
            margin_left = max(0.0, min(first_x0, body_x0) - rect.x0)
            text_indent = max(0.0, first_x0 - body_x0) if first_line_indent else 0.0
        return build_merged_text_element(
            group,
            page_rect,
            rect=rect,
            plain=plain,
            rich=rich,
            source_lines=source_lines,
            indent=ParagraphIndent(
                margin_left=min(margin_left, rect.width * PARAGRAPH_MAX_MARGIN_SHARE),
                text_indent=max(-rect.width * PARAGRAPH_MAX_HANGING_SHARE, min(text_indent, rect.width * PARAGRAPH_MAX_FIRST_LINE_SHARE)),
                first_line_indent=first_line_indent,
            ),
            toc_leader=_parse_pdf_toc_leader(plain),
            render_fontsize=fontsize,
            style=style,
            markers={
                "strong_semantic_continuation_merged": len(group),
                "table_hint": False,
                "preserve_source_style": False,
            },
        )

    return merge_groups(
        group_adjacent(
            elements,
            _pdf_strong_continuation_element_is_eligible,
            lambda group, elem: _pdf_strong_continuation_elements_can_merge(
                group, elem, page_rect
            ),
        ),
        merge,
    )


def _pdf_strong_continuation_residual_pairs(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[tuple[int, int]]:
    """Return strong same-paragraph pairs left behind after coalescing.

    Keeping this as a runtime invariant makes semantic paragraph extraction
    self-checking: a newly encountered PDF cannot silently fall back to
    line/block-level translation when the geometry and language signals say
    that two adjacent elements are unambiguously one paragraph.
    """
    return [
        (index, index + 1)
        for index in range(len(elements) - 1)
        if _pdf_strong_continuation_elements_can_merge(
            [elements[index]],
            elements[index + 1],
            page_rect,
        )
    ]


def _pdf_internal_paragraph_lead_residuals(
    elements: list[dict],
) -> list[tuple[int, int, str]]:
    """Find a new list/row lead buried after the first line of a paragraph.

    A paragraph builder or later coalescer must split before a numbered,
    bulleted, or repeated-hyphen lead.  Leaving such a lead inside the prior
    paragraph couples unrelated rows in one LLM request and redraw rectangle.
    """
    # Bibliography entries have their own stronger page-context and
    # author/year invariants below.  A hanging continuation may legitimately
    # begin with an author initial (``S. (eds).``) or a date (``2012.``), both
    # of which resemble ordinary list markers.  Exempt only elements already
    # confirmed as one reference lead in a bibliography context; multiple
    # entries in one element remain fail-closed through
    # ``_pdf_reference_entry_structure_residuals``.
    reference_context, reference_lead_counts = _pdf_reference_entry_lead_counts(
        elements
    )
    residuals = []
    for element_index, elem in enumerate(elements or []):
        if elem.get("type") != ElementType.TEXT:
            continue
        superscript_footnote_seen = False
        for paragraph in elem.get("paragraphs") or []:
            source_lines = [
                line
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            for line_index, line in enumerate(source_lines):
                marker = _pdf_source_line_footnote_definition_marker(line)
                if marker and superscript_footnote_seen:
                    plain = _collapsed_plain_text(line.get("plain", ""))
                    residuals.append((element_index, line_index, plain[:80]))
                superscript_footnote_seen = bool(superscript_footnote_seen or marker)
        # Multiple bullets or numbered instructions are legitimate inside one
        # detector-proven native table cell.  The exact cell boundary already
        # prevents the unsafe cross-row merge this invariant guards against,
        # while splitting those items into overlapping render boxes would
        # damage the table layout.
        if elem.get("semantic_native_table_cell"):
            continue
        if reference_context and reference_lead_counts.get(element_index) == 1:
            continue
        for paragraph in elem.get("paragraphs") or []:
            source_lines = [
                line
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            for line_index, line in enumerate(source_lines[1:], start=1):
                plain = _collapsed_plain_text(line.get("plain", ""))
                if _looks_like_pdf_paragraph_lead(plain):
                    residuals.append((element_index, line_index, plain[:80]))
    return residuals
