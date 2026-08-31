"""Semantic continuations helpers for deterministic PDF processing."""

from __future__ import annotations

import logging
import re

import fitz

from phoena_translator.pdf.types import (
    _PDF_BARE_FOOTNOTE_LEAD_RE,
    _PDF_SEMANTIC_TERMINAL_RE,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_primary_paragraph_rect,
    _get_pdf_render_bbox,
)
from phoena_translator.pdf.math_detection import (
    _looks_like_pdf_list_marker,
    _plain_text,
)

log = logging.getLogger("translator")

from phoena_translator.pdf.semantic_references import (
    _looks_like_pdf_reference_entry_lead,
    _pdf_reference_entry_lead_counts,
)
from phoena_translator.pdf.semantic_text import (
    _is_heading_like_elem,
    _is_pdf_single_visual_line_element,
    _join_pdf_line_fragments,
    _looks_like_pdf_paragraph_lead,
    _make_pdf_text_element_from_lines,
    _parse_pdf_toc_leader,
    _pdf_element_footnote_definition_marker,
    _pdf_source_line_footnote_definition_marker,
    _pdf_text_colors_semantically_compatible,
    _pdf_visual_line_info_from_element,
    _pick_pdf_dominant_value,
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
            elem.get("type") != "text"
            or elem.get("table_hint")
            or elem.get("preserve_source_style")
            or elem.get("non_horizontal")
        ):
            return None

    marker_plain = re.sub(r"\s+", " ", _plain_text(marker.get("content", ""))).strip()
    candidate_plain = re.sub(
        r"\s+", " ", _plain_text(candidate.get("content", ""))
    ).strip()
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

    fontsize = max(float(candidate.get("fontsize", 11.0)), 1.0)
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
        marker_plain = re.sub(
            r"\s+", " ", _plain_text(marker.get("content", ""))
        ).strip()
        if (
            marker.get("type") != "text"
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

        target_plain = re.sub(
            r"\s+", " ", _plain_text(target.get("content", ""))
        ).strip()
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
                    merged_rect.width * 0.35,
                ),
                "text_indent": max(
                    -merged_rect.width * 0.18,
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
                    float(target.get("fontsize", 11.0)),
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
            elem.get("type") != "text"
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

    first_plain = re.sub(r"\s+", " ", _plain_text(first.get("content", ""))).strip()
    current_plain = re.sub(r"\s+", " ", _plain_text(current.get("content", ""))).strip()
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
    fontsize = max(float(previous.get("fontsize", 11.0)), 1.0)
    current_fontsize = max(float(current.get("fontsize", 11.0)), 1.0)
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
    merged_elements: list[dict] = []
    index = 0
    while index < len(elements):
        first = elements[index]
        first_plain = re.sub(r"\s+", " ", _plain_text(first.get("content", ""))).strip()
        if not _looks_like_pdf_paragraph_lead(first_plain):
            merged_elements.append(first)
            index += 1
            continue

        group = [first]
        cursor = index + 1
        while cursor < len(elements) and _pdf_semantic_continuation_elements_can_merge(
            group,
            elements[cursor],
            page_rect,
        ):
            group.append(elements[cursor])
            cursor += 1
        if len(group) < 2:
            merged_elements.append(first)
            index += 1
            continue

        merged = dict(first)
        rect = _get_pdf_primary_paragraph_rect(first)
        for elem in group[1:]:
            rect |= _get_pdf_primary_paragraph_rect(elem)
        plain = _join_pdf_line_fragments(
            [
                re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
                for elem in group
            ]
        )
        rich = _join_pdf_line_fragments(
            [
                (elem.get("rich_content") or elem.get("content", "")).strip()
                for elem in group
            ]
        )
        source_lines = []
        for elem in group:
            paragraph = next(
                (
                    paragraph
                    for paragraph in (elem.get("paragraphs") or [])
                    if (paragraph.get("plain") or "").strip()
                ),
                {},
            )
            lines = [
                dict(line)
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            if lines:
                source_lines.extend(lines)
            else:
                elem_rect = _get_pdf_primary_paragraph_rect(elem)
                source_lines.append(
                    {
                        "plain": elem.get("content", ""),
                        "rich": elem.get("rich_content") or elem.get("content", ""),
                        "bbox": [float(value) for value in elem_rect],
                    }
                )
        first_x0 = float(source_lines[0]["bbox"][0])
        body_x0 = min(
            (float(line["bbox"][0]) for line in source_lines[1:]),
            default=first_x0,
        )
        paragraph = {
            "plain": plain,
            "rich": rich,
            "source_bbox": [float(value) for value in rect],
            "source_line_bboxes": [
                [float(value) for value in line["bbox"]] for line in source_lines
            ],
            "source_lines": source_lines,
            "margin_left": min(max(body_x0 - rect.x0, 0.0), rect.width * 0.35),
            "text_indent": max(-rect.width * 0.18, first_x0 - body_x0),
            "gap_before": 0.0,
            "text_align": "left",
            "nowrap": None,
            "first_line_indent": False,
            "toc_leader": _parse_pdf_toc_leader(plain),
        }
        merged.update(
            {
                "y": float(rect.y0),
                "x": float(rect.x0),
                "rect": fitz.Rect(rect),
                "bbox": [
                    float(rect.x0),
                    float(rect.y0),
                    float(rect.x1),
                    float(rect.y1),
                ],
                "render_bbox": _get_pdf_render_bbox(
                    rect,
                    page_rect,
                    [paragraph],
                    float(merged.get("fontsize", 11.0)),
                ),
                "content": plain,
                "rich_content": rich
                if any(elem.get("rich_content") for elem in group)
                else None,
                "paragraphs": [paragraph],
                "superscript_runs": [
                    dict(run)
                    for elem in group
                    for run in (elem.get("superscript_runs") or [])
                ],
                "inline_math_fragments": [
                    dict(fragment)
                    for elem in group
                    for fragment in (elem.get("inline_math_fragments") or [])
                ],
                "merged_visual_line_count": len(source_lines),
                "source_line_bboxes": [
                    [float(value) for value in line["bbox"]] for line in source_lines
                ],
                "bold": all(bool(elem.get("bold")) for elem in group),
            }
        )
        merged_elements.append(merged)
        index = cursor
    return merged_elements


def _pdf_wrapped_line_group_can_start(elem: dict, page_rect: fitz.Rect) -> bool:
    if not _is_pdf_single_visual_line_element(elem) or _is_heading_like_elem(elem):
        return False
    plain = re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
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

    current_plain = re.sub(r"\s+", " ", _plain_text(current.get("content", ""))).strip()
    if (
        not current_plain
        or _looks_like_pdf_paragraph_lead(current_plain)
        or _looks_like_pdf_list_marker(current_plain)
        or _pdf_element_footnote_definition_marker(current)
        or _is_heading_like_elem(current)
    ):
        return False

    first_plain = re.sub(r"\s+", " ", _plain_text(group[0].get("content", ""))).strip()
    if not _looks_like_pdf_paragraph_lead(first_plain) and len(first_plain) < 45:
        return False

    previous_rect = _get_pdf_elem_rect(previous)
    current_rect = _get_pdf_elem_rect(current)
    previous_fontsize = max(float(previous.get("fontsize", 11.0)), 1.0)
    current_fontsize = max(float(current.get("fontsize", 11.0)), 1.0)
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
    index = 0
    while index < len(elements):
        first = elements[index]
        if not _pdf_wrapped_line_group_can_start(first, page_rect):
            merged_elements.append(first)
            index += 1
            continue

        group = [first]
        cursor = index + 1
        while cursor < len(elements) and _pdf_wrapped_line_elements_can_merge(
            group,
            elements[cursor],
            page_rect,
        ):
            group.append(elements[cursor])
            cursor += 1

        if len(group) < 2:
            merged_elements.append(first)
            index += 1
            continue

        line_infos = [_pdf_visual_line_info_from_element(elem) for elem in group]
        merged = _make_pdf_text_element_from_lines(line_infos, page_rect)
        if merged is None:
            merged_elements.extend(group)
            index = cursor
            continue

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
            text_indent = max(-merged_rect.width * 0.18, first_x0 - body_x0)
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
            "margin_left": min(margin_left, merged_rect.width * 0.35),
            "text_indent": min(text_indent, merged_rect.width * 0.25),
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
            float(merged.get("fontsize", 11.0)),
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
        merged_elements.append(merged)
        index = cursor

    return merged_elements


def _pdf_strong_continuation_element_is_eligible(elem: dict) -> bool:
    if (
        elem.get("type") != "text"
        or elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
        or elem.get("watermark")
        or elem.get("layout_class") in {"heading", "footer", "table"}
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

    first_plain = re.sub(r"\s+", " ", _plain_text(first.get("content", ""))).strip()
    previous_plain = re.sub(
        r"\s+", " ", _plain_text(previous.get("content", ""))
    ).strip()
    current_plain = re.sub(r"\s+", " ", _plain_text(current.get("content", ""))).strip()
    if (
        len(first_plain) < 35
        or len(previous_plain) < 8
        or len(current_plain) < 12
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
    previous_fontsize = max(float(previous.get("fontsize", 11.0)), 1.0)
    current_fontsize = max(float(current.get("fontsize", 11.0)), 1.0)
    if abs(previous_fontsize - current_fontsize) > max(
        previous_fontsize * 0.15,
        1.0,
    ):
        return False

    vertical_gap = current_rect.y0 - previous_rect.y1
    if vertical_gap < -min(previous_rect.height, current_rect.height) * 0.40:
        return False
    if vertical_gap > max(previous_fontsize * 0.55, 4.5):
        return False

    x_shift = current_rect.x0 - previous_rect.x0
    if x_shift < -max(previous_fontsize * 0.35, 3.0):
        # Hanging references return left for the next entry.  That is a new
        # semantic paragraph, not a continuation of the prior entry.
        return False
    if x_shift > max(previous_fontsize * 4.0, page_rect.width * 0.08, 32.0):
        return False

    overlap = max(
        0.0,
        min(previous_rect.x1, current_rect.x1) - max(previous_rect.x0, current_rect.x0),
    )
    if overlap / max(min(previous_rect.width, current_rect.width), 1.0) < 0.35:
        return False

    # A same-column capitalized line is commonly a new participant, label or
    # reference entry.  Capitalized continuations remain safe when the source
    # uses an explicit hanging indent.
    lexical_probe = re.sub(r"^[\s\"'“”‘’(\[]+", "", current_plain)
    if (
        lexical_probe
        and (lexical_probe[0].isupper() or lexical_probe[0].isdigit())
        and x_shift < max(previous_fontsize * 0.80, 6.0)
    ):
        return False
    return True


def _merge_pdf_strong_continuation_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> tuple[list[dict], int]:
    """Merge post-classification strong semantic continuations."""
    merged_elements: list[dict] = []
    merge_count = 0
    index = 0
    while index < len(elements):
        first = elements[index]
        if not _pdf_strong_continuation_element_is_eligible(first):
            merged_elements.append(first)
            index += 1
            continue

        group = [first]
        cursor = index + 1
        while cursor < len(elements) and _pdf_strong_continuation_elements_can_merge(
            group,
            elements[cursor],
            page_rect,
        ):
            group.append(elements[cursor])
            cursor += 1

        if len(group) < 2:
            merged_elements.append(first)
            index += 1
            continue

        rect = _get_pdf_primary_paragraph_rect(group[0])
        for elem in group[1:]:
            rect |= _get_pdf_primary_paragraph_rect(elem)

        plain = _join_pdf_line_fragments(
            [
                re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
                for elem in group
            ]
        )
        rich = _join_pdf_line_fragments(
            [
                (elem.get("rich_content") or elem.get("content", "")).strip()
                for elem in group
            ]
        )
        source_lines: list[dict] = []
        for elem in group:
            paragraph = next(
                (
                    paragraph
                    for paragraph in (elem.get("paragraphs") or [])
                    if (paragraph.get("plain") or "").strip()
                ),
                {},
            )
            lines = [
                dict(line)
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            if lines:
                source_lines.extend(lines)
            else:
                elem_rect = _get_pdf_primary_paragraph_rect(elem)
                source_lines.append(
                    {
                        "plain": elem.get("content", ""),
                        "rich": elem.get("rich_content") or elem.get("content", ""),
                        "bbox": [float(value) for value in elem_rect],
                    }
                )

        first_x0 = float(source_lines[0]["bbox"][0])
        body_x0 = min(
            (float(line["bbox"][0]) for line in source_lines[1:]),
            default=first_x0,
        )
        fontsize = float(
            _pick_pdf_dominant_value(
                [
                    (
                        float(elem.get("fontsize", 11.0)),
                        max(len(_plain_text(elem.get("content", "")).strip()), 1),
                    )
                    for elem in group
                ],
                11.0,
            )
        )
        hanging_indent = body_x0 - first_x0 >= max(fontsize * 0.50, 3.0)
        first_line_indent = first_x0 - body_x0 >= max(fontsize * 0.50, 3.0)
        if hanging_indent:
            margin_left = max(0.0, body_x0 - rect.x0)
            text_indent = first_x0 - body_x0
        else:
            margin_left = max(0.0, min(first_x0, body_x0) - rect.x0)
            text_indent = max(0.0, first_x0 - body_x0) if first_line_indent else 0.0

        paragraph = {
            "plain": plain,
            "rich": rich,
            "source_bbox": [float(value) for value in rect],
            "source_line_bboxes": [
                [float(value) for value in line["bbox"]] for line in source_lines
            ],
            "source_lines": source_lines,
            "margin_left": min(margin_left, rect.width * 0.35),
            "text_indent": max(
                -rect.width * 0.18,
                min(text_indent, rect.width * 0.25),
            ),
            "gap_before": 0.0,
            "text_align": "left",
            "nowrap": None,
            "first_line_indent": first_line_indent,
            "toc_leader": _parse_pdf_toc_leader(plain),
        }
        merged = dict(group[0])
        merged.update(
            {
                "y": float(rect.y0),
                "x": float(rect.x0),
                "rect": fitz.Rect(rect),
                "bbox": [
                    float(rect.x0),
                    float(rect.y0),
                    float(rect.x1),
                    float(rect.y1),
                ],
                "render_bbox": _get_pdf_render_bbox(
                    rect,
                    page_rect,
                    [paragraph],
                    fontsize,
                ),
                "content": plain,
                "rich_content": rich
                if any(elem.get("rich_content") for elem in group)
                else None,
                "paragraphs": [paragraph],
                "fontsize": fontsize,
                "line_height": max(
                    float(elem.get("line_height", fontsize * 1.1)) for elem in group
                ),
                "color": int(
                    _pick_pdf_dominant_value(
                        [
                            (
                                int(elem.get("color", 0)),
                                max(
                                    len(_plain_text(elem.get("content", "")).strip()), 1
                                ),
                            )
                            for elem in group
                        ],
                        0,
                    )
                ),
                "bold": all(bool(elem.get("bold")) for elem in group),
                "superscript_runs": [
                    dict(run)
                    for elem in group
                    for run in (elem.get("superscript_runs") or [])
                ],
                "inline_math_fragments": [
                    dict(fragment)
                    for elem in group
                    for fragment in (elem.get("inline_math_fragments") or [])
                ],
                "merged_visual_line_count": len(source_lines),
                "source_line_bboxes": [
                    [float(value) for value in line["bbox"]] for line in source_lines
                ],
                "strong_semantic_continuation_merged": len(group),
                "table_hint": False,
                "preserve_source_style": False,
            }
        )
        merged_elements.append(merged)
        merge_count += len(group) - 1
        index = cursor

    return merged_elements, merge_count


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
        if elem.get("type") != "text":
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
                    plain = re.sub(
                        r"\s+", " ", _plain_text(line.get("plain", ""))
                    ).strip()
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
                plain = re.sub(r"\s+", " ", _plain_text(line.get("plain", ""))).strip()
                if _looks_like_pdf_paragraph_lead(plain):
                    residuals.append((element_index, line_index, plain[:80]))
    return residuals
