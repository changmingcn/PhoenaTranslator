"""Shared skeleton of the greedy semantic merge passes.

Several passes coalesce runs of adjacent text elements into one paragraph:
a run starts at an element a pass accepts as a lead and absorbs following
elements while the pass's pairwise rule holds.  ``group_adjacent`` owns
that loop; ``build_merged_text_element`` owns the shape of the resulting
element, so passes only state their own lead/join rules and indent policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import fitz

from phoena_translator.pdf.geometry import (
    _get_pdf_primary_paragraph_rect,
    _get_pdf_render_bbox,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.text_shape import (
    _join_pdf_line_fragments,
    _pick_pdf_dominant_value,
)


def group_adjacent(
    elements: list[dict],
    can_start: Callable[[dict], bool],
    can_join: Callable[[list[dict], dict], bool],
) -> Iterator[list[dict]]:
    """Yield ``elements`` in order as singletons or maximal mergeable runs."""
    index = 0
    while index < len(elements):
        first = elements[index]
        group = [first]
        if can_start(first):
            cursor = index + 1
            while cursor < len(elements) and can_join(group, elements[cursor]):
                group.append(elements[cursor])
                cursor += 1
        yield group
        index += len(group)


def merge_groups(
    groups: Iterable[list[dict]],
    merge: Callable[[list[dict]], dict],
) -> tuple[list[dict], int]:
    """Replace every multi-element run by ``merge(run)``; count absorbed elements."""
    merged_elements: list[dict] = []
    absorbed = 0
    for group in groups:
        if len(group) < 2:
            merged_elements.extend(group)
            continue
        merged_elements.append(merge(group))
        absorbed += len(group) - 1
    return merged_elements, absorbed


def group_rect(group: list[dict]) -> fitz.Rect:
    rect = _get_pdf_primary_paragraph_rect(group[0])
    for elem in group[1:]:
        rect |= _get_pdf_primary_paragraph_rect(elem)
    return rect


def group_plain_and_rich(group: list[dict]) -> tuple[str, str]:
    plain = _join_pdf_line_fragments(
        [_collapsed_plain_text(elem.get("content", "")) for elem in group]
    )
    rich = _join_pdf_line_fragments(
        [(elem.get("rich_content") or elem.get("content", "")).strip() for elem in group]
    )
    return plain, rich


def group_source_lines(group: list[dict]) -> list[dict]:
    """Concatenate each element's non-empty source lines (or its own box)."""
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
    return source_lines


def first_and_body_x0(source_lines: list[dict]) -> tuple[float, float]:
    """Left edge of the first line and of the leftmost continuation line."""
    first_x0 = float(source_lines[0]["bbox"][0])
    body_x0 = min(
        (float(line["bbox"][0]) for line in source_lines[1:]),
        default=first_x0,
    )
    return first_x0, body_x0


def _text_weight(elem: dict) -> int:
    return max(len(_plain_text(elem.get("content", "")).strip()), 1)


def dominant_group_style(group: list[dict]) -> dict[str, Any]:
    """Font size, line height and colour weighted by each element's text."""
    fontsize = float(
        _pick_pdf_dominant_value(
            [(float(elem.get("fontsize", 11.0)), _text_weight(elem)) for elem in group],
            11.0,
        )
    )
    return {
        "fontsize": fontsize,
        "line_height": max(
            float(elem.get("line_height", fontsize * 1.1)) for elem in group
        ),
        "color": int(
            _pick_pdf_dominant_value(
                [(int(elem.get("color", 0)), _text_weight(elem)) for elem in group],
                0,
            )
        ),
    }


@dataclass(frozen=True)
class ParagraphIndent:
    margin_left: float
    text_indent: float
    first_line_indent: bool = False


def build_merged_text_element(
    group: list[dict],
    page_rect: fitz.Rect,
    *,
    rect: fitz.Rect,
    plain: str,
    rich: str,
    source_lines: list[dict],
    indent: ParagraphIndent,
    toc_leader: dict | None,
    render_fontsize: float,
    style: dict[str, Any] | None = None,
    markers: dict[str, Any] | None = None,
) -> dict:
    """Return ``group[0]`` rewritten as one single-paragraph text element."""
    source_line_bboxes = [
        [float(value) for value in line["bbox"]] for line in source_lines
    ]
    paragraph = {
        "plain": plain,
        "rich": rich,
        "source_bbox": [float(value) for value in rect],
        "source_line_bboxes": source_line_bboxes,
        "source_lines": source_lines,
        "margin_left": indent.margin_left,
        "text_indent": indent.text_indent,
        "gap_before": 0.0,
        "text_align": "left",
        "nowrap": None,
        "first_line_indent": indent.first_line_indent,
        "toc_leader": toc_leader,
    }
    merged = dict(group[0])
    merged.update(
        {
            "y": float(rect.y0),
            "x": float(rect.x0),
            "rect": fitz.Rect(rect),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "render_bbox": _get_pdf_render_bbox(rect, page_rect, [paragraph], render_fontsize),
            "content": plain,
            "rich_content": rich if any(elem.get("rich_content") for elem in group) else None,
            "paragraphs": [paragraph],
            **(style or {}),
            "bold": all(bool(elem.get("bold")) for elem in group),
            "superscript_runs": [
                dict(run) for elem in group for run in (elem.get("superscript_runs") or [])
            ],
            "inline_math_fragments": [
                dict(fragment)
                for elem in group
                for fragment in (elem.get("inline_math_fragments") or [])
            ],
            "merged_visual_line_count": len(source_lines),
            "source_line_bboxes": [list(bbox) for bbox in source_line_bboxes],
            **(markers or {}),
        }
    )
    return merged


__all__ = [
    "ParagraphIndent",
    "build_merged_text_element",
    "dominant_group_style",
    "first_and_body_x0",
    "group_adjacent",
    "group_plain_and_rich",
    "group_rect",
    "group_source_lines",
    "merge_groups",
]
