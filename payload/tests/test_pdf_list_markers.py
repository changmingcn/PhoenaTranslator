"""Detached list markers: ``(b)``/bullet blocks rejoined with their paragraph.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import copy

import fitz
import pytest

from phoena_translator.pdf.model import (
    PARAGRAPH_MAX_HANGING_SHARE,
    PARAGRAPH_MAX_MARGIN_SHARE,
    ElementType,
)
from phoena_translator.pdf.semantics.continuations import (
    _merge_pdf_detached_list_marker_elements,
)

PAGE = fitz.Rect(0, 0, 612, 792)
BODY = "The dealers report their positions every quarter to the supervisor."


def _marker(text, bbox, **extra):
    return {
        "type": ElementType.TEXT,
        "content": text,
        "bbox": list(bbox),
        "x": bbox[0],
        "y": bbox[1],
        "fontsize": 10.0,
        "paragraphs": [{"plain": text, "source_bbox": list(bbox)}],
        **extra,
    }


def _body(text, bbox, *, lines=True, **extra):
    paragraph = {"plain": text, "rich": text, "source_bbox": list(bbox), "text_align": "justify"}
    if lines:
        x0, y0, x1, y1 = bbox
        paragraph["source_lines"] = [
            {"plain": text[:40], "rich": text[:40], "bbox": [x0, y0, x1, y0 + 12]},
            {"plain": text[40:], "bbox": [x0, y0 + 12, x1, y1]},
            {"plain": " ", "bbox": [x0, y1, x1, y1 + 1]},
        ]
    return {
        "type": ElementType.TEXT,
        "content": text,
        "bbox": list(bbox),
        "x": bbox[0],
        "y": bbox[1],
        "fontsize": 10.0,
        "paragraphs": [paragraph],
        **extra,
    }


def test_marker_joins_the_paragraph_whose_first_line_it_starts():
    marker = _marker("(b)", [72, 100, 84, 112], superscript_runs=[{"text": "1"}])
    body = _body(
        BODY,
        [90, 100, 540, 124],
        rich_content="<i>The dealers</i> report their positions every quarter to the supervisor.",
        inline_math_fragments=[{"text": "q"}],
        merged_visual_line_count=1,
    )
    heading = _body("Section heading text", [72, 60, 300, 72], lines=False)

    result = _merge_pdf_detached_list_marker_elements([body, marker, heading], PAGE)

    assert [elem["content"][:7] for elem in result] == ["Section", "(b) The"]
    merged = result[1]
    assert merged["content"] == f"(b) {BODY}"
    assert merged["rich_content"] == (
        "(b) <i>The dealers</i> report their positions every quarter to the supervisor."
    )
    assert merged["bbox"] == [72.0, 100.0, 540.0, 124.0]
    assert merged["detached_list_marker_merged"] is True
    assert merged["superscript_runs"] == [{"text": "1"}]
    assert merged["inline_math_fragments"] == [{"text": "q"}]
    (paragraph,) = merged["paragraphs"]
    first, second = paragraph["source_lines"]
    assert first["plain"] == f"(b) {BODY[:40]}" and first["bbox"] == [72.0, 100.0, 540.0, 112.0]
    # Later lines are kept as recorded; the blank positioning line is dropped.
    assert second == body["paragraphs"][0]["source_lines"][1]
    assert paragraph["margin_left"] == min(18.0, 468.0 * PARAGRAPH_MAX_MARGIN_SHARE)
    assert paragraph["text_indent"] == max(-468.0 * PARAGRAPH_MAX_HANGING_SHARE, -18.0)
    assert paragraph["text_align"] == "left" and paragraph["gap_before"] == 0.0
    assert merged["merged_visual_line_count"] == 2
    assert merged["source_line_bboxes"] == paragraph["source_line_bboxes"]


def test_marker_without_recorded_lines_gets_one_synthesized_line():
    marker = _marker("•", [72, 100, 78, 110])
    body = _body(BODY, [86, 99, 540, 111], lines=False)
    (merged,) = _merge_pdf_detached_list_marker_elements([marker, body], PAGE)
    (paragraph,) = merged["paragraphs"]
    assert paragraph["source_lines"] == [
        {"plain": f"• {BODY}", "rich": f"• {BODY}", "bbox": [72.0, 99.0, 540.0, 111.0]}
    ]
    assert merged["rich_content"] is None
    assert merged["merged_visual_line_count"] == 1


def test_marker_prefers_the_closest_aligned_paragraph():
    marker = _marker("(c)", [72, 200, 86, 212])
    same_row = _body(BODY, [92, 201, 540, 225])
    next_row = _body("Another paragraph that starts lower down", [92, 212, 540, 236])
    result = _merge_pdf_detached_list_marker_elements([next_row, same_row, marker], PAGE)
    assert [elem["content"].startswith("(c)") for elem in result] == [True, False]


def test_two_markers_never_merge_into_each_other():
    first = _marker("(a)", [72, 100, 86, 112])
    second = _marker("(b)", [90, 100, 104, 112])
    elements = [first, second]
    assert _merge_pdf_detached_list_marker_elements(copy.deepcopy(elements), PAGE) == elements


def _rejection_cases():
    def pair(marker_extra=None, body_extra=None, *, marker_text="(b)", body_text=BODY,
             marker_bbox=(72, 100, 84, 112), body_bbox=(90, 100, 540, 124), body_lines=True):
        return [
            _marker(marker_text, marker_bbox, **(marker_extra or {})),
            _body(body_text, list(body_bbox), lines=body_lines, **(body_extra or {})),
        ]

    two_paragraphs = pair()
    two_paragraphs[1]["paragraphs"].append({"plain": "second paragraph"})
    return {
        "not-a-marker": pair(marker_text="word"),
        "marker-table-hint": pair({"table_hint": True}),
        "marker-preserved-style": pair({"preserve_source_style": True}),
        "marker-rotated": pair({"non_horizontal": True}),
        "body-table-hint": pair(body_extra={"table_hint": True}),
        "body-is-a-marker": pair(body_text="(c)"),
        "body-is-a-numbered-lead": pair(body_text="1. The dealers report positions"),
        "body-too-short": pair(body_text="Dealers"),
        "body-two-paragraphs": two_paragraphs,
        "body-left-of-marker": pair(body_bbox=(60, 100, 540, 124)),
        "body-overlaps-marker": pair(body_bbox=(76, 100, 540, 124), marker_bbox=(72, 100, 90, 112)),
        "body-too-far-right": pair(body_bbox=(200, 100, 540, 124)),
        "body-on-another-row": pair(body_bbox=(90, 140, 540, 164)),
        "image-target": [
            _marker("(b)", [72, 100, 84, 112]),
            {"type": ElementType.IMAGE, "bbox": [90, 100, 540, 124], "y": 100.0, "content": BODY},
        ],
    }


@pytest.mark.parametrize("name", sorted(_rejection_cases()))
def test_marker_stays_alone_without_a_safe_target(name):
    elements = _rejection_cases()[name]
    before = copy.deepcopy(elements)
    result = _merge_pdf_detached_list_marker_elements(elements, PAGE)
    assert sorted(map(repr, result)) == sorted(map(repr, before))


def test_result_is_ordered_by_position_even_without_x():
    late = _body(BODY, [72, 300, 540, 324])
    early = _body(BODY, [300, 100, 540, 124])
    early.pop("x")
    other = {"type": ElementType.IMAGE, "bbox": (0, 100, 10, 110), "y": 100.0}
    result = _merge_pdf_detached_list_marker_elements([late, early, other], PAGE)
    assert result == [other, early, late]


def test_each_marker_of_a_list_joins_its_own_item():
    elements = [
        _marker("(a)", [72, 100, 86, 112]),
        _body(BODY, [92, 100, 540, 124]),
        _marker("(b)", [72, 140, 86, 152]),
        _body("Another item of the same list follows here", [92, 140, 540, 164]),
    ]
    result = _merge_pdf_detached_list_marker_elements(elements, PAGE)
    assert [elem["content"][:3] for elem in result] == ["(a)", "(b)"]
