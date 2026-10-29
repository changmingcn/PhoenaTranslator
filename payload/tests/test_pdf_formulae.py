"""Formula-adjacent prose: splitting paragraphs at formula contact, promoting
text that overlaps formula geometry, and rejoining wrapped inline-math lines.

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import copy

import fitz
import pytest

from phoena_translator.pdf.model import ElementType, SkipReason
from phoena_translator.pdf.semantics.formulae import (
    _merge_pdf_wrapped_inline_math_continuation_fragments,
    _promote_pdf_formula_overlaps,
    _split_pdf_formula_adjacent_paragraph_elements,
)

PAGE = fitz.Rect(0, 0, 612, 792)
PROSE = (
    "The dealers absorb the order flow of other traders while the market "
    "adjusts its quotes and the spread widens over the following sessions."
)


def _formula(x0, y0, x1, y1, *reasons):
    elem = {"type": ElementType.FORMULA_IMAGE, "bbox": [x0, y0, x1, y1]}
    if reasons:
        elem["math_reasons"] = list(reasons)
    return elem


def _line(plain, x0, y0, x1, y1, rich=None):
    line = {"plain": plain, "bbox": [x0, y0, x1, y1]}
    if rich is not None:
        line["rich"] = rich
    return line


def _union(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _paragraph(*lines, rich=None, margin_left=0.0):
    plain = " ".join(line["plain"] for line in lines)
    return {
        "plain": plain,
        "rich": rich or plain,
        "source_lines": list(lines),
        "source_bbox": _union([line["bbox"] for line in lines]),
        "margin_left": margin_left,
    }


def _text(*paragraphs, fontsize=10.0, bbox=None, content=None, **extra):
    bbox = bbox or _union([paragraph["source_bbox"] for paragraph in paragraphs])
    elem = {
        "type": ElementType.TEXT,
        "bbox": list(bbox),
        "x": bbox[0],
        "y": bbox[1],
        "content": content or " ".join(p["plain"] for p in paragraphs),
        "fontsize": fontsize,
        "paragraphs": list(paragraphs),
    }
    elem.update(extra)
    return elem


# ---------------------------------------------------------------------------
# _split_pdf_formula_adjacent_paragraph_elements
# ---------------------------------------------------------------------------


def _three_paragraph_block():
    first = _paragraph(
        _line("Order flow from other traders arrives first.", 72, 100, 540, 112),
        rich="Order flow from other traders arrives first.<sup>1</sup>",
    )
    second = _paragraph(
        _line("Dealers then absorb the net flow qE and σ risk.", 90, 130, 540, 142),
        margin_left=18.0,
    )
    third = _paragraph(
        _line("Bold closing sentence here", 72, 160, 300, 172),
        rich="<b>Bold closing sentence here</b><sup>2</sup>",
    )
    return _text(
        first,
        second,
        third,
        superscript_runs=[
            {"text": "1", "scale": 0.5},
            {"text": "2", "scale": 0.7},
        ],
        inline_math_fragments=[{"text": "σ"}, {"text": "qE"}],
    )


def test_split_without_formulas_is_a_no_op():
    elements = [_three_paragraph_block()]
    before = copy.deepcopy(elements)
    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 0
    assert elements == before


def test_split_isolates_each_paragraph_of_a_block_touching_a_formula():
    block = _three_paragraph_block()
    formula = _formula(300, 108, 400, 118, "unicode_math")
    elements = [formula, block]

    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 2

    # The page is re-sorted by (y, x); the formula sits below the first line.
    assert [elem["type"] for elem in elements] == [
        ElementType.TEXT,
        ElementType.FORMULA_IMAGE,
        ElementType.TEXT,
        ElementType.TEXT,
    ]
    assert elements[1] is formula
    texts = [elem for elem in elements if elem["type"] == ElementType.TEXT]
    assert [elem["content"] for elem in texts] == [
        paragraph["plain"] for paragraph in block["paragraphs"]
    ]
    assert [elem["bbox"] for elem in texts] == [
        paragraph["source_bbox"] for paragraph in block["paragraphs"]
    ]
    assert [elem["y"] for elem in texts] == [100.0, 130.0, 160.0]
    for fragment in texts:
        (paragraph,) = fragment["paragraphs"]
        assert paragraph["gap_before"] == 0.0
        assert fragment["top_padding"] == 0.0
    # Margins are re-expressed relative to each fragment's own left edge.
    assert texts[1]["paragraphs"][0]["margin_left"] == 0.0
    # Superscripts follow their markers; inline math follows its text order.
    assert texts[0]["superscript_runs"] == [{"text": "1", "scale": 0.5}]
    assert texts[0]["superscript_scale"] == 0.5
    assert texts[1]["superscript_runs"] == [] and texts[1]["superscript_scale"] is None
    assert texts[2]["superscript_runs"] == [{"text": "2", "scale": 0.7}]
    assert texts[1]["inline_math_fragments"] == [{"text": "qE"}, {"text": "σ"}]
    assert texts[0]["inline_math_fragments"] == []
    assert [elem["bold"] for elem in texts] == [False, False, True]
    assert texts[2]["rich_content"] == "<b>Bold closing sentence here</b><sup>2</sup>"
    assert texts[1]["rich_content"] is None


def test_split_breaks_one_paragraph_into_line_runs_around_a_formula():
    lines = [
        _line("x + y equals the first", 72, 100, 540, 112),
        _line("z term over the second", 72, 112, 540, 124),
        _line("w = v", 72, 124, 540, 136),
    ]
    elem = _text(_paragraph(*lines), content="x + y equals the first z term over the second w = v")
    elements = [_formula(100, 126, 500, 150), elem]

    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 1

    head, tail = [e for e in elements if e["type"] == ElementType.TEXT]
    assert head["content"] == "x + y equals the first z term over the second"
    assert head["paragraphs"][0]["source_line_bboxes"] == [
        [72.0, 100.0, 540.0, 112.0],
        [72.0, 112.0, 540.0, 124.0],
    ]
    assert head["bbox"] == [72.0, 100.0, 540.0, 124.0]
    assert tail["content"] == "w = v"
    assert tail["bbox"] == [72.0, 124.0, 540.0, 136.0]


def test_split_keeps_long_prose_whole_when_only_a_line_edge_grazes():
    lines = [
        _line(PROSE[:70], 72, 100, 540, 112),
        _line(PROSE[70:], 72, 112, 540, 124),
        _line(PROSE, 72, 124, 540, 136),
    ]
    elem = _text(_paragraph(*lines), content=PROSE + " " + PROSE)
    elements = [_formula(100, 134, 400, 170), elem]
    before = copy.deepcopy(elements)

    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 0
    assert elements == before


@pytest.mark.parametrize(
    "bad_source_bbox",
    [[72, 800, 540, 820], [None, 130, 540, 142]],
    ids=["outside-page", "non-numeric"],
)
def test_split_keeps_the_block_when_a_paragraph_box_is_unusable(bad_source_bbox):
    block = _three_paragraph_block()
    block["paragraphs"][1]["source_bbox"] = bad_source_bbox
    elements = [_formula(300, 108, 400, 118), block]
    before = copy.deepcopy(elements)

    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 0
    assert elements == before


def test_split_leaves_a_paragraph_whose_every_line_touches_the_formula():
    lines = [
        _line("a = b", 72, 100, 300, 112),
        _line("c = d", 72, 112, 300, 124),
    ]
    elem = _text(_paragraph(*lines))
    elements = [_formula(60, 95, 320, 130), elem]
    before = copy.deepcopy(elements)

    assert _split_pdf_formula_adjacent_paragraph_elements(elements, PAGE) == 0
    assert elements == before


# ---------------------------------------------------------------------------
# _promote_pdf_formula_overlaps
# ---------------------------------------------------------------------------


def _plain_text_elem(content, bbox, **extra):
    return {
        "type": ElementType.TEXT,
        "bbox": list(bbox),
        "content": content,
        "fontsize": 10.0,
        "paragraphs": [],
        **extra,
    }


def test_promote_ignores_text_that_does_not_touch_a_formula():
    elements = [_formula(72, 300, 300, 320), _plain_text_elem(PROSE, [72, 100, 540, 140])]
    before = copy.deepcopy(elements)
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elements == before


def test_promote_trusts_recorded_source_lines_over_a_diagonal_outer_box():
    elem = _text(
        _paragraph(
            _line("tail of a sentence on the right", 300, 100, 540, 112),
            _line("and its continuation", 72, 128, 200, 140),
        )
    )
    elements = [_formula(72, 100, 250, 125), elem]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elem["type"] == ElementType.TEXT
    assert elem["formula_adjacent_text"] is True
    assert elem["formula_overlap_ratio"] == 0.0


def test_promote_turns_deeply_overlapping_text_into_a_formula():
    elem = _plain_text_elem("x + y = z", [100, 100, 200, 120])
    elem["paragraphs"] = [{"plain": "x + y"}, {"plain": "= z"}]
    elements = [_formula(90, 95, 210, 125, "unicode_math", "font"), elem]

    assert _promote_pdf_formula_overlaps(elements) == 1
    assert elem["type"] == ElementType.FORMULA_IMAGE
    assert elem["math_reasons"] == ["font", "formula_geometry_overlap", "unicode_math"]
    assert elem["math_mixed"] is True
    assert elem["math_line_count"] == 2
    assert elem["math_candidate_symbol_count"] == 2
    assert elem["skip_translate_reason"] == SkipReason.FORMULA_GEOMETRY_OVERLAP


def test_promote_keeps_long_prose_with_a_shallow_edge_sliver():
    elem = _plain_text_elem(PROSE + " " + PROSE, [72, 100, 540, 140])
    elements = [_formula(100, 138, 400, 170), elem]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elem["formula_adjacent_text"] is True
    assert elem["formula_overlap_ratio"] == round((300 * 2) / (468 * 40), 6)


@pytest.mark.parametrize(
    "extra",
    [{"table_hint": True}, {"inline_math_fragments": [{"text": "x"}]}],
    ids=["table-hint", "inline-math-with-real-overlap"],
)
def test_promote_fails_closed_for_risky_prose(extra):
    elem = _plain_text_elem(PROSE + " " + PROSE, [72, 100, 540, 140], **extra)
    elements = [_formula(100, 138, 400, 170), elem]
    assert _promote_pdf_formula_overlaps(elements) == 1
    assert elem["type"] == ElementType.FORMULA_IMAGE


def test_promote_keeps_a_bold_heading_with_a_small_overlap():
    elem = _plain_text_elem("Market Structure Overview", [72, 100, 300, 112], bold=True)
    elements = [_formula(72, 110, 110, 130), elem]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elem["formula_overlap_ratio"] == round(76 / 2736, 6)


def test_promote_allows_a_wider_overlap_for_a_short_terminated_sentence():
    elem = _plain_text_elem("The dealers absorb the flow.", [72, 100, 300, 110])
    elements = [_formula(72, 108, 300, 120), elem]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elem["formula_overlap_ratio"] == 0.2
    # The same overlap on a longer prose line exceeds the ordinary allowance.
    elem = _plain_text_elem("The dealers absorb the flow", [72, 100, 300, 110])
    elements = [_formula(72, 108, 300, 120), elem]
    assert _promote_pdf_formula_overlaps(elements) == 1


def test_promote_keeps_a_citation_label_that_only_touches_the_guard():
    elem = _plain_text_elem("Available at: https://example.org/report", [72, 100, 300, 110])
    elements = [_formula(72, 110.3, 300, 130), elem]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert elem["formula_overlap_ratio"] == 0.0
    # Without padding the guard does not reach the line at all.
    del elem["formula_adjacent_text"], elem["formula_overlap_ratio"]
    assert _promote_pdf_formula_overlaps(elements, padding=0.0) == 0
    assert "formula_adjacent_text" not in elem


def test_promote_measures_a_table_cell_by_its_source_ink():
    cell = _text(
        _paragraph(_line("Units", 72, 100, 150, 110)),
        bbox=[72, 100, 300, 160],
        semantic_table_cell=True,
    )
    elements = [_formula(72, 110.3, 300, 130), cell]
    assert _promote_pdf_formula_overlaps(elements) == 0
    assert cell["formula_adjacent_text"] is True
    assert cell["formula_overlap_ratio"] == 0.0


# ---------------------------------------------------------------------------
# _merge_pdf_wrapped_inline_math_continuation_fragments
# ---------------------------------------------------------------------------

DANGLING = "In equilibrium the dealers absorb the net flow (qE"


def _row_elem(content, bbox, *, lines=True, fontsize=10.0, **extra):
    paragraphs = []
    if lines:
        paragraphs = [_paragraph(_line(content, *bbox))]
    elem = _text(*paragraphs, bbox=bbox, content=content, fontsize=fontsize, **extra)
    if not lines:
        elem["paragraphs"] = []
    return elem


def test_wrapped_merge_accepts_an_empty_page():
    assert _merge_pdf_wrapped_inline_math_continuation_fragments([], PAGE) == 0


def test_wrapped_merge_rejoins_a_same_row_script_continuation():
    previous = _row_elem(
        DANGLING,
        [72, 100, 320, 112],
        superscript_runs=[{"text": "1"}],
        layout_class="body",
    )
    inline = _formula(318, 98, 330, 110)
    current = _row_elem(
        "ls), other traders' positions",
        [322, 99, 450, 112],
        inline_math_fragments=[{"text": "qE"}],
    )
    elements = [previous, inline, current]

    assert _merge_pdf_wrapped_inline_math_continuation_fragments(elements, PAGE) == 1

    merged = elements[0]
    assert elements[1] is inline and len(elements) == 2
    assert merged["content"] == DANGLING + " ls), other traders' positions"
    assert merged["bbox"] == [72.0, 99.0, 450.0, 112.0]
    assert merged["wrapped_math_continuation_merged"] == 2
    assert "layout_class" not in merged
    (paragraph,) = merged["paragraphs"]
    assert len(paragraph["source_lines"]) == 2
    assert paragraph["source_bbox"] == [72.0, 99.0, 450.0, 112.0]
    assert paragraph["nowrap"] is None
    assert merged["superscript_runs"] == [{"text": "1"}]
    assert merged["inline_math_fragments"] == [{"text": "qE"}]


def test_wrapped_merge_rejoins_a_next_row_hyphen_wrap_without_recorded_lines():
    previous = _row_elem(
        "The dealers rebalance their positions across the inter-",
        [72, 100, 540, 112],
        lines=False,
    )
    current = _row_elem("dealer market and other venues", [72, 114, 300, 126], lines=False)
    elements = [previous, current]

    assert _merge_pdf_wrapped_inline_math_continuation_fragments(elements, PAGE) == 1

    (merged,) = elements
    (paragraph,) = merged["paragraphs"]
    assert paragraph["plain"] == merged["content"]
    assert paragraph["source_bbox"] == [72.0, 100.0, 540.0, 126.0]
    assert paragraph["source_lines"] == [
        {
            "plain": "dealer market and other venues",
            "rich": "dealer market and other venues",
            "bbox": [72.0, 114.0, 300.0, 126.0],
        }
    ]
    assert merged["rich_content"] is None


def test_wrapped_merge_folds_repeated_tails_into_one_sentence():
    elements = [
        _row_elem(DANGLING, [72, 100, 320, 112]),
        _row_elem("ls), and the flow of the dealers (qS", [322, 100, 470, 112]),
        _row_elem("ot), other traders' positions", [472, 100, 560, 112]),
    ]
    assert _merge_pdf_wrapped_inline_math_continuation_fragments(elements, PAGE) == 2
    (merged,) = elements
    assert merged["wrapped_math_continuation_merged"] == 3
    assert merged["content"].endswith("(qS ot), other traders' positions")


def _rejection_cases():
    def same_row(**changes):
        previous = _row_elem(DANGLING, [72, 100, 320, 112])
        current = _row_elem("ls), other traders' positions", [322, 99, 450, 112])
        middle = changes.pop("middle", None)
        previous.update(changes.pop("previous", {}))
        current.update(changes.pop("current", {}))
        return [previous, *([middle] if middle else []), current]

    return {
        "bold-mismatch": same_row(current={"bold": True}),
        "previous-table-hint": same_row(previous={"table_hint": True}),
        "no-dangling-tail": same_row(previous={"content": "In equilibrium the dealers absorb the flow"}),
        "multi-paragraph-fragment": same_row(
            current={"paragraphs": [{"plain": "ls), other"}, {"plain": "traders' positions"}]}
        ),
        "short-previous": same_row(previous={"content": "the net flow (qE"}),
        "paragraph-lead": same_row(current={"content": "2. other traders' positions"}),
        "no-prose-words": same_row(current={"content": "ls), x, y"}),
        "font-size-jump": same_row(current={"fontsize": 18.0}),
        "watermark": same_row(current={"watermark": True}),
        "table-between": same_row(middle={"type": ElementType.TABLE_IMAGE, "bbox": [0, 400, 10, 410]}),
        "display-formula": same_row(middle=_formula(100, 95, 440, 140)),
        "image-blocker": same_row() + [{"type": ElementType.IMAGE, "bbox": [300, 100, 340, 110]}],
        "hyphen-on-same-row": [
            _row_elem("The dealers rebalance their positions across the inter-", [72, 100, 320, 112]),
            _row_elem("dealer market and other venues", [322, 100, 450, 112]),
        ],
        "fragment-above": [
            _row_elem(DANGLING, [72, 100, 320, 112]),
            _row_elem("ls), other traders' positions", [72, 60, 300, 72]),
        ],
        "row-too-far": [
            _row_elem(DANGLING, [72, 100, 320, 112]),
            _row_elem("ls), other traders' positions", [72, 140, 300, 152]),
        ],
        "no-horizontal-overlap": [
            _row_elem("The dealers rebalance their positions across the inter-", [72, 100, 300, 112]),
            _row_elem("dealer market and other venues", [400, 114, 540, 126]),
        ],
    }


@pytest.mark.parametrize("name", sorted(_rejection_cases()))
def test_wrapped_merge_rejects_unsafe_pairs(name):
    elements = _rejection_cases()[name]
    before = copy.deepcopy(elements)
    assert _merge_pdf_wrapped_inline_math_continuation_fragments(elements, PAGE) == 0
    assert elements == before
