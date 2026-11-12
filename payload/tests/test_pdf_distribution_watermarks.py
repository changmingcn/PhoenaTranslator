"""Distribution watermark codes: detected at extraction, never translated,
removed from the delivered page.

Every token and document here is synthetic.  The detector is conservative:
each documented condition is shown to be necessary by an input that differs
from a detected code in that one respect.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import fitz
import pytest

from phoena_translator.pdf.extract.distribution_codes import (
    CONDITIONS,
    delivered_removed_watermarks,
    distribution_code_conditions,
    mark_distribution_codes,
    removed_watermarks_task_fields,
)
from phoena_translator.pdf.model import SkipReason

TESTS_DIR = Path(__file__).resolve().parent
PAYLOAD_DIR = TESTS_DIR.parent

CODE = "Zq4Wm8Xr2Lc7Hn"  # synthetic: random letters and digits, mixed case
PAGE = fitz.Rect(0, 0, 595.0, 794.0)
CORNER = [-3.0, 783.0, 70.0, 795.5]  # bottom-left corner, partly off the page
FOOTER_ON_PAGE = [20.0, 770.0, 95.0, 780.0]  # inside the bottom 6% band
BODY_AREA = [72.0, 400.0, 150.0, 410.0]


def _text(bbox, content, **extra):
    elem = {
        "type": "text",
        "bbox": list(bbox),
        "rect": list(bbox),
        "content": content,
        "rich_content": content,
        "fontsize": 9.0,
        "color": 0,
        "layout_class": "scattered",
        "paragraphs": [{"plain": content, "source_lines": [{"plain": content}]}],
    }
    elem.update(extra)
    return elem


def _body(y=100.0):
    return _text([72.0, y, 520.0, y + 40.0], "Demand stayed firm across the quarter.",
                 layout_class="body")


def _conditions(elem, others=(), elsewhere=(), invisible=False):
    elements = [elem, *others]
    return distribution_code_conditions(
        elem, 0, elements, PAGE, [fitz.Rect(r) for r in elsewhere], ink_invisible=invisible
    )


def _failed(elem, **kwargs):
    return [name for name, ok in _conditions(elem, **kwargs).items() if not ok]


# --- the positive cases ----------------------------------------------------


def test_code_partly_outside_the_page_passes_every_condition():
    assert _failed(_text(CORNER, CODE)) == []


def test_on_page_margin_code_needs_the_same_token_at_the_same_place_elsewhere():
    assert _failed(_text(FOOTER_ON_PAGE, CODE), invisible=True) == ["corroborated"]
    assert _failed(_text(FOOTER_ON_PAGE, CODE), elsewhere=[FOOTER_ON_PAGE], invisible=True) == []
    far = [400.0, 770.0, 475.0, 780.0]
    assert _failed(_text(FOOTER_ON_PAGE, CODE), elsewhere=[far], invisible=True) == ["corroborated"]


def test_a_visible_token_inside_the_page_is_never_a_code():
    assert _failed(_text(FOOTER_ON_PAGE, CODE), elsewhere=[FOOTER_ON_PAGE]) == [
        "invisible_or_outside"
    ]


def test_mark_sets_watermark_and_records_page_element_bbox_text():
    pages = {
        0: {"elements": [_body(), _text(CORNER, CODE)]},
        1: {"elements": [_body(), _text(FOOTER_ON_PAGE, CODE.lower() + "x")]},
    }
    removed = mark_distribution_codes(pages, {0: PAGE, 1: PAGE})
    assert removed == [{
        "page": 1, "element": 1, "bbox": [-3.0, 783.0, 70.0, 795.5],
        "text": CODE, "kind": "distribution_code",
    }]
    assert pages[0]["elements"][1]["skip_translate_reason"] == SkipReason.WATERMARK
    assert "skip_translate_reason" not in pages[0]["elements"][0]
    assert "skip_translate_reason" not in pages[1]["elements"][1]


def _footer_pdf(path, token, pages=3, **ink):
    """``pages`` pages of prose, each with ``token`` alone in the footer band;
    ``ink`` is passed to ``insert_text`` (``fill_opacity``, ``render_mode``)."""
    doc = fitz.open()
    for number in range(pages):
        page = doc.new_page(width=PAGE.width, height=PAGE.height)
        page.insert_text((72, 120), f"Demand stayed firm across quarter {number + 1}.",
                         fontsize=10, fontname="helv")
        page.insert_text((20, 778), token, fontsize=9, fontname="helv", **ink)
    doc.save(str(path))
    elements = {}
    for number, page in enumerate(doc):
        (word,) = [w for w in page.get_text("words") if w[4] == token]
        elements[number] = {"elements": [_body(), _text(word[:4], token)]}
    doc.close()
    return elements


def test_recurring_margin_code_is_marked_on_every_page(tmp_path):
    for ink in ({"fill_opacity": 0}, {"render_mode": 3}):
        pages = _footer_pdf(tmp_path / "codes.pdf", CODE, **ink)
        removed = mark_distribution_codes(
            pages, {n: PAGE for n in range(3)}, source_path=str(tmp_path / "codes.pdf")
        )
        assert [record["page"] for record in removed] == [1, 2, 3], ink


def test_a_visible_recurring_footer_token_is_never_removed(tmp_path):
    pages = _footer_pdf(tmp_path / "visible.pdf", CODE)
    rects = {n: PAGE for n in range(3)}
    assert mark_distribution_codes(pages, rects, source_path=str(tmp_path / "visible.pdf")) == []
    assert mark_distribution_codes(pages, rects) == []  # ink unreadable: kept
    assert not any("skip_translate_reason" in e for info in pages.values() for e in info["elements"])


RUNNING_FOOTERS = ["FY2024Q3Update", "Q3Earnings2024v2", "Report2024v3Final", "Model7Bv2"]


@pytest.mark.parametrize("footer", RUNNING_FOOTERS)
def test_a_token_holding_a_word_is_never_a_code(footer):
    # Partly outside the page (so visibility allows it): the word alone keeps it.
    assert _failed(_text(CORNER, footer)) == ["no_word_run"]
    assert _failed(_text(FOOTER_ON_PAGE, footer), elsewhere=[FOOTER_ON_PAGE], invisible=True) == [
        "no_word_run"
    ]


@pytest.mark.parametrize("footer", RUNNING_FOOTERS)
@pytest.mark.parametrize("ink", [{}, {"fill_opacity": 0}], ids=["visible", "invisible"])
def test_a_running_footer_on_two_pages_is_kept_also_on_a_cache_resume(tmp_path, footer, ink):
    pages = _footer_pdf(tmp_path / "footer.pdf", footer, pages=2, **ink)
    rects = {0: PAGE, 1: PAGE}
    assert mark_distribution_codes(pages, rects, source_path=str(tmp_path / "footer.pdf")) == []
    # A resume re-runs detection on the layout read back from its JSON cache.
    resumed = json.loads(json.dumps({str(k): v for k, v in pages.items()}))
    assert mark_distribution_codes(
        resumed, {"0": PAGE, "1": PAGE}, source_path=str(tmp_path / "footer.pdf")
    ) == []
    assert not any("skip_translate_reason" in e for info in resumed.values() for e in info["elements"])


def test_layout_json_string_keys_are_supported():
    pages = {"0": {"elements": [_text(CORNER, CODE)]}, "1": {"appendix": True}}
    removed = mark_distribution_codes(pages, {"0": PAGE, "1": PAGE})
    assert [record["page"] for record in removed] == [1]


# --- one condition at a time -----------------------------------------------


# ``exact``: the input fails only that condition.  Identifier-shaped inputs
# also fail ``alternating``; they are listed for the condition that names them.
@pytest.mark.parametrize(
    ("label", "elem", "condition", "exact"),
    [
        ("formula element", dict(_text(CORNER, CODE), type="formula_image"), "text_element", True),
        ("already kept", _text(CORNER, CODE, skip_translate_reason="table_preserved"), "text_element", True),
        ("body text", _text(CORNER, CODE, layout_class="body"), "not_content_class", True),
        ("heading", _text(CORNER, CODE, layout_class="heading"), "not_content_class", True),
        ("table cell", _text(CORNER, CODE, layout_class="table"), "not_content_class", True),
        ("table hint", _text(CORNER, CODE, table_hint=True), "not_content_class", True),
        ("footnote hint", _text(CORNER, CODE, footnote_hint=True), "not_content_class", True),
        ("two lines", _text(CORNER, CODE, paragraphs=[
            {"plain": CODE, "source_lines": [{"plain": CODE[:7]}, {"plain": CODE[7:]}]}]), "single_line", True),
        ("two words", _text(CORNER, "Zq4Wm8 Xr2Lc7Hn"), "alphanumeric", True),
        ("too short", _text(CORNER, "Zq4Wm8X"), "length", True),
        ("too long", _text(CORNER, (CODE * 3)[:41]), "length", True),
        ("punctuation", _text(CORNER, "Zq4W-m8Xr2"), "alphanumeric", True),
        ("non-ascii", _text(CORNER, "Zq4Wm8Xr2L\u00e97"), "alphanumeric", True),
        ("lower-case hash", _text(CORNER, "a3f9c2e1b7d4"), "mixed_case", True),
        ("upper-case code", _text(CORNER, "AB12CD34EF56"), "mixed_case", True),
        ("upper-case ISIN", _text(CORNER, "US0378331005"), "mixed_case", False),
        ("upper-case CUSIP", _text(CORNER, "38259P508"), "mixed_case", False),
        ("camel word with version", _text(CORNER, "MacroOutlook2026"), "alternating", False),
        ("word inside a code", _text(CORNER, "Zq4Update8Xr2"), "no_word_run", True),
        ("one digit", _text(CORNER, "ZqaWmbXrcLd7"), "alternating", True),
        ("mid-page", _text(BODY_AREA, CODE), "edge_placement", False),
        ("overlaps a neighbour", _text(CORNER, CODE), "isolated", True),
    ],
)
def test_each_condition_is_necessary(label, elem, condition, exact):
    others = [_text([60.0, 785.0, 200.0, 794.0], "Page 1 of 3")] if condition == "isolated" else []
    failed = _failed(elem, others=others)
    assert condition in failed, (label, failed)
    if exact:
        assert failed == [condition], (label, failed)


@pytest.mark.parametrize(
    "content",
    ["AAPL", "US0378331005", "B0YBKJ7", "12", "2026-11-12", "12Nov2026", "Q3FY24",
     "Global Markets Research", "Page 3 of 10", "iPhone15Pro", "COVID19"],
)
def test_never_takes_tickers_identifiers_page_numbers_dates_or_word_lines(content):
    for bbox in (CORNER, FOOTER_ON_PAGE):
        assert not all(_conditions(_text(bbox, content), elsewhere=[bbox]).values()), content


def test_detection_on_a_dense_page_evaluates_geometry_only_for_token_survivors(monkeypatch):
    # R6: 4,000 table cells (40 x 100) and one code.  Isolation compares an
    # element with every other one, so evaluating it for each cell is
    # quadratic (84 s at this size on the 2-core host); after the cheap token
    # conditions only the code is left, so detection is one pass plus one
    # isolation scan (about 0.1 s there).  The 2 s bound leaves ample room
    # for a loaded host while failing the quadratic path by a wide margin.
    import time

    from phoena_translator.pdf.extract import distribution_codes

    cells = [
        _text([20 + (i % 40) * 14, 60 + (i // 40) * 7, 32 + (i % 40) * 14, 66 + (i // 40) * 7],
              f"{i * 7 % 1000}.{i % 10}")
        for i in range(4000)
    ]
    pages = {0: {"elements": [*cells, _text(CORNER, CODE)]}}
    calls = []
    real = distribution_codes._isolated

    def counted(index, rect, elements):
        calls.append(index)
        assert len(calls) <= 1, "geometry evaluated for an element the token conditions reject"
        return real(index, rect, elements)

    monkeypatch.setattr(distribution_codes, "_isolated", counted)
    start = time.perf_counter()
    removed = mark_distribution_codes(pages, {0: PAGE})
    elapsed = time.perf_counter() - start

    assert [record["element"] for record in removed] == [4000]
    assert calls == [4000]
    assert elapsed < 2.0


def _code_shaped(i):
    """A synthetic token that passes every token condition (mixed case,
    alternating letters and digits, no word run)."""
    return f"Q{i % 10}k{i // 10 % 10}M{i // 100 % 10}x{i // 1000 % 10}"


def test_a_page_of_code_shaped_tokens_has_no_code_and_no_geometry(monkeypatch):
    # T6: 4,000 elements that all pass the token conditions.  A code is stamped
    # once per page, so none of them is one, and the geometry (quadratic per
    # page: 85 s at this size) is not evaluated for that page.  Same bound as
    # the R6 test above.
    import time

    from phoena_translator.pdf.extract import distribution_codes

    cells = [
        _text([20 + (i % 40) * 14, 60 + (i // 40) * 7, 32 + (i % 40) * 14, 66 + (i // 40) * 7],
              _code_shaped(i))
        for i in range(4000)
    ]
    pages = {0: {"elements": [*cells, _text(CORNER, CODE)]}}
    calls = []
    monkeypatch.setattr(
        distribution_codes, "_isolated", lambda *args: calls.append(args) or True
    )
    start = time.perf_counter()
    removed = mark_distribution_codes(pages, {0: PAGE})
    elapsed = time.perf_counter() - start

    assert removed == []
    assert calls == []
    assert elapsed < 2.0


@pytest.mark.parametrize("extra, removed_count", [(2, 1), (3, 0)])
def test_the_code_is_removed_beside_a_few_code_shaped_tokens_only(extra, removed_count):
    # The real code (one per page) beside two code-shaped content tokens is
    # removed; with three beside it the page is a page of identifiers.
    tokens = [_text([72.0, 300.0 + 20 * i, 150.0, 310.0 + 20 * i], _code_shaped(i + 1))
              for i in range(extra)]
    pages = {0: {"elements": [*tokens, _text(CORNER, CODE)]}}

    removed = mark_distribution_codes(pages, {0: PAGE})

    assert [record["element"] for record in removed] == [extra] * removed_count
    if removed_count == 0:
        failed = [name for name, ok in _conditions(
            _text(CORNER, CODE), others=tokens
        ).items() if not ok]
        assert failed == ["few_on_page"]


def test_conditions_cover_the_documented_list():
    assert tuple(_conditions(_text(CORNER, CODE))) == CONDITIONS


def test_task_record_lists_only_codes_on_delivered_pages():
    audit = {
        "removed_watermarks": [
            {"page": 1, "element": 4, "bbox": CORNER, "text": CODE, "kind": "distribution_code"},
            {"page": 2, "element": 3, "bbox": CORNER, "text": CODE, "kind": "distribution_code"},
        ],
        "source_page_fallbacks": [{"page": 2, "reason": "x"}],
    }
    assert [r["page"] for r in delivered_removed_watermarks(audit)] == [1]
    assert removed_watermarks_task_fields({"source_page_fallbacks": []}) == {}
    assert list(removed_watermarks_task_fields(audit)) == ["removed_watermarks"]


# --- end to end on a synthetic two-page PDF --------------------------------


def build_distribution_code_pdf(path: Path) -> None:
    """Two pages of report prose; page 1 carries a per-copy code stamped in
    the bottom-left corner, partly outside the page box, and a digit token in
    the footer that must stay."""
    import golden_support

    doc = fitz.open()
    for number in (1, 2):
        page = doc.new_page(width=595, height=794)
        page.insert_text((72, 80), f"Market outlook part {number}", fontsize=16, fontname="helv")
        golden_support._paragraph(
            page, 110,
            "Demand for duration stayed firm as investors added to long positions. "
            "Dealers reported steady flows from pension funds and insurers.",
        )
        page.insert_text((450, 770), "20261112", fontsize=8, fontname="helv")
    doc[0].insert_text((-3.0, 793.0), CODE, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


_RUNNER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import golden_support, test_pdf_distribution_watermarks as t
golden_support.CORPUS["distribution_code"] = t.build_distribution_code_pdf
raw = golden_support.run_pipeline("distribution_code", Path(sys.argv[2]), Path(sys.argv[3]))
Path(sys.argv[4]).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
"""


def build_running_identifier_pdf(path: Path) -> None:
    """Two pages of prose with the same visible identifier in the footer."""
    import golden_support

    doc = fitz.open()
    for number in (1, 2):
        page = doc.new_page(width=595, height=794)
        page.insert_text((72, 80), f"Market outlook part {number}", fontsize=16, fontname="helv")
        golden_support._paragraph(
            page, 110,
            "Demand for duration stayed firm as investors added to long positions. "
            "Dealers reported steady flows from pension funds and insurers.",
        )
        page.insert_text((72, 776), "Model7Bv2", fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


def _run_corpus(tmp_path, builder):
    import golden_support

    font = golden_support.build_test_font(tmp_path)
    if font is None:
        pytest.skip("no glyf-outline CJK system font available")
    workdir = tmp_path / "work"
    workdir.mkdir()
    snapshot = tmp_path / "snapshot.json"
    runner = _RUNNER.replace("t.build_distribution_code_pdf", f"t.{builder}")
    completed = subprocess.run(
        [sys.executable, "-c", runner, str(TESTS_DIR), str(workdir), str(font[0]), str(snapshot)],
        env={**os.environ, "PYTHONPATH": str(PAYLOAD_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return workdir, json.loads(snapshot.read_text(encoding="utf-8"))


def test_pipeline_keeps_a_visible_running_identifier(tmp_path):
    workdir, raw = _run_corpus(tmp_path, "build_running_identifier_pdf")
    assert raw["task"]["status"] == "completed"
    assert "removed_watermarks" not in raw["task"]
    assert not raw["audit"].get("removed_watermarks")
    output = fitz.open(str(workdir / "outputs" / "golden0000000000.pdf"))
    assert ["Model7Bv2" in page.get_text() for page in output] == [True, True]


def test_pipeline_removes_the_code_without_translating_it(tmp_path):
    import golden_support

    font = golden_support.build_test_font(tmp_path)
    if font is None:
        pytest.skip("no glyf-outline CJK system font available")
    workdir = tmp_path / "work"
    workdir.mkdir()
    snapshot = tmp_path / "snapshot.json"
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(TESTS_DIR), str(workdir), str(font[0]), str(snapshot)],
        env={**os.environ, "PYTHONPATH": str(PAYLOAD_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    task, audit = raw["task"], raw["audit"]
    assert task["status"] == "completed"
    assert task["source_page_fallbacks"] == []
    assert [(r["page"], r["text"]) for r in task["removed_watermarks"]] == [(1, CODE)]
    assert [(r["page"], r["text"]) for r in audit["removed_watermarks"]] == [(1, CODE)]
    assert not any(CODE in request for request in raw["requests"])
    assert not any(leak.get("page") == 1 for leak in task.get("untranslated_english_leaks", []))

    output = fitz.open(str(workdir / "outputs" / "golden0000000000.pdf"))
    source = fitz.open(str(workdir / "uploads" / "golden0000000000.pdf"))
    page_text = output[0].get_text("text", clip=fitz.Rect(-50, -50, 700, 900))
    assert CODE not in page_text
    assert "20261112" in page_text
    corner = fitz.Rect(0, 783, 70, 794)
    source_ink = source[0].get_pixmap(clip=corner, colorspace=fitz.csGRAY, alpha=False)
    output_ink = output[0].get_pixmap(clip=corner, colorspace=fitz.csGRAY, alpha=False)
    assert min(source_ink.samples) < 128  # the code is visible in the source
    assert min(output_ink.samples) >= 250  # and nothing is drawn in its place
