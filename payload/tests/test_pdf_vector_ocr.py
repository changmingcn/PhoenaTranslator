"""Vector-outline text recovery: region detection, element construction and
the page-level OCR pass (tesseract replaced by synthetic TSV output).

Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import re
import subprocess
import sys

import cv2
import fitz
import numpy as np
import pytest

from phoena_translator.pdf.extract import vector_ocr
from phoena_translator.pdf.model import ElementType

SCALE = 3.0  # px per point at the default 216 dpi
PAGE = fitz.Rect(0, 0, 612, 792)


def _blank(width_pt=612, height_pt=792):
    return np.full((int(height_pt * SCALE), int(width_pt * SCALE), 3), 255, np.uint8)


def _px(rect):
    return tuple(int(round(value * SCALE)) for value in rect)


def _box(image, rect, *, fill=None, border=3):
    x0, y0, x1, y1 = _px(rect)
    if fill is not None:
        cv2.rectangle(image, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), border)


def _line(text, x0, y0, x1, y1):
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def _tsv(lines):
    rows = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"]
    for number, (text, (x0, y0, x1, y1)) in enumerate(lines, 1):
        words = text.split()
        step = (x1 - x0) / len(words)
        for index, word in enumerate(words):
            rows.append(
                f"5\t1\t{number}\t1\t1\t{index + 1}\t{(x0 + index * step) * SCALE:.0f}\t"
                f"{y0 * SCALE:.0f}\t{step * SCALE * 0.9:.0f}\t{(y1 - y0) * SCALE:.0f}\t91\t{word}"
            )
    return "\n".join(rows) + "\n"


def _alpha_words(lines):
    return sum(len(re.findall(r"[A-Za-z]{2,}", text)) for text, _ in lines)


# ---------------------------------------------------------------------------
# _pdf_vector_ocr_detect_regions
# ---------------------------------------------------------------------------


def test_regions_keep_large_rectangular_cells_only():
    image = _blank()
    _box(image, (72, 100, 300, 220))  # diagram cell
    _box(image, (72, 300, 100, 320))  # too narrow
    _box(image, (72, 400, 300, 403))  # too flat
    cv2.circle(image, _px((450, 500, 0, 0))[:2], 150, (0, 0, 0), -1)  # not a rectangle
    (region,) = vector_ocr._pdf_vector_ocr_detect_regions(image, [], SCALE)
    assert region.contains(fitz.Rect(73, 101, 299, 219))
    assert fitz.Rect(71, 99, 301, 221).contains(region)


def _connected_flowchart():
    """Two light-filled cells joined by a connector: one non-rectangular blob."""
    image = _blank()
    _box(image, (72, 100, 300, 160), fill=(200, 200, 200))
    _box(image, (72, 220, 300, 280), fill=(200, 200, 200))
    x = int(186 * SCALE)
    cv2.line(image, (x, int(160 * SCALE)), (x, int(220 * SCALE)), (0, 0, 0), 9)
    return image


def test_regions_recover_flowchart_cells_from_their_borders():
    lines = [_line("first cell text", 90, 120, 200, 130), _line("second cell", 90, 240, 200, 250)]
    regions = vector_ocr._pdf_vector_ocr_detect_regions(_connected_flowchart(), lines, SCALE)
    assert len(regions) == 2
    for region, line in zip(regions, lines):
        assert region.contains(fitz.Rect(line["bbox"])) and region.height < 70


def test_border_pairs_need_ocr_text_inside_and_side_borders():
    assert vector_ocr._pdf_vector_ocr_detect_regions(_connected_flowchart(), [], SCALE) == []
    image = _blank()
    for y in (100, 160):
        cv2.line(image, _px((72, y, 0, 0))[:2], _px((0, 0, 300, y))[2:], (0, 0, 0), 3)
    assert vector_ocr._pdf_vector_ocr_detect_regions(
        image, [_line("between rules", 90, 120, 200, 130)], SCALE
    ) == []


def test_regions_inside_a_frame_collapse_into_the_frame():
    image = _blank()
    _box(image, (72, 100, 300, 160))
    _box(image, (72, 220, 300, 280))
    _box(image, (60, 90, 560, 300), border=2)
    (region,) = vector_ocr._pdf_vector_ocr_detect_regions(image, [], SCALE)
    assert region.contains(fitz.Rect(61, 91, 559, 299))
    # With text inside, border pairing proposes the frame again; the larger
    # of the near-identical proposals wins.
    lines = [_line("first cell text", 90, 120, 200, 130), _line("second cell", 90, 240, 200, 250)]
    (paired,) = vector_ocr._pdf_vector_ocr_detect_regions(image, lines, SCALE)
    assert paired.get_area() >= region.get_area()


def test_regions_merge_near_duplicates_keeping_the_larger():
    # A thick border yields nested outer/inner contours of one cell.
    image = _blank()
    _box(image, (72, 100, 300, 220), border=9)
    (region,) = vector_ocr._pdf_vector_ocr_detect_regions(image, [], SCALE)
    assert region.contains(fitz.Rect(72, 100, 300, 220))


def test_regions_require_opencv(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="OpenCV is required"):
        vector_ocr._pdf_vector_ocr_detect_regions(_blank(10, 10), [], SCALE)


# ---------------------------------------------------------------------------
# _make_pdf_vector_ocr_element
# ---------------------------------------------------------------------------


def test_element_from_only_artifacts_is_none():
    lines = [_line("|", 72, 100, 80, 110), _line("vv", 90, 100, 100, 110)]
    assert vector_ocr._make_pdf_vector_ocr_element(lines, PAGE, _blank(), SCALE) is None


def test_tall_diagram_cell_becomes_a_title_and_bullets():
    region = fitz.Rect(72, 100, 300, 220)
    lines = [
        _line("Market Structure", 76, 104, 200, 114),
        _line("Dealers absorb order flow", 80, 145, 290, 155),
        _line("and hedge inventory", 80, 157, 290, 167),
        _line("Spreads widen", 80, 190, 290, 200),
        _line("|", 80, 205, 85, 210),
    ]
    image = _blank()
    elem = vector_ocr._make_pdf_vector_ocr_element(lines, PAGE, image, SCALE, region=region)

    assert elem["content"] == (
        "Market Structure\n\n• Dealers absorb order flow and hedge inventory\n\n• Spreads widen"
    )
    assert elem["rich_content"].startswith("<b>Market Structure</b>\n\n• Dealers")
    titles = [paragraph["gap_before"] for paragraph in elem["paragraphs"]]
    assert titles == [0.0, 2.0, 2.0]
    assert elem["paragraphs"][1]["source_bbox"] == [80.0, 145.0, 290.0, 167.0]
    assert elem["bbox"] == [75.0, 103.0, 297.0, 217.0]
    assert elem["layout_class"] == "table" and elem["table_hint"] and elem["diagram_cell_hint"]
    (fill,) = elem["vector_ocr_fill_rects"]
    assert fill["bbox"] == elem["bbox"] and fill["fill"] == [1.0, 1.0, 1.0]
    assert elem["vector_ocr_erase_rects"] == []
    assert elem["color"] == 0x000000 and elem["merged_visual_line_count"] == 4
    assert re.fullmatch(r"[0-9a-f]{64}", elem["vector_ocr_source_signature"])


def test_short_coded_cell_on_a_dark_fill_is_one_paragraph_in_white():
    region = fitz.Rect(320, 100, 540, 150)
    image = _blank()
    _box(image, region, fill=(128, 51, 26))
    lines = [
        _line("EIA11: Funding costs rise", 326, 110, 530, 120),
        _line("when collateral is scarce", 326, 122, 530, 132),
    ]
    elem = vector_ocr._make_pdf_vector_ocr_element(lines, PAGE, image, SCALE, region=region)
    assert len(elem["paragraphs"]) == 1
    assert elem["rich_content"] == "<b>EIA11:</b> Funding costs rise when collateral is scarce"
    assert elem["inline_math_fragments"] == [{"text": "EIA11", "font_kind": "vector-ocr-code"}]
    assert elem["color"] == 0xFFFFFF


def test_outside_labels_erase_each_line_and_split_at_notes():
    lines = [
        _line("Source: dealer survey", 72, 300, 300, 310),
        _line("Note 1: monthly averages", 72, 312, 300, 322),
        _line("of positions", 72, 324, 300, 334),
        _line("Footer text far below", 2, 500, 300, 510),
    ]
    elem = vector_ocr._make_pdf_vector_ocr_element(lines, PAGE, _blank(), SCALE)
    assert [p["plain"] for p in elem["paragraphs"]] == [
        "Source: dealer survey",
        "Note 1: monthly averages of positions",
        "Footer text far below",
    ]
    assert elem["rich_content"] is None and elem["layout_class"] == "scattered"
    assert elem["vector_ocr_fill_rects"] == []
    erase = [spec["bbox"] for spec in elem["vector_ocr_erase_rects"]]
    assert erase[0] == [70.5, 299.0, 301.5, 311.0]
    assert erase[3][0] == 0.5  # clipped to the page
    other = vector_ocr._make_pdf_vector_ocr_element(lines[:1], PAGE, _blank(), SCALE)
    assert other["vector_ocr_source_signature"] != elem["vector_ocr_source_signature"]


# ---------------------------------------------------------------------------
# _extract_pdf_vector_ocr_elements
# ---------------------------------------------------------------------------

DIAGRAM_LINES = [
    ("Market Structure Overview", (76, 104, 200, 114)),
    ("Dealers absorb order flow from their institutional clients", (80, 145, 290, 155)),
    ("and hedge the remaining inventory in futures markets", (80, 157, 290, 167)),
    ("Spreads widen sharply when market volatility rises", (80, 190, 290, 200)),
    ("EIA11: Funding costs rise for every dealer bank", (326, 110, 530, 120)),
    ("when collateral becomes scarce in repo markets", (326, 122, 530, 132)),
    ("Note 1: values are monthly averages of dealer positions in bonds", (72, 300, 400, 310)),
    ("measured at quarter end dates for each reporting firm", (72, 312, 400, 322)),
    ("Source: survey of primary dealers and their largest clients", (72, 400, 400, 410)),
    ("vv", (420, 400, 430, 410)),
    ("Right margin label text here", (450, 412, 590, 422)),
    ("Q4", (500, 600, 520, 610)),
    ("Existing native caption", (80, 503, 200, 513)),
]


def _diagram_page():
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(72, 100, 300, 220), color=(0, 0, 0), fill=(1, 1, 1), width=1.5)
    page.draw_rect(fitz.Rect(320, 100, 540, 150), color=(0, 0, 0), fill=(0.1, 0.2, 0.5), width=1.5)
    for index in range(24):
        page.draw_line((72 + index * 10, 700), (72 + index * 10, 706), color=(0, 0, 0))
    page.insert_text((80, 112), "Market Structure Overview", fontsize=8)
    page.insert_text((80, 510), "Existing native caption", fontsize=8)
    return document, page


def _native_elements():
    return [
        {"type": ElementType.TEXT, "bbox": [80, 104, 200, 114], "content": "Market Structure Overview"},
        {"type": ElementType.TEXT, "bbox": [80, 503, 200, 513], "content": "Existing native caption"},
        {"type": ElementType.FORMULA_IMAGE, "bbox": [80, 104, 200, 114]},
        {"type": ElementType.TEXT, "bbox": [10, 10, 10, 10], "content": "x"},
    ]


@pytest.fixture
def ocr(monkeypatch):
    """Pretend tesseract is installed and answers with the given lines."""

    answer = {"lines": DIAGRAM_LINES}
    monkeypatch.setattr(vector_ocr.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(vector_ocr, "_run_pdf_tesseract", lambda _png, _fmt: _tsv(answer["lines"]))
    return answer


def test_page_ocr_recovers_cells_and_labels_and_supersedes_native_text(ocr):
    assert _alpha_words(DIAGRAM_LINES) >= 60
    document, page = _diagram_page()
    elements = _native_elements()
    additions = vector_ocr._extract_pdf_vector_ocr_elements(page, elements, [])
    document.close()

    assert [elem["layout_class"] for elem in additions] == [
        "table", "table", "scattered", "scattered", "scattered"
    ]
    cell, coded, note, source, margin = additions
    assert cell["content"].startswith("Market Structure Overview\n\n• Dealers absorb")
    assert coded["rich_content"].startswith("<b>EIA11:</b>") and coded["color"] == 0xFFFFFF
    assert note["content"] == (
        "Note 1: values are monthly averages of dealer positions in bonds "
        "measured at quarter end dates for each reporting firm"
    )
    assert source["content"] == "Source: survey of primary dealers and their largest clients"
    # A label on the next row but far to the right is its own group; a group
    # without two real words ("Q4") is dropped.
    assert margin["content"] == "Right margin label text here"
    # Native text under recovered paint is superseded; the matched caption stays.
    assert elements[0]["type"] == ElementType.MERGED_AWAY and elements[0]["vector_ocr_superseded"]
    assert elements[1]["type"] == ElementType.TEXT
    assert elements[2]["type"] == ElementType.FORMULA_IMAGE


def test_page_ocr_skips_pages_that_are_not_outline_text():
    document, page = _diagram_page()
    assert vector_ocr._extract_pdf_vector_ocr_elements(page, [], [fitz.Rect(0, 0, 10, 10)]) == []
    plain = document.new_page()
    assert vector_ocr._extract_pdf_vector_ocr_elements(plain, [], []) == []
    document.close()


def test_page_ocr_requires_tesseract_and_opencv(monkeypatch):
    document, page = _diagram_page()
    monkeypatch.setattr(vector_ocr.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires tesseract OCR"):
        vector_ocr._extract_pdf_vector_ocr_elements(page, [], [])
    monkeypatch.setattr(vector_ocr.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="requires OpenCV OCR support"):
        vector_ocr._extract_pdf_vector_ocr_elements(page, [], [])
    document.close()


def test_page_ocr_reports_a_failed_tesseract_run(ocr, monkeypatch):
    def fail(_png, _fmt):
        raise subprocess.CalledProcessError(1, "tesseract")

    monkeypatch.setattr(vector_ocr, "_run_pdf_tesseract", fail)
    document, page = _diagram_page()
    with pytest.raises(RuntimeError, match=r"vector-text OCR failed \(CalledProcessError\)"):
        vector_ocr._extract_pdf_vector_ocr_elements(page, [], [])
    document.close()


def test_page_ocr_gate_rejects_pages_with_few_ocr_words(ocr, caplog):
    ocr["lines"] = DIAGRAM_LINES[:3]
    document, page = _diagram_page()
    with caplog.at_level("INFO", logger="translator"):
        assert vector_ocr._extract_pdf_vector_ocr_elements(page, [], []) == []
    document.close()
    assert "vector OCR gate rejected chart page" in caplog.text


def test_page_ocr_reports_an_undecodable_raster(ocr, monkeypatch):
    monkeypatch.setattr(cv2, "imdecode", lambda *_args: None)
    document, page = _diagram_page()
    with pytest.raises(RuntimeError, match="raster decode failed"):
        vector_ocr._extract_pdf_vector_ocr_elements(page, [], [])
    document.close()


def test_page_ocr_without_new_text_is_an_error(ocr):
    # Every OCR line is already present as native text in place.
    ocr["lines"] = [(text, bbox) for text, bbox in DIAGRAM_LINES if text != "vv"]
    native = [
        {"type": ElementType.TEXT, "bbox": list(bbox), "content": text}
        for text, bbox in ocr["lines"]
    ]
    document, page = _diagram_page()
    with pytest.raises(RuntimeError, match="no semantic regions"):
        vector_ocr._extract_pdf_vector_ocr_elements(page, native, [])
    document.close()
