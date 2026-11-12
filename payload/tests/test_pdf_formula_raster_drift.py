"""Final formula audit: low-amplitude raster drift versus real changes.

PyMuPDF's redaction rewrites the page content stream, which can move the
antialiasing of untouched glyphs by a grey level or two.  The audit tolerates
that drift only when every vector fingerprint of the region matched and no
pixel moved by more than ``PDF_FORMULA_RASTER_DRIFT_MAX_DELTA``; overlays,
whiteouts and moved or removed glyphs still fail.  Every fixture is synthetic.
"""

from __future__ import annotations

import random

import fitz
import pytest

from phoena_translator.pdf.extract.signatures import _pdf_formula_region_signature
from phoena_translator.pdf.verify.output import (
    PDF_FORMULA_RASTER_DRIFT_MAX_DELTA,
    _audit_pdf_formula_regions,
    _pdf_formula_raster_difference_summary,
)

T = PDF_FORMULA_RASTER_DRIFT_MAX_DELTA
WIDTH, HEIGHT = 120, 30


def _pixmap(values: list[int]) -> fitz.Pixmap:
    return fitz.Pixmap(fitz.csGRAY, WIDTH, HEIGHT, bytes(values), False)


def _base_values() -> list[int]:
    """A glyph-like grey image: mid-grey strokes on a light background."""
    rng = random.Random(7)
    return [
        rng.choice((40, 90, 140, 200)) if (x // 6 + y // 5) % 3 == 0 else 245
        for y in range(HEIGHT)
        for x in range(WIDTH)
    ]


def _drift(values: list[int], share: float, amplitude: int, seed: int = 11) -> list[int]:
    """Move ``share`` of the pixels by +/-``amplitude`` grey levels."""
    rng = random.Random(seed)
    out = list(values)
    for index in rng.sample(range(len(out)), int(len(out) * share)):
        sign = rng.choice((-1, 1))
        out[index] = max(0, min(255, out[index] + sign * amplitude))
    return out


def _core_index(dx: int = 0, dy: int = 0) -> int:
    return (HEIGHT // 2 + dy) * WIDTH + (WIDTH // 2 + dx)


def _summary(source, observed, *, vector=True, spans=1) -> dict:
    return _pdf_formula_raster_difference_summary(
        _pixmap(source),
        _pixmap(observed),
        vector_fingerprints_match=vector,
        span_count=spans,
    )


def test_threshold_is_measured_and_capped():
    assert 1 <= T <= 4


def test_thirty_percent_of_pixels_at_one_level_is_tolerated_as_drift():
    base = _base_values()
    summary = _summary(base, _drift(base, 0.30, 1))
    assert summary["tolerable"] is True
    assert summary["tolerance"] == "low-amplitude-drift"
    assert summary["max_delta"] == 1
    assert 0.25 <= summary["diff_ratio"] <= 0.31
    assert summary["boundary_only"] is False


def test_drift_at_exactly_the_threshold_is_tolerated():
    base = _base_values()
    summary = _summary(base, _drift(base, 0.30, T))
    assert summary["tolerable"] is True
    assert summary["max_delta"] == T


def test_drift_one_level_above_the_threshold_fails():
    base = _base_values()
    summary = _summary(base, _drift(base, 0.30, T + 1))
    assert summary["tolerable"] is False
    assert summary["max_delta"] == T + 1


def test_one_core_pixel_above_the_threshold_fails_despite_global_drift():
    base = _base_values()
    observed = _drift(base, 0.30, 1)
    index = _core_index()
    observed[index] = base[index] + T + 1 if base[index] < 128 else base[index] - T - 1
    summary = _summary(base, observed)
    assert summary["tolerable"] is False
    assert summary["max_delta"] == T + 1


def test_white_box_over_the_core_fails():
    base = _base_values()
    observed = list(base)
    for y in range(HEIGHT // 4, 3 * HEIGHT // 4):
        for x in range(WIDTH // 4, 3 * WIDTH // 4):
            observed[y * WIDTH + x] = 255
    summary = _summary(base, observed)
    assert summary["tolerable"] is False
    assert summary["max_delta"] >= 50


def test_tiny_drift_without_vector_identity_fails():
    base = _base_values()
    assert _summary(base, _drift(base, 0.30, 1), vector=False)["tolerable"] is False


def test_tiny_drift_over_a_region_without_text_spans_fails():
    """With no span the vector match is vacuous; keep the exact raster rule."""
    base = _base_values()
    assert _summary(base, _drift(base, 0.30, 1), spans=0)["tolerable"] is False


def test_existing_sparse_and_boundary_rules_are_unchanged():
    base = _base_values()
    sparse = list(base)
    sparse[_core_index()] = 255 - sparse[_core_index()] // 2  # large single delta
    summary = _summary(base, sparse, vector=False, spans=0)
    assert summary["tolerable"] is True and summary["tolerance"] == "sparse"

    edge = list(base)
    for x in range(WIDTH):
        edge[x] = 0  # top row only, 0.8% of pixels
    summary = _summary(base, edge, vector=False, spans=0)
    assert summary["tolerable"] is True and summary["tolerance"] == "boundary"


def test_identical_and_mismatched_shapes():
    base = _base_values()
    assert _summary(base, list(base))["tolerance"] == "identical"
    other = fitz.Pixmap(fitz.csGRAY, WIDTH + 1, HEIGHT, bytes(len(base) + HEIGHT), False)
    summary = _pdf_formula_raster_difference_summary(
        _pixmap(base), other, vector_fingerprints_match=True, span_count=1
    )
    assert summary["tolerable"] is False and summary["max_delta"] is None


# ---------------------------------------------------------------------------
# Audit level: a synthetic PDF with one protected region.
# ---------------------------------------------------------------------------

FORMULA_BBOX = (72.0, 90.0, 190.0, 104.0)


def _source_pdf(text: str = "x+y-2=0", dx: float = 0.0) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((74.0 + dx, 101.0), text, fontsize=11, fontname="helv")
    page.insert_text((74.0, 140.0), "Body text below the formula.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def _audit(source_bytes: bytes, output_bytes: bytes):
    source = fitz.open("pdf", source_bytes)
    output = fitz.open("pdf", output_bytes)
    expectation = {
        "page": 1,
        "element": 3,
        "bbox": list(FORMULA_BBOX),
        "signature": _pdf_formula_region_signature(source[0], FORMULA_BBOX),
    }
    warnings: list[dict] = []
    verified, tolerated = _audit_pdf_formula_regions(
        output, source, [expectation], {}, warnings, set()
    )
    return verified, tolerated, warnings


def _modified(source_bytes: bytes, edit) -> bytes:
    doc = fitz.open("pdf", source_bytes)
    edit(doc[0])
    data = doc.tobytes()
    doc.close()
    return data


def _underlay_one_level(page) -> None:
    """A near-white fill *beneath* the content: glyphs untouched, background
    moves by one grey level -- the amplitude of the sanitize drift."""
    level = 254 / 255
    page.draw_rect(fitz.Rect(FORMULA_BBOX), color=None, fill=(level, level, level), overlay=False)


def test_audit_tolerates_one_level_drift_and_records_its_measure():
    source = _source_pdf()
    verified, tolerated, warnings = _audit(source, _modified(source, _underlay_one_level))
    assert warnings == []
    assert verified == 1
    assert len(tolerated) == 1
    record = tolerated[0]
    assert record["tolerance"] == "low-amplitude-drift"
    assert record["max_delta"] == 1
    assert record["diff_ratio"] > 0.10  # beyond both legacy rules


def test_audit_fails_a_white_box_painted_over_the_region():
    source = _source_pdf()

    def whiteout(page):
        page.draw_rect(fitz.Rect(FORMULA_BBOX), color=None, fill=(1, 1, 1), overlay=True)

    verified, tolerated, warnings = _audit(source, _modified(source, whiteout))
    assert verified == 0 and tolerated == []
    assert [w["type"] for w in warnings] == ["formula-integrity-mismatch"]
    # The text is still in the content stream under the box: only the raster
    # differs, and the warning carries the measured amplitude.
    assert warnings[0]["mismatches"] == ["raster_sha256"]
    assert warnings[0]["max_delta"] > T


def test_audit_fails_a_changed_glyph_even_with_tiny_drift():
    verified, tolerated, warnings = _audit(
        _source_pdf("x+y-2=0"),
        _modified(_source_pdf("x+y-3=0"), _underlay_one_level),
    )
    assert verified == 0 and tolerated == []
    assert "text_sha256" in warnings[0]["mismatches"]
    assert "max_delta" not in warnings[0]


def test_audit_fails_a_moved_glyph_run():
    verified, tolerated, warnings = _audit(_source_pdf(), _source_pdf(dx=1.0))
    assert verified == 0 and tolerated == []
    assert "geometry_sha256" in warnings[0]["mismatches"]


@pytest.mark.parametrize("fill_level", [(255 - T - 1) / 255, 230 / 255])
def test_audit_underlay_above_the_threshold_fails(fill_level):
    """Same mechanism as the tolerated case, but the background moves by more
    than ``T`` levels: a perceptible tint over the region is not drift."""
    source = _source_pdf()

    def tint(page):
        page.draw_rect(fitz.Rect(FORMULA_BBOX), color=None,
                       fill=(fill_level, fill_level, fill_level), overlay=False)

    verified, tolerated, warnings = _audit(source, _modified(source, tint))
    assert verified == 0 and tolerated == []
    assert warnings[0]["mismatches"] == ["raster_sha256"]
    assert warnings[0]["max_delta"] > T
