"""Math Detection for deterministic PDF processing."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter

import fitz
from phoena_translator.math_text import (
    MATH_FUNCTION_WORDS,
    is_math_block as _is_math_block,
    is_unicode_math_symbol as _is_unicode_math_symbol,
    iter_unicode_math_symbol_ranges as _iter_unicode_math_symbol_ranges,
    unicode_math_symbol_count as _unicode_math_symbol_count,
)

from phoena_translator.pdf.types import (
    PDF_BATCH_SEGMENT_RE,
    PDF_FORMULA_GUARD_PADDING,
    PDF_IDENTIFIER_PLACEHOLDER_RE,
    PDF_INLINE_MATH_PLACEHOLDER_RE,
    PDF_MATH_ITALIC_FONT_TOKENS,
    PDF_SIGNATURE_GEOMETRY_QUANTUM,
    PDF_STRONG_MATH_FONT_TOKENS,
    PDF_SUPERSCRIPT_PLACEHOLDER_RE,
    _PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE,
    _PRESERVED_IDENTIFIER_RE,
)
from phoena_translator.pdf.geometry import (
    _pdf_rect_intersects_protected,
    _subtract_pdf_protected_rects,
)

def _looks_like_pdf_wrapped_identifier_query_tail(text: str) -> bool:
    """Recognize a high-confidence query-string continuation line."""
    compact = re.sub(r"\s+", "", html.unescape(text or "")).strip()
    if (
        not compact
        or "&" not in compact
        or compact.count("=") < 2
        or not re.fullmatch(r"[A-Za-z0-9._~%+&=:#?/-]+[.]?", compact)
    ):
        return False
    keys = re.findall(
        r"(?:^|[?&])([A-Za-z][A-Za-z0-9._~-]*)=",
        compact,
    )
    # Requiring a descriptive key keeps ordinary compact equations such as
    # ``x=2&y=3`` outside this identifier exception while admitting real PDF
    # wraps such as ``D=2&PageID=43``.
    return len(keys) >= 2 and any(len(key) >= 3 for key in keys)


def _looks_like_pdf_identifier_line(text: str) -> bool:
    """Recognize URL/query lines before mathematical glyph protection.

    Bibliographies commonly wrap a URL onto its own native PDF line.  Query
    parameters such as ``abstract=1641387`` contain ``=`` (Unicode category
    ``Sm``), which otherwise makes the line look like an equation and can in
    turn promote the adjacent ``Accessed:`` line into protected formula
    geometry.  These patterns are opaque identifiers, not mathematical
    notation, so preserve their bytes as text and let the reference-entry
    merger keep the whole citation semantic.
    """
    plain = re.sub(r"\s+", " ", html.unescape(text or "")).strip()
    if not plain:
        return False
    if re.search(r"(?i)\b(?:https?://|www\.)", plain):
        return True
    if re.search(
        r"(?i)(?:^|[./])(?:com|org|gov|edu|net|int|eu|uk|co\.uk)/\S*[=?&%]",
        plain,
    ):
        return True
    if re.search(
        r"(?i)\b[A-Za-z0-9_-]+\.(?:do|aspx?|php|html?)\?\S*=",
        plain,
    ):
        return True
    if _looks_like_pdf_wrapped_identifier_query_tail(plain):
        return True
    return False


def _normalize_pdf_font_name(font_name: str) -> str:
    name = re.sub(r"^[A-Z]{6}\+", "", str(font_name or ""))
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _pdf_math_font_kind(font_name: str) -> str | None:
    normalized = _normalize_pdf_font_name(font_name)
    if not normalized:
        return None
    if any(token.replace("-", "") in normalized for token in PDF_STRONG_MATH_FONT_TOKENS):
        return "strong"
    if any(token in normalized for token in PDF_MATH_ITALIC_FONT_TOKENS):
        return "italic"
    return None


def _pdf_inline_math_span_fragments(spans: list[dict]) -> list[dict]:
    """Return exact, ordered fragments for safe inline-math placeholders."""
    fragments = []
    for span in spans or []:
        text = span.get("text", "")
        if not text or not text.strip():
            continue
        font_kind = _pdf_math_font_kind(span.get("font", ""))
        if font_kind is not None:
            fragments.append({"text": text, "font_kind": font_kind})
            continue
        for start, end in _iter_unicode_math_symbol_ranges(text):
            fragment = text[start:end]
            if fragment:
                fragments.append({"text": fragment, "font_kind": "unicode"})
    return fragments


def _looks_like_pdf_math_italic_run(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact or len(compact) > 24:
        return False
    if any(_is_unicode_math_symbol(char) for char in compact):
        return True
    if re.search(r"[\d_=<>+*/^{}\[\]()]", compact):
        return True
    tokens = re.findall(r"[A-Za-z]+", compact)
    if not tokens:
        return False
    def _token_is_math_like(token: str) -> bool:
        lowered = token.lower()
        return (
            len(token) <= 2
            or lowered in MATH_FUNCTION_WORDS
            or (len(token) == 3 and token.islower() and token.startswith("d"))
        )

    return len(tokens) <= 6 and all(_token_is_math_like(token) for token in tokens)


def _pdf_line_math_evidence(
    span_entries: list[dict],
    table_hint: bool = False,
) -> dict:
    """Return conservative, explainable math evidence for one native PDF line."""
    visible = [span for span in (span_entries or []) if span.get("text", "").strip()]
    text = "".join(span.get("text", "") for span in (span_entries or []))
    # ``Min`` and ``Max`` are ordinary statistic column headers when they are
    # printed in a regular text font inside a detected table. Treating the
    # bare words as mathematical functions creates immutable formula boxes;
    # neighboring Q1/Q3 headers can then be pulled into those boxes by formula
    # context propagation. Keep the exemption context-bound and retain math
    # protection for math-font runs and for the same words outside tables.
    plain_label = re.sub(r"\s+", " ", text).strip()
    if (
        table_hint
        and re.fullmatch(r"(?i)(?:min|max)\.?", plain_label)
        and visible
        and all(
            _pdf_math_font_kind(span.get("font", "")) is None
            for span in visible
        )
    ):
        return {
            "protected": False,
            "reasons": [],
            "mixed": False,
            "math_symbol_count": 0,
            "inline_math_fragments": [],
        }
    # Parenthesized list labels such as ``(b)`` are document structure, not
    # formulae.  Some regulatory PDFs store the marker in a standalone text
    # object next to the prose body; treating it as equation structure keeps
    # it outside the semantic paragraph and defeats whole-paragraph
    # translation/redraw.
    if _looks_like_pdf_list_marker(text):
        return {
            "protected": False,
            "reasons": [],
            "mixed": False,
            "math_symbol_count": 0,
            "inline_math_fragments": [],
        }
    if _looks_like_pdf_identifier_line(text):
        return {
            "protected": False,
            "reasons": [],
            "mixed": False,
            "math_symbol_count": 0,
            "inline_math_fragments": [],
        }
    compact = re.sub(r"\s+", "", text)
    reasons = set()
    strong_spans = [span for span in visible if _pdf_math_font_kind(span.get("font", "")) == "strong"]
    italic_spans = [span for span in visible if _pdf_math_font_kind(span.get("font", "")) == "italic"]

    if strong_spans:
        reasons.add("math_font")
    if "\ufffd" in text:
        reasons.add("decoded_symbol")
    math_symbol_count = _unicode_math_symbol_count(text)
    if math_symbol_count:
        reasons.add("math_symbol")
    if _is_math_block(text):
        reasons.add("equation_structure")

    # In Computer Modern Sans documents, CMSSI is used for variables while
    # surrounding prose stays CMSS. Preserve compact CMSSI runs, but do not
    # classify a long fully italic prose sentence as mathematics.
    italic_compact = re.sub(
        r"\s+", "", "".join(span.get("text", "") for span in italic_spans)
    )
    non_italic_visible = [
        span for span in visible
        if _pdf_math_font_kind(span.get("font", "")) != "italic"
    ]
    sizes = [float(span.get("size", 0.0)) for span in visible if float(span.get("size", 0.0)) > 0]
    size_variation = bool(sizes and min(sizes) <= max(sizes) * 0.82)
    if (
        italic_spans
        and italic_compact
        and _looks_like_pdf_math_italic_run(italic_compact)
        and (non_italic_visible or len(compact) <= 32 or size_variation)
    ):
        reasons.add("math_italic")

    if (strong_spans or italic_spans) and size_variation:
        reasons.add("script_geometry")

    prose_words = []
    for span in visible:
        if _pdf_math_font_kind(span.get("font", "")) is not None:
            continue
        for word in re.findall(r"[A-Za-z]{3,}", span.get("text", "")):
            if word.lower() not in MATH_FUNCTION_WORDS:
                prose_words.append(word)

    inline_fragments = _pdf_inline_math_span_fragments(visible)
    inline_math_text = "".join(fragment["text"] for fragment in inline_fragments)
    high_risk_inline_operator = bool(re.search(
        r"[\ufffd∑∫∏√⎧-⎭⎛-⎦]",
        inline_math_text,
    ))
    semantic_inline_math = bool(
        reasons
        and prose_words
        and len(prose_words) >= 5
        and "decoded_symbol" not in reasons
        and "equation_structure" not in reasons
        and not high_risk_inline_operator
        and math_symbol_count <= 6
        and len(re.sub(r"\s+", "", inline_math_text)) <= 24
    )

    return {
        "protected": bool(reasons) and not semantic_inline_math,
        "reasons": (
            ["semantic_inline_math"]
            if semantic_inline_math
            else sorted(reasons)
        ),
        "mixed": bool(reasons and prose_words),
        "math_symbol_count": math_symbol_count,
        "inline_math_fragments": inline_fragments if semantic_inline_math else [],
    }


def _demote_pdf_wrapped_identifier_math_lines(line_infos: list[dict]) -> list[dict]:
    """Restore a query-string tail split away from the preceding URL line."""
    for index, line in enumerate(line_infos or []):
        if not line.get("math_protected"):
            continue
        if not _looks_like_pdf_wrapped_identifier_query_tail(
            line.get("plain", "")
        ):
            continue
        previous_lines = line_infos[max(0, index - 2):index]
        if not any(
            _looks_like_pdf_identifier_line(previous.get("plain", ""))
            for previous in previous_lines
        ):
            continue
        line["math_protected"] = False
        line["math_reasons"] = []
        line["math_mixed"] = False
        line["math_symbol_count"] = 0
        line["inline_math_fragments"] = []
        line["wrapped_identifier_line"] = True
    return line_infos


def _propagate_pdf_math_line_context(line_infos: list[dict]) -> list[dict]:
    """Attach short equation fragments (bounds, labels) to nearby math lines."""
    protected_indices = [
        idx for idx, line in enumerate(line_infos)
        if line.get("math_protected")
    ]
    if not protected_indices:
        return line_infos

    for idx, line in enumerate(line_infos):
        if line.get("math_protected"):
            continue
        plain = re.sub(r"\s+", "", line.get("plain", ""))
        if not plain or len(plain) > 16:
            continue
        prose_words = [
            word for word in re.findall(r"[A-Za-z]{3,}", plain)
            if word.lower() not in MATH_FUNCTION_WORDS
        ]
        if prose_words:
            continue

        fontsize = max(float(line.get("fontsize", 0.0)), 6.0)
        for neighbor_idx in protected_indices:
            neighbor = line_infos[neighbor_idx]
            vertical_gap = max(
                0.0,
                float(line.get("y0", 0.0)) - float(neighbor.get("y1", 0.0)),
                float(neighbor.get("y0", 0.0)) - float(line.get("y1", 0.0)),
            )
            horizontal_gap = max(
                0.0,
                float(line.get("x0", 0.0)) - float(neighbor.get("x1", 0.0)),
                float(neighbor.get("x0", 0.0)) - float(line.get("x1", 0.0)),
            )
            if vertical_gap <= fontsize * 1.5 and horizontal_gap <= fontsize * 6.0:
                line["math_protected"] = True
                line["math_reasons"] = sorted(
                    set(line.get("math_reasons") or []) | {"equation_fragment_context"}
                )
                line["math_mixed"] = False
                break
    return line_infos


def _make_pdf_formula_element_from_lines(line_infos: list[dict]) -> dict | None:
    if not line_infos:
        return None
    rect = fitz.Rect(
        min(float(line["x0"]) for line in line_infos),
        min(float(line["y0"]) for line in line_infos),
        max(float(line["x1"]) for line in line_infos),
        max(float(line["y1"]) for line in line_infos),
    )
    reasons = sorted({
        reason
        for line in line_infos
        for reason in (line.get("math_reasons") or [])
    })
    return {
        "type": "formula_image",
        "y": rect.y0,
        "x": rect.x0,
        "rect": rect,
        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
        "math_reasons": reasons or ["equation_structure"],
        "math_mixed": any(bool(line.get("math_mixed")) for line in line_infos),
        "math_line_count": len(line_infos),
        "math_candidate_symbol_count": sum(
            int(line.get("math_symbol_count", 0)) for line in line_infos
        ),
    }


def _filter_pdf_formula_safe_rects(
    candidates: list[tuple[str, fitz.Rect]],
    protected_rects: list,
    padding: float = PDF_FORMULA_GUARD_PADDING,
    anchor_rect: fitz.Rect | None = None,
) -> list[tuple[str, fitz.Rect]]:
    safe = []
    clipped = []
    seen_clipped = set()

    for name, rect in candidates or []:
        candidate = fitz.Rect(rect)
        if not _pdf_rect_intersects_protected(candidate, protected_rects, padding):
            safe.append((name, candidate))
            continue

        # Adjacent PDF lines commonly have sub-point gaps. The formula guard can
        # therefore touch only a thin edge of an otherwise valid text box. Keep
        # the guard intact and trim the box to its largest usable safe piece
        # instead of treating one local collision as a document-wide failure.
        pieces = _subtract_pdf_protected_rects(candidate, protected_rects, padding)
        min_width = max(2.0, min(candidate.width * 0.35, 24.0))
        viable = [
            piece for piece in pieces
            if piece.width >= min_width and piece.height >= 2.0
        ]
        if not viable:
            continue

        # A ladder candidate can extend well beyond the source text to make
        # room for a longer translation.  If a formula cuts that extension in
        # two, choosing the globally largest remainder can move the paragraph
        # to the other side of the formula.  ASIC 452 p. 65 exposed the
        # failure: a one-line footnote continuation jumped below the equation
        # and overprinted the following citation.  Prefer the safe piece that
        # still overlaps the source anchor; use area only as a later tie-break.
        anchor = fitz.Rect(anchor_rect) if anchor_rect is not None else candidate
        best = max(
            viable,
            key=lambda piece: (
                (piece & anchor).get_area(),
                -abs(piece.y0 - anchor.y0),
                -abs(piece.x0 - anchor.x0),
                piece.get_area(),
                piece.width,
            ),
        )
        key = tuple(round(value, 3) for value in (best.x0, best.y0, best.x1, best.y1))
        if key in seen_clipped:
            continue
        seen_clipped.add(key)
        clipped.append((f"{name}+formula-trim", best))

    # Preserve the old preference for untouched rectangles. Formula-trimmed
    # candidates are a fail-safe only when every original candidate collides.
    return safe or clipped


def _pdf_signature_number(
    value: float,
    quantum: float = PDF_SIGNATURE_GEOMETRY_QUANTUM,
) -> float:
    """Quantize signature geometry and canonicalize signed zero.

    Incremental PyMuPDF saves can move an extracted coordinate by less than
    0.0001 pt. Decimal rounding is unstable exactly on a half-step (13.5500
    versus 13.5499), so use a quarter-point bucket instead.
    """
    quantum = max(float(quantum), 0.000001)
    rounded = round(round(float(value) / quantum) * quantum, 6)
    return 0.0 if rounded == 0 else rounded


def _pdf_formula_region_signature(page, bbox, page_dict: dict | None = None) -> dict:
    """Hash source text/font/geometry and raster pixels without exposing content."""
    rect = fitz.Rect(bbox) & page.rect
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        raise ValueError("empty formula region")
    if page_dict is None:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    records = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = re.sub(r"\s+", " ", span.get("text", "")).strip()
                if not span_text:
                    continue
                span_rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                if span_rect.is_empty:
                    continue
                inter = span_rect & rect
                center = fitz.Point(
                    (span_rect.x0 + span_rect.x1) / 2.0,
                    (span_rect.y0 + span_rect.y1) / 2.0,
                )
                if inter.is_empty and not rect.contains(center):
                    continue
                if not rect.contains(center) and inter.get_area() < span_rect.get_area() * 0.45:
                    continue
                records.append({
                    "text": span_text,
                    "font": _normalize_pdf_font_name(span.get("font", "")),
                    "size": round(float(span.get("size", 0.0)), 2),
                    "bbox": [
                        _pdf_signature_number(span_rect.x0 - rect.x0),
                        _pdf_signature_number(span_rect.y0 - rect.y0),
                        _pdf_signature_number(span_rect.x1 - rect.x0),
                        _pdf_signature_number(span_rect.y1 - rect.y0),
                    ],
                })
    records.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["font"], item["text"]))
    text_material = "\n".join(item["text"] for item in records)
    font_material = "\n".join(item["font"] for item in records)
    geometry_material = json.dumps(
        [{"font": item["font"], "size": item["size"], "bbox": item["bbox"]} for item in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    combined_material = json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2.0, 2.0),
        colorspace=fitz.csGRAY,
        alpha=False,
        clip=rect,
    )
    return {
        "signature_sha256": hashlib.sha256(combined_material.encode("utf-8")).hexdigest(),
        "text_sha256": hashlib.sha256(text_material.encode("utf-8")).hexdigest(),
        "font_sha256": hashlib.sha256(font_material.encode("utf-8")).hexdigest(),
        "geometry_sha256": hashlib.sha256(geometry_material.encode("utf-8")).hexdigest(),
        "raster_sha256": hashlib.sha256(pix.samples).hexdigest(),
        "span_count": len(records),
        "raster_width": pix.width,
        "raster_height": pix.height,
    }


def _plain_text(text: str) -> str:
    """Remove lightweight inline markup for translation checks."""
    # PDF rich text only emits a small, known markup vocabulary.  A generic
    # ``<...>`` stripper corrupts ordinary comparison prose such as
    # ``small (<£100,000) ... large (>£1,000,000)`` by treating the range as
    # one HTML tag.  Strip only actual lightweight tags and preserve literal
    # less-than / greater-than operators for translation and formula audit.
    return re.sub(
        r"(?is)<\s*/?\s*(?:b|strong|i|em|sup|sub|span|a)\b[^>]*>",
        "",
        text or "",
    )


def _pdf_span_origin_y(span: dict) -> float:
    origin = span.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
        return float(origin[1])
    return float(span.get("y1", span.get("bbox", (0.0, 0.0, 0.0, 0.0))[3]))


def _pdf_span_has_lexical_text(span: dict) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", span.get("text", "")))


def _pdf_nearby_superscript_reference(
    marker_span: dict,
    reference_spans: list[dict],
) -> dict | None:
    """Find the adjacent normal-size span that anchors a raised marker.

    MuPDF's HTML renderer may expose a visually inline ``sup`` as a separate
    line object. Coordinate proximity is therefore more reliable than line
    membership when validating the rendered marker.
    """
    marker_size = max(float(marker_span.get("size", 0.0)), 0.01)
    marker_rect = fitz.Rect(marker_span.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    candidates = []

    for reference in reference_spans:
        if reference is marker_span or not _pdf_span_has_lexical_text(reference):
            continue
        reference_size = max(float(reference.get("size", 0.0)), 0.01)
        scale = marker_size / reference_size
        if not 0.45 <= scale <= 0.80:
            continue

        raised_by = _pdf_span_origin_y(reference) - _pdf_span_origin_y(marker_span)
        if not reference_size * 0.12 <= raised_by <= reference_size * 0.80:
            continue

        reference_rect = fitz.Rect(reference.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        vertical_overlap = min(marker_rect.y1, reference_rect.y1) - max(marker_rect.y0, reference_rect.y0)
        if vertical_overlap <= 0:
            continue
        if reference_rect.x1 < marker_rect.x0:
            horizontal_gap = marker_rect.x0 - reference_rect.x1
        elif marker_rect.x1 < reference_rect.x0:
            horizontal_gap = reference_rect.x0 - marker_rect.x1
        else:
            horizontal_gap = 0.0
        if horizontal_gap > max(reference_size * 1.2, 8.0):
            continue

        candidates.append((
            horizontal_gap,
            abs(raised_by - reference_size * 0.30),
            -reference_size,
            reference,
        ))

    return min(candidates, key=lambda item: item[:3])[3] if candidates else None


def _pdf_native_superscript_fingerprints(
    page_data: dict,
    pattern: re.Pattern,
) -> Counter:
    """Fingerprint native superscript occurrences for exact source matching.

    An isolated mathematical superscript can have no adjacent normal-size span,
    so its relative scale is not measurable.  In that case we can still prove
    preservation without weakening the audit: require the output occurrence to
    retain the source font, absolute size, bbox, origin, and native superscript
    flag.  Counter intersection keeps duplicate markers one-to-one.
    """
    superscript_flag = int(getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1))
    fingerprints = Counter()
    for block in (page_data or {}).get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not (int(span.get("flags", 0)) & superscript_flag):
                    continue
                occurrences = len(list(pattern.finditer(span.get("text", ""))))
                if not occurrences:
                    continue
                bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                origin = span.get("origin", (0.0, 0.0))
                fingerprint = (
                    str(span.get("font", "")),
                    round(float(span.get("size", 0.0)), 3),
                    tuple(round(float(value), 2) for value in bbox),
                    tuple(round(float(value), 2) for value in origin[:2]),
                )
                fingerprints[fingerprint] += occurrences
    return fingerprints


def _pdf_exact_source_native_superscript_matches(
    source_page_data: dict,
    output_page_data: dict,
    pattern: re.Pattern,
) -> int:
    source = _pdf_native_superscript_fingerprints(source_page_data, pattern)
    output = _pdf_native_superscript_fingerprints(output_page_data, pattern)
    return sum((source & output).values())


def _pdf_is_semantic_superscript_marker(text: str) -> bool:
    """Reject corrupt font-decoding runs while retaining citation/math markers."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact or "\ufffd" in compact or len(compact) > 12:
        return False
    punctuation = frozenset("[](){}.,:;+-\u2212\u2013\u2014*\u2020\u2021\u00a7\u00b6")
    return (
        all(char.isalnum() or char in punctuation for char in compact)
        and any(char.isalnum() or char in "*\u2020\u2021\u00a7\u00b6" for char in compact)
    )


def _mark_pdf_superscript_spans(span_entries: list[dict]) -> list[dict]:
    """Mark semantic superscripts, preferring PyMuPDF's native font flag.

    Some generators omit that flag, so a conservative fallback accepts only a
    short marker that is both smaller and visibly raised next to normal inline
    text. Isolated line-leading footnote definitions therefore stay ordinary.
    """
    if not span_entries:
        return span_entries

    native_flag = int(getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1))
    non_empty_indices = [
        idx for idx, span in enumerate(span_entries)
        if span.get("text", "").strip()
    ]

    for idx, span in enumerate(span_entries):
        span["superscript"] = False
        span["superscript_source"] = None
        span["superscript_scale"] = None

        text = span.get("text", "")
        compact = re.sub(r"\s+", "", text)
        semantic_marker = _pdf_is_semantic_superscript_marker(compact)
        native = bool(int(span.get("flags", 0)) & native_flag) and semantic_marker
        inferred = False
        reference_size = None

        if not native and semantic_marker and re.fullmatch(r"[\[\(]?[A-Za-z0-9*\u2020\u2021]{1,4}[\]\)]?", compact):
            lexical_neighbors = []
            for neighbor_idx in reversed(non_empty_indices):
                if neighbor_idx >= idx:
                    continue
                neighbor = span_entries[neighbor_idx]
                if _pdf_span_has_lexical_text(neighbor):
                    lexical_neighbors.append(neighbor)
                    break
            for neighbor_idx in non_empty_indices:
                if neighbor_idx <= idx:
                    continue
                neighbor = span_entries[neighbor_idx]
                if _pdf_span_has_lexical_text(neighbor):
                    lexical_neighbors.append(neighbor)
                    break

            size = max(float(span.get("size", 0.0)), 0.01)
            for neighbor in lexical_neighbors:
                neighbor_size = max(float(neighbor.get("size", 0.0)), 0.01)
                scale = size / neighbor_size
                raised_by = _pdf_span_origin_y(neighbor) - _pdf_span_origin_y(span)
                if neighbor.get("x1", 0.0) <= span.get("x0", 0.0):
                    horizontal_gap = float(span.get("x0", 0.0)) - float(neighbor.get("x1", 0.0))
                elif span.get("x1", 0.0) <= neighbor.get("x0", 0.0):
                    horizontal_gap = float(neighbor.get("x0", 0.0)) - float(span.get("x1", 0.0))
                else:
                    horizontal_gap = 0.0

                if (
                    0.45 <= scale <= 0.80
                    and raised_by >= neighbor_size * 0.15
                    and horizontal_gap <= max(neighbor_size * 1.2, 8.0)
                ):
                    inferred = True
                    reference_size = neighbor_size
                    break

        if native:
            normal_sizes = [
                float(other.get("size", 0.0))
                for other in span_entries
                if other is not span
                and _pdf_span_has_lexical_text(other)
                and float(other.get("size", 0.0)) > float(span.get("size", 0.0)) * 1.15
            ]
            reference_size = max(normal_sizes, default=None)

        if native or inferred:
            size = max(float(span.get("size", 0.0)), 0.01)
            reference_size = max(float(reference_size or size / 0.60), 0.01)
            span["superscript"] = True
            span["superscript_source"] = "native" if native else "inferred"
            span["superscript_scale"] = max(0.45, min(size / reference_size, 0.80))

        rich = text
        if span.get("bold"):
            rich = f"<b>{rich}</b>"
        if span.get("superscript"):
            rich = f"<sup>{rich}</sup>"
        span["rich"] = rich

    return span_entries


def _looks_like_pdf_list_marker(text: str) -> bool:
    marker = re.sub(r'\s+', ' ', text or '').strip()
    if not marker or len(marker) > 12:
        return False
    patterns = [
        r'^[\u2022\u25AA\u25CF\u25E6\u2043\u2219\uF0B7\-\*\u00B7]+$',
        r'^\(?\d{1,3}[.)]\)?$',
        r'^\(?[A-Za-z][.)]\)?$',
        r'^\(?[ivxlcdmIVXLCDM]{1,8}[.)]\)?$',
        r'^[\u4e00-\u9fff]{1,3}[、.)]$',
    ]
    return any(re.fullmatch(pattern, marker) for pattern in patterns)


def _pdf_source_line_leading_superscript_footnote_marker(
    line: dict,
) -> str | None:
    """Return a line-leading superscript marker attached to footnote prose.

    A normal line beginning ``b Madison...`` is not enough evidence, while
    ``<sup>b</sup> Madison...`` is an explicit footnote-definition lead.
    Requiring a non-empty body also excludes detached marker-only objects.
    """
    if not isinstance(line, dict):
        return None
    rich = (line.get("rich") or "").strip()
    match = re.match(
        r"(?is)^<sup(?:\s[^>]*)?>(?P<marker>.*?)</sup>\s+(?P<body>\S.*)$",
        rich,
    )
    if not match:
        return None
    marker = re.sub(
        r"\s+", "", _plain_text(html.unescape(match.group("marker")))
    )
    body = re.sub(
        r"\s+", " ", _plain_text(html.unescape(match.group("body")))
    ).strip()
    if (
        not _PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE.fullmatch(marker)
        or len(body) < 4
    ):
        return None
    plain = re.sub(
        r"\s+", " ", _plain_text(line.get("plain", ""))
    ).strip()
    if not re.match(rf"^{re.escape(marker)}\s+", plain):
        return None
    return marker


def _pdf_element_leading_superscript_footnote_marker(
    elem: dict,
) -> str | None:
    for paragraph in elem.get("paragraphs") or []:
        source_lines = [
            line for line in (paragraph.get("source_lines") or [])
            if (line.get("plain") or "").strip()
        ]
        if source_lines:
            return _pdf_source_line_leading_superscript_footnote_marker(
                source_lines[0]
            )
    return None


def _select_pdf_inline_math_fragments_for_text(
    fragments: list[dict],
    plain: str,
) -> list[dict]:
    """Rebind parent-block inline math records to one split paragraph.

    Filtering only by symbol membership duplicates records when adjacent
    paragraphs both contain symbols such as epsilon.  Match occurrences in
    the paragraph's source order and consume at most the parent-record count
    for each exact fragment text.
    """
    queues: dict[str, list[dict]] = {}
    for fragment in fragments or []:
        token = (fragment.get("text") or "").strip()
        if token:
            queues.setdefault(token, []).append(dict(fragment))
    occurrences = []
    for token, records in queues.items():
        positions = [match.start() for match in re.finditer(re.escape(token), plain)]
        for position, record in zip(positions, records):
            occurrences.append((position, -len(token), record))
    occurrences.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in occurrences]


def _looks_like_compact_pdf_identifier(text: str) -> bool:
    """Detect compact mixed-case market/security codes such as ``TEL2b``."""
    plain = re.sub(r"\s+", "", _plain_text(text or "")).strip()
    if not 3 <= len(plain) <= 16 or not re.fullmatch(r"[A-Za-z0-9]+", plain):
        return False
    letters = [char for char in plain if char.isalpha()]
    return bool(
        any(char.isdigit() for char in plain)
        and sum(char.isupper() for char in letters) >= 2
        and sum(char.islower() for char in letters) <= 2
        and plain[:1].isupper()
    )


def _normalize_pdf_translation(text: str) -> str:
    """Keep only lightweight inline markup that we deliberately support in PDFs."""
    normalized = (text or "").strip()
    replacements = [
        (r'(?is)&lt;\s*b\b.*?&gt;', '<b>'),
        (r'(?is)&lt;\s*strong\b.*?&gt;', '<b>'),
        (r'(?is)&lt;\s*sup\b.*?&gt;', '<sup>'),
        (r'(?i)&lt;\s*b\s*&gt;', '<b>'),
        (r'(?i)&lt;\s*/\s*b\s*&gt;', '</b>'),
        (r'(?i)&lt;\s*strong\s*&gt;', '<b>'),
        (r'(?i)&lt;\s*/\s*strong\s*&gt;', '</b>'),
        (r'(?i)&lt;\s*sup\s*&gt;', '<sup>'),
        (r'(?i)&lt;\s*/\s*sup\s*&gt;', '</sup>'),
        (r'(?i)&lt;\s*br\s*/?\s*&gt;', '\n'),
        (r'(?is)<\s*b\b[^>]*>', '<b>'),
        (r'(?is)<\s*strong\b[^>]*>', '<b>'),
        (r'(?is)<\s*sup\b[^>]*>', '<sup>'),
        (r'(?i)<\s*strong\s*>', '<b>'),
        (r'(?i)<\s*/\s*strong\s*>', '</b>'),
        (r'(?i)<\s*b\s*>', '<b>'),
        (r'(?i)<\s*/\s*b\s*>', '</b>'),
        (r'(?i)<\s*sup\s*>', '<sup>'),
        (r'(?i)<\s*/\s*sup\s*>', '</sup>'),
        (r'(?i)<\s*br\s*/?\s*>', '\n'),
    ]
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized)

    normalized = re.sub(r'(?is)</?(?!b\b|sup\b|br\b)[a-z][^>]*>', '', normalized)
    normalized = re.sub(r'(?is)</?br\b[^>]*>', '\n', normalized)
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t]+\n', '\n', normalized)
    normalized = re.sub(r'\n[ \t]+', '\n', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = re.sub(r'[ \t]{2,}', ' ', normalized)
    return normalized


def _pdf_superscript_signature(text: str) -> tuple[str, ...]:
    normalized = _normalize_pdf_translation(text)
    return tuple(
        html.unescape(_plain_text(match.group(1))).strip()
        for match in re.finditer(r'(?is)<sup>(.*?)</sup>', normalized)
        if html.unescape(_plain_text(match.group(1))).strip()
    )


def _find_pdf_fragment_outside_control_tokens(
    text: str,
    fragment: str,
    cursor: int = 0,
) -> int:
    """Find a source fragment without matching markup or opaque tokens.

    Superscripts are protected before semantic inline formula runs. A
    one-letter formula fragment such as ``E`` must therefore never match the
    ``E`` inside ``PHOENA_SUP_...``. Recompute the small blocked-range list
    after each replacement so already-created inline-math tokens are opaque.
    """
    if not text or not fragment:
        return -1
    blocked = sorted(
        [
            match.span()
            for pattern in (
                PDF_BATCH_SEGMENT_RE,
                PDF_SUPERSCRIPT_PLACEHOLDER_RE,
                PDF_INLINE_MATH_PLACEHOLDER_RE,
                PDF_IDENTIFIER_PLACEHOLDER_RE,
                re.compile(r"<[^>]*>"),
            )
            for match in pattern.finditer(text)
        ]
    )
    position = text.find(fragment, max(0, int(cursor)))
    while position >= 0:
        end = position + len(fragment)
        collision = next(
            (
                (blocked_start, blocked_end)
                for blocked_start, blocked_end in blocked
                if position < blocked_end and end > blocked_start
            ),
            None,
        )
        if collision is None:
            return position
        position = text.find(fragment, max(position + 1, collision[1]))
    return -1


def _protect_pdf_superscript_fragments(text: str) -> tuple[str, list[dict]]:
    """Replace complete ``<sup>`` fragments with stable opaque tokens.

    Asking the model to copy inline tags is not sufficient for long prose: it
    commonly drops ordinal suffixes such as ``<sup>th</sup>`` while correctly
    translating the surrounding date.  Protect the complete source fragment
    before the API call so footnote numbers and ordinal superscripts can be
    restored byte-for-byte afterward.
    """
    normalized = _normalize_pdf_translation(text)
    matches = list(re.finditer(r"(?is)<sup>.*?</sup>", normalized))
    if not matches:
        return normalized, []

    records = []
    pieces = []
    cursor = 0
    for index, match in enumerate(matches):
        original = match.group(0)
        digest = hashlib.sha256(
            f"{index}:{match.start()}:{match.end()}:".encode("ascii")
            + original.encode("utf-8")
        ).hexdigest().upper()[:16]
        token = f"PHOENA_SUP_{index:04d}_{digest}"
        while token in normalized:
            digest = hashlib.sha256(
                (digest + original).encode("utf-8")
            ).hexdigest().upper()[:16]
            token = f"PHOENA_SUP_{index:04d}_{digest}"
        pieces.append(normalized[cursor:match.start()])
        pieces.append(token)
        source_prefix = normalized[:match.start()]
        source_suffix = normalized[match.end():]
        if not source_prefix.strip():
            anchor = "prefix"
            separator_match = re.match(r"\s*", source_suffix)
            anchor_separator = separator_match.group(0) if separator_match else ""
        elif not source_suffix.strip():
            anchor = "suffix"
            separator_match = re.search(r"\s*$", source_prefix)
            anchor_separator = separator_match.group(0) if separator_match else ""
        else:
            anchor = "inline"
            anchor_separator = ""
        records.append({
            "token": token,
            "original": original,
            "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "anchor": anchor,
            "anchor_separator": anchor_separator,
        })
        cursor = match.end()
    pieces.append(normalized[cursor:])
    return "".join(pieces), records


def _restore_pdf_superscript_fragments(
    translated_text: str,
    records: list[dict],
) -> str | None:
    """Restore protected PDF superscripts only after exact token validation."""
    if not records:
        return translated_text
    expected_tokens = [record["token"] for record in records]
    observed_tokens = PDF_SUPERSCRIPT_PLACEHOLDER_RE.findall(
        translated_text or ""
    )
    if observed_tokens != expected_tokens:
        return None
    restored = translated_text or ""

    # Models frequently move a leading footnote marker to the end of a
    # Chinese sentence even when they preserve its opaque token exactly.
    # Token-count validation alone cannot detect that layout corruption.
    # Re-anchor only source-edge superscripts; ordinary inline/ordinal markers
    # retain the model's surrounding word order.
    for record in records:
        token = record["token"]
        anchor = record.get("anchor")
        separator = record.get("anchor_separator", "")
        if anchor == "prefix" and not restored.lstrip().startswith(token):
            without = restored.replace(token, "", 1).lstrip()
            restored = token + separator + without
        elif anchor == "suffix" and not restored.rstrip().endswith(token):
            without = restored.replace(token, "", 1).rstrip()
            restored = without + separator + token

    for record in records:
        original = record.get("original", "")
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != record.get("sha256"):
            return None
        token = record["token"]
        if restored.count(token) != 1:
            return None
        restored = restored.replace(token, original, 1)
    if PDF_SUPERSCRIPT_PLACEHOLDER_RE.search(restored):
        return None
    return _normalize_pdf_translation(restored)


def _protect_pdf_inline_math_fragments(
    text: str,
    fragments: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Replace simple inline formula glyph runs with opaque API tokens.

    This path is deliberately limited to prose-dominant lines that extraction
    has classified as semantic inline math.  Standalone and complex formulae
    remain immutable source PDF objects.
    """
    protected = text or ""
    records = []
    cursor = 0
    for index, fragment in enumerate(fragments or []):
        original = fragment.get("text", "") if isinstance(fragment, dict) else ""
        if not original or not original.strip():
            continue
        start = _find_pdf_fragment_outside_control_tokens(
            protected,
            original,
            cursor,
        )
        if start < 0:
            # Rich markup may place a tag inside what was one native span.
            # Fail closed for that fragment and leave the source untouched;
            # the post-translation element validator will reject any change.
            continue
        end = start + len(original)
        digest = hashlib.sha256(
            f"{index}:{start}:{end}:".encode("ascii")
            + original.encode("utf-8")
        ).hexdigest().upper()[:16]
        token = f"PHOENA_IMATH_{index:04d}_{digest}"
        while token in protected:
            digest = hashlib.sha256(
                (digest + original).encode("utf-8")
            ).hexdigest().upper()[:16]
            token = f"PHOENA_IMATH_{index:04d}_{digest}"
        protected = protected[:start] + token + protected[end:]
        records.append({
            "token": token,
            "original": original,
            "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        })
        cursor = start + len(token)
    return protected, records


def _restore_pdf_inline_math_fragments(
    translated_text: str,
    records: list[dict],
) -> str | None:
    """Restore protected inline-math fragments after translation.

    Tolerant recovery: language models frequently mangle an opaque PHOENA_IMATH
    token (added spacing, changed case, a dropped tail hex char, an absorbed
    adjacent CJK glyph).  Instead of discarding the whole element when a token is
    imperfect, restore each fragment at its own position by matching that
    record's unique index with a fuzzy tail.  A prose paragraph carrying one or
    two inline-math fragments therefore still translates, with the math dropped
    back into place.  Returns None only when a fragment is genuinely missing or
    ambiguous, in which case the caller keeps just that element as source.
    """
    if not records:
        return translated_text
    restored = translated_text or ""
    for record in records:
        original = record.get("original", "")
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != record.get("sha256"):
            return None
        token = record["token"]
        if restored.count(token) == 1:
            restored = restored.replace(token, original, 1)
            continue
        if restored.count(token) > 1:
            return None
        index_match = re.search(r"IMATH_(\d{4})_", token)
        if not index_match:
            return None
        idx = index_match.group(1)
        fuzzy = re.compile(
            r"P[\s_]*H[\s_]*O[\s_]*E[\s_]*N[\s_]*A[\s_]*I?[\s_]*"
            r"M[\s_]*A[\s_]*T[\s_]*H[\s_]*" + idx + r"[\s_]*[0-9A-Fa-f]{0,16}",
            re.IGNORECASE,
        )
        matches = list(fuzzy.finditer(restored))
        if len(matches) != 1:
            return None
        span = matches[0]
        restored = restored[:span.start()] + original + restored[span.end():]
    if re.search(
        r"P[\s_]*H[\s_]*O[\s_]*E[\s_]*N[\s_]*A[\s_]*I?[\s_]*M[\s_]*A[\s_]*T[\s_]*H",
        restored,
        re.IGNORECASE,
    ):
        return None
    return restored


def _protect_pdf_identifier_fragments(text: str) -> tuple[str, list[dict]]:
    """Replace URLs, e-mail addresses and DOI strings with opaque tokens.

    Prompting a language model to copy a long identifier is not a reliable
    preservation mechanism: punctuation, case, or a single path character can
    be silently changed.  Protect the exact source bytes before translation
    and restore them only after the complete token sequence round-trips.
    """
    protected = text or ""
    matches = list(_PRESERVED_IDENTIFIER_RE.finditer(protected))
    if not matches:
        return protected, []

    records = []
    pieces = []
    cursor = 0
    for index, match in enumerate(matches):
        original = match.group(0)
        digest = hashlib.sha256(
            f"{index}:{match.start()}:{match.end()}:".encode("ascii")
            + original.encode("utf-8")
        ).hexdigest().upper()[:16]
        token = f"PHOENA_ID_{index:04d}_{digest}"
        while token in protected:
            digest = hashlib.sha256(
                (digest + original).encode("utf-8")
            ).hexdigest().upper()[:16]
            token = f"PHOENA_ID_{index:04d}_{digest}"
        pieces.append(protected[cursor:match.start()])
        pieces.append(token)
        source_prefix = protected[:match.start()]
        source_suffix = protected[match.end():]
        if not source_prefix.strip():
            anchor = "prefix"
            separator_match = re.match(r"\s*", source_suffix)
            anchor_separator = separator_match.group(0) if separator_match else ""
        elif not source_suffix.strip():
            anchor = "suffix"
            separator_match = re.search(r"\s*$", source_prefix)
            anchor_separator = separator_match.group(0) if separator_match else ""
        else:
            anchor = "inline"
            anchor_separator = ""
        records.append({
            "token": token,
            "original": original,
            "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "anchor": anchor,
            "anchor_separator": anchor_separator,
        })
        cursor = match.end()
    pieces.append(protected[cursor:])
    return "".join(pieces), records


def _restore_pdf_identifier_fragments(
    translated_text: str,
    records: list[dict],
) -> str | None:
    """Restore protected identifiers after exact count/order validation."""
    if not records:
        return translated_text
    expected_tokens = [record["token"] for record in records]
    observed_tokens = PDF_IDENTIFIER_PLACEHOLDER_RE.findall(
        translated_text or ""
    )
    if observed_tokens != expected_tokens:
        return None
    restored = translated_text or ""

    # Keep source-edge identifiers at the same edge.  This matters for URL
    # footnotes and bibliography entries, where moving the token changes the
    # semantic association even though its bytes remain intact.
    for record in records:
        token = record["token"]
        anchor = record.get("anchor")
        separator = record.get("anchor_separator", "")
        if anchor == "prefix" and not restored.lstrip().startswith(token):
            without = restored.replace(token, "", 1).lstrip()
            restored = token + separator + without
        elif anchor == "suffix" and not restored.rstrip().endswith(token):
            without = restored.replace(token, "", 1).rstrip()
            restored = without + separator + token

    for record in records:
        original = record.get("original", "")
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != record.get("sha256"):
            return None
        token = record["token"]
        if restored.count(token) != 1:
            return None
        restored = restored.replace(token, original, 1)
    if PDF_IDENTIFIER_PLACEHOLDER_RE.search(restored):
        return None
    return restored


def _pdf_inline_math_fragments_preserved(
    elem: dict,
    translated_text: str,
) -> bool:
    expected = [
        fragment.get("text", "")
        for fragment in (elem.get("inline_math_fragments") or [])
        if isinstance(fragment, dict) and fragment.get("text", "")
    ]
    if not expected:
        return True
    plain = _plain_text(_normalize_pdf_translation(translated_text))
    cursor = 0
    for fragment in expected:
        position = plain.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment)
    return True


def _pdf_inline_markup_preserved(source_text: str, translated_text: str) -> bool:
    source_signature = _pdf_superscript_signature(source_text)
    if not source_signature:
        return True
    return source_signature == _pdf_superscript_signature(translated_text)


def _pdf_marker_pattern(marker: str) -> re.Pattern:
    escaped = re.escape(marker)
    if marker and marker[0].isalnum() and marker[-1].isalnum():
        return re.compile(rf'(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])')
    return re.compile(escaped)


def _restore_pdf_superscript_markup(source_text: str, translated_text: str) -> str:
    """Restore dropped <sup> tags by the marker's source occurrence ordinal."""
    source = _normalize_pdf_translation(source_text)
    translated = _normalize_pdf_translation(translated_text)
    if _pdf_inline_markup_preserved(source, translated):
        return translated

    descriptors = []
    for match in re.finditer(r'(?is)<sup>(.*?)</sup>', source):
        marker = html.unescape(_plain_text(match.group(1))).strip()
        if not marker:
            continue
        source_prefix = _plain_text(source[:match.start()])
        ordinal = len(list(_pdf_marker_pattern(marker).finditer(source_prefix))) + 1
        descriptors.append((marker, ordinal))
    if not descriptors:
        return translated

    flat = re.sub(r'(?is)</?sup>', '', translated)
    selected_ranges = []
    for marker, ordinal in descriptors:
        matches = list(_pdf_marker_pattern(marker).finditer(flat))
        if ordinal <= 0 or ordinal > len(matches):
            return translated
        selected = matches[ordinal - 1]
        item = (selected.start(), selected.end())
        if item in selected_ranges:
            return translated
        selected_ranges.append(item)

    restored = flat
    for start, end in sorted(selected_ranges, reverse=True):
        restored = restored[:start] + "<sup>" + restored[start:end] + "</sup>" + restored[end:]
    return _normalize_pdf_translation(restored)
