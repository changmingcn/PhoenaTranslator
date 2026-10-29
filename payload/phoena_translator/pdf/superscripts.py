"""Detect footnote and ordinal superscripts and footnote-definition markers."""

from __future__ import annotations

import fitz
import re
import html
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from collections import Counter


_PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE = re.compile(
    r"^(?:[A-Za-z]|\d{1,3}|[*\u2020\u2021\u00a7\u00b6]{1,3})$"
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
    """Find the adjacent inline span that geometrically anchors a marker.

    MuPDF's HTML renderer may expose a visually inline ``sup`` as a separate
    line object. Coordinate proximity is therefore more reliable than line
    membership when validating the rendered marker. Some source PDFs retain
    the base font size for a raised glyph, so scale is supporting evidence,
    while local horizontal/vertical geometry is mandatory.
    """
    marker_size = max(float(marker_span.get("size", 0.0)), 0.01)
    marker_rect = fitz.Rect(marker_span.get("bbox") or (
        marker_span.get("x0", 0.0),
        marker_span.get("y0", 0.0),
        marker_span.get("x1", 0.0),
        marker_span.get("y1", 0.0),
    ))
    candidates = []

    for reference in reference_spans:
        if reference is marker_span or not _pdf_span_has_lexical_text(reference):
            continue
        reference_size = max(float(reference.get("size", 0.0)), 0.01)
        scale = marker_size / reference_size
        if scale < 0.45 - 1e-6 or scale > 1.10 + 1e-6:
            continue

        raised_by = _pdf_span_origin_y(reference) - _pdf_span_origin_y(marker_span)
        if (
            raised_by < reference_size * 0.12 - 1e-6
            or raised_by > reference_size * 0.80 + 1e-6
        ):
            continue

        reference_rect = fitz.Rect(reference.get("bbox") or (
            reference.get("x0", 0.0),
            reference.get("y0", 0.0),
            reference.get("x1", 0.0),
            reference.get("y1", 0.0),
        ))
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
    for span in span_entries:
        span["superscript"] = False
        span["superscript_source"] = None
        span["superscript_scale"] = None

        text = span.get("text", "")
        compact = re.sub(r"\s+", "", text)
        semantic_marker = _pdf_is_semantic_superscript_marker(compact)
        native_flagged = (
            bool(int(span.get("flags", 0)) & native_flag) and semantic_marker
        )
        native = False
        inferred = False
        reference_size = None

        short_marker = bool(re.fullmatch(
            r"[\[\(]?[A-Za-z0-9*\u2020\u2021]{1,4}[\]\)]?",
            compact,
        ))
        local_reference = _pdf_nearby_superscript_reference(
            span,
            span_entries,
        )

        if native_flagged and local_reference is not None:
            native = True
            reference_size = max(
                float(local_reference.get("size", 0.0)),
                0.01,
            )
        elif native_flagged and short_marker:
            # A genuine footnote/exponent may be emitted as a marker-only line.
            # Preserve that explicit native signal only when there is no same-
            # line lexical peer that should have supplied geometric proof. This
            # prevents a corrupt flag on ordinary multi-span prose from winning.
            lexical_peers = [
                other
                for other in span_entries
                if other is not span and _pdf_span_has_lexical_text(other)
            ]
            if not lexical_peers:
                native = True
                reference_size = max(
                    float(span.get("size", 0.0)) / 0.60,
                    0.01,
                )

        if (
            not native
            and semantic_marker
            and short_marker
            and local_reference is not None
        ):
            size = max(float(span.get("size", 0.0)), 0.01)
            neighbor_size = max(
                float(local_reference.get("size", 0.0)),
                0.01,
            )
            if 0.45 <= size / neighbor_size <= 0.80:
                inferred = True
                reference_size = neighbor_size

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
    body = _collapsed_plain_text(html.unescape(match.group("body")))
    if (
        not _PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE.fullmatch(marker)
        or len(body) < 4
    ):
        return None
    plain = _collapsed_plain_text(line.get("plain", ""))
    if not re.match(rf"^{re.escape(marker)}\s+", plain):
        return None
    return marker


def _pdf_source_line_footnote_definition_marker(
    line: dict,
) -> str | None:
    """Return an explicit footnote-definition marker for one source line."""
    superscript_marker = _pdf_source_line_leading_superscript_footnote_marker(line)
    if superscript_marker:
        return superscript_marker
    plain = _collapsed_plain_text((line or {}).get("plain", ""))
    match = re.match(
        r"^(?P<marker>[*\u2020\u2021\u00a7\u00b6]{1,3})\s+(?P<body>\S.*)$",
        plain,
    )
    if match and len(match.group("body")) >= 4:
        return match.group("marker")
    return None


def _pdf_element_footnote_definition_marker(elem: dict) -> str | None:
    for paragraph in elem.get("paragraphs") or []:
        source_lines = [
            line
            for line in (paragraph.get("source_lines") or [])
            if (line.get("plain") or "").strip()
        ]
        if source_lines:
            return _pdf_source_line_footnote_definition_marker(source_lines[0])
    return None
