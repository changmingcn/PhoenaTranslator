"""Detect mathematics in native PDF lines (fonts, symbols, identifiers) and
build formula elements from them."""

from __future__ import annotations

import html
import re

import fitz

from phoena_translator.math_text import (
    MATH_FUNCTION_WORDS,
)
from phoena_translator.math_text import (
    is_math_block as _is_math_block,
)
from phoena_translator.math_text import (
    is_unicode_math_symbol as _is_unicode_math_symbol,
)
from phoena_translator.math_text import (
    iter_unicode_math_symbol_ranges as _iter_unicode_math_symbol_ranges,
)
from phoena_translator.math_text import (
    unicode_math_symbol_count as _unicode_math_symbol_count,
)
from phoena_translator.pdf.model import ElementType
from phoena_translator.pdf.text_shape import (
    _looks_like_pdf_list_marker,
    _plain_text,
)

PDF_STRONG_MATH_FONT_TOKENS = (
    "cmex", "cmmi", "cmsy", "msam", "msbm", "eufm", "eurm",
    "math", "symbol", "euclid", "mt-extra", "mtextra", "mathematicalpi",
)
PDF_MATH_ITALIC_FONT_TOKENS = (
    "cmssi", "mathitalic", "mtmi", "texmathitalic",
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
    # A bare DOI such as ``doi=10.1257/jep.27.2.51`` carries no scheme, no TLD
    # and no script extension, so none of the shapes above fire -- yet its
    # single ``=`` was enough to promote the whole line into protected formula
    # geometry.  The raster expectation that created then failed on sub-pixel
    # font-subsetting drift (42 of 2907 pixels, visually identical), and the
    # accepted-merge closure reverted seven translated pages to source.
    # ``_PRESERVED_IDENTIFIER_RE`` already carries DOIs through translation
    # byte-for-byte, so they need no formula protection.
    if re.search(r"(?i)(?:\bdoi\s*[:=]\s*)?(?<!\d)10\.\d{4,9}/\S+", plain):
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
        "type": ElementType.FORMULA_IMAGE,
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
