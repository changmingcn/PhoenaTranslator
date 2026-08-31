"""Targets for deterministic PDF processing."""

from __future__ import annotations

import logging
import re
from collections import Counter

import fitz
from phoena_translator.math_text import (
    is_math_block as _is_math_block,
)

from phoena_translator.pdf.types import (
    PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE,
    PDF_FIXED_SHORT_LABEL_TRANSLATIONS,
    _PDF_DISPLAY_IDENTITY_END_WORDS,
    _PDF_PERSON_HONORIFICS,
    _PDF_PROPER_NAME_CONNECTORS,
    _PDF_PROPER_NAME_SUFFIXES,
    _PDF_SHORT_LABEL_CONNECTORS,
    _PDF_TRANSLATABLE_SHORT_LABELS,
    _PRESERVED_IDENTIFIER_RE,
    _PRESERVED_STRUCTURED_CODE_RE,
    _SHORT_CITATION_CONNECTORS,
    _SHORT_CITATION_WORD_RE,
    _SHORT_CITATION_YEAR_RE,
)
from phoena_translator.pdf.geometry import (
    _cluster_pdf_line_spans,
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
    _get_pdf_source_ink_rect,
    _line_overlaps_pdf_table_rects,
    _pdf_rotation_from_direction,
    _split_pdf_line_layout_cells,
)
from phoena_translator.pdf.math_detection import (
    _demote_pdf_wrapped_identifier_math_lines,
    _looks_like_compact_pdf_identifier,
    _make_pdf_formula_element_from_lines,
    _mark_pdf_superscript_spans,
    _pdf_formula_region_signature,
    _pdf_line_math_evidence,
    _plain_text,
    _propagate_pdf_math_line_context,
    _restore_pdf_superscript_markup,
)
from phoena_translator.pdf.extraction import (
    _extract_pdf_vector_ocr_elements,
    _find_pdf_table_regions,
    _mark_pdf_embedded_thumbnail_text_elements,
    _sanitize_pdf_hidden_text_artifact_spans,
)
from phoena_translator.pdf.semantics import (
    _bucket_pdf_fontsize,
    _complete_pdf_citation_separator_inline_math_fragments,
    _detect_pdf_drop_cap_span,
    _ends_with_sentence_boundary,
    _is_disclaimer_block,
    _looks_like_heading_text,
    _looks_like_split_layout_line,
    _make_pdf_text_element_from_lines,
    _mark_pdf_reference_entry_elements,
    _merge_adjacent_heading_elements,
    _merge_pdf_detached_list_marker_elements,
    _merge_pdf_list_marker_clusters,
    _merge_pdf_reference_entry_elements,
    _merge_pdf_semantic_continuation_elements,
    _merge_pdf_semantic_table_cells,
    _merge_pdf_strong_continuation_elements,
    _merge_pdf_wrapped_inline_math_continuation_fragments,
    _merge_pdf_wrapped_line_elements,
    _normalize_pdf_drop_cap_lines,
    _normalize_pdf_same_baseline_prose_fragments,
    _normalize_pdf_footnote_alignment,
    _normalize_pdf_glossary_cells,
    _pdf_element_prefers_single_line_heading,
    _pdf_internal_paragraph_lead_residuals,
    _pdf_reference_continuation_residual_pairs,
    _pdf_reference_entry_structure_residuals,
    _pdf_strong_continuation_residual_pairs,
    _pick_pdf_dominant_value,
    _promote_pdf_formula_overlaps,
    _split_pdf_disjoint_line_segments,
    _split_pdf_formula_adjacent_paragraph_elements,
    _split_pdf_reference_entry_elements,
    _split_pdf_text_elements_into_semantic_fragments,
    _translate_pdf_citation_labels,
)

log = logging.getLogger("translator")

def _mark_pdf_glossary_term_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> set[int]:
    """Mark lower-case term cells paired with explanatory definitions.

    Glossaries are often drawn as an open two-column table, so PyMuPDF does
    not set ``table_hint``.  The generic tiny-table-label exemption then used
    to preserve entries such as ``pinging``, ``profiling`` and ``spread`` in
    English even though the adjacent definition was translated.  Require both
    an explicit term/meaning header and at least four consistently aligned
    term/definition row pairs.  The header gate keeps regression-variable
    tables and ordinary two-column data tables out of this semantic class;
    the alignment gate excludes chart atoms, page markers and isolated table
    labels.  Proper names and all-uppercase identifiers remain immutable.
    """
    text_indices = [
        index
        for index, element in enumerate(elements or [])
        if element.get("type") == "text"
    ]
    if len(text_indices) < 8:
        return set()

    normalized_by_index = {
        index: re.sub(
            r"\s+",
            " ",
            _plain_text(elements[index].get("content", "")),
        ).strip().casefold()
        for index in text_indices
    }
    normalized_cells = set(normalized_by_index.values())
    has_term_header = bool(
        normalized_cells
        & {"term", "key term", "key terms", "glossary term", "glossary terms"}
    )
    has_meaning_header = any(
        cell in {"meaning", "definition"}
        or cell.startswith("meaning in ")
        for cell in normalized_cells
    )
    if not (has_term_header and has_meaning_header):
        return set()
    header_indices = [
        index
        for index, cell in normalized_by_index.items()
        if cell in {
            "term", "key term", "key terms", "glossary term", "glossary terms",
            "meaning", "definition",
        }
        or cell.startswith("meaning in ")
    ]
    glossary_header_bottom = max(
        (_get_pdf_elem_rect(elements[index]).y1 for index in header_indices),
        default=page_rect.y0 + page_rect.height * 0.12,
    )

    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    row_pairs = []
    for left_index in text_indices:
        left = elements[left_index]
        if left.get("layout_class") != "table":
            continue
        left_plain = re.sub(
            r"\s+", " ", _plain_text(left.get("content", ""))
        ).strip()
        if (
            not left_plain
            or len(left_plain) > 80
            or not left_plain[:1].islower()
            or not _looks_like_translatable_english(left_plain)
            or _looks_like_pdf_proper_name_label(left_plain)
        ):
            continue

        left_rect = _get_pdf_elem_rect(left)
        left_center_y = (left_rect.y0 + left_rect.y1) / 2.0
        if (
            left_rect.width > page_width * 0.38
            or left_center_y < page_rect.y0 + page_height * 0.16
            or left_center_y > page_rect.y0 + page_height * 0.90
        ):
            continue

        best = None
        for right_index in text_indices:
            if right_index == left_index:
                continue
            right = elements[right_index]
            right_plain = re.sub(
                r"\s+", " ", _plain_text(right.get("content", ""))
            ).strip()
            if (
                len(right_plain) < 24
                or not _looks_like_translatable_english(right_plain)
                or _is_disclaimer_block(right_plain)
            ):
                continue
            right_rect = _get_pdf_elem_rect(right)
            horizontal_gap = right_rect.x0 - left_rect.x1
            if horizontal_gap < max(12.0, page_width * 0.03):
                continue
            vertical_overlap = min(left_rect.y1, right_rect.y1) - max(
                left_rect.y0, right_rect.y0
            )
            row_tolerance = max(
                5.0,
                float(left.get("fontsize", 0.0)) * 0.85,
            )
            if (
                vertical_overlap <= 0
                and abs(right_rect.y0 - left_rect.y0) > row_tolerance
            ):
                continue
            candidate = (
                horizontal_gap,
                abs(right_rect.y0 - left_rect.y0),
                right_index,
                right_rect,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if best is not None:
            row_pairs.append(
                (
                    left_index,
                    best[2],
                    float(left_rect.x0),
                    float(best[3].x0),
                )
            )

    valid_columns = []
    for left_index, _, left_x, right_x in row_pairs:
        aligned_count = sum(
            1
            for _, _, other_left_x, other_right_x in row_pairs
            if abs(other_left_x - left_x) <= 18.0
            and abs(other_right_x - right_x) <= 30.0
        )
        if aligned_count >= 4:
            valid_columns.append((left_x, right_x))

    if not valid_columns:
        return set()

    marked = set()
    for index in text_indices:
        element = elements[index]
        plain = re.sub(
            r"\s+", " ", _plain_text(element.get("content", ""))
        ).strip()
        if (
            not plain
            or not _looks_like_translatable_english(plain)
            or _is_disclaimer_block(plain)
        ):
            continue
        rect = _get_pdf_elem_rect(element)
        center_y = (rect.y0 + rect.y1) / 2.0
        if center_y <= glossary_header_bottom or center_y > page_rect.y0 + page_height * 0.90:
            continue

        aligned_left = any(
            abs(float(rect.x0) - left_x) <= 18.0
            for left_x, _ in valid_columns
        )
        aligned_right = any(
            abs(float(rect.x0) - right_x) <= 30.0
            for _, right_x in valid_columns
        )
        semantic_cross_reference = bool(
            re.match(r"(?i)^see\b", plain)
        )
        proper_name = _looks_like_pdf_proper_name_label(plain)

        # Include split left-cell continuations and terms whose definitions
        # are too short to qualify as a row-pair anchor (for example
        # ``trades`` and ``liquidity`` in ASIC Report 452).
        if (
            aligned_left
            and plain[:1].islower()
            and not proper_name
        ):
            element["glossary_term_hint"] = True
            marked.add(index)
            continue

        # Short semantic definitions such as ``See 'NBBO'`` need the same
        # override as short terms.  Preserve pure institution/product names.
        if aligned_right and (not proper_name or semantic_cross_reference):
            element["glossary_definition_hint"] = True
            marked.add(index)
    return marked


def _classify_pdf_page_text_elements(elements: list[dict], page_rect: fitz.Rect) -> dict[int, str]:
    _normalize_pdf_glossary_cells(elements, page_rect)
    text_indices = [
        idx for idx, elem in enumerate(elements)
        if elem.get("type") == "text"
    ]
    if not text_indices:
        return {}

    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    overall_fontsize = _pick_pdf_dominant_value(
        [
            (_bucket_pdf_fontsize(elements[idx].get("fontsize", 11.0)), max(len(_plain_text(elements[idx].get("content", "")).strip()), 1))
            for idx in text_indices
        ],
        11.0,
    )

    for idx in text_indices:
        _normalize_pdf_footnote_alignment(elements[idx], page_rect)

    short_mid_candidates = []
    for idx in text_indices:
        elem = elements[idx]
        plain = re.sub(r'\s+', ' ', _plain_text(elem.get("content", ""))).strip()
        if not plain or _is_disclaimer_block(plain) or elem.get("skip_translate_reason") == "watermark":
            continue

        rect = _get_pdf_elem_rect(elem)
        center_y = ((rect.y0 + rect.y1) / 2.0 - page_rect.y0) / page_height
        width_ratio = rect.width / page_width
        paragraphs = [p for p in (elem.get("paragraphs") or []) if (p.get("plain") or "").strip()]
        para_count = len(paragraphs) or 1
        aligns = {p.get("text_align", "left") for p in paragraphs} or {"left"}
        nowrap = any(bool(p.get("nowrap")) for p in paragraphs)

        if (
            0.20 <= center_y <= 0.88
            and len(plain) <= 140
            and para_count <= 2
            and (width_ratio <= 0.45 or aligns != {"left"} or nowrap)
        ):
            short_mid_candidates.append(idx)

    dense_table_page = len(short_mid_candidates) >= 8
    table_zone = None
    for idx in short_mid_candidates:
        rect = _get_pdf_elem_rect(elements[idx])
        table_zone = rect if table_zone is None else (table_zone | rect)

    classifications = {}
    for idx in text_indices:
        elem = elements[idx]
        plain = re.sub(r'\s+', ' ', _plain_text(elem.get("content", ""))).strip()
        rect = _get_pdf_elem_rect(elem)
        center_y = ((rect.y0 + rect.y1) / 2.0 - page_rect.y0) / page_height
        width_ratio = rect.width / page_width
        fontsize = float(elem.get("fontsize", overall_fontsize))
        paragraphs = [p for p in (elem.get("paragraphs") or []) if (p.get("plain") or "").strip()]
        para_count = len(paragraphs) or 1
        aligns = {p.get("text_align", "left") for p in paragraphs} or {"left"}
        nowrap = any(bool(p.get("nowrap")) for p in paragraphs)
        para_lengths = [len(re.sub(r'\s+', ' ', p.get("plain", "")).strip()) for p in paragraphs if (p.get("plain") or "").strip()]
        short_block = len(plain) <= 140 and para_count <= 2
        header_footer = center_y <= 0.20 or center_y >= 0.90
        short_stacked_block = (
            para_count >= 2
            and len(plain) <= 140
            and width_ratio <= 0.45
            and para_lengths
            and max(para_lengths) <= 36
        )
        right_aligned_metadata = (
            center_y <= 0.32
            and width_ratio <= 0.40
            and aligns == {"right"}
            and len(plain) <= 180
        )

        is_body = (
            (para_count >= 2 and not short_stacked_block and not right_aligned_metadata)
            or len(plain) >= 180
            or (len(plain) >= 120 and width_ratio >= 0.42)
            or (len(plain) >= 90 and width_ratio >= 0.55 and rect.height >= max(fontsize * 2.0, 18.0))
        )

        in_table_zone = False
        if table_zone is not None:
            expanded_zone = fitz.Rect(table_zone.x0 - 12.0, table_zone.y0 - 12.0, table_zone.x1 + 12.0, table_zone.y1 + 12.0)
            in_table_zone = not (rect & expanded_zone).is_empty

        is_table = (
            not header_footer
            and not is_body
            and (
                in_table_zone
                or (short_block and dense_table_page)
                or (short_block and (width_ratio <= 0.35 or aligns != {"left"} or nowrap))
                or short_stacked_block
                or (len(plain) <= 40 and rect.width <= page_width * 0.25 and center_y >= 0.22)
                or (len(plain) <= 80 and fontsize <= overall_fontsize and width_ratio <= 0.30 and center_y >= 0.22)
            )
        )

        if elem.get("skip_translate_reason") == "watermark" or _is_disclaimer_block(plain):
            layout_class = "scattered"
        elif elem.get("table_hint"):
            layout_class = "table"
        elif is_body and not (header_footer and len(plain) < 180):
            layout_class = "body"
        elif is_table:
            layout_class = "table"
        else:
            layout_class = "scattered"

        elem["layout_class"] = layout_class
        single_line_heading = _pdf_element_prefers_single_line_heading(elem, page_rect)
        elem["single_line_heading"] = single_line_heading
        if single_line_heading:
            for paragraph in elem.get("paragraphs") or []:
                paragraph["nowrap"] = True
        classifications[idx] = layout_class

    _mark_pdf_glossary_term_elements(elements, page_rect)
    return classifications


def _is_preserved_nonlinguistic_text(text: str) -> bool:
    """Return True for one or more identifiers with only footnote markers.

    PDF extraction can join multiple URL-only footnotes into one visual text
    element, for example ``<sup>2</sup> URL <sup>3</sup> URL``.  Requiring one
    identifier to occupy the entire element sends that block to the model even
    though every meaningful byte must be preserved.
    """
    plain = re.sub(r'\s+', ' ', _plain_text(text)).strip()
    if not plain:
        return False

    if _PRESERVED_STRUCTURED_CODE_RE.fullmatch(plain.rstrip(".,;")):
        return True

    # Some PDFs insert a layout space at a wrapped URL hyphen.  Rejoin only
    # that unambiguous continuation for identifier detection; source bytes are
    # still retained unchanged in the output.
    detection_plain = re.sub(r"(?<=-)\s+(?=[A-Za-z0-9])", "", plain)

    matches = list(_PRESERVED_IDENTIFIER_RE.finditer(detection_plain))
    if not matches:
        return False

    residue = _PRESERVED_IDENTIFIER_RE.sub(" ", detection_plain)
    return bool(re.fullmatch(r"[\s\d.,;:()\[\]{}<>#*†‡|/\\\-]*", residue))


def _pdf_translation_language_views(
    source_text: str,
    translated_text: str,
) -> tuple[str, str, bool]:
    """Return prose-only views plus exact protected-identifier integrity.

    URLs, e-mail addresses and DOI strings can dominate a short footnote's
    byte count even though every byte must remain untranslated.  Language
    completeness therefore measures only the residue after those identifiers
    are removed, while a separate Counter equality keeps them byte-exact and
    prevents additions, losses or substitutions.
    """
    source_plain = re.sub(
        r"\s+", " ", _plain_text(source_text or "")
    ).strip()
    translated_plain = re.sub(
        r"\s+", " ", _plain_text(translated_text or "")
    ).strip()
    source_identifiers = Counter(
        match.group(0)
        for match in _PRESERVED_IDENTIFIER_RE.finditer(source_plain)
    )
    translated_identifiers = Counter(
        match.group(0)
        for match in _PRESERVED_IDENTIFIER_RE.finditer(translated_plain)
    )
    source_language = re.sub(
        r"\s+", " ", _PRESERVED_IDENTIFIER_RE.sub(" ", source_plain)
    ).strip()
    translated_language = re.sub(
        r"\s+", " ", _PRESERVED_IDENTIFIER_RE.sub(" ", translated_plain)
    ).strip()
    return (
        source_language,
        translated_language,
        source_identifiers == translated_identifiers,
    )


def _is_preserved_short_citation(text: str) -> bool:
    """Recognize a narrow author-year citation that must stay unchanged.

    This deliberately does not recognize sentences containing a citation.  It
    is limited to short author/year footnotes such as ``29 Hasbrouck & Saar
    (2011).`` so genuine English prose continues to fail closed.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text)).strip()
    if not plain or len(plain) > 180:
        return False

    marker_match = re.match(
        r"^(?:\[\s*\d{1,4}\s*\]|\d{1,4}(?:[.)])?)\s+",
        plain,
    )
    if marker_match:
        plain = plain[marker_match.end():].strip()

    years = list(_SHORT_CITATION_YEAR_RE.finditer(plain))
    if len(years) != 1:
        return False
    year = years[0]
    author_text = plain[:year.start()].strip(" ,")
    suffix = plain[year.end():]
    if not author_text or not re.fullmatch(
        r"(?i)\s*(?:(?:[,;:]?\s*)?(?:p{1,2}\.\s*)?"
        r"\d+(?:\s*[-–—,]\s*\d+)*)?"
        r"(?:\s*[,;:]?\s*op\.\s*cit\.)?\s*\.?\s*",
        suffix,
    ):
        return False

    words = _SHORT_CITATION_WORD_RE.findall(author_text)
    if not words:
        return False
    punctuation = _SHORT_CITATION_WORD_RE.sub("", author_text)
    if not re.fullmatch(r"[\s,&.'’\-]*", punctuation):
        return False

    author_words = []
    for word in words:
        normalized = word.rstrip(".")
        if normalized.casefold() in _SHORT_CITATION_CONNECTORS:
            continue
        if not normalized or not normalized[0].isupper():
            return False
        author_words.append(normalized)

    if not 1 <= len(author_words) <= 8:
        return False
    if not marker_match and len(author_words) > 1:
        has_author_joiner = bool(
            re.search(r"(?:&|\band\b|\bet\s+al\.?)", author_text, re.IGNORECASE)
        )
        if not has_author_joiner:
            return False
    return True


def _looks_like_translatable_english(text: str) -> bool:
    """Return True if the block is English prose worth translating."""
    plain = re.sub(r'\s+', ' ', _plain_text(text)).strip()
    if not plain:
        return False
    if (
        _is_disclaimer_block(plain)
        or _is_math_block(plain)
        or _is_preserved_nonlinguistic_text(plain)
        or _is_preserved_short_citation(plain)
    ):
        return False
    if re.fullmatch(r'[\W\d_]+', plain):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", plain)
    if not words:
        return False

    meaningful = [w for w in words if len(w) > 1]
    if not meaningful:
        return False

    if len(meaningful) == 1 and meaningful[0].isupper() and len(meaningful[0]) <= 8 and len(plain) <= 12:
        return False

    return True


def _looks_like_tiny_pdf_label(text: str) -> bool:
    """Detect chart/table labels where translation cost outweighs value."""
    plain = re.sub(r'\s+', ' ', _plain_text(text)).strip()
    if not plain:
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", plain)
    if not words:
        return False
    if len(words) <= 1 and len(plain) <= 12:
        return True
    if len(words) <= 2 and len(plain) <= 18:
        return True
    return False


_PDF_STRUCTURED_ROW_LABEL_WORDS = frozenset({
    "annex", "annexure", "appendix", "box", "chapter", "chart", "exhibit",
    "figure", "no", "note", "page", "panel", "part", "section", "see",
    "table", "vol",
})

_PDF_STRUCTURED_ROW_TOKEN_RE = re.compile(r"[A-Z0-9][A-Z0-9./%$&+\-]*")


def _looks_like_pdf_structured_identifier_row(text: str) -> bool:
    """Detect a data-row fragment made only of dates, codes and numbers.

    Deal tables in market research stack cells such as
    ``01/16/26 QTSII 2026-1A A2`` into one visual line.  Every token is an
    opaque identifier, so the exact source text IS the correct translation;
    sending such a row to the model spends the full retry ladder twice (page
    pass plus clean page retry) and still ends as a recorded source fallback.
    One deal-table page measured 2026-08-31 held 28 of these rows and alone
    dominated a two-hour run.

    Any lower-case letter disqualifies the row, so prose — including
    title-case headings and month names such as ``12 June 2026`` — keeps
    translating.  A FIGURE/TABLE-style caption word also disqualifies it,
    because those all-caps labels have deterministic translations and must
    keep flowing to the short-label rules.  At least half of the tokens must
    carry a digit so all-caps directive lines (``PLEASE SEE PAGE 37.``)
    still fail closed.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    if not plain or len(plain) > 80:
        return False
    if re.search(r"[a-z]", plain):
        return False
    tokens = []
    for raw_token in plain.split(" "):
        stripped = raw_token.strip(".,;:()[]")
        if stripped:
            tokens.append(stripped)
    if len(tokens) < 2:
        return False
    alpha_tokens = 0
    digit_tokens = 0
    for token in tokens:
        if token.casefold() in _PDF_STRUCTURED_ROW_LABEL_WORDS:
            return False
        if not _PDF_STRUCTURED_ROW_TOKEN_RE.fullmatch(token):
            return False
        if any(char.isalpha() for char in token):
            alpha_tokens += 1
        if any(char.isdigit() for char in token):
            digit_tokens += 1
    if not alpha_tokens:
        return False
    return digit_tokens * 2 >= len(tokens)


# A telephone token needs structure — a parenthesized country/area group or
# a hyphenated digit group.  A bare space-separated digit run must NOT match:
# chart axis sequences (``90 94 98 02 06``) and year ranges in terse captions
# (``Global Bond Supply 2010 2026``) would otherwise read as phone numbers.
_PDF_CONTACT_PHONE_RE = re.compile(
    r"\(\+?\d[\d\s/-]{0,14}\)[\s\d/-]*|\d[\d\s/]*-[\d\s/-]*\d"
)

_PDF_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

_PDF_DATE_DMY_RE = re.compile(r"(?i)(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_PDF_DATE_MDY_RE = re.compile(r"(?i)([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})")


def _pdf_full_date_label_translation(text: str) -> str | None:
    """Deterministically translate an element that is exactly one full date.

    ``05 August 2026`` runs as a page header through entire research series,
    so every page pays an API call — or, when the model echoes it, a full
    retry ladder — for a closed, unambiguous transformation.  Only complete
    ``day month year`` / ``month day, year`` elements qualify; partial dates
    and dates inside prose keep going to the model.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip().rstrip(".")
    match = _PDF_DATE_DMY_RE.fullmatch(plain)
    if match:
        day, month_word, year = match.group(1), match.group(2), match.group(3)
    else:
        match = _PDF_DATE_MDY_RE.fullmatch(plain)
        if not match:
            return None
        month_word, day, year = match.group(1), match.group(2), match.group(3)
    month = _PDF_MONTH_NUMBERS.get(month_word.casefold())
    if month is None or not 1 <= int(day) <= 31:
        return None
    return f"{year}年{month}月{int(day)}日"


def _looks_like_pdf_contact_line(text: str) -> bool:
    """Detect an analyst byline: proper-name words plus phone digits.

    ``Nikolaos Panigirtzoglou AC (44-20) 7134-7815 <e-mail>`` repeats as a
    running header on every page of a research series.  The e-mail is already
    protected as an identifier; the residue is a person or organization name
    plus a telephone number, which must stay verbatim.  Any lower-case prose
    word (beyond name connectors and organization suffixes) disqualifies the
    line, and without a phone-number token this rule never fires.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    if not plain or len(plain) > 120:
        return False
    residue = _PRESERVED_IDENTIFIER_RE.sub(" ", plain)
    if not _PDF_CONTACT_PHONE_RE.search(residue):
        return False
    without_phones = _PDF_CONTACT_PHONE_RE.sub(" ", residue)
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(without_phones)
    ]
    if not 1 <= len(words) <= 8:
        return False
    for word in words:
        normalized = word.strip("&.")
        if not normalized:
            continue
        if normalized.casefold() in _PDF_PROPER_NAME_CONNECTORS:
            continue
        if normalized.casefold() in _PDF_PROPER_NAME_SUFFIXES:
            continue
        if not (normalized[:1].isupper() or normalized.isupper()):
            return False
    leftover = _SHORT_CITATION_WORD_RE.sub("", without_phones)
    return bool(re.fullmatch(r"[\s,;:.&'’()\[\]/\\-]*", leftover))


def _looks_like_pdf_translatable_short_label(text: str) -> bool:
    """Recognize compact English headings/roles that still need translation.

    The tiny-label exemption exists for names and chart atoms, but semantic
    labels such as ``Contents``, ``Executive Summary`` and ``Project Manager``
    must never be accepted unchanged.  The lower-case-word rule distinguishes
    ordinary phrases (``Price manipulation``) from title-cased personal names
    (``Kevin Houstoun``), ignoring name-safe connectors such as ``of``.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    if not plain or len(plain) > 100:
        return False

    unnumbered = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z])\s*[:.)]\s*",
        "",
        plain,
    ).strip()
    normalized = unnumbered.strip(" \t\r\n:;,.!?").casefold()
    if not normalized:
        return False
    if normalized in _PDF_TRANSLATABLE_SHORT_LABELS:
        return True

    if re.match(r"(?i)^(?:date|place|urgency)\s*:\s*\S", unnumbered):
        return True

    # Participant directories often put a role and an organization in one
    # visual cell.  The organization suffix / trailing acronym must not make
    # the entire role line immutable (for example ``CEO, ...`` or
    # ``Head of ..., ESMA``).
    if re.match(r"(?i)^(?:ceo|head\s+of|partner)\b(?:\s*,|\s+)", unnumbered):
        return True

    if re.fullmatch(
        r"(?i)(?:annex|annexure|appendix|chapter|section|part|figure|table|chart)"
        r"\s+[A-Z0-9][A-Za-z0-9.\-]*(?:\s*:)?",
        unnumbered,
    ):
        return True

    if re.fullmatch(r"(?i)see\s+chapter\s+\d+(?:\.\d+)*\.?", unnumbered):
        return True

    if re.match(r"^\s*\d+(?:\.\d+)*\s+[A-Za-z]", plain):
        return True

    if unnumbered != plain and re.search(r"[A-Za-z]", unnumbered):
        return True

    # Use Unicode letters so names such as ``Sandås`` remain one token.  The
    # former ASCII-only tokenizer split that surname into ``Sand`` + ``s``
    # and mistook the trailing fragment for a lower-case semantic word.
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(unnumbered)
    ]
    if len(words) >= 2:
        meaningful_tail = [
            word
            for word in words[1:]
            if word.casefold() not in _PDF_SHORT_LABEL_CONNECTORS
        ]
        if any(word[:1].islower() for word in meaningful_tail):
            return True

    # Sentence-like one-word fragments are language, not immutable chart/name
    # tokens.  Requiring a lower-case lead avoids acronyms and document codes.
    if (
        len(words) == 1
        and words[0][:1].islower()
        and bool(re.search(r"[.!?]\s*$", plain))
    ):
        return True
    return False


def _looks_like_pdf_proper_name_label(text: str) -> bool:
    """Detect a short organization/security/person label, never prose."""
    raw_plain = _plain_text(text).strip()
    plain = re.sub(r"\s+", " ", raw_plain).strip().rstrip(".;")
    if not plain or len(plain) > 80:
        return False
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(plain)
    ]
    if not 1 <= len(words) <= 8:
        return False
    for word in words:
        normalized = word.strip("&.")
        if not normalized:
            continue
        if normalized.casefold() in _PDF_PROPER_NAME_CONNECTORS:
            continue
        if not (normalized[0].isupper() or normalized.isupper()):
            return False

    residue = _SHORT_CITATION_WORD_RE.sub("", plain)
    if not re.fullmatch(r"[\s,&'’\-]*", residue):
        return False

    has_organization_marker = any(
        word.casefold() in _PDF_PROPER_NAME_SUFFIXES
        for word in words
    )
    has_honorific = (
        len(words) >= 2
        and words[0].casefold() in _PDF_PERSON_HONORIFICS
    )
    is_single_non_ascii_person_name = (
        len(words) == 1
        and words[0][:1].isupper()
        and any(ord(char) > 127 for char in words[0])
    )
    is_single_uppercase_identifier = (
        len(words) == 1
        and words[0].isupper()
        and 9 <= len(words[0]) <= 24
    )
    has_trailing_acronym = bool(
        re.search(r",\s*[A-Z][A-Z0-9&.\-]{1,11}\s*$", plain)
    )
    is_particle_surname = (
        len(words) >= 2
        and all(
            word.casefold() in _PDF_PROPER_NAME_CONNECTORS
            for word in words[:-1]
        )
        and words[-1][:1].isupper()
    )
    is_hyphenated_two_part_person_name = (
        len(words) == 2
        and "-" in words[0]
        and all(word[:1].isupper() for word in words)
    )

    # Some participant lists are extracted as two stacked name cells inside
    # one element (for example ``Sintiani Dewi\n\nTeddy``).  Limit this to
    # short, punctuation-free name fragments so ordinary multiline prose and
    # headings still require translation.
    stacked_parts = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n+", raw_plain)
        if part.strip()
    ]
    is_stacked_person_name = (
        len(stacked_parts) == 2
        and [len(_SHORT_CITATION_WORD_RE.findall(part)) for part in stacked_parts]
        == [2, 1]
        and not re.search(r"[,;:!?()]", raw_plain)
    )
    return (
        has_organization_marker
        or has_honorific
        or is_single_non_ascii_person_name
        or is_single_uppercase_identifier
        or has_trailing_acronym
        or is_particle_surname
        or is_hyphenated_two_part_person_name
        or is_stacked_person_name
    )


def _pdf_element_is_multiline_display_title(elem: dict) -> bool:
    """Recognize a large wrapped work title before proper-name preservation.

    Sparse front matter is often classified as ``scattered`` rather than as a
    heading.  A report title split over several source lines can then resemble
    an organization name token by token (especially when it ends in
    ``Markets``) and be restored to English from an otherwise valid cache.
    Require display typography and multiple visual lines, while excluding
    explicit people, directory rows, tables, and unambiguous institution-head
    endings.  Ordinary one-line company/name labels retain the existing
    preservation behavior.
    """
    if (
        elem.get("type") != "text"
        or elem.get("table_hint")
        or elem.get("entity_directory_hint")
        or elem.get("skip_translate_reason")
        or elem.get("non_horizontal")
        or float(elem.get("fontsize", 0.0)) < 15.0
    ):
        return False

    plain = re.sub(
        r"\s+", " ", _plain_text(elem.get("content", ""))
    ).strip().rstrip(".;:")
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(plain)
    ]
    if not 4 <= len(words) <= 20 or len(plain) > 180:
        return False

    source_line_count = max(
        int(elem.get("merged_visual_line_count") or 0),
        max(
            (
                len(paragraph.get("source_lines") or [])
                for paragraph in (elem.get("paragraphs") or [])
                if isinstance(paragraph, dict)
            ),
            default=0,
        ),
    )
    if source_line_count < 2:
        return False

    if words[0].casefold() in _PDF_PERSON_HONORIFICS:
        return False
    if re.search(r",\s*[A-Z][A-Z0-9&.\-]{1,11}\s*$", plain):
        return False
    if words[-1].casefold() in _PDF_DISPLAY_IDENTITY_END_WORDS:
        return False
    return True


def _looks_like_pdf_entity_directory_row(text: str) -> bool:
    """Return whether one short line can be an entity in a declared directory.

    This deliberately does not decide preservation on its own.  Generic title-
    cased phrases such as ``Market Liquidity Crisis`` still need translation.
    A caller must additionally establish page-level directory language and a
    sufficiently long, aligned run of peer rows.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    if not plain or len(plain) > 90 or _PRESERVED_IDENTIFIER_RE.search(plain):
        return False
    # A detached one-letter footnote marker is part of several company rows
    # (for example ``Knight Capital a``), not a lower-case prose word.
    lexical = re.sub(r"\s+[a-z]\s*$", "", plain).strip()
    words = [
        word.rstrip(".")
        for word in _SHORT_CITATION_WORD_RE.findall(lexical)
    ]
    if not 1 <= len(words) <= 10:
        return False
    if not any(any(char.isalpha() for char in word) for word in words):
        return False
    for word in words:
        normalized = word.strip("&.")
        if not normalized:
            continue
        if normalized.casefold() in _PDF_PROPER_NAME_CONNECTORS:
            continue
        if not (normalized[:1].isupper() or normalized.isupper()):
            return False
    residue = _SHORT_CITATION_WORD_RE.sub("", lexical)
    return bool(re.fullmatch(r"[\s,&'’()\-./]*", residue))


def _mark_pdf_entity_directory_elements(elements: list[dict]) -> int:
    """Mark aligned entity-name rows under explicit list/directory prose.

    Some PDFs contain an explanatory paragraph followed by a one-column list
    of firms.  Individual names such as ``All Options International`` are
    linguistically ambiguous without that page context and were repeatedly
    sent to the LLM.  Require both explicit directory language and at least
    five tightly aligned peer rows so ordinary short headings remain outside
    this preservation rule.
    """
    page_plain = re.sub(
        r"\s+",
        " ",
        " ".join(
            _plain_text(element.get("content", ""))
            for element in elements or []
            if element.get("type") == "text"
        ),
    ).strip()
    page_folded = page_plain.casefold()
    entity_noun = (
        r"(?:companies|company|entities|firms|hfts|institutions|members|"
        r"organisations|organizations|participants|traders)"
    )
    has_directory_context = bool(
        re.search(rf"\b(?:list|directory)\s+of\s+(?:the\s+)?{entity_noun}\b", page_folded)
        or re.search(
            r"\bmembers?\s+of\s+(?:the\s+)?(?:study|working|advisory|"
            r"steering)\s+group\b",
            page_folded,
        )
        or re.search(
            rf"\b{entity_noun}\b[^.!?]{{0,100}}\b(?:presented|listed|ordered)\s+"
            r"in\s+alphabetical\s+order\b",
            page_folded,
        )
    )
    if not has_directory_context:
        return 0

    def row_candidate(element: dict) -> bool:
        return bool(
            element.get("type") == "text"
            and element.get("layout_class") in {"table", "scattered"}
            and not element.get("bold")
            and _looks_like_pdf_entity_directory_row(element.get("content", ""))
        )

    candidates = [
        element for element in (elements or []) if row_candidate(element)
    ]
    columns: list[list[dict]] = []
    for element in sorted(
        candidates,
        key=lambda candidate: (
            _get_pdf_source_ink_rect(candidate).x0,
            _get_pdf_source_ink_rect(candidate).y0,
        ),
    ):
        rect = _get_pdf_source_ink_rect(element)
        fontsize = max(float(element.get("fontsize", 9.0)), 1.0)
        compatible_columns = [
            column
            for column in columns
            if abs(
                rect.x0 - _get_pdf_source_ink_rect(column[0]).x0
            ) <= max(12.0, fontsize * 1.5)
        ]
        if compatible_columns:
            min(
                compatible_columns,
                key=lambda column: abs(
                    rect.x0 - _get_pdf_source_ink_rect(column[0]).x0
                ),
            ).append(element)
        else:
            columns.append([element])

    runs: list[list[dict]] = []
    for column in columns:
        current: list[dict] = []
        for element in sorted(
            column,
            key=lambda candidate: _get_pdf_source_ink_rect(candidate).y0,
        ):
            rect = _get_pdf_source_ink_rect(element)
            if current:
                previous = current[-1]
                previous_rect = _get_pdf_source_ink_rect(previous)
                fontsize = max(
                    float(element.get("fontsize", 9.0)),
                    float(previous.get("fontsize", 9.0)),
                    1.0,
                )
                compact_gap = (
                    rect.y0 - previous_rect.y1
                    <= max(42.0, fontsize * 5.0)
                )
                forward = rect.y0 >= previous_rect.y0 - 0.5
                if not (compact_gap and forward):
                    runs.append(current)
                    current = []
            current.append(element)
        if current:
            runs.append(current)

    marked = 0
    for run in runs:
        if len(run) < 5:
            continue
        for element in run:
            if not element.get("entity_directory_hint"):
                element["entity_directory_hint"] = True
                marked += 1
    return marked


def _pdf_element_requires_translation(elem: dict) -> bool:
    """Return whether a text element needs an API-produced translation."""
    if elem.get("type") != "text" or elem.get("skip_translate_reason"):
        return False
    text = elem.get("content", "")
    if _is_disclaimer_block(text) or not _looks_like_translatable_english(text):
        return False
    plain = re.sub(r"\s+", " ", _plain_text(text)).strip()
    if (
        elem.get("layout_class") in {"table", "scattered"}
        and not elem.get("bold")
        and not elem.get("single_line_heading")
        and _looks_like_compact_pdf_identifier(plain)
    ):
        return False
    if (
        elem.get("entity_directory_hint")
        and elem.get("layout_class") in {"table", "scattered"}
        and not elem.get("bold")
    ):
        return False
    # Unlike the compact-identifier gate above, these deliberately ignore
    # ``single_line_heading`` and ``table_hint``: the measured deal-table rows
    # and analyst bylines carried those flags, and a row with no lower-case
    # prose cannot be a real heading the flags are meant to protect.  A
    # full-date element leaves the queue because its translation is
    # deterministic (applied by ``_translate_pdf_deterministic_labels``).
    if elem.get("layout_class") in {"table", "scattered"} and not elem.get("bold"):
        if _looks_like_pdf_structured_identifier_row(plain):
            return False
        if _looks_like_pdf_contact_line(text):
            return False
        if _pdf_full_date_label_translation(plain) is not None:
            return False
    if _pdf_element_is_multiline_display_title(elem):
        return True
    if _looks_like_pdf_translatable_short_label(text):
        return True
    if elem.get("glossary_term_hint") or elem.get("glossary_definition_hint"):
        return True
    if (
        elem.get("table_hint")
        and re.fullmatch(r"[A-Z][A-Z0-9]{8,23}", plain)
    ):
        return False
    if (
        not elem.get("table_hint")
        and elem.get("layout_class") in {"table", "scattered"}
        and not elem.get("bold")
        and not elem.get("single_line_heading")
        and _looks_like_pdf_proper_name_label(text)
    ):
        return False
    if (
        not elem.get("table_hint")
        and elem.get("layout_class") in {"table", "scattered"}
        and not elem.get("bold")
        and _looks_like_tiny_pdf_label(text)
    ):
        return False
    return True


def _dedupe_pdf_translation_targets(
    targets: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], dict[int, list[int]]]:
    """Deduplicate identical page-local API inputs and map copies to primaries."""
    unique_targets = []
    primary_by_text = {}
    duplicate_indices = {}
    for idx, text in targets:
        primary_idx = primary_by_text.get(text)
        if primary_idx is None:
            primary_by_text[text] = idx
            unique_targets.append((idx, text))
            continue
        duplicate_indices.setdefault(primary_idx, []).append(idx)
    return unique_targets, duplicate_indices


def _translate_pdf_fixed_short_label(text: str) -> str:
    """Translate a closed set of standalone chart/table labels offline.

    The ordinary tiny-label exemption protects company names, series codes,
    and mathematical identifiers from context-free API translation.  A small
    controlled vocabulary of unambiguous functional labels can still be
    translated deterministically, provided the *whole* element matches.
    """
    raw = text or ""
    plain = re.sub(r"\s+", " ", _plain_text(raw)).strip()
    replacement = PDF_FIXED_SHORT_LABEL_TRANSLATIONS.get(plain.casefold())
    if replacement is not None:
        return _restore_pdf_superscript_markup(raw, replacement)

    date_translation = _pdf_full_date_label_translation(plain)
    if date_translation is not None:
        return _restore_pdf_superscript_markup(raw, date_translation)

    band_match = re.fullmatch(
        r"(?i)band\s+(\d+)(\s*\([^)]*\))?",
        plain,
    )
    if band_match:
        suffix = band_match.group(2) or ""
        return f"第{band_match.group(1)}档{suffix}"

    times_match = re.fullmatch(
        r"([+−\-–—]?\d+(?:[.,]\d+)?(?:\s*[−\-–—]\s*\d+(?:[.,]\d+)?)?)\s+times?",
        plain,
        re.IGNORECASE,
    )
    if times_match:
        return f"{times_match.group(1)}倍"
    return raw


def _translate_pdf_deterministic_labels(text: str) -> str:
    """Apply every context-free PDF label translation in one stable order."""
    return _translate_pdf_fixed_short_label(
        _translate_pdf_citation_labels(text or "")
    )


def _extract_pdf_table_elements(
    page,
) -> tuple[list[dict], list[fitz.Rect], list[dict]]:
    """Return table placeholders and their geometry for one page."""
    elements: list[dict] = []
    table_rects: list[fitz.Rect] = []
    table_regions: list[dict] = []
    try:
        table_regions = _find_pdf_table_regions(page)
        for region in table_regions:
            rect = fitz.Rect(region["rect"])
            table_rects.append(rect)
            elements.append({
                "type": "table_image",
                "y": rect.y0,
                "rect": rect,
                "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
            })
    except Exception as exc:
        log.warning(f"Table detection failed: {exc}")
    return elements, table_rects, table_regions


def _extract_pdf_image_elements(
    page,
    table_rects: list[fitz.Rect],
) -> tuple[list[dict], list[fitz.Rect]]:
    """Return image placeholders, excluding images substantially inside tables."""
    elements: list[dict] = []
    image_rects: list[fitz.Rect] = []
    try:
        for image_info in page.get_images(full=True):
            image_rects_for_xref = page.get_image_rects(image_info[0])
            if not image_rects_for_xref:
                continue
            try:
                for image_rect in image_rects_for_xref:
                    rect = fitz.Rect(image_rect)
                    image_rects.append(fitz.Rect(rect))
                    overlaps_table = any(
                        not (intersection := rect & table_rect).is_empty
                        and (intersection.width * intersection.height)
                        / max(rect.width * rect.height, 1)
                        > 0.3
                        for table_rect in table_rects
                    )
                    if not overlaps_table:
                        elements.append({
                            "type": "image",
                            "y": rect.y0,
                            "rect": rect,
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        })
            except Exception as image_exc:
                log.warning(
                    "Page %s: skipped unreadable image placement: %s",
                    page.number + 1,
                    image_exc,
                )
                continue
    except Exception as exc:
        log.warning(
            "Page %s: image enumeration failed; continuing without image "
            "placeholders: %s",
            page.number + 1,
            exc,
        )
    return elements, image_rects


def _pdf_span_entry(span: dict) -> dict | None:
    """Normalize a PyMuPDF span into the internal rich-text representation."""
    text = span["text"]
    if not text:
        return None
    span_bold = bool(span.get("flags", 0) & (1 << 4))
    font_name = span.get("font", "").lower()
    is_bold = (
        span_bold
        or "bold" in font_name
        or "heavy" in font_name
        or "black" in font_name
    )
    return {
        "text": text,
        "font": str(span.get("font", "")),
        "x0": float(span["bbox"][0]),
        "x1": float(span["bbox"][2]),
        "y0": float(span["bbox"][1]),
        "y1": float(span["bbox"][3]),
        "size": float(span["size"]),
        "color": span["color"],
        "flags": int(span.get("flags", 0)),
        "origin": tuple(span.get("origin", (span["bbox"][0], span["bbox"][3]))),
        "bold": is_bold,
    }


def _make_pdf_split_layout_element(
    *,
    page_rect: fitz.Rect,
    line: dict,
    cell: dict,
    line_fontsize: float,
    line_in_table: bool,
    line_non_horizontal: bool,
    cluster_math: dict,
) -> dict | None:
    """Build one semantic cell from a visually split PDF line."""
    cluster = cell["cluster"]
    cluster_plain = re.sub(
        r"\s+",
        " ",
        "".join(span["text"] for span in cluster),
    ).strip()
    if not cluster_plain:
        return None
    cluster_rich = "".join(span["rich"] for span in cluster).strip()
    cluster_x0 = float(cell["x0"])
    cluster_x1 = float(cell["x1"])
    cluster_y0 = min(span["y0"] for span in cluster)
    cluster_y1 = max(span["y1"] for span in cluster)
    cluster_sizes = [span["size"] for span in cluster]
    cluster_colors = [span["color"] for span in cluster]
    cluster_bold_chars = sum(
        len(span["text"].strip()) for span in cluster if span["bold"]
    )
    cluster_total_chars = sum(len(span["text"].strip()) for span in cluster)
    cluster_superscript_runs = [
        {
            "text": span["text"].strip(),
            "scale": float(span.get("superscript_scale") or 0.60),
            "source": span.get("superscript_source"),
        }
        for span in cluster
        if span.get("superscript") and span.get("text", "").strip()
    ]
    cluster_superscript_scale = _pick_pdf_dominant_value(
        [
            (round(run["scale"], 3), max(len(run["text"]), 1))
            for run in cluster_superscript_runs
        ],
        0.60,
    )
    rect = fitz.Rect(cluster_x0, cluster_y0, cluster_x1, cluster_y1)
    return {
        "type": "text",
        "y": cluster_y0,
        "x": cluster_x0,
        "rect": rect,
        "bbox": [cluster_x0, cluster_y0, cluster_x1, cluster_y1],
        "render_bbox": _get_pdf_render_bbox(
            rect,
            page_rect,
            [{
                "plain": cluster_plain,
                "text_align": cell["align"],
                "nowrap": True,
            }],
            line_fontsize,
        ),
        "content": cluster_plain,
        "rich_content": (
            cluster_rich
            if (
                cluster_superscript_runs
                or 0 < cluster_bold_chars < cluster_total_chars * 0.9
            )
            else None
        ),
        "superscript_runs": cluster_superscript_runs,
        "superscript_scale": (
            float(cluster_superscript_scale)
            if cluster_superscript_runs
            else None
        ),
        "paragraphs": [{
            "plain": cluster_plain,
            "rich": cluster_rich,
            "margin_left": 0.0,
            "text_indent": 0.0,
            "gap_before": 0.0,
            "text_align": cell["align"],
            "nowrap": True,
        }],
        "top_padding": max(0.0, cluster_y0 - float(line["bbox"][1])),
        "fontsize": (
            max(set(cluster_sizes), key=cluster_sizes.count)
            if cluster_sizes
            else line_fontsize
        ),
        "line_height": max(cluster_y1 - cluster_y0, line_fontsize * 1.05),
        "color": (
            max(set(cluster_colors), key=cluster_colors.count)
            if cluster_colors
            else 0
        ),
        "bold": (
            cluster_bold_chars > cluster_total_chars * 0.5
            if cluster_total_chars > 0
            else False
        ),
        "non_horizontal": line_non_horizontal,
        "rotation": _pdf_rotation_from_direction(line.get("dir")),
        "inline_math_fragments": [
            dict(fragment)
            for fragment in (cluster_math.get("inline_math_fragments") or [])
        ],
        "preserve_source_style": bool(line_in_table),
        "table_hint": bool(line_in_table),
    }


def _extract_pdf_line_record(
    line: dict,
    block_rect: fitz.Rect,
    table_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
) -> tuple[list[dict], dict | None]:
    """Extract either split-layout elements or one normal line record."""
    span_entries = [
        entry
        for span in line.get("spans", [])
        if (entry := _pdf_span_entry(span)) is not None
    ]
    if not span_entries:
        return [], None

    _mark_pdf_superscript_spans(span_entries)
    line_rect = fitz.Rect(line["bbox"])
    line_in_table = _line_overlaps_pdf_table_rects(line_rect, table_rects)
    line_math = _pdf_line_math_evidence(
        span_entries,
        table_hint=line_in_table,
    )
    line_superscript_runs = [
        {
            "text": span["text"].strip(),
            "scale": float(span.get("superscript_scale") or 0.60),
            "source": span.get("superscript_source"),
        }
        for span in span_entries
        if span.get("superscript") and span.get("text", "").strip()
    ]
    line_fontsize = float(_pick_pdf_dominant_value(
        [
            (float(span["size"]), max(len(span["text"].strip()), 1))
            for span in span_entries
        ],
        float(span_entries[0]["size"]),
    ))
    line_color = max(
        set(span["color"] for span in span_entries),
        key=[span["color"] for span in span_entries].count,
    )
    line_plain_chars = sum(len(span["text"].strip()) for span in span_entries)
    line_bold_chars = sum(
        len(span["text"].strip()) for span in span_entries if span["bold"]
    )
    line_rotation = _pdf_rotation_from_direction(line.get("dir"))
    line_non_horizontal = line_rotation in {90, 270}
    clusters = _merge_pdf_list_marker_clusters(
        _cluster_pdf_line_spans(span_entries, line_fontsize)
    )
    if _looks_like_split_layout_line(
        clusters,
        line["bbox"],
        block_rect,
        line_fontsize,
    ):
        split_elements: list[dict] = []
        for cell in _split_pdf_line_layout_cells(clusters, line["bbox"]):
            cluster_math = _pdf_line_math_evidence(
                cell["cluster"],
                table_hint=line_in_table,
            )
            if cluster_math["protected"]:
                cluster = cell["cluster"]
                formula_element = _make_pdf_formula_element_from_lines([{
                    "x0": min(span["x0"] for span in cluster),
                    "x1": max(span["x1"] for span in cluster),
                    "y0": min(span["y0"] for span in cluster),
                    "y1": max(span["y1"] for span in cluster),
                    "math_reasons": cluster_math["reasons"],
                    "math_mixed": cluster_math["mixed"],
                    "math_symbol_count": cluster_math["math_symbol_count"],
                }])
                if formula_element:
                    split_elements.append(formula_element)
                continue
            element = _make_pdf_split_layout_element(
                page_rect=page_rect,
                line=line,
                cell=cell,
                line_fontsize=line_fontsize,
                line_in_table=line_in_table,
                line_non_horizontal=line_non_horizontal,
                cluster_math=cluster_math,
            )
            if element:
                split_elements.append(element)
        return split_elements, None

    return [], {
        "plain": "".join(span["text"] for span in span_entries),
        "rich": "".join(span["rich"] for span in span_entries),
        "x0": float(line["bbox"][0]),
        "x1": float(line["bbox"][2]),
        "y0": float(line["bbox"][1]),
        "y1": float(line["bbox"][3]),
        "fontsize": line_fontsize,
        "line_height": float(line["bbox"][3]) - float(line["bbox"][1]),
        "color": line_color,
        "bold": (
            line_bold_chars > line_plain_chars * 0.5
            if line_plain_chars > 0
            else False
        ),
        "non_horizontal": line_non_horizontal,
        "rotation": line_rotation,
        "table_hint": bool(line_in_table),
        "drop_cap": _detect_pdf_drop_cap_span(span_entries),
        "superscript_runs": line_superscript_runs,
        "inline_math_fragments": [
            dict(fragment)
            for fragment in (line_math.get("inline_math_fragments") or [])
        ],
        "math_protected": bool(line_math["protected"]),
        "math_reasons": list(line_math["reasons"]),
        "math_mixed": bool(line_math["mixed"]),
        "math_symbol_count": int(line_math["math_symbol_count"]),
    }


def _group_pdf_block_lines(
    line_infos: list[dict],
) -> list[tuple[tuple[bool, bool], list[dict]]]:
    """Group consecutive lines by table and formula protection state."""
    line_groups: list[tuple[tuple[bool, bool], list[dict]]] = []
    current_group: list[dict] = []
    current_key: tuple[bool, bool] | None = None
    for line_info in line_infos:
        key = (
            bool(line_info.get("table_hint")),
            bool(line_info.get("math_protected")),
        )
        if current_group and key != current_key:
            line_groups.append((current_key, current_group))
            current_group = [line_info]
            current_key = key
            continue
        if not current_group:
            current_key = key
        current_group.append(line_info)
    if current_group and current_key is not None:
        line_groups.append((current_key, current_group))
    return line_groups


def _pdf_bold_run_in_wraps_into_mixed_line(
    heading_lines: list[dict],
    transition_line: dict,
) -> bool:
    """Return whether a bold run-in sentence crosses a visual line boundary.

    A common report layout bolds the opening sentence of a paragraph.  When
    that sentence wraps, the first visual line is entirely bold while the next
    line starts bold and changes to regular text after the sentence-ending
    punctuation.  The generic heading splitter used to treat the fully bold
    first line as a standalone heading, so the two halves of one sentence were
    translated independently.

    Require the mixed line to contain both a leading bold run and a regular
    suffix, and require that the leading bold run completes a sentence.  Those
    signals distinguish a wrapped run-in sentence from an ordinary standalone
    heading followed by body prose.
    """
    if not heading_lines or not transition_line:
        return False
    previous = heading_lines[-1]
    if (
        not previous.get("bold")
        or _ends_with_sentence_boundary(previous.get("plain", ""))
    ):
        return False

    rich_remainder = str(transition_line.get("rich") or "").lstrip()
    bold_fragments: list[str] = []
    while rich_remainder:
        match = re.match(r"(?is)^<b\b[^>]*>(.*?)</b>", rich_remainder)
        if match is None:
            break
        fragment = re.sub(r"\s+", " ", _plain_text(match.group(1))).strip()
        if fragment:
            bold_fragments.append(fragment)
        rich_remainder = rich_remainder[match.end():].lstrip()

    bold_prefix = " ".join(bold_fragments).strip()
    regular_suffix = re.sub(r"\s+", " ", _plain_text(rich_remainder)).strip()
    if (
        not bold_prefix
        or not regular_suffix
        or not _ends_with_sentence_boundary(bold_prefix)
    ):
        return False

    previous_size = max(float(previous.get("fontsize", 11.0)), 1.0)
    transition_size = max(float(transition_line.get("fontsize", 11.0)), 1.0)
    if abs(previous_size - transition_size) > max(previous_size * 0.08, 0.7):
        return False

    previous_height = max(
        float(previous.get("y1", 0.0)) - float(previous.get("y0", 0.0)),
        1.0,
    )
    transition_height = max(
        float(transition_line.get("y1", 0.0))
        - float(transition_line.get("y0", 0.0)),
        1.0,
    )
    vertical_gap = float(transition_line.get("y0", 0.0)) - float(
        previous.get("y1", 0.0)
    )
    if vertical_gap < -max(previous_height, transition_height) * 0.35:
        return False
    if vertical_gap > max(previous_size * 0.95, previous_height, transition_height):
        return False

    return abs(
        float(transition_line.get("x0", 0.0)) - float(previous.get("x0", 0.0))
    ) <= max(previous_size * 0.60, 6.0)


def _split_pdf_heading_segments(group_lines: list[dict]) -> list[list[dict]]:
    """Separate a short bold heading glued to following body text."""
    dominant_size = float(_pick_pdf_dominant_value(
        [
            (
                float(line.get("fontsize", 11.0)),
                max(len(line.get("plain", "").strip()), 1),
            )
            for line in group_lines
        ],
        11.0,
    ))
    segments = _split_pdf_disjoint_line_segments(group_lines, dominant_size)
    split_segments: list[list[dict]] = []
    for candidate_lines in segments:
        if len(candidate_lines) < 2:
            split_segments.append(candidate_lines)
            continue
        heading_count = 0
        while (
            heading_count < len(candidate_lines)
            and heading_count < 2
            and not candidate_lines[heading_count].get("drop_cap")
            and (
                candidate_lines[heading_count].get("bold")
                or _looks_like_heading_text(
                    candidate_lines[heading_count].get("plain", "")
                )
            )
        ):
            heading_count += 1
        next_line_is_body = (
            0 < heading_count < len(candidate_lines)
            and not (
                candidate_lines[heading_count].get("bold")
                or _looks_like_heading_text(
                    candidate_lines[heading_count].get("plain", "")
                )
            )
        )
        if next_line_is_body and _pdf_bold_run_in_wraps_into_mixed_line(
            candidate_lines[:heading_count],
            candidate_lines[heading_count],
        ):
            next_line_is_body = False
        if next_line_is_body:
            split_segments.extend([
                candidate_lines[:heading_count],
                candidate_lines[heading_count:],
            ])
        else:
            split_segments.append(candidate_lines)
    return split_segments


def _append_pdf_grouped_line_elements(
    elements: list[dict],
    line_infos: list[dict],
    page_rect: fitz.Rect,
) -> None:
    """Convert normalized line groups into formula or text elements."""
    for (table_hint, math_protected), group_lines in _group_pdf_block_lines(
        line_infos
    ):
        if math_protected:
            formula_element = _make_pdf_formula_element_from_lines(group_lines)
            if formula_element:
                elements.append(formula_element)
            continue
        if table_hint:
            for group_line in group_lines:
                element = _make_pdf_text_element_from_lines(
                    [group_line],
                    page_rect,
                    preserve_source_style=True,
                    table_hint=True,
                )
                if element:
                    elements.append(element)
            continue
        for segment_lines in _split_pdf_heading_segments(group_lines):
            element = _make_pdf_text_element_from_lines(
                segment_lines,
                page_rect,
            )
            if element:
                elements.append(element)


def _append_pdf_text_block_elements(
    elements: list[dict],
    block: dict,
    page,
    table_rects: list[fitz.Rect],
) -> None:
    """Append all semantic elements represented by one PDF text block."""
    block, removed_hidden_spans = _sanitize_pdf_hidden_text_artifact_spans(
        block,
        fitz.Rect(page.rect),
    )
    if removed_hidden_spans:
        log.info(
            "Page %s: ignored %s oversized hidden text artifact span(s)",
            page.number + 1,
            removed_hidden_spans,
        )
    if not block:
        return
    block_rect = fitz.Rect(block["bbox"])
    line_infos: list[dict] = []
    for line in block.get("lines", []):
        split_elements, line_info = _extract_pdf_line_record(
            line,
            block_rect,
            table_rects,
            fitz.Rect(page.rect),
        )
        elements.extend(split_elements)
        if line_info is not None:
            line_infos.append(line_info)

    line_infos = _normalize_pdf_drop_cap_lines(line_infos, block_rect)
    line_infos = _normalize_pdf_same_baseline_prose_fragments(
        line_infos,
        block_rect,
    )
    _demote_pdf_wrapped_identifier_math_lines(line_infos)
    _propagate_pdf_math_line_context(line_infos)
    if line_infos:
        _append_pdf_grouped_line_elements(
            elements,
            line_infos,
            fitz.Rect(page.rect),
        )


def _drop_pdf_duplicate_glyph_spans(page_dict: dict) -> tuple[dict, int]:
    """Keep one copy of text that the generator stroked twice in place.

    Apache FOP 2.7 emits every body line as two identical spans at the same
    coordinates.  Extraction then produced two overlapping elements per line
    plus stray half-line fragments, and because redaction erases a glyph whose
    box is merely clipped, the overlap-redraw closure in ``rendering`` grew
    until it reached a ``formula_risk_preserved`` element and failed the whole
    page.  One such page then dragged its entire accepted cross-page merge
    component back to untranslated source copies -- 19 pages from a single
    duplicated line.

    Only redundant ink is removed: same text, same font, same size, and the
    same position within ``PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE``.  Text
    repeated elsewhere on the page -- table cells, running heads, a column of
    ``0.00%`` -- sits at a different bbox and survives.

    Colour is deliberately NOT part of the identity, because the observed
    duplication is a white knockout copy at ``0xffffff`` followed by the real
    ``0x231f20`` copy.  The survivor must therefore be the LAST occurrence:
    PDF paints in stream order, so the final copy is the one a reader sees.
    Keeping the first copy instead inherited the white colour and redrew every
    translated paragraph in white on white -- a fully blank page that still
    extracted perfect Chinese text.
    """
    spans_in_order: list[dict] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans_in_order.append(span)

    winner_by_key: dict[tuple, list[list]] = {}
    for position, span in enumerate(spans_in_order):
        text = span.get("text", "")
        if not text.strip():
            continue
        try:
            bbox = tuple(float(value) for value in span.get("bbox", ()))
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        key = (
            text,
            span.get("font", ""),
            round(float(span.get("size", 0.0)), 2),
        )
        for entry in winner_by_key.setdefault(key, []):
            if all(
                abs(bbox[axis] - entry[0][axis])
                <= PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE
                for axis in range(4)
            ):
                entry[1].append(position)
                break
        else:
            winner_by_key[key].append([bbox, [position]])

    discarded: set[int] = set()
    for entries in winner_by_key.values():
        for _bbox, positions in entries:
            discarded.update(positions[:-1])
    if not discarded:
        return page_dict, 0

    # Rebuild with a counter that mirrors the walk above, so a span is matched
    # by its position in stream order rather than by identity or value.
    position = 0
    blocks = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            blocks.append(block)
            continue
        lines = []
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                if position not in discarded:
                    spans.append(span)
                position += 1
            if spans:
                lines.append({**line, "spans": spans})
        if lines:
            blocks.append({**block, "lines": lines})
    return {**page_dict, "blocks": blocks}, len(discarded)


def _extract_pdf_text_elements(
    page,
    table_rects: list[fitz.Rect],
) -> list[dict]:
    """Extract rich text and formula elements from every text block.

    A ``get_text`` failure must propagate: the extraction stage wraps it into
    ``PDFPageExtractionError`` so the recovery policy preserves the exact
    source page with an audit record, instead of silently emitting a page
    with no text elements.
    """
    elements: list[dict] = []
    page_dict = page.get_text(
        "dict",
        flags=fitz.TEXT_PRESERVE_WHITESPACE,
    )
    page_dict, duplicate_spans = _drop_pdf_duplicate_glyph_spans(page_dict)
    if duplicate_spans:
        log.info(
            "Page %s: dropped %s duplicate glyph span(s) stroked in place",
            page.number + 1,
            duplicate_spans,
        )
    dropped_blocks = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        try:
            _append_pdf_text_block_elements(
                elements,
                block,
                page,
                table_rects,
            )
        except Exception as block_exc:
            dropped_blocks += 1
            log.warning(
                "PDF text block extraction failed at "
                f"{block.get('bbox')}: {block_exc}"
            )
    if dropped_blocks:
        log.warning(
            "Page %s: dropped %s unextractable text block(s)",
            page.number + 1,
            dropped_blocks,
        )
    return elements


def _pdf_element_sort_key(element: dict) -> tuple[float, float]:
    bbox = element.get("bbox")
    fallback_x = bbox[0] if isinstance(bbox, list) else 0
    return element["y"], element.get("x", fallback_x)


def _merge_and_validate_pdf_page_semantics(
    page,
    elements: list[dict],
    table_rects: list[fitz.Rect],
    table_regions: list[dict],
) -> list[dict]:
    """Run ordered semantic merges and enforce their postconditions."""
    page_rect = fitz.Rect(page.rect)
    elements.sort(key=_pdf_element_sort_key)
    elements, semantic_table_cell_count = _merge_pdf_semantic_table_cells(
        elements,
        table_rects,
        page_rect,
        table_regions=table_regions,
    )
    if semantic_table_cell_count:
        log.info(
            "Page %s: merged %s wrapped semantic table cell(s)",
            page.number + 1,
            semantic_table_cell_count,
        )
    elements = _split_pdf_text_elements_into_semantic_fragments(
        elements,
        page_rect,
    )
    elements.sort(key=_pdf_element_sort_key)
    elements = _merge_pdf_detached_list_marker_elements(elements, page_rect)
    elements = _merge_pdf_wrapped_line_elements(elements, page_rect)
    elements = _merge_pdf_semantic_continuation_elements(elements, page_rect)
    elements = _merge_adjacent_heading_elements(elements)
    wrapped_math_count = _merge_pdf_wrapped_inline_math_continuation_fragments(
        elements,
        page_rect,
    )
    if wrapped_math_count:
        log.info(
            "Page %s: merged %s wrapped inline-math continuation fragment(s)",
            page.number + 1,
            wrapped_math_count,
        )

    _classify_pdf_page_text_elements(elements, page_rect)
    elements, strong_count = _merge_pdf_strong_continuation_elements(
        elements,
        page_rect,
    )
    if strong_count:
        _classify_pdf_page_text_elements(elements, page_rect)
        log.info(
            "Page %s: merged %s strong post-classification paragraph "
            "continuation(s)",
            page.number + 1,
            strong_count,
        )
    elements, reference_split_count = _split_pdf_reference_entry_elements(
        elements,
        page_rect,
    )
    if reference_split_count:
        _classify_pdf_page_text_elements(elements, page_rect)
        log.info(
            "Page %s: split %s coupled bibliography entry boundary/boundaries",
            page.number + 1,
            reference_split_count,
        )
    elements, reference_merge_count = _merge_pdf_reference_entry_elements(
        elements,
        page_rect,
    )
    if reference_merge_count:
        _classify_pdf_page_text_elements(elements, page_rect)
        log.info(
            "Page %s: merged %s hanging reference continuation(s)",
            page.number + 1,
            reference_merge_count,
        )

    strong_residuals = _pdf_strong_continuation_residual_pairs(
        elements,
        page_rect,
    )
    if strong_residuals:
        raise RuntimeError(
            f"Page {page.number + 1}: semantic paragraph coalescing invariant "
            f"failed ({len(strong_residuals)} unmerged strong continuation "
            "pair(s))"
        )
    paragraph_residuals = _pdf_internal_paragraph_lead_residuals(elements)
    if paragraph_residuals:
        raise RuntimeError(
            f"Page {page.number + 1}: semantic paragraph boundary invariant "
            f"failed ({len(paragraph_residuals)} embedded successor lead(s))"
        )
    reference_residuals = _pdf_reference_continuation_residual_pairs(
        elements,
        page_rect,
    )
    if reference_residuals:
        raise RuntimeError(
            f"Page {page.number + 1}: reference-entry coalescing invariant "
            f"failed ({len(reference_residuals)} unmerged hanging continuation "
            "pair(s))"
        )
    reference_structure_residuals = _pdf_reference_entry_structure_residuals(
        elements
    )
    if reference_structure_residuals:
        raise RuntimeError(
            f"Page {page.number + 1}: reference-entry semantic invariant "
            f"failed ({len(reference_structure_residuals)} element(s) still "
            "contain multiple bibliography entries)"
        )

    entity_count = _mark_pdf_entity_directory_elements(elements)
    if entity_count:
        log.info(
            "Page %s: marked %s aligned entity-directory row(s)",
            page.number + 1,
            entity_count,
        )
    paragraph_split_count = _split_pdf_formula_adjacent_paragraph_elements(
        elements,
        page_rect,
    )
    if paragraph_split_count:
        log.info(
            "Page %s: isolated %s formula-adjacent paragraph fragment(s)",
            page.number + 1,
            paragraph_split_count,
        )
    promoted_count = _promote_pdf_formula_overlaps(elements)
    if promoted_count:
        log.info(
            "Page %s: conservatively preserved %s text element(s) touching "
            "formula geometry",
            page.number + 1,
            promoted_count,
        )
    _mark_pdf_reference_entry_elements(elements)
    return elements


def _enrich_pdf_page_elements(
    page,
    elements: list[dict],
    image_rects: list[fitz.Rect],
) -> list[dict]:
    """Add image OCR, citation, and immutable-formula audit metadata."""
    page_rect = fitz.Rect(page.rect)
    embedded_count = _mark_pdf_embedded_thumbnail_text_elements(
        elements,
        image_rects,
        page_rect,
    )
    if embedded_count:
        log.info(
            "Page %s: preserved %s tiny text element(s) inside embedded image "
            "thumbnail(s)",
            page.number + 1,
            embedded_count,
        )

    vector_elements = _extract_pdf_vector_ocr_elements(
        page,
        elements,
        image_rects,
    )
    if vector_elements:
        elements.extend(vector_elements)
        elements.sort(key=lambda element: (
            float(element.get("y", _get_pdf_elem_rect(element).y0)),
            float(element.get("x", _get_pdf_elem_rect(element).x0)),
        ))

    completed_separators = (
        _complete_pdf_citation_separator_inline_math_fragments(elements)
    )
    if completed_separators:
        log.info(
            "Page %s: restored %s merged citation-separator protection "
            "record(s)",
            page.number + 1,
            completed_separators,
        )

    try:
        page_dict = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE,
        )
    except Exception as page_dict_exc:
        log.warning(
            f"Page {page.number + 1}: formula signature extraction failed: "
            f"{type(page_dict_exc).__name__}"
        )
        return elements

    # Signatures are per-element evidence, so they must fail per element.  A
    # shared ``try`` around the loop let one unsignable region silently strip
    # the signature from every later formula on the page, and the audit then
    # rejected the finished document with one
    # ``source-formula-signature-missing`` warning per stripped element.
    for index, element in enumerate(elements):
        if element.get("type") != "formula_image":
            continue
        try:
            element["math_signature"] = _pdf_formula_region_signature(
                page,
                element.get("bbox", element.get("rect")),
                page_dict=page_dict,
            )
        except Exception as signature_exc:
            log.warning(
                f"Page {page.number + 1} elem {index}: formula signature "
                f"extraction failed: {type(signature_exc).__name__}: "
                f"{signature_exc}"
            )
    return elements


def _extract_page_elements(page, doc):
    """Extract and semantically normalize ordered elements from one PDF page."""
    del doc  # retained for the legacy callback contract
    table_elements, table_rects, table_regions = _extract_pdf_table_elements(
        page
    )
    image_elements, image_rects = _extract_pdf_image_elements(
        page,
        table_rects,
    )
    elements = [
        *table_elements,
        *image_elements,
        *_extract_pdf_text_elements(page, table_rects),
    ]
    elements = _merge_and_validate_pdf_page_semantics(
        page,
        elements,
        table_rects,
        table_regions,
    )
    return _enrich_pdf_page_elements(page, elements, image_rects)
