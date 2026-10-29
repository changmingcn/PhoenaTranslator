"""Decide which text elements must be translated and which keep their
source text (identifiers, citations, names, contact lines, tiny labels)."""

from __future__ import annotations

import re
from collections import Counter

from phoena_translator.math_text import is_math_block as _is_math_block
from phoena_translator.pdf.extract.math_detection import (
    _looks_like_compact_pdf_identifier,
)
from phoena_translator.pdf.model import ElementType, LayoutClass
from phoena_translator.pdf.patterns import (
    _PDF_PROPER_NAME_CONNECTORS,
    _PRESERVED_IDENTIFIER_RE,
)
from phoena_translator.pdf.semantics.watermarks import _is_disclaimer_block
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.translate.deterministic_labels import (
    _pdf_full_date_label_translation,
)

_PRESERVED_STRUCTURED_CODE_RE = re.compile(
    r"(?i)(?=[A-Z0-9/_\-]*\d)(?=(?:.*[/_\-]){2,})"
    r"[A-Z0-9]+(?:[/_\-][A-Z0-9]+){2,}[/_\-]?"
)
_SHORT_CITATION_YEAR_RE = re.compile(r"\((?:18|19|20)\d{2}[a-z]?\)", re.IGNORECASE)
_SHORT_CITATION_WORD_RE = re.compile(
    r"[^\W\d_]+(?:['’\-][^\W\d_]+)*\.?",
    re.UNICODE,
)
_SHORT_CITATION_CONNECTORS = {
    "al", "and", "da", "de", "del", "der", "di", "dos", "du", "et",
    "jr", "la", "le", "st", "van", "von",
}
_PDF_TRANSLATABLE_SHORT_LABELS = {
    "abstract",
    "acknowledgement",
    "acknowledgements",
    "annex",
    "annexure",
    "appendix",
    "background",
    "bibliography",
    "chapter",
    "chart",
    "conclusion",
    "conclusions",
    "contents",
    "definition",
    "definitions",
    "discussion",
    "graphical analysis",
    "executive summary",
    "figure",
    "findings",
    "foreword",
    "glossary",
    "implication",
    "implications",
    "index",
    "intern",
    "introduction",
    "interim order",
    "methodology",
    "model",
    "models",
    "note",
    "overview",
    "part",
    "preface",
    "pfutp regulations",
    "project co-ordinator",
    "project coordinator",
    "project leader",
    "project manager",
    "project researcher",
    "recommendation",
    "recommendations",
    "reference",
    "references",
    "ratio",
    "result",
    "results",
    "section",
    "sebi act",
    "source",
    "table",
    "test",
    "tests",
    "terms",
    "acronyms",
    "chapter title",
    "workshop reports",
    "year",
    "urgency",
    "volatility",
}
_PDF_SHORT_LABEL_CONNECTORS = {
    "and", "at", "by", "da", "de", "der", "di", "for", "from", "in", "of",
    "on", "the", "to", "van", "von", "with",
}
_PDF_PROPER_NAME_SUFFIXES = {
    "advisors", "asset", "bank", "capital", "company", "consulting", "corp",
    "corporation", "exchange", "fund", "group", "holdings", "inc", "investments",
    "limited", "llc", "lp", "ltd", "management", "markets", "partners", "plc",
    "securities", "trust", "university", "institute", "college", "school",
}
_PDF_PERSON_HONORIFICS = {
    "dame", "dr", "mr", "mrs", "ms", "prof", "professor", "sir",
}
_PDF_DISPLAY_IDENTITY_END_WORDS = {
    "advisors", "association", "authority", "bank", "commission", "commissions",
    "committee", "company",
    "consulting", "corporation", "council", "department", "exchange", "fund",
    "government", "group", "holdings", "institute", "management", "ministry",
    "office", "organization", "organisation", "partners", "school", "trust",
    "university",
}
_PDF_STRUCTURED_ROW_LABEL_WORDS = frozenset({
    "annex", "annexure", "appendix", "box", "chapter", "chart", "exhibit",
    "figure", "no", "note", "page", "panel", "part", "section", "see",
    "table", "vol",
})
_PDF_STRUCTURED_ROW_TOKEN_RE = re.compile(r"[A-Z0-9][A-Z0-9./%$&+\-]*")
# A telephone token needs structure — a parenthesized country/area group or
# a hyphenated digit group.  A bare space-separated digit run must NOT match:
# chart axis sequences (``90 94 98 02 06``) and year ranges in terse captions
# (``Global Bond Supply 2010 2026``) would otherwise read as phone numbers.
_PDF_CONTACT_PHONE_RE = re.compile(
    r"\(\+?\d[\d\s/-]{0,14}\)[\s\d/-]*|\d[\d\s/]*-[\d\s/-]*\d"
)


def _is_preserved_nonlinguistic_text(text: str) -> bool:
    """Return True for one or more identifiers with only footnote markers.

    PDF extraction can join multiple URL-only footnotes into one visual text
    element, for example ``<sup>2</sup> URL <sup>3</sup> URL``.  Requiring one
    identifier to occupy the entire element sends that block to the model even
    though every meaningful byte must be preserved.
    """
    plain = _collapsed_plain_text(text)
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
    source_plain = _collapsed_plain_text(source_text or "")
    translated_plain = _collapsed_plain_text(translated_text or "")
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
    plain = _collapsed_plain_text(text)
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
    plain = _collapsed_plain_text(text)
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
    plain = _collapsed_plain_text(text)
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
    plain = _collapsed_plain_text(text or "")
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


def _looks_like_pdf_contact_line(text: str) -> bool:
    """Detect an analyst byline: proper-name words plus phone digits.

    ``Nikolaos Panigirtzoglou AC (44-20) 7134-7815 <e-mail>`` repeats as a
    running header on every page of a research series.  The e-mail is already
    protected as an identifier; the residue is a person or organization name
    plus a telephone number, which must stay verbatim.  Any lower-case prose
    word (beyond name connectors and organization suffixes) disqualifies the
    line, and without a phone-number token this rule never fires.
    """
    plain = _collapsed_plain_text(text or "")
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
    plain = _collapsed_plain_text(text or "")
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
        elem.get("type") != ElementType.TEXT
        or elem.get("table_hint")
        or elem.get("entity_directory_hint")
        or elem.get("skip_translate_reason")
        or elem.get("non_horizontal")
        or float(elem.get("fontsize", 0.0)) < 15.0
    ):
        return False

    plain = _collapsed_plain_text(elem.get("content", "")).rstrip(".;:")
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


def _pdf_element_requires_translation(elem: dict) -> bool:
    """Return whether a text element needs an API-produced translation."""
    if elem.get("type") != ElementType.TEXT or elem.get("skip_translate_reason"):
        return False
    text = elem.get("content", "")
    if _is_disclaimer_block(text) or not _looks_like_translatable_english(text):
        return False
    plain = _collapsed_plain_text(text)
    if (
        elem.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED}
        and not elem.get("bold")
        and not elem.get("single_line_heading")
        and _looks_like_compact_pdf_identifier(plain)
    ):
        return False
    if (
        elem.get("entity_directory_hint")
        and elem.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED}
        and not elem.get("bold")
    ):
        return False
    # Unlike the compact-identifier gate above, these deliberately ignore
    # ``single_line_heading`` and ``table_hint``: the measured deal-table rows
    # and analyst bylines carried those flags, and a row with no lower-case
    # prose cannot be a real heading the flags are meant to protect.  A
    # full-date element leaves the queue because its translation is
    # deterministic (applied by ``_translate_pdf_deterministic_labels``).
    if elem.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED} and not elem.get("bold"):
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
        and elem.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED}
        and not elem.get("bold")
        and not elem.get("single_line_heading")
        and _looks_like_pdf_proper_name_label(text)
    ):
        return False
    if (
        not elem.get("table_hint")
        and elem.get("layout_class") in {LayoutClass.TABLE, LayoutClass.SCATTERED}
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
