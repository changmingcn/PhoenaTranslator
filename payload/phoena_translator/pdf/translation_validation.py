"""Decide whether a model translation is acceptable or must be retried:
untranslated English, lost number anchors, refusals, TOC rows,
reference entries, length expansion and batch envelopes."""

from __future__ import annotations

import re
import math
from phoena_translator.pdf.translation_targets import (
    _looks_like_pdf_contact_line,
    _looks_like_pdf_structured_identifier_row,
    _looks_like_pdf_translatable_short_label,
    _looks_like_translatable_english,
    _pdf_element_requires_translation,
    _pdf_translation_language_views,
)
from phoena_translator.pdf.batch_codec import (
    PDF_BATCH_SEPARATOR,
    _parse_pdf_batch_segment_envelope,
    _pdf_batch_segment_records_from_source,
)
from phoena_translator.pdf.patterns import (
    PDF_TRANSLATABLE_CITATION_LABELS,
    _PDF_PROPER_NAME_CONNECTORS,
    _PRESERVED_IDENTIFIER_RE,
)
from phoena_translator.pdf.toc import _pdf_toc_entries, _pdf_toc_trailing_text
from phoena_translator.pdf.inline_markup import (
    _normalize_pdf_translation,
    _pdf_inline_markup_preserved,
    _pdf_inline_math_fragments_preserved,
)
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.semantic_references import (
    _looks_like_pdf_reference_entry_lead,
)
from phoena_translator.pdf.errors import PDFTranslationIntegrityError


# A citation's work title is the quoted run inside a bibliography entry.  Its
# presence separates "this entry has something to translate" from "this entry
# is only authors, initials and a year", where returning the source unchanged
# is the correct translation rather than a failure.
_PDF_QUOTED_TITLE_RE = re.compile(r"[“\"]([^”\"]{2,})[”\"]?")
PDF_MAX_TRANSLATION_EXPANSION_RATIO = 1.40
PDF_MAX_TRANSLATION_EXPANSION_SLACK = 64
_PDF_MIXED_TRANSLATION_ENGLISH_RUN_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z'’\-]*\b(?:\s+|[,;:()–—-]+)){4,}"
    r"\b[A-Za-z][A-Za-z'’\-]*\b"
)
_PDF_WORK_TITLE_QUOTE_CHARS = "'\"‘’“”"
_PDF_REFERENCE_TRANSLATION_YEAR_RE = re.compile(
    r"(?<!\d)(?:18\d{2}|19\d{2}|20[0-2]\d)[a-z]?(?!\d)",
    re.IGNORECASE,
)
_PDF_REFUSAL_META_RE = re.compile(
    r"^(?:好的|当然|明白了)[，,、]?\s*(?:以下|这)是"
    r"|^以下是[^。]{0,12}(?:译文|翻译)"
    r"|^译文[:：]"
    r"|^(?:抱歉|对不起)[，,]?\s*(?:我)?(?:无法|不能)"
    r"|^(?:我)?(?:无法|不能)(?:翻译|提供|处理|完成)"
    r"|作为一?[个個]?(?:AI|人工智能|大?语言模型)"
)


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def _pdf_semantic_number_anchor_preserved(
    source_number: str,
    translated_text: str,
) -> bool:
    if source_number in translated_text:
        return True

    # English decades are conventionally localized rather than copied as an
    # exact digit token: ``the 1960s`` -> ``20世纪60年代``.  This is not the
    # neighboring-row drift that the number-anchor gate is intended to catch.
    decade_match = re.fullmatch(r"((?:18|19|20)\d0)[sS]", source_number)
    if decade_match:
        year = int(decade_match.group(1))
        century = year // 100 + 1
        decade = year % 100
        localized = {
            f"{century}世纪{decade}年代",
            f"{century}世紀{decade}年代",
            f"{year}年代",
        }
        return any(candidate in translated_text for candidate in localized)
    return False


def _pdf_number_anchor_plain_text(text: str) -> str:
    """Flatten text without joining base numbers to superscript markers.

    ``Table 2.<sup>32</sup>`` must yield separate ``2`` and ``32`` tokens,
    never the synthetic decimal ``2.32``.  Superscript integrity is enforced
    independently; this view exists only for semantic number comparisons.
    """
    separated = re.sub(
        r"(?is)<sup(?:\s[^>]*)?>(.*?)</sup>",
        lambda match: " " + _plain_text(match.group(1)) + " ",
        text or "",
    )
    return _collapsed_plain_text(separated)


def _pdf_quoted_title_word_count(text: str) -> int:
    """Count the words inside a citation's quoted work title."""
    match = _PDF_QUOTED_TITLE_RE.search(text or "")
    if not match:
        return 0
    return len(re.findall(r"[A-Za-z][A-Za-z’'-]*", match.group(1)))


def _pdf_citation_apparatus_only(source_language: str) -> bool:
    """Return whether a fragment carries no translatable prose at all.

    A bibliography wraps onto several native PDF lines, so a line is often
    just authors, initials and a year -- ``Khandani, Amir E., and Andrew W.
    Lo. 2007.``  Prompt rule 6 keeps exactly those verbatim, so an identical
    response is the CORRECT translation, yet the word-overlap verdict below
    read it as an untranslated echo.  Measured on one bibliography, that
    rejected 5 of 14 correct translations and spent four retries plus backoff
    on each; the two reference pages dominated an 85-minute run.

    A quoted title of two or more words IS translatable, so entries carrying
    one keep going through the normal ladder -- including the real catch in
    that sample, whose title came back untranslated.
    """
    if not _pdf_english_run_looks_like_proper_name(source_language):
        return False
    return _pdf_quoted_title_word_count(source_language) < 2


def _short_translation_needs_retry(source_text: str, translated_text: str) -> bool:
    """Detect short blocks that came back untranslated."""
    if not _looks_like_translatable_english(source_text):
        return False

    translated_plain = _collapsed_plain_text(translated_text)
    source_language, translated_language, identifiers_preserved = (
        _pdf_translation_language_views(source_text, translated_text)
    )
    if not identifiers_preserved:
        return True
    if not _looks_like_translatable_english(source_language):
        return False
    if _looks_like_pdf_structured_identifier_row(source_language):
        # ``01/16/26 QTSII 2026-1A A2``-style data rows: the source text is
        # the correct translation, so an exact echo must not start the retry
        # ladder — batch segments containing one such row previously failed
        # the whole batch into per-element ladders.
        return False
    if _looks_like_pdf_contact_line(source_text):
        # Analyst bylines (name + phone + protected e-mail) repeat on every
        # page of a research series and must stay verbatim.
        return False

    source_words = re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*", source_language
    )
    if (
        not _looks_like_pdf_translatable_short_label(source_language)
        and (
            len(source_language) <= 12
            or (len(source_words) <= 2 and len(source_language) <= 18)
        )
    ):
        return False

    if not translated_plain or not translated_language:
        return True
    if _has_chinese(translated_language):
        # A Chinese refusal or meta-preamble ("抱歉，我无法翻译此内容",
        # "以下是译文：…") contains Chinese and, for short digit-free
        # sources, violates none of the length/anchor checks below — yet it
        # is not a translation.  Force the retry ladder instead.
        if _PDF_REFUSAL_META_RE.search(translated_plain):
            return True
        input_len = len(source_language)
        output_len = len(translated_language)
        source_alpha_chars = len(re.findall(r"[A-Za-z]", source_language))
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", translated_language))

        # A medium prose cell mapped to the next row's short term still has
        # Chinese, but is grossly too small to be its own translation.  This
        # exact failure escaped the old yes/no Chinese gate on ASIC page 81.
        if (
            source_alpha_chars >= 80
            and chinese_chars < max(8, math.ceil(source_alpha_chars * 0.07))
        ):
            return True

        # Conversely, a short term mapped to a neighboring definition is
        # implausibly expansive.  Keep a generous floor for legitimate titles.
        if (
            input_len <= 60
            and len(source_words) <= 8
            and output_len > max(24, math.ceil(input_len * 1.45))
        ):
            return True

        source_anchor_plain = _pdf_number_anchor_plain_text(
            _PRESERVED_IDENTIFIER_RE.sub(" ", source_text)
        )
        translated_anchor_plain = _pdf_number_anchor_plain_text(
            _PRESERVED_IDENTIFIER_RE.sub(" ", translated_text)
        )
        source_numbers = re.findall(
            r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)+(?:[A-Za-z])?|\d{2,}[A-Za-z]?)(?![A-Za-z0-9])",
            source_anchor_plain,
        )
        if any(
            not _pdf_semantic_number_anchor_preserved(
                number,
                translated_anchor_plain,
            )
            for number in source_numbers
        ):
            return True
        return False
    if _pdf_citation_apparatus_only(source_language):
        return False

    if translated_language.casefold() == source_language.casefold():
        return True

    src_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*", source_language.casefold()
    ))
    out_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*", translated_language.casefold()
    ))
    if src_words and out_words:
        overlap = len(src_words & out_words) / max(len(src_words), 1)
        if overlap >= 0.6:
            return True

    return False


def _pdf_english_run_looks_like_proper_name(run: str) -> bool:
    """Return whether a long English run is only names or an entity title.

    A translated academic paragraph may legitimately retain author lists,
    journal names, and institutions such as ``International Organization of
    Securities Commissions``.  Those runs are title-cased apart from narrow
    connectors.  Sentence-case work titles contain at least one substantive
    lower-case word and must not receive this exemption.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", run or "")
    if len(words) < 5:
        return False
    substantive_words = 0
    for word in words:
        normalized = word.strip("-'’")
        if normalized.casefold() in _PDF_PROPER_NAME_CONNECTORS:
            continue
        substantive_words += 1
        if not (
            len(normalized) == 1
            or normalized.isupper()
            or normalized[:1].isupper()
        ):
            return False
    return substantive_words >= 2


def _pdf_mixed_translation_has_untranslated_english_run(
    source_text: str,
    translated_text: str,
) -> bool:
    """Detect a residual English sentence/work title inside Chinese output.

    Aggregate Chinese counts can pass when a model translates the surrounding
    footnote prose but silently leaves an article title in English.  Require a
    retry for any uninterrupted five-word English run, while preserving pure
    proper-name, institution, and journal-name runs.  A quoted title remains
    translatable even when every word is title-cased.
    """
    source_language, translated_language, identifiers_preserved = (
        _pdf_translation_language_views(source_text, translated_text)
    )
    if (
        not identifiers_preserved
        or not _has_chinese(translated_language)
        or not _looks_like_translatable_english(source_language)
    ):
        return False

    for match in _PDF_MIXED_TRANSLATION_ENGLISH_RUN_RE.finditer(
        translated_language
    ):
        run = match.group(0).strip()
        if not _pdf_english_run_looks_like_proper_name(run):
            return True

        # Quotation marks are strong work-title evidence.  Do not let a
        # title-cased article name evade the residue gate merely because it
        # resembles an institution name token by token.
        surrounding = translated_language[
            max(0, match.start() - 2):min(len(translated_language), match.end() + 2)
        ]
        if any(char in surrounding for char in _PDF_WORK_TITLE_QUOTE_CHARS):
            return True
    return False


def _split_pdf_reference_entry_translation_parts(
    text: str,
) -> tuple[str, str] | None:
    """Split a complete citation into immutable author/year and translatable tail.

    Bibliographic identity depends on the author/year lead.  Translate only
    the material after that lead so a model cannot create Chinese author
    transliterations that happen to satisfy the generic Chinese-count gate.
    """
    normalized = _normalize_pdf_translation(text or "")
    plain = _collapsed_plain_text(normalized)
    if not _looks_like_pdf_reference_entry_lead(plain):
        return None
    year_match = _PDF_REFERENCE_TRANSLATION_YEAR_RE.search(normalized[:240])
    if not year_match:
        return None
    cursor = year_match.end()
    # Include ``)``/punctuation and following whitespace in the immutable
    # prefix: ``Authors (2000). Title`` and ``Authors, 2000, Title`` then
    # reconstruct byte-for-byte around the translated tail.
    while cursor < len(normalized) and normalized[cursor] in ")],.;: \t\r\n":
        cursor += 1
    prefix = normalized[:cursor]
    tail = normalized[cursor:]
    if not prefix or not _looks_like_translatable_english(tail):
        return None
    return prefix, tail


def _pdf_reference_entry_translation_needs_retry(
    source_text: str,
    translated_text: str,
) -> bool:
    """Validate a hybrid citation translation at its semantic boundary."""
    parts = _split_pdf_reference_entry_translation_parts(source_text)
    if parts is None:
        return _translation_part_needs_retry(source_text, translated_text)
    source_prefix, source_tail = parts
    candidate = _normalize_pdf_translation(translated_text or "")
    if not candidate.startswith(source_prefix):
        return True
    translated_tail = candidate[len(source_prefix):].lstrip()
    if not translated_tail or translated_tail.casefold() == source_tail.casefold():
        return True

    # URLs and DOI strings are searchable bibliographic identifiers.  Number
    # anchors and these opaque identifiers must remain exact.  Use a reference-
    # specific Chinese threshold: a short translated title followed by a long
    # URL must not be rejected by the ordinary long-prose minimum.
    source_identifiers = re.findall(
        r"(?i)(?:https?://\S+|www\.\S+|doi\s*:\s*\S+)",
        source_tail,
    )
    if any(identifier not in translated_tail for identifier in source_identifiers):
        return True
    source_numbers = re.findall(
        r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)+(?:[A-Za-z])?|\d{2,}[A-Za-z]?)(?![A-Za-z0-9])",
        _pdf_number_anchor_plain_text(source_tail),
    )
    translated_anchor_plain = _pdf_number_anchor_plain_text(translated_tail)
    if any(
        not _pdf_semantic_number_anchor_preserved(
            number,
            translated_anchor_plain,
        )
        for number in source_numbers
    ):
        return True

    source_without_ids = source_tail
    for identifier in source_identifiers:
        source_without_ids = source_without_ids.replace(identifier, " ")
    source_alpha_chars = len(re.findall(r"[A-Za-z]", source_without_ids))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", translated_tail))
    minimum_chinese = max(2, min(12, math.ceil(source_alpha_chars * 0.05)))
    if chinese_chars < minimum_chinese:
        return True

    # The work title is the first semantic material after author/year.  This
    # prevents a model from leaving that title English and translating only a
    # trailing ``Working Paper`` label.
    early_window = translated_tail[:max(32, math.ceil(len(translated_tail) * 0.60))]
    if len(re.findall(r"[\u4e00-\u9fff]", early_window)) < 2:
        return True

    source_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*",
        source_without_ids.casefold(),
    ))
    translated_without_ids = translated_tail
    for identifier in source_identifiers:
        translated_without_ids = translated_without_ids.replace(identifier, " ")
    translated_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*",
        translated_without_ids.casefold(),
    ))
    if source_words and translated_words:
        overlap = len(source_words & translated_words) / len(source_words)
        if overlap >= 0.90 and chinese_chars < max(6, minimum_chinese * 2):
            return True
    return False


def _pdf_element_translation_needs_retry(
    elem: dict,
    source_text: str,
    translated_text: str,
) -> bool:
    """Apply element-aware integrity checks after generic translation checks."""
    if elem.get("reference_entry_hint"):
        return _pdf_reference_entry_translation_needs_retry(
            source_text,
            translated_text,
        )
    if _translation_part_needs_retry(source_text, translated_text):
        return True
    if not _pdf_inline_math_fragments_preserved(elem, translated_text):
        return True
    if elem.get("vector_ocr"):
        source_plain = _collapsed_plain_text(source_text)
        translated_plain = _collapsed_plain_text(translated_text)
        if (
            re.search(r"(?i)\bdriver\s+reviews?\b", source_plain)
            and re.search(r"司机|驾驶员", translated_plain)
        ):
            return True
        expected_paragraphs = len([
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ])
        if expected_paragraphs > 1:
            normalized = _normalize_pdf_translation(translated_text)
            blank_line_parts = [
                part for part in re.split(r"\n\s*\n+", normalized)
                if part.strip()
            ]
            line_parts = [line for line in normalized.splitlines() if line.strip()]
            if (
                len(blank_line_parts) != expected_paragraphs
                and len(line_parts) != expected_paragraphs
            ):
                return True
    if elem.get("glossary_term_hint") or elem.get("glossary_definition_hint"):
        translated_plain = _collapsed_plain_text(translated_text)
        return not _has_chinese(translated_plain)
    return False


def _pdf_toc_translation_needs_retry(
    source_text: str,
    translated_text: str,
) -> bool:
    """Validate dotted TOC entries by labels and page references, not dot count.

    Hundreds of leader dots can make a small TOC block look like a long prose
    paragraph.  A fixed 20-Chinese-character threshold then rejects a complete
    translation such as three short labels containing 19 Chinese characters.
    Validate each label independently and require the ordered page numbers to
    remain exact instead.
    """
    source_entries = _pdf_toc_entries(source_text)
    translated_entries = _pdf_toc_entries(translated_text)
    if not source_entries or len(source_entries) != len(translated_entries):
        return True
    if _pdf_toc_trailing_text(translated_text).strip(" .…·"):
        return True
    if [page for _, page in source_entries] != [page for _, page in translated_entries]:
        return True

    for (source_label, _), (translated_label, _) in zip(source_entries, translated_entries):
        if not _looks_like_translatable_english(source_label):
            continue
        translated_plain = _collapsed_plain_text(translated_label)
        if not translated_plain:
            return True
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", translated_plain))
        source_alpha_chars = len(re.findall(r"[A-Za-z]", _plain_text(source_label)))
        minimum_cn_chars = max(1, math.ceil(source_alpha_chars * 0.12))
        if cn_chars < minimum_cn_chars:
            return True
        if translated_plain.casefold() == _collapsed_plain_text(source_label).casefold():
            return True
        source_words = set(re.findall(
            r"[A-Za-z][A-Za-z0-9'/-]*",
            _plain_text(source_label).casefold(),
        ))
        output_words = set(re.findall(
            r"[A-Za-z][A-Za-z0-9'/-]*",
            translated_plain.casefold(),
        ))
        if source_words and output_words:
            overlap = len(source_words & output_words) / len(source_words)
            if overlap >= 0.7:
                return True
    return False


def _pdf_has_untranslated_citation_label(
    source_text: str,
    translated_text: str,
) -> bool:
    return any(
        pattern.search(source_text or "")
        and pattern.search(translated_text or "")
        for pattern, _ in PDF_TRANSLATABLE_CITATION_LABELS
    )


def _translation_part_needs_retry(source_text: str, translated_text: str) -> bool:
    if not _looks_like_translatable_english(source_text):
        return False

    translated_plain = _collapsed_plain_text(translated_text)
    if not translated_plain:
        return True
    if _pdf_has_untranslated_citation_label(source_text, translated_text):
        return True
    source_language, translated_language, identifiers_preserved = (
        _pdf_translation_language_views(source_text, translated_text)
    )
    if not identifiers_preserved:
        return True
    if not _looks_like_translatable_english(source_language):
        return False
    if _looks_like_pdf_toc_leader_block(source_text):
        return _pdf_toc_translation_needs_retry(source_text, translated_text)
    if _pdf_mixed_translation_has_untranslated_english_run(
        source_text,
        translated_text,
    ):
        return True

    input_len = len(source_language)
    output_len = len(translated_language)
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', translated_language))
    src_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*", source_language.casefold()
    ))
    out_words = set(re.findall(
        r"[A-Za-z][A-Za-z0-9'/-]*", translated_language.casefold()
    ))
    overlap = len(src_words & out_words) / max(len(src_words), 1) if src_words and out_words else 0.0

    # English-to-Chinese prose normally contracts in character count.  A
    # large expansion is strong evidence that a batch segment copied text from
    # its neighbor.  Keep generous fixed slack for short labels and markup,
    # but reject the candidate instead of truncating it into a plausible-looking
    # fragment that can pass the remaining integrity checks.
    max_output_len = max(
        math.ceil(input_len * PDF_MAX_TRANSLATION_EXPANSION_RATIO),
        input_len + PDF_MAX_TRANSLATION_EXPANSION_SLACK,
    )
    if _has_chinese(translated_language) and output_len > max_output_len:
        return True

    if input_len < 200:
        return _short_translation_needs_retry(source_text, translated_text)
    if cn_chars < max(20, int(input_len * 0.06)):
        return True
    if translated_language.casefold() == source_language.casefold():
        return True
    if overlap >= 0.7 and cn_chars < max(24, int(output_len * 0.15)):
        return True
    if output_len / max(input_len, 1) < 0.12:
        return True
    return False


def _pdf_batch_translation_candidate_is_valid(
    source_text: str,
    translated_text: str,
) -> bool:
    """Validate a PDF batch at its real element boundaries.

    Aggregate Chinese-character thresholds are misleading for batches that
    contain names, citations, symbols, or numeric table cells.  The page-level
    integrity gate repeats this check before anything is cached.
    """
    if PDF_BATCH_SEPARATOR not in source_text:
        return False
    segment_records = _pdf_batch_segment_records_from_source(source_text)
    if segment_records:
        translated_parts = _parse_pdf_batch_segment_envelope(
            translated_text,
            segment_records,
        )
        if translated_parts is None:
            return False
        return all(
            not _translation_part_needs_retry(record["source"], candidate)
            and _pdf_inline_markup_preserved(record["source"], candidate)
            for record, candidate in zip(segment_records, translated_parts)
        )
    source_parts = [part.strip() for part in source_text.split(PDF_BATCH_SEPARATOR)]
    translated_parts = [part.strip() for part in translated_text.split(PDF_BATCH_SEPARATOR)]
    if len(source_parts) != len(translated_parts) or not source_parts:
        return False
    return all(
        candidate
        and not _translation_part_needs_retry(source, candidate)
        and _pdf_inline_markup_preserved(source, candidate)
        for source, candidate in zip(source_parts, translated_parts)
    )


def _validate_pdf_page_translations(
    page_number: int,
    elements: list[dict],
    translations: dict,
    *,
    allow_missing: bool = False,
) -> None:
    """Fail closed unless each present API result is safe.

    ``allow_missing`` is used only while migrating a compatible cache.  It
    lets the caller retain already verified element translations and request
    the newly exposed elements incrementally; malformed translations still
    fail closed and invalidate the page seed.
    """
    normalized = {str(key): value for key, value in (translations or {}).items()}
    issues = []
    for elem_index, elem in enumerate(elements or []):
        if not _pdf_element_requires_translation(elem):
            continue
        source = elem.get("rich_content") or elem.get("content", "")
        candidate = normalized.get(str(elem_index))
        if not isinstance(candidate, str) or not candidate.strip():
            if allow_missing:
                continue
            issues.append({"element": elem_index + 1, "reason": "missing-translation"})
            continue
        if _pdf_element_translation_needs_retry(elem, source, candidate):
            issues.append({"element": elem_index + 1, "reason": "untranslated-or-incomplete"})
            continue
        if not _pdf_inline_markup_preserved(source, candidate):
            issues.append({"element": elem_index + 1, "reason": "inline-markup-mismatch"})
    if issues:
        raise PDFTranslationIntegrityError(page_number, issues, translations)


def _looks_like_pdf_toc_leader_block(text: str) -> bool:
    entries = _pdf_toc_entries(text)
    return bool(
        entries
        and any(_looks_like_translatable_english(label) for label, _ in entries)
        and not _looks_like_translatable_english(_pdf_toc_trailing_text(text))
    )
