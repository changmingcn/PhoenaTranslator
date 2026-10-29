"""Find English runs left untranslated in delivered pages."""

from __future__ import annotations

import re
from phoena_translator.pdf.translation_validation import (
    _has_chinese,
    _pdf_english_run_looks_like_proper_name,
)
from phoena_translator.pdf.patterns import _PDF_PROPER_NAME_CONNECTORS
from phoena_translator.pdf.translation_targets import (
    _looks_like_translatable_english,
    _pdf_element_requires_translation,
    _pdf_translation_language_views,
)
from phoena_translator.pdf.model import ElementType
from phoena_translator.math_text import MATH_FUNCTION_WORDS


_PDF_UNTRANSLATED_LEAK_WORD_RE = re.compile(
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*"
)
_PDF_UNTRANSLATED_LEAK_RUN_RE = re.compile(
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*"
    r"(?:[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]+"
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*)+"
)
_PDF_UNTRANSLATED_LEAK_LONE_WORD_RE = re.compile(
    r"[a-z]{3,}(?:['\u2019][a-z]+)*"
)
_PDF_UNTRANSLATED_LEAK_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
_PDF_UNTRANSLATED_LEAK_UNIT_WORDS = {
    "bps", "ppm", "pts", "bbl", "mmbtu", "kwh", "mwh", "gwh", "twh",
}
_PDF_UNTRANSLATED_LEAK_SYMBOL_WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega", "rho", "vega",
}
_PDF_UNTRANSLATED_LEAK_NEIGHBOR_BEFORE_RE = re.compile(
    r"[A-Za-z]{3,}[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]*$"
)
_PDF_UNTRANSLATED_LEAK_NEIGHBOR_AFTER_RE = re.compile(
    r"^[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]*[A-Za-z]{3,}"
)


def _pdf_untranslated_english_leak_runs(
    source_text: str,
    translated_text: str,
) -> list[str]:
    """Return English prose that survived untranslated into final output.

    Generalized capture of the wrapped-inline-math defect family: a sentence
    fragment sent to the model in isolation can come back with real prose
    words (for example ``other``) still in English inside otherwise-final
    output.  Two rules:

    1. any run of >=2 adjacent lowercase/Title-case ASCII words of >=3
       letters; and
    2. a LONE lowercase word of >=3 letters wedged against CJK text in an
       otherwise-translated (Chinese-bearing) output, when the same word
       exists in the source.

    Excluded: math-function tokens, units, Greek-letter names, all-caps or
    mixed-caps identifiers (the word shapes reject them), hyphen/apostrophe
    compounds such as ``E-mini``, preserved URL/e-mail/DOI identifiers
    (stripped by the language view), Title-case proper-name runs already
    present in the source, and elements whose final text intentionally equals
    the source.  Advisory only; callers never raise.
    """
    source_language, translated_language, _identifiers_preserved = (
        _pdf_translation_language_views(source_text, translated_text)
    )
    if not translated_language:
        return []
    if not _looks_like_translatable_english(source_language):
        return []
    if translated_language.casefold() == source_language.casefold():
        # Element-level source fallback: intentionally kept as exact source.
        return []
    source_casefold = source_language.casefold()
    leaks = []
    for match in _PDF_UNTRANSLATED_LEAK_RUN_RE.finditer(translated_language):
        run = match.group(0).strip()
        words = _PDF_UNTRANSLATED_LEAK_WORD_RE.findall(run)
        substantive = [
            word for word in words
            if word.lower() not in MATH_FUNCTION_WORDS
            and word.lower() not in _PDF_UNTRANSLATED_LEAK_UNIT_WORDS
        ]
        if len(substantive) < 2:
            continue
        title_case_run = all(
            word[:1].isupper() or word.casefold() in _PDF_PROPER_NAME_CONNECTORS
            for word in words
        )
        if title_case_run and run.casefold() in source_casefold:
            continue
        if _pdf_english_run_looks_like_proper_name(run):
            continue
        leaks.append(run)

    if _has_chinese(translated_language):
        for match in _PDF_UNTRANSLATED_LEAK_LONE_WORD_RE.finditer(
            translated_language
        ):
            word = match.group(0)
            if (
                word in MATH_FUNCTION_WORDS
                or word in _PDF_UNTRANSLATED_LEAK_UNIT_WORDS
                or word in _PDF_UNTRANSLATED_LEAK_SYMBOL_WORDS
            ):
                continue
            start, end = match.span()
            prev_char = translated_language[start - 1] if start else ""
            next_char = (
                translated_language[end]
                if end < len(translated_language)
                else ""
            )
            # Part of a larger token or compound (E-mini, traders').
            if prev_char and (
                prev_char.isalnum() or prev_char in "-'\u2019"
            ):
                continue
            if next_char and (
                next_char.isalnum() or next_char in "-'\u2019"
            ):
                continue
            window_before = translated_language[max(0, start - 6):start]
            window_after = translated_language[end:end + 6]
            if not (
                _PDF_UNTRANSLATED_LEAK_CJK_RE.search(window_before)
                or _PDF_UNTRANSLATED_LEAK_CJK_RE.search(window_after)
            ):
                continue
            # A neighboring English word means the multi-word rule owns it.
            if _PDF_UNTRANSLATED_LEAK_NEIGHBOR_BEFORE_RE.search(
                window_before
            ) or _PDF_UNTRANSLATED_LEAK_NEIGHBOR_AFTER_RE.search(
                window_after
            ):
                continue
            if not re.search(
                r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                source_language,
                re.IGNORECASE,
            ):
                continue
            leaks.append(word)

    seen = set()
    unique_leaks = []
    for leak in leaks:
        key = leak.casefold()
        if key not in seen:
            seen.add(key)
            unique_leaks.append(leak)
    return unique_leaks


def _pdf_collect_untranslated_english_leaks(
    page_number: int,
    elements: list[dict],
    translations: dict,
    max_entries: int = 20,
) -> list[dict]:
    """Audit one completed page for translatable prose left in English.

    Scans each text element's final rendered text (the translation that will
    be drawn).  Skips elements intentionally kept as source: any
    ``skip_translate_reason``, formula images (never text), elements the
    translation-queue predicate exempts, and element-level source fallbacks
    (final text equals source).  Fail-open by design: any internal error
    yields the findings collected so far and delivery never blocks on this.
    """
    entries: list[dict] = []
    try:
        for index, elem in enumerate(elements or []):
            if len(entries) >= max_entries:
                break
            if elem.get("type") != ElementType.TEXT or elem.get("skip_translate_reason"):
                continue
            final_text = (translations or {}).get(index)
            if final_text is None:
                final_text = (translations or {}).get(str(index))
            if not final_text or not str(final_text).strip():
                continue
            if not _pdf_element_requires_translation(elem):
                continue
            runs = _pdf_untranslated_english_leak_runs(
                elem.get("content", ""),
                str(final_text),
            )
            for run in runs[: max(1, max_entries - len(entries))]:
                entries.append({
                    "page": int(page_number),
                    "element": int(index),
                    "text": run,
                    "layout_class": str(elem.get("layout_class") or ""),
                })
    except Exception:
        return entries
    return entries
