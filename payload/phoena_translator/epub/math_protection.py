"""Locate XHTML MathML, LaTeX and Unicode-math spans that must stay verbatim."""

from __future__ import annotations

import html
import re

from ..math_text import (
    MATH_FUNCTION_WORDS,
    is_math_block,
    is_unicode_math_symbol,
    iter_unicode_math_symbol_ranges,
)
from .xhtml import visible_text_ranges


def looks_like_latex_dollar_content(content: str) -> bool:
    stripped = (content or "").strip()
    if not stripped or len(stripped) > 400:
        return False
    # A pair of currency amounts is not a LaTeX delimiter pair. In particular,
    # a hyphen in ordinary prose ("$30 billion ... lead-up ... $110 billion")
    # must not protect the entire intervening sentence from translation.
    if re.match(
        r"\d[\d,.]*(?:\s*(?:thousand|million|billion|trillion|dollars?|USD|US))\b",
        stripped,
        re.IGNORECASE,
    ):
        return False
    prose_words = re.findall(r"[A-Za-z]+", stripped)
    if not re.search(r"\\[A-Za-z]+|[\^_{}]", stripped):
        long_words = [
            word for word in prose_words
            if len(word) > 3 and word.lower() not in MATH_FUNCTION_WORDS
        ]
        if len(prose_words) > 8 or len(long_words) >= 2:
            return False
    if re.match(r"\d", stripped) and not re.search(r"[\\^_{}=<>+*/-]", stripped):
        return False
    if re.search(r"[\\^_{}=<>+*/-]", stripped):
        return True
    words = re.findall(r"[A-Za-z]+", stripped)
    return bool(words) and len(words) <= 3 and all(
        len(word) <= 3 or word.lower() in MATH_FUNCTION_WORDS for word in words
    )


def find_xhtml_math_intervals(text: str) -> list[tuple[int, int, str]]:
    """Locate explicit math fragments while retaining their exact source text."""
    if not text:
        return []
    candidates: list[tuple[int, int, int, str]] = []
    mathml_pattern = re.compile(
        r"(?is)<(?:[A-Za-z_][\w.-]*:)?math\b[^>]*>.*?</(?:[A-Za-z_][\w.-]*:)?math\s*>"
    )
    for match in mathml_pattern.finditer(text):
        candidates.append((match.start(), match.end(), 0, "mathml"))

    latex_patterns = (
        (re.compile(r"(?s)\\begin\{([A-Za-z*]+)\}.*?\\end\{\1\}"), "latex_environment"),
        (re.compile(r"(?s)\$\$.+?\$\$"), "latex_display_dollar"),
        (re.compile(r"(?s)\\\[.+?\\\]"), "latex_display_bracket"),
        (re.compile(r"(?s)\\\(.+?\\\)"), "latex_inline_paren"),
    )
    math_entity_pattern = re.compile(
        r"(?i)&(?:sum|prod|int|radic|part|nabla|infin|asymp|ne|le|ge|plusmn|"
        r"times|divide|isin|notin|sub|sup|sube|supe|forall|exist|empty|rarr|larr|"
        r"harr|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|xi|pi|rho|sigma|"
        r"tau|phi|chi|psi|omega);"
    )

    for range_start, range_end in visible_text_ranges(text):
        segment = text[range_start:range_end]
        for pattern, reason in latex_patterns:
            for match in pattern.finditer(segment):
                candidates.append(
                    (
                        range_start + match.start(),
                        range_start + match.end(),
                        1,
                        reason,
                    )
                )
        # Scan candidate openings individually. A rejected currency pair must
        # not consume the opening delimiter of a following genuine formula.
        dollar_positions = [
            match.start() for match in re.finditer(r"(?<![\\$])\$(?!\$)", segment)
        ]
        dollar_index = 0
        while dollar_index + 1 < len(dollar_positions):
            opening, closing = dollar_positions[dollar_index:dollar_index + 2]
            content = segment[opening + 1:closing]
            if looks_like_latex_dollar_content(content):
                candidates.append(
                    (
                        range_start + opening,
                        range_start + closing + 1,
                        2,
                        "latex_inline_dollar",
                    )
                )
                dollar_index += 2
            else:
                dollar_index += 1
        if segment.strip() and is_math_block(html.unescape(segment)):
            candidates.append((range_start, range_end, 3, "math_text_node"))
        for symbol_start, symbol_end in iter_unicode_math_symbol_ranges(segment):
            source_slice = segment[symbol_start:symbol_end]
            if source_slice and all(is_unicode_math_symbol(char) for char in source_slice):
                candidates.append(
                    (
                        range_start + symbol_start,
                        range_start + symbol_end,
                        4,
                        "unicode_math_symbol",
                    )
                )
        for match in math_entity_pattern.finditer(segment):
            candidates.append(
                (
                    range_start + match.start(),
                    range_start + match.end(),
                    4,
                    "html_math_entity",
                )
            )

    selected: list[tuple[int, int, str]] = []
    for start, end, _priority, reason in sorted(
        candidates, key=lambda item: (item[2], item[0], -(item[1] - item[0]))
    ):
        if start >= end:
            continue
        if any(
            start < chosen_end and end > chosen_start
            for chosen_start, chosen_end, _ in selected
        ):
            continue
        selected.append((start, end, reason))
    return sorted(selected, key=lambda item: item[0])


__all__ = [
    "find_xhtml_math_intervals",
    "looks_like_latex_dollar_content",
]
