"""Deterministic math-text classification shared by EPUB and PDF code."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterator


MATH_SYMBOLS = frozenset(
    "∑∫∏√∂∇∆∞≈≠≤≥±∓÷×∈∉⊂⊃⊆⊇∧∨∀∃∅→←↔⟹⟸⇒⇐"
    "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
)
MATH_LETTERLIKE_SYMBOLS = frozenset("ℂℍℕℙℚℝℤℵℏℓ℘ℑℜ℧")
MATH_FUNCTION_WORDS = frozenset(
    {
        "arg",
        "cos",
        "cov",
        "det",
        "exp",
        "inf",
        "int",
        "lim",
        "ln",
        "log",
        "max",
        "min",
        "prod",
        "sin",
        "sqrt",
        "sum",
        "sup",
        "tan",
        "var",
    }
)
SHORT_PROSE_WORDS = frozenset(
    {
        "a",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "if",
        "in",
        "is",
        "it",
        "me",
        "no",
        "of",
        "on",
        "or",
        "so",
        "to",
        "up",
        "us",
        "we",
    }
)


def is_unicode_math_symbol(char: str) -> bool:
    if not char or len(char) != 1:
        return False
    return (
        char in MATH_SYMBOLS
        or char in MATH_LETTERLIKE_SYMBOLS
        or unicodedata.category(char) == "Sm"
    )


def unicode_math_symbol_count(text: str) -> int:
    return sum(1 for char in (text or "") if is_unicode_math_symbol(char))


def iter_unicode_math_symbol_ranges(text: str) -> Iterator[tuple[int, int]]:
    start = None
    for index, char in enumerate(text or ""):
        if is_unicode_math_symbol(char):
            if start is None:
                start = index
        elif start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(text)


def is_math_block(text: str) -> bool:
    """Return whether a short text block is predominantly mathematical."""
    stripped = html.unescape(text or "")
    stripped = re.sub(r"(?is)</?(?:b|sup)\b[^>]*>", "", stripped).strip()
    if not stripped or len(stripped) > 300:
        return False
    if re.search(
        r"\\(?:frac|sqrt|sum|prod|int|lim|log|ln|sin|cos|tan|exp|sup|inf|max|min|arg|det)\b",
        stripped,
    ):
        return True
    if re.search(
        r"\$(?!\s*\d)[^$\n]{0,159}[a-zA-Z\\^_{}=<>+*/-][^$\n]{0,159}\$",
        stripped,
    ):
        return True
    if re.search(r"\\\(.*\\\)", stripped, re.DOTALL):
        return True
    if re.search(r"\\\[.*\\\]", stripped, re.DOTALL):
        return True

    math_chars = sum(
        1 for char in stripped if is_unicode_math_symbol(char) or char in "{}^_()[]|"
    )
    total = len(stripped)
    unicode_math = unicode_math_symbol_count(stripped)
    replacement_chars = stripped.count("\ufffd")
    relation_chars = sum(1 for char in stripped if char in "=<>≤≥≈≠→←↔⇒⇐")
    operator_chars = sum(1 for char in stripped if char in "+−*/^_")
    grouping_chars = sum(1 for char in stripped if char in "()[]{}|")
    digit_chars = sum(1 for char in stripped if char.isdigit())

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']*", stripped)
    prose_tokens: list[str] = []
    variable_tokens: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if (
            lowered in MATH_FUNCTION_WORDS
            or len(token) == 1
            or re.fullmatch(r"[A-Za-z]{1,3}\d+", token)
            or (len(token) == 2 and lowered not in SHORT_PROSE_WORDS)
        ):
            variable_tokens.append(token)
        else:
            prose_tokens.append(token)

    math_weight = (
        unicode_math * 2
        + replacement_chars * 3
        + relation_chars * 2
        + operator_chars
        + min(grouping_chars, 4)
        + min(digit_chars, 4)
        + sum(len(token) for token in variable_tokens)
    )
    prose_weight = sum(len(token) for token in prose_tokens)
    has_structure = bool(
        unicode_math
        or replacement_chars
        or relation_chars
        or operator_chars >= 2
        or grouping_chars >= 4
    )

    if (
        not prose_tokens
        and tokens
        and any(token.lower() in MATH_FUNCTION_WORDS for token in tokens)
        and all(
            token.lower() in MATH_FUNCTION_WORDS
            or len(token) <= 2
            or re.fullmatch(r"[A-Za-z]{1,3}\d+", token)
            for token in tokens
        )
        and len(re.sub(r"\s+", "", stripped)) <= 16
    ):
        return True
    if has_structure and not prose_tokens:
        return True

    relation_match = re.search(r"[=<>≤≥≈≠→←↔⇒⇐]", stripped)
    if relation_match and len(stripped) <= 90 and len(prose_tokens) <= 4:
        tail_tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9']*", stripped[relation_match.end() :]
        )
        tail_prose_tokens = [
            token
            for token in tail_tokens
            if not (
                token.lower() in MATH_FUNCTION_WORDS
                or len(token) == 1
                or re.fullmatch(r"[A-Za-z]{1,3}\d+", token)
                or (len(token) == 2 and token.lower() not in SHORT_PROSE_WORDS)
            )
        ]
        if not tail_prose_tokens and (
            unicode_math or replacement_chars or variable_tokens
        ):
            return True

    if (
        has_structure
        and len(prose_tokens) <= 1
        and math_weight >= 10
        and math_weight >= prose_weight * 1.15
    ):
        return True
    return math_chars / total > 0.3


__all__ = [
    "MATH_FUNCTION_WORDS",
    "is_math_block",
    "is_unicode_math_symbol",
    "iter_unicode_math_symbol_ranges",
    "unicode_math_symbol_count",
]
