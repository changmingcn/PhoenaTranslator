"""Byte-exact XHTML MathML, LaTeX and Unicode-math protection."""

from __future__ import annotations

import hashlib
import html
import re

from ..math_text import (
    MATH_FUNCTION_WORDS,
    is_math_block,
    is_unicode_math_symbol,
    iter_unicode_math_symbol_ranges,
)
from .xhtml import visible_text_ranges


XHTML_MATH_PLACEHOLDER_RE = re.compile(r"PHOENA_MATH_\d{4}_[0-9A-F]{16}")


def looks_like_latex_dollar_content(content: str) -> bool:
    stripped = (content or "").strip()
    if not stripped or len(stripped) > 400:
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
        for match in re.finditer(
            r"(?s)(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", segment
        ):
            if looks_like_latex_dollar_content(match.group(1)):
                candidates.append(
                    (
                        range_start + match.start(),
                        range_start + match.end(),
                        2,
                        "latex_inline_dollar",
                    )
                )
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


def protect_xhtml_math_fragments(text: str) -> tuple[str, list[dict]]:
    intervals = find_xhtml_math_intervals(text)
    if not intervals:
        return text, []
    records: list[dict] = []
    pieces: list[str] = []
    cursor = 0
    for index, (start, end, reason) in enumerate(intervals):
        original = text[start:end]
        digest = hashlib.sha256(
            f"{start}:{end}:".encode("ascii") + original.encode("utf-8")
        ).hexdigest().upper()[:16]
        token = f"PHOENA_MATH_{index:04d}_{digest}"
        while token in text:
            digest = hashlib.sha256(
                (digest + original).encode("utf-8")
            ).hexdigest().upper()[:16]
            token = f"PHOENA_MATH_{index:04d}_{digest}"
        pieces.extend((text[cursor:start], token))
        records.append(
            {
                "token": token,
                "original": original,
                "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                "reason": reason,
            }
        )
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), records


def restore_xhtml_math_fragments(translated: str, records: list[dict]) -> str | None:
    if not records:
        return translated
    expected_tokens = [record["token"] for record in records]
    observed_tokens = XHTML_MATH_PLACEHOLDER_RE.findall(translated or "")
    if observed_tokens != expected_tokens:
        return None
    for record in records:
        original = record.get("original", "")
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != record.get("sha256"):
            return None
        if (translated or "").count(record["token"]) != 1:
            return None
    restored = translated
    for record in records:
        restored = restored.replace(record["token"], record["original"], 1)
    if XHTML_MATH_PLACEHOLDER_RE.search(restored):
        return None
    return restored


__all__ = [
    "XHTML_MATH_PLACEHOLDER_RE",
    "find_xhtml_math_intervals",
    "looks_like_latex_dollar_content",
    "protect_xhtml_math_fragments",
    "restore_xhtml_math_fragments",
]
