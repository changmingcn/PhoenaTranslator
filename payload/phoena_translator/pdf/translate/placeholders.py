"""Opaque placeholder tokens that carry exact source bytes through translation.

Every protected fragment is replaced by ``PHOENA_<KIND>_<index>_<digest>``
before the model call and restored only after the token sequence survives
validation, so the model never has to copy markup, identifiers or formulae.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from phoena_translator.pdf.patterns import _PRESERVED_IDENTIFIER_RE
from phoena_translator.pdf.translate.inline_markup import _normalize_pdf_translation

PDF_BATCH_SEGMENT_RE = re.compile(r"PHOENA_SEG_\d{4}_[A-F0-9]{16}")
PDF_SUPERSCRIPT_PLACEHOLDER_RE = re.compile(
    r"PHOENA_SUP_\d{4}_[0-9A-F]{16}"
)
PDF_INLINE_MATH_PLACEHOLDER_RE = re.compile(
    r"PHOENA_IMATH_\d{4}_[0-9A-F]{16}"
)
PDF_IDENTIFIER_PLACEHOLDER_RE = re.compile(
    r"PHOENA_ID_\d{4}_[0-9A-F]{16}"
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()[:16]


def mint_placeholder_token(
    kind: str,
    index: int,
    start: int,
    end: int,
    original: str,
    text: str,
) -> str:
    """Return a token for ``original`` at ``start:end`` that is absent from ``text``."""
    digest = _digest(f"{index}:{start}:{end}:".encode("ascii") + original.encode("utf-8"))
    token = f"PHOENA_{kind}_{index:04d}_{digest}"
    while token in text:
        digest = _digest((digest + original).encode("utf-8"))
        token = f"PHOENA_{kind}_{index:04d}_{digest}"
    return token


def placeholder_record(token: str, original: str, **extra) -> dict:
    return {
        "token": token,
        "original": original,
        "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        **extra,
    }


def record_original_intact(record: dict) -> bool:
    original = record.get("original", "")
    return hashlib.sha256(original.encode("utf-8")).hexdigest() == record.get("sha256")


def _edge_anchor(text: str, start: int, end: int) -> tuple[str, str]:
    """Classify a fragment as sitting at the source prefix, suffix or inline."""
    source_prefix = text[:start]
    source_suffix = text[end:]
    if not source_prefix.strip():
        separator_match = re.match(r"\s*", source_suffix)
        return "prefix", separator_match.group(0) if separator_match else ""
    if not source_suffix.strip():
        separator_match = re.search(r"\s*$", source_prefix)
        return "suffix", separator_match.group(0) if separator_match else ""
    return "inline", ""


def protect_anchored_matches(
    text: str,
    matches: Iterable[re.Match],
    kind: str,
) -> tuple[str, list[dict]]:
    """Tokenise every match and remember whether it sat at a source edge."""
    records = []
    pieces = []
    cursor = 0
    for index, match in enumerate(matches):
        original = match.group(0)
        token = mint_placeholder_token(kind, index, match.start(), match.end(), original, text)
        pieces.append(text[cursor:match.start()])
        pieces.append(token)
        anchor, anchor_separator = _edge_anchor(text, match.start(), match.end())
        records.append(
            placeholder_record(token, original, anchor=anchor, anchor_separator=anchor_separator)
        )
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), records


def restore_anchored_tokens(
    translated_text: str,
    records: list[dict],
    token_re: re.Pattern,
) -> str | None:
    """Restore tokens after exact order validation, re-anchoring edge fragments.

    Models often move a leading footnote marker or a trailing URL while
    keeping its token intact; edge fragments return to their source edge,
    inline ones keep the model's word order.  Returns ``None`` on any loss,
    duplication or reordering.
    """
    expected_tokens = [record["token"] for record in records]
    if token_re.findall(translated_text or "") != expected_tokens:
        return None
    restored = translated_text or ""
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
        if not record_original_intact(record):
            return None
        token = record["token"]
        if restored.count(token) != 1:
            return None
        restored = restored.replace(token, record.get("original", ""), 1)
    if token_re.search(restored):
        return None
    return restored


__all__ = [
    "mint_placeholder_token",
    "placeholder_record",
    "protect_anchored_matches",
    "record_original_intact",
    "restore_anchored_tokens",
]


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
                # Only supported rich-text tags are control markup.  A broad
                # ``<...>`` range also blocks the angle delimiters in lexical
                # source notation such as ``<JPY short position>``, leaving
                # them unprotected while the validator still requires them.
                re.compile(
                    r"(?is)<\s*/?\s*(?:b|strong|sup|br)\b[^>]*>"
                ),
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
    return protect_anchored_matches(normalized, matches, "SUP")


def _restore_pdf_superscript_fragments(
    translated_text: str,
    records: list[dict],
) -> str | None:
    """Restore protected PDF superscripts only after exact token validation.

    Models frequently move a leading footnote marker to the end of a Chinese
    sentence even when they preserve its opaque token exactly; the shared
    restorer re-anchors such source-edge markers.
    """
    if not records:
        return translated_text
    restored = restore_anchored_tokens(
        translated_text, records, PDF_SUPERSCRIPT_PLACEHOLDER_RE
    )
    return None if restored is None else _normalize_pdf_translation(restored)


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
        token = mint_placeholder_token("IMATH", index, start, end, original, protected)
        protected = protected[:start] + token + protected[end:]
        records.append(placeholder_record(token, original))
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
        if not record_original_intact(record):
            return None
        original = record.get("original", "")
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
    return protect_anchored_matches(protected, matches, "ID")


def _restore_pdf_identifier_fragments(
    translated_text: str,
    records: list[dict],
) -> str | None:
    """Restore protected identifiers after exact count/order validation.

    Source-edge identifiers stay at the same edge: for URL footnotes and
    bibliography entries moving the token changes the semantic association
    even though its bytes remain intact.
    """
    if not records:
        return translated_text
    return restore_anchored_tokens(translated_text, records, PDF_IDENTIFIER_PLACEHOLDER_RE)
