"""Opaque placeholder tokens that carry exact source bytes through translation.

Every protected fragment is replaced by ``PHOENA_<KIND>_<index>_<digest>``
before the model call and restored only after the token sequence survives
validation, so the model never has to copy markup, identifiers or formulae.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


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
