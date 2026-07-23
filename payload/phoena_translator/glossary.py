"""Deterministic term selection and explicit glossary configuration."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_ACRONYM_RE = re.compile(r"(?<![\w.-])[A-Z][A-Z0-9&.-]{1,15}(?![\w.-])")
_PROPER_RE = re.compile(
    r"(?<![\w.-])(?:[A-Z][A-Za-z'’.-]{2,}"
    r"(?:\s+(?:of|the|and|for|in|on|de|[A-Z][A-Za-z'’.-]{2,})){0,4})(?![\w.-])"
)
_MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TermCandidate:
    term: str
    count: int
    first_offset: int


def extract_term_candidates(
    text: str,
    *,
    min_count: int = 1,
    max_terms: int = 500,
) -> list[TermCandidate]:
    """Rank likely terms using only lexical rules, counts and first occurrence."""
    found_spans: set[tuple[int, int, str]] = set()
    for pattern in (_ACRONYM_RE, _PROPER_RE):
        found_spans.update(
            (match.start(), match.end(), match.group(0).strip())
            for match in pattern.finditer(text)
        )
    found = sorted((start, term) for start, _end, term in found_spans)
    counts = Counter(term for _offset, term in found)
    first: dict[str, int] = {}
    for offset, term in found:
        first.setdefault(term, offset)
    ranked = [
        TermCandidate(term=term, count=count, first_offset=first[term])
        for term, count in counts.items()
        if count >= min_count
    ]
    ranked.sort(key=lambda item: (-item.count, item.first_offset, item.term.casefold()))
    return ranked[:max_terms]


def load_explicit_glossary(path: Path | None) -> dict[str, str]:
    """Load a bounded JSON object. No implicit translation or model fallback exists."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Glossary JSON does not exist: {path}")
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("Glossary JSON exceeds 1 MiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Glossary JSON must be an object of source-to-target strings")
    result: dict[str, str] = {}
    for source, target in payload.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Glossary keys and values must be strings")
        source = source.strip()
        target = target.strip()
        if not source or not target or "\n" in source or "\n" in target:
            raise ValueError("Glossary entries must be non-empty single-line strings")
        result[source] = target
    return result


def build_glossary_text(
    text: str,
    mapping: Mapping[str, str],
    *,
    max_terms: int = 500,
) -> str:
    """Select configured entries that occur in the source and format prompt lines."""
    if not mapping or not text:
        return ""
    candidates = {item.term: item for item in extract_term_candidates(text, max_terms=max_terms * 4)}
    selected: list[tuple[int, int, str, str]] = []
    for order, (source, target) in enumerate(mapping.items()):
        # Whole-term occurrences only: plain substring counting matched "Art"
        # inside "Article" and skewed both selection and ranking.
        term_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
        )
        first_match = term_pattern.search(text)
        if first_match is None:
            continue
        occurrences = len(term_pattern.findall(text))
        first_offset = first_match.start()
        candidate = candidates.get(source)
        candidate_count = candidate.count if candidate is not None else occurrences
        selected.append((-candidate_count, first_offset, source, target))
    selected.sort(key=lambda item: (item[0], item[1], item[2].casefold()))
    return "\n".join(
        f"{source} = {target}"
        for _count, _offset, source, target in selected[:max_terms]
    )
