"""Normalise and compare the lightweight inline markup (<b>, <sup>) that
translations carry, and repair superscript markers a model dropped."""

from __future__ import annotations

import re
import html
from phoena_translator.pdf.text_shape import _plain_text


def _normalize_pdf_translation(text: str) -> str:
    """Keep only lightweight inline markup that we deliberately support in PDFs."""
    normalized = (text or "").strip()
    replacements = [
        (r'(?is)&lt;\s*b\b.*?&gt;', '<b>'),
        (r'(?is)&lt;\s*strong\b.*?&gt;', '<b>'),
        (r'(?is)&lt;\s*sup\b.*?&gt;', '<sup>'),
        (r'(?i)&lt;\s*b\s*&gt;', '<b>'),
        (r'(?i)&lt;\s*/\s*b\s*&gt;', '</b>'),
        (r'(?i)&lt;\s*strong\s*&gt;', '<b>'),
        (r'(?i)&lt;\s*/\s*strong\s*&gt;', '</b>'),
        (r'(?i)&lt;\s*sup\s*&gt;', '<sup>'),
        (r'(?i)&lt;\s*/\s*sup\s*&gt;', '</sup>'),
        (r'(?i)&lt;\s*br\s*/?\s*&gt;', '\n'),
        (r'(?is)<\s*b\b[^>]*>', '<b>'),
        (r'(?is)<\s*strong\b[^>]*>', '<b>'),
        (r'(?is)<\s*sup\b[^>]*>', '<sup>'),
        (r'(?i)<\s*strong\s*>', '<b>'),
        (r'(?i)<\s*/\s*strong\s*>', '</b>'),
        (r'(?i)<\s*b\s*>', '<b>'),
        (r'(?i)<\s*/\s*b\s*>', '</b>'),
        (r'(?i)<\s*sup\s*>', '<sup>'),
        (r'(?i)<\s*/\s*sup\s*>', '</sup>'),
        (r'(?i)<\s*br\s*/?\s*>', '\n'),
    ]
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized)

    # Strip known presentation/wrapper tags, but keep unknown angle-bracketed
    # source text literal.  Financial prose uses notation such as
    # ``<JPY short position>``; the former generic ``<[a-z]...>`` sanitizer
    # mistook that complete phrase for HTML and silently removed it before
    # translation.  Rendering escapes unknown tags, so preserving them here is
    # both lossless and safe.
    normalized = re.sub(
        r"(?is)<\s*/?\s*(?:"
        r"a|abbr|address|article|aside|blockquote|body|caption|code|dd|div|dl|dt|"
        r"em|figcaption|figure|font|footer|form|h[1-6]|head|header|html|i|label|"
        r"li|main|nav|ol|p|pre|section|small|span|sub|table|tbody|td|tfoot|th|"
        r"thead|title|tr|u|ul|translation|answer"
        r")\b[^>]*>",
        "",
        normalized,
    )
    normalized = re.sub(r'(?is)</?br\b[^>]*>', '\n', normalized)
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t]+\n', '\n', normalized)
    normalized = re.sub(r'\n[ \t]+', '\n', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = re.sub(r'[ \t]{2,}', ' ', normalized)
    return normalized


def _pdf_superscript_signature(text: str) -> tuple[str, ...]:
    normalized = _normalize_pdf_translation(text)
    return tuple(
        html.unescape(_plain_text(match.group(1))).strip()
        for match in re.finditer(r'(?is)<sup>(.*?)</sup>', normalized)
        if html.unescape(_plain_text(match.group(1))).strip()
    )


def _pdf_inline_math_fragments_preserved(
    elem: dict,
    translated_text: str,
) -> bool:
    expected = [
        fragment.get("text", "")
        for fragment in (elem.get("inline_math_fragments") or [])
        if isinstance(fragment, dict) and fragment.get("text", "")
    ]
    if not expected:
        return True
    plain = _plain_text(_normalize_pdf_translation(translated_text))
    cursor = 0
    for fragment in expected:
        position = plain.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment)
    return True


def _pdf_inline_markup_preserved(source_text: str, translated_text: str) -> bool:
    source_signature = _pdf_superscript_signature(source_text)
    if not source_signature:
        return True
    return source_signature == _pdf_superscript_signature(translated_text)


def _pdf_marker_pattern(marker: str) -> re.Pattern:
    escaped = re.escape(marker)
    if marker and marker[0].isalnum() and marker[-1].isalnum():
        return re.compile(rf'(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])')
    return re.compile(escaped)


def _restore_pdf_superscript_markup(source_text: str, translated_text: str) -> str:
    """Restore dropped <sup> tags by the marker's source occurrence ordinal."""
    source = _normalize_pdf_translation(source_text)
    translated = _normalize_pdf_translation(translated_text)
    if _pdf_inline_markup_preserved(source, translated):
        return translated

    descriptors = []
    for match in re.finditer(r'(?is)<sup>(.*?)</sup>', source):
        marker = html.unescape(_plain_text(match.group(1))).strip()
        if not marker:
            continue
        source_prefix = _plain_text(source[:match.start()])
        ordinal = len(list(_pdf_marker_pattern(marker).finditer(source_prefix))) + 1
        descriptors.append((marker, ordinal))
    if not descriptors:
        return translated

    flat = re.sub(r'(?is)</?sup>', '', translated)
    selected_ranges = []
    for marker, ordinal in descriptors:
        matches = list(_pdf_marker_pattern(marker).finditer(flat))
        if ordinal <= 0 or ordinal > len(matches):
            return translated
        selected = matches[ordinal - 1]
        item = (selected.start(), selected.end())
        if item in selected_ranges:
            return translated
        selected_ranges.append(item)

    restored = flat
    for start, end in sorted(selected_ranges, reverse=True):
        restored = restored[:start] + "<sup>" + restored[start:end] + "</sup>" + restored[end:]
    return _normalize_pdf_translation(restored)
