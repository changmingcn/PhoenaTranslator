"""XHTML markup lexing shared by EPUB segmentation and math protection."""

from __future__ import annotations


# UTF-8 byte budget for one translation request.
DEFAULT_CHUNK_MAX_BYTES = 20_000

_DELIMITED_MARKUP = (("<!--", "-->"), ("<![CDATA[", "]]>"), ("<?", "?>"))


def _tag_end(text: str, start: int) -> int:
    """Return the end of the tag at ``start``, honouring quotes and DTD subsets."""
    quote: str | None = None
    depth = 0
    is_doctype = text[start:start + 9].upper() == "<!DOCTYPE"
    cursor = start + 1
    while cursor < len(text):
        char = text[cursor]
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif is_doctype and char == "[":
            depth += 1
        elif is_doctype and char == "]":
            depth -= 1
        elif char == ">" and depth == 0:
            return cursor + 1
        cursor += 1
    return cursor


def markup_ranges(text: str, *, strict: bool = False) -> list[tuple[int, int]]:
    """Lex markup ranges while respecting quotes, comments, CDATA and DOCTYPE.

    ``strict`` is for input already validated as XML: an unterminated comment,
    CDATA section or processing instruction raises ``ValueError``.  Otherwise
    such a construct extends to the end of ``text``.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while (start := text.find("<", cursor)) >= 0:
        end = None
        for prefix, suffix in _DELIMITED_MARKUP:
            if text.startswith(prefix, start):
                stop = text.find(suffix, start + len(prefix))
                if stop < 0:
                    if strict:
                        raise ValueError("Invalid XHTML markup boundary")
                    end = len(text)
                else:
                    end = stop + len(suffix)
                break
        if end is None:
            end = _tag_end(text, start)
        ranges.append((start, end))
        cursor = end
    return ranges


def visible_text_ranges(text: str) -> list[tuple[int, int]]:
    """Return source ranges outside tags so attributes are never masked."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in markup_ranges(text):
        if start > cursor:
            ranges.append((cursor, start))
        cursor = end
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return ranges


def markup_tokens(text: str) -> list[str]:
    """Return exact markup tokens for deterministic structure comparison."""
    return [text[start:end] for start, end in markup_ranges(text)]


__all__ = [
    "DEFAULT_CHUNK_MAX_BYTES",
    "markup_ranges",
    "markup_tokens",
    "visible_text_ranges",
]
