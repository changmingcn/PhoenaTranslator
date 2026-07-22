"""Pure-Python XHTML classification, repair, chunking and merge helpers."""

from __future__ import annotations

import logging
import re
from pathlib import Path


DEFAULT_CHUNK_MAX_BYTES = 20_000
SKIP_SECTION_PATTERNS = re.compile(
    r"(?i)\b(bibliography|references|index|endnotes?|glossary|acronyms|abbreviations)\b"
)
_BODY_RE = re.compile(r"(<body[^>]*>)(.*?)(</body>)", re.DOTALL | re.IGNORECASE)


def is_appendix_xhtml(filepath: str, content: str) -> bool:
    """Detect bibliography/reference/index material that should not be translated."""
    filename = Path(filepath).name.lower()
    if any(
        pattern in filename
        for pattern in ("bibliograph", "reference", "index", "endnote", "glossary")
    ):
        return True
    title_match = re.search(
        r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL
    )
    if title_match and SKIP_SECTION_PATTERNS.search(title_match.group(1)):
        return True
    heading_match = re.search(
        r"<h[12][^>]*>(.*?)</h[12]>", content, re.IGNORECASE | re.DOTALL
    )
    if heading_match:
        heading_text = re.sub(r"<[^>]+>", "", heading_match.group(1))
        if SKIP_SECTION_PATTERNS.search(heading_text):
            return True
    return False


def _byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def _split_oversized_text(value: str, max_bytes: int) -> list[str]:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for char in value:
        char_bytes = len(char.encode("utf-8"))
        if current and current_bytes + char_bytes > max_bytes:
            chunks.append("".join(current))
            current = []
            current_bytes = 0
        current.append(char)
        current_bytes += char_bytes
    if current:
        chunks.append("".join(current))
    return chunks


def split_raw(text: str, max_size: int) -> list[str]:
    """Fallback byte-bounded split that never cuts a UTF-8 code point."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True) or [text]:
        if _byte_length(line) > max_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_oversized_text(line, max_size))
            continue
        if current and _byte_length(current) + _byte_length(line) > max_size:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def split_xhtml_to_chunks(
    content: str,
    max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
    *,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Split XHTML at structural boundaries while retaining full document wrappers."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if _byte_length(content) <= max_bytes:
        return [content]

    body_match = _BODY_RE.search(content)
    if not body_match:
        return split_raw(content, max_bytes)
    pre_body = content[: body_match.start(2)]
    body_content = body_match.group(2)
    post_body = content[body_match.end(2) :]
    wrapper_bytes = _byte_length(pre_body) + _byte_length(post_body)
    body_limit = max_bytes - wrapper_bytes
    if body_limit < 1:
        return split_raw(content, max_bytes)

    split_pattern = (
        r'(?=<(?:h[1-3]|section|div\s+class="(?:chapter|section|part)")[^>]*>)'
    )
    sections = [item for item in re.split(split_pattern, body_content) if item.strip()]
    if len(sections) <= 1:
        sections = [
            item
            for item in re.split(
                r"(?=<(?:p|div|blockquote|table|ul|ol)[^>]*>)", body_content
            )
            if item.strip()
        ]

    body_chunks: list[str] = []
    current = ""
    for section in sections:
        if current and _byte_length(current) + _byte_length(section) > body_limit:
            body_chunks.append(current)
            current = section
        else:
            current += section
    if current:
        body_chunks.append(current)

    final_chunks: list[str] = []
    for body_chunk in body_chunks:
        if _byte_length(body_chunk) <= body_limit:
            final_chunks.append(pre_body + body_chunk + post_body)
            continue
        parts = re.split(r"(</(?:p|li|tr|div|br)>)", body_chunk)
        merged_parts: list[str] = []
        for part in parts:
            if re.fullmatch(r"</(?:p|li|tr|div|br)>", part):
                if merged_parts:
                    merged_parts[-1] += part
                else:
                    merged_parts.append(part)
            elif part:
                merged_parts.append(part)
        parts = [part for part in merged_parts if part.strip()]
        if len(parts) <= 1:
            # There is no safe structural split point. Preserve the element as
            # one valid chunk instead of slicing through markup.
            final_chunks.append(pre_body + body_chunk + post_body)
            continue
        current = ""
        for part in parts:
            if current and _byte_length(current) + _byte_length(part) > body_limit:
                final_chunks.append(pre_body + current + post_body)
                current = part
            else:
                current += part
        if current:
            final_chunks.append(pre_body + current + post_body)

    if logger is not None:
        logger.info(
            "Split XHTML into %s chunks (UTF-8 bytes: %s)",
            len(final_chunks),
            [_byte_length(chunk) for chunk in final_chunks],
        )
    return final_chunks


def merge_translated_chunks(chunks: list[str]) -> str:
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    first_match = _BODY_RE.search(chunks[0])
    if not first_match:
        return "\n".join(chunks)
    pre_body = chunks[0][: first_match.start(2)]
    post_body = chunks[0][first_match.end(2) :]
    bodies: list[str] = []
    for chunk in chunks:
        match = re.search(
            r"<body[^>]*>(.*?)</body>", chunk, re.DOTALL | re.IGNORECASE
        )
        bodies.append(match.group(1) if match else chunk)
    return pre_body + "".join(bodies) + post_body


def fix_xhtml_entities(text: str) -> str:
    """Escape bare ampersands without double-escaping valid XML entities."""
    return re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)",
        "&amp;",
        text,
    )


def fix_xhtml_tags(text: str) -> str:
    """Repair the narrow set of known translation-provider tag corruptions."""
    text = re.sub(r"</([a-zA-Z]+)([^\x00-\x7F])", r"</\1>\2", text)
    text = text.replace("</>", "</a>")
    text = re.sub(
        r'<a (href="#fn\d+" id="ft\d+")>(\d+)</sup>',
        r"<sup><a \1>\2</a></sup>",
        text,
    )
    return re.sub(
        r'(?<!<sup>)(<a href="#fn\d+" id="ft\d+">\d+</a></sup>)',
        r"<sup>\1",
        text,
    )


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


def markup_ranges(text: str) -> list[tuple[int, int]]:
    """Lex XHTML markup ranges while respecting quotes, comments and CDATA."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = text.find("<", index)
        if start < 0:
            break
        if text.startswith("<!--", start):
            stop = text.find("-->", start + 4)
            end = len(text) if stop < 0 else stop + 3
        elif text.startswith("<![CDATA[", start):
            stop = text.find("]]>", start + 9)
            end = len(text) if stop < 0 else stop + 3
        elif text.startswith("<?", start):
            stop = text.find("?>", start + 2)
            end = len(text) if stop < 0 else stop + 2
        else:
            quote: str | None = None
            cursor = start + 1
            while cursor < len(text):
                char = text[cursor]
                if quote:
                    if char == quote:
                        quote = None
                elif char in {'"', "'"}:
                    quote = char
                elif char == ">":
                    cursor += 1
                    break
                cursor += 1
            end = cursor
        ranges.append((start, end))
        index = max(end, start + 1)
    return ranges


def markup_tokens(text: str) -> list[str]:
    """Return exact markup tokens for deterministic structure comparison."""
    return [text[start:end] for start, end in markup_ranges(text)]


__all__ = [
    "DEFAULT_CHUNK_MAX_BYTES",
    "fix_xhtml_entities",
    "fix_xhtml_tags",
    "is_appendix_xhtml",
    "merge_translated_chunks",
    "markup_ranges",
    "markup_tokens",
    "split_raw",
    "split_xhtml_to_chunks",
    "visible_text_ranges",
]
