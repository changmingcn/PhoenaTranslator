"""Translate semantic XHTML text units without asking a model to rewrite XML.

Offsets and immutable fragments belong to the local source.  Only the JSON
string values travel to the translation service; restoring them cannot change
attributes, links, element nesting, or the document wrapper.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Collection, Mapping, Sequence

from .math_protection import find_xhtml_math_intervals
from .xhtml import markup_ranges


SEGMENT_SYSTEM_PROMPT = """你是专业英文图书中文译者。输入是 JSON 对象，键是稳定段落 ID，值是一个完整语义段落、标题或列表项。
只输出同一组 ID 对应的 JSON 对象，所有值为中文译文字符串。不得增删或重复 ID，不得输出 Markdown 代码围栏、解释、总结、译者说明或 JSON 之外的任何内容。
完整翻译所有可见正文、章标题、小标题、目录文字、引文标题及注释说明，不得省略、概括或原文回填。人名、机构专名、编号、URL 和数学公式可以保留。
参考文献中的书名、论文名、新闻标题、视频标题必须译成中文，包括引号内及斜体标记内的作品标题。它们不是可以整体保留的专名；仅作者、期刊/出版社/媒体名称、网址和编号可保留。不要以“这是参考文献”为由保留英文作品标题。
形如 [[PH_0000_ABCDEF12]] 的不可变占位符代表本地保留的内联标签、链接、数学或其他原文片段。每个占位符必须逐字保留一次，数量和顺序严格不变；不要输出实际 HTML 标签或转义后的 HTML 实体。
占位符之间的文字属于同一个段落：综合整段理解后翻译。尤其不要把首字下沉标签内的单个字母与后面的词分开理解。可调整占位符间的译文以保持自然中文，但不能移动占位符。
只含人名、数字、缩写或 URL 的项可保持原值；有叙述内容的项必须完整译成中文。"""

TOKEN_RE = re.compile(r"\[\[PH_\d{4,}_[0-9A-F]{8}\]\]")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_URL_RE = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"']+|"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "caption", "dd",
    "details", "dialog", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table",
    "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
}
_OPAQUE_TAGS = {"script", "style", "svg", "math"}
_PROSE_WORDS = set(
    "a an the this that these those is are was were be been being have has had "
    "do does did will would shall should can could may might must not no for "
    "to of in on at by from with without as if then than but or because when "
    "where which who what why how into over under between through about our "
    "your their its it we you they he she i us them my his her all every only "
    "also more most less very such new introduction contents notes references "
    "acknowledgments acknowledgements index chapter copyright conclusion".split()
)
_NAME_CONNECTORS = {"and", "of", "the", "for", "de", "del", "der", "di", "da", "van", "von", "la", "et", "al"}
_SINGLE_LETTER_RE = re.compile(r"[A-Za-z]\.?")
# Only this quality rule may be waived, and only for English kept from the
# source (an article title in a citation). Coverage failures (refusals,
# truncations) are never waived.
_QUALITY_REASONS = ("untranslated long English passage",)
_ENGLISH_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\s,'’\"“”.!?():;–—/\-]*")
_META_RE = re.compile(
    r"(?:^|\n)\s*(?:以下(?:是|为).{0,16}(?:翻译|译文)|翻译(?:完成|如下)|"
    r"译者(?:说明|注)|我(?:已|已经).{0,16}翻译|"
    r"here\s+is\s+(?:the|your)\s+translation|translation\s+(?:complete|follows))",
    re.IGNORECASE,
)


class SegmentValidationError(ValueError):
    """A model response cannot safely replace the source segment(s)."""


@dataclass(frozen=True)
class ProtectedFragment:
    token: str
    original: str


@dataclass(frozen=True)
class TextSegment:
    id: str
    start: int
    end: int
    kind: str
    source: str
    original: str
    fragments: tuple[ProtectedFragment, ...]

    @property
    def plain_text(self) -> str:
        return TOKEN_RE.sub("", self.source).strip()

    @property
    def has_words(self) -> bool:
        """Whether the visible text contains a Latin word (pre-2026-10-28 request rule).

        The glossary source keeps using this rule so the checkpoint signature of
        books translated before the single-letter exemption stays valid.
        """
        return bool(_WORD_RE.search(self.plain_text))

    @property
    def needs_request(self) -> bool:
        # Numbers, already-Chinese text, pure URLs/math, empty markup and a
        # lone Latin letter (index section headings "E", list labels "A.")
        # need no provider call. Proper names still travel with their paragraph.
        if _SINGLE_LETTER_RE.fullmatch(self.plain_text):
            return False
        return self.has_words


@dataclass(frozen=True)
class XHTMLDocument:
    source: str
    segments: tuple[TextSegment, ...]

    def render(self, translations: Mapping[str, str], *, check_quality: bool = True,
               lenient_ids: Collection[str] = ()) -> str:
        """Restore every segment; ``lenient_ids`` skip only the quality rules."""
        expected = {segment.id for segment in self.segments}
        if set(translations) != expected:
            raise SegmentValidationError("render requires exactly all segment IDs")
        lenient = set(lenient_ids)
        pieces: list[str] = []
        cursor = 0
        for segment in self.segments:
            translated = translations[segment.id]
            validate_segment_translation(segment, translated,
                                         check_quality=check_quality and segment.id not in lenient)
            pieces.extend((self.source[cursor:segment.start], _restore(segment, translated)))
            cursor = segment.end
        pieces.append(self.source[cursor:])
        result = "".join(pieces)
        validate_xhtml(result)
        return result


@dataclass
class BatchValidation:
    translations: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    # Structurally valid responses that failed only a quality rule:
    # id -> (translated text, quality reason). Their ids are also in failures.
    quality_rejects: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Ids whose response text changed through invisible-anchor repair.
    repaired: set[str] = field(default_factory=set)


@dataclass
class _Node:
    name: str
    start: int
    inner_start: int
    inner_end: int
    end: int
    children: list[_Node] = field(default_factory=list)


def validate_xhtml(source: str) -> None:
    """Strict XML parsing also rejects explanatory text after </html>."""
    try:
        root = ET.fromstring(source)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError(f"Invalid XHTML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "html":
        raise ValueError("Invalid XHTML: root element must be html")


def _nodes(source: str) -> list[_Node]:
    roots: list[_Node] = []
    stack: list[_Node] = []
    for start, end in markup_ranges(source, strict=True):
        raw = source[start:end]
        if raw.startswith(("<!", "<?")):
            continue
        if raw.startswith("</"):
            node = stack.pop()
            node.inner_end, node.end = start, end
            continue
        match = re.match(r"<([^\s/>]+)", raw)
        if not match:
            raise ValueError("Invalid XHTML opening tag")
        node = _Node(match.group(1).rsplit(":", 1)[-1].lower(), start, end, end, end)
        (stack[-1].children if stack else roots).append(node)
        if not raw.rstrip().endswith("/>"):
            stack.append(node)
    return roots


def _protect(raw: str) -> tuple[str, tuple[ProtectedFragment, ...]]:
    candidates: list[tuple[int, int, int]] = []
    def protect_reference_markers(node: _Node) -> None:
        if node.name in {"a", "sup"}:
            inner = raw[node.inner_start:node.inner_end]
            visible: list[str] = []
            cursor = 0
            for start, end in markup_ranges(inner, strict=True):
                visible.append(inner[cursor:start])
                cursor = end
            visible.append(inner[cursor:])
            label = html.unescape("".join(visible)).strip()
            if re.fullmatch(r"(?:\[?\d+(?:[–—,\-]\d+)*\]?|[*†‡]+)", label):
                candidates.append((0, node.start, node.end))
                return
        for child in node.children:
            protect_reference_markers(child)

    for node in _nodes(raw):
        protect_reference_markers(node)
    # Non-text namespaces and executable/style content are never translated.
    for match in re.finditer(
        r"<(?:[\w.-]+:)?(script|style|svg|math)\b[^>]*>.*?</(?:[\w.-]+:)?\1\s*>",
        raw, re.IGNORECASE | re.DOTALL,
    ):
        candidates.append((0, match.start(), match.end()))
    candidates.extend((1, start, end) for start, end, _reason in find_xhtml_math_intervals(raw))
    markup = markup_ranges(raw, strict=True)
    for start, end in markup:
        if raw.startswith("<![CDATA[", start):
            candidates.extend(((2, start, start + 9), (2, end - 3, end)))
        else:
            candidates.append((2, start, end))
    for match in _URL_RE.finditer(raw):
        end = match.end()
        while end > match.start() and raw[end - 1] in ".,;:!?":
            end -= 1
        while end > match.start() and raw[end - 1] in ")]}":
            pair = {")": "(", "]": "[", "}": "{"}[raw[end - 1]]
            if raw[match.start():end].count(pair) >= raw[match.start():end].count(raw[end - 1]):
                break
            end -= 1
        # Protect the whole visible URL before math punctuation such as '='.
        # Attribute URLs already belong to an immutable markup token.
        if not any(match.start() < stop and end > begin for begin, stop in markup):
            candidates.append((0, match.start(), end))
    # A literal reserved-looking token in a book is itself protected.
    candidates.extend((3, match.start(), match.end()) for match in TOKEN_RE.finditer(raw))
    chosen: list[tuple[int, int]] = []
    for _priority, start, end in sorted(candidates, key=lambda item: (item[0], item[1], -item[2])):
        if start < end and not any(start < stop and end > begin for begin, stop in chosen):
            chosen.append((start, end))
    fragments: list[ProtectedFragment] = []
    parts: list[str] = []
    cursor = 0
    cdata = False
    for index, (start, end) in enumerate(sorted(chosen)):
        original = raw[start:end]
        digest = hashlib.sha256(f"{start}:{end}:{original}".encode()).hexdigest()[:8].upper()
        token = f"[[PH_{index:04d}_{digest}]]"
        while token in raw:
            digest = hashlib.sha256(digest.encode()).hexdigest()[:8].upper()
            token = f"[[PH_{index:04d}_{digest}]]"
        plain = raw[cursor:start]
        parts.extend((plain if cdata else html.unescape(plain), token))
        fragments.append(ProtectedFragment(token, original))
        if original == "<![CDATA[":
            cdata = True
        elif original == "]]>":
            cdata = False
        cursor = end
    tail = raw[cursor:]
    parts.append(tail if cdata else html.unescape(tail))
    return "".join(parts), tuple(fragments)


def extract_segments(source: str) -> XHTMLDocument:
    validate_xhtml(source)
    roots = _nodes(source)
    ranges: list[tuple[int, int, str]] = []

    def contains_block(node: _Node) -> bool:
        return node.name in _BLOCK_TAGS or any(contains_block(child) for child in node.children)

    def walk(node: _Node) -> None:
        if node.name in _OPAQUE_TAGS:
            return
        boundaries = [child for child in node.children if contains_block(child)]
        cursor = node.inner_start
        for child in boundaries:
            if cursor < child.start:
                ranges.append((cursor, child.start, node.name))
            walk(child)
            cursor = child.end
        if cursor < node.inner_end:
            ranges.append((cursor, node.inner_end, node.name))

    found_body = False
    def visit(node: _Node) -> None:
        nonlocal found_body
        if node.name == "body":
            found_body = True
            walk(node)
        elif node.name == "title":
            walk(node)
        else:
            for child in node.children:
                visit(child)

    for root in roots:
        visit(root)
    if not found_body:
        raise ValueError("Invalid XHTML: missing body element")
    segments: list[TextSegment] = []
    for start, end, kind in sorted(ranges):
        original = source[start:end]
        protected, fragments = _protect(original)
        if not TOKEN_RE.sub("", protected).strip():
            continue
        segments.append(TextSegment(f"s{len(segments) + 1:06d}", start, end, kind,
                                    protected, original, fragments))
    return XHTMLDocument(source, tuple(segments))


def _looks_like_names(text: str) -> bool:
    # Middle initials and all-capital ampersand company names are names, not
    # the prose words "a"/"at" (John A. Smith, AT&T).
    name_text = re.sub(r"\b[A-Z]\.\s*", "", text)
    name_text = re.sub(r"\b[A-Z]{1,5}(?:&[A-Z]{1,5})+\b", "Company", name_text)
    # An index's trailing page references are not English prose.
    name_text = re.sub(r',\s*[ivxlcdm]+(?:[–—-][ivxlcdm]+)?[\s,;–—-]*$', '', name_text, flags=re.IGNORECASE)
    words = _WORD_RE.findall(name_text)
    if not words or any(word.lower() in (_PROSE_WORDS - _NAME_CONNECTORS - {"new", "may", "will"}) for word in words):
        return False
    if re.search(r"[\"“”!?:]|(?:^|\s)[‘']", text):
        return False
    return all(word.lower() in _NAME_CONNECTORS or word[0].isupper()
               or re.search(r"[a-z][A-Z]", word) for word in words)


def _requires_chinese(segment: TextSegment) -> bool:
    text = segment.plain_text
    words = _WORD_RE.findall(text)
    if not words:
        return False
    if _SINGLE_LETTER_RE.fullmatch(text):
        return False  # preserved without a request (see needs_request), like numbers
    if re.fullmatch(r"[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}", text):
        return False  # UK postal identifier, e.g. EC1A 7JQ
    if segment.kind in {"title", "h1", "h2", "h3", "h4", "h5", "h6"}:
        # Ordinary uppercase headings (CLOUD, POWER, MIX) are not acronyms or
        # page numbers merely because of their capitalization.
        return text not in {"AI", "US", "UK", "EU", "UN", "GDP", "CPU", "GPU", "IT",
                            "API", "ISBN", "DOI", "PDF", "EPUB", "CSS", "HTML"}
    if text and re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", text, re.IGNORECASE):
        return False
    if len(words) == 1 and words[0].isupper() and len(words[0]) <= 5:
        return words[0].lower() in _PROSE_WORDS
    if segment.kind not in {"title", "h1", "h2", "h3", "h4", "h5", "h6"} and _looks_like_names(text):
        return False
    return True


def validate_segment_translation(segment: TextSegment, translated: str, *, check_quality: bool = True) -> None:
    if not isinstance(translated, str) or not translated.strip():
        raise SegmentValidationError("empty or non-string translation")
    if TOKEN_RE.findall(translated) != [fragment.token for fragment in segment.fragments]:
        raise SegmentValidationError("placeholder count/order changed")
    if "[[PH_" in TOKEN_RE.sub("", translated):
        raise SegmentValidationError("unknown or malformed placeholder")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", translated):
        raise SegmentValidationError("invalid XML character in translation")
    plain = TOKEN_RE.sub("", translated).strip()
    if not plain:
        raise SegmentValidationError("visible text was removed")
    if not check_quality:
        return
    if _META_RE.search(plain) and not _META_RE.search(segment.plain_text):
        raise SegmentValidationError("assistant explanation in translation")
    if _requires_chinese(segment):
        count = len(_CJK_RE.findall(plain))
        words = _WORD_RE.findall(segment.plain_text)
        minimum = max(1, len(words) // 5) if len(words) >= 16 else 1
        if count < minimum:
            raise SegmentValidationError(f"insufficient Chinese coverage ({count} < {minimum})")
    # Stripping placeholders first joins dropcaps/italic spans back into words.
    for run in _ENGLISH_RUN_RE.findall(plain):
        if len(_WORD_RE.findall(run)) >= 8 and not _looks_like_names(run):
            raise SegmentValidationError("untranslated long English passage")


def _restore(segment: TextSegment, translated: str) -> str:
    if translated == segment.source:
        return segment.original  # identity preserves even original entity spelling
    fragments = {fragment.token: fragment.original for fragment in segment.fragments}
    parts: list[str] = []
    cursor = 0
    cdata = False
    for match in TOKEN_RE.finditer(translated):
        plain = translated[cursor:match.start()]
        if cdata and "]]>" in plain:
            raise SegmentValidationError("CDATA terminator in translation")
        parts.append(plain if cdata else html.escape(plain, quote=False))
        original = fragments[match.group()]
        parts.append(original)
        if original == "<![CDATA[":
            cdata = True
        elif original == "]]>":
            cdata = False
        cursor = match.end()
    tail = translated[cursor:]
    parts.append(tail if cdata else html.escape(tail, quote=False))
    return "".join(parts)


def make_payload(segments: Sequence[TextSegment]) -> str:
    return json.dumps({segment.id: segment.source for segment in segments}, ensure_ascii=False)


def _normalize_dropcap(segment: TextSegment, translated: str) -> str:
    """Move the Chinese initial into the source's decorative drop-cap span.

    Some providers retain the Latin initial while translating its complete
    word after the span (B + 到2022年 for 'By 2022'). Only repair an initial
    source span followed by the rest of a lowercase word; standalone initials
    and acronyms are unaffected.
    """
    original = re.match(r'^\s*<span\b[^>]*>([A-Za-z])</span>(?=[a-z])', segment.original)
    if not original or len(segment.fragments) < 2 or not isinstance(translated, str):
        return translated
    first, second = segment.fragments[:2]
    if not first.original.startswith('<span') or second.original != '</span>':
        return translated
    pattern = (r'^(\s*)' + re.escape(first.token) + re.escape(original.group(1))
               + re.escape(second.token) + r'\s*([\u3400-\u4dbf\u4e00-\u9fff])')
    return re.sub(pattern, lambda match: match.group(1) + first.token + match.group(2) + second.token,
                  translated, count=1)


_INVISIBLE_ANCHOR_RE = re.compile(
    r"<(a|span)(\s[^<>]*?)?\s*(?:/>|>\s*</\1\s*>)", re.IGNORECASE | re.DOTALL)
_INVISIBLE_OPEN_RE = re.compile(r"<(a|span)(\s[^<>]*?)?\s*>", re.IGNORECASE | re.DOTALL)
_HREF_RE = re.compile(r"(?:^|[\s:])href\s*=", re.IGNORECASE)  # also xlink:href
_STYLE_RE = re.compile(r"(?:^|\s)style\s*=", re.IGNORECASE)  # styled empties can render


def _anchor_attributes_invisible(attributes: str | None) -> bool:
    return not _HREF_RE.search(attributes or "") and not _STYLE_RE.search(attributes or "")


def _invisible_anchor_units(segment: TextSegment) -> list[tuple[int, ...]]:
    """Fragment index groups that are empty ``a``/``span`` elements without href.

    A unit is one fragment (``<span id="x"/>``, ``<a id="x"/>``, ``<a id="x"></a>``)
    or an opening tag immediately followed in the source by its closing tag.
    """
    fragments = segment.fragments
    units: list[tuple[int, ...]] = []
    index = 0
    while index < len(fragments):
        original = fragments[index].original
        match = _INVISIBLE_ANCHOR_RE.fullmatch(original)
        if match and _anchor_attributes_invisible(match.group(2)):
            units.append((index,))
            index += 1
            continue
        match = _INVISIBLE_OPEN_RE.fullmatch(original)
        if (match and _anchor_attributes_invisible(match.group(2)) and index + 1 < len(fragments)
                and re.fullmatch(r"</" + match.group(1) + r"\s*>", fragments[index + 1].original,
                                 re.IGNORECASE)
                and fragments[index].token + fragments[index + 1].token in segment.source):
            units.append((index, index + 1))
            index += 2
            continue
        index += 1
    return units


def _english_kept_from_source(segment: TextSegment, translated: str) -> bool:
    """Every 8+ word English run is a contiguous word sequence of the source.

    Words are compared as _WORD_RE lists (curly/straight apostrophes folded),
    so a kept title survives changed quote marks while appended notes do not.
    """
    def words(value: str) -> list[str]:
        return [word.replace("’", "'") for word in _WORD_RE.findall(value)]

    source = words(segment.plain_text)
    for run in _ENGLISH_RUN_RE.findall(TOKEN_RE.sub("", translated)):
        run_words = words(run)
        if len(run_words) < 8:
            continue
        width = len(run_words)
        if not any(source[start:start + width] == run_words
                   for start in range(len(source) - width + 1)):
            return False
    return True


def repair_invisible_anchors(segment: TextSegment, translated: str) -> str:
    """Put dropped or moved zero-width anchors back between their visible neighbours.

    Only presence and order of an empty anchor matter, so a response whose
    visible placeholders are exactly right gets every invisible anchor
    re-inserted in source order, proportionally within its window. Anything
    else is returned unchanged for validation to judge.
    """
    if not isinstance(translated, str):
        return translated
    expected = [fragment.token for fragment in segment.fragments]
    if TOKEN_RE.findall(translated) == expected:
        return translated
    units = _invisible_anchor_units(segment)
    if not units or TOKEN_RE.findall(segment.source) != expected:
        return translated
    hidden = {index for unit in units for index in unit}
    stripped = translated
    for index in sorted(hidden):
        stripped = stripped.replace(expected[index], "")
    visible = [token for index, token in enumerate(expected) if index not in hidden]
    if TOKEN_RE.findall(stripped) != visible:
        return translated
    # Source layout: plain length of each window between visible tokens and
    # the plain offset of every invisible unit inside its window.
    unit_at = {unit[0]: "".join(expected[index] for index in unit) for unit in units}
    lengths = [0] * (len(visible) + 1)
    offsets: list[list[tuple[int, str]]] = [[] for _ in lengths]
    window = cursor = 0
    for index, match in enumerate(TOKEN_RE.finditer(segment.source)):
        lengths[window] += match.start() - cursor
        cursor = match.end()
        if index in unit_at:
            offsets[window].append((lengths[window], unit_at[index]))
        elif index not in hidden:
            window += 1
    lengths[window] += len(segment.source) - cursor
    windows = TOKEN_RE.split(stripped)
    pieces: list[str] = []
    for window, text in enumerate(windows):
        last = 0
        for offset, tokens in offsets[window]:
            position = int(offset * len(text) / lengths[window] + 0.5) if lengths[window] else 0
            position = max(last, min(len(text), position))
            pieces.extend((text[last:position], tokens))
            last = position
        pieces.append(text[last:])
        if window < len(visible):
            pieces.append(visible[window])
    return "".join(pieces)


def batch_segments(segments: Sequence[TextSegment], max_bytes: int) -> list[tuple[TextSegment, ...]]:
    if max_bytes < 1:
        raise ValueError("batch size must be positive")
    batches: list[tuple[TextSegment, ...]] = []
    current: list[TextSegment] = []
    for segment in segments:
        if current and len(make_payload([*current, segment]).encode("utf-8")) > max_bytes:
            batches.append(tuple(current))
            current = []
        current.append(segment)
    if current:
        batches.append(tuple(current))
    # A single oversized paragraph stays whole; never split an inline context.
    return batches


def parse_translation_response(response: str, segments: Sequence[TextSegment]) -> BatchValidation:
    expected = {segment.id for segment in segments}

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SegmentValidationError(f"duplicate segment ID: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(response, object_pairs_hook=unique_object)
        if not isinstance(decoded, dict):
            raise SegmentValidationError("response must be a JSON ID-to-string object")
        unknown = set(decoded) - expected
        if unknown:
            raise SegmentValidationError("unexpected segment IDs")
    except (TypeError, ValueError) as exc:
        reason = str(exc)[:240]
        return BatchValidation(failures={segment.id: reason for segment in segments})
    result = BatchValidation()
    for segment in segments:
        if segment.id not in decoded:
            result.failures[segment.id] = "missing segment ID"
            continue
        normalized = _normalize_dropcap(segment, decoded[segment.id])
        translated = repair_invisible_anchors(segment, normalized)
        if translated != normalized:
            result.repaired.add(segment.id)
        try:
            validate_segment_translation(segment, translated)
        except SegmentValidationError as exc:
            reason = str(exc)[:240]
            result.failures[segment.id] = reason
            if reason.startswith(_QUALITY_REASONS) and _english_kept_from_source(segment, translated):
                # Kept for the pipeline's bounded final-attempt tolerance;
                # structural rules are re-checked and never waived.
                try:
                    validate_segment_translation(segment, translated, check_quality=False)
                except SegmentValidationError:
                    pass
                else:
                    result.quality_rejects[segment.id] = (translated, reason)
        else:
            result.translations[segment.id] = translated
    return result
