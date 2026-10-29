"""Turn translated paragraphs into the HTML and CSS that PyMuPDF's
htmlbox renders."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from phoena_translator.pdf.model import LayoutClass
from phoena_translator.pdf.semantics.toc import _pdf_toc_entries, _strip_pdf_toc_label
from phoena_translator.pdf.text_shape import _plain_text
from phoena_translator.pdf.translate.inline_markup import _normalize_pdf_translation

PDF_HTMLBOX_SUPERSCRIPT_CSS = (
    'sup{line-height:0;vertical-align:super;white-space:nowrap;}'
)


def _pdf_plain_text_to_html(text: str) -> str:
    normalized = _plain_text(_normalize_pdf_translation(text))
    normalized = re.sub(r'\s*\n\s*', ' ', normalized)
    escaped = html.escape(normalized)
    return _wrap_pdf_latin_runs(escaped)


def _wrap_pdf_latin_runs(text: str) -> str:
    parts = re.split(r'(&(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);)', text)
    wrapped = []
    for part in parts:
        if not part:
            continue
        if part.startswith("&") and part.endswith(";"):
            wrapped.append(part)
            continue
        wrapped.append(re.sub(
            r'([A-Za-z][A-Za-z0-9.,\-\'&;/():%]*(?:\s+[A-Za-z][A-Za-z0-9.,\-\'&;/():%]*)*)',
            r'<span class="en">\1</span>',
            part,
        ))
    return "".join(wrapped)


def _pdf_text_to_html(text: str, superscript_scale: float = 0.60) -> str:
    """Escape translated text while preserving supported inline markup."""
    parser = _PDFInlineHTMLParser(superscript_scale)
    parser.feed(_normalize_pdf_translation(text))
    parser.close()
    return parser.finish()


def _build_paragraph_html(
    element: dict,
    source_paragraphs: list[dict],
    translated_paragraphs: list[str],
    fontsize: float,
    color_hex: str,
    line_height: float,
) -> list[str]:
    """Render paragraph metadata into deterministic HTML fragments."""

    fragments = []
    for metadata, translated_text in zip(source_paragraphs, translated_paragraphs):
        if not translated_text.strip():
            continue
        margin_top = max(0.0, metadata.get("gap_before", 0.0))
        margin_left = max(0.0, metadata.get("margin_left", 0.0))
        text_indent = metadata.get("text_indent", 0.0)
        text_align = metadata.get("text_align", "left")
        nowrap = metadata.get("nowrap", False)
        toc_metadata = metadata.get("toc_leader")
        source_toc_entries = _pdf_toc_entries(metadata.get("plain", ""))
        translated_toc_entries = _pdf_toc_entries(translated_text)
        entries_match = (
            len(source_toc_entries) > 1
            and len(source_toc_entries) == len(translated_toc_entries)
            and [page for _, page in source_toc_entries]
            == [page for _, page in translated_toc_entries]
        )
        if entries_match:
            rows = []
            for (_, page_reference), (translated_label, _) in zip(
                source_toc_entries,
                translated_toc_entries,
            ):
                safe_label = _pdf_plain_text_to_html(translated_label)
                safe_page = html.escape(page_reference)
                rows.append(
                    "<tr>"
                    f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{safe_label}</td>'
                    '<td style="padding:0 4px;vertical-align:baseline;white-space:nowrap;'
                    'width:3.2em;text-align:center;">......</td>'
                    f'<td style="padding:0;vertical-align:baseline;white-space:nowrap;'
                    f'width:3.2em;text-align:right;">{safe_page}</td>'
                    "</tr>"
                )
            fragments.append(
                f'<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;padding:0;"
                f"font-size:{fontsize:.1f}px;color:{color_hex};"
                f'line-height:{line_height:.1f}px;">{"".join(rows)}</table>'
            )
        elif toc_metadata:
            label = (
                _strip_pdf_toc_label(translated_text, toc_metadata["page"])
                or toc_metadata["label"]
            )
            safe_label = _pdf_plain_text_to_html(label)
            safe_page = html.escape(toc_metadata["page"])
            fragments.append(
                f'<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;padding:0;"
                f"font-size:{fontsize:.1f}px;color:{color_hex};"
                f'line-height:{line_height:.1f}px;"><tr>'
                f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{safe_label}</td>'
                '<td style="padding:0 4px;vertical-align:baseline;white-space:nowrap;'
                'width:3.2em;text-align:center;">......</td>'
                '<td style="padding:0;vertical-align:baseline;white-space:nowrap;'
                f'width:3.2em;text-align:right;">{safe_page}</td></tr></table>'
            )
        else:
            safe_text = _pdf_text_to_html(
                translated_text,
                superscript_scale=float(element.get("superscript_scale") or 0.60),
            )
            single_line_heading = bool(element.get("single_line_heading"))
            white_space = (
                "white-space:nowrap;"
                if single_line_heading
                or (element.get("layout_class", "body") != LayoutClass.BODY and nowrap)
                else ""
            )
            fragments.append(
                f'<p style="font-size:{fontsize:.1f}px;color:{color_hex};'
                f"line-height:{line_height:.1f}px;"
                f"margin:{margin_top:.1f}px 0 0 {margin_left:.1f}px;"
                f"padding:0;text-indent:{text_indent:.1f}px;"
                f'text-align:{text_align};{white_space}">{safe_text}</p>'
            )
    return fragments


def _split_pdf_translated_paragraphs(text: str, expected_count: int) -> list[str]:
    normalized = _normalize_pdf_translation(text)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', normalized) if p.strip()]

    if expected_count <= 1:
        if not paragraphs and normalized.strip():
            return [re.sub(r'\s+', ' ', normalized).strip()]
        return [re.sub(r'\s+', ' ', p).strip() for p in paragraphs] or [""]

    if len(paragraphs) != expected_count:
        line_paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in normalized.splitlines() if p.strip()]
        if len(line_paragraphs) == expected_count:
            paragraphs = line_paragraphs

    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        paragraphs = [re.sub(r'\s+', ' ', normalized).strip()] if normalized.strip() else [""]

    if len(paragraphs) < expected_count:
        paragraphs.extend([""] * (expected_count - len(paragraphs)))
    elif len(paragraphs) > expected_count:
        paragraphs = paragraphs[:expected_count - 1] + [" ".join(paragraphs[expected_count - 1:]).strip()]

    return paragraphs


class _PDFInlineHTMLParser(HTMLParser):
    def __init__(self, superscript_scale: float):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
        self.superscript_scale = max(0.45, min(float(superscript_scale), 0.80))

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in {"b", "sup"}:
            # Unknown angle-bracketed material is source text, not markup.
            # Preserve it visibly and escape it before insert_htmlbox; this
            # covers domain notation such as ``<JPY short position>`` without
            # allowing an arbitrary provider-emitted tag into the HTML.
            self.parts.append(html.escape(
                self.get_starttag_text() or f"<{tag}>"
            ))
            return
        if tag == "b":
            self.parts.append("<b>")
        else:
            percent = self.superscript_scale * 100.0
            self.parts.append(
                '&#8288;'
                f'<sup style="font-size:{percent:.1f}%;line-height:0;vertical-align:super">'
            )
        self.stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag not in {"b", "sup"}:
            self.parts.append(html.escape(f"</{tag}>"))
            return
        if tag not in self.stack:
            return
        while self.stack:
            open_tag = self.stack.pop()
            self.parts.append(f"</{open_tag}>")
            if open_tag == tag:
                break

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        if tag not in {"b", "sup"}:
            self.parts.append(html.escape(
                self.get_starttag_text() or f"<{tag}/>"
            ))
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        escaped = html.escape(re.sub(r'\s*\n\s*', ' ', data))
        self.parts.append(_wrap_pdf_latin_runs(escaped))

    def finish(self) -> str:
        while self.stack:
            self.parts.append(f"</{self.stack.pop()}>")
        return "".join(self.parts)
