"""Semantic references helpers for deterministic PDF processing."""

from __future__ import annotations

import html
import logging
import re

import fitz
from phoena_translator.math_text import (
    is_unicode_math_symbol as _is_unicode_math_symbol,
)

from phoena_translator.pdf.types import (
    PDF_TRANSLATABLE_CITATION_LABELS,
    _PDF_CITATION_PUBLISHER_CONNECTORS,
)
from phoena_translator.pdf.geometry import (
    _get_pdf_primary_paragraph_rect,
    _get_pdf_render_bbox,
)
from phoena_translator.pdf.math_detection import (
    _plain_text,
    _select_pdf_inline_math_fragments_for_text,
)

log = logging.getLogger("translator")

from phoena_translator.pdf.semantic_text import (
    _is_heading_like_elem,
    _join_pdf_line_fragments,
    _pick_pdf_dominant_value,
)


def _looks_like_pdf_reference_entry_lead(text: str) -> bool:
    """Recognize the first visual line of a bibliography entry.

    Bibliographies commonly use a left-aligned author/year lead followed by
    indented continuation lines.  The year alone is not enough evidence (it
    also occurs in ordinary prose), so require an uppercase initial and
    author-like punctuation before a nearby four-digit year.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text or "")).strip()
    if len(plain) < 18:
        return False
    if not plain[0].isalpha() or not plain[0].isupper():
        return False
    year_match = re.search(
        r"(?<!\d)(?:18\d{2}|19\d{2}|20[0-2]\d)[a-z]?(?!\d)",
        plain[:180],
    )
    if not year_match:
        return False
    author_prefix = plain[: year_match.start()]
    return bool(
        "," in author_prefix
        or "(" in author_prefix
        or "&" in author_prefix
        or re.search(r"\b(?:and|et\s+al)\b", author_prefix, re.IGNORECASE)
    )


def _looks_like_pdf_citation_publisher(text: str) -> bool:
    """Recognize a short institutional publisher after a citation separator."""
    publisher = (text or "").strip(" \t\r\n.,;:")
    if not re.fullmatch(
        r"[A-Za-z][A-Za-z&.'’/-]*(?:\s+[A-Za-z][A-Za-z&.'’/-]*){1,9}",
        publisher,
    ):
        return False
    words = publisher.split()
    return all(
        word.lower() in _PDF_CITATION_PUBLISHER_CONNECTORS
        or word[0].isupper()
        or word.isupper()
        for word in words
    )


def _pdf_citation_separator_pipe_count(text: str) -> int:
    """Count pipes that separate numbered footnotes from their publishers.

    Some PDFs place two consecutive footnotes in one final semantic element.
    The native line extractor records each ``|`` correctly, but later merging
    can retain only the first inline-math record.  This recognizer is narrow:
    it requires a numbered citation lead, consecutive entry numbers, only
    pipe-shaped math symbols, whitespace-delimited separators, and a short
    title-cased institutional publisher after every separator.
    """
    plain = re.sub(r"\s+", " ", _plain_text(text)).strip()
    lead = re.match(r"^(?P<number>\d{1,3})\s+\S", plain)
    if not lead:
        return 0
    pipes = list(re.finditer(r"(?<!\S)\|(?!\S)", plain))
    if not pipes or len(pipes) != plain.count("|"):
        return 0
    math_symbols = [char for char in plain if _is_unicode_math_symbol(char)]
    if math_symbols != ["|"] * len(pipes):
        return 0

    entry_number = int(lead.group("number"))
    for index, pipe in enumerate(pipes):
        if index + 1 < len(pipes):
            between = plain[pipe.end() : pipes[index + 1].start()]
            next_entry = re.search(r"\s+(\d{1,3})\s+(?=[A-Z])", between)
            if not next_entry or int(next_entry.group(1)) != entry_number + 1:
                return 0
            publisher = between[: next_entry.start()]
            entry_number = int(next_entry.group(1))
        else:
            publisher = plain[pipe.end() :]
        if not _looks_like_pdf_citation_publisher(publisher):
            return 0
    return len(pipes)


def _complete_pdf_citation_separator_inline_math_fragments(
    elements: list[dict],
) -> int:
    """Restore citation-separator records lost by late semantic coalescing."""
    added = 0
    for elem in elements or []:
        if elem.get("type") != "text":
            continue
        plain = _plain_text(elem.get("content", ""))
        expected = _pdf_citation_separator_pipe_count(plain)
        if not expected:
            continue
        fragments = [
            dict(fragment)
            for fragment in (elem.get("inline_math_fragments") or [])
            if isinstance(fragment, dict)
        ]
        existing = sum(
            1 for fragment in fragments if (fragment.get("text") or "").strip() == "|"
        )
        missing = max(expected - existing, 0)
        if not missing:
            continue
        fragments.extend(
            {"text": "|", "font_kind": "citation_separator"} for _ in range(missing)
        )
        elem["inline_math_fragments"] = _select_pdf_inline_math_fragments_for_text(
            fragments, plain
        )
        added += missing
    return added


def _pdf_reference_entry_lead_counts(
    elements: list[dict],
) -> tuple[bool, dict[int, int]]:
    """Return reference-page context and author/year lead counts by element."""
    text_elements = [
        (index, elem)
        for index, elem in enumerate(elements)
        if elem.get("type") == "text"
    ]
    normalized_texts = [
        re.sub(
            r"\s+",
            " ",
            _plain_text(elem.get("content", "")),
        )
        .strip()
        .lower()
        for _, elem in text_elements
    ]
    heading_present = any(
        text in {"references", "bibliography"}
        or (len(text) <= 40 and re.search(r"\b(?:references|bibliography)$", text))
        for text in normalized_texts
    )
    candidates: list[tuple[int, float]] = []
    for index, elem in text_elements:
        lines: list[dict] = []
        for paragraph in elem.get("paragraphs") or []:
            lines.extend(
                line
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            )
        if not lines:
            elem_rect = _get_pdf_primary_paragraph_rect(elem)
            lines = [
                {
                    "plain": elem.get("content", ""),
                    "bbox": [float(value) for value in elem_rect],
                }
            ]
        for line in lines:
            if not _looks_like_pdf_reference_entry_lead(line.get("plain", "")):
                continue
            bbox = line.get("bbox") or _get_pdf_primary_paragraph_rect(elem)
            x0 = float(bbox[0])
            candidates.append((index, x0))

    x_clusters: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item[1]):
        if x_clusters and abs(candidate[1] - x_clusters[-1]["mean_x0"]) <= 6.0:
            x_clusters[-1]["members"].append(candidate)
            x_clusters[-1]["mean_x0"] = sum(
                member[1] for member in x_clusters[-1]["members"]
            ) / len(x_clusters[-1]["members"])
        else:
            x_clusters.append({"mean_x0": candidate[1], "members": [candidate]})

    dense_clusters = [cluster for cluster in x_clusters if len(cluster["members"]) >= 2]
    if dense_clusters:
        accepted_clusters = [
            cluster
            for cluster in dense_clusters
            if not any(
                8.0 <= cluster["mean_x0"] - other["mean_x0"] <= 40.0
                for other in dense_clusters
            )
        ]
    elif heading_present:
        accepted_clusters = x_clusters
    else:
        accepted_clusters = []
    accepted_candidates = [
        candidate for cluster in accepted_clusters for candidate in cluster["members"]
    ]
    lead_counts: dict[int, int] = {}
    for index, _ in accepted_candidates:
        lead_counts[index] = lead_counts.get(index, 0) + 1

    lead_total = sum(lead_counts.values())
    reference_context = heading_present or (
        lead_total >= 4 and lead_total * 3 >= max(len(text_elements), 1)
    )
    return bool(lead_counts) and reference_context, lead_counts


def _mark_pdf_reference_entry_elements(elements: list[dict]) -> int:
    """Mark final text elements that are complete bibliography entries.

    The page-context gate prevents an ordinary author-year sentence in body
    prose from receiving reference-specific translation rules.  Marking is
    intentionally done after all semantic splitting/merging and formula
    promotion, so only final API-bound text objects receive the hint.
    """
    reference_context, lead_counts = _pdf_reference_entry_lead_counts(elements)
    marked = 0
    for index, elem in enumerate(elements):
        elem.pop("reference_entry_hint", None)
        if (
            reference_context
            and elem.get("type") == "text"
            and lead_counts.get(index, 0) == 1
            and _looks_like_pdf_reference_entry_lead(
                _plain_text(elem.get("content", ""))
            )
        ):
            elem["reference_entry_hint"] = True
            marked += 1
    return marked


def _make_pdf_reference_fragment_from_source_lines(
    parent: dict,
    source_lines: list[dict],
    page_rect: fitz.Rect,
) -> dict:
    """Clone one bibliography entry from a parent containing several."""
    lines = [dict(line) for line in source_lines if (line.get("plain") or "").strip()]
    rect = fitz.Rect(lines[0]["bbox"])
    for line in lines[1:]:
        rect |= fitz.Rect(line["bbox"])
    plain = _join_pdf_line_fragments(
        [_plain_text(line.get("plain", "")) for line in lines]
    )
    rich = _join_pdf_line_fragments(
        [line.get("rich") or line.get("plain", "") for line in lines]
    )
    fontsize = max(float(parent.get("fontsize", 11.0)), 1.0)
    first_x0 = float(lines[0]["bbox"][0])
    body_x0 = min(
        (float(line["bbox"][0]) for line in lines[1:]),
        default=first_x0,
    )
    hanging_indent = body_x0 - first_x0 >= max(fontsize * 0.50, 3.0)
    if hanging_indent:
        margin_left = max(0.0, body_x0 - rect.x0)
        text_indent = first_x0 - body_x0
    else:
        margin_left = 0.0
        text_indent = 0.0
    paragraph = {
        "plain": plain,
        "rich": rich,
        "source_bbox": [float(value) for value in rect],
        "source_line_bboxes": [
            [float(value) for value in line["bbox"]] for line in lines
        ],
        "source_lines": lines,
        "margin_left": min(margin_left, rect.width * 0.35),
        "text_indent": max(-rect.width * 0.18, text_indent),
        "gap_before": 0.0,
        "text_align": "left",
        "nowrap": None,
        "first_line_indent": False,
        "toc_leader": None,
    }
    superscript_texts = [
        _plain_text(html.unescape(match)).strip()
        for match in re.findall(
            r"<sup(?:\s[^>]*)?>(.*?)</sup>",
            rich,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if _plain_text(html.unescape(match)).strip()
    ]
    superscript_scale = float(parent.get("superscript_scale") or 0.60)
    superscript_runs = [
        {
            "text": text,
            "scale": superscript_scale,
            "source": "reference-entry-split",
        }
        for text in superscript_texts
    ]
    clone = dict(parent)
    for stale_key in (
        "strong_semantic_continuation_merged",
        "reference_entry_merged",
        "skip_translate_reason",
    ):
        clone.pop(stale_key, None)
    clone.update(
        {
            "y": float(rect.y0),
            "x": float(rect.x0),
            "rect": fitz.Rect(rect),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "render_bbox": _get_pdf_render_bbox(
                rect,
                page_rect,
                [paragraph],
                fontsize,
            ),
            "content": plain,
            "rich_content": rich if rich != plain or superscript_runs else None,
            "paragraphs": [paragraph],
            "superscript_runs": superscript_runs,
            "superscript_scale": superscript_scale if superscript_runs else None,
            "inline_math_fragments": _select_pdf_inline_math_fragments_for_text(
                parent.get("inline_math_fragments") or [],
                plain,
            ),
            "merged_visual_line_count": len(lines),
            "source_line_bboxes": [
                [float(value) for value in line["bbox"]] for line in lines
            ],
            "reference_entry_split": True,
            "table_hint": False,
            "preserve_source_style": False,
        }
    )
    return clone


def _split_pdf_reference_entry_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> tuple[list[dict], int]:
    """Split a PDF element that contains two or more bibliography entries."""
    reference_context, lead_counts = _pdf_reference_entry_lead_counts(elements)
    if not reference_context:
        return elements, 0

    fragments: list[dict] = []
    split_count = 0
    for index, elem in enumerate(elements):
        if elem.get("type") != "text" or lead_counts.get(index, 0) < 1:
            fragments.append(elem)
            continue
        source_lines = [
            dict(line)
            for paragraph in (elem.get("paragraphs") or [])
            for line in (paragraph.get("source_lines") or [])
            if (line.get("plain") or "").strip()
        ]
        candidate_indices = [
            line_index
            for line_index, line in enumerate(source_lines)
            if _looks_like_pdf_reference_entry_lead(line.get("plain", ""))
        ]
        if not candidate_indices:
            fragments.append(elem)
            continue
        lead_x0 = min(float(source_lines[pos]["bbox"][0]) for pos in candidate_indices)
        lead_indices = [
            pos
            for pos in candidate_indices
            if abs(float(source_lines[pos]["bbox"][0]) - lead_x0) <= 6.0
        ]
        if lead_indices == [0]:
            fragments.append(elem)
            continue

        boundaries = sorted(set([0, *lead_indices, len(source_lines)]))
        made = 0
        for start, end in zip(boundaries, boundaries[1:]):
            segment = source_lines[start:end]
            if not segment:
                continue
            fragments.append(
                _make_pdf_reference_fragment_from_source_lines(
                    elem,
                    segment,
                    page_rect,
                )
            )
            made += 1
        split_count += max(made - 1, 0)
    fragments.sort(
        key=lambda elem: (
            elem["y"],
            elem.get(
                "x",
                elem.get("bbox", [0])[0] if isinstance(elem.get("bbox"), list) else 0,
            ),
        )
    )
    return fragments, split_count


def _pdf_reference_entry_elements_can_merge(
    group: list[dict],
    current: dict,
    page_rect: fitz.Rect,
) -> bool:
    """Return whether ``current`` is an indented continuation of a citation."""
    if not group:
        return False
    first = group[0]
    previous = group[-1]
    for elem in (first, previous, current):
        if (
            elem.get("type") != "text"
            or elem.get("non_horizontal")
            or elem.get("watermark")
            or elem.get("layout_class") in {"heading", "footer"}
            or _is_heading_like_elem(elem)
        ):
            return False
        paragraphs = [
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ]
        if len(paragraphs) != 1 or paragraphs[0].get("toc_leader"):
            return False

    first_plain = re.sub(r"\s+", " ", _plain_text(first.get("content", ""))).strip()
    current_plain = re.sub(r"\s+", " ", _plain_text(current.get("content", ""))).strip()
    if not _looks_like_pdf_reference_entry_lead(first_plain) or not current_plain:
        return False

    first_rect = _get_pdf_primary_paragraph_rect(first)
    previous_rect = _get_pdf_primary_paragraph_rect(previous)
    current_rect = _get_pdf_primary_paragraph_rect(current)
    fontsize = max(float(previous.get("fontsize", 11.0)), 1.0)
    current_fontsize = max(float(current.get("fontsize", 11.0)), 1.0)
    if abs(fontsize - current_fontsize) > max(fontsize * 0.18, 1.1):
        return False

    # A new bibliography entry returns to the author margin.  A continuation
    # is normally indented, but an OCR/style split may remain on that margin;
    # an explicit author/year lead is therefore the decisive stop signal.
    if _looks_like_pdf_reference_entry_lead(current_plain) and abs(
        current_rect.x0 - first_rect.x0
    ) <= max(fontsize * 0.70, 5.0):
        return False

    vertical_gap = current_rect.y0 - previous_rect.y1
    if vertical_gap < -min(previous_rect.height, current_rect.height) * 0.30:
        return False
    if vertical_gap > max(fontsize * 0.65, 6.5):
        return False

    x_shift_from_lead = current_rect.x0 - first_rect.x0
    if x_shift_from_lead < -max(fontsize * 0.30, 3.0):
        return False
    if x_shift_from_lead > max(fontsize * 6.0, page_rect.width * 0.12, 54.0):
        return False

    overlap = max(
        0.0,
        min(previous_rect.x1, current_rect.x1) - max(previous_rect.x0, current_rect.x0),
    )
    if overlap / max(min(previous_rect.width, current_rect.width), 1.0) < 0.20:
        return False
    return True


def _merge_pdf_reference_entry_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> tuple[list[dict], int]:
    """Coalesce every hanging bibliography entry into one semantic element."""
    reference_context, _ = _pdf_reference_entry_lead_counts(elements)
    if not reference_context:
        return elements, 0

    merged_elements: list[dict] = []
    merge_count = 0
    index = 0
    while index < len(elements):
        first = elements[index]
        first_plain = re.sub(r"\s+", " ", _plain_text(first.get("content", ""))).strip()
        if not _looks_like_pdf_reference_entry_lead(first_plain):
            merged_elements.append(first)
            index += 1
            continue

        group = [first]
        cursor = index + 1
        while cursor < len(elements) and _pdf_reference_entry_elements_can_merge(
            group,
            elements[cursor],
            page_rect,
        ):
            group.append(elements[cursor])
            cursor += 1

        if len(group) < 2:
            merged_elements.append(first)
            index += 1
            continue

        rect = _get_pdf_primary_paragraph_rect(group[0])
        for elem in group[1:]:
            rect |= _get_pdf_primary_paragraph_rect(elem)
        plain = _join_pdf_line_fragments(
            [
                re.sub(r"\s+", " ", _plain_text(elem.get("content", ""))).strip()
                for elem in group
            ]
        )
        rich = _join_pdf_line_fragments(
            [
                (elem.get("rich_content") or elem.get("content", "")).strip()
                for elem in group
            ]
        )
        source_lines: list[dict] = []
        for elem in group:
            paragraph = next(
                (
                    paragraph
                    for paragraph in (elem.get("paragraphs") or [])
                    if (paragraph.get("plain") or "").strip()
                ),
                {},
            )
            lines = [
                dict(line)
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            if lines:
                source_lines.extend(lines)
            else:
                elem_rect = _get_pdf_primary_paragraph_rect(elem)
                source_lines.append(
                    {
                        "plain": elem.get("content", ""),
                        "rich": elem.get("rich_content") or elem.get("content", ""),
                        "bbox": [float(value) for value in elem_rect],
                    }
                )

        first_x0 = float(source_lines[0]["bbox"][0])
        body_x0 = min(
            (float(line["bbox"][0]) for line in source_lines[1:]),
            default=first_x0,
        )
        fontsize = float(
            _pick_pdf_dominant_value(
                [
                    (
                        float(elem.get("fontsize", 11.0)),
                        max(len(_plain_text(elem.get("content", "")).strip()), 1),
                    )
                    for elem in group
                ],
                11.0,
            )
        )
        hanging_indent = body_x0 - first_x0 >= max(fontsize * 0.50, 3.0)
        if hanging_indent:
            margin_left = max(0.0, body_x0 - rect.x0)
            text_indent = first_x0 - body_x0
        else:
            margin_left = 0.0
            text_indent = 0.0
        paragraph = {
            "plain": plain,
            "rich": rich,
            "source_bbox": [float(value) for value in rect],
            "source_line_bboxes": [
                [float(value) for value in line["bbox"]] for line in source_lines
            ],
            "source_lines": source_lines,
            "margin_left": min(margin_left, rect.width * 0.35),
            "text_indent": max(-rect.width * 0.18, text_indent),
            "gap_before": 0.0,
            "text_align": "left",
            "nowrap": None,
            "first_line_indent": False,
            "toc_leader": None,
        }
        merged = dict(group[0])
        merged.update(
            {
                "y": float(rect.y0),
                "x": float(rect.x0),
                "rect": fitz.Rect(rect),
                "bbox": [
                    float(rect.x0),
                    float(rect.y0),
                    float(rect.x1),
                    float(rect.y1),
                ],
                "render_bbox": _get_pdf_render_bbox(
                    rect,
                    page_rect,
                    [paragraph],
                    fontsize,
                ),
                "content": plain,
                "rich_content": rich
                if any(elem.get("rich_content") for elem in group)
                else None,
                "paragraphs": [paragraph],
                "fontsize": fontsize,
                "line_height": max(
                    float(elem.get("line_height", fontsize * 1.1)) for elem in group
                ),
                "color": int(
                    _pick_pdf_dominant_value(
                        [
                            (
                                int(elem.get("color", 0)),
                                max(
                                    len(_plain_text(elem.get("content", "")).strip()), 1
                                ),
                            )
                            for elem in group
                        ],
                        0,
                    )
                ),
                "bold": all(bool(elem.get("bold")) for elem in group),
                "superscript_runs": [
                    dict(run)
                    for elem in group
                    for run in (elem.get("superscript_runs") or [])
                ],
                "inline_math_fragments": [
                    dict(fragment)
                    for elem in group
                    for fragment in (elem.get("inline_math_fragments") or [])
                ],
                "merged_visual_line_count": len(source_lines),
                "source_line_bboxes": [
                    [float(value) for value in line["bbox"]] for line in source_lines
                ],
                "reference_entry_merged": len(group),
                "table_hint": False,
                "preserve_source_style": False,
            }
        )
        merged_elements.append(merged)
        merge_count += len(group) - 1
        index = cursor
    return merged_elements, merge_count


def _pdf_reference_continuation_residual_pairs(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[tuple[int, int]]:
    reference_context, _ = _pdf_reference_entry_lead_counts(elements)
    if not reference_context:
        return []
    return [
        (index, index + 1)
        for index in range(len(elements) - 1)
        if _pdf_reference_entry_elements_can_merge(
            [elements[index]],
            elements[index + 1],
            page_rect,
        )
    ]


def _pdf_reference_entry_structure_residuals(
    elements: list[dict],
) -> list[tuple[int, int]]:
    """Find bibliography elements that still contain multiple entries.

    A reference page is recognized by its heading or by a dense population
    of author/year lead lines.  Within that context, more than one such lead
    in one redraw element proves that two semantic entries were coupled.
    """
    reference_context, lead_counts = _pdf_reference_entry_lead_counts(elements)
    if not reference_context:
        return []
    return [(index, count) for index, count in lead_counts.items() if count > 1]


def _translate_pdf_citation_labels(text: str) -> str:
    """Translate fixed bibliographic labels without touching dates or URLs."""
    translated = text or ""
    for pattern, replacement in PDF_TRANSLATABLE_CITATION_LABELS:
        translated = pattern.sub(replacement, translated)
    return translated
