"""Build paragraphs from native lines: line grouping, list-marker and
drop-cap handling, paragraph geometry, and splitting elements into
semantic fragments."""

from __future__ import annotations

import fitz
import re
import logging
import html
from phoena_translator.pdf.text_shape import (
    _collapsed_plain_text,
    _ends_with_sentence_boundary,
    _is_heading_like_elem,
    _join_pdf_line_fragments,
    _looks_like_heading_text,
    _looks_like_pdf_list_marker,
    _looks_like_pdf_paragraph_lead,
    _pdf_has_terminal_sentence_boundary,
    _pdf_text_colors_semantically_compatible,
    _pick_pdf_dominant_value,
    _plain_text,
)
from phoena_translator.pdf.superscripts import (
    _pdf_source_line_footnote_definition_marker,
)
from phoena_translator.pdf.model import ElementType
from phoena_translator.pdf.toc import _parse_pdf_toc_leader
from phoena_translator.pdf.placeholders import (
    _select_pdf_inline_math_fragments_for_text,
)
from phoena_translator.pdf.geometry import (
    _detect_pdf_text_align,
    _get_pdf_elem_rect,
    _get_pdf_render_bbox,
)


_PDF_FOOTNOTE_MARKER_RE = re.compile(r"^(?:\d{1,3}|[*\u2020\u2021\u00a7\u00b6]{1,3})$")
log = logging.getLogger("translator")


def _looks_like_pdf_drop_cap_token(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    letters = re.findall(r"[A-Za-z]", compact)
    if len(letters) != 1:
        return False
    if letters[0] != letters[0].upper():
        return False
    return len(compact) <= 3


def _detect_pdf_drop_cap_span(span_entries: list[dict]) -> dict | None:
    non_empty = [span for span in span_entries if span.get("text", "").strip()]
    if len(non_empty) < 2:
        return None

    first = non_empty[0]
    if not _looks_like_pdf_drop_cap_token(first.get("text", "")):
        return None

    body_entries = [
        span
        for span in non_empty[1:]
        if re.search(r"[A-Za-z0-9]", span.get("text", ""))
    ]
    if not body_entries:
        body_entries = non_empty[1:]
    if not body_entries:
        return None

    body_size = _pick_pdf_dominant_value(
        [
            (float(span.get("size", 0.0)), max(len(span.get("text", "").strip()), 1))
            for span in body_entries
        ],
        float(body_entries[0].get("size", 0.0)),
    )
    if not body_size:
        return None

    gap = max(float(body_entries[0]["x0"]) - float(first["x1"]), 0.0)
    body_text = "".join(span.get("text", "") for span in body_entries)
    if len(re.sub(r"\s+", "", body_text)) < 3:
        return None
    if float(first.get("size", 0.0)) < float(body_size) * 1.6:
        return None
    if gap > max(float(body_size) * 2.0, 18.0):
        return None

    return {
        "body_size": float(body_size),
        "body_y0": min(float(span["y0"]) for span in body_entries),
        "body_y1": max(float(span["y1"]) for span in body_entries),
    }


def _make_pdf_space_span(prev_span: dict, next_span: dict) -> dict:
    return {
        "text": " ",
        "rich": " ",
        "x0": prev_span["x1"],
        "x1": next_span["x0"],
        "y0": min(prev_span["y0"], next_span["y0"]),
        "y1": max(prev_span["y1"], next_span["y1"]),
        "size": prev_span["size"],
        "color": prev_span["color"],
        "bold": False,
        "flags": 0,
        "superscript": False,
        "superscript_source": None,
        "superscript_scale": None,
    }


def _merge_pdf_list_marker_clusters(clusters: list[list[dict]]) -> list[list[dict]]:
    if len(clusters) < 2:
        return clusters

    first_text = "".join(span["text"] for span in clusters[0])
    second_text = "".join(span["text"] for span in clusters[1])
    if not _looks_like_pdf_list_marker(first_text):
        return clusters
    if not second_text.strip():
        return clusters

    merged = list(clusters[0])
    merged.append(_make_pdf_space_span(clusters[0][-1], clusters[1][0]))
    merged.extend(clusters[1])
    return [merged] + clusters[2:]


def _looks_like_split_layout_line(
    clusters: list[list[dict]], line_bbox, block_rect, fontsize: float
) -> bool:
    if len(clusters) < 2 or len(clusters) > 3:
        return False

    cell_texts = [
        re.sub(r"\s+", " ", "".join(span["text"] for span in cluster)).strip()
        for cluster in clusters
    ]
    if any(not text for text in cell_texts):
        return False
    if (
        max(len(text) for text in cell_texts) > 40
        or sum(len(text) for text in cell_texts) > 120
    ):
        return False

    token_count = sum(
        len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)) for text in cell_texts
    )
    if token_count > 12:
        return False

    max_gap = max(
        clusters[idx + 1][0]["x0"] - clusters[idx][-1]["x1"]
        for idx in range(len(clusters) - 1)
    )
    line_width = max(float(line_bbox[2]) - float(line_bbox[0]), 1.0)
    occupied_width = sum(
        max(cluster[-1]["x1"] - cluster[0]["x0"], 1.0) for cluster in clusters
    )
    width_ratio = occupied_width / line_width
    right_span = (clusters[-1][-1]["x1"] - block_rect.x0) / max(block_rect.width, 1.0)

    return (
        max_gap >= max(fontsize * 5.0, block_rect.width * 0.12, 36.0)
        and width_ratio <= 0.75
        and right_span >= 0.6
    )


def _should_keep_pdf_lines_separate(
    line_infos: list[dict], block_rect, fontsize: float
) -> bool:
    if len(line_infos) < 2 or len(line_infos) > 8:
        return False

    plain_lines = [re.sub(r"\s+", " ", line["plain"]).strip() for line in line_infos]
    if any(not text for text in plain_lines):
        return False
    if max(len(text) for text in plain_lines) > 40:
        return False
    if sum(len(text) for text in plain_lines) > 180:
        return False

    align = _detect_pdf_text_align(line_infos, block_rect, fontsize)
    if align == "left":
        return False

    x_spread = max(line["x0"] for line in line_infos) - min(
        line["x0"] for line in line_infos
    )
    return x_spread <= max(fontsize * 2.5, 24.0)


def _is_new_pdf_paragraph(
    prev_line: dict,
    curr_line: dict,
    baseline_x0: float,
    avg_height: float,
    fontsize: float,
) -> bool:
    indent_threshold = max(fontsize * 0.9, avg_height * 0.8, 8.0)
    gap_threshold = max(fontsize * 0.55, avg_height * 0.4, 4.0)

    vertical_gap = curr_line["y0"] - prev_line["y1"]
    if vertical_gap > gap_threshold:
        return True

    x_shift = curr_line["x0"] - prev_line["x0"]
    curr_from_base = curr_line["x0"] - baseline_x0
    prev_from_base = prev_line["x0"] - baseline_x0
    if abs(x_shift) > indent_threshold and (
        curr_from_base > indent_threshold * 0.5
        or prev_from_base > indent_threshold * 0.5
    ):
        return True

    if curr_from_base > indent_threshold and prev_from_base <= indent_threshold * 0.5:
        return True

    return False


def _split_pdf_disjoint_line_segments(
    line_infos: list[dict],
    fontsize: float,
) -> list[list[dict]]:
    """Separate unrelated table cells that PyMuPDF put in one text block.

    Some compact tables encode several visual columns as consecutive PDF
    lines.  Joining two short lines whose horizontal ranges are completely
    disjoint creates a large diagonal bounding box.  That box can cross a
    formula in a neighboring column, causing the whole block to be preserved
    as math and allowing a later redaction to clip unrelated source glyphs.
    Keep ordinary wrapped prose together; split only short, nearby lines with
    a material horizontal gap and no shared text column.
    """
    if len(line_infos) < 2:
        return [list(line_infos)] if line_infos else []

    segments: list[list[dict]] = []
    current = [line_infos[0]]
    for line in line_infos[1:]:
        previous = current[-1]
        previous_text = re.sub(r"\s+", " ", previous.get("plain", "")).strip()
        current_text = re.sub(r"\s+", " ", line.get("plain", "")).strip()
        previous_width = max(float(previous["x1"]) - float(previous["x0"]), 1.0)
        current_width = max(float(line["x1"]) - float(line["x0"]), 1.0)
        horizontal_overlap = max(
            0.0,
            min(float(previous["x1"]), float(line["x1"]))
            - max(float(previous["x0"]), float(line["x0"])),
        )
        overlap_ratio = horizontal_overlap / min(previous_width, current_width)
        horizontal_gap = max(
            float(line["x0"]) - float(previous["x1"]),
            float(previous["x0"]) - float(line["x1"]),
            0.0,
        )
        previous_height = max(float(previous["y1"]) - float(previous["y0"]), 1.0)
        current_height = max(float(line["y1"]) - float(line["y0"]), 1.0)
        vertical_gap = float(line["y0"]) - float(previous["y1"])
        nearby = vertical_gap <= max(fontsize * 1.25, previous_height, current_height)
        short_cells = (
            bool(previous_text)
            and bool(current_text)
            and len(previous_text) <= 80
            and len(current_text) <= 80
        )
        split_here = (
            short_cells
            and nearby
            and overlap_ratio < 0.10
            and horizontal_gap >= max(fontsize * 1.2, 10.0)
            and not _looks_like_pdf_list_marker(previous_text)
            and not _pdf_marked_run_in_wraps_to_next_line(
                previous,
                line,
                fontsize,
            )
        )
        if split_here:
            segments.append(current)
            current = [line]
        else:
            current.append(line)
    segments.append(current)
    return segments


def _pdf_marked_run_in_wraps_to_next_line(
    first_line: dict,
    next_line: dict,
    fontsize: float,
) -> bool:
    """Protect a proven run-in start from the generic table-cell splitter."""
    if not first_line.get("same_baseline_paragraph_start"):
        return False
    if (
        first_line.get("table_hint")
        or next_line.get("table_hint")
        or first_line.get("math_protected")
        or next_line.get("math_protected")
        or first_line.get("non_horizontal")
        or next_line.get("non_horizontal")
        or first_line.get("drop_cap")
        or next_line.get("drop_cap")
    ):
        return False

    first_plain = re.sub(r"\s+", " ", first_line.get("plain", "")).strip()
    next_plain = re.sub(r"\s+", " ", next_line.get("plain", "")).strip()
    if (
        not first_plain
        or not next_plain
        or _pdf_has_terminal_sentence_boundary(first_plain)
        or _looks_like_pdf_paragraph_lead(next_plain)
        or _looks_like_pdf_list_marker(next_plain)
        or _pdf_source_line_footnote_definition_marker(next_line)
    ):
        return False

    first_size = max(float(first_line.get("fontsize", fontsize)), 1.0)
    next_size = max(float(next_line.get("fontsize", fontsize)), 1.0)
    if abs(first_size - next_size) > max(fontsize * 0.12, 0.8):
        return False
    if bool(first_line.get("bold")) != bool(next_line.get("bold")):
        return False
    if not _pdf_text_colors_semantically_compatible(
        first_line.get("color", 0),
        next_line.get("color", 0),
    ):
        return False

    first_height = max(
        float(first_line["y1"]) - float(first_line["y0"]),
        1.0,
    )
    next_height = max(
        float(next_line["y1"]) - float(next_line["y0"]),
        1.0,
    )
    vertical_gap = float(next_line["y0"]) - float(first_line["y1"])
    if vertical_gap < -max(first_height, next_height) * 0.35:
        return False
    if vertical_gap > max(fontsize * 0.95, first_height, next_height, 8.0):
        return False

    indent_threshold = max(fontsize * 0.9, min(first_height, next_height) * 0.8, 8.0)
    return float(first_line["x0"]) - float(next_line["x0"]) >= indent_threshold * 0.55


def _is_pdf_first_line_indent_continuation(
    first_line: dict,
    next_line: dict,
    baseline_x0: float,
    block_rect: fitz.Rect,
    avg_height: float,
    fontsize: float,
) -> bool:
    """Recognize a normal indented first line followed by its body line.

    Fully justified prose commonly has a first line inset from the left while
    still touching the right edge. Treating that geometry as a paragraph
    boundary leaves the first line alone, where it is easily misclassified as
    right-aligned. Keep the lines together and let ``text_indent`` preserve the
    inset instead.
    """
    indent_threshold = max(fontsize * 0.9, avg_height * 0.8, 8.0)
    gap_threshold = max(fontsize * 0.55, avg_height * 0.4, 4.0)
    vertical_gap = float(next_line["y0"]) - float(first_line["y1"])
    if vertical_gap > gap_threshold or vertical_gap < -avg_height * 0.6:
        return False

    first_indent = float(first_line["x0"]) - float(baseline_x0)
    next_from_base = abs(float(next_line["x0"]) - float(baseline_x0))
    if first_indent < indent_threshold * 0.55:
        return False
    if next_from_base > indent_threshold * 0.45:
        return False

    block_width = max(float(block_rect.width), 1.0)
    first_width = max(float(first_line["x1"]) - float(first_line["x0"]), 0.0)
    right_gap = max(float(block_rect.x1) - float(first_line["x1"]), 0.0)
    marked_run_in_start = bool(first_line.get("same_baseline_paragraph_start"))
    if not marked_run_in_start and first_width < block_width * 0.65:
        return False
    if right_gap > max(fontsize * 0.65, 6.0):
        return False

    first_size = float(first_line.get("fontsize", fontsize))
    next_size = float(next_line.get("fontsize", fontsize))
    if abs(first_size - next_size) > max(fontsize * 0.12, 0.8):
        return False
    if bool(first_line.get("bold")) != bool(next_line.get("bold")):
        return False

    first_plain = re.sub(r"\s+", " ", first_line.get("plain", "")).strip()
    if (
        (not marked_run_in_start and len(first_plain) < 24)
        or _looks_like_pdf_list_marker(first_plain)
        or _pdf_has_terminal_sentence_boundary(first_plain)
    ):
        return False
    return True


def _is_pdf_hanging_list_continuation(
    first_line: dict,
    next_line: dict,
    block_rect: fitz.Rect,
    avg_height: float,
    fontsize: float,
) -> bool:
    """Recognize the body line after ``7. ...`` / ``7.1. ...``.

    The first line starts at the list marker while following visual lines use
    a hanging indent.  Treating that x shift as a new paragraph is the exact
    failure mode that made SEBI page 5 translate and center every source line
    independently.
    """
    first_plain = re.sub(r"\s+", " ", first_line.get("plain", "")).strip()
    next_plain = re.sub(r"\s+", " ", next_line.get("plain", "")).strip()
    if not _looks_like_pdf_paragraph_lead(first_plain) or not next_plain:
        return False
    if _looks_like_pdf_paragraph_lead(next_plain) or _looks_like_pdf_list_marker(
        next_plain
    ):
        return False

    vertical_gap = float(next_line["y0"]) - float(first_line["y1"])
    if vertical_gap < -avg_height * 0.35:
        return False
    if vertical_gap > max(fontsize * 0.95, avg_height * 0.90, 8.0):
        return False

    indent = float(next_line["x0"]) - float(first_line["x0"])
    if indent < -max(fontsize * 0.25, 2.0):
        return False
    if indent > max(float(block_rect.width) * 0.28, fontsize * 5.0, 36.0):
        return False

    first_width = max(float(first_line["x1"]) - float(first_line["x0"]), 1.0)
    next_width = max(float(next_line["x1"]) - float(next_line["x0"]), 1.0)
    overlap = max(
        0.0,
        min(float(first_line["x1"]), float(next_line["x1"]))
        - max(float(first_line["x0"]), float(next_line["x0"])),
    )
    return overlap / min(first_width, next_width) >= 0.35


def _build_pdf_paragraphs(
    line_infos: list[dict], block_rect, fontsize: float
) -> tuple[list[dict], float]:
    if not line_infos:
        return [], 0.0

    baseline_x0 = min(line["x0"] for line in line_infos)
    avg_height = sum(max(1.0, line["y1"] - line["y0"]) for line in line_infos) / len(
        line_infos
    )

    top_padding = max(0.0, line_infos[0]["y0"] - block_rect.y0)
    separate_lines_mode = _should_keep_pdf_lines_separate(
        line_infos, block_rect, fontsize
    )
    separate_line_align = (
        _detect_pdf_text_align(line_infos, block_rect, fontsize)
        if separate_lines_mode
        else None
    )
    if separate_lines_mode:
        grouped_lines = [[line] for line in line_infos]
    else:
        grouped_lines = []
        current = [line_infos[0]]
        for line in line_infos[1:]:
            if _pdf_source_line_footnote_definition_marker(line) and any(
                _pdf_source_line_footnote_definition_marker(prior_line)
                for prior_line in current
            ):
                grouped_lines.append(current)
                current = [line]
                continue
            if _looks_like_pdf_paragraph_lead(line.get("plain", "")):
                grouped_lines.append(current)
                current = [line]
                continue
            if line.get("same_baseline_paragraph_start"):
                grouped_lines.append(current)
                current = [line]
                continue
            if len(current) == 1 and _is_pdf_hanging_list_continuation(
                current[0],
                line,
                fitz.Rect(block_rect),
                avg_height,
                fontsize,
            ):
                current.append(line)
                continue
            if len(current) == 1 and _is_pdf_first_line_indent_continuation(
                current[0],
                line,
                baseline_x0,
                fitz.Rect(block_rect),
                avg_height,
                fontsize,
            ):
                current.append(line)
                continue
            if _is_new_pdf_paragraph(
                current[-1], line, baseline_x0, avg_height, fontsize
            ):
                grouped_lines.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            grouped_lines.append(current)

        merged_groups = []
        idx = 0
        while idx < len(grouped_lines):
            current_group = grouped_lines[idx]
            current_text = _join_pdf_line_fragments(
                [line["plain"] for line in current_group]
            )
            if (
                idx + 1 < len(grouped_lines)
                and len(current_group) <= 2
                and _looks_like_pdf_list_marker(current_text)
            ):
                next_group = grouped_lines[idx + 1]
                vertical_gap = next_group[0]["y0"] - current_group[-1]["y1"]
                if vertical_gap <= max(avg_height * 0.9, fontsize * 0.8, 8.0):
                    merged_groups.append(current_group + next_group)
                    idx += 2
                    continue
            merged_groups.append(current_group)
            idx += 1
        grouped_lines = merged_groups

    paragraphs = []
    prev_last_y1 = None
    for lines in grouped_lines:
        plain_text = _join_pdf_line_fragments([line["plain"] for line in lines])
        rich_text = _join_pdf_line_fragments([line["rich"] for line in lines])
        if not plain_text:
            continue

        first_x0 = lines[0]["x0"]
        body_x0 = min((line["x0"] for line in lines[1:]), default=first_x0)
        first_plain = re.sub(r"\s+", " ", lines[0]["plain"]).strip()
        is_list_lead = len(lines) > 1 and (
            _looks_like_pdf_list_marker(first_plain)
            or _looks_like_pdf_paragraph_lead(first_plain)
        )
        has_first_line_indent = len(
            lines
        ) > 1 and _is_pdf_first_line_indent_continuation(
            lines[0],
            lines[1],
            min(line["x0"] for line in lines),
            fitz.Rect(block_rect),
            avg_height,
            fontsize,
        )
        if is_list_lead:
            margin_left = max(0.0, body_x0 - block_rect.x0)
            text_indent = max(-block_rect.width * 0.18, first_x0 - body_x0)
        else:
            margin_left = max(0.0, min(body_x0, first_x0) - block_rect.x0)
            text_indent = max(0.0, first_x0 - body_x0)
        gap_before = (
            0.0 if prev_last_y1 is None else max(0.0, lines[0]["y0"] - prev_last_y1)
        )
        prev_last_y1 = lines[-1]["y1"]
        if is_list_lead:
            text_align = "left"
        elif has_first_line_indent:
            # Stable right edges are a consequence of justified prose here,
            # not evidence that the paragraph itself is right-aligned.
            text_align = "left"
        elif lines[0].get("same_baseline_paragraph_start"):
            # A proven run-in can be the final source line on a page, so it
            # may not yet have a following line from which indentation can be
            # inferred.  Its shared-baseline evidence still rules out a true
            # right-aligned paragraph.
            text_align = "left"
        elif separate_line_align:
            text_align = separate_line_align
        else:
            text_align = _detect_pdf_text_align(lines, block_rect, fontsize)

        paragraphs.append(
            {
                "plain": plain_text,
                "rich": rich_text,
                "source_bbox": [
                    min(float(line["x0"]) for line in lines),
                    min(float(line["y0"]) for line in lines),
                    max(float(line["x1"]) for line in lines),
                    max(float(line["y1"]) for line in lines),
                ],
                "source_line_bboxes": [
                    [
                        float(line["x0"]),
                        float(line["y0"]),
                        float(line["x1"]),
                        float(line["y1"]),
                    ]
                    for line in lines
                ],
                "source_lines": [
                    {
                        "plain": line.get("plain", ""),
                        "rich": line.get("rich", line.get("plain", "")),
                        "bbox": [
                            float(line["x0"]),
                            float(line["y0"]),
                            float(line["x1"]),
                            float(line["y1"]),
                        ],
                        "same_baseline_fragments": [
                            dict(fragment)
                            for fragment in (line.get("same_baseline_fragments") or [])
                        ],
                    }
                    for line in lines
                ],
                "margin_left": min(margin_left, block_rect.width * 0.35),
                "text_indent": max(
                    -block_rect.width * 0.18, min(text_indent, block_rect.width * 0.25)
                ),
                "gap_before": min(gap_before, avg_height * 2.0),
                "text_align": text_align,
                "nowrap": True if separate_lines_mode else None,
                "first_line_indent": has_first_line_indent,
                "same_baseline_paragraph_start": bool(
                    lines[0].get("same_baseline_paragraph_start")
                ),
                "toc_leader": _parse_pdf_toc_leader(plain_text),
            }
        )

    return paragraphs, top_padding


def _normalize_pdf_footnote_alignment(elem: dict, page_rect: fitz.Rect) -> bool:
    """Force bottom-of-page footnote definitions to use a stable left edge.

    A one-line footnote body can have the same right edge as its enclosing PDF
    block. Generic alignment inference then mistakes it for right-aligned text,
    while neighboring wrapped footnotes are detected as left-aligned.
    """
    if elem.get("type") != ElementType.TEXT or elem.get("table_hint"):
        return False

    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) < 2:
        return False

    marker = re.sub(r"\s+", "", paragraphs[0].get("plain", ""))
    if not _PDF_FOOTNOTE_MARKER_RE.fullmatch(marker):
        return False

    rect = _get_pdf_elem_rect(elem)
    page_height = max(page_rect.height, 1.0)
    if rect.y0 < page_rect.y0 + page_height * 0.55:
        return False

    fontsize = max(float(elem.get("fontsize", 0.0)), 1.0)
    marker_margin = float(paragraphs[0].get("margin_left", 0.0))
    body_margins = [
        float(paragraph.get("margin_left", 0.0)) for paragraph in paragraphs[1:]
    ]
    if not body_margins or min(body_margins) < marker_margin + max(fontsize * 0.6, 4.0):
        return False

    body_text = " ".join(
        (paragraph.get("plain") or "").strip() for paragraph in paragraphs[1:]
    ).strip()
    if len(body_text) < 4:
        return False

    for paragraph in paragraphs:
        paragraph["text_align"] = "left"
    elem["footnote_hint"] = True
    return True


def _make_pdf_text_element_from_lines(
    line_infos: list[dict],
    page_rect: fitz.Rect,
    preserve_source_style: bool = False,
    table_hint: bool = False,
) -> dict | None:
    if not line_infos:
        return None

    # Whitespace-only PDF lines are often positioning artifacts at the far
    # edge of an open table.  Excluding them before geometry is computed keeps
    # the element/redaction rectangle tied to visible glyphs.  Previously a
    # left-hand row label could inherit an x1 near the right page edge and its
    # redaction erased every numeric cell on the row.
    line_infos = [line for line in line_infos if (line.get("plain") or "").strip()]
    if not line_infos:
        return None

    group_rect = fitz.Rect(
        min(line["x0"] for line in line_infos),
        min(line["y0"] for line in line_infos),
        max(line["x1"] for line in line_infos),
        max(line["y1"] for line in line_infos),
    )

    font_weights = [
        (
            float(line.get("fontsize", 11.0)),
            max(len((line.get("plain") or "").strip()), 1),
        )
        for line in line_infos
    ]
    fontsize = float(_pick_pdf_dominant_value(font_weights, 11.0))

    paragraph_lines = [
        {
            "plain": line.get("plain", ""),
            "rich": line.get("rich", ""),
            "x0": line["x0"],
            "x1": line["x1"],
            "y0": line["y0"],
            "y1": line["y1"],
            "fontsize": line.get("fontsize", fontsize),
            "bold": bool(line.get("bold")),
            "color": line.get("color", 0),
            "same_baseline_paragraph_start": bool(
                line.get("same_baseline_paragraph_start")
            ),
            "same_baseline_fragments": [
                dict(fragment)
                for fragment in (line.get("same_baseline_fragments") or [])
            ],
        }
        for line in line_infos
    ]

    paragraphs, top_padding = _build_pdf_paragraphs(
        paragraph_lines, group_rect, fontsize
    )
    if table_hint:
        # Table-aware extraction emits one element per visual cell / source
        # line. Keep it on one line so fitting scales inside the cell instead
        # of turning neighboring columns into extra rows.
        for paragraph in paragraphs:
            paragraph["nowrap"] = True
    plain_text = "\n\n".join(
        paragraph["plain"] for paragraph in paragraphs if paragraph["plain"]
    ).strip()
    rich_text = "\n\n".join(
        paragraph["rich"] for paragraph in paragraphs if paragraph["rich"]
    ).strip()
    if not plain_text:
        return None

    color_weights = [
        (line.get("color", 0), max(len((line.get("plain") or "").strip()), 1))
        for line in line_infos
    ]
    color = int(_pick_pdf_dominant_value(color_weights, 0))
    plain_chars = sum(len((line.get("plain") or "").strip()) for line in line_infos)
    bold_chars = sum(
        len((line.get("plain") or "").strip())
        for line in line_infos
        if line.get("bold")
    )
    superscript_runs = [
        dict(run) for line in line_infos for run in (line.get("superscript_runs") or [])
    ]
    inline_math_fragments = [
        dict(fragment)
        for line in line_infos
        for fragment in (line.get("inline_math_fragments") or [])
        if (fragment.get("text") or "").strip()
    ]
    has_inline_bold = bool(
        rich_text
        and rich_text != plain_text
        and 0
        < sum(
            len(_plain_text(match).strip())
            for match in re.findall(
                r"(?is)<b\b[^>]*>(.*?)</b>",
                rich_text,
            )
        )
        < plain_chars * 0.9
    )
    has_inline_markup = has_inline_bold or bool(superscript_runs)
    superscript_scale = _pick_pdf_dominant_value(
        [
            (
                round(float(run.get("scale", 0.60)), 3),
                max(len(run.get("text", "").strip()), 1),
            )
            for run in superscript_runs
        ],
        0.60,
    )
    line_height_values = [
        max(
            float(line.get("line_height", 0.0)),
            float(line.get("fontsize", fontsize)) * 1.05,
        )
        for line in line_infos
    ]
    avg_line_height = (
        sum(line_height_values) / len(line_height_values)
        if line_height_values
        else fontsize * 1.2
    )

    return {
        "type": ElementType.TEXT,
        "y": group_rect.y0,
        "x": group_rect.x0,
        "rect": group_rect,
        "bbox": [group_rect.x0, group_rect.y0, group_rect.x1, group_rect.y1],
        "render_bbox": _get_pdf_render_bbox(
            group_rect, page_rect, paragraphs, fontsize
        ),
        "content": plain_text,
        "rich_content": rich_text if has_inline_markup else None,
        "superscript_runs": superscript_runs,
        "superscript_scale": float(superscript_scale) if superscript_runs else None,
        "paragraphs": paragraphs,
        "top_padding": top_padding,
        "fontsize": fontsize,
        "line_height": avg_line_height,
        "color": color,
        "bold": (bold_chars > plain_chars * 0.5) if plain_chars > 0 else False,
        "non_horizontal": any(bool(line.get("non_horizontal")) for line in line_infos),
        "rotation": int(
            _pick_pdf_dominant_value(
                [
                    (
                        int(line.get("rotation", 0)),
                        max(len((line.get("plain") or "").strip()), 1),
                    )
                    for line in line_infos
                ],
                0,
            )
        ),
        "inline_math_fragments": inline_math_fragments,
        "preserve_source_style": bool(preserve_source_style),
        "table_hint": bool(table_hint),
    }


def _is_pdf_single_visual_line_element(elem: dict) -> bool:
    """Return whether an extracted text element represents one visual line."""
    if (
        elem.get("type") != ElementType.TEXT
        or elem.get("table_hint")
        or elem.get("preserve_source_style")
        or elem.get("non_horizontal")
        or elem.get("skip_translate_reason")
    ):
        return False
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) != 1 or paragraphs[0].get("toc_leader"):
        return False
    plain = _collapsed_plain_text(elem.get("content", ""))
    if not plain or "\n" in (elem.get("content") or ""):
        return False
    rect = _get_pdf_elem_rect(elem)
    fontsize = max(float(elem.get("fontsize", 11.0)), 1.0)
    line_height = max(float(elem.get("line_height", fontsize * 1.1)), 1.0)
    return rect.height <= max(fontsize * 1.85, line_height * 1.45, 22.0)


def _pdf_visual_line_info_from_element(elem: dict) -> dict:
    rect = _get_pdf_elem_rect(elem)
    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    paragraph = paragraphs[0] if paragraphs else {}
    plain = _collapsed_plain_text(elem.get("content", ""))
    rich = elem.get("rich_content") or paragraph.get("rich") or plain
    return {
        "plain": plain,
        "rich": rich,
        "x0": float(rect.x0),
        "x1": float(rect.x1),
        "y0": float(rect.y0),
        "y1": float(rect.y1),
        "fontsize": float(elem.get("fontsize", 11.0)),
        "line_height": float(elem.get("line_height", max(rect.height, 1.0))),
        "color": int(elem.get("color", 0)),
        "bold": bool(elem.get("bold")),
        "non_horizontal": bool(elem.get("non_horizontal")),
        "rotation": int(elem.get("rotation", 0)),
        "table_hint": False,
        "drop_cap": None,
        "superscript_runs": [dict(run) for run in (elem.get("superscript_runs") or [])],
        "inline_math_fragments": [
            dict(fragment) for fragment in (elem.get("inline_math_fragments") or [])
        ],
        "math_protected": False,
        "math_reasons": [],
        "math_mixed": False,
        "math_symbol_count": 0,
    }


def _rebase_pdf_split_paragraph_layout(
    paragraph: dict,
    paragraph_rect: fitz.Rect,
) -> dict:
    """Rebase paragraph offsets after detaching it from a parent text block.

    ``margin_left`` and ``text_indent`` are relative to the rectangle in which
    HTML is rendered.  A paragraph split out of a wide parent block therefore
    cannot retain the parent's offsets.  Reconstruct hanging / first-line
    indentation from its own source-line geometry and otherwise start at zero.
    """
    rebased = dict(paragraph)
    source_lines = [
        dict(line)
        for line in (paragraph.get("source_lines") or [])
        if (line.get("plain") or "").strip() and line.get("bbox")
    ]
    if not source_lines:
        source_lines = [
            {
                "plain": paragraph.get("plain", ""),
                "bbox": [
                    float(paragraph_rect.x0),
                    float(paragraph_rect.y0),
                    float(paragraph_rect.x1),
                    float(paragraph_rect.y1),
                ],
            }
        ]

    first_x0 = float(source_lines[0]["bbox"][0])
    body_x0 = min(
        (float(line["bbox"][0]) for line in source_lines[1:]),
        default=first_x0,
    )
    first_plain = _collapsed_plain_text(source_lines[0].get("plain", ""))
    is_hanging_lead = len(source_lines) > 1 and (
        _looks_like_pdf_list_marker(first_plain)
        or _looks_like_pdf_paragraph_lead(first_plain)
    )
    has_first_line_indent = bool(
        len(source_lines) > 1 and paragraph.get("first_line_indent")
    )

    if is_hanging_lead:
        margin_left = max(0.0, body_x0 - paragraph_rect.x0)
        text_indent = first_x0 - body_x0
    elif has_first_line_indent:
        margin_left = max(0.0, min(first_x0, body_x0) - paragraph_rect.x0)
        text_indent = first_x0 - body_x0
    else:
        margin_left = max(
            0.0,
            min(float(line["bbox"][0]) for line in source_lines) - paragraph_rect.x0,
        )
        text_indent = 0.0

    rebased["margin_left"] = min(
        margin_left,
        max(paragraph_rect.width * 0.35, 0.0),
    )
    rebased["text_indent"] = max(
        -paragraph_rect.width * 0.18,
        min(text_indent, paragraph_rect.width * 0.25),
    )
    return rebased


def _split_pdf_text_elements_into_semantic_fragments(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """Emit one positioned text element per semantic source paragraph.

    A PDF text block may contain several paragraphs.  Keeping that block as a
    single translation unit couples unrelated paragraphs, while drawing it as
    one rectangle also makes natural paragraph wrapping impossible.  The
    paragraph builder already records exact source geometry, so split only
    elements with complete paragraph geometry and leave tables, rotated text,
    and marker-plus-body footnotes intact.
    """
    fragments: list[dict] = []
    for elem in elements:
        paragraphs = [
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ]
        if (
            elem.get("type") != ElementType.TEXT
            or len(paragraphs) <= 1
            or elem.get("table_hint")
            or elem.get("preserve_source_style")
            or elem.get("non_horizontal")
            or any(not paragraph.get("source_bbox") for paragraph in paragraphs)
        ):
            fragments.append(elem)
            continue

        footnote_probe = dict(elem)
        footnote_probe["paragraphs"] = [dict(paragraph) for paragraph in paragraphs]
        if _normalize_pdf_footnote_alignment(footnote_probe, page_rect):
            fragments.append(elem)
            continue

        for paragraph in paragraphs:
            plain = _collapsed_plain_text(paragraph.get("plain", ""))
            rich = (paragraph.get("rich") or plain).strip()
            if not plain:
                continue
            rect = fitz.Rect(paragraph["source_bbox"])
            clone = dict(elem)
            paragraph_copy = dict(paragraph)
            paragraph_copy.update(
                {
                    "plain": plain,
                    "rich": rich,
                    "source_bbox": [float(value) for value in rect],
                    "gap_before": 0.0,
                }
            )
            paragraph_copy = _rebase_pdf_split_paragraph_layout(
                paragraph_copy,
                rect,
            )
            source_lines = [
                dict(line)
                for line in (paragraph.get("source_lines") or [])
                if (line.get("plain") or "").strip()
            ]
            source_line_bboxes = [
                [float(value) for value in bbox]
                for bbox in (paragraph.get("source_line_bboxes") or [])
            ]
            superscript_texts = [
                _plain_text(html.unescape(match)).strip()
                for match in re.findall(
                    r"<sup(?:\s[^>]*)?>(.*?)</sup>",
                    rich,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if _plain_text(html.unescape(match)).strip()
            ]
            superscript_scale = float(elem.get("superscript_scale") or 0.60)
            superscript_runs = [
                {
                    "text": text,
                    "scale": superscript_scale,
                    "source": "semantic-paragraph-split",
                }
                for text in superscript_texts
            ]
            inline_math_fragments = _select_pdf_inline_math_fragments_for_text(
                elem.get("inline_math_fragments") or [],
                plain,
            )
            clone.update(
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
                        [paragraph_copy],
                        float(elem.get("fontsize", 11.0)),
                    ),
                    "content": plain,
                    "rich_content": rich if rich != plain or superscript_runs else None,
                    "superscript_runs": superscript_runs,
                    "superscript_scale": superscript_scale
                    if superscript_runs
                    else None,
                    "inline_math_fragments": inline_math_fragments,
                    "paragraphs": [paragraph_copy],
                    "top_padding": 0.0,
                    "merged_visual_line_count": max(
                        len(source_lines), len(source_line_bboxes), 1
                    ),
                    "source_line_bboxes": source_line_bboxes
                    or [
                        [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                    ],
                }
            )
            fragments.append(clone)
    return fragments


def _normalize_pdf_drop_cap_lines(
    line_infos: list[dict], block_rect: fitz.Rect
) -> list[dict]:
    if not line_infos:
        return []

    normalized = [dict(line) for line in line_infos]
    first_drop_cap = normalized[0].get("drop_cap") or {}
    if len(normalized) == 1 and not first_drop_cap:
        return normalized

    body_fontsize = _pick_pdf_dominant_value(
        [
            (
                float(line.get("fontsize", 0.0)),
                max(len(re.sub(r"\s+", "", line.get("plain", ""))), 1),
            )
            for line in normalized[1:]
            if (line.get("plain") or "").strip()
        ],
        float(first_drop_cap.get("body_size", normalized[0].get("fontsize", 11.0))),
    )
    body_fontsize = max(
        float(body_fontsize or normalized[0].get("fontsize", 11.0)), 6.0
    )

    first = normalized[0]
    drop_cap = first.get("drop_cap") or {}
    if drop_cap:
        first["x0"] = float(block_rect.x0)
        first["fontsize"] = body_fontsize
        first["y0"] = max(
            float(block_rect.y0), float(drop_cap.get("body_y0", first["y0"]))
        )
        body_y1 = float(drop_cap.get("body_y1", first["y1"]))
        first["y1"] = max(first["y0"] + max(body_fontsize * 1.05, 1.0), body_y1)
        first["line_height"] = max(first["y1"] - first["y0"], body_fontsize * 1.1)
        first["drop_cap_normalized"] = True

    first_plain = re.sub(r"\s+", "", first.get("plain", ""))
    if (
        len(normalized) >= 2
        and _looks_like_pdf_drop_cap_token(first_plain)
        and float(first.get("fontsize", 0.0)) >= body_fontsize * 1.6
    ):
        second = dict(normalized[1])
        second["plain"] = (
            f"{first.get('plain', '').strip()}{second.get('plain', '').lstrip()}"
        )
        second["rich"] = (
            f"{first.get('rich', '').strip()}{second.get('rich', '').lstrip()}"
        )
        second["x0"] = float(block_rect.x0)
        second["fontsize"] = body_fontsize
        second["line_height"] = max(
            float(second.get("line_height", 0.0)), body_fontsize * 1.1
        )
        second["drop_cap_normalized"] = True
        normalized = [second] + normalized[2:]

    return normalized


def _pdf_element_prefers_single_line_heading(elem: dict, page_rect: fitz.Rect) -> bool:
    """Return whether a short source heading should stay on one rendered line."""
    if (
        elem.get("type") != ElementType.TEXT
        or elem.get("table_hint")
        or elem.get("non_horizontal")
    ):
        return False

    paragraphs = [
        paragraph
        for paragraph in (elem.get("paragraphs") or [])
        if (paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) != 1:
        return False

    plain = _collapsed_plain_text(elem.get("content", ""))
    if not plain or len(plain) > 96:
        return False

    rect = _get_pdf_elem_rect(elem)
    fontsize = max(float(elem.get("fontsize", 11.0)), 1.0)
    if rect.height > max(fontsize * 1.9, 24.0):
        return False
    if rect.width > max(page_rect.width * 0.78, fontsize * 42.0):
        return False

    word_count = len(re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]", plain))
    bold_short_heading = (
        bool(elem.get("bold"))
        and word_count <= 16
        and not re.search(r"[.!?。！？]\s*$", plain)
    )
    return _looks_like_heading_text(plain) or bold_short_heading


def _merge_adjacent_heading_elements(elements: list[dict]) -> list[dict]:
    """Re-join a heading that wraps onto a second line.

    PDF extraction often yields the wrapped heading as two separate short bold
    elements; if the first one ends mid-phrase (no terminal punctuation) and
    the next bold line sits directly below at the same size, translate them as
    one unit so the title reads as a single sentence."""
    merged = []
    for elem in elements:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("type") == ElementType.TEXT
            and elem.get("type") == ElementType.TEXT
            and _is_heading_like_elem(prev)
            and _is_heading_like_elem(elem)
            and not prev.get("table_hint")
            and not elem.get("table_hint")
            and not prev.get("non_horizontal")
            and not elem.get("non_horizontal")
            and len(_plain_text(prev.get("content", ""))) <= 200
            and len(_plain_text(elem.get("content", ""))) <= 200
            and abs(
                float(prev.get("fontsize", 11.0)) - float(elem.get("fontsize", 11.0))
            )
            <= 0.6
            and not _ends_with_sentence_boundary(prev.get("content", ""))
        ):
            prev_rect = fitz.Rect(prev["bbox"])
            cur_rect = fitz.Rect(elem["bbox"])
            gap = cur_rect.y0 - prev_rect.y1
            fontsize = float(prev.get("fontsize", 11.0))
            x_overlap = min(prev_rect.x1, cur_rect.x1) - max(prev_rect.x0, cur_rect.x0)
            if -fontsize * 0.6 <= gap <= fontsize * 1.6 and x_overlap > 10.0:
                union = prev_rect | cur_rect
                text = prev["content"].rstrip() + " " + elem["content"].lstrip()
                first_para = (prev.get("paragraphs") or [{}])[0]
                prev["content"] = text
                prev["rich_content"] = None
                prev["paragraphs"] = [
                    {
                        "plain": text,
                        "rich": text,
                        "margin_left": 0.0,
                        "text_indent": 0.0,
                        "gap_before": 0.0,
                        "text_align": first_para.get("text_align", "center"),
                    }
                ]
                prev["rect"] = union
                prev["bbox"] = [union.x0, union.y0, union.x1, union.y1]
                prev["render_bbox"] = [union.x0, union.y0, union.x1, union.y1]
                prev["line_height"] = max(
                    float(prev.get("line_height", fontsize * 1.2)),
                    float(elem.get("line_height", fontsize * 1.2)),
                )
                continue
        merged.append(elem)
    return merged
