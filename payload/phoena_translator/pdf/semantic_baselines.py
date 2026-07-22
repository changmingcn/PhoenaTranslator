"""Deterministic reconstruction of prose split across one PDF baseline."""

from __future__ import annotations

import re

import fitz

from phoena_translator.pdf.math_detection import _looks_like_pdf_list_marker
from phoena_translator.pdf.semantic_text import (
    _join_pdf_line_fragments,
    _looks_like_pdf_paragraph_lead,
    _pdf_has_terminal_sentence_boundary,
    _pdf_source_line_footnote_definition_marker,
    _pdf_text_colors_semantically_compatible,
)


def _pdf_same_baseline_geometry_compatible(first: dict, second: dict) -> bool:
    """Return whether two fragments may form one row despite bold changes."""
    if (
        first.get("table_hint")
        or second.get("table_hint")
        or first.get("non_horizontal")
        or second.get("non_horizontal")
        or first.get("drop_cap")
        or second.get("drop_cap")
    ):
        return False

    first_size = max(float(first.get("fontsize", 11.0)), 1.0)
    second_size = max(float(second.get("fontsize", 11.0)), 1.0)
    if abs(first_size - second_size) > max(first_size * 0.08, 0.7):
        return False
    return _pdf_text_colors_semantically_compatible(
        first.get("color", 0),
        second.get("color", 0),
    )


def _pdf_same_baseline_style_compatible(first: dict, second: dict) -> bool:
    return (
        _pdf_same_baseline_geometry_compatible(first, second)
        and bool(first.get("bold")) == bool(second.get("bold"))
    )


def _pdf_same_baseline_candidate_group(
    line_infos: list[dict],
    index: int,
) -> tuple[list[dict], int]:
    """Collect at most three adjacent records on one visual baseline."""
    group = [dict(line_infos[index])]
    cursor = index + 1
    while cursor < len(line_infos) and len(group) < 3:
        previous = group[-1]
        current = line_infos[cursor]
        previous_height = max(
            float(previous["y1"]) - float(previous["y0"]),
            1.0,
        )
        current_height = max(
            float(current["y1"]) - float(current["y0"]),
            1.0,
        )
        vertical_overlap = max(
            0.0,
            min(float(previous["y1"]), float(current["y1"]))
            - max(float(previous["y0"]), float(current["y0"])),
        )
        overlap_ratio = vertical_overlap / min(previous_height, current_height)
        gap = float(current["x0"]) - float(previous["x1"])
        fontsize = max(
            float(previous.get("fontsize", 11.0)),
            float(current.get("fontsize", 11.0)),
            1.0,
        )
        if (
            overlap_ratio < 0.80
            or gap < 0.0
            or gap > max(fontsize * 1.25, 12.0)
            or not _pdf_same_baseline_geometry_compatible(previous, current)
        ):
            break
        group.append(dict(current))
        cursor += 1
    return group, cursor


def _pdf_repeated_column_group_starts(line_infos: list[dict]) -> set[int]:
    """Find repeated same-baseline columns that are likely an undetected table."""
    candidates: list[tuple[int, list[dict]]] = []
    index = 0
    while index < len(line_infos):
        group, cursor = _pdf_same_baseline_candidate_group(line_infos, index)
        if len(group) >= 2:
            candidates.append((index, group))
            index = cursor
        else:
            index += 1

    repeated: set[int] = set()
    for (first_index, first), (second_index, second) in zip(
        candidates,
        candidates[1:],
    ):
        if len(first) != len(second) or any(
            line.get("math_protected") for line in (*first, *second)
        ):
            continue
        first_y = min(float(line["y0"]) for line in first)
        second_y = min(float(line["y0"]) for line in second)
        fontsize = max(
            float(line.get("fontsize", 11.0)) for line in (*first, *second)
        )
        if second_y <= first_y + 0.5 or second_y - first_y > max(
            fontsize * 2.2,
            18.0,
        ):
            continue

        tolerance = max(2.0, fontsize * 0.35)
        outer_edges_repeat = (
            abs(float(first[0]["x0"]) - float(second[0]["x0"])) <= tolerance
            and abs(float(first[-1]["x1"]) - float(second[-1]["x1"]))
            <= tolerance
        )
        corresponding_columns_repeat = all(
            abs(float(left["x0"]) - float(right["x0"])) <= tolerance
            or abs(float(left["x1"]) - float(right["x1"])) <= tolerance
            for left, right in zip(first, second)
        )
        dividers_repeat = all(
            abs(float(first[col]["x1"]) - float(second[col]["x1"]))
            <= tolerance
            and abs(float(first[col + 1]["x0"]) - float(second[col + 1]["x0"]))
            <= tolerance
            for col in range(len(first) - 1)
        )
        previous_tail = re.sub(
            r"\s+",
            " ",
            first[-1].get("plain", ""),
        ).strip()
        next_head = re.sub(
            r"\s+",
            " ",
            second[0].get("plain", ""),
        ).strip()
        prose_wrap = bool(
            previous_tail
            and next_head
            and not _pdf_has_terminal_sentence_boundary(previous_tail)
            and next_head[0].islower()
        )
        if (
            outer_edges_repeat
            and corresponding_columns_repeat
            and dividers_repeat
            and not prose_wrap
        ):
            repeated.update({first_index, second_index})
    return repeated


def _looks_like_pdf_bare_footnote_run_in_group(lines: list[dict]) -> bool:
    """Keep a bare numeric footnote marker separate from its definition."""
    if len(lines) < 2:
        return False
    first = re.sub(r"\s+", " ", lines[0].get("plain", "")).strip()
    fontsize = max(float(lines[0].get("fontsize", 11.0)), 1.0)
    return bool(
        re.fullmatch(r"\d{1,3}", first)
        and float(lines[0]["x1"]) - float(lines[0]["x0"])
        <= max(fontsize * 2.0, 18.0)
    )


def _pdf_same_baseline_fragment_snapshot(line: dict) -> dict:
    return {
        "plain": line.get("plain", ""),
        "rich": line.get("rich", line.get("plain", "")),
        "bbox": [
            float(line.get("x0", 0.0)),
            float(line.get("y0", 0.0)),
            float(line.get("x1", 0.0)),
            float(line.get("y1", 0.0)),
        ],
    }


def _merge_pdf_same_baseline_line_records(first: dict, second: dict) -> dict:
    """Coalesce two proven fragments while retaining their raw geometry."""
    merged = dict(first)
    first_fragments = [
        dict(fragment)
        for fragment in (
            first.get("same_baseline_fragments")
            or [_pdf_same_baseline_fragment_snapshot(first)]
        )
    ]
    second_fragments = [
        dict(fragment)
        for fragment in (
            second.get("same_baseline_fragments")
            or [_pdf_same_baseline_fragment_snapshot(second)]
        )
    ]
    first_plain = re.sub(r"\s+", " ", first.get("plain", "")).strip()
    second_plain = re.sub(r"\s+", " ", second.get("plain", "")).strip()
    first_rich = (first.get("rich") or first_plain).strip()
    second_rich = (second.get("rich") or second_plain).strip()
    merged.update({
        "plain": _join_pdf_line_fragments([first_plain, second_plain]),
        "rich": _join_pdf_line_fragments([first_rich, second_rich]),
        "x0": min(float(first["x0"]), float(second["x0"])),
        "x1": max(float(first["x1"]), float(second["x1"])),
        "y0": min(float(first["y0"]), float(second["y0"])),
        "y1": max(float(first["y1"]), float(second["y1"])),
        "line_height": max(
            float(first.get("line_height", 0.0)),
            float(second.get("line_height", 0.0)),
            float(first["y1"]) - float(first["y0"]),
            float(second["y1"]) - float(second["y0"]),
        ),
        "superscript_runs": [
            dict(run)
            for line in (first, second)
            for run in (line.get("superscript_runs") or [])
        ],
        "inline_math_fragments": [
            dict(fragment)
            for line in (first, second)
            for fragment in (line.get("inline_math_fragments") or [])
        ],
        "math_reasons": sorted(
            set(first.get("math_reasons") or [])
            | set(second.get("math_reasons") or [])
        ),
        "math_mixed": bool(first.get("math_mixed") or second.get("math_mixed")),
        "math_symbol_count": int(first.get("math_symbol_count") or 0)
        + int(second.get("math_symbol_count") or 0),
        "same_baseline_fragments": first_fragments + second_fragments,
        "same_baseline_coalesced": True,
        "same_baseline_paragraph_start": bool(
            first.get("same_baseline_paragraph_start")
        ),
    })
    return merged


def _looks_like_pdf_reference_run_in_group(lines: list[dict]) -> bool:
    if len(lines) < 2:
        return False
    first = re.sub(r"\s+", " ", lines[0].get("plain", "")).strip()
    rest = " ".join(
        re.sub(r"\s+", " ", line.get("plain", "")).strip()
        for line in lines[1:]
    )
    return bool(
        re.match(r"(?i)^and\s+", rest)
        and re.search(r"\b(?:18|19|20)\d{2}\b", rest)
        and ("," in first or re.search(r"\bet\s+al\.?$", first, re.IGNORECASE))
    )


def _normalize_pdf_same_baseline_prose_fragments(
    line_infos: list[dict],
    block_rect: fitz.Rect,
) -> list[dict]:
    """Reconstruct full-width prose rows before paragraph inference.

    PyMuPDF can expose adjacent pieces on one visual baseline as separate line
    records.  This pass merges a continuing sentence or marks a new right-hand
    sentence so it joins its following left-margin line.  Repeated columns,
    footnotes, references, tables, rotated text, and protected math fail closed.
    """
    if len(line_infos) < 2:
        return [dict(line) for line in line_infos]

    repeated_column_starts = _pdf_repeated_column_group_starts(line_infos)
    normalized: list[dict] = []
    index = 0
    block_width = max(float(block_rect.width), 1.0)
    while index < len(line_infos):
        group, cursor = _pdf_same_baseline_candidate_group(line_infos, index)
        if len(group) < 2:
            normalized.append(dict(line_infos[index]))
            index += 1
            continue

        combined_plain = _join_pdf_line_fragments(
            [line.get("plain", "") for line in group]
        )
        alpha_words = re.findall(r"[A-Za-z]{2,}", combined_plain)
        right_words = re.findall(r"[A-Za-z]{2,}", group[-1].get("plain", ""))
        coverage = (float(group[-1]["x1"]) - float(group[0]["x0"])) / block_width
        left_gap = float(group[0]["x0"]) - float(block_rect.x0)
        right_gap = float(block_rect.x1) - float(group[-1]["x1"])
        fontsize = max(float(line.get("fontsize", 11.0)) for line in group)
        structural_prose_row = (
            index not in repeated_column_starts
            and coverage >= 0.88
            and left_gap <= max(fontsize * 1.5, block_width * 0.05)
            and right_gap <= max(fontsize * 1.5, block_width * 0.05)
            and len(alpha_words) >= 6
            and bool(right_words)
            and not _pdf_has_terminal_sentence_boundary(group[-1].get("plain", ""))
            and not _looks_like_pdf_reference_run_in_group(group)
            and not _looks_like_pdf_bare_footnote_run_in_group(group)
            and not any(
                _looks_like_pdf_paragraph_lead(line.get("plain", ""))
                or _looks_like_pdf_list_marker(line.get("plain", ""))
                or _pdf_source_line_footnote_definition_marker(line)
                for line in group
            )
        )
        if not structural_prose_row:
            normalized.extend(group)
            index = cursor
            continue

        group_is_final = cursor == len(line_infos)
        current = group[0]
        for following in group[1:]:
            sentence_boundary = _pdf_has_terminal_sentence_boundary(
                current.get("plain", "")
            )
            bold_to_regular_boundary = bool(current.get("bold")) and not bool(
                following.get("bold")
            )
            can_join = (
                _pdf_same_baseline_style_compatible(current, following)
                and not current.get("math_protected")
                and not following.get("math_protected")
            )
            if (
                (sentence_boundary and not group_is_final)
                or bold_to_regular_boundary
            ):
                normalized.append(current)
                current = dict(following)
                current["same_baseline_paragraph_start"] = True
                current["same_baseline_left_context"] = (
                    _pdf_same_baseline_fragment_snapshot(group[0])
                )
                continue
            if can_join:
                current = _merge_pdf_same_baseline_line_records(current, following)
                continue
            normalized.append(current)
            current = dict(following)
        normalized.append(current)
        index = cursor

    return normalized


__all__ = ["_normalize_pdf_same_baseline_prose_fragments"]
