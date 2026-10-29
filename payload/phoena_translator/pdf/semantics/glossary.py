"""Unruled Term/Meaning glossaries: detect the two columns, coalesce each
cell and mark lower-case term cells for translation."""

from __future__ import annotations

import re

import fitz

from phoena_translator.pdf.geometry import _get_pdf_elem_rect, _get_pdf_render_bbox
from phoena_translator.pdf.model import ElementType, LayoutClass
from phoena_translator.pdf.semantics.tables import (
    _apply_pdf_semantic_table_replacements,
    _merge_pdf_table_cell_group,
)
from phoena_translator.pdf.semantics.watermarks import _is_disclaimer_block
from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text
from phoena_translator.pdf.translate.targets import (
    _looks_like_pdf_proper_name_label,
    _looks_like_translatable_english,
)

_PDF_GLOSSARY_TERM_HEADERS = frozenset(
    {"term", "key term", "key terms", "glossary term", "glossary terms"}
)


def _merge_pdf_glossary_cell_group(
    group: list[dict],
    page_rect: fitz.Rect,
) -> dict:
    """Coalesce visual fragments belonging to one open-glossary cell."""
    return _merge_pdf_table_cell_group(
        group,
        page_rect,
        glossary_cell_hint=True,
    )


def _make_pdf_glossary_column_fragment(
    elem: dict,
    paragraphs: list[dict],
    rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> dict:
    """Clone one visual column out of a cross-column glossary block."""
    clone = dict(elem)
    normalized_paragraphs = []
    for position, paragraph in enumerate(paragraphs):
        normalized = dict(paragraph)
        normalized.update(
            {
                "margin_left": 0.0,
                "text_indent": 0.0,
                "gap_before": 0.0
                if position == 0
                else float(paragraph.get("gap_before", 0.0)),
                "text_align": "left",
            }
        )
        normalized_paragraphs.append(normalized)

    plain = " ".join(
        re.sub(r"\s+", " ", paragraph.get("plain", "")).strip()
        for paragraph in normalized_paragraphs
        if paragraph.get("plain", "").strip()
    )
    rich = " ".join(
        (paragraph.get("rich") or paragraph.get("plain", "")).strip()
        for paragraph in normalized_paragraphs
        if (paragraph.get("rich") or paragraph.get("plain", "")).strip()
    )
    superscript_runs = [
        dict(run)
        for run in (elem.get("superscript_runs") or [])
        if run.get("text", "").strip()
        and run.get("text", "").strip() in _plain_text(rich)
    ]
    fontsize = float(clone.get("fontsize", 9.0))
    clone.update(
        {
            "y": float(rect.y0),
            "x": float(rect.x0),
            "rect": fitz.Rect(rect),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "render_bbox": _get_pdf_render_bbox(
                rect,
                page_rect,
                normalized_paragraphs,
                fontsize,
            ),
            "content": plain,
            "rich_content": rich
            if (elem.get("rich_content") or superscript_runs)
            else None,
            "paragraphs": normalized_paragraphs,
            "superscript_runs": superscript_runs,
        }
    )
    return clone


def _prepare_pdf_glossary_column_geometry(
    elements: list[dict],
    page_rect: fitz.Rect,
    left_x: float,
    right_x: float,
    header_bottom: float,
    body_bottom: float,
) -> bool:
    """Repair column geometry only after explicit glossary headers match.

    A few tagged PDFs put two same-baseline glossary cells in one block, or
    include an empty full-width line in a right-cell block.  Keeping the
    generic extractor stable is important for page-cache identity, so these
    two corrections are deliberately scoped to a confirmed Term/Meaning page.
    """
    changed = False
    prepared = []
    column_gap = right_x - left_x
    for elem in elements:
        if elem.get("type") != ElementType.TEXT:
            prepared.append(elem)
            continue

        rect = _get_pdf_elem_rect(elem)
        if not (header_bottom < rect.y0 < body_bottom):
            prepared.append(elem)
            continue
        paragraphs = [
            paragraph
            for paragraph in (elem.get("paragraphs") or [])
            if (paragraph.get("plain") or "").strip()
        ]

        # A single PDF block may contain the left and right cells as distinct
        # paragraphs whose margins point to the real column origins.
        if (
            len(paragraphs) >= 2
            and abs(rect.x0 - left_x) <= 18.0
            and rect.x1 >= right_x + 24.0
        ):
            left_paragraphs = []
            right_paragraphs = []
            for paragraph in paragraphs:
                paragraph_x = rect.x0 + float(paragraph.get("margin_left", 0.0))
                if abs(paragraph_x - right_x) < abs(paragraph_x - left_x):
                    right_paragraphs.append(paragraph)
                else:
                    left_paragraphs.append(paragraph)
            if left_paragraphs and right_paragraphs:
                fontsize = max(float(elem.get("fontsize", 9.0)), 1.0)
                left_rect = fitz.Rect(
                    left_x,
                    rect.y0,
                    min(rect.x1, right_x - max(fontsize, 8.0)),
                    rect.y1,
                )
                right_rect = fitz.Rect(right_x, rect.y0, rect.x1, rect.y1)
                prepared.extend(
                    [
                        _make_pdf_glossary_column_fragment(
                            elem, left_paragraphs, left_rect, page_rect
                        ),
                        _make_pdf_glossary_column_fragment(
                            elem, right_paragraphs, right_rect, page_rect
                        ),
                    ]
                )
                changed = True
                continue

        # A whitespace-only line at the page margin can enlarge a right-cell
        # block to full width.  The surviving paragraph margin remains strong
        # evidence that its lexical text begins in the Meaning column.
        if (
            len(paragraphs) == 1
            and rect.x0 < left_x - 30.0
            and rect.x1 >= right_x + 24.0
            and float(paragraphs[0].get("margin_left", 0.0))
            >= max(column_gap * 0.75, 54.0)
        ):
            shifted_rect = fitz.Rect(right_x, rect.y0, rect.x1, rect.y1)
            prepared.append(
                _make_pdf_glossary_column_fragment(
                    elem, paragraphs, shifted_rect, page_rect
                )
            )
            changed = True
            continue

        prepared.append(elem)

    if changed:
        prepared.sort(
            key=lambda element: (
                float(element.get("y", _get_pdf_elem_rect(element).y0)),
                float(element.get("x", _get_pdf_elem_rect(element).x0)),
            )
        )
        elements[:] = prepared
    return changed


def _is_pdf_glossary_term_header(cell: str) -> bool:
    """``cell`` is casefolded, whitespace-collapsed plain text."""
    return cell in _PDF_GLOSSARY_TERM_HEADERS


def _is_pdf_glossary_meaning_header(cell: str) -> bool:
    return cell in {"meaning", "definition"} or cell.startswith("meaning in ")


def _normalize_pdf_glossary_cells(
    elements: list[dict],
    page_rect: fitz.Rect,
    _geometry_prepared: bool = False,
) -> bool:
    """Detect an unruled Term/Meaning table and coalesce each row cell.

    PDF generators frequently encode the first visual line under the ruled
    header as a table cell and the remaining lines as an unrelated text block.
    Translating those fragments independently can shift batch results and lets
    each fragment extend into the following row.  Explicit headers plus four
    aligned row pairs are required before mutating geometry.
    """
    text_indices = [
        index
        for index, element in enumerate(elements or [])
        if element.get("type") == ElementType.TEXT
    ]
    normalized = {
        index: _collapsed_plain_text(elements[index].get("content", ""))
        .casefold()
        for index in text_indices
    }
    term_headers = [
        index for index, text in normalized.items() if _is_pdf_glossary_term_header(text)
    ]
    meaning_headers = [
        index for index, text in normalized.items() if _is_pdf_glossary_meaning_header(text)
    ]
    if not term_headers or not meaning_headers:
        return False

    best_headers = None
    for left_index in term_headers:
        left_rect = _get_pdf_elem_rect(elements[left_index])
        for right_index in meaning_headers:
            right_rect = _get_pdf_elem_rect(elements[right_index])
            if right_rect.x0 <= left_rect.x0:
                continue
            y_delta = abs(right_rect.y0 - left_rect.y0)
            gap = right_rect.x0 - left_rect.x0
            if y_delta > max(float(elements[left_index].get("fontsize", 9.0)), 9.0):
                continue
            candidate = (y_delta, -gap, left_index, right_index)
            if best_headers is None or candidate[:2] < best_headers[:2]:
                best_headers = candidate
    if best_headers is None:
        return False

    left_header_index, right_header_index = best_headers[2], best_headers[3]
    left_header_rect = _get_pdf_elem_rect(elements[left_header_index])
    right_header_rect = _get_pdf_elem_rect(elements[right_header_index])
    left_x = float(left_header_rect.x0)
    right_x = float(right_header_rect.x0)
    if right_x - left_x < max(page_rect.width * 0.10, 48.0):
        return False

    header_bottom = max(left_header_rect.y1, right_header_rect.y1)
    body_bottom = page_rect.y0 + page_rect.height * 0.90
    if not _geometry_prepared and _prepare_pdf_glossary_column_geometry(
        elements,
        page_rect,
        left_x,
        right_x,
        header_bottom,
        body_bottom,
    ):
        return _normalize_pdf_glossary_cells(
            elements,
            page_rect,
            _geometry_prepared=True,
        )

    left_candidates = []
    right_candidates = []
    for index in text_indices:
        if index in {left_header_index, right_header_index}:
            continue
        rect = _get_pdf_elem_rect(elements[index])
        if rect.y0 <= header_bottom or rect.y0 >= body_bottom:
            continue
        if abs(rect.x0 - left_x) <= 18.0:
            left_candidates.append(index)
        if abs(rect.x0 - right_x) <= 30.0:
            right_candidates.append(index)

    row_starts = []
    for left_index in left_candidates:
        left = elements[left_index]
        left_rect = _get_pdf_elem_rect(left)
        tolerance = max(3.0, float(left.get("fontsize", 9.0)) * 0.55)
        matches = [
            right_index
            for right_index in right_candidates
            if abs(_get_pdf_elem_rect(elements[right_index]).y0 - left_rect.y0)
            <= tolerance
        ]
        if matches:
            row_starts.append(float(left_rect.y0))

    clustered_starts = []
    for y_value in sorted(row_starts):
        if not clustered_starts or y_value - clustered_starts[-1] > 3.0:
            clustered_starts.append(y_value)
    if len(clustered_starts) < 4:
        return False

    replacements = {}
    removed_indices = set()
    for position, row_y in enumerate(clustered_starts):
        next_y = (
            clustered_starts[position + 1]
            if position + 1 < len(clustered_starts)
            else body_bottom
        )
        lower = row_y - 3.0
        upper = next_y - 3.0
        for column_candidates in (left_candidates, right_candidates):
            group_indices = [
                index
                for index in column_candidates
                if lower <= _get_pdf_elem_rect(elements[index]).y0 < upper
            ]
            if not group_indices:
                continue
            group_indices.sort(
                key=lambda index: (
                    _get_pdf_elem_rect(elements[index]).y0,
                    _get_pdf_elem_rect(elements[index]).x0,
                )
            )
            primary = group_indices[0]
            replacements[primary] = _merge_pdf_glossary_cell_group(
                [elements[index] for index in group_indices],
                page_rect,
            )
            removed_indices.update(group_indices[1:])

    if not replacements:
        return False
    elements[left_header_index]["glossary_cell_hint"] = True
    elements[right_header_index]["glossary_cell_hint"] = True
    elements[:] = _apply_pdf_semantic_table_replacements(
        elements, replacements, removed_indices
    )
    return True


def _mark_pdf_glossary_term_elements(
    elements: list[dict],
    page_rect: fitz.Rect,
) -> set[int]:
    """Mark lower-case term cells paired with explanatory definitions.

    Glossaries are often drawn as an open two-column table, so PyMuPDF does
    not set ``table_hint``.  The generic tiny-table-label exemption then used
    to preserve entries such as ``pinging``, ``profiling`` and ``spread`` in
    English even though the adjacent definition was translated.  Require both
    an explicit term/meaning header and at least four consistently aligned
    term/definition row pairs.  The header gate keeps regression-variable
    tables and ordinary two-column data tables out of this semantic class;
    the alignment gate excludes chart atoms, page markers and isolated table
    labels.  Proper names and all-uppercase identifiers remain immutable.
    """
    text_indices = [
        index
        for index, element in enumerate(elements or [])
        if element.get("type") == ElementType.TEXT
    ]
    if len(text_indices) < 8:
        return set()
    header_bottom = _pdf_glossary_header_bottom(elements, text_indices, page_rect)
    if header_bottom is None:
        return set()
    row_pairs = [
        pair
        for left_index in text_indices
        if (pair := _pdf_glossary_row_pair(elements, text_indices, left_index, page_rect))
    ]
    valid_columns = _pdf_glossary_aligned_columns(row_pairs)
    if not valid_columns:
        return set()
    return _mark_pdf_glossary_column_cells(
        elements, text_indices, valid_columns, header_bottom, page_rect
    )


def _pdf_glossary_header_bottom(
    elements: list[dict], text_indices: list[int], page_rect: fitz.Rect
) -> float | None:
    """Bottom of the Term/Meaning header row, or ``None`` without both headers."""
    normalized_by_index = {
        index: _collapsed_plain_text(elements[index].get("content", "")).casefold()
        for index in text_indices
    }
    normalized_cells = set(normalized_by_index.values())
    has_term_header = any(map(_is_pdf_glossary_term_header, normalized_cells))
    has_meaning_header = any(map(_is_pdf_glossary_meaning_header, normalized_cells))
    if not (has_term_header and has_meaning_header):
        return None
    header_indices = [
        index
        for index, cell in normalized_by_index.items()
        if _is_pdf_glossary_term_header(cell) or _is_pdf_glossary_meaning_header(cell)
    ]
    return max(
        (_get_pdf_elem_rect(elements[index]).y1 for index in header_indices),
        default=page_rect.y0 + page_rect.height * 0.12,
    )


def _pdf_glossary_row_pair(
    elements: list[dict], text_indices: list[int], left_index: int, page_rect: fitz.Rect
) -> tuple[int, int, float, float] | None:
    """Pair a lower-case term cell with the nearest definition to its right."""
    page_width = max(page_rect.width, 1.0)
    page_height = max(page_rect.height, 1.0)
    left = elements[left_index]
    if left.get("layout_class") != LayoutClass.TABLE:
        return None
    left_plain = _collapsed_plain_text(left.get("content", ""))
    if (
        not left_plain
        or len(left_plain) > 80
        or not left_plain[:1].islower()
        or not _looks_like_translatable_english(left_plain)
        or _looks_like_pdf_proper_name_label(left_plain)
    ):
        return None
    left_rect = _get_pdf_elem_rect(left)
    left_center_y = (left_rect.y0 + left_rect.y1) / 2.0
    if (
        left_rect.width > page_width * 0.38
        or left_center_y < page_rect.y0 + page_height * 0.16
        or left_center_y > page_rect.y0 + page_height * 0.90
    ):
        return None

    best = None
    for right_index in text_indices:
        if right_index == left_index:
            continue
        right = elements[right_index]
        right_plain = _collapsed_plain_text(right.get("content", ""))
        if (
            len(right_plain) < 24
            or not _looks_like_translatable_english(right_plain)
            or _is_disclaimer_block(right_plain)
        ):
            continue
        right_rect = _get_pdf_elem_rect(right)
        horizontal_gap = right_rect.x0 - left_rect.x1
        if horizontal_gap < max(12.0, page_width * 0.03):
            continue
        vertical_overlap = min(left_rect.y1, right_rect.y1) - max(left_rect.y0, right_rect.y0)
        row_tolerance = max(5.0, float(left.get("fontsize", 0.0)) * 0.85)
        if vertical_overlap <= 0 and abs(right_rect.y0 - left_rect.y0) > row_tolerance:
            continue
        candidate = (horizontal_gap, abs(right_rect.y0 - left_rect.y0), right_index, right_rect)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return left_index, best[2], float(left_rect.x0), float(best[3].x0)


def _pdf_glossary_aligned_columns(row_pairs) -> list[tuple[float, float]]:
    """Column x-positions shared by at least four term/definition pairs."""
    valid_columns = []
    for _left_index, _right_index, left_x, right_x in row_pairs:
        aligned_count = sum(
            1
            for _, _, other_left_x, other_right_x in row_pairs
            if abs(other_left_x - left_x) <= 18.0 and abs(other_right_x - right_x) <= 30.0
        )
        if aligned_count >= 4:
            valid_columns.append((left_x, right_x))
    return valid_columns


def _mark_pdf_glossary_column_cells(
    elements: list[dict],
    text_indices: list[int],
    valid_columns: list[tuple[float, float]],
    header_bottom: float,
    page_rect: fitz.Rect,
) -> set[int]:
    page_height = max(page_rect.height, 1.0)
    marked = set()
    for index in text_indices:
        element = elements[index]
        plain = _collapsed_plain_text(element.get("content", ""))
        if not plain or not _looks_like_translatable_english(plain) or _is_disclaimer_block(plain):
            continue
        rect = _get_pdf_elem_rect(element)
        center_y = (rect.y0 + rect.y1) / 2.0
        if center_y <= header_bottom or center_y > page_rect.y0 + page_height * 0.90:
            continue
        aligned_left = any(abs(float(rect.x0) - left_x) <= 18.0 for left_x, _ in valid_columns)
        aligned_right = any(abs(float(rect.x0) - right_x) <= 30.0 for _, right_x in valid_columns)
        semantic_cross_reference = bool(re.match(r"(?i)^see\b", plain))
        proper_name = _looks_like_pdf_proper_name_label(plain)

        # Include split left-cell continuations and terms whose definitions
        # are too short to qualify as a row-pair anchor (for example
        # ``trades`` and ``liquidity`` in ASIC Report 452).
        if aligned_left and plain[:1].islower() and not proper_name:
            element["glossary_term_hint"] = True
            marked.add(index)
            continue

        # Short semantic definitions such as ``See 'NBBO'`` need the same
        # override as short terms.  Preserve pure institution/product names.
        if aligned_right and (not proper_name or semantic_cross_reference):
            element["glossary_definition_hint"] = True
            marked.add(index)
    return marked
