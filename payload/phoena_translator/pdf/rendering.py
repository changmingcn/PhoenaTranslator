"""Rendering for deterministic PDF processing."""

from __future__ import annotations

import logging
import html
import hashlib
import math
import os
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
import fitz
from phoena_translator.pdf.geometry import (
    _get_pdf_elem_rect,
    _get_pdf_source_ink_rects,
    _pdf_rect_intersects_protected,
    _subtract_pdf_protected_rects,
)
from phoena_translator.pdf.math_detection import (
    _normalize_pdf_translation,
    _plain_text,
)
from phoena_translator.pdf.types import (
    PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE,
    PDF_TEXT_NEIGHBOR_GUARD_PADDING,
)

log = logging.getLogger("translator")

def _collect_pdf_overlap_redraw_indices(
    elements: list[dict],
    removal_indices: set[int],
) -> set[int]:
    """Find untouched text that PDF redaction would erase as collateral.

    PyMuPDF redaction removes a whole glyph when the redaction rectangle clips
    any part of its glyph box. Chart headings are often extracted as adjacent
    stacked elements whose boxes overlap by a few points, so deleting a
    translated heading can otherwise erase an unchanged second line (including
    its native superscript). Redraw the complete overlap-connected text group.
    """
    active = {
        int(index) for index in (removal_indices or set())
        if 0 <= int(index) < len(elements or [])
    }
    active_rects = []
    for index in sorted(active):
        elem = elements[index]
        try:
            rects = (
                _get_pdf_source_ink_rects(elem)
                if elem.get("type") in {"text", "text_merged_away"}
                else [_get_pdf_elem_rect(elem)]
            )
        except Exception:
            continue
        active_rects.extend(rect for rect in rects if not rect.is_empty)

    redraw = set()
    while active_rects:
        newly_added = []
        for index, elem in enumerate(elements or []):
            if index in active or elem.get("type") != "text":
                continue
            try:
                rects = [
                    rect
                    for rect in _get_pdf_source_ink_rects(elem)
                    if not rect.is_empty
                ]
            except Exception:
                continue
            if not rects:
                continue
            if not any(
                not (rect & removal_rect).is_empty
                and (rect & removal_rect).get_area() > 0
                for rect in rects
                for removal_rect in active_rects
            ):
                continue
            active.add(index)
            redraw.add(index)
            newly_added.extend(rects)

        if not newly_added:
            break
        active_rects = newly_added

    return redraw


def _expand_pdf_htmlbox_rect(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    text: str,
    fontsize: float,
    paragraphs: list[dict],
    aggressive: bool = False,
) -> fitz.Rect:
    expanded = fitz.Rect(rect)
    plain = re.sub(r"\s+", " ", _plain_text(text)).strip()
    compact = re.sub(r"\s+", "", plain)
    if not compact:
        return expanded

    non_empty_paragraphs = [p for p in (paragraphs or []) if (p.get("plain") or "").strip()]
    aligns = {p.get("text_align", "left") for p in non_empty_paragraphs} or {"left"}
    single_paragraph = len(non_empty_paragraphs) <= 1
    narrow_box = expanded.width <= max(fontsize * 2.2, 20.0)
    shortish_text = len(compact) <= (28 if aggressive else 22)

    if not aggressive and not (single_paragraph and narrow_box and shortish_text):
        return expanded

    if aligns == {"right"}:
        target_width = max(expanded.width, fontsize * (4.0 if not aggressive else 4.8), 30.0 if not aggressive else 40.0)
    elif aligns == {"center"}:
        target_width = max(expanded.width, fontsize * (4.2 if not aggressive else 5.0), 30.0 if not aggressive else 40.0)
    else:
        target_width = max(expanded.width, fontsize * (3.8 if not aggressive else 4.6), 30.0 if not aggressive else 40.0)

    target_width = min(target_width, page_rect.width * (0.18 if not aggressive else 0.24))
    if target_width <= expanded.width + 0.1:
        return expanded

    extra = target_width - expanded.width
    if aligns == {"right"}:
        expanded.x0 = max(page_rect.x0 + 4.0, expanded.x0 - extra)
    elif aligns == {"left"}:
        expanded.x1 = min(page_rect.x1 - 4.0, expanded.x1 + extra)
    else:
        expanded.x0 = max(page_rect.x0 + 4.0, expanded.x0 - extra / 2.0)
        expanded.x1 = min(page_rect.x1 - 4.0, expanded.x1 + extra / 2.0)

    if expanded.x1 <= expanded.x0:
        return fitz.Rect(rect)
    return expanded


def _sanitize_pdf_text_rect(rect: fitz.Rect, page_rect: fitz.Rect, min_width: float = 2.0, min_height: float = 2.0) -> fitz.Rect | None:
    coords = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
    if not all(math.isfinite(value) for value in coords):
        return None

    x0, x1 = sorted((coords[0], coords[2]))
    y0, y1 = sorted((coords[1], coords[3]))
    page_x0 = float(page_rect.x0) + 2.0
    page_y0 = float(page_rect.y0) + 2.0
    page_x1 = float(page_rect.x1) - 2.0
    page_y1 = float(page_rect.y1) - 2.0
    if page_x1 <= page_x0 or page_y1 <= page_y0:
        return None

    x0 = min(max(x0, page_x0), page_x1)
    y0 = min(max(y0, page_y0), page_y1)
    x1 = min(max(x1, page_x0), page_x1)
    y1 = min(max(y1, page_y0), page_y1)

    if x1 - x0 < min_width:
        x1 = min(page_x1, x0 + min_width)
        x0 = max(page_x0, x1 - min_width)
    if y1 - y0 < min_height:
        y1 = min(page_y1, y0 + min_height)
        y0 = max(page_y0, y1 - min_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def _expand_pdf_textbox_fit_rect(rect: fitz.Rect, page_rect: fitz.Rect, aggressive: bool = False) -> fitz.Rect:
    expanded = fitz.Rect(rect)
    right_pad = page_rect.x1 - expanded.x1 - 2.0
    bottom_pad = page_rect.y1 - expanded.y1 - 2.0

    width_extra = max(expanded.width * (0.35 if aggressive else 0.18), 24.0 if aggressive else 10.0)
    height_extra = max(expanded.height * (1.0 if aggressive else 0.45), 48.0 if aggressive else 18.0)
    expanded.x1 += min(max(right_pad, 0.0), width_extra)
    expanded.y1 += min(max(bottom_pad, 0.0), height_extra)
    return _sanitize_pdf_text_rect(expanded, page_rect) or rect


def _insert_pdf_htmlbox_at_readable_scale(
    page,
    rect: fitz.Rect,
    box_html: str,
    *,
    css: str,
    archive,
) -> tuple[float, float]:
    """Attempt the final HTML fit without ever committing unreadable text."""
    return page.insert_htmlbox(
        rect,
        box_html,
        css=css,
        archive=archive,
        scale_low=PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE,
    )


def _insert_pdf_textbox_with_fit(
    page,
    rect: fitz.Rect,
    text: str,
    *,
    align: int,
    color: tuple[float, float, float],
    fontfile: str,
    fontsize: float,
    lineheight_points: float,
    protected_rects: list | None = None,
    allow_compact_fontsize: bool = False,
    fit_rects: list | None = None,
) -> tuple[bool, str, float | None]:
    """Insert text with PyMuPDF's textbox API, checking the no-draw return path.

    PyMuPDF's lineheight argument is a factor, while our layout metrics are in
    points for HTML/CSS. Passing point values makes multi-line text silently not
    commit, because insert_textbox returns a negative deficit instead of raising.
    """
    fallback_text = (text or "").strip()
    if not fallback_text:
        return False, "empty text", None

    page_rect = fitz.Rect(page.rect)
    requested_rect = _sanitize_pdf_text_rect(fitz.Rect(rect), page_rect)
    if requested_rect is None:
        return False, "invalid text rect", None

    base_fontsize = max(float(fontsize), 1.0)
    lineheight_factor = max(float(lineheight_points) / base_fontsize, 1.0)
    lineheight_factor = min(lineheight_factor, 1.65)

    if fit_rects is None:
        rects = [
            requested_rect,
            _expand_pdf_textbox_fit_rect(requested_rect, page_rect),
            _expand_pdf_textbox_fit_rect(
                requested_rect,
                page_rect,
                aggressive=True,
            ),
        ]
    else:
        # The page assembler has already trimmed these rectangles against
        # neighboring text and protected formula regions.  Re-expanding them
        # here would silently discard that safety boundary.
        rects = [
            sanitized
            for candidate in fit_rects
            if (
                sanitized := _sanitize_pdf_text_rect(
                    fitz.Rect(candidate),
                    page_rect,
                )
            ) is not None
        ]
    seen_rects = set()
    unique_rects = []
    for candidate in rects:
        key = tuple(round(value, 2) for value in (candidate.x0, candidate.y0, candidate.x1, candidate.y1))
        if (
            key not in seen_rects
            and not _pdf_rect_intersects_protected(candidate, protected_rects or [])
        ):
            seen_rects.add(key)
            unique_rects.append(candidate)

    if not unique_rects:
        return False, "no formula-safe text rect", None
    base_rect = unique_rects[0]

    # ``fontfile=`` does not by itself select that font in PyMuPDF's textbox
    # APIs: when ``fontname`` is omitted they continue to use the built-in
    # Helvetica face, whose missing CJK glyphs are written as literal ``?``.
    # Register and explicitly select a stable custom font name before either
    # the normal textbox path or the last-resort single-line path can draw.
    font_digest = hashlib.sha256(
        os.path.abspath(fontfile).encode("utf-8")
    ).hexdigest()[:8]
    fontname = f"PhoenaFallback{font_digest}"
    try:
        page.insert_font(fontname=fontname, fontfile=fontfile)
    except Exception as exc:
        return False, f"textbox font registration failed: {exc}", None

    last_msg = ""
    scales = tuple(
        scale
        for scale in (1.0, 0.92, 0.84, 0.76, 0.68, 0.60)
        if scale + 1e-9 >= PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE
    )
    minimum_fontsize = (
        max(base_fontsize * PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE, 1.0)
        if allow_compact_fontsize
        else 5.5
    )
    for candidate_rect in unique_rects:
        for scale in scales:
            trial_fontsize = max(base_fontsize * scale, minimum_fontsize)
            trial_lineheight = max(min(lineheight_factor * min(1.0, scale + 0.08), 1.45), 1.05)
            try:
                rc = page.insert_textbox(
                    candidate_rect,
                    fallback_text,
                    align=align,
                    color=color,
                    fontname=fontname,
                    fontfile=fontfile,
                    fontsize=trial_fontsize,
                    lineheight=trial_lineheight,
                )
            except Exception as exc:
                last_msg = str(exc)
                continue
            if rc >= 0:
                if candidate_rect != base_rect or scale != 1.0:
                    return (
                        True,
                        f"fit after resize fontsize={trial_fontsize:.1f}, spare={rc:.1f}",
                        trial_fontsize,
                    )
                return True, f"spare={rc:.1f}", trial_fontsize
            last_msg = f"textbox deficit {-rc:.1f}"

    try:
        fallback_fontsize = max(min(base_fontsize, 8.0), 5.5)
        if (
            fallback_fontsize / max(base_fontsize, 1.0)
            < PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE
        ):
            return (
                False,
                "single-line fallback would violate minimum render scale",
                None,
            )
        line_text = re.sub(r"\s+", " ", fallback_text)[:1200]
        # insert_text draws an unclipped line: measure it first so it can
        # neither run off the page nor overwrite a protected formula region.
        try:
            line_width = fitz.Font(fontfile=fontfile).text_length(
                line_text, fontsize=fallback_fontsize
            )
        except Exception:
            line_width = page_rect.x1 - base_rect.x0
        if base_rect.x0 + line_width > page_rect.x1 - 2.0:
            return (
                False,
                "single-line fallback would overflow the page width",
                None,
            )
        baseline_y = min(base_rect.y1, base_rect.y0 + max(base_fontsize, 5.5))
        line_rect = fitz.Rect(
            base_rect.x0,
            baseline_y - fallback_fontsize,
            base_rect.x0 + line_width,
            baseline_y + fallback_fontsize * 0.35,
        )
        if fit_rects is not None and not any(
            line_rect.x0 >= candidate.x0 - 0.05
            and line_rect.y0 >= candidate.y0 - 0.05
            and line_rect.x1 <= candidate.x1 + 0.05
            and line_rect.y1 <= candidate.y1 + 0.05
            for candidate in unique_rects
        ):
            return (
                False,
                "single-line fallback would leave vetted text rects",
                None,
            )
        if _pdf_rect_intersects_protected(line_rect, protected_rects or []):
            return (
                False,
                "single-line fallback would cross a protected formula region",
                None,
            )
        point = fitz.Point(base_rect.x0, baseline_y)
        page.insert_text(
            point,
            line_text,
            color=color,
            fontname=fontname,
            fontfile=fontfile,
            fontsize=fallback_fontsize,
        )
        return True, f"used single-line fallback after {last_msg}", fallback_fontsize
    except Exception as exc:
        return False, f"{last_msg}; single-line fallback failed: {exc}", None


def _restore_pdf_source_text_element(
    page,
    elem: dict,
    rect: fitz.Rect,
    *,
    color: tuple[float, float, float],
    fontfile: str,
    fontsize: float,
    lineheight_points: float,
    protected_rects: list | None = None,
) -> tuple[bool, str]:
    """Write the element's exact source text back into its own region.

    Last-resort recovery after a translated text cannot be placed and the
    original ink was already redacted: the source prose occupied this
    rectangle in the original file, so re-inserting it (plain, without style
    runs) keeps the page complete and readable instead of failing the whole
    page back to an exact source copy.
    """
    source_text = _plain_text(
        _normalize_pdf_translation(elem.get("content") or "")
    ).strip()
    if not source_text:
        return False, "empty source text"

    ok, fit_msg, _ = _insert_pdf_textbox_with_fit(
        page,
        rect,
        source_text,
        align=0,
        color=color,
        fontfile=fontfile,
        fontsize=fontsize,
        lineheight_points=lineheight_points,
        protected_rects=protected_rects,
    )
    if ok:
        return True, f"source text restored via textbox ({fit_msg})"

    # The source text fit this rectangle in the original file, so a direct
    # unfiltered walk over modest scales is safe: it cannot cover more page
    # area than the original ink did.
    base_rect = _sanitize_pdf_text_rect(
        fitz.Rect(rect), fitz.Rect(page.rect)
    )
    if base_rect is None:
        return False, f"{fit_msg}; invalid source rect"
    font_digest = hashlib.sha256(
        os.path.abspath(fontfile).encode("utf-8")
    ).hexdigest()[:8]
    fontname = f"PhoenaFallback{font_digest}"
    try:
        page.insert_font(fontname=fontname, fontfile=fontfile)
    except Exception as exc:
        return False, f"{fit_msg}; source restore font registration failed: {exc}"
    last_deficit = ""
    for scale in (1.0, 0.92, 0.84, 0.76, 0.68):
        trial_fontsize = max(fontsize * scale, 5.0)
        try:
            rc = page.insert_textbox(
                base_rect,
                source_text,
                align=0,
                color=color,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=trial_fontsize,
                lineheight=1.12,
            )
        except Exception as exc:
            last_deficit = str(exc)
            continue
        if rc >= 0:
            return True, (
                f"source text restored directly at fontsize {trial_fontsize:.1f}"
            )
        last_deficit = f"textbox deficit {-rc:.1f}"
    return False, f"{fit_msg}; source restore no-fit ({last_deficit})"


def _build_pdf_rotated_text_rects(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    fontsize: float,
    rotation: int,
    protected_rects: list | None = None,
) -> list[tuple[str, fitz.Rect]]:
    """Return formula-safe fit rectangles for a rotated chart/table label."""
    base = _sanitize_pdf_text_rect(
        fitz.Rect(rect),
        fitz.Rect(page_rect),
        min_width=1.0,
        min_height=1.0,
    )
    if base is None:
        return []

    candidates = [("base", base)]
    cross_axis = fitz.Rect(base)
    if rotation in {90, 270}:
        target_width = max(cross_axis.width, max(float(fontsize), 1.0) * 1.55, 9.0)
        extra = max(0.0, target_width - cross_axis.width)
        cross_axis.x0 -= extra / 2.0
        cross_axis.x1 += extra / 2.0
    else:
        target_height = max(cross_axis.height, max(float(fontsize), 1.0) * 1.55, 9.0)
        extra = max(0.0, target_height - cross_axis.height)
        cross_axis.y0 -= extra / 2.0
        cross_axis.y1 += extra / 2.0
    cross_axis = _sanitize_pdf_text_rect(
        cross_axis,
        fitz.Rect(page_rect),
        min_width=1.0,
        min_height=1.0,
    )
    if cross_axis is not None:
        candidates.append(("cross-axis", cross_axis))

        length_axis = fitz.Rect(cross_axis)
        length_extra = max(float(fontsize) * 0.8, 4.0)
        if rotation in {90, 270}:
            length_axis.y0 -= length_extra
            length_axis.y1 += length_extra
        else:
            length_axis.x0 -= length_extra
            length_axis.x1 += length_extra
        length_axis = _sanitize_pdf_text_rect(
            length_axis,
            fitz.Rect(page_rect),
            min_width=1.0,
            min_height=1.0,
        )
        if length_axis is not None:
            candidates.append(("cross+length", length_axis))

    unique = []
    seen = set()
    for name, candidate in candidates:
        key = tuple(round(value, 2) for value in candidate)
        if key in seen or _pdf_rect_intersects_protected(
            candidate,
            protected_rects or [],
        ):
            continue
        seen.add(key)
        unique.append((name, candidate))
    return unique


def _insert_pdf_rotated_textbox_with_fit(
    page,
    rect: fitz.Rect,
    text: str,
    *,
    rotation: int,
    color: tuple[float, float, float],
    fontfile: str,
    fontsize: float,
    protected_rects: list | None = None,
) -> tuple[bool, str]:
    """Draw one translated non-horizontal label in its native direction."""
    fallback_text = re.sub(
        r"\s+", " ", _plain_text(_normalize_pdf_translation(text or ""))
    ).strip()
    if not fallback_text:
        return False, "empty rotated text"
    rotation = int(rotation or 0) % 360
    if rotation not in {90, 180, 270}:
        return False, f"unsupported rotation={rotation}"

    candidates = _build_pdf_rotated_text_rects(
        fitz.Rect(rect),
        fitz.Rect(page.rect),
        fontsize,
        rotation,
        protected_rects,
    )
    if not candidates:
        return False, "no formula-safe rotated text rect"

    font_digest = hashlib.sha256(
        os.path.abspath(fontfile).encode("utf-8")
    ).hexdigest()[:8]
    fontname = f"PhoenaVert{font_digest}"
    try:
        page.insert_font(fontname=fontname, fontfile=fontfile)
    except Exception as exc:
        return False, f"rotated font registration failed: {exc}"

    last_msg = ""
    for candidate_name, candidate_rect in candidates:
        for scale in (1.0, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64):
            trial_fontsize = max(float(fontsize) * scale, 4.8)
            try:
                spare = page.insert_textbox(
                    candidate_rect,
                    fallback_text,
                    align=1,
                    color=color,
                    fontname=fontname,
                    fontsize=trial_fontsize,
                    lineheight=1.05,
                    rotate=rotation,
                )
            except Exception as exc:
                last_msg = str(exc)
                continue
            if spare >= 0:
                return True, (
                    f"{candidate_name} rotation={rotation} "
                    f"fontsize={trial_fontsize:.1f} spare={spare:.1f}"
                )
            last_msg = f"rotated textbox deficit {-spare:.1f}"
    return False, last_msg or "rotated textbox did not fit"


def _build_htmlbox_rect_ladder(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    fontsize: float,
    *,
    prefer_widen: bool = False,
) -> list[tuple[str, fitz.Rect]]:
    """Candidate rects for no-shrink htmlbox insertion.

    Keep the original box first, then progressively extend downward with a
    transparent background (content may overlap vacated space below), and
    finally widen toward the right margin. Text is only ever drawn at source
    font size while walking this ladder; scaling down is the caller's separate,
    logged last resort.
    """
    base = fitz.Rect(rect)
    line = max(fontsize * 1.5, 12.0)
    bottom_limit = page_rect.y1 - 8.0
    right_limit = page_rect.x1 - 24.0

    steps = [("base", base)]
    if prefer_widen:
        early_widen = fitz.Rect(base)
        early_widen.x1 = min(
            max(
                early_widen.x1,
                base.x1 + base.width * 0.25,
                base.x1 + 3.0 * line,
            ),
            right_limit,
        )
        # The source glyph box is often slightly shorter than one HTML line.
        # Add one line of vertical breathing room while trying horizontal space
        # first, before any candidate that permits a two-line wrap.
        early_widen.y1 = min(base.y1 + line, bottom_limit)
        if early_widen.x1 > base.x1 + 1.0:
            steps.append(("widen+1line", early_widen))

    for name, extra in (
        ("down+2lines", 2.0 * line),
        ("down+60%", max(base.height * 0.6, 4.0 * line)),
        ("down+150%", max(base.height * 1.5, 8.0 * line)),
    ):
        candidate = fitz.Rect(base)
        candidate.y1 = min(base.y1 + extra, bottom_limit)
        if not any(
            abs(candidate.x0 - existing.x0) <= 0.1
            and abs(candidate.x1 - existing.x1) <= 0.1
            and abs(candidate.y0 - existing.y0) <= 0.1
            and abs(candidate.y1 - existing.y1) <= 0.1
            for _, existing in steps
        ):
            steps.append((name, candidate))

    widened = fitz.Rect(steps[-1][1])
    widened.x1 = min(max(widened.x1, base.x1 + base.width * 0.25, base.x1 + 3.0 * line), right_limit)
    if widened.x1 > steps[-1][1].x1 + 1.0:
        steps.append(("widen", widened))
    return steps


def _constrain_pdf_htmlbox_rect_ladder_to_neighbors(
    ladder: list[tuple[str, fitz.Rect]],
    anchor_rect: fitz.Rect,
    elements: list[dict],
    element_index: int,
    fontsize: float,
    *,
    ignored_indices: set[int] | None = None,
) -> list[tuple[str, fitz.Rect]]:
    """Keep an HTML box inside genuinely vacant space around its source block.

    PDF glyph boxes from consecutive paragraphs often overlap by a point or
    two, while the HTML no-shrink ladder can add several complete lines below
    the source rectangle.  Extending every dense footnote independently then
    makes one translated definition overwrite the next even though each
    individual ``insert_htmlbox`` call reports success.  Treat the next
    horizontally overlapping text block as a hard boundary and trim every
    candidate -- including ``base`` -- to that boundary.  The caller can then
    scale the complete semantic paragraph *inside its own slot*.

    Rightward widening is likewise stopped before a same-row text neighbor.
    Table cells do not use this helper: their existing base-only policy keeps
    them inside the ruled cell geometry.
    """
    anchor = fitz.Rect(anchor_rect)
    ignored = set(ignored_indices or ())
    guard = PDF_TEXT_NEIGHBOR_GUARD_PADDING
    row_tolerance = max(float(fontsize) * 0.35, 2.0)
    minimum_height = max(float(fontsize) * 0.60, 4.0)
    neighbors: list[fitz.Rect] = []

    for other_index, other in enumerate(elements or []):
        if other_index == element_index or other_index in ignored:
            continue
        if other.get("type") != "text":
            continue
        try:
            other_rect = _get_pdf_elem_rect(other)
        except (KeyError, TypeError, ValueError):
            continue
        if other_rect.is_empty or other_rect.is_infinite:
            continue
        neighbors.append(other_rect)

    constrained: list[tuple[str, fitz.Rect]] = []
    for name, raw_candidate in ladder:
        candidate = fitz.Rect(raw_candidate)
        changed = False

        # Downward growth may use only the gap before the next block in the
        # same reading column.  Use the candidate width so a widened heading
        # also respects text that becomes horizontally reachable.
        safe_bottom = candidate.y1
        for other in neighbors:
            if other.y0 <= anchor.y0 + row_tolerance:
                continue
            horizontal_overlap = max(
                0.0,
                min(candidate.x1, other.x1) - max(candidate.x0, other.x0),
            )
            if horizontal_overlap <= 0.0:
                continue
            overlap_ratio = horizontal_overlap / max(
                min(candidate.width, other.width),
                1.0,
            )
            if overlap_ratio < 0.25:
                continue
            safe_bottom = min(safe_bottom, other.y0 - guard)

        if safe_bottom < candidate.y1 - 0.05:
            candidate.y1 = safe_bottom
            changed = True

        # ``prefer_widen`` grows only toward the right.  Do not let that growth
        # enter a same-row label or a neighboring column.
        if candidate.x1 > anchor.x1 + 0.05:
            safe_right = candidate.x1
            for other in neighbors:
                if other.x0 < anchor.x1 - guard:
                    continue
                vertical_overlap = max(
                    0.0,
                    min(candidate.y1, other.y1) - max(candidate.y0, other.y0),
                )
                if vertical_overlap <= 0.0:
                    continue
                safe_right = min(safe_right, other.x0 - guard)
            if safe_right < candidate.x1 - 0.05:
                candidate.x1 = max(anchor.x1, safe_right)
                changed = True

        if candidate.height < minimum_height or candidate.width < 4.0:
            continue
        constrained_name = f"{name}+neighbor-trim" if changed else name
        if any(
            abs(candidate.x0 - existing.x0) <= 0.1
            and abs(candidate.x1 - existing.x1) <= 0.1
            and abs(candidate.y0 - existing.y0) <= 0.1
            and abs(candidate.y1 - existing.y1) <= 0.1
            for _, existing in constrained
        ):
            continue
        constrained.append((constrained_name, candidate))
    return constrained


def _pdf_plain_text_to_html(text: str) -> str:
    normalized = _plain_text(_normalize_pdf_translation(text))
    normalized = re.sub(r'\s*\n\s*', ' ', normalized)
    escaped = html.escape(normalized)
    return _wrap_pdf_latin_runs(escaped)


def _resolve_pdf_render_style(elem: dict, layout_styles: dict[str, dict]) -> dict:
    layout_class = elem.get("layout_class", "body")
    class_style = layout_styles.get(layout_class, layout_styles.get("body", {}))
    fontsize = max(float(elem.get("fontsize", class_style.get("fontsize", 11.0))), 6.5)
    color = int(elem.get("color", class_style.get("color", 0)))
    line_height = max(float(elem.get("line_height", fontsize * 1.18)), fontsize * 1.05)
    r_val = (color >> 16) & 0xFF
    g_val = (color >> 8) & 0xFF
    b_val = color & 0xFF
    return {
        "fontsize": fontsize,
        "line_height": line_height,
        "color": color,
        "color_hex": f"#{r_val:02x}{g_val:02x}{b_val:02x}",
        "padding_ratio": 0.14 if len(elem.get("paragraphs") or []) <= 1 else 0.20,
        "preserve_source_style": True,
    }


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


class _PDFInlineHTMLParser(HTMLParser):
    def __init__(self, superscript_scale: float):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
        self.superscript_scale = max(0.45, min(float(superscript_scale), 0.80))

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in {"b", "sup"}:
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
        if tag not in self.stack:
            return
        while self.stack:
            open_tag = self.stack.pop()
            self.parts.append(f"</{open_tag}>")
            if open_tag == tag:
                break

    def handle_data(self, data):
        escaped = html.escape(re.sub(r'\s*\n\s*', ' ', data))
        self.parts.append(_wrap_pdf_latin_runs(escaped))

    def finish(self) -> str:
        while self.stack:
            self.parts.append(f"</{self.stack.pop()}>")
        return "".join(self.parts)


def _pdf_text_to_html(text: str, superscript_scale: float = 0.60) -> str:
    """Escape translated text while preserving supported inline markup."""
    parser = _PDFInlineHTMLParser(superscript_scale)
    parser.feed(_normalize_pdf_translation(text))
    parser.close()
    return parser.finish()


def _paint_pdf_vector_ocr_element_background(
    out_page,
    elem: dict,
    protected_formula_rects: list[fitz.Rect] | None = None,
) -> int:
    """Erase visible outline glyphs while leaving cell borders untouched."""
    if not elem.get("vector_ocr"):
        return 0
    painted = 0
    paint_specs = (
        list(elem.get("vector_ocr_fill_rects") or [])
        + list(elem.get("vector_ocr_erase_rects") or [])
    )
    for spec in paint_specs:
        paint_rect = fitz.Rect(spec.get("bbox", (0, 0, 0, 0)))
        if paint_rect.is_empty:
            continue
        raw_fill = spec.get("fill") or [1.0, 1.0, 1.0]
        fill = tuple(
            max(0.0, min(1.0, float(value)))
            for value in raw_fill[:3]
        )
        for safe_rect in _subtract_pdf_protected_rects(
            paint_rect,
            protected_formula_rects or [],
        ):
            shape = out_page.new_shape()
            shape.draw_rect(safe_rect)
            shape.finish(fill=fill, color=fill)
            shape.commit()
            painted += 1
    return painted


def _compact_pdf_file_in_place(path: str, task_id: str = "") -> None:
    """Deduplicate repeated PDF objects after htmlbox incremental writes."""
    if not path or not os.path.exists(path):
        return

    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = None
    doc = None
    try:
        before_size = os.path.getsize(path)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{Path(path).stem}_compact_",
            suffix=".pdf",
            dir=directory,
        )
        os.close(fd)

        doc = fitz.open(path)
        doc.save(tmp_path, deflate=True, garbage=4, clean=True)
        doc.close()
        doc = None

        after_size = os.path.getsize(tmp_path)
        if after_size > 0 and after_size < before_size:
            os.replace(tmp_path, path)
            tmp_path = None
            log.info(
                f"[{task_id}] Compacted PDF from {before_size / 1024 / 1024:.1f} MB "
                f"to {after_size / 1024 / 1024:.1f} MB"
            )
        else:
            log.info(
                f"[{task_id}] PDF compaction kept original "
                f"({before_size / 1024 / 1024:.1f} MB -> {after_size / 1024 / 1024:.1f} MB)"
            )
    except Exception as exc:
        log.warning(f"[{task_id}] PDF compaction failed, keeping original: {exc}")
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
