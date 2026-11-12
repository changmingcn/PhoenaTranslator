"""Distribution watermark codes: per-copy tracking tokens stamped in a page
corner by research portals.

Such a code is one random-looking token of letters and digits, printed at the
edge of the page (often partly outside the page box).  It is not content: it
must not be sent for translation, and it must not survive in the delivered
page.  A false positive deletes real content while a miss only leaves a stray
code, so the detector is deliberately narrow.  An element is a distribution
code only when *every* condition below holds:

``text_element``      a TEXT element with no ``skip_translate_reason`` yet.
``not_content_class`` not classed body, heading or table, and none of the
                      table, heading, footnote or reference hints.
``single_line``       one paragraph with one source line (or no paragraph
                      record and no line break).
``length``            8 to 40 characters.
``alphanumeric``      only ``A-Z``, ``a-z`` and ``0-9`` after HTML markup is
                      removed (so a single token: no whitespace).
``mixed_case``        both upper- and lower-case letters.  Tickers, ISIN,
                      CUSIP and SEDOL codes, report numbers and fiscal labels
                      are upper case, so this alone keeps them out.
``alternating``       at least three switches between a letter and a digit
                      (which implies at least two of each), so camel-case
                      words with a version number fail.
``no_word_run``       no word-shaped letter run of four or more letters (a
                      capitalised or lower-case run such as ``Update`` or
                      ``final``), so running footers such as ``FY2024Q3Update``
                      or ``Model7Bv2`` are never codes.
``few_on_page``       at most ``DISTRIBUTION_CODE_MAX_PER_PAGE`` (3) elements of
                      the page pass the eight conditions above.  A code is
                      stamped once per page; a page with more such tokens is
                      a page of identifiers (a key table, a list of codes),
                      where removing one would delete content.  The geometry
                      below is not evaluated for such a page.
``edge_placement``    the box sticks out of the page box by at least 0.5 pt,
                      or lies wholly inside the outer 6% band of the page.
``isolated``          no other element's box intersects it, so removing its
                      ink cannot touch neighbouring content.
``corroborated``      the box sticks out of the page box, or the same token
                      sits at the same place (within 3% of the page size) on
                      another page.  A token wholly on the page needs that
                      repetition before it is removed.
``invisible_or_outside`` the box sticks out of the page box, or every glyph in
                      it is invisible in the source (fill opacity 0 or text
                      render mode 3).  A visible token inside the page is
                      never removed, whatever its shape.  When the source
                      cannot be read the ink counts as visible.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import fitz

from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import ElementType, LayoutClass, SkipReason
from phoena_translator.pdf.text_shape import _collapsed_plain_text

DISTRIBUTION_CODE_MIN_LENGTH = 8
DISTRIBUTION_CODE_MAX_LENGTH = 40
DISTRIBUTION_CODE_MARGIN_SHARE = 0.06
DISTRIBUTION_CODE_OUTSIDE_TOLERANCE = 0.5
DISTRIBUTION_CODE_SAME_PLACE_SHARE = 0.03
DISTRIBUTION_CODE_MIN_SWITCHES = 3
# One code per page, and room for two code-shaped content tokens beside it
# (they fail the geometry); more than this is a page of identifiers.
DISTRIBUTION_CODE_MAX_PER_PAGE = 3

CONDITIONS = (
    "text_element",
    "not_content_class",
    "single_line",
    "length",
    "alphanumeric",
    "mixed_case",
    "alternating",
    "no_word_run",
    "few_on_page",
    "edge_placement",
    "isolated",
    "corroborated",
    "invisible_or_outside",
)

_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_WORD_RUN_RE = re.compile(r"[A-Z][a-z]{3,}|[a-z]{4,}")
_CONTENT_CLASSES = {LayoutClass.BODY, LayoutClass.HEADING, LayoutClass.TABLE}
_CONTENT_HINTS = (
    "table_hint",
    "semantic_table_cell",
    "semantic_native_table_cell",
    "single_line_heading",
    "footnote_hint",
    "reference_entry_hint",
)


def _token(elem: dict) -> str:
    return _collapsed_plain_text(elem.get("content") or "").strip()


def _letter_digit_switches(token: str) -> int:
    kinds = ["d" if char.isdigit() else "a" for char in token]
    return sum(1 for left, right in zip(kinds, kinds[1:]) if left != right)


def _single_line(elem: dict) -> bool:
    paragraphs = elem.get("paragraphs")
    if not paragraphs:
        return "\n" not in (elem.get("content") or "").strip()
    if len(paragraphs) != 1:
        return False
    lines = paragraphs[0].get("source_lines")
    if lines is None:
        lines = paragraphs[0].get("source_line_bboxes") or [None]
    return len(lines) == 1


def _outside_page(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    tol = DISTRIBUTION_CODE_OUTSIDE_TOLERANCE
    return (
        rect.x0 < page_rect.x0 - tol
        or rect.y0 < page_rect.y0 - tol
        or rect.x1 > page_rect.x1 + tol
        or rect.y1 > page_rect.y1 + tol
    )


def _in_margin_band(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    band_x = page_rect.width * DISTRIBUTION_CODE_MARGIN_SHARE
    band_y = page_rect.height * DISTRIBUTION_CODE_MARGIN_SHARE
    return (
        rect.y1 <= page_rect.y0 + band_y
        or rect.y0 >= page_rect.y1 - band_y
        or rect.x1 <= page_rect.x0 + band_x
        or rect.x0 >= page_rect.x1 - band_x
    )


def _isolated(index: int, rect: fitz.Rect, elements: list[dict]) -> bool:
    for other_index, other in enumerate(elements):
        if other_index == index or other.get("type") == ElementType.MERGED_AWAY:
            continue
        try:
            other_rect = _get_pdf_elem_rect(other)
        except Exception:
            continue
        if other_rect.is_empty:
            continue
        overlap = fitz.Rect(rect) & other_rect
        if not overlap.is_empty and overlap.width > 0 and overlap.height > 0:
            return False
    return True


def _ink_invisible(trace: list[dict], rect: fitz.Rect) -> bool:
    """Every glyph centred in ``rect`` is drawn with fill opacity 0 or in the
    invisible text render mode (3); ``False`` when no glyph is found."""
    found = False
    for span in trace:
        invisible = span.get("type") == 3 or span.get("opacity", 1.0) == 0
        for char in span.get("chars", ()):
            if chr(char[0]).isspace():
                continue
            box = fitz.Rect(char[3])
            centre = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            if not rect.contains(centre):
                continue
            if not invisible:
                return False
            found = True
    return found


class _SourceInk:
    """The source document, opened on first use, answering whether the ink in
    a box is invisible.  Any failure to read it counts as visible ink."""

    def __init__(self, source_path: str | None, password: str = ""):
        self._path = source_path
        self._password = password or ""
        self._doc = None
        self._traces: dict[int, list[dict]] = {}

    def invisible(self, page_num: int, rect: fitz.Rect) -> bool:
        if not self._path or rect.is_empty:
            return False
        try:
            page_num = int(page_num)
            if page_num not in self._traces:
                if self._doc is None:
                    self._doc = fitz.open(self._path)
                    if self._doc.needs_pass:
                        self._doc.authenticate(self._password)
                self._traces[page_num] = self._doc[page_num].get_texttrace()
        except Exception:
            return False
        return _ink_invisible(self._traces[page_num], rect)

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None


def _same_place(rect: fitz.Rect, other: fitz.Rect, page_rect: fitz.Rect) -> bool:
    tol_x = page_rect.width * DISTRIBUTION_CODE_SAME_PLACE_SHARE
    tol_y = page_rect.height * DISTRIBUTION_CODE_SAME_PLACE_SHARE
    return abs(rect.x0 - other.x0) <= tol_x and abs(rect.y0 - other.y0) <= tol_y


def _token_conditions(elem: dict) -> dict[str, bool]:
    """The cheap conditions, read from the element alone."""
    token = _token(elem)
    return {
        "text_element": elem.get("type") == ElementType.TEXT
        and not elem.get("skip_translate_reason"),
        "not_content_class": elem.get("layout_class") not in _CONTENT_CLASSES
        and not any(elem.get(hint) for hint in _CONTENT_HINTS),
        "single_line": _single_line(elem),
        "length": DISTRIBUTION_CODE_MIN_LENGTH <= len(token) <= DISTRIBUTION_CODE_MAX_LENGTH,
        "alphanumeric": bool(_ALNUM_RE.fullmatch(token)) and token.isascii(),
        "mixed_case": any(char.isupper() for char in token)
        and any(char.islower() for char in token),
        "alternating": _letter_digit_switches(token) >= DISTRIBUTION_CODE_MIN_SWITCHES,
        "no_word_run": not _WORD_RUN_RE.search(token),
    }


def _token_survivors(elements: list[dict]) -> list[int]:
    """The page's elements that pass every token condition."""
    return [
        index for index, elem in enumerate(elements) if all(_token_conditions(elem).values())
    ]


def distribution_code_conditions(
    elem: dict,
    index: int,
    elements: list[dict],
    page_rect: fitz.Rect,
    same_token_rects_elsewhere: Iterable[fitz.Rect] = (),
    ink_invisible: bool = False,
    token_survivors: list[int] | None = None,
) -> dict[str, bool]:
    """Evaluate every documented condition for one element (all are computed,
    so an audit can show which ones a near miss failed).  ``ink_invisible``:
    the source draws every glyph in the box invisibly (``_SourceInk``).
    ``token_survivors``: the page's ``_token_survivors``, if already known."""
    if token_survivors is None:
        token_survivors = _token_survivors(elements)
    try:
        rect = _get_pdf_elem_rect(elem)
    except Exception:
        rect = fitz.Rect()
    page_rect = fitz.Rect(page_rect)
    outside = not rect.is_empty and _outside_page(rect, page_rect)
    repeated = any(
        _same_place(rect, other, page_rect) for other in same_token_rects_elsewhere
    )
    return {
        **_token_conditions(elem),
        "few_on_page": len(token_survivors) <= DISTRIBUTION_CODE_MAX_PER_PAGE,
        "edge_placement": not rect.is_empty
        and (outside or _in_margin_band(rect, page_rect)),
        "isolated": not rect.is_empty and _isolated(index, rect, elements),
        "corroborated": outside or repeated,
        "invisible_or_outside": outside or bool(ink_invisible),
    }


def _token_rects_by_page(page_extractions: dict) -> dict[str, list[tuple[int, fitz.Rect]]]:
    found: dict[str, list[tuple[int, fitz.Rect]]] = {}
    for page_num, info in page_extractions.items():
        for elem in info.get("elements", []) or []:
            if elem.get("type") != ElementType.TEXT:
                continue
            token = _token(elem)
            if not token or len(token) < DISTRIBUTION_CODE_MIN_LENGTH or re.search(r"\s", token):
                continue
            try:
                found.setdefault(token, []).append((page_num, _get_pdf_elem_rect(elem)))
            except Exception:
                continue
    return found


def evaluate_distribution_codes(
    page_extractions: dict,
    page_rects: dict,
    source_path: str | None = None,
    password: str = "",
) -> list[dict[str, Any]]:
    """Per-element condition table for every TEXT element (for audits only:
    it evaluates the geometry of every element, which is quadratic per page).

    The source ink is read only for an element inside the page that fails at
    most one other condition, which is all a near-miss audit can show."""
    ink = _SourceInk(source_path, password)
    try:
        return _evaluate_distribution_codes(page_extractions, page_rects, ink)
    finally:
        ink.close()


def _evaluate_distribution_codes(page_extractions, page_rects, ink):
    by_token = _token_rects_by_page(page_extractions)
    rows = []
    for page_num in sorted(page_extractions, key=int):
        info = page_extractions[page_num]
        elements = info.get("elements", []) or []
        page_rect = page_rects.get(page_num)
        if page_rect is None:
            continue
        survivors = _token_survivors(elements)
        for index, elem in enumerate(elements):
            if elem.get("type") != ElementType.TEXT:
                continue
            token = _token(elem)
            elsewhere = [
                rect for other_page, rect in by_token.get(token, []) if other_page != page_num
            ]
            conditions = distribution_code_conditions(
                elem, index, elements, page_rect, elsewhere, token_survivors=survivors
            )
            failed_others = [
                name for name, ok in conditions.items()
                if not ok and name != "invisible_or_outside"
            ]
            if not conditions["invisible_or_outside"] and len(failed_others) <= 1:
                conditions["invisible_or_outside"] = ink.invisible(
                    page_num, _get_pdf_elem_rect(elem)
                )
            rows.append({
                "page_num": int(page_num),
                "element": index,
                "bbox": [round(float(v), 1) for v in _get_pdf_elem_rect(elem)],
                "text": token,
                "conditions": conditions,
            })
    return rows


def mark_distribution_codes(
    page_extractions: dict,
    page_rects: dict,
    source_path: str | None = None,
    password: str = "",
) -> list[dict[str, Any]]:
    """Mark every distribution code as a watermark and return audit records.

    A marked element keeps its index; ``SkipReason.WATERMARK`` makes the
    translation stage skip it, the cache bind empty it, the leak and
    untranslated checks ignore it, and the assembly redact its ink without
    drawing anything in its place.

    The cheap token conditions are evaluated first; the geometry only for
    the elements that pass them on a page with at most
    ``DISTRIBUTION_CODE_MAX_PER_PAGE`` such elements (``few_on_page``), and
    the source ink only for those that pass every other condition, so any
    page costs one pass plus at most that many isolation scans.
    """
    removed = []
    by_token = None
    ink = _SourceInk(source_path, password)
    try:
        for page_num in sorted(page_extractions, key=int):
            elements = page_extractions[page_num].get("elements", []) or []
            page_rect = page_rects.get(page_num)
            if page_rect is None:
                continue
            survivors = _token_survivors(elements)
            if len(survivors) > DISTRIBUTION_CODE_MAX_PER_PAGE:
                continue  # a page of identifiers: no code, no geometry
            for index in survivors:
                elem = elements[index]
                if by_token is None:
                    by_token = _token_rects_by_page(page_extractions)
                token = _token(elem)
                elsewhere = [
                    rect for other_page, rect in by_token.get(token, []) if other_page != page_num
                ]
                conditions = distribution_code_conditions(
                    elem, index, elements, page_rect, elsewhere, token_survivors=survivors
                )
                if not all(ok for name, ok in conditions.items() if name != "invisible_or_outside"):
                    continue
                if not conditions["invisible_or_outside"] and not ink.invisible(
                    page_num, _get_pdf_elem_rect(elem)
                ):
                    continue
                elem["skip_translate_reason"] = SkipReason.WATERMARK
                removed.append({
                    "page": int(page_num) + 1,
                    "element": index,
                    "bbox": [round(float(v), 1) for v in _get_pdf_elem_rect(elem)],
                    "text": token,
                    "kind": "distribution_code",
                })
    finally:
        ink.close()
    return removed


def removed_watermarks_task_fields(audit: dict) -> dict[str, Any]:
    """Task-record fields: ``removed_watermarks`` on delivered pages, or ``{}``
    so records of tasks without one keep their exact shape."""
    delivered = delivered_removed_watermarks(audit)
    return {"removed_watermarks": delivered} if delivered else {}


def reconcile_removed_watermarks(audit: dict, page_extractions: dict) -> None:
    """Before the audit is saved: ``removed_watermarks`` keeps only pages
    delivered with the code removed.  A page restored to its source shows the
    code again, so its record goes and its element loses the watermark skip
    reason (only elements named by a record: other watermarks keep theirs)."""
    records = audit.get("removed_watermarks")
    if not records:
        return
    delivered = delivered_removed_watermarks(audit)
    kept = {(record.get("page"), record.get("element")) for record in delivered}
    for record in records:
        if (record.get("page"), record.get("element")) in kept:
            continue
        page_num = int(record.get("page", 0) or 0) - 1
        info = page_extractions.get(page_num) or page_extractions.get(str(page_num)) or {}
        elements = info.get("elements") or []
        index = record.get("element")
        if isinstance(index, int) and 0 <= index < len(elements):
            if elements[index].get("skip_translate_reason") == SkipReason.WATERMARK:
                elements[index].pop("skip_translate_reason", None)
    if delivered:
        audit["removed_watermarks"] = delivered
    else:
        del audit["removed_watermarks"]


def delivered_removed_watermarks(audit: dict) -> list[dict[str, Any]]:
    """``removed_watermarks`` minus pages delivered as the exact source page."""
    fallback_pages = set()
    for entry in audit.get("source_page_fallbacks") or []:
        try:
            fallback_pages.add(int(entry.get("page", 0) or 0))
        except (AttributeError, TypeError, ValueError):
            continue
    return [
        dict(record)
        for record in audit.get("removed_watermarks") or []
        if int(record.get("page", 0) or 0) not in fallback_pages
    ]
