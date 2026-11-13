"""Text the source page does not show is neither translated nor redrawn nor
painted over (acceptance item B4).  All fixtures are generated."""

from __future__ import annotations

import fitz
import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)

from phoena_translator.pdf.extract.page import (
    PDF_INVISIBLE_TEXT_SCAN_SHARE,
    _extract_page_elements,
    _mark_pdf_invisible_source,
    _pdf_page_is_scan_layer,
    _pdf_span_invisible,
    _pdf_stream_text_modes_are_non_clipping,
    _pdf_trace_occurrence_invisible,
)
from phoena_translator.pdf.render import assembly
from phoena_translator.pdf.model import ElementType, SkipReason

BODY_LINE = "Body text line that every reader of the page can see clearly."


def _page_with(writer):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    writer(page)
    return doc, page


def _reasons(doc, page):
    return {
        elem.get("content", ""): elem.get("skip_translate_reason")
        for elem in _extract_page_elements(page, doc)
        if elem.get("type") == ElementType.TEXT
    }


def _body_lines(page, count=20, y0=300):
    for row in range(count):
        page.insert_text((72, y0 + 18 * row), BODY_LINE, fontsize=10)


def test_fill_opacity_zero_and_an_invisible_render_mode_are_marked_semi_transparent_is_not():
    def write(page):
        page.insert_text((72, 100), "Visible heading words", fontsize=12)
        page.insert_text((72, 140), "Hidden opacity zero words", fontsize=12, fill_opacity=0)
        page.insert_text((72, 180), "Hidden render mode words", fontsize=12, render_mode=3)
        page.insert_text((72, 220), "Semi transparent words", fontsize=12, fill_opacity=0.7)
        _body_lines(page)

    reasons = _reasons(*_page_with(write))

    assert reasons["Hidden opacity zero words"] == SkipReason.INVISIBLE_SOURCE
    assert reasons["Hidden render mode words"] == SkipReason.INVISIBLE_SOURCE
    assert reasons["Visible heading words"] is None
    assert reasons["Semi transparent words"] is None
    body = [reason for content, reason in reasons.items() if content.startswith("Body text")]
    assert body and all(reason is None for reason in body)


def test_a_line_mixing_visible_and_invisible_spans_is_visible_text():
    def write(page):
        writer = fitz.TextWriter(page.rect)
        writer.append((72, 140), "Hidden part ", fontsize=12)
        writer.write_text(page, opacity=0)
        writer = fitz.TextWriter(page.rect)
        writer.append((150, 140), "shown part", fontsize=12)
        writer.write_text(page)
        _body_lines(page)

    reasons = _reasons(*_page_with(write))

    assert SkipReason.INVISIBLE_SOURCE not in reasons.values()


def test_a_large_invisible_letter_behind_a_logo_is_marked():
    def write(page):
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(42, 38, 288, 80))
        shape.finish(fill=(0.2, 0.3, 0.7), color=None)
        shape.commit()
        page.insert_text((60, 90), "M", fontsize=100, fill_opacity=0, color=(0.4, 0.45, 0.7))
        page.insert_text((300, 60), "9 April 2030", fontsize=9)
        _body_lines(page)

    reasons = _reasons(*_page_with(write))

    assert reasons["M"] == SkipReason.INVISIBLE_SOURCE
    assert reasons["9 April 2030"] is None


def _scan_layer_page(invisible_lines, visible_lines):
    def write(page):
        for row in range(invisible_lines):
            page.insert_text((72, 80 + 18 * row), BODY_LINE, fontsize=10, render_mode=3)
        for row in range(visible_lines):
            page.insert_text((72, 80 + 18 * (invisible_lines + row)), BODY_LINE, fontsize=10)

    return _page_with(write)


def test_a_page_of_mostly_invisible_text_is_a_scan_layer_and_is_translated_as_before():
    # 30 invisible lines, 2 visible: 94% of the page's text is invisible.
    reasons = _extract_page_elements(*reversed(_scan_layer_page(30, 2)))
    assert not any(
        elem.get("skip_translate_reason") == SkipReason.INVISIBLE_SOURCE for elem in reasons
    )


def _span_dict(*spans):
    return {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": text, "paint_invisible": hidden} for text, hidden in spans]}
                ],
            }
        ]
    }


def test_the_scan_layer_share_decides_on_both_sides():
    assert PDF_INVISIBLE_TEXT_SCAN_SHARE == 0.5
    assert not _pdf_page_is_scan_layer(_span_dict(("x" * 49, True), ("y" * 51, False)))
    assert _pdf_page_is_scan_layer(_span_dict(("x" * 50, True), ("y" * 50, False)))
    assert not _pdf_page_is_scan_layer(_span_dict(("   ", True)))

    def elem(content):
        return {"type": ElementType.TEXT, "content": content, "source_invisible_text": content}

    below = [elem("x" * 49), {"type": ElementType.TEXT, "content": "y" * 51}]
    assert _mark_pdf_invisible_source(below, scan_layer=False) == 1
    assert below[0]["skip_translate_reason"] == SkipReason.INVISIBLE_SOURCE
    assert all("source_invisible_text" not in item for item in below)

    at = [elem("x" * 50), {"type": ElementType.TEXT, "content": "y" * 50}]
    assert _mark_pdf_invisible_source(at, scan_layer=True) == 0
    assert "skip_translate_reason" not in at[0]
    assert all("source_invisible_text" not in item for item in at)


def test_the_scan_exemption_is_decided_before_elements_are_merged():
    """An invisible body that semantic merging joins with a visible line must
    still count as invisible text: the page is a scan layer and a separate
    hidden heading is not marked (R-11)."""

    def write(page):
        page.insert_text((72, 80), "Hidden chapter heading", fontsize=18, render_mode=3)
        for row in range(20):
            page.insert_text(
                (72, 150 + 18 * row),
                "The quarterly report reviews activity across the region and "
                "explains the changes in demand.",
                fontsize=10,
                render_mode=3,
            )
        page.insert_text((72, 510), "A visible concluding note.", fontsize=10)

    doc, page = _page_with(write)
    elements = _extract_page_elements(page, doc)

    assert not any(
        elem.get("skip_translate_reason") == SkipReason.INVISIBLE_SOURCE for elem in elements
    )


def test_a_note_is_honoured_only_while_the_element_holds_exactly_its_text():
    merged = {
        "type": ElementType.TEXT,
        "content": "Hidden words and a visible continuation",
        "source_invisible_text": "Hidden words",
    }
    filler = {"type": ElementType.TEXT, "content": "y" * 400}
    assert _mark_pdf_invisible_source([merged, filler], scan_layer=False) == 0
    assert "skip_translate_reason" not in merged


def test_an_invisible_distribution_code_is_left_to_its_own_detector():
    code = {
        "type": ElementType.TEXT,
        "content": "Q7xe2k9Rt4mzWp",
        "source_invisible_text": "Q7xe2k9Rt4mzWp",
        "layout_class": "scattered",
        "paragraphs": [{"plain": "Q7xe2k9Rt4mzWp", "source_lines": [{"plain": "Q7xe2k9Rt4mzWp"}]}],
    }
    filler = {"type": ElementType.TEXT, "content": "y" * 400}
    assert _mark_pdf_invisible_source([code, filler], scan_layer=False) == 0
    assert "skip_translate_reason" not in code


def test_an_element_with_another_skip_reason_keeps_it():
    elem = {
        "type": ElementType.TEXT,
        "content": "Hidden words",
        "source_invisible_text": "Hidden words",
        "skip_translate_reason": SkipReason.EMBEDDED_THUMBNAIL_TEXT,
    }
    filler = {"type": ElementType.TEXT, "content": "y" * 400}
    _mark_pdf_invisible_source([elem, filler], scan_layer=False)
    assert elem["skip_translate_reason"] == SkipReason.EMBEDDED_THUMBNAIL_TEXT


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

LOGO = fitz.Rect(42, 38, 288, 80)
LOGO_COLOUR = (0.2, 0.3, 0.7)


def _logo_document(tmp_path, font_path, *, invisible_reason):
    letter = _text([42.0, -13.3, 287.8, 94.0], "M", fontsize=107.0, layout_class="scattered")
    if invisible_reason:
        letter["skip_translate_reason"] = SkipReason.INVISIBLE_SOURCE
    date_line = _text([200, 84, 300, 93], "9 April 2030", fontsize=9.0, layout_class="scattered")
    title = _text([72, 120, 540, 135], "A translated page title", fontsize=12.0, layout_class="heading")
    elements = [letter, date_line, title]
    document = Document(
        tmp_path, font_path, [(elements, {0: "M", 1: "2030年4月9日", 2: ZH[:12]})]
    )
    # Replace the generated source with one drawing the logo and the letter
    # with fill opacity 0, as the source page does.
    document.context.out_doc.close()
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    shape = page.new_shape()
    shape.draw_rect(LOGO)
    shape.finish(fill=LOGO_COLOUR, color=None)
    shape.commit()
    page.insert_text((60, 92), "M", fontsize=100, fill_opacity=0, color=(0.4, 0.45, 0.7))
    page.insert_text((200, 92), "9 April 2030", fontsize=9)
    page.insert_text((72, 134), "A translated page title", fontsize=12)
    source.save(str(document.source_path))
    source.save(str(document.work_path))
    source.close()
    document.context.out_doc = fitz.open(str(document.work_path))
    return document


def _visible_letter_spans(path):
    with fitz.open(str(path)) as doc:
        return [
            span
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip() == "M" and span.get("alpha", 255) > 0
        ]


def _logo_pixel(path, point):
    with fitz.open(str(path)) as doc:
        pix = doc[0].get_pixmap(dpi=72, clip=fitz.Rect(point, point.x + 1, point.y + 1))
        return pix.pixel(0, 0)


def test_an_invisible_element_touched_by_a_translated_neighbour_is_not_redrawn_or_painted(
    tmp_path, font_path
):
    document = _logo_document(tmp_path, font_path, invisible_reason=True)

    (text,) = document.assemble()

    assert document.fallback_pages() == []
    assert "2030年4月9日" in _squash(text)
    assert _visible_letter_spans(document.work_path) == []
    redraws = [
        event
        for event in document.audit.get("render_scale_events", [])
        if event.get("element") == 0
    ]
    assert redraws == []
    # The logo under the letter's box is not painted white.
    red, green, blue = _logo_pixel(document.work_path, fitz.Point(60, 50))
    assert blue > 120 and red < 100


def test_without_the_mark_the_letter_is_redrawn_visibly(tmp_path, font_path):
    """Today's behaviour, kept as the contrast the mark removes."""
    document = _logo_document(tmp_path, font_path, invisible_reason=False)

    document.assemble()

    assert _visible_letter_spans(document.work_path) != []


@pytest.mark.parametrize("translation", ["M", "米"])
def test_an_invisible_element_is_never_a_translation_candidate(tmp_path, font_path, translation):
    document = _logo_document(tmp_path, font_path, invisible_reason=True)
    document.context.page_extractions[0]["cached"]["0"] = translation

    (text,) = document.assemble()

    assert "米" not in _squash(text)
    assert _visible_letter_spans(document.work_path) == []


@pytest.mark.parametrize(
    ("occurrence", "invisible"),
    [
        ({"type": 3, "opacity": 1.0}, True),  # the invisible render mode
        ({"type": 0, "opacity": 0.0}, True),  # a fill with fill opacity 0
        ({"type": 0, "opacity": 0.5}, False),
        ({"type": 0, "opacity": 1.0}, False),
        ({"type": 1, "opacity": 1.0}, False),  # a stroke is visible
        ({"type": 1, "opacity": 0.0}, False),  # not one of the two invisible kinds
    ],
)
def test_a_paint_occurrence_is_invisible_only_in_the_two_named_kinds(occurrence, invisible):
    assert _pdf_trace_occurrence_invisible(occurrence) is invisible


def test_span_flags_no_longer_decide_visibility():
    """The extraction's span flags lose a stroke over a transparent fill; the
    span is invisible only by the record read from the text trace."""
    assert not _pdf_span_invisible({"alpha": 0, "char_flags": 16})
    assert not _pdf_span_invisible({"alpha": 255, "char_flags": 0})
    assert _pdf_span_invisible({"alpha": 0, "char_flags": 16, "paint_invisible": True})


def test_an_invisible_element_is_no_candidate_of_any_assembly_net():
    letter = _text([42, 0, 288, 94], "M", fontsize=100.0)
    letter["skip_translate_reason"] = SkipReason.INVISIBLE_SOURCE
    assert assembly._assembly_net_candidates([letter], {0: "米"}, {}, ()) == []


# ---------------------------------------------------------------------------
# R-7: visibility is read from how the text is painted
# ---------------------------------------------------------------------------

CASE_TEXT = "Wpq"


def _append_content(page, stream: bytes) -> None:
    doc = page.parent
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, stream)
    contents = [*page.get_contents(), xref]
    doc.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in contents) + "]")


def _case_fill_opacity_zero(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, fill_opacity=0)


def _case_invisible_mode(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, render_mode=3)


def _case_stroke_over_transparent_fill(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, render_mode=2, fill_opacity=0)


def _case_stroke_clip_over_transparent_fill(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, render_mode=6, fill_opacity=0)


def _case_stroke_only(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, render_mode=1, fill_opacity=0)


def _case_transparent_then_opaque(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, fill_opacity=0)
    page.insert_text((72, 140), CASE_TEXT, fontsize=60)


def _case_opaque_then_transparent(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60)
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, fill_opacity=0)


def _case_fill_clip_transparent(page):
    page.insert_text((72, 140), CASE_TEXT, fontsize=60, render_mode=4, fill_opacity=0)


def _case_clip_with_paint_through(page):
    # Render mode 7: the glyphs only clip; a blue fill is painted through them.
    _append_content(
        page,
        b"q BT /helv 60 Tf 7 Tr 1 0 0 1 72 652 Tm (" + CASE_TEXT.encode() + b") Tj ET "
        b"0 0 1 rg 60 630 220 90 re f Q",
    )


# (writer, marked, shows ink).  A mode that clips is never marked even when
# nothing is painted through it here: content painted later through the clip
# would show (the paint-through case).
PAINT_CASES = [
    (_case_fill_opacity_zero, True, False),
    (_case_invisible_mode, True, False),
    (_case_stroke_over_transparent_fill, False, True),
    (_case_stroke_clip_over_transparent_fill, False, True),
    (_case_stroke_only, False, True),
    (_case_transparent_then_opaque, False, True),
    (_case_opaque_then_transparent, False, True),
    (_case_fill_clip_transparent, False, False),
    (_case_clip_with_paint_through, False, True),
]


def _paint_case_page(writer):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), "Visible body text.", fontsize=10)  # registers the font
    writer(page)
    _body_lines(page, count=12, y0=340)
    doc = fitz.open("pdf", doc.tobytes())
    return doc, doc[0]


def _ink(page, rect) -> int:
    pix = page.get_pixmap(dpi=72, clip=rect, alpha=False)
    samples = pix.samples
    step = pix.n
    return sum(1 for index in range(0, len(samples), step) if min(samples[index : index + step]) < 200)


CASE_RECT = fitz.Rect(60, 80, 300, 160)


@pytest.mark.parametrize(
    ("writer", "marked", "shows_ink"),
    PAINT_CASES,
    ids=[case[0].__name__.removeprefix("_case_") for case in PAINT_CASES],
)
def test_each_paint_case_is_marked_only_when_erasing_it_loses_no_ink(writer, marked, shows_ink):
    doc, page = _paint_case_page(writer)
    elements = _extract_page_elements(page, doc)
    case = [
        elem
        for elem in elements
        if elem.get("type") == ElementType.TEXT and CASE_TEXT in (elem.get("content") or "")
    ]
    flagged = [elem for elem in case if elem.get("skip_translate_reason") == SkipReason.INVISIBLE_SOURCE]
    assert bool(flagged) is marked
    if marked:
        assert flagged == case

    # Erase the case's glyphs the way a neighbour's redaction does and count
    # the ink before and after.
    before = _ink(page, CASE_RECT)
    page.add_redact_annot(CASE_RECT, fill=False)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    after = _ink(page, CASE_RECT)
    if marked or not shows_ink:
        assert before == 0 and after == 0
    else:
        # Each visible case shows ink, and erasing it would lose that ink:
        # this is why it is never marked.
        assert before > 0 and after < before


# ---------------------------------------------------------------------------
# R-7b, R-7c: uncertain evidence of clipping means nothing is marked.  A
# page's text is marked invisible only when every text render mode operator
# in its content and forms is read with certainty as carrying a non-clipping
# mode (0-3).  The streams are read by a small lexer that skips comments,
# literal strings and hexadecimal strings with certainty: what they hold is
# never taken for the operator.
# ---------------------------------------------------------------------------

LETTER = "M"


def _clip_paint_through_page(spelling: bytes):
    """The second reader's reproduction: a transparent glyph in clip mode 4,
    a blue fill painted through its clip, the mode operator respelt."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((85, 110), "Neighbouring note", fontsize=9)
    for row in range(12):
        page.insert_text((72, 340 + 18 * row), "Plain words of the body stay on the page.", fontsize=10)
    page.insert_text((72, 140), LETTER, fontsize=60, render_mode=4, fill_opacity=0)
    xref = page.get_contents()[-1]
    raw = doc.xref_stream(xref)
    state = doc.get_new_xref()
    doc.update_object(state, "<< /Type /ExtGState /ca 1 /CA 1 >>")
    resources = int(doc.xref_get_key(page.xref, "Resources")[1].split()[0])
    doc.xref_set_key(resources, "ExtGState/OPAQUE", f"{state} 0 R")
    assert raw.count(b"4 Tr") == 1
    raw = raw.replace(b"4 Tr", spelling).replace(
        b"ET\nQ", b"ET\n/OPAQUE gs\n0 0 1 rg\n60 630 150 90 re f\nQ"
    )
    doc.update_stream(xref, raw)
    doc = fitz.open("pdf", doc.tobytes())
    return doc, doc[0]


def _blue(page) -> int:
    pix = page.get_pixmap(dpi=72, clip=fitz.Rect(60, 70, 230, 155), alpha=False)
    samples = pix.samples
    return sum(
        1
        for index in range(0, len(samples), 3)
        if samples[index + 2] - max(samples[index], samples[index + 1]) > 40
    )


CLIP_SPELLINGS = {
    "canonical": b"4 Tr",
    "comment": b"4 % legal comment\n Tr",
    "signed": b"+4 Tr",
    "padded": b"04 Tr",
    "decimal": b"4.0 Tr",
    "other-white-space": b"4\x0c\x00Tr",
    "glued": b" 4Tr",
    # MuPDF takes the first of two operands, so this clips.
    "two-operands": b"4 1 Tr",
    "after-a-string": b"(x) 4 Tr",
}


@pytest.mark.parametrize("spelling", list(CLIP_SPELLINGS.values()), ids=list(CLIP_SPELLINGS))
def test_text_painted_through_a_clip_is_never_marked_however_the_mode_is_spelt(spelling):
    doc, page = _clip_paint_through_page(spelling)
    assert _blue(page) > 0  # the clip shows the blue fill: the glyph is visible
    reasons = _reasons(doc, page)
    assert reasons[LETTER] is None
    assert SkipReason.INVISIBLE_SOURCE not in reasons.values()


def _logo_page(extra: bytes | None = None, form: bytes | None = None):
    """The logo lettering of the earlier tests (a transparent letter, no mode
    operator), with an extra content stream or a form XObject."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 90), LETTER, fontsize=100, fill_opacity=0, color=(0.4, 0.45, 0.7))
    page.insert_text((300, 60), "9 April 2030", fontsize=9)
    _body_lines(page)
    if extra is not None:
        _append_content(page, extra)
    if form is not None:
        xref = doc.get_new_xref()
        doc.update_object(xref, "<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] >>")
        doc.update_stream(xref, form)
        resources = int(doc.xref_get_key(page.xref, "Resources")[1].split()[0])
        doc.xref_set_key(resources, "XObject/FmT", f"{xref} 0 R")
        _append_content(page, b"q /FmT Do Q")
    doc = fitz.open("pdf", doc.tobytes())
    return doc, doc[0]


def test_the_logo_lettering_is_still_marked_and_a_page_without_the_operator_is_unaffected():
    doc, page = _logo_page()
    assert _pdf_stream_text_modes_are_non_clipping(page.read_contents())
    assert b"Tr" not in page.read_contents()
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
    # Every non-clipping mode spelt plainly: still marked.
    plain = b"q BT /helv 9 Tf 0 Tr 1 Tr 2 Tr 3 Tr ET Q"
    doc, page = _logo_page(extra=plain)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
    doc, page = _logo_page(form=plain)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
    # Other legal spellings of a non-clipping mode, read by the lexer.
    respelt = b"q BT +3 Tr 03 Tr -0 Tr 2 % note\n Tr 1\x0c\x00Tr ET Q"
    doc, page = _logo_page(extra=respelt)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
    doc, page = _logo_page(form=respelt)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE


TEXT = b"/helv 9 Tf 1 0 0 1 72 40 Tm "
LOOK_ALIKES = {
    "literal-string": b"q BT " + TEXT + b"(see 4 Tr) Tj ET Q",
    "kerned-word-in-a-form": b"q BT " + TEXT + b"[(Tr) -20 ( Tr) 15 (ack)] TJ ET Q",
    "nested-and-escaped": b"q BT " + TEXT + b"(a (4 Tr) \\) 5 Tr \\( b) Tj ET Q",
    "escaped-backslash": b"q BT " + TEXT + b"(x\\\\) Tj 0 Tr ET Q",
    "hex-string": b"q BT 0 Tr " + TEXT + b"<34 20 54 72> Tj ET Q",
    "comment": b"q % 4 Tr in a comment\nBT 0 Tr ET Q",
    "comment-with-a-parenthesis": b"% (4 Tr\nq BT 1 Tr ET Q",
    "dictionary-and-name": b"/Span << /ActualText (4 Tr) /Tr 4 >> BDC BT 0 Tr ET EMC",
}


@pytest.mark.parametrize("stream", list(LOOK_ALIKES.values()), ids=list(LOOK_ALIKES))
def test_a_string_or_comment_holding_the_operators_letters_does_not_prevent_marking(stream):
    """R-7c: the lexer skips strings (nested, escaped, hexadecimal) and
    comments with certainty, so the operator's letters inside them -- the
    kerning strings of a word that begins ``Tr``, set in a form -- leave the
    logo lettering marked."""
    assert _pdf_stream_text_modes_are_non_clipping(stream)
    doc, page = _logo_page(extra=stream)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
    doc, page = _logo_page(form=stream)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE


INLINE_IMAGE = b"q 2 0 0 2 500 700 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x80 EI Q"


@pytest.mark.parametrize(
    "stream",
    [
        b"q BT 5 Tr ET Q",
        b"q BT 3.0 Tr ET Q",
        b"q BT 1 3 Tr ET Q",
        b"q BT /M 3 Tr ET Q",
        b"q BT (3) Tr ET Q",
        b"q BT Tr ET Q",
        b"q BT 3Tr ET Q",
        INLINE_IMAGE + b" q BT 0 Tr ET Q",
        b"q BT 0 Tr ET Q (open 4 Tr",
        b"q BT (x\\) Tj 0 Tr ET Q",
        b"q ) BT 0 Tr ET Q",
        b"q BT 0 Tr <4 Tr> Tj ET Q",
        b"q BT 1.2.3 Tc 0 Tr ET Q",
    ],
    ids=["clip-mode", "real-operand", "two-operands", "name-before-operand", "string-operand",
         "no-operand", "glued", "inline-image", "unbalanced-string", "escaped-close",
         "stray-close", "invalid-hex-string", "malformed-number"],
)
def test_any_operator_not_read_with_certainty_unmarks_the_whole_page(stream):
    doc, page = _logo_page(extra=stream)
    assert _reasons(doc, page)[LETTER] is None
    doc, page = _logo_page(form=stream)
    assert _reasons(doc, page)[LETTER] is None


def test_an_inline_image_prevents_marking_unless_no_operator_can_follow_it():
    """R-7c: an inline image's data cannot be skipped with certainty, so an
    inline image ahead of the operator's letters leaves the page unmarked.
    After the stream's last ``Tr`` bytes nothing can be the operator: an
    inline image there (closing a page's content, say) leaves the mark, and
    so does one in a stream without the operator."""
    for stream, marked in (
        (INLINE_IMAGE + b" q BT 0 Tr ET Q", False),
        (b"q BT 0 Tr ET Q " + INLINE_IMAGE, True),
        (INLINE_IMAGE, True),
    ):
        for where in ("extra", "form"):
            doc, page = _logo_page(**{where: stream})
            reason = _reasons(doc, page)[LETTER]
            assert (reason == SkipReason.INVISIBLE_SOURCE) is marked, (stream, where, reason)


@pytest.mark.parametrize(
    "stream, certain",
    [
        # Read with certainty: no operator, or one integer operand 0-3.
        (b"BT 0 Tr 1 Tr\n2\tTr 3\x00Tr ET", True),
        (b"3 Tr", True),
        (b"q 1 0 0 1 0 0 cm 2 Tr", True),
        (b"+3 Tr", True),
        (b"03 Tr", True),
        (b"-0 Tr", True),
        (b"3 %c\n Tr", True),
        (b"3%c\rTr", True),
        (b"/TrimBox 1 Tra", True),
        (b"/Tr 3", True),
        (b"(3 Tr) Tj", True),
        (b"[(Tr) 5 ( Tr)] TJ", True),
        (b"(a (b) \\) 4 Tr) Tj", True),
        (b"(x\\\\) Tj 0 Tr", True),
        (b"(\\\n4 Tr) Tj", True),
        (b"<54 72> Tj <> Tj 1 Tr", True),
        (b"<< /Tr 4 /S (4 Tr) >> BDC 0 Tr", True),
        (b"% 4 Tr", True),
        # Nothing after the last ``Tr`` bytes can be the operator.
        (b"BI /W 1 /H 1 ID x EI", True),
        (b"0 Tr BI /W 1 /H 1 ID (Tq EI", True),
        (b"0 Tr (open", True),
        (b"0 Tr 1.2.3 Tc", True),
        # Uncertain.
        (b"4 Tr", False),
        (b"7 Tr", False),
        (b"-3 Tr", False),
        (b"13 Tr", False),
        (b"3.0 Tr", False),
        (b"3. Tr", False),
        (b"Tr", False),
        (b"3Tr", False),
        (b"+3Tr", False),
        (b"]3 Tr", False),
        (b"[3] Tr", False),
        (b"1 3 Tr", False),
        (b"/N 3 Tr", False),
        (b"(x) 3 Tr", False),
        (b"<41> 3 Tr", False),
        (b"true 3 Tr", False),
        (b"BI /W 1 /H 1 ID 0 Tr EI", False),
        (b"BI /W 1 /H 1 ID x EI 0 Tr", False),
        (b"(open 0 Tr", False),
        (b"(x\\) 0 Tr", False),
        (b") 0 Tr", False),
        (b"> 0 Tr", False),
        (b"<4 Tr> 0 Tr", False),
        (b"<41 0 Tr", False),
        (b"1.2.3 Tc 0 Tr", False),
    ],
)
def test_the_lexer_reads_the_operator_and_its_one_operand_with_certainty(stream, certain):
    assert _pdf_stream_text_modes_are_non_clipping(stream) is certain


def test_image_data_that_looks_like_the_operator_is_not_page_content():
    """Only the page's content streams and its forms are read: the samples of
    an image XObject are data, whatever bytes they hold."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 90), LETTER, fontsize=100, fill_opacity=0, color=(0.4, 0.45, 0.7))
    page.insert_text((300, 60), "9 April 2030", fontsize=9)
    _body_lines(page)
    samples = b" 4 Tr " * 3  # six RGB pixels whose bytes read "4 Tr"
    pixmap = fitz.Pixmap(fitz.csRGB, 6, 1, samples, False)
    page.insert_image(fitz.Rect(400, 700, 460, 710), pixmap=pixmap)
    doc = fitz.open("pdf", doc.tobytes())
    page = doc[0]
    (image,) = [item[0] for item in page.get_images()]
    assert b"4 Tr" in doc.xref_stream(image)
    assert _reasons(doc, page)[LETTER] == SkipReason.INVISIBLE_SOURCE
