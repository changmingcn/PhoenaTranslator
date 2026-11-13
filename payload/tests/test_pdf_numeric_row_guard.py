"""A table row keeps its numbers, or it keeps its source.

Condition N: a numeric-dominant source must come back with the same sequence
of numbers (a model comment in place of the row, a dropped value or a
neighbour's row fails).  Condition M: a source without any digit must not
come back with three or more numbers (a row moved into a label).  Translation
validation keeps a failing element as exact source text; assembly never draws
a failing cached translation and keeps the element's source ink.

Synthetic fixtures only.  Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_render_element_fallback import _assert_kept_alone, _kept_records

from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.errors import PDFTranslationIntegrityError
from phoena_translator.pdf.extract import stage as extract_stage
from phoena_translator.pdf.model import SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.translate import stage as translate_stage
from phoena_translator.pdf.translate.numeric_guard import (
    PDF_LABEL_NUMBERS_REASON,
    PDF_NUMERIC_ROW_REASON,
    pdf_numbers,
    pdf_numeric_guard_failure,
    pdf_text_is_numeric_dominant,
)
from phoena_translator.pdf.translate.validation import _validate_pdf_page_translations

log = logging.getLogger("test-numeric-row-guard")

ROW = "Coupons 9.9 5.0 -4.9 9.2 55%"
ZH_ROW = "票息 9.9 5.0 -4.9 9.2 55%"
# A merged table column, mostly values and long enough for the long-text
# checks: a comment in its place passed every check before condition N.
COLUMN = "* Gap Perf " + " ".join(
    ["6 n.a. -2 -4 -2 -6 3 n.a. 3 18 12 n.a. -3 -2 15 -5 16 n.a. 16 -6 10 3 n.a. 4"] * 3
)
LABEL = "Fund demand breakdown"
ZH_LABEL = "基金需求明细"
COMMENT = "这段内容是表格中的一行数据，由数字和缩写构成，没有可翻译的英文句子，因此没有新增的中文译文内容。"


# ---------------------------------------------------------------------------
# The two conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, translated",
    [
        # A model comment in place of the row.
        ("* Gap Perf 16 n.a. 19 12 12 -5 n.a. 15 -5 5", COMMENT),
        # A value dropped.
        ("Widget 7 '43 - Gadget 2 9/16 '38", "Widget 2043年到期 - Gadget 2 9/16厘 2038年到期"),
        # The neighbouring row's values.
        ("Coupons 9.9 5.0 -4.9 9.2 55%", "票息 3.4 3.3 -0.1 2.0 82%"),
        # Values reordered or a sign lost.
        ("12.5 13.1 -0.4 61%", "13.1 12.5 -0.4 61%"),
        ("12.5 13.1 -0.4 61%", "12.5 13.1 0.4 61%"),
        # Empty dates column answered with a comment.
        ("2027-03-01 2027-03-01 2027-03-02", "这是日期列。"),
    ],
)
def test_a_numeric_row_that_loses_or_moves_numbers_fails_n(source, translated):
    assert pdf_text_is_numeric_dominant(source)
    assert pdf_numeric_guard_failure(source, translated) == PDF_NUMERIC_ROW_REASON


@pytest.mark.parametrize(
    "source, translated",
    [
        # Only a label next to unchanged numbers is translated.
        (ROW, ZH_ROW),
        ("4.5 FRN 4.5 4.75 FRN 5.1", "4.5 浮动利率票据 4.5 4.75 浮动利率票据 5.1"),
        # Placeholders for no data, thousands separators, full-width forms.
        ("Total 1,250 n.a. -3 4.5%", "合计 1250 不适用 -3 ４.５％"),
        # Apostrophe years and localized dates.
        ("XYZ (4.25 '33 - 3.9 '29)", "XYZ（4.25% 2033年到期 - 3.9% 2029年到期）"),
        ("Volumes on 3-Mar-27 ($mn) Buy 1,204 - 1,204 377 0 52", "2027年3月3日交易量（百万美元） 买入 1,204 - 1,204 377 0 52"),
        ("4-Feb 24-Feb 16-Mar 5-Apr", "2月4日 2月24日 3月16日 4月5日"),
        ("May-25 Jul-25 Sep-25 Nov-25", "2025年5月 2025年7月 2025年9月 2025年11月"),
        # The source returned verbatim.
        ("12.5 13.1 -0.4 61%", "12.5 13.1 -0.4 61%"),
    ],
)
def test_a_numeric_row_that_keeps_its_numbers_passes(source, translated):
    assert pdf_text_is_numeric_dominant(source)
    assert pdf_numeric_guard_failure(source, translated) is None


@pytest.mark.parametrize(
    "source, translated",
    [
        # Prose with unit conversions is not numeric-dominant.
        (
            "Issuers sold $67bn of bonds in 2027, up from $52bn a year earlier and 4% above plan.",
            "发行人在2027年发行了670亿美元债券，高于上年的520亿美元，比计划高出4%。",
        ),
        # A date line.
        ("Published: 14 Mar 2027 09:30 GMT", "发布时间：2027年3月14日 09:30 GMT"),
        ("Distributed on: 03/14/2027 09:30:05 GMT", "发布时间：2027年3月14日 09:30:05 GMT"),
        # A numbered heading.
        ("3.2 Supply outlook for 2027", "3.2 2027年供给展望"),
        # A label with a number keeps it.
        ("12wk %tile", "12周百分位"),
        # Fewer than three numbers is not a numeric row: signs put into words.
        ("+25bp -10bp", "上升25个基点 下降10个基点"),
    ],
)
def test_prose_dates_and_headings_are_untouched(source, translated):
    assert pdf_numeric_guard_failure(source, translated) is None


def test_a_label_that_acquires_three_numbers_fails_m():
    assert pdf_numeric_guard_failure(LABEL, f"{ZH_LABEL} 9.9 5.0 -4.9 9.2 55%") == (
        PDF_LABEL_NUMBERS_REASON
    )
    assert pdf_numeric_guard_failure("Demand", "需求 36.5 27.1 2.0") == PDF_LABEL_NUMBERS_REASON


@pytest.mark.parametrize(
    "source, translated",
    [
        (LABEL, ZH_LABEL),
        # Fewer than three numbers.
        ("Total demand", "总需求（2027年，十亿美元）"),
        ("Fund demand breakdown", "基金需求明细 9.9 5.0"),
        # Spelled-out numbers and month names may become digits.
        ("Episode four was taped on March ninth", "第4期录制于3月9日"),
        ("seventy-seven hundred, two and a half percent below eighty-one fifty", "7700点，比8150低2.5%"),
        # A source with a digit is outside M.
        ("Top 3 issuers", "前3大发行人 1 2 3"),
    ],
)
def test_labels_that_do_not_acquire_numbers_pass(source, translated):
    assert pdf_numeric_guard_failure(source, translated) is None


def test_numbers_are_read_in_one_canonical_spelling():
    assert pdf_numbers("Baa2 4NC3 1,250 02 +7 -0.30 36% (1.5)") == [
        "4", "1250", "2", "7", "-0.30", "36%", "-1.5",
    ]
    # Footnote markers are excluded, never joined to the number before them.
    assert pdf_numbers("Total<sup>12</sup> 4.0<sup>1</sup> 5.5") == ["4.0", "5.5"]
    assert pdf_numbers("5-Nov-26 2026年11月5日 Nov-26 '31") == [
        "D26-11-5", "D2026-11-5", "D26-11-", "Y31",
    ]


# ---------------------------------------------------------------------------
# Layer 1: translation validation
# ---------------------------------------------------------------------------


def _cell(content, y):
    return {
        "type": "text",
        "layout_class": "table",
        "content": content,
        "bbox": [72.0, y, 400.0, y + 12.0],
        "fontsize": 9.0,
    }


SEG_RE = re.compile(r"PHOENA_SEG_\d{4}_[0-9A-F]{16}")


def _commenting_provider(calls):
    """The row (element 1) is always answered with a comment."""

    def translate_text(text, system_prompt=""):
        calls.append(text)
        if "PHOENA_SEG_" in text:
            seg = SEG_RE.findall(text)
            return f"{seg[0]}\n{ZH_LABEL}\n≡≡≡SPLIT≡≡≡\n{seg[1]}\n{COMMENT}"
        return COMMENT if "Gap Perf" in text else ZH_LABEL

    return translate_text


def _shifting_provider(calls):
    """The row's numbers move into the label (element 1), every time."""

    def translate_text(text, system_prompt=""):
        calls.append(text)
        if "PHOENA_SEG_" in text:
            seg = SEG_RE.findall(text)
            return (
                f"{seg[0]}\n票息\n≡≡≡SPLIT≡≡≡\n"
                f"{seg[1]}\n{ZH_LABEL} 9.9 5.0 -4.9 9.2 55%"
            )
        return ZH_ROW if "Coupons" in text else f"{ZH_LABEL} 9.9 5.0 -4.9 9.2 55%"

    return translate_text


SCENARIOS = {
    # name: (element 0, element 1 = the flawed one, provider, reason, element 0's answer)
    "N": (LABEL, COLUMN, _commenting_provider, PDF_NUMERIC_ROW_REASON, ZH_LABEL),
    "M": (ROW, LABEL, _shifting_provider, PDF_LABEL_NUMBERS_REASON, ZH_ROW),
}


def _run_translation(monkeypatch, scenario, merge_decisions=()):
    first, flawed, provider, _reason, _answer = SCENARIOS[scenario]
    saved = {}
    monkeypatch.setattr(
        translate_stage,
        "_save_pdf_page_translation_cache",
        lambda _task, page, translations, _sha, elements=None: saved.update(
            {page: dict(translations)}
        ),
    )
    elements = [_cell(first, 100.0), _cell(flawed, 112.0)]
    audit = _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = list(merge_decisions)
    context = translate_stage.PDFTranslationStageContext(
        task_id="t",
        total_pages=3,
        source_sha256="0" * 64,
        workers=1,
        fail_open_to_source_page=True,
        logger=log,
        page_extractions={
            0: {"elements": elements},
            1: {"elements": [], "cached": {}},
            2: {"elements": [], "cached": {}},
        },
        pdf_audit=audit,
        completed_indices=set(),
        save_translation_progress=lambda *args, **kwargs: None,
    )
    calls = []
    page_context = PDFPageTranslationContext(
        task_id="t", prompt="翻译", logger=log, translate_text=provider(calls)
    )
    translate_stage.translate_pages(context, page_context, [0])
    return context, elements, saved, calls


@pytest.mark.parametrize("scenario", ["N", "M"])
def test_a_failing_element_is_kept_as_exact_source_and_the_rest_is_translated(
    monkeypatch, scenario
):
    _first, flawed, _provider, reason, first_answer = SCENARIOS[scenario]
    context, elements, saved, calls = _run_translation(monkeypatch, scenario)

    committed = saved[0]
    assert committed[0] == first_answer
    assert committed[1] == flawed
    assert elements[1]["skip_translate_reason"] == SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    (entry,) = context.pdf_audit["element_source_fallbacks"]
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "translation")
    assert entry["reason"] == reason
    assert context.pdf_audit["source_page_fallbacks"] == []
    # Retried as today: individually after the batch, then one clean page retry.
    assert len(calls) > 2


def test_a_failing_merge_endpoint_keeps_the_endpoint_rule(monkeypatch):
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 0,
    }
    context, _elements, saved, _calls = _run_translation(monkeypatch, "N", [merge])

    assert saved == {}
    assert [entry["page"] for entry in context.pdf_audit["source_page_fallbacks"]] == [1, 2]
    assert context.pdf_audit.get("element_source_fallbacks", []) == []


def test_a_batch_segment_failing_a_condition_is_retried_individually():
    from phoena_translator.pdf.translate import page

    context = SimpleNamespace(task_id="t", logger=log)
    row = _cell(ROW, 100.0)
    label = _cell(LABEL, 112.0)
    assert page._restore_batch_part(context, 0, row, 0, ROW, COMMENT, ((), (), ())) is None
    assert page._restore_batch_part(context, 0, row, 0, ROW, ZH_ROW, ((), (), ())) == ZH_ROW
    assert page._restore_batch_part(
        context, 0, label, 1, LABEL, f"{ZH_LABEL} 9.9 5.0 -4.9", ((), (), ())
    ) is None


def test_page_validation_names_the_condition_and_the_cache_rebind_defers_it():
    elements = [_cell(COLUMN, 100.0), _cell(LABEL, 112.0)]
    flawed = {0: COMMENT, 1: f"{ZH_LABEL} 9.9 5.0 -4.9 9.2 55%"}
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, elements, flawed)
    assert raised.value.issues == [
        {"element": 1, "reason": PDF_NUMERIC_ROW_REASON},
        {"element": 2, "reason": PDF_LABEL_NUMBERS_REASON},
    ]
    _validate_pdf_page_translations(1, elements, flawed, reject_numeric_flaws=False)


def test_an_older_page_cache_with_a_numeric_flaw_is_kept_for_assembly(monkeypatch, tmp_path):
    elements = [_cell(LABEL, 100.0), _cell(COLUMN, 112.0)]
    flawed = {"0": ZH_LABEL, "1": COMMENT}
    cache_file = tmp_path / "page_0.json"
    cache_file.write_text("{}")
    discarded = []
    monkeypatch.setattr(extract_stage, "_pdf_page_cache_path", lambda _task, _page: str(cache_file))
    monkeypatch.setattr(
        extract_stage, "_load_pdf_page_translation_cache", lambda *_a, **_k: dict(flawed)
    )
    monkeypatch.setattr(
        extract_stage, "_discard_pdf_page_translation_cache", lambda *a: discarded.append(a)
    )
    monkeypatch.setattr(extract_stage, "_save_pdf_page_translation_cache", lambda *a, **k: None)
    result = SimpleNamespace(
        completed_indices={0},
        cached_indices={0},
        page_extractions={0: {"elements": elements}},
        source_sha256="0" * 64,
        total_pages=1,
    )
    extract_stage._bind_and_reconcile_caches(SimpleNamespace(task_id="t", logger=log), result)

    assert discarded == []
    assert result.completed_indices == {0}
    assert result.page_extractions[0]["cached"]["1"] == COMMENT


# ---------------------------------------------------------------------------
# Layer 2: assembly safety net
# ---------------------------------------------------------------------------


ASSEMBLY_CASES = {
    "N": (COLUMN, COMMENT, PDF_NUMERIC_ROW_REASON),
    "M": ("Fund demand breakdown", "基金需求明细 9.9 5.0 -4.9 9.2 55%", PDF_LABEL_NUMBERS_REASON),
}


def _flawed_cached_page(scenario):
    source, _cached, _reason = ASSEMBLY_CASES[scenario]
    kept = _text([72, 200, 400, 212], source, layout_class="table")
    return [_body(100), kept, _body(400)], kept


@pytest.mark.parametrize("scenario", ["N", "M"])
def test_a_flawed_cached_translation_is_never_drawn_and_keeps_source_ink(
    tmp_path, font_path, scenario
):
    _source, cached, reason = ASSEMBLY_CASES[scenario]
    elements, kept = _flawed_cached_page(scenario)
    document = Document(
        tmp_path, font_path, [(elements, {0: ZH[:30], 1: cached, 2: ZH[5:30]})]
    )

    (text,) = document.assemble()

    assert cached.split(" ")[0][:6] not in _squash(text)
    assert document.fallback_pages() == []
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert entry["reason"].startswith(reason)
    assert "translation withdrawn" in entry["reason"]
    assert kept["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE


def test_a_flawed_cached_merge_endpoint_takes_the_merge_closure(tmp_path, font_path):
    elements, _kept = _flawed_cached_page("N")
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (elements, {0: ZH[:30], 1: ASSEMBLY_CASES["N"][1], 2: ZH[5:30]}),
            ([_body(100), _body(300)], {0: ZH[6:30], 1: ZH[9:30]}),
        ],
        merge_decisions=[merge],
    )

    first, second = document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert ASSEMBLY_CASES["N"][1] not in _squash(first + second)
    assert _kept_records(document) == []


def test_a_correct_cached_row_is_drawn(tmp_path, font_path):
    elements, _kept = _flawed_cached_page("N")
    translated = COLUMN.replace("Gap Perf", "缺口 表现")
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:30], 1: translated, 2: ZH[5:30]})])

    (text,) = document.assemble()

    assert "缺口" in _squash(text)
    assert _kept_records(document) == []


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------

from test_pdf_figure_table_fallback import (  # noqa: E402
    LONG_CELL,
    _cell as _table_cell,
    _formula_over_right_edge,
    _region,
)

from phoena_translator.pdf.render import assembly  # noqa: E402

# S1: an element already kept in source is never a candidate.  Extraction
# glues a superscript digit onto the number before it in the plain text.
SUP_RICH = "Spreads 36.5 27.1 -2.0 30.0<sup>1</sup> 21.8%"
SUP_PLAIN = "Spreads 36.5 27.1 -2.0 30.01 21.8%"


def test_a_plain_source_return_is_not_a_candidate_of_either_condition():
    assert pdf_numeric_guard_failure(SUP_RICH, SUP_PLAIN) is None
    assert pdf_numeric_guard_failure(SUP_RICH, "价差 36.5 27.1 -2.0 30.0<sup>1</sup> 21.8%") is None
    assert pdf_numeric_guard_failure(SUP_RICH, "价差 36.5 27.1 -2.0 31.0<sup>1</sup> 21.8%") == (
        PDF_NUMERIC_ROW_REASON
    )


def _sup_page(content=SUP_PLAIN, **extra):
    kept = _text([72, 200, 400, 212], content, layout_class="table", rich_content=SUP_RICH, **extra)
    return [_body(100), kept, _body(400)]


# The plain text may also hold the superscript as a separate token.
@pytest.mark.parametrize("content", [SUP_PLAIN, SUP_PLAIN.replace("30.01", "30.0 1")])
def test_an_untranslated_element_with_a_superscript_digit_gets_no_record(
    tmp_path, font_path, content
):
    elements = _sup_page(content, skip_translate_reason=SkipReason.TABLE_PRESERVED)
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:30], 1: content, 2: ZH[5:30]})])

    document.assemble()

    assert document.audit.get("element_source_fallbacks", []) == []
    assert document.fallback_pages() == []


def test_an_element_kept_at_translation_for_another_reason_is_recorded_once(tmp_path, font_path):
    elements = _sup_page(skip_translate_reason=SkipReason.TRANSLATION_INTEGRITY_FALLBACK)
    document = Document(tmp_path, font_path, [(elements, {0: ZH[:30], 1: SUP_PLAIN, 2: ZH[5:30]})])
    assembly._record_pdf_element_source_fallback(
        document.audit, 1, 1, "translation", "inline-markup-mismatch", source_text=SUP_PLAIN
    )

    document.assemble()

    entries = [e for e in document.audit["element_source_fallbacks"] if e["element"] == 1]
    assert [(e["stage"], e["reason"]) for e in entries] == [("translation", "inline-markup-mismatch")]
    assert document.fallback_pages() == []


# T4: one eligibility rule for both assembly nets.  A member of a retained
# group is never a candidate of either net, whichever flaw its translation
# has; its group record stands and nothing is recorded twice.
@pytest.mark.parametrize(
    "source, translation",
    [
        ("Net profit 63 77 41", "净利润 64 77 41"),
        ("Cash flows", "摘要PHOENA_<b>IMATH</b>。"),
        ("Cash flows", "摘要PHOENA_IMATH。"),
        ("Cash flows", "摘要PHOENA_IMATH_0000_0123456789ABCDEF。"),
    ],
    ids=["numeric", "markup-token", "cut-token", "whole-token"],
)
def test_a_numeric_flaw_inside_a_retained_table_is_recorded_once(
    tmp_path, font_path, source, translation
):
    elements = [
        _body(100),
        _region([60, 200, 560, 260]),
        _table_cell(72, 210, "Ticker", table_hint=True),
        _table_cell(200, 210, "GA AUM", table_hint=True),
        # Too small for its translation and for its own source text.
        _table_cell(330, 210, LONG_CELL, width=24.0, table_hint=True),
        _table_cell(72, 230, source, width=150.0, table_hint=True),
        _body(400),
    ]
    elements.append(_formula_over_right_edge(elements[4]))
    translations = {0: ZH[:30], 2: "代码", 3: "资产", 4: ZH * 4, 5: translation, 6: ZH[5:30]}
    document = Document(tmp_path, font_path, [(elements, translations)])

    document.assemble()

    (record,) = document.figure_tables()
    assert record["elements"] == [2, 3, 4, 5]
    entries = document.audit.get("element_source_fallbacks", [])
    assert [e["stage"] for e in entries if e["element"] == 5] == ["render-figure-table"]
    indices = [e["element"] for e in entries]
    assert len(indices) == len(set(indices))
    assert document.fallback_pages() == []


def test_an_element_kept_at_translation_is_not_a_token_net_candidate(tmp_path, font_path):
    # Its source shows a cut-off token, which no source owns: the translation
    # stage kept the element as its source text; assembly adds no record.
    content = "Field PHOENA_IMATH is reserved for the diagnostic output"
    kept = _text([72, 200, 540, 212], content, layout_class="body",
                 skip_translate_reason=SkipReason.TRANSLATION_INTEGRITY_FALLBACK)
    document = Document(
        tmp_path, font_path, [([_body(100), kept, _body(400)], {0: ZH[:30], 1: content, 2: ZH[5:30]})]
    )
    assembly._record_pdf_element_source_fallback(
        document.audit, 1, 1, "translation", "internal-placeholder-token", source_text=content
    )

    document.assemble()

    entries = document.audit["element_source_fallbacks"]
    assert [(e["element"], e["stage"]) for e in entries] == [(1, "translation")]
    assert document.fallback_pages() == []


# T3: one sign grammar.  The class is every common spelling of one negative
# value: it reads as that negative value, and as one data token, in source
# and candidate alike.  Each spelling with its positive counterpart.
SINGLE_NEGATIVES = [
    ("({v})", "{v}"),
    ("( {v} )", "{v}"),
    ("({v}bn)", "{v}bn"),
    ("({v}%)", "{v}%"),
    ("(${v})", "${v}"),
    ("$({v})", "${v}"),
    ("-{v}", "{v}"),
    ("- {v}", "{v}"),
    ("\u2212{v}", "{v}"),
    ("\u2013{v}", "{v}"),
    ("-${v}", "${v}"),
    ("$-{v}", "${v}"),
]
NEGATIVE_IDS = [spelling.format(v="v") for spelling, _ in SINGLE_NEGATIVES]
ROW_VALUES = ("1.2", "24.6", "48.9")


def _signed_row(label, spelling):
    return " ".join([label, *(spelling.format(v=value) for value in ROW_VALUES)])


def _kept_unit(spelling):
    """What a negative keeps of its spelling: a percent sign."""
    return "%" if "%" in spelling else ""


def _partners(index):
    """Two other spellings of the same value (a percent value pairs with
    percent spellings: a percent sign may be added, never dropped)."""
    spelling = SINGLE_NEGATIVES[index][0]
    if "%" in spelling:
        return ["-{v}%", "\u2212{v}%"]
    others = [s for s, _ in SINGLE_NEGATIVES if "%" not in s and s != spelling]
    return [others[index % len(others)], others[(index + 4) % len(others)]]


@pytest.mark.parametrize(("spelling", "_positive"), SINGLE_NEGATIVES, ids=NEGATIVE_IDS)
def test_every_spelling_of_one_negative_is_that_value_and_one_data_token(spelling, _positive):
    from phoena_translator.pdf.translate.numeric_guard import (
        _counted_tokens,
        _is_data_token,
        _numeric_view,
    )

    expected = [f"-{value}{_kept_unit(spelling)}" for value in ROW_VALUES]
    for label in ("Net profit", "净利润"):
        assert pdf_numbers(_signed_row(label, spelling)) == expected
    tokens = _counted_tokens(_numeric_view(spelling.format(v="1.2")))
    assert len(tokens) == 1 and _is_data_token(tokens[0]), tokens
    assert pdf_text_is_numeric_dominant(_signed_row("Net profit", spelling))


@pytest.mark.parametrize("index", range(len(SINGLE_NEGATIVES)), ids=NEGATIVE_IDS)
def test_an_equivalent_spelling_of_a_negative_passes_both_ways(index):
    spelling = SINGLE_NEGATIVES[index][0]
    for partner in _partners(index):
        assert pdf_numeric_guard_failure(
            _signed_row("Net profit", spelling), _signed_row("净利润", partner)
        ) is None, (spelling, partner)
        assert pdf_numeric_guard_failure(
            _signed_row("Net profit", partner), _signed_row("净利润", spelling)
        ) is None, (partner, spelling)


@pytest.mark.parametrize(("spelling", "positive"), SINGLE_NEGATIVES, ids=NEGATIVE_IDS)
def test_a_negative_turned_positive_fails_both_ways(spelling, positive):
    assert pdf_numeric_guard_failure(
        _signed_row("Net profit", spelling), _signed_row("净利润", positive)
    ) == PDF_NUMERIC_ROW_REASON
    assert pdf_numeric_guard_failure(
        _signed_row("Net profit", positive), _signed_row("净利润", spelling)
    ) == PDF_NUMERIC_ROW_REASON


@pytest.mark.parametrize(
    "text, numbers",
    [
        ("Range (1.2, 24.6) and (3.4 to 5.6)", ["1.2", "24.6", "3.4", "5.6"]),
        ("Pair ( 1.2 24.6 ) (see note 7)", ["1.2", "24.6", "7"]),
        ("Range 1.5 \u2013 2.5 and 2025-26", ["1.5", "2.5", "2025", "26"]),
    ],
    ids=["comma-and-words", "several-values-and-words", "ranges"],
)
def test_grouping_parentheses_and_ranges_stay_unsigned(text, numbers):
    assert pdf_numbers(text) == numbers


# S2: signs.
@pytest.mark.parametrize(
    "source, translated",
    [
        # Accounting negatives that lose their parentheses.
        ("Net income (30.0) (21.8) 24.62", "净利润 30.0 21.8 24.62"),
        # En-dash negatives that lose their sign.
        ("Change –52.7 14.05 –53.8", "变化 52.7 14.05 53.8"),
        # A sign before a currency symbol lost.
        ("Net profit -$33.6 -$40.6 -$17.7", "净利润 $33.6 $40.6 $17.7"),
        # A dropped percent sign.
        ("Margin 5.14% 32.9% 45.21%", "利润率 5.14 32.9 45.21"),
    ],
)
def test_a_lost_sign_or_percent_fails_n(source, translated):
    assert pdf_text_is_numeric_dominant(source)
    assert pdf_numeric_guard_failure(source, translated) == PDF_NUMERIC_ROW_REASON


@pytest.mark.parametrize(
    "source, translated",
    [
        ("Net income (30.0) (21.8) 24.62", "净利润 -30.0 -21.8 24.62"),
        ("Net income (30.0) (21.8) 24.62", "净利润 （30.0） （21.8） 24.62"),
        ("Change –52.7 14.05 −53.8", "变化 -52.7 14.05 -53.8"),
        ("Net profit -$33.6 $-40.6 −$17.7", "净利润 -33.6 -40.6 -17.7 美元"),
        ("Margin 5.14% 32.9% 45.21%", "利润率 5.14% 32.9％ 45.21%"),
    ],
)
def test_an_equivalent_negative_spelling_passes(source, translated):
    assert pdf_numeric_guard_failure(source, translated) is None


def test_grouping_parentheses_stay_distinguishable_from_an_accounting_negative():
    assert pdf_numbers("Spread (12.3 34.3) (41.3) -$5 $-6 (7%) –8 2027–28") == [
        "12.3", "34.3", "-41.3", "-5", "-6", "-7%", "-8", "2027", "28",
    ]


# S3: leading decimals.
def test_a_leading_decimal_is_the_same_number_and_its_row_is_checked():
    assert pdf_numbers(".5 0.5 -.25 $.75") == ["0.5", "0.5", "-0.25", "0.75"]
    assert pdf_text_is_numeric_dominant("Gap .34 .57 .75")
    assert pdf_numeric_guard_failure("Net profits .34 .57 .75", "净利润 0.34 0.57 0.75") is None
    assert pdf_numeric_guard_failure("Net profits .34 .57 .75", "净利润 34 57 75") == (
        PDF_NUMERIC_ROW_REASON
    )
    assert pdf_numeric_guard_failure("Net profits .34 .57 .75 -- -- --", "这是表格中的统计数据。") == (
        PDF_NUMERIC_ROW_REASON
    )


# S4: full years keep their century.
ZH_DATES = "日期 2027年2月18日 2027年2月28日 2027年2月16日"


def test_a_full_year_keeps_its_century():
    assert pdf_numbers("2027-02-18 18-Feb-27 2027年2月18日") == [
        "D2027-2-18", "D27-2-18", "D2027-2-18",
    ]
    assert pdf_numeric_guard_failure("Dates 1927-02-18 2027-02-28 2027-02-16", ZH_DATES) == (
        PDF_NUMERIC_ROW_REASON
    )
    assert pdf_numeric_guard_failure("Dates 2027-02-18 2027-02-28 2027-02-16", ZH_DATES) is None
    # A short year still matches the full year it abbreviates.
    assert pdf_numeric_guard_failure("Dates 18-Feb-27 28-Feb-27 16-Feb-27", ZH_DATES) is None


# S6: the number-word waiver of condition M.
@pytest.mark.parametrize(
    "source, translated",
    [
        ("First half", "上半年 53.9 36.53 43.38 12.3"),
        ("Second-quarter revenue", "第二季度收入 53.9 36.53 43.38"),
        ("Duration seconds", "时长（秒） 63 77 41"),
        ("Demand may improve", "需求可能改善 63 77 41"),
        ("Issued in March", "三月发行 63 77 41"),
        # One cardinal word exempts one acquired number, not a row.
        ("One-off items", "一次性项目 1 63 77 41"),
    ],
)
def test_ordinary_words_do_not_waive_m_and_number_words_waive_one_each(source, translated):
    assert pdf_numeric_guard_failure(source, translated) == PDF_LABEL_NUMBERS_REASON


# S5: unambiguous dates.
@pytest.mark.parametrize(
    "source",
    [
        "Issue dates 18/02/2027 28/02/2027 16/02/2027",
        "Issue dates 02/18/2027 02/28/2027 02/16/2027",
        "Issue dates 18-02-2027 28-02-2027 16-02-2027",
        "Issue dates 02-18-2027 02-28-2027 02-16-2027",
        "Issue dates 18.02.2027 28.02.2027 16.02.2027",
    ],
)
def test_an_unambiguous_date_in_either_order_passes_its_translation(source):
    assert pdf_text_is_numeric_dominant(source)
    assert pdf_numeric_guard_failure(source, "发行" + ZH_DATES) is None
    assert pdf_numeric_guard_failure(source, "发行" + ZH_DATES.replace("18日", "19日")) == (
        PDF_NUMERIC_ROW_REASON
    )


def test_an_ambiguous_date_keeps_its_reading():
    assert pdf_numbers("11/10/2027 11-10-2027 11.10.2027") == [
        "D2027-11-10", "11", "10", "2027", "11.10",
    ]


# S8: the merge-endpoint membership is computed once per assembly.
class _CountingList(list):
    def __init__(self, items):
        super().__init__(items)
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return super().__iter__()


def _assemble_counting_merge_passes(tmp_path, font_path, flawed):
    tmp_path.mkdir()
    rows = [
        _text([72, 150 + 30 * k, 400, 162 + 30 * k], f"Row {k} 63 77 41 13", layout_class="table")
        for k in range(6)
    ]
    translations = {
        0: ZH[:30],
        **{1 + k: COMMENT if flawed else f"行 {k} 63 77 41 13" for k in range(6)},
    }
    document = Document(tmp_path, font_path, [([_body(80), *rows], translations)])
    decisions = _CountingList(
        {
            "decision": "accepted",
            "source_page": 9,
            "source_element": k,
            "destination_page": 10,
            "destination_element": k,
        }
        for k in range(50)
    )
    document.audit["merge_decisions"] = decisions
    document.assemble()
    return document, decisions


def test_the_merge_endpoint_lookup_is_computed_once_per_assembly(tmp_path, font_path):
    _clean, clean = _assemble_counting_merge_passes(tmp_path / "clean", font_path, False)
    document, flawed = _assemble_counting_merge_passes(tmp_path / "flawed", font_path, True)

    assert len(_kept_records(document)) == 6
    # One pass over the merge list for six withdrawn rows, not one per row.
    assert flawed.passes - clean.passes == 1


# ---------------------------------------------------------------------------
# Review leftovers F1 and F3 (B5), each as a class.
# ---------------------------------------------------------------------------
# F1, the class: a sign never decides whether a token is data.  Every operand
# of a range or ratio may be a signed value (any separator, either or both
# operands, every minus spelling, accounting parentheses, a currency, a unit
# or percent sign); each negative token is data exactly as its positive
# counterpart is, and its row is checked.
SIGNED_OPERANDS = [
    ("1.2/-2.3", "1.2/2.3", ["1.2", "-2.3"]),
    ("1.2:-2.3", "1.2:2.3", ["1.2", "-2.3"]),
    ("1.2–-2.3", "1.2–2.3", ["1.2", "-2.3"]),
    ("1.2--2.3", "1.2-2.3", ["1.2", "-2.3"]),
    ("-1.2/-2.3", "1.2/2.3", ["-1.2", "-2.3"]),
    ("-1.2/2.3", "1.2/2.3", ["-1.2", "2.3"]),
    ("1.2/−2.3", "1.2/2.3", ["1.2", "-2.3"]),
    ("1.2/–2.3", "1.2/2.3", ["1.2", "-2.3"]),
    ("1.2/(2.3)", "1.2/2.3", ["1.2", "-2.3"]),
    ("(1.2)/(2.3)", "1.2/2.3", ["-1.2", "-2.3"]),
    ("$1.2/-$2.3", "$1.2/$2.3", ["1.2", "-2.3"]),
    ("1.2x/-2.3x", "1.2x/2.3x", ["1.2", "-2.3"]),
    ("-1.2%/-2.3%", "1.2%/2.3%", ["-1.2%", "-2.3%"]),
    ("1.2bp/-2.3bp", "1.2bp/2.3bp", ["1.2", "-2.3"]),
    ("1.2/-2.3%", "1.2/2.3%", ["1.2", "-2.3%"]),
    ("1.2/-2.3/-3.4", "1.2/2.3/3.4", ["1.2", "-2.3", "-3.4"]),
]
SIGNED_OPERAND_IDS = [negative for negative, _positive, _values in SIGNED_OPERANDS]


def _operand_row(label, token):
    return f"{label} {token} 24.6 48.9"


@pytest.mark.parametrize(("negative", "positive", "values"), SIGNED_OPERANDS, ids=SIGNED_OPERAND_IDS)
def test_f1_a_signed_operand_keeps_its_token_data_and_its_row_checked(negative, positive, values):
    from phoena_translator.pdf.translate.numeric_guard import (
        _counted_tokens,
        _is_data_token,
        _numeric_view,
    )

    for token in (negative, positive):
        (counted,) = _counted_tokens(_numeric_view(token))
        assert _is_data_token(counted), (token, counted)
        assert pdf_text_is_numeric_dominant(_operand_row("Net profit", token)), token
    assert pdf_numbers(_operand_row("Net profit", negative)) == [*values, "24.6", "48.9"]


@pytest.mark.parametrize(("negative", "positive", "_values"), SIGNED_OPERANDS, ids=SIGNED_OPERAND_IDS)
def test_f1_a_signed_operand_turned_positive_fails_both_ways(negative, positive, _values):
    assert pdf_numeric_guard_failure(
        _operand_row("Net profit", negative), _operand_row("净利润", positive)
    ) == PDF_NUMERIC_ROW_REASON
    assert pdf_numeric_guard_failure(
        _operand_row("Net profit", positive), _operand_row("净利润", negative)
    ) == PDF_NUMERIC_ROW_REASON
    assert pdf_numeric_guard_failure(
        _operand_row("Net profit", negative), _operand_row("净利润", negative)
    ) is None


@pytest.mark.parametrize(
    "token",
    ["3-in-1", "A1/B2", "1.2/abc", "abc/1.2", "x-1.2", "4NC3"],
)
def test_f1_a_token_with_a_word_operand_stays_text(token):
    from phoena_translator.pdf.translate.numeric_guard import (
        _counted_tokens,
        _is_data_token,
        _numeric_view,
    )

    (counted,) = _counted_tokens(_numeric_view(token))
    assert not _is_data_token(counted)


# F3, the class: a negative stays negative whatever stands between its sign or
# accounting parentheses and its digits (a letter-prefixed or ISO currency
# before the parentheses or before or after the sign; spaces after a
# currency; several spaces or a tab after a minus; a space before a percent
# sign inside parentheses; a figure or em dash).  Parentheses that hold a
# letter read as the live release reads them (R-9d,
# ``test_pdf_numeric_units_currency.py``).  Each spelling is that negative value and
# one data token; its positive counterpart is one data token too.
MORE_NEGATIVES = [
    ("-US${v}", "US${v}"),
    ("US$-{v}", "US${v}"),
    ("US$({v})", "US${v}"),
    ("HK$ ({v})", "HK${v}"),
    ("($ {v})", "${v}"),
    ("( $ {v} )", "${v}"),
    ("(€ {v})", "€{v}"),
    ("-€ {v}", "€{v}"),
    ("−$ {v}", "${v}"),
    ("-US$ {v}", "US${v}"),
    ("-EUR {v}", "EUR{v}"),
    ("EUR -{v}", "EUR{v}"),
    ("-RMB{v}", "RMB{v}"),
    ("RMB-{v}", "RMB{v}"),
    ("RMB({v})", "RMB{v}"),
    ("-\t{v}", "{v}"),
    ("-  {v}", "{v}"),
    ("−  {v}", "{v}"),
    ("- \t${v}", "${v}"),
    ("({v} %)", "{v}%"),
    ("‒{v}", "{v}"),
    ("—{v}", "{v}"),
]
MORE_NEGATIVE_IDS = [spelling.format(v="v") for spelling, _ in MORE_NEGATIVES]
# Every spelling of the old matrix and the new one, without percent signs: a
# value written in any of them passes against any other, both ways.
# A spelling with a letter inside its parentheses (``({v}bn)``) puts its pair
# under the live release's reading (R-9d): it is no partner of the new ones.
_LETTERED = re.compile(r"\([^()]*[A-Za-z][^()]*\)")
_PLAIN_FAMILY = [
    s for s, _ in SINGLE_NEGATIVES + MORE_NEGATIVES if "%" not in s and not _LETTERED.search(s)
]
_PERCENT_FAMILY = [s for s, _ in SINGLE_NEGATIVES + MORE_NEGATIVES if "%" in s]


@pytest.mark.parametrize(("spelling", "positive"), MORE_NEGATIVES, ids=MORE_NEGATIVE_IDS)
def test_f3_every_spelling_of_one_negative_is_that_value_and_one_data_token(spelling, positive):
    from phoena_translator.pdf.translate.numeric_guard import (
        _counted_tokens,
        _is_data_token,
        _numeric_view,
    )

    percent = "%" if "%" in spelling else ""
    expected = [f"-{value}{_kept_unit(spelling)}" for value in ROW_VALUES]
    for label in ("Net profit", "净利润"):
        assert pdf_numbers(_signed_row(label, spelling)) == expected
    for written in (spelling, positive):
        tokens = _counted_tokens(_numeric_view(written.format(v="1.2")))
        assert len(tokens) == 1 and _is_data_token(tokens[0]), (written, tokens)
    assert pdf_numbers(_signed_row("Net profit", positive)) == [
        f"{value}{percent}" for value in ROW_VALUES
    ]
    assert pdf_text_is_numeric_dominant(_signed_row("Net profit", spelling))
    assert pdf_text_is_numeric_dominant(_signed_row("Net profit", positive))


@pytest.mark.parametrize(("spelling", "positive"), MORE_NEGATIVES, ids=MORE_NEGATIVE_IDS)
def test_f3_a_negative_turned_positive_fails_both_ways(spelling, positive):
    assert pdf_numeric_guard_failure(
        _signed_row("Net profit", spelling), _signed_row("净利润", positive)
    ) == PDF_NUMERIC_ROW_REASON
    assert pdf_numeric_guard_failure(
        _signed_row("Net profit", positive), _signed_row("净利润", spelling)
    ) == PDF_NUMERIC_ROW_REASON


@pytest.mark.parametrize(("spelling", "_positive"), MORE_NEGATIVES, ids=MORE_NEGATIVE_IDS)
def test_f3_every_equivalent_spelling_passes_both_ways(spelling, _positive):
    family = _PERCENT_FAMILY if "%" in spelling else _PLAIN_FAMILY
    for partner in family:
        assert pdf_numeric_guard_failure(
            _signed_row("Net profit", spelling), _signed_row("净利润", partner)
        ) is None, (spelling, partner)
        assert pdf_numeric_guard_failure(
            _signed_row("Net profit", partner), _signed_row("净利润", spelling)
        ) is None, (partner, spelling)


@pytest.mark.parametrize(("spelling", "_positive"), MORE_NEGATIVES, ids=MORE_NEGATIVE_IDS)
def test_f3_a_pair_with_a_lettered_spelling_gets_the_live_verdict(spelling, _positive):
    """``(1.2bn)`` on one side: the pair is read as the live release reads it,
    which knows a plain minus and none of the new spellings."""
    lettered = _signed_row("Net profit", "({v}bn)")
    assert pdf_numeric_guard_failure(lettered, _signed_row("净利润", "-{v}")) is None
    live_spellings = ("-{v}", "- {v}", "-${v}", "$-{v}", "US$-{v}", "EUR -{v}")
    expected = None if spelling in live_spellings else PDF_NUMERIC_ROW_REASON
    if "%" not in spelling and not spelling.startswith(("(", "US$(", "HK$ (", "RMB(")):
        assert pdf_numeric_guard_failure(lettered, _signed_row("净利润", spelling)) == expected


@pytest.mark.parametrize(
    "text, numbers",
    [
        # A spaced en or em dash is a range, however many spaces.
        ("Range 1.5 – 2.5 and 1.5  –  2.5", ["1.5", "2.5", "1.5", "2.5"]),
        ("Range 1.5 — 2.5 and 2025—26", ["1.5", "2.5", "2025", "26"]),
        ("Range 1.5‒2.5 and 2025-26", ["1.5", "2.5", "2025", "26"]),
        # A dash after a letter or code is part of the code.
        ("Code ABC-1.2 and Baa-2", ["1.2", "2"]),
        # Letters that are not a currency leave parentheses grouping.
        ("Note (EURO 1.2) (FY 2.3) (usd 3.4) (Q1 4.5)", ["1.2", "2.3", "3.4", "4.5"]),
        # Two numbers inside parentheses are grouping; a word inside them
        # leaves them grouping as in the live release (R-9d).
        ("Note (1.2 bn and 2.3) (1.2 bnx)", ["1.2", "2.3", "1.2"]),
    ],
    ids=["spaced-dash-range", "em-dash-range", "figure-dash-between-digits", "codes",
         "non-currency-letters", "words-inside"],
)
def test_f3_ranges_codes_and_grouping_stay_unsigned(text, numbers):
    assert pdf_numbers(text) == numbers
