"""The sign check does not exist in the translation stage (ruling R-20).

A protected sign that touched an amount in the source must still have a
figure next to it in the translation (``prose-number-missing``).  That check
runs only at the assembly net, which keeps the element's source ink through
the stacked-cell rule's keep-or-waive path or records a waiver.  The
translation stage treats such a translation exactly as the previous release
does: no clean page retry because of it, no element kept in source because
of it, no other translation replaced because of it, and the provider called
exactly as often.

Every scenario drives the real ``translate_pages`` (first pass, the clean
page retry, the element fallbacks) with a stub provider; the end-to-end
cases then run the real assembly on what the stage committed.  All texts
are synthetic.  ``run_scenario`` and ``run_pages`` import only what the
previous release also has, so the same scenarios run against it side by
side.
"""

from __future__ import annotations

import inspect
import logging
import re

import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_render_element_fallback import _kept_records

from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.translate import stage as translate_stage
from phoena_translator.pdf.translate import validation

log = logging.getLogger("test-sign-check-stage-paths")

SIGN = "prose-number-missing"
UNTRANSLATED = "untranslated-or-incomplete"
WARNINGS = "delivered_element_warnings"
SEG_RE = re.compile(r"PHOENA_SEG_\d{4}_[0-9A-F]{16}")
IMATH_RE = re.compile(r"PHOENA_IMATH_\d{4}_[0-9A-F]{16}")


# ---------------------------------------------------------------------------
# A synthetic page: a title, two paragraphs with protected signs that touch
# amounts, a paragraph the stub provider can leave in English, and one with
# a plain figure the provider can change.
# ---------------------------------------------------------------------------

REST_A = "随着厂商消化订单，预计各地仓库库存将在春季恢复正常，而更新需求依然稳固。"
REST_N = "地区办公室计划明年保持同样的安排，同时在各个仓库推行新的报表格式。"
TEXTS = {
    "T": {
        "match": "dealer survey",
        "source": "Half-year dealer survey summary",
        "signs": (),
        "forms": {"zh": "半年度经销商调查摘要"},
    },
    "A": {
        "match": "farm machinery",
        "source": (
            "Shipments of farm machinery rose +$7.4bn over the half year, and the "
            "dealers' association expects a further +$2.6bn before the season "
            "closes. Inventories at regional depots should normalise by spring as "
            "producers work through their order books, while replacement demand "
            "stays firm."
        ),
        "signs": ("+", "+"),
        "forms": {
            "zh": "上半年农机出货量增加了{}74亿美元，经销商协会预计本季结束前还会再增加{}26亿美元。"
            + REST_A,
            "lossy": "上半年农机出货量增加了{}，经销商协会预计本季结束前还会再增加{}。" + REST_A,
            # Too short to pass as a translation, and its signs lost their
            # amounts too: one element, two flaws.
            "short": "出货{}，{}。",
        },
    },
    "B": {
        "match": "gearboxes",
        "source": (
            "Lead times for imported gearboxes lengthened by ~30bp of the contract "
            "value during the period, according to the survey, and buyers now place "
            "orders earlier in the season to secure delivery slots before the spring "
            "planting work begins in earnest."
        ),
        "signs": ("~",),
        "forms": {
            "zh": "调查显示，本期进口齿轮箱的交货期延长了合同金额的{}30个基点，"
            "买家如今在季初就提前下单，以便在春耕全面开始前锁定交货档期。",
            "lossy": "调查显示，本期进口齿轮箱的交货期延长了合同金额的{}个基点，"
            "买家如今在季初就提前下单，以便在春耕全面开始前锁定交货档期。",
        },
    },
    "S": {
        "match": "warehouse staffing",
        "source": (
            "Several distributors said that warehouse staffing remained tight "
            "through the period, and that training programmes for new service "
            "technicians were extended to cover the additional repair centres "
            "opened in the northern districts during the previous year."
        ),
        "signs": (),
        "forms": {
            "zh": "多家经销商表示，本期仓库人手依然紧张，新服务技师的培训计划已延长，"
            "以覆盖去年在北部地区新开设的维修中心。",
        },
    },
    "N": {
        "match": "field crews",
        "source": (
            "The northern field crews completed 340 inspections during the season, "
            "and the regional office plans to keep the same schedule next year while "
            "new reporting forms are introduced at every depot."
        ),
        "signs": (),
        "forms": {
            "zh": "北部外勤队本季完成了340次检查，" + REST_N,
            # The same sentence with the figure ten times too large: no
            # validation compares figures in running text.
            "tenfold": "北部外勤队本季完成了3400次检查，" + REST_N,
        },
    },
}


def _element(key: str, y: float, height: float = 30.0) -> dict:
    text = TEXTS[key]
    return {
        "type": "text",
        "layout_class": "body",
        "content": text["source"],
        "bbox": [72.0, y, 540.0, y + height],
        "fontsize": 9.0,
        "inline_math_fragments": [
            {"text": sign, "font_kind": "unicode"} for sign in text["signs"]
        ],
    }


def _rendered(key: str, kind: str) -> str:
    """The translation the page holds once the signs are restored."""
    return TEXTS[key]["forms"][kind].format(*TEXTS[key]["signs"])


def _key_of(part: str):
    return next((key for key, text in TEXTS.items() if text["match"] in part), None)


def _provider(answers: dict, calls: list, batches: list, retry_raises: bool):
    """Answers element ``key`` with ``answers[key][attempt]``: attempt 0 is the
    first pass, 1 the clean page retry (counted by batch requests).  ``"en"``
    (and any text not in ``TEXTS``) returns the English input unchanged.
    ``retry_raises`` makes the retry's batch request fail in transport."""

    def answer(part: str) -> str:
        key = _key_of(part)
        if key is None:
            return part.strip()
        kinds = answers.get(key, ("zh", "zh"))
        kind = kinds[min(max(len(batches) - 1, 0), len(kinds) - 1)]
        if kind == "en":
            return part.strip()
        return TEXTS[key]["forms"][kind].format(*IMATH_RE.findall(part))

    def translate_text(text, system_prompt=""):
        calls.append(text)
        markers = SEG_RE.findall(text)
        if not markers:
            return answer(text)
        batches.append(text)
        if retry_raises and len(batches) > 1:
            raise RuntimeError("synthetic transport failure on the clean retry")
        parts = SEG_RE.split(text)[1:]
        return "\n≡≡≡SPLIT≡≡≡\n".join(
            f"{marker}\n{answer(part)}" for marker, part in zip(markers, parts)
        )

    return translate_text


def _merge(element_index: int) -> dict:
    """An accepted cross-page merge whose page-1 endpoint is ``element_index``."""
    return {
        "decision": "accepted",
        "source_page": 1,
        "source_element": element_index,
        "destination_page": 2,
        "destination_element": 0,
    }


def _held(elements: list[dict], page: dict) -> dict:
    """What each keyed element of a committed page holds: ``"source"``, the
    name of its translation form, or ``"other"``."""
    held = {}
    for index, element in enumerate(elements):
        key = next(
            (key for key, text in TEXTS.items() if text["source"] == element.get("content")),
            None,
        )
        if key is None:
            continue
        value = page.get(index, page.get(str(index)))
        if value == TEXTS[key]["source"]:
            held[key] = "source"
        else:
            held[key] = next(
                (kind for kind in TEXTS[key]["forms"] if value == _rendered(key, kind)),
                "other",
            )
    return held


def _stage_context(pages: dict, audit, fail_open, total_pages):
    return translate_stage.PDFTranslationStageContext(
        task_id="t",
        total_pages=total_pages,
        source_sha256="0" * 64,
        workers=1,
        fail_open_to_source_page=fail_open,
        logger=log,
        page_extractions=pages,
        pdf_audit=audit,
        completed_indices=set(),
        save_translation_progress=lambda *args, **kwargs: None,
    )


def _patch_cache_writer(patch, saved):
    patch(
        translate_stage,
        "_save_pdf_page_translation_cache",
        lambda _task, page, translations, _sha, elements=None: saved.update(
            {page: dict(translations)}
        ),
    )


def run_scenario(
    patch,
    layout,
    answers,
    *,
    merges=(),
    retry_raises=False,
    fail_open=True,
    seed=None,
    elements=None,
    trace=None,
):
    """Drive ``translate_pages`` on page 1 of a three-page document (pages 2
    and 3 come from the cache) and describe the outcome.

    ``patch(obj, name, value)`` replaces the cache writer (``monkeypatch.setattr``).
    ``elements`` replaces the page built from ``layout``.  ``seed`` maps
    element keys to the translations a partial page cache holds.  ``trace``,
    a dict, receives the provider requests, the committed pages, the audit
    and the elements.
    """
    saved = {}
    _patch_cache_writer(patch, saved)
    if elements is None:
        elements = [_element(key, 100.0 + 50.0 * index) for index, key in enumerate(layout)]
    page_info = {"elements": elements}
    if seed:
        page_info["cached_seed"] = {
            str(layout.index(key)): _rendered(key, kind) for key, kind in seed.items()
        }
    audit = _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = [_merge(index) for index in merges]
    context = _stage_context(
        {0: page_info, 1: {"elements": [], "cached": {}}, 2: {"elements": [], "cached": {}}},
        audit,
        fail_open,
        3,
    )
    calls = []
    batches = []
    page_context = PDFPageTranslationContext(
        task_id="t",
        prompt="翻译",
        logger=log,
        translate_text=_provider(answers, calls, batches, retry_raises),
    )
    error = None
    try:
        translate_stage.translate_pages(context, page_context, [0])
    except Exception as raised:  # the task fails on any escape
        error = raised

    if trace is not None:
        trace.update(calls=calls, saved=saved, audit=audit, elements=elements, error=error)
    page = saved.get(0)
    return {
        "status": "failed" if error is not None else "completed",
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "delivered": [0 + 1] if page is not None else [],
        "restored": sorted(
            entry["page"] for entry in audit.get("source_page_fallbacks", [])
        ),
        "source": sorted(
            (entry["element"], entry["reason"])
            for entry in audit.get("element_source_fallbacks", [])
            if entry["page"] == 1
        ),
        "held": _held(elements, page) if page is not None else {},
        "calls": len(calls),
        "batches": len(batches),
    }


def _expected(held, source=(), calls=0, delivered=(1,), restored=(), batches=2):
    return {
        "status": "completed",
        "error": None,
        "delivered": list(delivered),
        "restored": list(restored),
        "source": sorted(source),
        "held": held,
        "calls": calls,
        "batches": batches,
    }


# ---------------------------------------------------------------------------
# Each path the sign check used to take through the stage now ends as the
# previous release ends it (the expected outcomes are the previous release's).
# ---------------------------------------------------------------------------

# (name, layout, answers, options, expected outcome)
SCENARIOS = [
    (
        # The live failure: after the clean retry one element is still
        # untranslated and two others lost their amounts.
        "retry-untranslated-elsewhere-with-two-sign-failures",
        ("T", "A", "B", "S"),
        {"A": ("lossy", "lossy"), "B": ("lossy", "lossy"), "S": ("en", "en")},
        {},
        _expected(
            {"T": "zh", "A": "lossy", "B": "lossy", "S": "source"}, [(3, UNTRANSLATED)], calls=6
        ),
    ),
    (
        # The live sequence: the first pass has only the untranslated
        # element; the retry loses an amount.
        "first-pass-without-sign-issue-retry-loses-an-amount",
        ("T", "A", "S"),
        {"A": ("zh", "lossy"), "S": ("en", "en")},
        {},
        _expected({"T": "zh", "A": "lossy", "S": "source"}, [(2, UNTRANSLATED)], calls=6),
    ),
    (
        "nothing-else-translated-sign-and-untranslated",
        ("A", "S"),
        {"A": ("lossy", "lossy"), "S": ("en", "en")},
        {},
        _expected({"A": "lossy", "S": "source"}, [(1, UNTRANSLATED)], calls=6),
    ),
    (
        # A lost amount on the first pass triggers no retry.
        "first-pass-sign-only-retry-untranslated-elsewhere",
        ("T", "A", "S"),
        {"A": ("lossy", "lossy"), "S": ("zh", "en")},
        {},
        _expected({"T": "zh", "A": "lossy", "S": "zh"}, calls=1, batches=1),
    ),
    (
        "first-pass-sign-only-retry-transport-failure",
        ("T", "A"),
        {"A": ("lossy", "lossy")},
        {"retry_raises": True},
        _expected({"T": "zh", "A": "lossy"}, calls=1, batches=1),
    ),
    (
        "first-pass-sign-only-retry-transport-failure-fail-open-off",
        ("T", "A"),
        {"A": ("lossy", "lossy")},
        {"retry_raises": True, "fail_open": False},
        _expected({"T": "zh", "A": "lossy"}, calls=1, batches=1),
    ),
    (
        "first-pass-sign-only-retry-untranslated-merge-endpoint",
        ("T", "A", "S"),
        {"A": ("lossy", "lossy"), "S": ("zh", "en")},
        {"merges": (2,)},
        _expected({"T": "zh", "A": "lossy", "S": "zh"}, calls=1, batches=1),
    ),
    (
        "faithful-page-committed-on-the-first-pass",
        ("T", "A"),
        {"A": ("zh", "zh")},
        {},
        _expected({"T": "zh", "A": "zh"}, calls=1, batches=1),
    ),
    (
        "first-pass-sign-alone-retry-still-loses-the-amount",
        ("T", "A"),
        {"A": ("lossy", "lossy")},
        {},
        _expected({"T": "zh", "A": "lossy"}, calls=1, batches=1),
    ),
    (
        # No retry, so the retry's faithful answer is never asked for.
        "first-pass-sign-alone-retry-recovers",
        ("T", "A"),
        {"A": ("lossy", "zh")},
        {},
        _expected({"T": "zh", "A": "lossy"}, calls=1, batches=1),
    ),
    (
        # One element, two flaws: too short, and its signs lost their
        # amounts.  Kept in source for the first, as before.
        "same-element-untranslated-and-sign",
        ("T", "A"),
        {"A": ("short", "short")},
        {},
        _expected({"T": "zh", "A": "source"}, [(1, UNTRANSLATED)], calls=6),
    ),
    (
        "nothing-else-translated-sign-alone",
        ("A",),
        {"A": ("lossy", "lossy")},
        {},
        _expected({"A": "lossy"}, calls=1, batches=0),
    ),
    (
        "nothing-else-translated-two-signs-retry-transport-failure",
        ("A", "B"),
        {"A": ("lossy", "lossy"), "B": ("lossy", "lossy")},
        {"retry_raises": True},
        _expected({"A": "lossy", "B": "lossy"}, calls=1, batches=1),
    ),
    (
        "merge-endpoint-sign-element-with-untranslated-elsewhere",
        ("T", "A", "S"),
        {"A": ("lossy", "lossy"), "S": ("en", "en")},
        {"merges": (1,)},
        _expected({"T": "zh", "A": "lossy", "S": "source"}, [(2, UNTRANSLATED)], calls=6),
    ),
    (
        # An untranslated merge endpoint escalates to the merge closure.
        "merge-endpoint-untranslated-with-sign-elsewhere",
        ("T", "A", "S"),
        {"A": ("lossy", "lossy"), "S": ("en", "en")},
        {"merges": (2,)},
        _expected({}, calls=6, delivered=(), restored=(1, 2)),
    ),
    (
        "first-pass-sign-only-retry-sign-on-another-element",
        ("T", "A", "B"),
        {"A": ("lossy", "zh"), "B": ("zh", "lossy")},
        {},
        _expected({"T": "zh", "A": "lossy", "B": "zh"}, calls=1, batches=1),
    ),
    (
        # Another failure, then a transport failure: the page restore.
        "first-pass-untranslated-retry-transport-failure",
        ("T", "A", "S"),
        {"A": ("lossy", "lossy"), "S": ("en", "en")},
        {"retry_raises": True},
        _expected({}, calls=4, delivered=(), restored=(1,)),
    ),
    (
        # A partially cached page (resume) whose cached element lost its
        # amounts: kept for the assembly net, the missing element translated.
        "partial-cache-seed-fails-the-sign-check",
        ("T", "A", "S"),
        {"S": ("zh", "zh")},
        {"seed": {"T": "zh", "A": "lossy"}},
        _expected({"T": "zh", "A": "lossy", "S": "zh"}, calls=1, batches=0),
    ),
    (
        # Second reader's F2: a retry the sign check triggered replaced a
        # sound translation (N) with a wrong figure.  No retry now.
        "f2-sign-failure-retry-would-change-another-figure",
        ("T", "A", "N"),
        {"A": ("lossy", "lossy"), "N": ("zh", "tenfold")},
        {},
        _expected({"T": "zh", "A": "lossy", "N": "zh"}, calls=1, batches=1),
    ),
    (
        "f2-sign-failure-retry-would-recover-and-change-another-figure",
        ("T", "A", "N"),
        {"A": ("lossy", "zh"), "N": ("zh", "tenfold")},
        {},
        _expected({"T": "zh", "A": "lossy", "N": "zh"}, calls=1, batches=1),
    ),
]


@pytest.mark.parametrize(
    ("layout", "answers", "options", "expected"),
    [row[1:] for row in SCENARIOS],
    ids=[row[0] for row in SCENARIOS],
)
def test_the_stage_ends_each_path_as_the_previous_release(
    monkeypatch, layout, answers, options, expected
):
    outcome = run_scenario(monkeypatch.setattr, layout, answers, **options)
    assert outcome == expected
    assert SIGN not in {reason for _element_index, reason in outcome["source"]}


def test_translation_validation_never_reports_the_sign_check():
    elements = [_element("T", 100.0), _element("A", 150.0)]
    translations = {0: _rendered("T", "zh"), 1: _rendered("A", "lossy")}
    validation._validate_pdf_page_translations(1, elements, translations)
    assert "sign_exempt" not in inspect.signature(
        validation._validate_pdf_page_translations
    ).parameters
    assert not hasattr(validation, "pdf_sign_amount_failure")


def test_the_live_failure_no_longer_reaches_the_task_record(monkeypatch):
    """The live escape went through the real recovery to a failed task record."""
    from phoena_translator.pdf import recovery

    name, layout, answers, options, _expected_outcome = SCENARIOS[0]
    escaped = []
    original = translate_stage.translate_pages

    def watched(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as error:
            escaped.append(error)
            raise

    monkeypatch.setattr(translate_stage, "translate_pages", watched)
    outcome = run_scenario(monkeypatch.setattr, layout, answers, **options)
    records = []
    if escaped:
        monkeypatch.setattr(recovery, "_save_recovery_audit", lambda _context: None)
        recovery.recover_pdf_failure(
            recovery.PDFRecoveryContext(
                task_id="t",
                structure_retry=0,
                fail_open_to_source_page=True,
                pdf_audit=_new_pdf_audit("t", 0.6, []),
                load_task=lambda _task_id: {"progress": 1, "total": 3},
                logger=log,
                save_progress=lambda _task_id, record: records.append(record),
                trim_process_memory=lambda: None,
            ),
            escaped[0],
        )
    assert [record["status"] for record in records] == [], records
    assert outcome["status"] == "completed", name


SWEEP_OPTIONS = ({}, {"retry_raises": True}, {"merges": (1,)}, {"merges": (3,)})


def _sweep():
    """Every combination of first-pass and retry answers on a page with two
    sign elements and one that can stay in English: plain, with a retry
    transport failure, with a sign element or the other one as an accepted
    merge endpoint.  (The one-element-two-flaws answer has its own row.)"""
    kinds = {"A": ("zh", "lossy"), "B": ("zh", "lossy"), "S": ("zh", "en")}
    per_try = [
        {"A": a, "B": b, "S": s} for a in kinds["A"] for b in kinds["B"] for s in kinds["S"]
    ]
    for first in per_try:
        for retry in per_try:
            answers = {key: (first[key], retry[key]) for key in kinds}
            for options in SWEEP_OPTIONS:
                yield answers, options


def _without_sign_failures(answers):
    return {
        key: tuple("zh" if kind == "lossy" else kind for kind in kinds)
        for key, kinds in answers.items()
    }


def test_no_combination_lets_the_sign_check_change_the_stage(monkeypatch):
    """The same page with every amount kept is the yardstick: a translation
    whose sign lost its amount changes nothing in the stage.  Same status,
    pages delivered and restored, elements kept in source and reasons,
    provider requests and batch requests; each element holds what the
    yardstick holds, or the lossy form where the yardstick holds the
    faithful one and the provider answered it lossy."""
    layout = ("T", "A", "B", "S")
    monkeypatch.setattr(log, "disabled", True)  # 240 runs; the rows above log
    yardsticks = {}
    checked = 0
    for answers, options in _sweep():
        if answers == _without_sign_failures(answers):
            continue
        outcome = run_scenario(monkeypatch.setattr, layout, answers, **options)
        key = repr((_without_sign_failures(answers), options))
        if key not in yardsticks:
            yardsticks[key] = run_scenario(
                monkeypatch.setattr, layout, _without_sign_failures(answers), **options
            )
        yardstick = yardsticks[key]
        case = (answers, options)
        for field in ("status", "error", "delivered", "restored", "source", "calls", "batches"):
            assert outcome[field] == yardstick[field], (field, case, outcome, yardstick)
        assert set(outcome["held"]) == set(yardstick["held"]), (case, outcome, yardstick)
        for element_key, held in outcome["held"].items():
            assert held == yardstick["held"][element_key] or (
                held == "lossy"
                and yardstick["held"][element_key] == "zh"
                and "lossy" in answers[element_key]
            ), (case, outcome, yardstick)
        checked += 1
    assert checked == len(SWEEP_OPTIONS) * (8 * 8 - 2 * 2)


def test_a_cached_page_failing_the_sign_check_is_resumed_not_discarded(monkeypatch, tmp_path):
    """Resume from cache: the rebind never runs the sign check, so a cached
    translation failing it is kept for the assembly net (which keeps the
    element's source ink), and the page is not translated again."""
    from types import SimpleNamespace

    from phoena_translator.pdf.extract import stage as extract_stage

    layout = ("T", "A")
    elements = [_element(key, 100.0 + 50.0 * index) for index, key in enumerate(layout)]
    cached = {"0": _rendered("T", "zh"), "1": _rendered("A", "lossy")}
    cache_file = tmp_path / "page_0.json"
    cache_file.write_text("{}")
    discarded = []
    monkeypatch.setattr(extract_stage, "_pdf_page_cache_path", lambda _task, _page: str(cache_file))
    monkeypatch.setattr(
        extract_stage, "_load_pdf_page_translation_cache", lambda *_a, **_k: dict(cached)
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
    assert result.page_extractions[0]["cached"] == cached
    assert "cached_seed" not in result.page_extractions[0]


# ---------------------------------------------------------------------------
# Second reader's F1: a request the sign check added could starve another
# page sharing the provider.  Several pages, one provider that answers a
# fixed number of requests and then fails.
# ---------------------------------------------------------------------------


def run_pages(patch, layouts, answers, *, limit, fail_open=True, trace=None):
    """Drive ``translate_pages`` on every page of ``layouts`` with one shared
    provider.  ``answers[page][key]`` lists the element's answers per batch
    request of its page (an individual request repeats the latest); after
    ``limit`` requests every request fails."""
    saved = {}
    _patch_cache_writer(patch, saved)
    pages = {
        number: {"elements": [_element(key, 100.0 + 50.0 * index) for index, key in enumerate(keys)]}
        for number, keys in enumerate(layouts)
    }
    audit = _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = []
    context = _stage_context(pages, audit, fail_open, len(layouts))
    calls = []
    page_batches = {}

    def page_of(part):
        key = _key_of(part)
        return next(number for number, keys in enumerate(layouts) if key in keys and key != "T")

    def answer(number, part):
        key = _key_of(part)
        kinds = answers[number].get(key, ("zh",))
        kind = kinds[min(max(page_batches.get(number, 0) - 1, 0), len(kinds) - 1)]
        if kind == "en":
            return part.strip()
        return TEXTS[key]["forms"][kind].format(*IMATH_RE.findall(part))

    def translate_text(text, system_prompt=""):
        calls.append(text)
        if len(calls) > limit:
            raise RuntimeError(f"synthetic provider quota exhausted after {limit} calls")
        markers = SEG_RE.findall(text)
        if not markers:
            return answer(page_of(text), text)
        parts = SEG_RE.split(text)[1:]
        number = next(page_of(part) for part in parts if _key_of(part) != "T")
        page_batches[number] = page_batches.get(number, 0) + 1
        return "\n≡≡≡SPLIT≡≡≡\n".join(
            f"{marker}\n{answer(number, part)}" for marker, part in zip(markers, parts)
        )

    page_context = PDFPageTranslationContext(
        task_id="t", prompt="翻译", logger=log, translate_text=translate_text
    )
    error = None
    try:
        translate_stage.translate_pages(context, page_context, list(pages))
    except Exception as raised:
        error = raised
    if trace is not None:
        trace.update(calls=calls, saved=saved, audit=audit, error=error)
    return {
        "status": "failed" if error is not None else "completed",
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "delivered": sorted(page + 1 for page in saved),
        "restored": sorted(entry["page"] for entry in audit.get("source_page_fallbacks", [])),
        "source": sorted(
            (entry["page"], entry["element"], entry["reason"])
            for entry in audit.get("element_source_fallbacks", [])
        ),
        "held": {page + 1: _held(pages[page]["elements"], saved[page]) for page in sorted(saved)},
        "calls": len(calls),
    }


# Page 1 always loses its amounts; page 2's paragraph stays English on its
# first batch and individual retries, and is translated by its clean retry.
F1_LAYOUTS = (("T", "A"), ("T", "S"))
F1_ANSWERS = {0: {"A": ("lossy",)}, 1: {"S": ("en", "zh")}}


@pytest.mark.parametrize("fail_open", [True, False], ids=["fail-open", "fail-open-off"])
def test_f1_no_request_for_the_sign_check_starves_another_page(monkeypatch, fail_open):
    outcome = run_pages(
        monkeypatch.setattr, F1_LAYOUTS, F1_ANSWERS, limit=5, fail_open=fail_open
    )

    # The previous release: page 1 committed on its first pass, the fifth
    # request repairs page 2.
    assert outcome == {
        "status": "completed",
        "error": None,
        "delivered": [1, 2],
        "restored": [],
        "source": [],
        "held": {1: {"T": "zh", "A": "lossy"}, 2: {"T": "zh", "S": "zh"}},
        "calls": 5,
    }


def test_f1_control_one_request_fewer_restores_page_two_in_both_releases(monkeypatch):
    outcome = run_pages(monkeypatch.setattr, F1_LAYOUTS, F1_ANSWERS, limit=4)

    assert (outcome["status"], outcome["delivered"], outcome["restored"]) == (
        "completed", [1], [2]
    )
    assert outcome["calls"] == 5  # the fifth request fails


# ---------------------------------------------------------------------------
# End to end: the real stage, then the real assembly of what it committed.
# ---------------------------------------------------------------------------


def _stage_then_assembly(monkeypatch, tmp_path, font_path, elements, answers, *, merges=(),
                         second_page=None):
    trace = {}
    outcome = run_scenario(
        monkeypatch.setattr, None, answers, merges=merges, elements=elements, trace=trace
    )
    pages = [(trace["elements"], trace["saved"][0])]
    if second_page is not None:
        pages.append(second_page)
    document = Document(tmp_path, font_path, pages, merge_decisions=[_merge(i) for i in merges])
    document.audit.update(trace["audit"])
    texts = document.assemble()
    return outcome, trace, document, _squash(texts[0])


def _label(y, content, rich):
    """A short element that does not read as English prose: the page
    translator stores its source (``rich_content`` when present) untranslated."""
    return {
        "type": "text",
        "layout_class": "body",
        "content": content,
        "rich_content": rich,
        "bbox": [72.0, y, 400.0, y + 14.0],
        "fontsize": 9.0,
        "inline_math_fragments": [],
    }


def _only_warning(document, element):
    (warning,) = document.audit[WARNINGS]
    assert (warning["page"], warning["element"], warning["stage"], warning["reason"]) == (
        1, element, "render", SIGN
    )
    return warning


def test_end_to_end_a_paragraph_that_lost_its_amount_keeps_its_source_ink(
    monkeypatch, tmp_path, font_path
):
    elements = [_element("T", 60.0), _element("A", 150.0, 60.0), _element("S", 300.0, 60.0)]
    outcome, _trace, document, text = _stage_then_assembly(
        monkeypatch, tmp_path, font_path, elements, {"A": ("lossy", "lossy")}
    )

    # The stage: as the previous release, one request, the translation committed.
    assert outcome == _expected({"T": "zh", "A": "lossy", "S": "zh"}, calls=1, batches=1)
    # The assembly net: the paragraph keeps its source ink, one record; the
    # page is delivered with its other translations.
    assert document.fallback_pages() == []
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert entry["reason"].startswith(SIGN)
    assert [e["element"] for e in document.audit["element_source_fallbacks"]] == [1]
    assert "农机" not in text
    assert "仓库人手" in text and "经销商调查" in text
    assert WARNINGS not in document.audit


def test_end_to_end_a_page_whose_only_paragraph_lost_its_amount_is_drawn_with_one_warning(
    monkeypatch, tmp_path, font_path
):
    elements = [_element("A", 150.0, 60.0)]
    outcome, _trace, document, text = _stage_then_assembly(
        monkeypatch, tmp_path, font_path, elements, {"A": ("lossy", "lossy")}
    )

    assert outcome == _expected({"A": "lossy"}, calls=1, batches=0)
    assert document.fallback_pages() == []
    assert "农机" in text
    assert _kept_records(document) == []
    assert "no other translation" in _only_warning(document, 0)["waived"]


MARKUP_ECHOES = pytest.mark.parametrize(
    ("content", "rich"),
    [
        # The fifteenth path (review round 6): a label whose markup makes its
        # stored source differ from its plain content.
        ("2025E", "2025<sup>E</sup>"),
        # The second reader's F3 variant: a disclaimer kept in English.
        (
            "Copyright 2026 Sample Analytics. All rights reserved.",
            "Copyright <b>2026</b> Sample Analytics. All rights reserved.",
        ),
    ],
    ids=["label-with-markup", "disclaimer-with-markup"],
)


def _markup_echo_page(monkeypatch, tmp_path, font_path, content, rich):
    elements = [_element("A", 150.0, 60.0), _label(300.0, content, rich)]
    return _stage_then_assembly(
        monkeypatch, tmp_path, font_path, elements, {"A": ("lossy", "lossy")}
    )


@MARKUP_ECHOES
def test_end_to_end_the_stage_commits_a_paragraph_beside_a_markup_echo_as_before(
    monkeypatch, tmp_path, font_path, content, rich
):
    outcome, trace, document, _text = _markup_echo_page(
        monkeypatch, tmp_path, font_path, content, rich
    )

    # The stage commits the paragraph as the previous release does; the
    # other element holds its own source with its markup.
    assert outcome == _expected({"A": "lossy"}, calls=1, batches=0)
    assert trace["audit"].get("element_source_fallbacks", []) == []
    assert re.sub(r"</?\w+>", "", trace["saved"][0][1]) == content
    assert document.fallback_pages() == []


@MARKUP_ECHOES
def test_end_to_end_a_paragraph_beside_a_markup_echo_is_drawn_with_one_warning(
    monkeypatch, tmp_path, font_path, content, rich
):
    # Ruling R-21: an echo of the source is not a translation, whatever
    # markup it carries, so keeping the paragraph would leave the page with
    # nothing translated and the paragraph is drawn as the previous release
    # draws it.
    _outcome, _trace, document, text = _markup_echo_page(
        monkeypatch, tmp_path, font_path, content, rich
    )

    assert document.fallback_pages() == []
    assert "农机" in text
    assert _kept_records(document) == []
    assert "no other translation" in _only_warning(document, 0)["waived"]


@MARKUP_ECHOES
def test_end_to_end_a_paragraph_beside_a_markup_echo_and_a_translation_keeps_its_source_ink(
    monkeypatch, tmp_path, font_path, content, rich
):
    # The mirror: a real translation elsewhere on the page, so the paragraph
    # keeps its source ink as before.
    elements = [_element("A", 150.0, 60.0), _label(300.0, content, rich), _element("S", 400.0, 60.0)]
    outcome, _trace, document, text = _stage_then_assembly(
        monkeypatch, tmp_path, font_path, elements, {"A": ("lossy", "lossy")}
    )

    assert outcome == _expected({"A": "lossy", "S": "zh"}, calls=1, batches=1)
    assert document.fallback_pages() == []
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 0, "render")
    assert entry["reason"].startswith(SIGN)
    assert "农机" not in text
    assert "仓库人手" in text
    assert WARNINGS not in document.audit


def test_end_to_end_an_accepted_merge_end_is_untouched(monkeypatch, tmp_path, font_path):
    elements = [_element("T", 60.0), _element("A", 150.0, 60.0)]
    outcome, _trace, document, text = _stage_then_assembly(
        monkeypatch,
        tmp_path,
        font_path,
        elements,
        {"A": ("lossy", "lossy")},
        merges=(1,),
        second_page=([_body(100)], {0: ZH[:30]}),
    )

    assert outcome == _expected({"T": "zh", "A": "lossy"}, calls=1, batches=1)
    assert document.fallback_pages() == []
    assert "农机" in text
    assert _kept_records(document) == []
    assert WARNINGS not in document.audit


def test_end_to_end_f3_a_later_refusal_never_costs_the_page(monkeypatch, tmp_path, font_path):
    """Second reader's F3: the other translation on the page (B) has no
    formula-safe place.  The stage no longer withdraws A beforehand, so the
    assembly still holds A's translation when it learns that B is refused."""
    elements = [
        {**_element("A", 80.0), "bbox": [72.0, 80.0, 540.0, 160.0]},
        {**_element("B", 400.0), "bbox": [72.0, 400.0, 540.0, 480.0]},
        {"type": "formula_image", "bbox": [0.0, 350.0, 612.0, 792.0]},
    ]
    outcome, _trace, document, text = _stage_then_assembly(
        monkeypatch, tmp_path, font_path, elements, {"A": ("lossy", "lossy"), "B": ("zh", "zh")}
    )

    assert outcome == _expected({"A": "lossy", "B": "zh"}, calls=1, batches=1)
    assert document.fallback_pages() == []
    assert "农机" in text
