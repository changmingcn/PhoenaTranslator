"""Parentheses that hold a letter read as the live release reads them
(R-9d); the currency codes are a closed list (R-10).

R-9d: the sign of a parenthesised number never depends on what else is
written inside the parentheses.  Parentheses that hold nothing but one
number, currency symbols, a percent sign and white space are accounting
parentheses: one negative value, however the white space falls.  A text in
which a pair of parentheses holds a letter of any script (a unit, a
qualifier before or after the number, a date, a quarter, a footnote mark, a
currency code), and a pair of texts with such a text on either side, is read
exactly as the live release (857aed9) reads it: the readings, the numeric
dominance and the verdicts pinned below were taken by running the live
module, and the release task's differential run compares the two modules
over a generated corpus and the cached documents.

R-10: a currency code is one of the active ISO 4217 codes (or RMB, CNH),
before the parentheses or before or after a sign; any other upper-case word
is not a currency.

Synthetic fixtures only.  Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import re

import pytest

from phoena_translator.pdf.translate.numeric_guard import (
    _CURRENCY_CODES,
    _ISO_4217_CODES,
    _WITHDRAWN_ISO_4217_CODES,
    PDF_NUMERIC_ROW_REASON,
    pdf_numbers,
    pdf_numeric_guard_failure,
    pdf_text_is_numeric_dominant,
)

ROW_VALUES = ("1.2", "24.6", "48.9")

# (text, the live release's reading, its numeric-dominance verdict).
LIVE_READINGS = [
    # The reviewers' rows of round 3: a qualifier or unit before or after the
    # number, in either language, leaves the parentheses grouping.
    ("Tenor (5 yrs) 2.3 3.4", ["5", "2.3", "3.4"], True),
    ("期限（为期5年）2.3 3.4", ["5", "2.3", "3.4"], False),
    ("Revenue (2025 Q3) 1.2 3.4", ["2025", "1.2", "3.4"], True),
    ("收入（2025年第三季度）1.2 3.4", ["2025", "1.2", "3.4"], False),
    ("Revenue growth (5% YoY) 2.3 3.4", ["5%", "2.3", "3.4"], False),
    ("收入增长（同比5%）2.3 3.4", ["5%", "2.3", "3.4"], False),
    ("Spread (5 bps lower) 2.3 3.4", ["5", "2.3", "3.4"], False),
    ("利差（低5个基点）2.3 3.4", ["5", "2.3", "3.4"], False),
    ("Est. (2025E) 2.3 3.4", ["2025", "2.3", "3.4"], False),
    ("估计（预计2025年）2.3 3.4", ["2025", "2.3", "3.4"], False),
    ("Change (1.2 per-unit) 2.3 3.4 4.5", ["1.2", "2.3", "3.4", "4.5"], True),
    ("变化（1.2每单位）2.3 3.4 4.5", ["1.2", "2.3", "3.4", "4.5"], True),
    ("Window (2000-current) 2.3 3.4 4.5", ["2000", "2.3", "3.4", "4.5"], True),
    ("区间（2000年至今）2.3 3.4 4.5", ["2000", "2.3", "3.4", "4.5"], True),
    ("Rank (1st) 2.3 3.4 4.5", ["1", "2.3", "3.4", "4.5"], True),
    ("排名（第1）2.3 3.4 4.5", ["1", "2.3", "3.4", "4.5"], True),
    # Unit words of rounds 1 and 2, and a unit suffix spaced off: grouping.
    ("Loss (1.2 mln) 2.3 3.4", ["1.2", "2.3", "3.4"], True),
    ("亏损（1.2百万）2.3 3.4", ["1.2", "2.3", "3.4"], False),
    ("Net (1.2 bn) 2.3 3.4", ["1.2", "2.3", "3.4"], True),
    ("Margin (5 pct) 2.3 3.4", ["5", "2.3", "3.4"], True),
    ("Rate (5 фунт) 17 28", ["5", "17", "28"], True),
    # A unit suffix glued to the number: accounting, the suffix left after
    # the mark (so a next number glued to the bracket is not read).
    ("Net (1.2bn) 2.3 3.4", ["-1.2", "2.3", "3.4"], True),
    ("Net ($1.2m) 2.3 3.4", ["-1.2", "2.3", "3.4"], True),
    ("Net (1.2%pp) 2.3 3.4", ["-1.2%", "2.3", "3.4"], True),
    ("净额（1.2bn）17 28", ["-1.2", "28"], False),
    ("Note (12) (13 bn) 2.3", ["-12", "13", "2.3"], True),
    # A currency code or a letter-prefixed currency inside: grouping.
    ("Cost (US$1.2) 2.3 3.4", ["1.2", "2.3", "3.4"], False),
    ("Cost (EUR 1.2) 2.3 3.4", ["1.2", "2.3", "3.4"], True),
    ("Cost (RMB1.2) (RMB2.3) 3.4", ["3.4"], False),
    # Ranges, dates, footnote marks.
    ("Range (5-6%) 17 28", ["5", "6%", "17", "28"], True),
    ("区间（5亿至6亿）17 28", ["5", "6", "17", "28"], False),
    ("Item (1.2bn-3.4bn) 17", ["1.2", "3.4", "17"], False),
    ("Date (5-Nov-26) 2.3 3.4", ["D26-11-5", "2.3", "3.4"], True),
    ("Note (a) 2.3 3.4 4.5", ["2.3", "3.4", "4.5"], True),
    # Outside the lettered parentheses the text is read as live too.
    ("Net (5 yrs) -$ 1.2 \u20122.3 3.4", ["5", "1.2", "2.3", "3.4"], False),
    # The only letter a superscript, Latin or CJK, either width (R-9e).
    ("Net (5<sup>a</sup>) \u20142.3 3.4", ["-5", "2.3", "3.4"], False),
    ("Net （5<sup>注</sup>） \u20142.3 3.4", ["-5", "2.3", "3.4"], False),
    ("Net (<sup>b</sup>1.2) \u20142.3 3.4", ["-1.2", "2.3", "3.4"], False),
]


@pytest.mark.parametrize(("text", "numbers", "dominant"), LIVE_READINGS)
def test_a_text_with_lettered_parentheses_reads_as_the_live_release(text, numbers, dominant):
    assert pdf_numbers(text) == numbers
    assert pdf_text_is_numeric_dominant(text) is dominant


# (source, translation, the live release's verdict): the reviewers' pairs
# of all three rounds.
LIVE_VERDICTS = [
    ("Tenor (5 yrs) 2.3 3.4", "期限（为期5年）2.3 3.4", None),
    ("Revenue (2025 Q3) 1.2 3.4", "收入（2025年第三季度）1.2 3.4", None),
    ("Revenue (2025 H1) 1.2 3.4", "收入（2025年上半年）1.2 3.4", None),
    ("Revenue growth (5% YoY) 2.3 3.4", "收入增长（同比5%）2.3 3.4", None),
    ("Spread (5 bps lower) 2.3 3.4", "利差（低5个基点）2.3 3.4", None),
    ("Est. (2025E) 2.3 3.4", "估计（预计2025年）2.3 3.4", None),
    ("Change (1.2 per-unit) 2.3 3.4 4.5", "变化（1.2每单位）2.3 3.4 4.5", None),
    ("Change (1.2 m2) 2.3 3.4 4.5", "变化（1.2平方米）2.3 3.4 4.5", None),
    ("Window (2000-current) 2.3 3.4 4.5", "区间（2000年至今）2.3 3.4 4.5", None),
    ("Rank (1st) 2.3 3.4 4.5", "排名（第1）2.3 3.4 4.5", None),
    ("Utilities (63 bps) 17 28", "公用事业（63个基点）17 28", None),
    ("Loss (1.2 mln) 2.3 3.4", "亏损（1.2百万）2.3 3.4", None),
    ("Margin (5 pct) 2.3 3.4", "利率5% 2.3 3.4", None),
    # A changed unit: the percent sign dropped, as live.
    ("Loss (5%) 17 28", "亏损（5‰）17 28", PDF_NUMERIC_ROW_REASON),
    ("Loss (5%) 17 28", "亏损（5元）17 28", PDF_NUMERIC_ROW_REASON),
    # A glued unit suffix and a translation that drops its parentheses.
    ("Net (1.2bn) 2.3 3.4", "净额 1.2bn 2.3 3.4", PDF_NUMERIC_ROW_REASON),
    ("Interest cover (2.5x) 1 2 3", "利息保障倍数（2.5倍）1 2 3", PDF_NUMERIC_ROW_REASON),
    ("Net result (CZK 1.2) 2.3 3.4", "净结果1.2 2.3 3.4", None),
    # The second reader's round-4 F3: the only letter a superscript (R-9e).
    ("Net (5<sup>a</sup>) \u20142.3 3.4", "净额 (5<sup>a</sup>) 2.3 3.4", None),
    ("净额 (5<sup>a</sup>) 2.3 3.4", "Net (5<sup>a</sup>) \u20142.3 3.4", None),
]


@pytest.mark.parametrize(("source", "translated", "verdict"), LIVE_VERDICTS)
def test_a_pair_with_lettered_parentheses_gets_the_live_verdict(source, translated, verdict):
    assert pdf_numeric_guard_failure(source, translated) == verdict


def test_whether_parentheses_hold_a_letter_is_decided_before_superscripts_are_removed():
    # A superscript letter selects the live reading; a superscript digit
    # holds no letter, so the parentheses do not.
    assert pdf_numbers("Net (5<sup>a</sup>) \u20142.3 3.4") == ["-5", "2.3", "3.4"]
    assert pdf_numbers("Net (5<sup>1</sup>) \u20142.3 3.4") == ["-5", "-2.3", "3.4"]


@pytest.mark.parametrize(("source", "translated", "_verdict"), LIVE_VERDICTS[:12])
def test_the_reviewers_correct_pairs_pass_the_other_way_too(source, translated, _verdict):
    assert pdf_numeric_guard_failure(translated, source) is None


# The one extension that stays for parentheses: white space inside
# parentheses that hold one number, currency symbols and a percent sign.
WHITE_SPACE_SPELLINGS = [
    ("($ {v})", ""), ("( $ {v} )", ""), ("$ ({v})", ""), ("（ {v} ）", ""),
    ("（$ {v}）", ""), ("( \t{v}\t)", ""), ("({v} %)", "%"), ("( {v} % )", "%"),
]


@pytest.mark.parametrize(("spelling", "percent"), WHITE_SPACE_SPELLINGS)
def test_white_space_does_not_prevent_the_accounting_reading(spelling, percent):
    row = " ".join(["Net profit", *(spelling.format(v=value) for value in ROW_VALUES)])
    assert pdf_numbers(row) == [f"-{value}{percent}" for value in ROW_VALUES]
    assert pdf_text_is_numeric_dominant(row)
    flipped = "净利润 " + " ".join(f"{value}{percent}" for value in ROW_VALUES)
    assert pdf_numeric_guard_failure(row, flipped) == PDF_NUMERIC_ROW_REASON
    kept = "净利润 " + " ".join(f"-{value}{percent}" for value in ROW_VALUES)
    assert pdf_numeric_guard_failure(row, kept) is None


@pytest.mark.parametrize(
    "spelling", ["($ {v}bn)", "({v} bn)", "({v} mn)", "(US${v})", "(EUR {v})", "({v} \u2030)"]
)
def test_a_letter_or_another_sign_inside_leaves_the_parentheses_grouping(spelling):
    row = " ".join(["Net profit", *(spelling.format(v=value) for value in ROW_VALUES)])
    assert pdf_numbers(row) == list(ROW_VALUES)


# ---------------------------------------------------------------------------
# R-10: the currency codes are a closed list
# ---------------------------------------------------------------------------

# Outside the parentheses only: a code inside them is a letter (R-9d).
CODE_SPELLINGS = ["{c} ({v})", "-{c} {v}", "{c} -{v}", "{c}-{v}"]


def test_the_codes_are_the_active_iso_list_plus_the_yuan_market_names():
    assert len(set(_ISO_4217_CODES)) == len(_ISO_4217_CODES) == 175
    assert all(re.fullmatch("[A-Z]{3}", code) for code in _ISO_4217_CODES)
    assert _ISO_4217_CODES == sorted(_ISO_4217_CODES)
    for code in ("CZK", "PLN", "HUF", "ILS", "NGN", "XCG", "ZWG", "SLE", "VED", "XAU", "XDR"):
        assert code in _ISO_4217_CODES, code
    # Testing and no-currency codes, and codes withdrawn before 2021, are not in it.
    for code in ("XTS", "XXX", "VEF", "MRO", "STD", "LTL"):
        assert code not in _CURRENCY_CODES, code
    # Codes withdrawn since 2021 stay currencies beside their successors.
    for old, new in (("HRK", "EUR"), ("BGN", "EUR"), ("SLL", "SLE"), ("ZWL", "ZWG"), ("ANG", "XCG"), ("CUC", "CUP")):
        assert old in _WITHDRAWN_ISO_4217_CODES and old not in _ISO_4217_CODES, old
        assert old in _CURRENCY_CODES and new in _CURRENCY_CODES, (old, new)
    assert set(_CURRENCY_CODES) - set(_ISO_4217_CODES) == {*_WITHDRAWN_ISO_4217_CODES, "RMB", "CNH"}


@pytest.mark.parametrize("code", sorted(_CURRENCY_CODES))
def test_every_listed_code_carries_a_negative_and_its_flip_is_caught(code):
    for spelling in CODE_SPELLINGS:
        row = " ".join(["Net result", *(spelling.format(c=code, v=value) for value in ROW_VALUES)])
        assert pdf_numbers(row) == [f"-{value}" for value in ROW_VALUES], row
        assert pdf_text_is_numeric_dominant(row), row
        flipped = "净结果 " + " ".join(f"{code} {value}" for value in ROW_VALUES)
        assert pdf_numeric_guard_failure(row, flipped) == PDF_NUMERIC_ROW_REASON, row
        kept = "净结果 " + " ".join(f"-{value}" for value in ROW_VALUES)
        assert pdf_numeric_guard_failure(row, kept) is None, row


def test_the_reviewers_unlisted_code_carries_its_negative_outside_the_parentheses():
    source = "Net result CZK -1.2 2.3 3.4"
    assert pdf_numbers(source) == ["-1.2", "2.3", "3.4"]
    assert pdf_numeric_guard_failure(source, "净结果1.2 2.3 3.4") == PDF_NUMERIC_ROW_REASON
    assert pdf_numeric_guard_failure(source, "净结果-1.2 2.3 3.4") is None


@pytest.mark.parametrize("code", ["CZK", "EUR", "RMB", "USD"])
def test_a_code_inside_the_parentheses_reads_as_the_live_release(code):
    # The live release knows no code: the parentheses stay grouping, and a
    # code glued to the number hides it.
    assert pdf_numbers(f"Net result ({code} 1.2) 2.3 3.4") == ["1.2", "2.3", "3.4"]
    assert pdf_numbers(f"Net result ({code}1.2) 2.3 3.4") == ["2.3", "3.4"]


@pytest.mark.parametrize(
    "word", ["XXX", "XTS", "ABC", "NEW", "EURO", "CZKX", "czk", "Usd", "VEF", "hrk", "FY"]
)
def test_any_other_word_is_not_a_currency(word):
    # Inside the parentheses it leaves them grouping: the number stays unsigned.
    assert pdf_numbers(f"Note ({word} 1.2) ({word} 2.3)") == ["1.2", "2.3"]
    row = f"Net result ({word} 1.2) ({word} 2.3) ({word} 3.4)"
    assert pdf_numeric_guard_failure(row, "净结果 1.2 2.3 3.4") is None
