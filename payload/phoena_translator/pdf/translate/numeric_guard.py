"""A table row keeps its numbers; a text label does not acquire numbers.

Two element-level conditions, decided from an element's own source text and
its candidate translation and nothing else:

* N, ``numeric-row-numbers-changed``: a numeric-dominant source (most of its
  tokens are data tokens, and it holds at least three numbers) must come
  back with exactly the same sequence of numbers.  A model comment in place
  of the row, a dropped value or a row shifted in from a neighbour fails.
* M, ``label-acquired-numbers``: a source without any digit must not come
  back with three or more numbers (a neighbour's row moved into a label).

Prose is never numeric-dominant, so unit conversions in running text
(``$67bn`` -> ``670亿美元``) are untouched; condition M needs a source with
no digit at all.  A translation that fails either condition is not drawn:
translation validation keeps the element as exact source text, assembly keeps
an older cached one in its source ink.  A translation whose plain text is
the element's own plain source text is the element kept in source, never a
candidate of either condition.

Limits of the rule as defined:

* A correct translation of a numeric row that renders a label's integer as a
  word (``Tier 1`` -> ``一级``) or converts units inside the row (``$12.5bn``
  -> ``125亿美元``) changes the sequence and is rejected; the row stays in
  English with its numbers intact.
* A comment that repeats every number of the row in order passes N.
* A row whose label is long enough that data tokens are not the majority, or
  that holds fewer than three numbers, is not numeric-dominant.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Mapping

from phoena_translator.pdf.text_shape import _collapsed_plain_text, _plain_text

PDF_NUMERIC_ROW_REASON = "numeric-row-numbers-changed"
PDF_LABEL_NUMBERS_REASON = "label-acquired-numbers"

# Condition N: at least this many numbers, and data tokens are strictly more
# than half of the counted tokens.
NUMERIC_ROW_MIN_NUMBERS = 3
# Condition M: a digit-free source whose translation holds this many numbers.
LABEL_ACQUIRED_MIN_NUMBERS = 3

# Footnote and reference markers live in superscript markup; their integrity
# is checked by the superscript gate, so they are not part of the sequence.
_SUPERSCRIPT_RE = re.compile(r"(?is)<sup(?:\s[^>]*)?>.*?</sup>")

# Dates are one value each, compared by (year as written, month, day) so that
# a localized date is the same value: ``5-Nov-26`` == ``2026年11月5日``,
# ``Nov-26`` == ``2026年11月``, ``10/11/2026`` == ``2026年10月11日``.  A
# numeric date whose first field is above 12 is day-first (``25/10/2026`` ==
# ``2026年10月25日``); a hyphen or dot date is one only when its order is
# unambiguous (``14-11-2026``, ``11.14.2026``).
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"), ("feb", "february"), ("mar", "march"),
            ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
            ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
            ("nov", "november"), ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_MON = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<![\d.])(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})(?![\d.])",
        r"(?P<y>\d{2,4})年(?P<m>\d{1,2})月(?:(?P<d>\d{1,2})日)?",
        r"(?<![\d年])(?P<m>\d{1,2})月(?P<d>\d{1,2})日",
        r"(?<![\d./])(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4}|\d{2})(?![\d./])",
        r"(?<![\d./-])(?P<a>\d{1,2})(?P<sep>[-.])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4})(?![\d./-])",
        r"(?<![\d./])(?P<m>\d{1,2})/(?P<y>\d{4})(?![\d./])",
        # ``15-Jun``, ``5-Nov-26``, ``11 Nov 2026``; a year never takes the
        # next cell's day (``5-Jul 25-Jul``).
        r"(?<![\w./-])(?P<d>\d{1,2})[- ]" + _MON
        + r"(?:[- ,]+['’]?(?P<y>\d{4}|\d{2})(?![\w./-]))?(?![\w])",
        r"(?<![\w])" + _MON + r"\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})(?!\d)",
        # ``Nov-26``, ``Nov '26``, ``Nov 2026``; ``Jun 15`` is a day.
        r"(?<![\w])" + _MON + r"(?:-|\s+['’]|\s+(?=\d{4}))(?P<y>\d{4}|\d{2})(?![\w./-])",
        r"(?<![\w])" + _MON + r"\s+(?P<d>\d{1,2})(?![\w./,-])",
    )
]
_DATE_MARK = "\u27e6D{}\u27e7"
# A two-digit year after an apostrophe (``'36``) is the year 1936 or 2036.
_SHORT_YEAR_RE = re.compile(r"(?<![\w])['’‘](\d{2})(?![\d.,%])")
_SHORT_YEAR_MARK = "\u27e6Y{}\u27e7"
# A currency written before a number: a symbol, a symbol with a letter prefix
# (``US$`` ``HK$`` ``A$``), or an upper-case currency code from a closed list
# (``EUR`` ``CZK`` ``RMB``).  It is matched case-sensitively wherever it is
# used; any other upper-case word is not a currency.
#
# The codes: the active ISO 4217 codes (currencies, funds, precious metals,
# bond-market and settlement units; the testing code XTS and the no-currency
# code XXX are not currencies), the codes withdrawn since 2021 (a document
# written before a currency changed still uses its old code; a superset is
# harmless), plus the market names of the Chinese yuan, RMB and CNH.
_ISO_4217_CODES = """
    AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND BOB BOV
    BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP
    CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF
    GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR
    KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT
    MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN
    PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE
    SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX
    USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU XBA XBB XBC XBD
    XCD XCG XDR XOF XPD XPF XPT XSU XUA YER ZAR ZMW ZWG
""".split()
# Withdrawn since 2021: CUC (2021), HRK (2023), SLL and ZWL (2024), ANG (2025),
# BGN (2026); their successors (SLE, ZWG, XCG, EUR) are in the active list.
_WITHDRAWN_ISO_4217_CODES = ("ANG", "BGN", "CUC", "HRK", "SLL", "ZWL")
_CURRENCY_CODES = (*_ISO_4217_CODES, *_WITHDRAWN_ISO_4217_CODES, "RMB", "CNH")
_CURRENCY = r"(?-i:[A-Z]{0,3}[$€£¥]|(?:" + "|".join(_CURRENCY_CODES) + r")(?![A-Za-z]))"
# A currency and the spaces or tabs a writer may put after it (``$ 1.2``).
_SPACED_CURRENCY = rf"(?:{_CURRENCY}[ \t]*)"
# A number: an integer part with optional thousands separators and an
# optional decimal part, or a decimal part alone (``.1`` is ``0.1``).
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+"
_UNIT_SUFFIX = r"(?:x|bps?|pp|bn|mn|[mkb])"
# A sign written right against its number or currency: hyphen-minus, plus,
# minus sign, figure dash, en dash, em dash.
_SIGN = "[-+\u2212\u2012\u2013\u2014]"
# The one sign grammar.  A signed value, in every common spelling, is read
# once into a value mark ``⟦V<value>⟧`` that value extraction and data-token
# classification both see:
#
# * accounting parentheses (either width) that hold nothing but one number,
#   currency symbols, a percent sign and white space, with or without a
#   currency before them: ``(1.2)`` ``( 1.2 )`` ``(1.2%)`` ``(1.2 %)``
#   ``($1.2)`` ``($ 1.2)`` ``$(1.2)`` ``$ (1.2)`` ``EUR (1.2)`` are all
#   ``-1.2`` (``-1.2%`` with the percent sign);
# * a sign before or after a currency: ``-1.2`` ``−1.2`` ``–1.2`` ``‒1.2``
#   ``—1.2`` ``-$1.2`` ``-$ 1.2`` ``-US$1.2`` ``-EUR 1.2`` ``$-1.2``
#   ``US$-1.2`` (a plus sign leaves the value positive);
# * a hyphen or minus sign spaced off its number by spaces or tabs:
#   ``- 1.2`` ``-  1.2`` ``- $1.2``.
#
# A sign never follows a letter, digit or decimal point (a dash between
# digits is a separator: ``2025-26``), and a spaced en or em dash is a range
# (``1.5 – 2.5``).  Parentheses around several values, or around anything
# else, are grouping: their numbers stay unsigned.  A text with parentheses
# that hold a letter is read by the live release's grammar instead
# (``_LIVE_GRAMMAR``).
_SIGNED_VALUE_RE = re.compile(
    rf"(?<![0-9.]){_SPACED_CURRENCY}?\(\s*(?:[$€£¥][ \t]*)?(?P<acct>{_NUMBER})"
    rf"(?:[ \t]*(?P<acct_pct>%))?\s*\)"
    rf"|(?<![A-Za-z0-9.]){_SPACED_CURRENCY}?(?P<sign>{_SIGN}|[-\u2212][ \t]+)"
    rf"{_SPACED_CURRENCY}?(?P<num>{_NUMBER})(?P<pct>%?)"
)
_VALUE_MARK = "\u27e6V{}\u27e7"
# One value: a date, short-year or signed-value mark, or an unsigned number
# with an optional currency symbol and percent sign.  A number never starts
# inside a word, a code (``Baa2``; ``4NC3`` yields only its leading ``4``) or
# after a decimal point; letters may follow (``25bp``, ``12wk``).
_VALUE_RE = re.compile(
    r"\u27e6(?P<mark>[DYV][^\u27e7]*)\u27e7"
    rf"|(?<![A-Za-z0-9.]){_CURRENCY}?(?P<num>{_NUMBER})(?P<pct>%?)"
)
# One operand of a range or ratio: a signed-value mark or a number, with an
# optional currency, percent sign and short unit suffix.
_OPERAND = (
    rf"(?:\u27e6V[^\u27e7]*\u27e7|{_CURRENCY}?(?:\d|\.\d)[\d,.]*)%?{_UNIT_SUFFIX}?"
)
# A data token, after enclosing brackets and trailing punctuation are
# stripped: a date mark, or a number (or signed-value mark), range or ratio
# with an optional currency sign, percent sign or short unit suffix.  Every
# operand of a range or ratio may be a signed value, so a sign never decides
# whether a token is data: ``1.2/-2.3`` ``-1.2%/-2.3%`` ``$1.2/-$2.3`` are data
# as ``1.2/2.3`` ``1.2%/2.3%`` ``$1.2/$2.3`` are.
_DATA_TOKEN_RE = re.compile(
    r"\u27e6[DY][^\u27e7]*\u27e7"
    rf"|(?:\u27e6V[^\u27e7]*\u27e7|{_CURRENCY}?\.?\d)[\d,./:\-\u2013]*%?{_UNIT_SUFFIX}?"
    rf"|{_OPERAND}(?:[/:\-\u2013]{_OPERAND})+",
    re.IGNORECASE,
)
# Parentheses that hold a letter (R-9d).  The sign of a parenthesised number
# must not depend on what else is written inside the parentheses, and the
# two languages write the same unit or qualifier in different shapes and
# places (``(5 yrs)`` against ``（为期5年）``, ``(2025 Q3)`` against
# ``（2025年第三季度）``): no reading of the letters gives both sides the same
# sign.  So a text in which a pair of parentheses holds a letter of any
# script, and a pair of texts with such a text on either side, is read
# exactly as the live release reads it, by its grammar copied unchanged:
# currency symbols only, four sign characters, accounting parentheses
# around one number with a percent sign or a unit suffix glued to it
# (``(1.2bn)`` is ``-1.2`` with ``bn`` left after the mark), no signed
# operand in a range or ratio.
_LIVE_CURRENCY = "[$€£¥]"
_LIVE_GRAMMAR = (
    re.compile(
        rf"(?<![0-9.]){_LIVE_CURRENCY}?\(\s*{_LIVE_CURRENCY}?(?P<acct>{_NUMBER})(?P<acct_pct>%?)"
        rf"(?P<acct_unit>(?i:{_UNIT_SUFFIX})?)\s*\)"
        rf"|(?<![A-Za-z0-9.]){_LIVE_CURRENCY}?(?P<sign>[-+\u2212\u2013]|[-\u2212] ){_LIVE_CURRENCY}?"
        rf"(?P<num>{_NUMBER})(?P<pct>%?)"
    ),
    re.compile(
        r"\u27e6(?P<mark>[DYV][^\u27e7]*)\u27e7"
        rf"|(?<![A-Za-z0-9.]){_LIVE_CURRENCY}?(?P<num>{_NUMBER})(?P<pct>%?)"
    ),
    re.compile(
        r"\u27e6[DY][^\u27e7]*\u27e7"
        rf"|(?:\u27e6V[^\u27e7]*\u27e7|{_LIVE_CURRENCY}?\.?\d)[\d,./:\-\u2013]*%?{_UNIT_SUFFIX}?",
        re.IGNORECASE,
    ),
)
# The sign grammar, value pattern and data-token pattern, by whether the
# text (or pair) has parentheses that hold a letter.
_GRAMMARS = {False: (_SIGNED_VALUE_RE, _VALUE_RE, _DATA_TOKEN_RE), True: _LIVE_GRAMMAR}
# The usual placeholders for "no data" in a table cell.
_NO_DATA_TOKEN_RE = re.compile(
    r"n\.?a\.?|n/a|n\.?m\.?|n/m|nmf|-{1,3}|[\u2013\u2014]",
    re.IGNORECASE,
)
_TOKEN_EDGE_CHARS = "()[]{}*†‡,;:'\"“”‘’"
_ALNUM_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff\u27e6]")
# Cardinal number words, which a translator legitimately renders as digits
# (``seventy-seven hundred`` -> ``7700``).  Condition M exempts at most as many
# acquired numbers as a digit-free source holds such words; ordinals,
# fractions, month names and modals are ordinary vocabulary and exempt none.
_SPELLED_NUMBER_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    r"|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
    r"|million|billion|trillion|dozen)s?\b",
    re.IGNORECASE,
)


def _date_mark(match: re.Match) -> str:
    groups = match.groupdict()
    if groups.get("a"):
        first, second = int(groups["a"]), int(groups["b"])
        if first > 12 >= second:
            month, day = second, str(first)  # day-first, unambiguous
        elif groups.get("sep") and not second > 12 >= first:
            return match.group(0)  # an ambiguous hyphen or dot date
        else:
            month, day = first, str(second)
    else:
        month = groups.get("m")
        month = int(month) if month else _MONTHS[groups["mon"].lower()]
        day = groups.get("d")
    year = groups.get("y")
    if not 1 <= month <= 12 or (day and not 1 <= int(day) <= 31):
        return match.group(0)
    # A full year keeps its century; a short year stays short.
    year = year.lstrip("0").rjust(2, "0") if year else ""
    return _DATE_MARK.format(f"{year}-{month}-{int(day) if day else ''}")


def _canonical_value(number: str, negative: bool, percent: str) -> str:
    integer, dot, decimal = number.partition(".")
    integer = integer.replace(",", "").lstrip("0") or "0"
    return f"{'-' if negative else ''}{integer}{dot}{decimal}{percent}"


def _signed_value_mark(match: re.Match) -> str:
    if match.group("acct") is not None:
        value = _canonical_value(match.group("acct"), True, match.group("acct_pct") or "")
        # The live grammar leaves a unit suffix after the mark.
        return _VALUE_MARK.format(value) + (match.groupdict().get("acct_unit") or "")
    negative = match.group("sign").strip() != "+"
    return _VALUE_MARK.format(_canonical_value(match.group("num"), negative, match.group("pct")))


def _folded(text: str) -> str:
    """Plain text without superscripts, full-width forms folded."""
    without_superscripts = _SUPERSCRIPT_RE.sub(" ", text or "")
    return unicodedata.normalize("NFKC", _plain_text(without_superscripts))


def _lettered_parentheses(text: str) -> bool:
    """Whether a pair of parentheses (either width) in ``text`` holds a
    letter of any script (R-9d), decided on the text as written: before
    superscript content is removed (R-9e)."""
    opened: list[bool] = []
    for char in unicodedata.normalize("NFKC", _plain_text(text or "")):
        if char == "(":
            opened.append(False)
        elif char == ")" and opened:
            if opened.pop():
                return True
        elif opened and unicodedata.category(char).startswith("L"):
            opened[-1] = True
    return False


def _numeric_view(text: str, live: bool = False) -> str:
    """Plain text without superscripts, full-width forms folded, dates,
    apostrophe years and signed values marked as single values."""
    view = _folded(text)
    for pattern in _DATE_PATTERNS:
        view = pattern.sub(_date_mark, view)
    view = _SHORT_YEAR_RE.sub(lambda m: _SHORT_YEAR_MARK.format(m.group(1)), view)
    return _GRAMMARS[live][0].sub(_signed_value_mark, view)


def pdf_numbers(text: str) -> list[str]:
    """The values of ``text`` in order, each in one canonical spelling.

    A number drops thousands separators, leading zeros of its integer part, a
    plus sign and a currency symbol; a minus sign (every spelling the sign
    grammar ``_SIGNED_VALUE_RE`` reads), the decimal part as written and a
    percent sign are kept, so ``1,250`` == ``1250``, ``02`` == ``2``, ``.5`` ==
    ``0.5``, ``( 1.2 )`` == ``$(1.2)`` == ``- 1.2`` == ``-1.2`` but ``0.30`` != ``0.3``,
    ``-2`` != ``2`` and ``36%`` is ``36%``.  A date is ``D<year>-<m>-<d>``
    with the year as written, an apostrophe year ``Y<yy>``.  A text in which
    a pair of parentheses holds a letter is read as the live release reads
    it (``_LIVE_GRAMMAR``).
    """
    return _numbers(text, _lettered_parentheses(text))


def _numbers(text: str, live: bool) -> list[str]:
    values = []
    for match in _GRAMMARS[live][1].finditer(_numeric_view(text, live)):
        mark = match.group("mark")
        if mark:
            values.append(mark[1:] if mark.startswith("V") else mark)
            continue
        values.append(_canonical_value(match.group("num"), False, match.group("pct")))
    return values


def _same_value(source: str, translated: str) -> bool:
    if source == translated:
        return True
    if source.startswith("D") and translated.startswith("D"):
        # A short year matches the full year it abbreviates; two full years
        # must be equal (they are not, or the marks would be).
        source_year, _, source_rest = source[1:].partition("-")
        translated_year, _, translated_rest = translated[1:].partition("-")
        return (
            source_rest == translated_rest
            and bool(source_year) == bool(translated_year)
            and min(len(source_year), len(translated_year)) == 2
            and source_year[-2:] == translated_year[-2:]
        )
    # A percent sign may be added, never dropped.
    if translated == f"{source}%":
        return True
    # ``'36`` may come back as ``2036`` (or ``1936``).
    return (
        source.startswith("Y")
        and translated.rstrip("%") in {source[1:], f"19{source[1:]}", f"20{source[1:]}"}
    )


def _same_numbers(source_text: str, translated_text: str, live: bool) -> bool:
    source = _numbers(source_text, live)
    translated = _numbers(translated_text, live)
    return len(source) == len(translated) and all(
        _same_value(s, t) for s, t in zip(source, translated)
    )


def _counted_tokens(view: str) -> list[str]:
    tokens = []
    for raw in view.split():
        token = raw.strip(_TOKEN_EDGE_CHARS)
        if _NO_DATA_TOKEN_RE.fullmatch(token):
            tokens.append(token)
        elif token and _ALNUM_RE.search(token):
            tokens.append(token)
        # A token of punctuation only (``*``, ``|``) is neither data nor text.
    return tokens


def _is_data_token(token: str, live: bool = False) -> bool:
    return bool(_GRAMMARS[live][2].fullmatch(token) or _NO_DATA_TOKEN_RE.fullmatch(token))


def _numeric_dominant(source_text: str, live: bool) -> bool:
    view = _numeric_view(source_text, live)
    if len(_numbers(source_text, live)) < NUMERIC_ROW_MIN_NUMBERS:
        return False
    tokens = _counted_tokens(view)
    data = sum(1 for token in tokens if _is_data_token(token, live))
    return 2 * data > len(tokens)


def pdf_text_is_numeric_dominant(source_text: str) -> bool:
    """Whether most tokens of ``source_text`` are data and it holds >= 3 numbers."""
    return _numeric_dominant(source_text, _lettered_parentheses(source_text))


def pdf_numeric_guard_failure(source_text: str, translated_text: str) -> str | None:
    """The condition a candidate translation fails, or ``None``.

    N: a numeric-dominant source must keep exactly its sequence of numbers.
    M: a source without any digit must not acquire three or more numbers
    beyond one per cardinal number word it holds.
    """
    if _plain_text(translated_text or "") == _plain_text(source_text or ""):
        # The element kept in source (its plain text where the source holds
        # inline markup): not a translation of the row.
        return None
    # Parentheses that hold a letter on either side: the live release's reading.
    live = _lettered_parentheses(source_text) or _lettered_parentheses(translated_text)
    if _numeric_dominant(source_text, live):
        if not _same_numbers(source_text, translated_text, live):
            return PDF_NUMERIC_ROW_REASON
        return None
    source_view = _numeric_view(source_text, live)
    if (
        not re.search(r"\d", source_view)
        and len(_numbers(translated_text, live))
        - len(_SPELLED_NUMBER_RE.findall(source_view))
        >= LABEL_ACQUIRED_MIN_NUMBERS
    ):
        return PDF_LABEL_NUMBERS_REASON
    return None


# --- a protected sign still has a figure near it (condition S) ---
#
# The one measured way running text loses an amount: extraction protects a
# sign-like glyph of the text (``+`` ``~`` ``−`` ``–``) as an inline-math
# fragment, its placeholder touches the amount (``+$7.3bn``, ``$250+bn``),
# the model keeps the placeholder and drops the amount, and the restore puts
# the sign back alone (``为+，``, ``+亿美元``, ``+bn``).
#
# S, ``prose-number-missing``, is a proximity check: in an element that is
# not numeric-dominant (rows are condition N's), each protected fragment made
# only of sign-like characters (``PDF_SIGN_CHARACTERS``) that touches an
# amount in the source (a digit or a currency symbol right after it, or a
# digit right before it) must still have a figure near it in the
# translation: right before it (a magnitude word such as ``亿`` may stand
# between, ``2500亿+美元``), or within ``PDF_SIGN_FIGURE_DISTANCE`` characters
# after it with no punctuation mark other than an opening bracket and no
# line break between.  It does not establish that the amount survived: any
# figure near the sign satisfies it, and occurrences are counted per
# fragment text, so the order of the clauses may change.  Nothing else is
# compared: no values, no units, no neighbours.
PDF_PROSE_NUMBERS_REASON = "prose-number-missing"
# The sign-like fragments the inline-math detection produces
# (``_pdf_inline_math_span_fragments``): plus, minus, plus-minus, tilde,
# approximately, less and greater than in their ASCII, mathematical,
# full-width, small, superscript and subscript forms (category Sm, the
# Unicode path), and the hyphen-minus and dash spellings a math font's span
# protects as they are (the strong-font path): every character whose
# compatibility form is ``-`` ``+`` ``−`` ``‒`` ``–`` ``—`` ``~`` ``<`` ``>``,
# and the hyphens ``‐`` ``‑`` and comparisons ``≲`` ``≳``.
PDF_SIGN_CHARACTERS = frozenset(
    "+＋﹢−∓±~∼～≈≃≅≒<>≤≥≦≧⩽⩾＜＞≪≫"
    "-\ufe63\uff0d\u2012\u2013\u2014\ufe31\ufe32\ufe58"
    "\u207a\u207b\u208a\u208b\ufb29\ufe64\ufe65"
    "\u2010\u2011\u2272\u2273"
)
# The figure is one of this many characters after the sign, and no
# punctuation mark (Unicode category P) other than an opening bracket and no
# line break stands between them: room for a currency or approximation
# phrase, a space and a bracket (``+约人民币50亿元``, ``+ US$5bn``,
# ``+（50亿美元）``); none for a sign left before punctuation and a figure
# of the next clause (``+，为9``), a magnitude word or a unit alone, or the
# end of the text.  Measured on the cached translations: from 2 to 12
# characters the check rejects the same nine elements.
PDF_SIGN_FIGURE_DISTANCE = 8
# A figure (R-18c): a character to which Unicode gives a numeric value, read
# in its compatibility form (digits of any width or script, the Chinese
# numerals in their ordinary, financial and variant forms, 廿 卅 卌), except:
# * a magnitude word, an ideograph worth ten thousand or more (万 亿 萬 億 兆),
#   is never a figure on its own: a sign followed by ``亿美元`` is the
#   measured loss; it may stand between a figure and the sign after it;
# * an ideograph worth a hundred or a thousand (百 千 佰 仟) is a figure
#   (``百个基点`` ``百美元`` ``三百`` ``千万``) except as the first character
#   of a unit or percentage word (R-18d): a hundred before 分 (``百分点``
#   ``百分比`` ``百分之``), a thousand before a metric unit (``千瓦时``
#   ``千克`` ``千米``), so a sign followed by such a word alone stays a loss;
# * an everyday numeral, a character with a numeric value that is also an
#   ordinary character of common words (``参与`` ``什么`` ``大陆`` ``队伍``
#   ``收拾`` ``拐点`` ``漏洞`` ``挂钩`` ``陌生``), is a figure only next to
#   another figure or a magnitude word (R-18e: ``陆亿`` ``伍佰`` ``陆拾亿``),
#   another everyday numeral included (R-18g: ``伍拾`` ``肆拾伍``, and so
#   ``阡陌`` ``陆陆续续``); alone in a word (``大陆``) it is none;
# * 半, which has no numeric value, is a figure wherever it stands (R-18f:
#   ``半个`` ``半厘`` ``半bp`` ``三半``, and ``半导体`` too); it is no company
#   for an everyday numeral.
# R-18d: the second character of the unit or percentage words a hundred or
# a thousand begins (the percent words; watt, gram, metre, calorie, joule,
# hertz, volt, ampere, tonne in either script, litre).
_UNIT_WORD_SECOND = {
    "hundred": frozenset("分"),
    "thousand": frozenset("瓦克米卡焦赫伏安吨噸升"),
}
# R-18e: of the ideographs with a numeric value below ten thousand, those
# among the common characters (GB 2312 level 1 or Big5 level 1) whose common
# words use them for something other than a quantity, in both scripts, and
# 鈎, the other traditional spelling of 钩.
_EVERYDAY_NUMERALS = frozenset("参參什伍拾肆陆陸拐洞漆钩鈎陌阡")
# An opening bracket of either width does not stop the search.
_OPENING_BRACKETS = frozenset("(（")
# The characters ``str.splitlines`` breaks on.
_LINE_BREAKS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")
# The audit list of translations drawn although they fail S (assembly only).
PDF_DELIVERED_ELEMENT_WARNINGS = "delivered_element_warnings"


def _figure_kind(char: str) -> str | None:
    """``"figure"``, ``"magnitude"``, ``"hundred"``, ``"thousand"``,
    ``"everyday"``, ``"half"`` or ``None``."""
    if char == "半":
        return "half"
    form = unicodedata.normalize("NFKC", char)
    value = unicodedata.numeric(form, None) if len(form) == 1 else None
    if value is None and len(char) == 1:
        value = unicodedata.numeric(char, None)
    if value is None:
        return None
    ideograph = unicodedata.category(form[:1]) == "Lo"
    if ideograph and value >= 10_000:
        return "magnitude"
    if form in _EVERYDAY_NUMERALS:
        return "everyday"
    if ideograph and value == 100:
        return "hundred"
    if ideograph and value == 1000:
        return "thousand"
    return "figure"


def _is_figure(text: str, index: int) -> bool:
    """Whether ``text[index]`` counts as a figure where it stands."""
    kind = _figure_kind(text[index])
    if kind in (None, "figure", "magnitude", "half"):
        return kind in ("figure", "half")
    if kind == "everyday":
        return _has_company(text, index)
    after = text[index + 1:index + 2]
    return unicodedata.normalize("NFKC", after)[:1] not in _UNIT_WORD_SECOND[kind]


def _has_company(text: str, index: int) -> bool:
    """Whether the everyday numeral ``text[index]`` stands next to another
    figure or a magnitude word: the character beside it on either side is
    a figure, another everyday numeral (R-18g), a magnitude word, or a
    hundred or a thousand that is a figure where it stands."""
    for at in (index - 1, index + 1):
        if not 0 <= at < len(text):
            continue
        kind = _figure_kind(text[at])
        if kind in ("figure", "everyday", "magnitude"):
            return True
        if kind in _UNIT_WORD_SECOND and _is_figure(text, at):
            return True
    return False


def _sign_view(text: str) -> str:
    return html.unescape(_plain_text(text or ""))


def _sign_touches_amount(text: str, start: int, end: int) -> bool:
    after = text[end:end + 1]
    return bool(
        after and (after.isdecimal() or unicodedata.category(after) == "Sc")
        or start and text[start - 1].isdecimal()
    )


def _sign_touches_figure(text: str, start: int, end: int) -> bool:
    before = start
    while before and _figure_kind(text[before - 1]) == "magnitude":
        before -= 1
    if before and _is_figure(text, before - 1):
        return True
    for index in range(end, min(end + PDF_SIGN_FIGURE_DISTANCE, len(text))):
        char = text[index]
        if _is_figure(text, index):
            return True
        if char in _LINE_BREAKS or (
            unicodedata.category(char).startswith("P") and char not in _OPENING_BRACKETS
        ):
            return False
    return False


def _occurrences(text: str, fragment: str):
    position = text.find(fragment)
    while position >= 0:
        yield position, position + len(fragment)
        position = text.find(fragment, position + len(fragment))


def pdf_sign_amount_failure(
    element: Mapping, source_text: str, translated_text: str
) -> str | None:
    """``PDF_PROSE_NUMBERS_REASON`` if the translation keeps a protected sign
    of ``element`` that touches an amount in ``source_text`` with no figure
    near it (condition S, a proximity check: it does not establish that the
    amount survived), else ``None``."""
    fragments = [
        fragment.get("text", "")
        for fragment in (element.get("inline_math_fragments") or [])
        if isinstance(fragment, dict) and str(fragment.get("text", "")).strip()
    ]
    if not fragments:
        return None
    source, translated = _sign_view(source_text), _sign_view(translated_text)
    if _plain_text(translated_text or "") == _plain_text(source_text or ""):
        return None  # the element kept in source
    if pdf_text_is_numeric_dominant(source_text):
        return None  # condition N's
    # The protected occurrences, found in order as the protection finds them.
    needed: dict[str, int] = {}
    cursor = 0
    for fragment in fragments:
        start = source.find(fragment, cursor)
        if start < 0:
            continue
        cursor = start + len(fragment)
        if set(fragment) <= PDF_SIGN_CHARACTERS and _sign_touches_amount(source, start, cursor):
            needed[fragment] = needed.get(fragment, 0) + 1
    for fragment, count in needed.items():
        attached = sum(
            1 for start, end in _occurrences(translated, fragment)
            if _sign_touches_figure(translated, start, end)
        )
        if attached < count:
            return PDF_PROSE_NUMBERS_REASON
    return None


def record_pdf_delivered_prose_loss(
    audit, page_number: int, element_index: int, stage: str, source_text: str = "", why: str = ""
) -> None:
    """Record, once per page and element, a translation drawn although it
    fails condition S, the proximity check (keeping its source would cost
    more than the element)."""
    entries = audit.setdefault(PDF_DELIVERED_ELEMENT_WARNINGS, [])
    if any(
        int(entry.get("page", 0) or 0) == int(page_number)
        and int(entry.get("element", -1)) == int(element_index)
        and entry.get("reason") == PDF_PROSE_NUMBERS_REASON
        for entry in entries
    ):
        return
    entry = {
        "page": int(page_number),
        "element": int(element_index),
        "stage": str(stage),
        "reason": PDF_PROSE_NUMBERS_REASON,
        "detail": (
            "translation drawn although a protected sign that touches an amount "
            "in its source text has no figure near it (a proximity check; it "
            "does not establish whether the amount survived)"
        ),
    }
    if why:
        entry["waived"] = str(why)[:200]
    if source_text:
        entry["source_text"] = _collapsed_plain_text(source_text)[:160]
    entries.append(entry)


__all__ = [
    "LABEL_ACQUIRED_MIN_NUMBERS",
    "NUMERIC_ROW_MIN_NUMBERS",
    "PDF_LABEL_NUMBERS_REASON",
    "PDF_NUMERIC_ROW_REASON",
    "PDF_DELIVERED_ELEMENT_WARNINGS",
    "PDF_PROSE_NUMBERS_REASON",
    "PDF_SIGN_CHARACTERS",
    "PDF_SIGN_FIGURE_DISTANCE",
    "pdf_numbers",
    "pdf_numeric_guard_failure",
    "pdf_sign_amount_failure",
    "pdf_text_is_numeric_dominant",
    "record_pdf_delivered_prose_loss",
]
