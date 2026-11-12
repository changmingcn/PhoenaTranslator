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

import re
import unicodedata

from phoena_translator.pdf.text_shape import _plain_text

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
_CURRENCY = "[$€£¥]"
# A number: an integer part with optional thousands separators and an
# optional decimal part, or a decimal part alone (``.1`` is ``0.1``).
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+"
_UNIT_SUFFIX = r"(?:x|bps?|pp|bn|mn|[mkb])"
_SIGN = "[-+\u2212\u2013]"
# The one sign grammar.  A signed value, in every common spelling, is read
# once into a value mark ``⟦V<value>⟧`` that value extraction and data-token
# classification both see:
#
# * accounting parentheses around one number, with or without inner spaces,
#   a currency symbol inside or before them, a percent sign or a unit suffix
#   inside: ``(1.2)`` ``( 1.2 )`` ``(1.2bn)`` ``(1.2%)`` ``($1.2)``
#   ``$(1.2)`` are all ``-1.2`` (``-1.2%`` with the percent sign);
# * a sign before or after a currency symbol: ``-1.2`` ``−1.2`` ``–1.2``
#   ``-$1.2`` ``$-1.2`` (a plus sign leaves the value positive);
# * a hyphen or minus sign spaced off its number: ``- 1.2``.
#
# A sign never follows a letter, digit or decimal point (a dash between
# digits is a separator: ``2025-26``), and a spaced en dash is a range
# (``1.5 – 2.5``).  Parentheses around several values or around words are
# grouping: their numbers stay unsigned.
_SIGNED_VALUE_RE = re.compile(
    rf"(?<![0-9.]){_CURRENCY}?\(\s*{_CURRENCY}?(?P<acct>{_NUMBER})(?P<acct_pct>%?)"
    rf"(?P<acct_unit>(?i:{_UNIT_SUFFIX})?)\s*\)"
    rf"|(?<![A-Za-z0-9.]){_CURRENCY}?(?P<sign>{_SIGN}|[-\u2212] ){_CURRENCY}?"
    rf"(?P<num>{_NUMBER})(?P<pct>%?)"
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
# A data token, after enclosing brackets and trailing punctuation are
# stripped: a date mark, or a number (or signed-value mark), range or ratio
# with an optional currency sign, percent sign or short unit suffix.
_DATA_TOKEN_RE = re.compile(
    r"\u27e6[DY][^\u27e7]*\u27e7"
    rf"|(?:\u27e6V[^\u27e7]*\u27e7|{_CURRENCY}?\.?\d)[\d,./:\-\u2013]*%?{_UNIT_SUFFIX}?",
    re.IGNORECASE,
)
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
        value = _canonical_value(match.group("acct"), True, match.group("acct_pct"))
        return _VALUE_MARK.format(value) + match.group("acct_unit")
    negative = match.group("sign").strip() != "+"
    return _VALUE_MARK.format(_canonical_value(match.group("num"), negative, match.group("pct")))


def _numeric_view(text: str) -> str:
    """Plain text without superscripts, full-width forms folded, dates,
    apostrophe years and signed values marked as single values."""
    without_superscripts = _SUPERSCRIPT_RE.sub(" ", text or "")
    view = unicodedata.normalize("NFKC", _plain_text(without_superscripts))
    for pattern in _DATE_PATTERNS:
        view = pattern.sub(_date_mark, view)
    view = _SHORT_YEAR_RE.sub(lambda m: _SHORT_YEAR_MARK.format(m.group(1)), view)
    return _SIGNED_VALUE_RE.sub(_signed_value_mark, view)


def pdf_numbers(text: str) -> list[str]:
    """The values of ``text`` in order, each in one canonical spelling.

    A number drops thousands separators, leading zeros of its integer part, a
    plus sign and a currency symbol; a minus sign (every spelling the sign
    grammar ``_SIGNED_VALUE_RE`` reads), the decimal part as written and a
    percent sign are kept, so ``1,250`` == ``1250``, ``02`` == ``2``, ``.5`` ==
    ``0.5``, ``( 1.2 )`` == ``$(1.2)`` == ``- 1.2`` == ``-1.2`` but ``0.30`` != ``0.3``,
    ``-2`` != ``2`` and ``36%`` is ``36%``.  A date is ``D<year>-<m>-<d>``
    with the year as written, an apostrophe year ``Y<yy>``.
    """
    values = []
    for match in _VALUE_RE.finditer(_numeric_view(text)):
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


def _same_numbers(source_text: str, translated_text: str) -> bool:
    source = pdf_numbers(source_text)
    translated = pdf_numbers(translated_text)
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


def _is_data_token(token: str) -> bool:
    return bool(_DATA_TOKEN_RE.fullmatch(token) or _NO_DATA_TOKEN_RE.fullmatch(token))


def pdf_text_is_numeric_dominant(source_text: str) -> bool:
    """Whether most tokens of ``source_text`` are data and it holds >= 3 numbers."""
    view = _numeric_view(source_text)
    if len(pdf_numbers(source_text)) < NUMERIC_ROW_MIN_NUMBERS:
        return False
    tokens = _counted_tokens(view)
    data = sum(1 for token in tokens if _is_data_token(token))
    return 2 * data > len(tokens)


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
    if pdf_text_is_numeric_dominant(source_text):
        if not _same_numbers(source_text, translated_text):
            return PDF_NUMERIC_ROW_REASON
        return None
    source_view = _numeric_view(source_text)
    if (
        not re.search(r"\d", source_view)
        and len(pdf_numbers(translated_text))
        - len(_SPELLED_NUMBER_RE.findall(source_view))
        >= LABEL_ACQUIRED_MIN_NUMBERS
    ):
        return PDF_LABEL_NUMBERS_REASON
    return None


__all__ = [
    "LABEL_ACQUIRED_MIN_NUMBERS",
    "NUMERIC_ROW_MIN_NUMBERS",
    "PDF_LABEL_NUMBERS_REASON",
    "PDF_NUMERIC_ROW_REASON",
    "pdf_numbers",
    "pdf_numeric_guard_failure",
    "pdf_text_is_numeric_dominant",
]
