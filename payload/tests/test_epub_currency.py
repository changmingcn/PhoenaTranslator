from phoena_translator.epub.math_protection import (
    find_xhtml_math_intervals,
    protect_xhtml_math_fragments,
    restore_xhtml_math_fragments,
)


def test_two_dollar_amounts_do_not_hide_ordinary_prose():
    text = (
        '<p>Investment grew from $30 billion a year in the lead-up to the '
        'financial crash to an annual expenditure of more than $110 billion.</p>'
    )
    protected, records = protect_xhtml_math_fragments(text)
    assert protected == text
    assert records == []


def test_currency_does_not_consume_a_later_formula():
    text = '<p>It costs $30, while the formula is $x-y$ and $a^2+b^2=c^2$.</p>'
    intervals = find_xhtml_math_intervals(text)
    formulas = [text[start:end] for start, end, reason in intervals
                if reason == 'latex_inline_dollar']
    assert formulas == ['$x-y$', '$a^2+b^2=c^2$']


def test_genuine_formulas_remain_exact_after_translation():
    text = '<p>Value $x-y$, $\\frac{a}{b}$, and $x=2$.</p>'
    protected, records = protect_xhtml_math_fragments(text)
    assert len(records) == 3
    assert restore_xhtml_math_fragments(protected.replace('Value', '数值'), records) == text.replace('Value', '数值')
