from phoena_translator.epub.math_protection import find_xhtml_math_intervals


def _dollar_formulas(text):
    return [
        text[start:end]
        for start, end, reason in find_xhtml_math_intervals(text)
        if reason == "latex_inline_dollar"
    ]


def test_two_dollar_amounts_do_not_hide_ordinary_prose():
    text = (
        '<p>Investment grew from $30 billion a year in the lead-up to the '
        'financial crash to an annual expenditure of more than $110 billion.</p>'
    )
    assert find_xhtml_math_intervals(text) == []


def test_currency_does_not_consume_a_later_formula():
    text = '<p>It costs $30, while the formula is $x-y$ and $a^2+b^2=c^2$.</p>'
    assert _dollar_formulas(text) == ['$x-y$', '$a^2+b^2=c^2$']


def test_genuine_formulas_are_all_located():
    text = '<p>Value $x-y$, $\\frac{a}{b}$, and $x=2$.</p>'
    assert _dollar_formulas(text) == ['$x-y$', '$\\frac{a}{b}$', '$x=2$']
