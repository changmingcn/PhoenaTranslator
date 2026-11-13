"""A protected sign keeps the amount it touches, or the element keeps its source.

Condition S (reason ``prose-number-missing``): extraction protects sign-like
glyphs of running text (``+`` ``~`` and their kind) as inline-math
fragments.  Where such a fragment touches an amount in the source, the
translation must still show it touching a number: a digit or Chinese
numeral right before it, or one within ``PDF_SIGN_FIGURE_DISTANCE``
characters after it with no punctuation mark and no line break between.
Nothing else is compared.  Numeric-dominant rows stay
with condition N; an accepted cross-page merge endpoint is never checked.
Translation validation retries and then keeps the element as exact source
text; the batch restore re-translates the part; assembly keeps the element's
source ink.  Where the inline-math restore consumes the digits after a
damaged placeholder of a sign (a dropped digest followed by the amount), the
sign is left without a figure and the check rejects the result.

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
from phoena_translator.pdf.model import SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.translate import page as translate_page
from phoena_translator.pdf.translate import stage as translate_stage
from phoena_translator.pdf.translate.numeric_guard import (
    PDF_NUMERIC_ROW_REASON,
    PDF_PROSE_NUMBERS_REASON,
    PDF_SIGN_FIGURE_DISTANCE,
    pdf_numeric_guard_failure,
    pdf_sign_amount_failure,
)
from phoena_translator.pdf.translate.placeholders import (
    _protect_pdf_inline_math_fragments,
    _restore_pdf_inline_math_fragments,
)
from phoena_translator.pdf.translate.validation import _validate_pdf_page_translations

log = logging.getLogger("test-sign-amount-check")


def _signs(*signs):
    return [{"text": sign, "font_kind": "unicode"} for sign in signs]


def _elem(source, *signs, y=140.0):
    return {
        "type": "text",
        "layout_class": "body",
        "content": source,
        "bbox": [72.0, y, 540.0, y + 30.0],
        "fontsize": 9.0,
        "inline_math_fragments": _signs(*signs),
    }


def _fails(source, translated, *signs):
    return pdf_sign_amount_failure(_elem(source, *signs), source, translated) == PDF_PROSE_NUMBERS_REASON


ORDERS = "Machine tool orders rose +$3.2bn in May, after +$1.1bn in April."
LEAD = "Lead times for the parts lengthened by ~40bp of the order value"
PLUS_AFTER = "A grid upgrade would need $250+bn of cable over five years."


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "translated", "signs"),
    [
        # The sign followed by punctuation.
        (ORDERS, "五月机床订单增加+，四月为+。", "++"),
        # By a magnitude word alone.
        (ORDERS, "五月机床订单增加+亿美元，四月为+亿美元。", "++"),
        # By a unit alone.
        (LEAD, "零件交货期延长了订单金额的~bp", "~"),
        (LEAD, "零件交货期延长了订单金额的~个基点", "~"),
        # At the end of the text.
        (LEAD, "零件交货期延长了订单金额的~", "~"),
        # The figure-before-sign form.
        (PLUS_AFTER, "电网升级五年内需要+bn的电缆。", "+"),
        # One of two signs lost its amount.
        (ORDERS, "五月机床订单增加+32亿美元，四月为+。", "++"),
        # A comma, then a figure of another clause: not the sign's figure.
        (ORDERS, "五月机床订单增加+，为32亿美元，四月为+，即11亿美元。", "++"),
        (ORDERS, "五月机床订单增加+,3家工厂扩产，四月为+,1家。", "++"),
        # A magnitude word alone, then punctuation and a figure within reach.
        (ORDERS, "五月机床订单增加+亿，3月持平，四月为+亿。", "++"),
        # A unit alone, then punctuation or a line break and a figure.
        (LEAD, "零件交货期延长了订单金额的~bp，2年来最长", "~"),
        (LEAD, "零件交货期延长了~bp\n2年来最长", "~"),
        # A line break between the sign and the figure.
        (ORDERS, "五月机床订单增加+\n32亿美元，四月为+\r\n11亿美元。", "++"),
        # The figure nine characters after the sign.
        (ORDERS, "五月机床订单增加+亿美元的订单规模达3倍，四月为+亿美元的订单规模达1倍。", "++"),
        # A magnitude word before the sign, with no figure before it.
        (PLUS_AFTER, "电网升级五年内需要亿+美元的电缆。", "+"),
        # A magnitude word alone after the sign, in either script.
        (ORDERS, "五月机床订单增加+万亿美元，四月为+億美元。", "++"),
    ],
)
def test_a_sign_left_without_its_amount_fails(source, translated, signs):
    assert _fails(source, translated, *signs)


@pytest.mark.parametrize(
    ("source", "translated", "signs"),
    [
        # A converted unit.
        (ORDERS, "五月机床订单增加+32亿美元，四月为+11亿美元。", "++"),
        # The amount as written, a currency mark between sign and figure.
        (ORDERS, "五月机床订单增加+$3.2bn，四月为+$1.1bn。", "++"),
        # A space between.
        (ORDERS, "五月机床订单增加+ 32亿美元，四月为+ 11亿美元。", "++"),
        # An approximation word between.
        ("Spending of >$5bn is planned.", "计划支出>约50亿美元。", ">"),
        # A currency code with its symbol.
        ("Spending of >$5bn is planned.", "计划支出>US$5bn。", ">"),
        (ORDERS, "五月机床订单增加+HK$32亿，四月为+US$11亿。", "++"),
        # A two-character approximation word.
        ("Spending of >$5bn is planned.", "计划支出>大约50亿美元。", ">"),
        ("Lead times grew ~3 months this year.", "今年交货期延长了~大约三个月。", "~"),
        # An approximation word followed by a currency mark.
        ("Spending of >$5bn is planned.", "计划支出>大约$5bn。", ">"),
        ("Spending of >$5bn is planned.", "计划支出>约 $5bn。", ">"),
        # A Chinese numeral.
        ("Lead times grew ~3 months this year.", "今年交货期延长了~三个月。", "~"),
        ("Orders rose +$1bn in May.", "五月订单增加+十亿美元。", "+"),
        # The figure moved in front of the sign.
        (PLUS_AFTER, "电网升级五年内需要2500+亿美元的电缆。", "+"),
        ("Some 25+ issuers came to market.", "约25+家发行人进入市场。", "+"),
        # The clauses swapped.
        (ORDERS, "四月机床订单增加+11亿美元，五月为+32亿美元。", "++"),
        # Other signs of the kind.
        ("Rates moved −0.4% and ±2% bands held; spread ≈135bp.", "利率变动−0.4%，±2%区间守住，利差≈135bp。", "−±≈"),
    ],
)
def test_a_sign_still_touching_a_figure_passes(source, translated, signs):
    assert not _fails(source, translated, *signs)


def test_the_distance_is_eight_characters():
    assert PDF_SIGN_FIGURE_DISTANCE == 8


# Within eight characters after the sign, white space and an opening bracket
# of either width in between (R-18b).
@pytest.mark.parametrize(
    "between",
    ["", "约", "大约", "大约$", "约人民币", " US$", "（", "(", "（约", " （", "\t", "　",
     "大约人民币共计"],
)
def test_a_figure_among_the_eight_characters_after_the_sign_counts(between):
    assert not _fails("Orders rose +$1bn in May.", f"五月订单增加+{between}10亿美元。", "+")


def test_a_figure_nine_characters_after_the_sign_does_not_count():
    assert _fails("Orders rose +$1bn in May.", "五月订单增加+大约人民币共计合10亿。", "+")


@pytest.mark.parametrize(
    "between",
    ["，", ",", "。", "、", "；", "：", "-", "）", "\n", "\r", " ", "约，", "约\n", " ,",
     "（，", "约人民币，"],
)
def test_punctuation_or_a_line_break_before_the_figure_cuts_the_sign_off(between):
    assert _fails("Orders rose +$1bn in May.", f"五月订单增加+{between}10亿美元。", "+")


# A figure as Chinese writes it (R-18b, R-18c, R-18d): the financial
# numerals, 廿 卅, 百 千 outside a unit word, 半 wherever it stands; the
# magnitude words 万 亿 alone are not.
@pytest.mark.parametrize(
    "figure",
    ["百亿美元", "千亿美元", "百万美元", "千万美元", "半个百分点", "壹佰亿元", "贰拾亿元", "叁仟万元",
     "貳拾億元", "廿亿元", "卅年来", "两千亿", "１０亿美元"],
)
def test_a_chinese_figure_after_the_sign_counts(figure):
    assert not _fails("Orders rose +$10bn this spring.", f"今年春季订单增加+{figure}。", "+")
    assert not _fails("Orders rose ~$10bn this spring.", f"今年春季订单增加~{figure}。", "~")


def test_the_reviewers_hundred_and_thousand_pairs_pass():
    source = "Orders for new machine tools climbed +$10bn, with room for a further +$100bn."
    assert not _fails(source, "新机床订单增加了+百亿美元，还有+千亿美元的空间。", *"++")
    assert not _fails("Yields rose +0.5pp in May.", "五月收益率上升+半个百分点。", "+")
    assert _fails(source, "新机床订单增加了+亿美元，还有+千亿美元的空间。", *"++")
    # R-18d: 百 is the whole figure here, so these correct forms pass.
    assert not _fails("Rates moved ~1pp over the year.", "全年利率变动~百个基点。", "~")
    assert not _fails("Orders rose +$100 per unit.", "单位订单增加+百美元。", "+")


@pytest.mark.parametrize(
    "translated",
    ["五月订单增加+亿美元。", "五月订单增加+万美元。", "五月订单增加+万亿。", "五月订单增加+億美元。",
     "五月订单增加+萬元。", "五月订单增加+兆元。", "五月订单增加+萬億元。"],
)
def test_a_magnitude_word_alone_is_still_no_figure(translated):
    assert _fails("Orders rose +$1bn in May.", translated, "+")


# R-18c: a figure is a character to which Unicode gives a numeric value, read
# in its compatibility form, not a member of a list.
NUMERIC_SOURCE = "Orders rose +$0.2bn this spring, while demand was steady across the region."
# R-18e: the characters with a numeric value that are also ordinary
# characters of common words (参与 什么 大陆 队伍 收拾 肆意 拐点 漏洞 油漆
# 挂钩 陌生 阡陌), in both scripts.
EVERYDAY_NUMERALS = "参參什伍拾肆陆陸拐洞漆钩鈎陌阡"


@pytest.mark.parametrize(
    "numeral",
    ["\u8cae", "\u53c4", "\u534c", "\uf978", "\uf96b", "\uf9d3"],
    ids=["U+8CAE", "U+53C4", "U+534C", "U+F978", "U+F96B", "U+F9D3"],
)
def test_the_second_readers_numerals_count(numeral):
    # 貮 叄 卌, and the compatibility forms of 兩 參 陸.
    translated = f"本季度订单增加+{numeral}亿美元，而整个区域的需求保持稳定，企业经营情况依然平稳。"
    assert not _fails(NUMERIC_SOURCE, translated, "+")
    element = _elem(NUMERIC_SOURCE, "+")
    _validate_pdf_page_translations(1, [element], {0: translated}, set())


@pytest.mark.parametrize("figure", ["٣", "३", "௩", "Ⅲ", "ⅷ", "③", "〇", "弌", "陆", "拾", "皕", "５"])
def test_a_figure_of_any_script_counts(figure):
    assert not _fails(NUMERIC_SOURCE, f"本季度订单增加+{figure}亿美元。", "+")


def test_every_character_with_a_numeric_value_counts_but_the_qualified_ideographs():
    import unicodedata

    counted = 0
    for code in range(0x110000):
        char = chr(code)
        form = unicodedata.normalize("NFKC", char)
        value = unicodedata.numeric(form if len(form) == 1 else char, None)
        if value is None:
            continue
        ideograph = unicodedata.category(form[:1]) == "Lo"
        # Next to no other figure and no magnitude word: a magnitude word or
        # an everyday numeral alone is no figure (R-18e); every other
        # character with a numeric value is one, 百 and 千 included (R-18d).
        lost = _fails(NUMERIC_SOURCE, f"本季度订单增加+{char}美元。", "+")
        assert lost == (ideograph and value >= 10_000 or form in EVERYDAY_NUMERALS), (hex(code), value)
        counted += not lost
    assert counted > 1000


# 百 千 and their financial forms: a figure, except as the first character
# of a unit or percentage word, 百 before 分 or 千 before a metric unit
# (R-18d).
@pytest.mark.parametrize(
    "figure",
    ["百亿美元", "千万美元", "三百亿", "两千亿", "3千亿", "壹佰万元", "仟万元", "佰亿元", "百分之五", "千分之三"],
)
def test_a_hundred_or_thousand_next_to_a_figure_or_magnitude_word_counts(figure):
    assert not _fails(NUMERIC_SOURCE, f"本季度订单增加+{figure}。", "+")


@pytest.mark.parametrize(
    "figure",
    [
        # Before a measure word.
        "百个基点", "千个基点", "百倍", "千倍", "百年来",
        # Before a currency name.
        "百美元", "千美元", "百元", "千欧元", "百港元", "佰元", "仟元",
        # Before a magnitude word.
        "百亿美元", "千亿美元", "百万元", "千万元",
    ],
)
def test_a_hundred_or_thousand_that_is_the_whole_figure_counts(figure):
    assert not _fails(NUMERIC_SOURCE, f"本季度订单增加+{figure}。", "+")
    assert not _fails("Costs moved ~$0.1bn over the year.", f"全年变动~{figure}。", "~")


@pytest.mark.parametrize(
    ("source", "translated"),
    [
        ("Margins widened +0.5pp over the year.", "利润率全年扩大+个百分点。"),
        ("Margins widened +0.5pp over the year.", "利润率全年扩大+百分点。"),
        ("The share rose +3pp over the year.", "份额全年上升+百分比。"),
        ("The share rose +3% over the year.", "份额全年上升+百分之的水平。"),
        ("Output grew +1.2bn kWh over the year.", "全年发电量增加+亿千瓦时。"),
        ("Output grew +1.2m kWh over the year.", "全年发电量增加+万千瓦时。"),
        ("Output grew +12kW per unit.", "单机功率增加+千瓦。"),
        ("Loads grew +300kg per unit.", "单位载荷增加+千克。"),
        ("The range grew +40km this year.", "今年续航增加+千米。"),
        ("Intake grew +200kcal per day.", "每日摄入增加+千卡。"),
        ("Intake grew +800kJ per day.", "每日摄入增加+千焦。"),
        ("The band grew +5kHz this year.", "今年频宽增加+千赫。"),
        ("The line rose +10kV this year.", "今年线路电压提高+千伏。"),
        ("The load rose +3kA this year.", "今年负载电流提高+千安。"),
        ("Output rose +500t this year.", "今年产量增加+千吨。"),
        ("Output rose +500t this year.", "今年產量增加+千噸。"),
        ("Sales rose +900L this year.", "今年销量增加+千升。"),
        # The unit in a compatibility form (KANGXI RADICAL TILE).
        ("Output grew +12kW per unit.", "单机功率增加+千⽡。"),
    ],
)
def test_a_hundred_or_thousand_in_a_unit_or_percentage_word_is_no_figure(source, translated):
    assert _fails(source, translated, "+")


def test_an_ordinary_word_that_begins_with_a_hundred_counts():
    # R-18d: 百 outside the percent words is a figure, in 百货 and 百姓 too.
    assert not _fails("Chip sales grew +$5bn over the year.", "全年芯片销售额增长+亿美元的百货。", "+")
    assert not _fails("Outlays rose +$3bn for households.", "家庭支出增加+亿元百姓受益。", "+")


# Everyday numerals: next to another figure, another everyday numeral
# included, or a magnitude word only (R-18e, R-18g).
@pytest.mark.parametrize(
    "word",
    ["参与", "參與", "什么", "队伍", "入伍", "收拾", "拾取", "肆意", "陆地", "陸地", "大陆", "陆续",
     "拐点", "洞察", "漏洞", "漆器", "油漆", "钩子", "挂钩", "掛鈎", "陌生"],
)
def test_an_everyday_numeral_alone_in_a_word_is_no_figure(word):
    assert _fails(NUMERIC_SOURCE, f"本季度订单增加+{word}的企业增多。", "+")
    assert _fails(NUMERIC_SOURCE, f"本季度订单增加+亿美元{word}的企业增多。", "+")


def test_an_everyday_numeral_alone_before_the_sign_is_no_figure():
    assert _fails(PLUS_AFTER, "电网升级五年内需要大陆+美元的电缆。", "+")
    assert _fails(PLUS_AFTER, "电网升级五年内需要参与+亿美元的电缆。", "+")


@pytest.mark.parametrize(
    "figure",
    ["陆亿美元", "拾万元", "参亿元", "參億元", "伍万元", "肆亿元", "什亿", "陆拾亿元", "陸拾萬元", "伍佰元",
     "陆千元", "拾百个", "叁拾伍亿元", "壹拾陆元", "陆拾伍万元"],
)
def test_an_everyday_numeral_next_to_another_figure_or_magnitude_word_counts(figure):
    assert not _fails(NUMERIC_SOURCE, f"本季度订单增加+{figure}。", "+")


# R-18g: everyday numerals keep each other company, so an amount written
# only in them passes (the second reader's reproductions).
@pytest.mark.parametrize(
    "figure", ["伍拾元", "陆拾美元", "肆拾伍美元", "伍拾美元", "陸拾元", "拾伍个百分点"]
)
def test_everyday_numerals_keep_each_other_company(figure):
    assert not _fails(NUMERIC_SOURCE, f"本季度订单增加+{figure}。", "+")
    assert not _fails("Costs moved ~$0.1bn over the year.", f"全年变动~{figure}。", "~")
    assert not _fails(PLUS_AFTER, f"电网升级五年内需要{figure[:2]}+美元的电缆。", "+")


def test_the_second_readers_compound_amount_passes_page_validation():
    # The second reader's pair, with its prose tails.
    source = (
        "The measured cost increased +$50 this quarter while demand stayed firm across the region."
        " The survey covered the same locations as before, and the team used consistent"
        " procedures throughout the period. The report describes the underlying activity and"
        " the changes observed across the region."
    )
    translated = (
        "本季度测量成本增加+伍拾美元，各地区需求持续保持稳健，相关项目按照计划有序展开。"
        "本次调查涵盖与此前相同的地点，团队在整个期间采用一致的程序，"
        "报告介绍各地区的基础活动以及观察到的变化。"
    )
    assert not _fails(source, translated, "+")
    _validate_pdf_page_translations(1, [_elem(source, "+")], {0: translated}, set())


def test_ordinary_words_of_two_everyday_numerals_count_too():
    # R-18g: the accepted cost, a loss followed by 阡陌 or 陆陆续续 passes.
    assert not _fails(NUMERIC_SOURCE, "本季度订单增加+阡陌的企业增多。", "+")
    assert not _fails(NUMERIC_SOURCE, "本季度订单增加+陆陆续续的企业增多。", "+")
    assert not _fails(PLUS_AFTER, "电网升级五年内需要陆陆+美元的电缆。", "+")


@pytest.mark.parametrize(
    "translated", ["需要陆亿+美元。", "需要陆拾亿+美元。", "需要伍佰+美元。", "需要壹拾陆+美元。"]
)
def test_an_everyday_numeral_with_company_before_the_sign_counts(translated):
    assert not _fails(PLUS_AFTER, translated, "+")


@pytest.mark.parametrize(
    "numeral",
    list(EVERYDAY_NUMERALS) + ["參", "拾", "陸", "什", "洞"],
    ids=[f"U+{ord(c):04X}" for c in EVERYDAY_NUMERALS] + ["U+F96B", "U+F973", "U+F9D3", "U+F9FD", "U+FA05"],
)
def test_each_everyday_numeral_needs_company(numeral):
    from phoena_translator.pdf.translate.numeric_guard import _is_figure

    assert not _is_figure(numeral, 0) and not _is_figure(f"的{numeral}的", 1)
    assert _is_figure(f"{numeral}亿", 0) and _is_figure(f"万{numeral}", 1)
    assert _is_figure(f"3{numeral}", 1) and _is_figure(f"{numeral}五", 0)
    assert _is_figure(f"{numeral}百", 0) and _is_figure(f"{numeral}千元", 0)
    # Another everyday numeral is company too (R-18g), on either side.
    assert _is_figure(f"{numeral}拾亿", 0) and _is_figure(f"二陆{numeral}", 2)
    assert _is_figure(f"陆{numeral}", 1) and _is_figure(f"{numeral}陌生", 0)
    assert _is_figure(f"的{numeral}拾的", 1) and _is_figure(f"的伍{numeral}的", 2)


def test_a_hundred_or_thousand_in_a_unit_word_keeps_an_everyday_numeral_no_company():
    from phoena_translator.pdf.translate.numeric_guard import _is_figure

    assert not _is_figure("陆千瓦", 0) and not _is_figure("拾百分", 0) and not _is_figure("陆半", 0)
    assert _is_figure("陆千元", 0) and _is_figure("拾百个", 0)


# 半: a figure wherever it stands (R-18f), before any measure word or unit.
@pytest.mark.parametrize(
    "quantity",
    [
        # Before a measure word.
        "半个点", "半年", "半成", "半个百分点", "半小时",
        # The reviewers' reproductions: measure words and units of either
        # script, Latin units with or without a space, the percent signs.
        "半厘", "半点", "半點", "半档", "半檔", "半级", "半級", "半格", "半次", "半岁", "半歲", "半分",
        "半毫", "半米", "半吨", "半公斤", "半基点", "半亩", "半英里", "半bp", "半pp", "半 pp", "半%",
        "半‰",
    ],
)
def test_half_counts_before_any_measure_word_or_unit(quantity):
    assert not _fails("Yields rose +0.5pp in May.", f"五月收益率上升+{quantity}。", "+")
    assert not _fails("Yields rose ~0.5pp in May.", f"五月收益率上升~{quantity}。", "~")


def test_the_reviewers_half_pairs_pass_page_validation():
    pairs = [
        ("Policy rates were cut +0.5pp in the quarter, as the central bank responded to weaker "
         "demand across the region.", "本季度政策利率下调+半厘，央行回应了全区需求走弱。"),
        ("The measured load increased +0.5kg this quarter as work continued steadily across "
         "the region.", "本季度测量载荷增加+半公斤，各地区工作持续平稳推进，相关项目按照计划有序展开。"),
    ]
    for source, translated in pairs:
        assert not _fails(source, translated, "+")
        _validate_pdf_page_translations(1, [_elem(source, "+")], {0: translated}, set())


@pytest.mark.parametrize(
    "translated", ["全年芯片销售额增长+亿美元的半导体。", "份额上升+过半。", "份额上升+半。"]
)
def test_half_in_another_word_or_alone_counts_too(translated):
    # R-18f: the accepted cost, a loss followed by 半 in an ordinary word
    # passes, as before R-18c.
    assert not _fails("Chip sales grew +$5bn over the year.", translated, "+")


def test_a_hundred_or_thousand_before_a_figure_counts():
    from phoena_translator.pdf.translate.numeric_guard import _is_figure

    assert _is_figure("百5", 0) and _is_figure("千三", 0) and _is_figure("百亿", 0)
    assert not _is_figure("亿千瓦", 1) and not _is_figure("百分", 0)
    # R-18d: alone, at the end and in the financial forms; the unit words by
    # value, each metric unit after a thousand.
    assert _is_figure("百", 0) and _is_figure("千", 0) and _is_figure("百个", 0) and _is_figure("千美", 0)
    assert _is_figure("佰元", 0) and _is_figure("仟元", 0) and _is_figure("千分", 0)
    assert not _is_figure("佰分", 0) and not _is_figure("仟克", 0) and _is_figure("百克", 0)
    for unit in "瓦克米卡焦赫伏安吨噸升":
        assert not _is_figure(f"千{unit}", 0), unit


def test_half_is_a_figure_wherever_it_stands():
    from phoena_translator.pdf.translate.numeric_guard import _is_figure

    assert _is_figure("三半", 1) and _is_figure("3半", 1) and _is_figure("陆半", 1)
    assert _is_figure("半个", 0) and _is_figure("过半", 1) and _is_figure("半导体", 0)
    assert _is_figure("半", 0) and _is_figure("半厘", 0) and _is_figure("半bp", 0)


# Before the sign: the figure right before it, or a magnitude word between.
@pytest.mark.parametrize("translated", ["电网升级五年内需要2500亿+美元的电缆。", "需要2500万亿+美元。",
                                        "需要三千亿+美元。", "需要2500+亿美元。", "需要2兆+美元。"])
def test_a_figure_then_a_magnitude_word_before_the_sign_counts(translated):
    assert not _fails(PLUS_AFTER, translated, "+")


def test_a_figure_of_the_next_clause_behind_a_comma_is_still_out_of_reach():
    assert _fails(ORDERS, "五月机床订单增加+，3家工厂扩产，四月为+，1家。", *"++")
    assert _fails(ORDERS, "五月机床订单增加+（，3家），四月为+（，1家）。", *"++")


# Each sign spelling the inline-math detection protects (R-18b): the minus
# and dash spellings of a math font's span, and the Unicode forms.
ADDED_SIGNS = ["-", "\ufe63", "\uff0d", "\u2012", "\u2013", "\u2014", "\ufe31", "\ufe32",
               "\ufe58", "\u207a", "\u207b", "\u208a", "\u208b", "\ufb29", "\ufe64", "\ufe65",
               "\u2010", "\u2011", "\u2272", "\u2273"]


@pytest.mark.parametrize("sign", ADDED_SIGNS, ids=[f"U+{ord(c):04X}" for c in ADDED_SIGNS])
def test_each_added_sign_spelling_is_checked(sign):
    source = f"Costs moved {sign}$3.2bn this spring as demand changed across the region."
    assert _fails(source, f"本季度成本变动{sign}亿美元，区域需求发生变化。", sign)
    assert not _fails(source, f"本季度成本变动{sign}32亿美元，区域需求发生变化。", sign)


def test_the_sign_set_is_pinned_to_the_inline_math_detection():
    import unicodedata

    from phoena_translator.pdf.extract.math_detection import _pdf_inline_math_span_fragments
    from phoena_translator.pdf.translate.numeric_guard import PDF_SIGN_CHARACTERS

    for sign in PDF_SIGN_CHARACTERS:
        # A math font's span is protected as it is.
        assert _pdf_inline_math_span_fragments([{"text": sign, "font": "CambriaMath"}]) == [
            {"text": sign, "font_kind": "strong"}
        ], sign
        # Category Sm is protected by the Unicode path in any font.
        unicode_path = _pdf_inline_math_span_fragments([{"text": f"up {sign}5", "font": "Helvetica"}])
        assert (unicode_path == [{"text": sign, "font_kind": "unicode"}]) == (
            unicodedata.category(sign) == "Sm"
        ), sign
    # The set: the closed list of sign-like Sm characters of the earlier
    # check, every character whose compatibility form is a sign the row
    # guard reads or ``~ < >``, and the hyphens and comparisons a math font's
    # span protects as they are (the second reader's round-4 F4).
    signs = set("-+\u2212\u2012\u2013\u2014~<>")
    compatible = {
        chr(code) for code in range(0x110000) if unicodedata.normalize("NFKC", chr(code)) in signs
    }
    earlier = set("+\uff0b\ufe62\u2212\u2213\u00b1~\u223c\uff5e\u2248\u2243\u2245\u2252<>\u2264\u2265\u2266\u2267"
                  "\u2a7d\u2a7e\uff1c\uff1e\u226a\u226b")
    assert PDF_SIGN_CHARACTERS == earlier | compatible | set("\u2010\u2011\u2272\u2273")


@pytest.mark.parametrize(
    ("source", "translated", "signs"),
    [
        # No protected fragment: never checked, whatever the translation does.
        (ORDERS, "五月机床订单增加+，四月为+。", ""),
        # A fragment that is not sign-like.
        ("Set x=5 for the base case of $3bn.", "基本情形设x=，金额为。", "="),
        # A sign that touches no amount in the source.
        ("The issuer kept its A+ rating and grew 5%.", "该发行人保持A+评级。", "+"),
        ("Returns of + 3% were seen.", "回报率为+。", "+"),
    ],
)
def test_an_element_without_a_sign_touching_an_amount_is_never_checked(source, translated, signs):
    assert not _fails(source, translated, *signs)


def test_a_numeric_dominant_row_is_left_to_the_row_guard():
    row = "Net flows +1.2 +3.4 +5.6 +7.8"
    lossy = "净流量 + + + +"
    assert pdf_sign_amount_failure(_elem(row, "+", "+", "+", "+"), row, lossy) is None
    assert pdf_numeric_guard_failure(row, lossy) == PDF_NUMERIC_ROW_REASON


def test_the_element_kept_in_source_is_no_candidate():
    source = "Fees of +$ apply."  # a sign before a currency mark and no figure
    assert pdf_sign_amount_failure(_elem(source, "+"), source, source) is None


@pytest.mark.parametrize(
    ("source", "translated"),
    [
        # The general prose rule's false rejections (review rounds 1 and 2):
        # nothing of this kind is compared any more.
        ("Sales expanded by .5% over the entire reporting period.", "整个报告期销售额增长了0.5%。"),
        ("Revenue was $5-6bn in the full reporting period.", "报告期内收入为五十至六十亿美元。"),
        ("See section 2.1 for the method.", "方法见第2.1节。"),
        ("Margins of 5-6% are typical.", "利润率通常为百分之五到六。"),
        ("Each tranche priced 25bp wider.", "各档定价均宽了各25个基点。"),
        ("Output doubled to $8bn from $4bn.", "产出翻了一番。"),
        (
            "Revenue was $5bn and the coverage ratio was 5.0x during the reporting period.",
            "报告期内收入为50亿美元，覆盖率为5.0倍。",
        ),
        ("The meeting starts at 14:00 in the 1990s hall.", "会议在九十年代大厅下午两点开始。"),
    ],
)
def test_the_general_rules_false_rejections_pass(source, translated):
    element = _elem(source)
    assert pdf_sign_amount_failure(element, source, translated) is None
    try:
        _validate_pdf_page_translations(1, [element], {0: translated}, set())
    except PDFTranslationIntegrityError as error:  # a live check may still apply
        assert PDF_PROSE_NUMBERS_REASON not in {issue["reason"] for issue in error.issues}


# ---------------------------------------------------------------------------
# Translation validation and the batch restore
# ---------------------------------------------------------------------------

PROSE = (
    "Orders for new machine tools climbed +$12.3bn over the quarter, and the "
    "trade group sees room for a further +$4.5bn in the next quarter. Delivery "
    "times are expected to ease into the winter as suppliers clear their "
    "backlogs, while demand from export buyers holds up well."
)
ZH_REST = "随着供应商消化积压订单，预计入冬后交货时间将缩短，而出口买家的需求依然稳健。"
LOSSY_FORM = "本季度新机床订单增加了{}，行业协会认为下季度还有{}的空间。" + ZH_REST
FAITHFUL_FORM = "本季度新机床订单增加了{}123亿美元，行业协会认为下季度还有{}45亿美元的空间。" + ZH_REST
LOSSY = LOSSY_FORM.format("+", "+")
FAITHFUL = FAITHFUL_FORM.format("+", "+")
TITLE = "Quarterly order book review"
ZH_TITLE = "季度订单回顾"
SEG_RE = re.compile(r"PHOENA_SEG_\d{4}_[0-9A-F]{16}")
IMATH_RE = re.compile(r"PHOENA_IMATH_\d{4}_[0-9A-F]{16}")


def _prose(y=140.0):
    return _elem(PROSE, "+", "+", y=y)


def _title(y=100.0):
    return {**_elem(TITLE, y=y), "inline_math_fragments": []}


def _provider(form):
    """Answers the prose with ``form`` (its tokens kept), the title as such."""

    def answer(text):
        return form.format(*IMATH_RE.findall(text)) if "machine tools" in text else ZH_TITLE

    def build(calls):
        def translate_text(text, system_prompt=""):
            calls.append(text)
            markers = SEG_RE.findall(text)
            if not markers:
                return answer(text)
            parts = SEG_RE.split(text)[1:]
            return "\n≡≡≡SPLIT≡≡≡\n".join(
                f"{marker}\n{answer(part)}" for marker, part in zip(markers, parts)
            )

        return translate_text

    return build


def _run_translation(monkeypatch, provider, merge_decisions=(), elements=None):
    saved = {}
    monkeypatch.setattr(
        translate_stage,
        "_save_pdf_page_translation_cache",
        lambda _task, page, translations, _sha, elements=None: saved.update(
            {page: dict(translations)}
        ),
    )
    if elements is None:
        elements = [_title(), _prose()]
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


MERGE_1_2 = {
    "decision": "accepted",
    "source_page": 1,
    "source_element": 1,
    "destination_page": 2,
    "destination_element": 0,
}


def test_validation_names_the_condition_and_skips_merge_endpoints():
    elements = [_title(), _prose()]
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, elements, {0: ZH_TITLE, 1: LOSSY}, set())
    assert raised.value.issues == [{"element": 2, "reason": PDF_PROSE_NUMBERS_REASON}]
    _validate_pdf_page_translations(1, elements, {0: ZH_TITLE, 1: FAITHFUL}, set())
    # A merge endpoint is never checked; the cache rebind defers the check;
    # the page translator leaves it to the stage (no endpoints known).
    _validate_pdf_page_translations(1, elements, {0: ZH_TITLE, 1: LOSSY}, {1})
    _validate_pdf_page_translations(
        1, elements, {0: ZH_TITLE, 1: LOSSY}, set(), reject_numeric_flaws=False
    )
    _validate_pdf_page_translations(1, elements, {0: ZH_TITLE, 1: LOSSY})


def test_a_paragraph_that_keeps_losing_its_amounts_is_kept_as_exact_source(monkeypatch):
    context, elements, saved, calls = _run_translation(monkeypatch, _provider(LOSSY_FORM))

    assert saved[0] == {0: ZH_TITLE, 1: PROSE}
    assert elements[1]["skip_translate_reason"] == SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    (entry,) = context.pdf_audit["element_source_fallbacks"]
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "translation")
    assert entry["reason"] == PDF_PROSE_NUMBERS_REASON
    assert context.pdf_audit["source_page_fallbacks"] == []
    assert "delivered_element_warnings" not in context.pdf_audit
    # The batch, then one clean page retry: the batch restore does not run
    # the sign check, so no part is re-translated individually (R-19b).
    assert len(calls) == 2


def test_a_faithful_paragraph_is_committed_on_the_first_pass(monkeypatch):
    context, _elements, saved, calls = _run_translation(monkeypatch, _provider(FAITHFUL_FORM))

    assert saved[0] == {0: ZH_TITLE, 1: FAITHFUL}
    assert context.pdf_audit.get("element_source_fallbacks", []) == []
    assert len(calls) == 1


def test_a_merge_endpoint_is_not_checked_at_translation(monkeypatch):
    context, elements, saved, calls = _run_translation(
        monkeypatch, _provider(LOSSY_FORM), [MERGE_1_2]
    )

    assert saved[0] == {0: ZH_TITLE, 1: LOSSY}  # as the live release keeps it
    assert "skip_translate_reason" not in elements[1]
    assert context.pdf_audit["source_page_fallbacks"] == []
    assert context.pdf_audit.get("element_source_fallbacks", []) == []
    assert "delivered_element_warnings" not in context.pdf_audit
    assert len(calls) == 1  # the batch only; no individual call, no page retry


def test_the_batch_restore_does_not_run_the_sign_check(monkeypatch):
    """R-19b: a batch part that keeps a protected sign without a figure near
    it is restored as it is; the check runs at page validation only."""
    from phoena_translator.pdf.translate import numeric_guard

    def refuse(*_args, **_kwargs):
        raise AssertionError("the batch restore ran the sign check")

    monkeypatch.setattr(numeric_guard, "pdf_sign_amount_failure", refuse)
    assert not hasattr(translate_page, "pdf_sign_amount_failure")
    context = SimpleNamespace(task_id="t", logger=log)
    prose = _prose()
    protected, records = _protect_pdf_inline_math_fragments(PROSE, prose["inline_math_fragments"])
    tokens = [record["token"] for record in records]
    restore = translate_page._restore_batch_part
    protection = ((), records, ())
    assert restore(context, 0, prose, 1, PROSE, FAITHFUL_FORM.format(*tokens), protection) == FAITHFUL
    assert restore(context, 0, prose, 1, PROSE, LOSSY_FORM.format(*tokens), protection) == LOSSY
    assert protected.count("PHOENA_IMATH_") == 2


def _individual_call_fails(form):
    """``_provider(form)``, except that an individual (unsegmented) request
    raises, as the second reader's F4 reproductions do."""

    def build(calls):
        batch = _provider(form)(calls)

        def translate_text(text, system_prompt=""):
            if not SEG_RE.search(text):
                calls.append(text)
                raise RuntimeError("synthetic retry transport failure")
            return batch(text, system_prompt)

        return translate_text

    return build


def test_f4_an_exempt_merge_end_whose_individual_retry_would_fail_ends_as_live(monkeypatch):
    context, elements, saved, calls = _run_translation(
        monkeypatch, _individual_call_fails(LOSSY_FORM), [MERGE_1_2]
    )

    # The live release: one batch call, the translation committed.
    assert saved[0] == {0: ZH_TITLE, 1: LOSSY}
    assert len(calls) == 1
    assert "skip_translate_reason" not in elements[1]
    assert context.pdf_audit["source_page_fallbacks"] == []
    assert context.pdf_audit.get("element_source_fallbacks", []) == []


def test_f4_an_ordinary_paragraph_whose_individual_retry_would_fail_keeps_only_itself(monkeypatch):
    context, elements, saved, calls = _run_translation(
        monkeypatch, _individual_call_fails(LOSSY_FORM)
    )

    # No individual call: the element alone is kept in source, the page is
    # delivered with its other translation.
    assert not any(not SEG_RE.search(call) for call in calls)
    assert saved[0] == {0: ZH_TITLE, 1: PROSE}
    assert elements[1]["skip_translate_reason"] == SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    assert context.pdf_audit["source_page_fallbacks"] == []
    (entry,) = context.pdf_audit["element_source_fallbacks"]
    assert (entry["element"], entry["reason"]) == (1, PDF_PROSE_NUMBERS_REASON)


# ---------------------------------------------------------------------------
# The assembly net
# ---------------------------------------------------------------------------


def _cached_page(translation):
    kept = _text(
        [72, 200, 540, 240], PROSE, layout_class="body", inline_math_fragments=_signs("+", "+")
    )
    return [_body(100), kept, _body(400)], kept, {0: ZH[:30], 1: translation, 2: ZH[5:30]}


def test_a_cached_paragraph_that_lost_its_amounts_keeps_its_source_ink(tmp_path, font_path):
    elements, kept, translations = _cached_page(LOSSY)
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert "机床订单" not in _squash(text)
    assert document.fallback_pages() == []
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert entry["reason"].startswith(PDF_PROSE_NUMBERS_REASON)
    assert "protected sign" in entry["reason"]
    assert kept["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE
    assert "delivered_element_warnings" not in document.audit


def test_a_cached_paragraph_with_converted_units_is_drawn(tmp_path, font_path):
    elements, _kept, translations = _cached_page(FAITHFUL)
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert "123亿美元" in _squash(text)
    assert _kept_records(document) == []


def test_a_cached_merge_endpoint_is_not_checked(tmp_path, font_path):
    elements, kept, translations = _cached_page(LOSSY)
    document = Document(
        tmp_path,
        font_path,
        [(elements, translations), ([_body(100)], {0: ZH[:30]})],
        merge_decisions=[MERGE_1_2],
    )

    (text, _second) = document.assemble()

    assert document.fallback_pages() == []
    assert "机床订单" in _squash(text)
    assert _kept_records(document) == []
    assert "delivered_element_warnings" not in document.audit
    assert "skip_translate_reason" not in kept



# ---------------------------------------------------------------------------
# The restore may consume the digits after a damaged placeholder (R-4f)
# ---------------------------------------------------------------------------
#
# The inline-math restore is the live release's: after the record's index its
# fuzzy tail takes up to sixteen hexadecimal characters, so where a model
# drops the digest and writes the amount right after the index, the amount's
# leading digits go with the token.  What is left is a sign without a figure,
# which the check rejects.


def _dropped_digest(form, source, *signs):
    """``form`` with each protected sign's token written without its digest,
    and the protection records."""
    _protected_text, records = _protect_pdf_inline_math_fragments(source, _signs(*signs))
    return form.format(*(record["token"].rsplit("_", 1)[0] for record in records)), records


@pytest.mark.parametrize(
    ("source", "form", "consumed"),
    [
        # The reviewers' reproductions: an amount, a percentage, a decimal.
        ("Orders rose +$47bn in May.", "五月订单增加{}470亿美元。", "五月订单增加+亿美元。"),
        ("Orders rose +$47bn in May.", "五月订单增加{} 470亿美元。", "五月订单增加+亿美元。"),
        ("Orders rose +$47bn in May.", "五月订单增加{}_470亿美元。", "五月订单增加+亿美元。"),
        ("Margins rose +3% in May.", "五月利润率上升{}3%。", "五月利润率上升+%。"),
        ("Orders rose +$7.3bn in May.", "五月订单增加{}7.3bn。", "五月订单增加+.3bn。"),
        ("Yields rose +0.5pp in May.", "五月收益率上升{}0.5个百分点。", "五月收益率上升+.5个百分点。"),
        ("Orders rose +$130bn in May.", "五月订单增加{}1,300亿美元。", "五月订单增加+,300亿美元。"),
    ],
)
def test_digits_the_restore_consumes_leave_a_sign_the_check_rejects(source, form, consumed):
    answer, records = _dropped_digest(form, source, "+")
    restored = _restore_pdf_inline_math_fragments(answer, records)
    assert restored == consumed
    assert _fails(source, restored, "+")


def _dropped_digest_provider():
    """Answers the prose with the amounts written right after each token's
    index, the digest dropped; the title as such."""

    def build(calls):
        def answer(text):
            if "machine tools" not in text:
                return ZH_TITLE
            prefixes = [token.rsplit("_", 1)[0] for token in IMATH_RE.findall(text)]
            return FAITHFUL_FORM.format(*prefixes)

        def translate_text(text, system_prompt=""):
            calls.append(text)
            markers = SEG_RE.findall(text)
            if not markers:
                return answer(text)
            parts = SEG_RE.split(text)[1:]
            return "\n≡≡≡SPLIT≡≡≡\n".join(
                f"{marker}\n{answer(part)}" for marker, part in zip(markers, parts)
            )

        return translate_text

    return build


def test_page_validation_rejects_an_answer_whose_digits_the_restore_consumed(monkeypatch):
    answer, records = _dropped_digest(FAITHFUL_FORM, PROSE, "+", "+")
    restored = _restore_pdf_inline_math_fragments(answer, records)
    assert restored == "本季度新机床订单增加了+亿美元，行业协会认为下季度还有+亿美元的空间。" + ZH_REST
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, [_title(), _prose()], {0: ZH_TITLE, 1: restored}, set())
    assert raised.value.issues == [{"element": 2, "reason": PDF_PROSE_NUMBERS_REASON}]

    context, elements, saved, calls = _run_translation(monkeypatch, _dropped_digest_provider())

    assert saved[0] == {0: ZH_TITLE, 1: PROSE}
    assert elements[1]["skip_translate_reason"] == SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    (entry,) = context.pdf_audit["element_source_fallbacks"]
    assert (entry["page"], entry["element"], entry["reason"]) == (1, 1, PDF_PROSE_NUMBERS_REASON)
    assert context.pdf_audit["source_page_fallbacks"] == []
    assert len(calls) == 2  # the batch, then one page retry


def test_the_assembly_net_keeps_the_source_of_a_cached_answer_whose_digits_were_consumed(
    tmp_path, font_path
):
    answer, records = _dropped_digest(FAITHFUL_FORM, PROSE, "+", "+")
    restored = _restore_pdf_inline_math_fragments(answer, records)
    assert restored == "本季度新机床订单增加了+亿美元，行业协会认为下季度还有+亿美元的空间。" + ZH_REST
    elements, kept, translations = _cached_page(restored)
    document = Document(tmp_path, font_path, [(elements, translations)])

    (text,) = document.assemble()

    assert "机床订单" not in _squash(text)
    assert document.fallback_pages() == []
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert entry["reason"].startswith(PDF_PROSE_NUMBERS_REASON)
    assert "delivered_element_warnings" not in document.audit
