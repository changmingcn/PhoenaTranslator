"""Prompt rules: no added glosses + swap glossary on every translation path.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import threading
from types import SimpleNamespace

import pytest

from phoena_translator import application
from phoena_translator.api_runtime import (
    APIRetryPolicy,
    CompletionDependencies,
    create_chat_completion,
)
from phoena_translator.epub.pipeline import translate_single_chunk
from phoena_translator.llm import DeepSeekTranslationAdapter, LLMSettings
from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.page_translation import (
    _translate_batch,
    _translate_element,
    _translate_one,
)
from phoena_translator.pdf.translation import (
    PDFTranslationDependencies,
    translate_text as pdf_translate_text,
)
from phoena_translator.prompts import (
    FIDELITY_RULES,
    PROMPT_RULE_BLOCKS,
    SWAP_GLOSSARY,
    SYSTEM_PROMPT_TEXT,
    SYSTEM_PROMPT_XHTML,
)

log = logging.getLogger("test")

# sha256 of the 2026.08.31 / 2026.10.11 base prompts (git e8b6317).
LEGACY_TEXT_SHA256 = "03f7e7bd0d652a312628e571aa9bbe7921eb66d828c770bf2bb7592470760881"
LEGACY_XHTML_SHA256 = "9132cf80dcac0dbc8ae4c9df414ce18b0d5974569505949872e0281d5f869335"
# The only legacy text removed: it licensed exactly the glosses being banned.
REMOVED_GLOSS_CLAUSE = "，可在首次出现时括号注释中文"
SWAP_TEXT = "Swap spreads widened as Treasuries cheapened to swaps."
CHINESE_REPLY = "互换利差走阔，美债相对互换变便宜。"


def _has_rules(prompt: str) -> bool:
    return FIDELITY_RULES in prompt and SWAP_GLOSSARY in prompt


# ---------------------------------------------------------------------------
# Rule content and "all other rules unchanged"
# ---------------------------------------------------------------------------


def test_fidelity_rules_content():
    for needle in (
        "不得添加原文没有的解释、释义、括号说明或注释",
        "原文已有的括号照常保留",
        "译文不得新增括号",
        "QE、WAM、RV",
        "只有原文本身给出了全称",
        "Bottom line",
        "“结论”或“小结”",
        "不得译为“底线”",
    ):
        assert needle in FIDELITY_RULES, needle


def test_swap_glossary_content():
    for needle in (
        "interest rate swap → 利率互换",
        "swap spread(s) → 互换利差",
        "swap rate → 互换利率",
        "swap curve → 互换曲线",
        "SOFR swap(s) → SOFR互换",
        "swaption → 互换期权",
        "Treasuries cheapen to swaps → 美债相对互换变便宜",
        "FX swap → 外汇掉期",
        "swap points / forward points → 掉期点",
        "cross-currency swap → 交叉货币互换",
        "cross-currency basis → 交叉货币基差",
        "central-bank swap line → 货币互换额度",
        "绝不能译为“掉期”或“掉期利差”",
    ):
        assert needle in SWAP_GLOSSARY, needle


@pytest.mark.parametrize(
    "prompt,legacy_sha,anchor",
    [
        (SYSTEM_PROMPT_TEXT, LEGACY_TEXT_SHA256, "保留英文原文；但首字母大写"),
        (SYSTEM_PROMPT_XHTML, LEGACY_XHTML_SHA256, "保留英文原文\n5."),
    ],
)
def test_base_prompt_is_legacy_minus_gloss_clause_plus_rule_blocks(
    prompt, legacy_sha, anchor
):
    assert prompt.endswith(PROMPT_RULE_BLOCKS)
    body = prompt[: -len(PROMPT_RULE_BLOCKS)]
    assert "括号注释" not in body
    assert body.count(anchor) == 1
    restored = body.replace(
        anchor, anchor.replace("保留英文原文", "保留英文原文" + REMOVED_GLOSS_CLAUSE)
    )
    assert hashlib.sha256(restored.encode("utf-8")).hexdigest() == legacy_sha


# ---------------------------------------------------------------------------
# Every semantic-translation prompt path carries both rule blocks
# ---------------------------------------------------------------------------


def _fake_completion(replies, systems):
    replies = iter(replies)

    def create(*, messages, max_tokens, temperature, **_):
        systems.append(messages[0]["content"])
        content = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return create


def test_pdf_translate_text_default_and_retry_attempts_carry_rules():
    systems: list[str] = []
    dependencies = PDFTranslationDependencies(
        create_chat_completion=_fake_completion([SWAP_TEXT, CHINESE_REPLY], systems),
        strip_think_tags=lambda text: text,
        system_prompt_text=SYSTEM_PROMPT_TEXT,
        logger=log,
    )
    assert pdf_translate_text(SWAP_TEXT, dependencies=dependencies) == CHINESE_REPLY
    assert len(systems) == 2
    assert all(_has_rules(system) for system in systems)
    assert "上一次输出未充分翻译" in systems[1]


def test_application_facade_and_wiring_use_rule_prompts(monkeypatch):
    systems: list[str] = []
    monkeypatch.setattr(
        application, "_create_chat_completion", _fake_completion([CHINESE_REPLY], systems)
    )
    application.translate_text(SWAP_TEXT)
    assert len(systems) == 1 and _has_rules(systems[0])
    assert application.SYSTEM_PROMPT_TEXT is SYSTEM_PROMPT_TEXT
    assert application.SYSTEM_PROMPT_XHTML is SYSTEM_PROMPT_XHTML
    glossary_prompt = application._make_prompt_with_glossary(
        SYSTEM_PROMPT_TEXT, "SOFR → SOFR"
    )
    assert glossary_prompt.startswith(SYSTEM_PROMPT_TEXT)
    source = inspect.getsource(application)
    assert source.count("system_prompt_text=SYSTEM_PROMPT_TEXT") >= 2
    assert "system_prompt_xhtml=SYSTEM_PROMPT_XHTML" in source


def _page_context(systems: list[str], reply: str) -> PDFPageTranslationContext:
    def translate(text, system_prompt=None, **_):
        systems.append(system_prompt)
        return reply

    return PDFPageTranslationContext(
        task_id="t",
        prompt=application._make_prompt_with_glossary(SYSTEM_PROMPT_TEXT, "SOFR → SOFR"),
        logger=log,
        translate_text=translate,
    )


def test_pdf_page_batch_prompt_carries_rules():
    systems: list[str] = []
    context = _page_context(systems, "")
    elements = [{"text": SWAP_TEXT}, {"text": "Bottom line for swap spreads"}]
    try:
        _translate_batch(context, 0, elements, [(0, SWAP_TEXT), (1, "Bottom line for swap spreads")])
    except Exception:
        pass  # the empty reply is invalid; only the prompt matters here
    assert systems and all(_has_rules(system) for system in systems)
    assert "≡≡≡SPLIT≡≡≡" in systems[0]


def test_pdf_element_one_and_strict_retry_prompts_carry_rules():
    systems: list[str] = []
    context = _page_context(systems, SWAP_TEXT)  # stays English -> retry ladder
    _translate_one(context, 0, SWAP_TEXT, element={}, extra_hint="\n\nHINT")
    long_text = " ".join([SWAP_TEXT] * 6)
    try:
        _translate_element(context, 0, [{"text": long_text}], 0, long_text)
    except Exception:
        pass
    assert len(systems) >= 2
    assert all(_has_rules(system) for system in systems)
    assert systems[0].endswith("HINT")


@pytest.mark.parametrize("strict", [False, True])
def test_epub_chunk_prompts_carry_rules(strict):
    systems: list[str] = []

    def translate(text, system_prompt=None, **_):
        systems.append(system_prompt)
        return "<p>互换利差</p>"

    xhtml_prompt = application._make_prompt_with_glossary(SYSTEM_PROMPT_XHTML, "")
    translate_single_chunk(
        "t", "f.xhtml", 0, 1, "<html><body><p>Swap spreads.</p></body></html>",
        xhtml_prompt, translate_text=translate, logger=log, strict_retry=strict,
    )
    assert systems and all(_has_rules(system) for system in systems)


# ---------------------------------------------------------------------------
# DeepSeek request path: identical apart from the system prompt text
# ---------------------------------------------------------------------------


def test_deepseek_request_identical_apart_from_prompt_text():
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=CHINESE_REPLY))],
                usage=None,
            )

    adapter = DeepSeekTranslationAdapter(
        LLMSettings(api_key="k", base_url="http://localhost", model="deepseek-v4-flash"),
        client_factory=lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())),
    )
    policy = APIRetryPolicy(3, 60, 5, 12, 90, 30)
    runtime = CompletionDependencies(
        adapter=adapter,
        concurrency_semaphore=threading.BoundedSemaphore(1),
        policy=policy,
        logger=log,
        wait_for_slot=lambda: None,
        record_usage=lambda response, model: None,
        record_event=lambda kind: {
            "rate_limit_count": 0, "error_count": 0, "success_count": 0, "window_seconds": 60,
        },
        record_rate_limit=lambda delay, exc: None,
        is_rate_limit_error=lambda exc: False,
        is_retryable_error=lambda exc: False,
        retry_delay=lambda exc, attempt: 0.0,
        sleep=lambda seconds: None,
    )

    def create(**kwargs):
        return create_chat_completion(model="deepseek-v4-flash", dependencies=runtime, **kwargs)

    dependencies = PDFTranslationDependencies(
        create_chat_completion=create,
        strip_think_tags=lambda text: text,
        system_prompt_text=SYSTEM_PROMPT_TEXT,
        logger=log,
    )
    assert pdf_translate_text(SWAP_TEXT, dependencies=dependencies) == CHINESE_REPLY
    assert calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": SWAP_TEXT},
            ],
            "max_tokens": 65536,
            "temperature": 0.3,
        }
    ]
