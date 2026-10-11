"""Claude provider: request mapping, contract, retries and ledger (no network).

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from phoena_translator.api_runtime import (
    APIRetryPolicy,
    APIUsagePrices,
    CompletionDependencies,
    completion_usage_record,
    create_chat_completion,
    extract_retry_delay,
    is_rate_limit_error,
    is_retryable_api_error,
)
from phoena_translator.config import AppConfig
from phoena_translator.llm import (
    CLAUDE_FALLBACK_BETA,
    CLAUDE_MIN_MAX_TOKENS,
    ClaudeCompletion,
    ClaudeResponseError,
    ClaudeSettings,
    ClaudeTranslationAdapter,
    DeepSeekTranslationAdapter,
    LLMSettings,
    build_claude_request,
)

log = logging.getLogger("test")

MESSAGES = [
    {"role": "system", "content": "你是翻译。"},
    {"role": "user", "content": "Rates rallied."},
]
SAMPLING_KEYS = {"temperature", "top_p", "top_k"}


# ---------------------------------------------------------------------------
# Configuration: provider switch, model, effort and per-model prices
# ---------------------------------------------------------------------------


def test_provider_defaults_to_deepseek():
    config = AppConfig.from_env({})
    assert config.llm_provider == "deepseek"
    assert config.claude_model == "claude-opus-5-5"
    assert config.claude_effort is None
    assert config.claude_fallbacks is True


def test_provider_switch_and_fail_closed_on_typo():
    assert AppConfig.from_env({"TRANSLATOR_LLM_PROVIDER": " Claude "}).llm_provider == "claude"
    with pytest.raises(ValueError, match="TRANSLATOR_LLM_PROVIDER"):
        AppConfig.from_env({"TRANSLATOR_LLM_PROVIDER": "claud"})
    with pytest.raises(ValueError, match="CLAUDE_EFFORT"):
        AppConfig.from_env({"CLAUDE_EFFORT": "extreme"})


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-5-5", (4.0, 20.0, 0.20, 5.0)),
        ("claude-sonnet-5-5", (2.0, 10.0, 0.20, 2.5)),
        ("claude-haiku-5-5", (0.10, 0.50, 0.01, 0.125)),
        ("claude-unknown-9", (4.0, 20.0, 0.20, 5.0)),
    ],
)
def test_claude_price_defaults_follow_model(model, expected):
    config = AppConfig.from_env({"CLAUDE_MODEL": model})
    actual = (
        config.claude_price_input_usd_per_m,
        config.claude_price_output_usd_per_m,
        config.claude_price_cache_read_usd_per_m,
        config.claude_price_cache_write_usd_per_m,
    )
    assert actual == pytest.approx(expected)


def test_claude_price_env_overrides():
    config = AppConfig.from_env(
        {
            "CLAUDE_MODEL": "claude-sonnet-5-5",
            "CLAUDE_PRICE_INPUT_USD_PER_M": "3",
            "CLAUDE_PRICE_CACHE_WRITE_USD_PER_M": "9",
        }
    )
    assert config.claude_price_input_usd_per_m == 3.0
    assert config.claude_price_cache_write_usd_per_m == 9.0
    assert config.claude_price_output_usd_per_m == 10.0


def test_application_wiring_defaults_to_deepseek():
    from phoena_translator import application

    if application.LLM_PROVIDER != "deepseek":
        pytest.skip("test process configured with a non-default provider")
    assert isinstance(application._llm_adapter, DeepSeekTranslationAdapter)
    assert application.ACTIVE_LLM_MODEL == application.DEEPSEEK_MODEL
    prices = application._current_api_prices()
    assert prices.cache_write_usd_per_million is None


# ---------------------------------------------------------------------------
# DeepSeek request path is unchanged, argument for argument
# ---------------------------------------------------------------------------


def test_deepseek_request_is_argument_for_argument_identical():
    calls: list[dict] = []
    sentinel = object()

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return sentinel

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    factory_kwargs: dict = {}

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return client

    adapter = DeepSeekTranslationAdapter(
        LLMSettings(api_key="k", base_url="http://localhost", model="deepseek-v4-flash"),
        client_factory=factory,
    )
    result = adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    assert result is sentinel
    assert calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": MESSAGES,
            "max_tokens": 65536,
            "temperature": 0.3,
        }
    ]
    assert factory_kwargs == {
        "api_key": "k",
        "base_url": "http://localhost",
        "timeout": 120.0,
        "max_retries": 0,
    }


# ---------------------------------------------------------------------------
# Claude request mapping
# ---------------------------------------------------------------------------


def test_request_maps_system_and_never_sends_sampling():
    params = build_claude_request(MESSAGES, model="claude-sonnet-5-5", max_tokens=65536)
    assert params["system"] == [
        {"type": "text", "text": "你是翻译。", "cache_control": {"type": "ephemeral"}}
    ]
    assert params["messages"] == [{"role": "user", "content": "Rates rallied."}]
    assert not SAMPLING_KEYS & set(params)
    plain = build_claude_request(
        MESSAGES, model="claude-sonnet-5-5", max_tokens=65536, prompt_cache=False
    )
    assert plain["system"] == "你是翻译。"


def test_request_max_tokens_floor_and_ceiling():
    low = build_claude_request(MESSAGES, model="claude-opus-5-5", max_tokens=512)
    assert low["max_tokens"] == CLAUDE_MIN_MAX_TOKENS
    same = build_claude_request(MESSAGES, model="claude-opus-5-5", max_tokens=65536)
    assert same["max_tokens"] == 65536
    high = build_claude_request(MESSAGES, model="claude-opus-5-5", max_tokens=10**6)
    assert high["max_tokens"] == 128_000


@pytest.mark.parametrize(
    "model,effort,thinking,output_config",
    [
        ("claude-opus-5-5", None, None, {"effort": "low"}),
        ("claude-opus-5-5", "medium", None, {"effort": "medium"}),
        ("claude-sonnet-5-5", None, {"type": "between_tools"}, None),
        ("claude-sonnet-5-5", "high", {"type": "between_tools"}, {"effort": "high"}),
        ("claude-sonnet-5-5", "xhigh", None, {"effort": "xhigh"}),
        ("claude-haiku-5-5", None, {"type": "disabled"}, None),
        ("claude-haiku-5-5", "low", {"type": "disabled"}, {"effort": "low"}),
        ("claude-haiku-5-5", "max", None, {"effort": "max"}),
        ("claude-sonnet-5-5-20261001", None, {"type": "between_tools"}, None),
        ("claude-other", None, None, None),
    ],
)
def test_request_thinking_and_effort_per_model(model, effort, thinking, output_config):
    params = build_claude_request(MESSAGES, model=model, max_tokens=65536, effort=effort)
    assert params.get("thinking") == thinking
    assert params.get("output_config") == output_config
    assert not SAMPLING_KEYS & set(params)


# ---------------------------------------------------------------------------
# Adapter: fallbacks wiring, SDK retries off, stop_reason handling
# ---------------------------------------------------------------------------


def _message(stop_reason="end_turn", blocks=None, model="claude-sonnet-5-5", **usage):
    usage_fields = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    usage_fields.update(usage)
    return SimpleNamespace(
        id="msg_1",
        model=model,
        stop_reason=stop_reason,
        stop_details=None,
        content=blocks
        if blocks is not None
        else [SimpleNamespace(type="text", text="利率上涨。")],
        usage=SimpleNamespace(**usage_fields),
    )


class _FakeAnthropic:
    def __init__(self, response):
        self.calls: list[tuple[str, dict]] = []
        outer = self

        class _Create:
            def __init__(self, kind):
                self.kind = kind

            def create(self, **kwargs):
                outer.calls.append((self.kind, kwargs))
                return response

        self.messages = _Create("messages")
        self.beta = SimpleNamespace(messages=_Create("beta"))


def _adapter(response, **settings):
    fake = _FakeAnthropic(response)
    factory_kwargs: dict = {}

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return fake

    config = {"api_key": "k", "model": "claude-sonnet-5-5"}
    config.update(settings)
    adapter = ClaudeTranslationAdapter(ClaudeSettings(**config), client_factory=factory)
    return adapter, fake, factory_kwargs


@pytest.mark.parametrize(
    "model,expect_beta",
    [("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-haiku-5-5", False)],
)
def test_fallbacks_wired_for_opus_and_sonnet_only(model, expect_beta):
    adapter, fake, factory_kwargs = _adapter(_message(model=model), model=model)
    result = adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    kind, kwargs = fake.calls[0]
    assert factory_kwargs["max_retries"] == 0
    assert not SAMPLING_KEYS & set(kwargs)
    if expect_beta:
        assert kind == "beta"
        assert kwargs["betas"] == [CLAUDE_FALLBACK_BETA]
        assert kwargs["fallbacks"] == "default"
    else:
        assert kind == "messages"
        assert "fallbacks" not in kwargs and "betas" not in kwargs
    assert result.choices[0].message.content == "利率上涨。"


def test_fallbacks_opt_out():
    adapter, fake, _ = _adapter(_message(), fallbacks=False)
    adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    assert fake.calls[0][0] == "messages"
    assert "fallbacks" not in fake.calls[0][1]


def test_only_text_blocks_are_read():
    blocks = [
        SimpleNamespace(type="thinking", thinking="internal"),
        SimpleNamespace(type="text", text="第一段"),
        SimpleNamespace(type="text", text="第二段"),
    ]
    adapter, _, _ = _adapter(_message(blocks=blocks))
    result = adapter.translate(messages=MESSAGES, max_tokens=100, temperature=0.3)
    assert result.choices[0].message.content == "第一段第二段"
    assert result.choices[0].finish_reason == "end_turn"


@pytest.mark.parametrize(
    "stop_reason", ["refusal", "max_tokens", "pause_turn", "model_context_window_exceeded"]
)
def test_unusable_stop_reasons_raise_non_retryable(stop_reason):
    adapter, _, _ = _adapter(_message(stop_reason=stop_reason))
    with pytest.raises(ClaudeResponseError) as caught:
        adapter.translate(messages=MESSAGES, max_tokens=100, temperature=0.3)
    assert caught.value.stop_reason == stop_reason
    assert not is_retryable_api_error(caught.value)
    assert not is_rate_limit_error(caught.value)


def test_fallback_served_is_recorded_in_ledger_fields():
    blocks = [
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="text", text="译文"),
    ]
    adapter, _, _ = _adapter(_message(blocks=blocks, model="claude-sonnet-5"))
    result = adapter.translate(messages=MESSAGES, max_tokens=100, temperature=0.3)
    assert result.fallback_served is True
    assert result.ledger_fields == {
        "provider": "claude",
        "served_model": "claude-sonnet-5",
        "fallback_served": True,
    }


def test_missing_anthropic_key_is_a_configuration_error():
    from phoena_translator.llm import LLMConfigurationError

    adapter = ClaudeTranslationAdapter(ClaudeSettings(api_key=""))
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        adapter.translate(messages=MESSAGES, max_tokens=100, temperature=0.3)


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


def test_claude_ledger_record_prices_cache_reads_and_writes():
    adapter, _, _ = _adapter(
        _message(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=600,
            cache_read_input_tokens=2000,
        )
    )
    response = adapter.translate(messages=MESSAGES, max_tokens=100, temperature=0.3)
    prices = APIUsagePrices(
        cache_hit_usd_per_million=0.20,
        cache_miss_usd_per_million=2.0,
        output_usd_per_million=10.0,
        cache_write_usd_per_million=2.5,
    )
    record = completion_usage_record(response, "claude-sonnet-5-5", prices, timestamp=1.0)
    assert record == {
        "schema": 2,
        "ts": 1.0,
        "model": "claude-sonnet-5-5",
        "prompt_tokens": 3600,
        "prompt_cache_hit_tokens": 2000,
        "prompt_cache_miss_tokens": 1600,
        "completion_tokens": 500,
        "total_tokens": 4100,
        "conservative_cost_usd": pytest.approx(0.0089),
        "cache_creation_input_tokens": 600,
        "provider": "claude",
    }
    json.dumps(record)


def test_deepseek_ledger_record_unchanged_schema_1():
    response = SimpleNamespace(
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_cache_hit_tokens": 400,
            "prompt_cache_miss_tokens": 600,
        }
    )
    prices = APIUsagePrices(0.0028, 0.14, 0.28)
    record = completion_usage_record(response, "deepseek-v4-flash", prices, timestamp=1.0)
    assert record == {
        "schema": 1,
        "ts": 1.0,
        "model": "deepseek-v4-flash",
        "prompt_tokens": 1000,
        "prompt_cache_hit_tokens": 400,
        "prompt_cache_miss_tokens": 600,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "conservative_cost_usd": pytest.approx(
            (400 * 0.0028 + 600 * 0.14 + 100 * 0.28) / 1_000_000
        ),
    }


# ---------------------------------------------------------------------------
# Retry classification for Anthropic exception classes
# ---------------------------------------------------------------------------

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, status, message="error", headers=None):
    response = httpx2.Response(status, request=_REQUEST, headers=headers or {})
    return cls(message, response=response, body=None)


def test_anthropic_rate_limit_and_transient_classification():
    rate = _status_error(anthropic.RateLimitError, 429, "rate_limit_error")
    overloaded = _status_error(anthropic.InternalServerError, 529, "Overloaded")
    server = _status_error(anthropic.InternalServerError, 500, "api_error")
    connection = anthropic.APIConnectionError(request=_REQUEST)
    timeout = anthropic.APITimeoutError(request=_REQUEST)
    assert is_rate_limit_error(rate) and not is_retryable_api_error(rate)
    for exc in (overloaded, server, connection, timeout):
        assert is_retryable_api_error(exc), exc
        assert not is_rate_limit_error(exc), exc


@pytest.mark.parametrize(
    "cls,status",
    [
        ("BadRequestError", 400),
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("NotFoundError", 404),
    ],
)
def test_anthropic_client_errors_never_retry(cls, status):
    # The message deliberately contains words the text heuristics treat as
    # transient; the typed classification must win.
    exc = _status_error(
        getattr(anthropic, cls), status, "connection timeout temporarily 429"
    )
    assert not is_retryable_api_error(exc)
    assert not is_rate_limit_error(exc)


def test_anthropic_retry_after_header_is_honoured():
    policy = APIRetryPolicy(6, 60, 5, 12, 90, 30)
    rate = _status_error(
        anthropic.RateLimitError, 429, "rate_limit_error", headers={"retry-after": "7"}
    )
    assert extract_retry_delay(rate, 0, policy) == 8.0
    bare = _status_error(anthropic.RateLimitError, 429, "rate_limit_error")
    assert extract_retry_delay(bare, 1, policy) == 24


def _dependencies(adapter, sleeps):
    policy = APIRetryPolicy(3, 60, 5, 12, 90, 30)
    return CompletionDependencies(
        adapter=adapter,
        concurrency_semaphore=threading.BoundedSemaphore(1),
        policy=policy,
        logger=log,
        wait_for_slot=lambda: None,
        record_usage=lambda response, model: None,
        record_event=lambda kind: {
            "rate_limit_count": 0,
            "error_count": 0,
            "success_count": 0,
            "window_seconds": 60,
        },
        record_rate_limit=lambda delay, exc: None,
        is_rate_limit_error=is_rate_limit_error,
        is_retryable_error=is_retryable_api_error,
        retry_delay=lambda exc, attempt: extract_retry_delay(exc, attempt, policy),
        sleep=sleeps.append,
    )


class _ScriptedAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def translate(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_runtime_retries_overloaded_then_succeeds():
    ok = SimpleNamespace(choices=[])
    adapter = _ScriptedAdapter(
        [_status_error(anthropic.InternalServerError, 529, "Overloaded"), ok]
    )
    sleeps: list[float] = []
    result = create_chat_completion(
        messages=MESSAGES,
        max_tokens=100,
        temperature=0.3,
        model="claude-sonnet-5-5",
        dependencies=_dependencies(adapter, sleeps),
    )
    assert result is ok and adapter.calls == 2 and sleeps == [3]


def test_runtime_does_not_retry_bad_request():
    adapter = _ScriptedAdapter(
        [_status_error(anthropic.BadRequestError, 400, "invalid_request_error")]
    )
    with pytest.raises(anthropic.BadRequestError):
        create_chat_completion(
            messages=MESSAGES,
            max_tokens=100,
            temperature=0.3,
            model="claude-sonnet-5-5",
            dependencies=_dependencies(adapter, []),
        )
    assert adapter.calls == 1


# ---------------------------------------------------------------------------
# Wire format through the real SDK with a mock transport (no network)
# ---------------------------------------------------------------------------


def _wire_capture(model, body):
    captured: dict = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=body, request=request)

    def factory(**kwargs):
        return anthropic.Anthropic(
            **kwargs,
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(handler)
            ),
        )

    adapter = ClaudeTranslationAdapter(
        ClaudeSettings(api_key="test-key", model=model), client_factory=factory
    )
    result = adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    return captured, result


def _wire_body(model):
    return {
        "id": "msg_wire",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "利率上涨。"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 7,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def test_wire_request_sonnet_uses_fallback_beta_and_no_sampling():
    captured, result = _wire_capture("claude-sonnet-5-5", _wire_body("claude-sonnet-5-5"))
    body = captured["body"]
    assert CLAUDE_FALLBACK_BETA in captured["headers"].get("anthropic-beta", "")
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "between_tools"}
    assert body["max_tokens"] == 65536
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert not SAMPLING_KEYS & set(body)
    assert isinstance(result, ClaudeCompletion)
    assert result.choices[0].message.content == "利率上涨。"
    assert result.usage["prompt_tokens"] == 12


def test_wire_request_haiku_plain_messages_endpoint():
    captured, _ = _wire_capture("claude-haiku-5-5", _wire_body("claude-haiku-5-5"))
    body = captured["body"]
    assert "anthropic-beta" not in captured["headers"]
    assert "fallbacks" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert not SAMPLING_KEYS & set(body)
