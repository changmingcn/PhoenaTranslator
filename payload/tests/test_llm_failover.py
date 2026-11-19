"""Primary/backup key slots: shape detection, OpenAI adapter, failover rule (no network).

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import openai
import pytest

from phoena_translator.api_runtime import (
    APIRetryPolicy,
    APIRuntimeState,
    APIUsagePrices,
    CompletionDependencies,
    FailoverLLMGateway,
    LLMGateway,
    LLMKeysUnusableError,
    LLMRetryBudgetExhausted,
    LLMSlot,
    create_chat_completion,
    extract_retry_delay,
    is_rate_limit_error,
    is_retryable_api_error,
    key_unusable_reason,
    raise_key_unusable_immediately,
)
from phoena_translator.config import AppConfig, detect_slot_provider
from phoena_translator.llm import (
    OPENAI_MIN_MAX_TOKENS,
    ClaudeResponseError,
    ClaudeTranslationAdapter,
    DeepSeekTranslationAdapter,
    OpenAICompletion,
    OpenAIResponseError,
    OpenAISettings,
    OpenAITranslationAdapter,
    build_openai_request,
)
from phoena_translator.runtime import build_llm_gateway

log = logging.getLogger("test.failover")

MESSAGES = [
    {"role": "system", "content": "你是翻译。"},
    {"role": "user", "content": "Rates rallied."},
]
# Shape-only fixtures: filler, never real keys.
CLAUDE_KEY = "sk-ant-api03-" + "x" * 40
CLAUDE_KEY_2 = "sk-ant-api03-" + "y" * 40
OPENAI_KEY = "sk-proj-" + "z" * 40
DEEPSEEK_KEY = "sk-" + "0123456789abcdef" * 2


# ---------------------------------------------------------------------------
# Configuration: slot mode, provider detection, legacy mode unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,provider",
    [
        (CLAUDE_KEY, "claude"),
        ("sk-ant-usr-" + "q" * 20, "claude"),
        (OPENAI_KEY, "openai"),
        ("sk-svcacct-" + "a" * 30, "openai"),
        ("sk-admin-" + "b" * 30, "openai"),
        (DEEPSEEK_KEY, "deepseek"),
        ("sk-" + "ABCDEF0123456789" * 2, "deepseek"),
        ("sk-" + "0123456789abcdef" * 2 + "0", "openai"),  # 33 hex: not DeepSeek
        ("sk-" + "g" * 32, "openai"),  # 32 chars, not hex
        ("sk-" + "c" * 48, "openai"),
    ],
)
def test_provider_detected_from_key_shape(key, provider):
    assert detect_slot_provider("primary", key) == provider


def test_provider_override_wins_and_is_validated():
    assert detect_slot_provider("backup", "not-a-known-shape", " DeepSeek ") == "deepseek"
    assert detect_slot_provider("primary", CLAUDE_KEY, "openai") == "openai"
    with pytest.raises(ValueError, match="LLM_BACKUP_PROVIDER"):
        detect_slot_provider("backup", OPENAI_KEY, "gemini")


def test_unknown_key_shape_names_variables_not_the_key():
    secret_like = "AIzaSy-unknown-shape-filler"
    with pytest.raises(ValueError) as caught:
        AppConfig.from_env({"LLM_PRIMARY_API_KEY": secret_like})
    message = str(caught.value)
    assert "LLM_PRIMARY_API_KEY" in message and "LLM_PRIMARY_PROVIDER" in message
    assert secret_like not in message and "AIza" not in message
    with pytest.raises(ValueError, match="LLM_BACKUP_API_KEY.*LLM_BACKUP_PROVIDER"):
        AppConfig.from_env(
            {"LLM_PRIMARY_API_KEY": CLAUDE_KEY, "LLM_BACKUP_API_KEY": secret_like}
        )


def test_slot_mode_needs_primary_and_keys_stay_out_of_repr():
    config = AppConfig.from_env(
        {"LLM_PRIMARY_API_KEY": f" {CLAUDE_KEY}\n", "LLM_BACKUP_API_KEY": OPENAI_KEY}
    )
    assert config.llm_slot_mode
    assert config.llm_primary_api_key == CLAUDE_KEY
    assert (config.llm_primary_provider, config.llm_backup_provider) == ("claude", "openai")
    assert CLAUDE_KEY not in repr(config) and OPENAI_KEY not in repr(config)
    assert config.openai_model == "gpt-6.2-sol"
    assert config.openai_base_url == "https://api.openai.com/v1"
    assert config.openai_reasoning_effort == "low"
    assert config.llm_failover_retry_primary_seconds == 3600.0


def test_backup_without_primary_is_legacy_mode_with_one_warning(tmp_path, caplog):
    # A malformed backup key is not even parsed outside slot mode.
    config = AppConfig.from_env({"LLM_BACKUP_API_KEY": "anything"}, home=tmp_path)
    assert not config.llm_slot_mode and config.llm_backup_provider == ""
    with caplog.at_level(logging.WARNING, logger=log.name):
        gateway = build_llm_gateway(config, log)
    assert isinstance(gateway, LLMGateway)
    assert isinstance(gateway.adapter, DeepSeekTranslationAdapter)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "LLM_BACKUP_API_KEY" in warnings[0].getMessage()
    assert "anything" not in warnings[0].getMessage()


@pytest.mark.parametrize(
    "env,adapter_cls,model_attr",
    [
        ({}, DeepSeekTranslationAdapter, "deepseek_model"),
        ({"TRANSLATOR_LLM_PROVIDER": "claude"}, ClaudeTranslationAdapter, "claude_model"),
    ],
)
def test_legacy_mode_gateway_is_unchanged(tmp_path, env, adapter_cls, model_attr):
    config = AppConfig.from_env(
        {**env, "ANTHROPIC_API_KEY": CLAUDE_KEY, "DEEPSEEK_API_KEY": DEEPSEEK_KEY},
        home=tmp_path,
    )
    gateway = build_llm_gateway(config, log)
    assert type(gateway) is LLMGateway
    assert isinstance(gateway.adapter, adapter_cls)
    assert gateway.model == getattr(config, model_attr)
    assert gateway.ledger_fields == {}
    assert gateway._dependencies.raise_immediately(_openai_error(429, "insufficient_quota")) is False
    expected_key = CLAUDE_KEY if adapter_cls is ClaudeTranslationAdapter else DEEPSEEK_KEY
    assert gateway.adapter._settings.api_key == expected_key


def test_single_slot_is_one_gateway_without_failover(tmp_path):
    config = AppConfig.from_env({"LLM_PRIMARY_API_KEY": OPENAI_KEY}, home=tmp_path)
    gateway = build_llm_gateway(config, log)
    assert type(gateway) is LLMGateway
    assert isinstance(gateway.adapter, OpenAITranslationAdapter)
    assert gateway.model == "gpt-6.2-sol"
    assert gateway.ledger_fields == {"key_slot": "primary", "provider": "openai"}
    # Nothing to switch to: quota errors keep today's retry handling.
    assert gateway._dependencies.raise_immediately(_openai_error(429, "insufficient_quota")) is False


def test_two_slots_build_failover_with_shared_cap_and_own_pacing(tmp_path):
    config = AppConfig.from_env(
        {
            "LLM_PRIMARY_API_KEY": CLAUDE_KEY,
            "LLM_BACKUP_API_KEY": OPENAI_KEY,
            "OPENAI_MODEL": "gpt-6-luna",
            "LLM_FAILOVER_RETRY_PRIMARY_SECONDS": "60",
        },
        home=tmp_path,
    )
    gateway = build_llm_gateway(config, log)
    assert isinstance(gateway, FailoverLLMGateway)
    primary, backup = gateway.slots
    assert (primary.name, primary.provider, primary.gateway.model) == (
        "primary",
        "claude",
        config.claude_model,
    )
    assert (backup.name, backup.provider, backup.gateway.model) == (
        "backup",
        "openai",
        "gpt-6-luna",
    )
    assert isinstance(primary.gateway.adapter, ClaudeTranslationAdapter)
    assert isinstance(backup.gateway.adapter, OpenAITranslationAdapter)
    assert primary.gateway.prices.cache_write_usd_per_million is not None
    assert backup.gateway.prices.cache_miss_usd_per_million == config.openai_price_input_usd_per_m
    assert primary.gateway.state.concurrency_semaphore is backup.gateway.state.concurrency_semaphore
    assert primary.gateway.state.usage_lock is backup.gateway.state.usage_lock
    assert primary.gateway.state.rate_limit_lock is not backup.gateway.state.rate_limit_lock
    assert primary.gateway.usage_log == backup.gateway.usage_log
    assert gateway.retry_primary_seconds == 60.0


def test_two_keys_of_one_provider(tmp_path):
    config = AppConfig.from_env(
        {"LLM_PRIMARY_API_KEY": CLAUDE_KEY, "LLM_BACKUP_API_KEY": CLAUDE_KEY_2},
        home=tmp_path,
    )
    gateway = build_llm_gateway(config, log)
    keys = [slot.gateway.adapter._settings.api_key for slot in gateway.slots]
    assert [slot.provider for slot in gateway.slots] == ["claude", "claude"]
    assert keys == [CLAUDE_KEY, CLAUDE_KEY_2]


# ---------------------------------------------------------------------------
# OpenAI adapter: request shape, wire format, unusable output, ledger
# ---------------------------------------------------------------------------


def test_openai_request_never_sends_temperature_or_max_tokens():
    params = build_openai_request(MESSAGES, model="gpt-6.2-sol", max_tokens=100)
    assert params == {
        "model": "gpt-6.2-sol",
        "messages": MESSAGES,
        "max_completion_tokens": OPENAI_MIN_MAX_TOKENS,
        "reasoning_effort": "low",
    }
    big = build_openai_request(
        MESSAGES, model="m", max_tokens=65536, reasoning_effort=None
    )
    assert big["max_completion_tokens"] == 65536 and "reasoning_effort" not in big


def _openai_wire(body, *, effort="low"):
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body, request=request)

    factory_kwargs: dict = {}

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return openai.OpenAI(
            **kwargs,
            http_client=openai.DefaultHttpxClient(transport=httpx.MockTransport(handler)),
        )

    adapter = OpenAITranslationAdapter(
        OpenAISettings(api_key="test-key", reasoning_effort=effort),
        client_factory=factory,
    )
    return adapter, captured, factory_kwargs


def _openai_body(content="利率上涨。", finish_reason="stop", refusal=None):
    return {
        "id": "chatcmpl-wire",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6.2-sol-2026-11-01",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "total_tokens": 1300,
            "prompt_tokens_details": {"cached_tokens": 400},
            "completion_tokens_details": {"reasoning_tokens": 120},
        },
    }


def test_openai_wire_request_and_normalised_result():
    adapter, captured, factory_kwargs = _openai_wire(_openai_body())
    result = adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    body = captured["body"]
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-6.2-sol"
    assert body["max_completion_tokens"] == 65536
    assert body["reasoning_effort"] == "low"
    assert not {"temperature", "max_tokens", "top_p"} & set(body)
    assert factory_kwargs["max_retries"] == 0
    assert isinstance(result, OpenAICompletion)
    assert result.choices[0].message.content == "利率上涨。"
    assert result.usage == {
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "total_tokens": 1300,
        "prompt_cache_hit_tokens": 400,
        "prompt_cache_miss_tokens": 600,
    }
    assert result.ledger_fields == {
        "provider": "openai",
        "served_model": "gpt-6.2-sol-2026-11-01",
    }


@pytest.mark.parametrize(
    "body",
    [
        _openai_body(content="", finish_reason="length"),
        _openai_body(content="partial", finish_reason="content_filter"),
        _openai_body(content=None, refusal="I can't help with that."),
    ],
)
def test_openai_unusable_output_raises_non_retryable(body):
    adapter, _, _ = _openai_wire(body)
    with pytest.raises(OpenAIResponseError) as caught:
        adapter.translate(messages=MESSAGES, max_tokens=65536, temperature=0.3)
    assert not is_retryable_api_error(caught.value)
    assert not is_rate_limit_error(caught.value)
    assert key_unusable_reason(caught.value) is None


def test_openai_reasoning_effort_env_override():
    config = AppConfig.from_env({"OPENAI_REASONING_EFFORT": " Medium "})
    assert config.openai_reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# Failover rule with real SDK exception classes
# ---------------------------------------------------------------------------

_A_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
_O_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
FAST_POLICY = APIRetryPolicy(3, 60, 5, 0, 0, 0)  # no sleeping in tests


def _anthropic_error(cls, status, error_type, message="error"):
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    response = httpx2.Response(status, request=_A_REQUEST)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def _openai_error(status, code, message="error", cls=None):
    body = {"message": message, "type": "invalid_request_error", "code": code}
    if code == "insufficient_quota":
        body["type"] = "insufficient_quota"
    cls = cls or {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(status, openai.APIStatusError)
    response = httpx.Response(status, request=_O_REQUEST)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


KEY_UNUSABLE = {
    "anthropic-401": (lambda: _anthropic_error(anthropic.AuthenticationError, 401, "authentication_error", "invalid x-api-key"), 401, "authentication_error"),
    "anthropic-403": (lambda: _anthropic_error(anthropic.PermissionDeniedError, 403, "permission_error"), 403, "permission_error"),
    "anthropic-400-credit": (lambda: _anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API."), 400, "invalid_request_error"),
    "anthropic-400-usage-limit": (lambda: _anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", "You have reached your specified API usage limits. You will regain access later."), 400, "invalid_request_error"),
    "anthropic-400-workspace-usage-limit": (lambda: _anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", "You have reached your specified workspace API usage limits. You will regain access later."), 400, "invalid_request_error"),
    "openai-401": (lambda: _openai_error(401, "invalid_api_key"), 401, "invalid_api_key"),
    "openai-429-quota": (lambda: _openai_error(429, "insufficient_quota", "You exceeded your current quota."), 429, "insufficient_quota"),
    "deepseek-402": (lambda: _openai_error(402, "invalid_request_error", "Insufficient Balance"), 402, "invalid_request_error"),
}

NEVER_SWITCH = {
    "anthropic-connection": lambda: anthropic.APIConnectionError(request=_A_REQUEST),
    "anthropic-timeout": lambda: anthropic.APITimeoutError(request=_A_REQUEST),
    "anthropic-529": lambda: _anthropic_error(anthropic.InternalServerError, 529, "overloaded_error"),
    "anthropic-500": lambda: _anthropic_error(anthropic.InternalServerError, 500, "api_error"),
    "anthropic-400": lambda: _anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", "messages: field required"),
    "anthropic-404": lambda: _anthropic_error(anthropic.NotFoundError, 404, "not_found_error", "model: not found"),
    "openai-connection": lambda: openai.APIConnectionError(request=_O_REQUEST),
    "openai-timeout": lambda: openai.APITimeoutError(request=_O_REQUEST),
    "openai-503": lambda: _openai_error(503, "server_error", "Service unavailable"),
    "openai-400": lambda: _openai_error(400, "unsupported_parameter", "Unsupported parameter: 'max_tokens'"),
    "claude-refusal": lambda: ClaudeResponseError("Claude refusal", stop_reason="refusal", model="m"),
    "claude-truncated": lambda: ClaudeResponseError("Claude output truncated", stop_reason="max_tokens", model="m"),
    "openai-truncated": lambda: OpenAIResponseError("OpenAI output truncated", stop_reason="length", model="m"),
}


class _Adapter:
    """Scripted adapter: pops outcomes; ``default`` repeats once the script ends."""

    def __init__(self, provider, outcomes=(), default=None):
        self.provider = provider
        self.outcomes = list(outcomes)
        self.default = default
        self.calls = 0
        self.lock = threading.Lock()

    def translate(self, **kwargs):
        with self.lock:
            self.calls += 1
            outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok(provider, text="好"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ledger_fields={"provider": provider},
        served_by=provider,
    )


PRICES = APIUsagePrices(1.0, 2.0, 3.0)


def _failover(tmp_path, primary, backup, *, clock=None, retry_after=3600.0):
    shared = APIRuntimeState.create(2)
    slots = []
    for name, adapter in (("primary", primary), ("backup", backup)):
        gateway = LLMGateway(
            adapter,
            model=f"{adapter.provider}-model",
            prices=PRICES,
            usage_log=tmp_path / "deepseek_usage.jsonl",
            max_concurrency=2,
            logger=log,
            policy=FAST_POLICY,
            state=APIRuntimeState(
                concurrency_semaphore=shared.concurrency_semaphore,
                usage_lock=shared.usage_lock,
            ),
            ledger_fields={"key_slot": name, "provider": adapter.provider},
            raise_immediately=raise_key_unusable_immediately,
        )
        slots.append(LLMSlot(name, adapter.provider, gateway))
    kwargs = {} if clock is None else {"clock": clock}
    return FailoverLLMGateway(slots, retry_primary_seconds=retry_after, logger=log, **kwargs)


def _complete(gateway):
    return gateway.complete(messages=MESSAGES, max_tokens=65536, temperature=0.3)


def _failover_lines(caplog):
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("llm-failover:")
    ]


@pytest.mark.parametrize("case", sorted(KEY_UNUSABLE))
def test_key_unusable_switches_once_to_backup(tmp_path, caplog, case):
    make_error, status, code = KEY_UNUSABLE[case]
    primary = _Adapter("claude", default=make_error)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        result = _complete(gateway)
    assert result.served_by == "openai"
    assert primary.calls == 1, "key-unusable errors are not retried on the dead slot"
    assert backup.calls == 1
    lines = _failover_lines(caplog)
    assert lines == [
        f"llm-failover: from=primary(claude) to=backup(openai) status={status} code={code}"
    ]
    assert [r.levelno for r in caplog.records if r.getMessage() in lines] == [logging.WARNING]
    assert gateway.active_slot == "backup"


def test_ordinary_429_switches_only_after_retry_budget(tmp_path, caplog):
    rate = lambda: _anthropic_error(anthropic.RateLimitError, 429, "rate_limit_error")
    primary = _Adapter("claude", default=rate)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        result = _complete(gateway)
    assert result.served_by == "openai"
    assert primary.calls == FAST_POLICY.attempts
    assert _failover_lines(caplog) == [
        "llm-failover: from=primary(claude) to=backup(openai) status=429 "
        "code=rate_limit_exhausted"
    ]


def test_ordinary_429_recovering_inside_budget_does_not_switch(tmp_path, caplog):
    rate = _anthropic_error(anthropic.RateLimitError, 429, "rate_limit_error")
    primary = _Adapter("claude", [rate], default=_ok("claude"))
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "claude"
    assert (primary.calls, backup.calls) == (2, 0)
    assert _failover_lines(caplog) == []


@pytest.mark.parametrize("case", sorted(NEVER_SWITCH))
def test_never_switch_errors_behave_as_today(tmp_path, caplog, case):
    primary = _Adapter("claude", default=NEVER_SWITCH[case])
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        with pytest.raises(Exception) as caught:
            _complete(gateway)
    assert not isinstance(caught.value, LLMKeysUnusableError)
    assert backup.calls == 0
    assert _failover_lines(caplog) == []
    assert gateway.active_slot == "primary"
    # Transient errors were retried by api_runtime exactly as before.
    expected = NEVER_SWITCH[case]()
    if is_retryable_api_error(expected):
        assert isinstance(caught.value, LLMRetryBudgetExhausted)
        assert primary.calls == FAST_POLICY.attempts
    else:
        assert type(caught.value) is type(expected)
        assert primary.calls == 1


def test_both_slots_dead_raises_naming_both_status_codes(tmp_path, caplog):
    primary = _Adapter("claude", default=KEY_UNUSABLE["anthropic-401"][0])
    backup = _Adapter("openai", default=KEY_UNUSABLE["openai-401"][0])
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        with pytest.raises(LLMKeysUnusableError) as caught:
            _complete(gateway)
    message = str(caught.value)
    assert "primary(claude) HTTP 401 authentication_error" in message
    assert "backup(openai) HTTP 401 invalid_api_key" in message
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert [type(e) for e in caught.value.errors] == [
        anthropic.AuthenticationError,
        openai.AuthenticationError,
    ]
    assert (primary.calls, backup.calls) == (1, 1), "no ping-pong"
    assert len(_failover_lines(caplog)) == 1


def test_backup_dying_switches_back_to_primary(tmp_path, caplog):
    primary = _Adapter(
        "claude", [KEY_UNUSABLE["anthropic-401"][0]], default=_ok("claude")
    )
    backup = _Adapter(
        "openai", [_ok("openai")], default=KEY_UNUSABLE["openai-429-quota"][0]
    )
    gateway = _failover(tmp_path, primary, backup)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai"
        assert _complete(gateway).served_by == "claude"
    assert gateway.active_slot == "primary"
    assert _failover_lines(caplog)[-1] == (
        "llm-failover: from=backup(openai) to=primary(claude) status=429 "
        "code=insufficient_quota"
    )


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_stays_on_backup_then_retries_primary_after_window(tmp_path, caplog):
    clock = _Clock()
    primary = _Adapter(
        "claude", [KEY_UNUSABLE["anthropic-401"][0]], default=_ok("claude")
    )
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup, clock=clock, retry_after=3600)
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai"
        clock.now += 3599
        assert _complete(gateway).served_by == "openai"
        assert primary.calls == 1, "no primary traffic inside the window"
        clock.now += 1
        assert _complete(gateway).served_by == "claude"
    assert gateway.active_slot == "primary"
    assert _failover_lines(caplog)[-1] == (
        "llm-failover: from=backup(openai) to=primary(claude) status=200 "
        "code=primary_retry_ok"
    )


def test_primary_still_dead_after_window_resets_the_timer(tmp_path, caplog):
    clock = _Clock()
    primary = _Adapter("claude", default=KEY_UNUSABLE["anthropic-401"][0])
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup, clock=clock, retry_after=60)
    _complete(gateway)
    clock.now += 60
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai"
    assert primary.calls == 2 and gateway.active_slot == "backup"
    assert len(_failover_lines(caplog)) == 1
    clock.now += 59
    _complete(gateway)
    assert primary.calls == 2, "timer restarted at the failed retry"


@pytest.mark.parametrize(
    ("case", "logged"),
    [
        ("anthropic-529", "status=529 code=overloaded_error error=LLMRetryBudgetExhausted/InternalServerError"),
        ("anthropic-timeout", "status=- code=- error=LLMRetryBudgetExhausted/APITimeoutError"),
    ],
)
def test_failed_return_to_primary_is_served_by_backup(tmp_path, caplog, case, logged):
    clock = _Clock()
    dead = KEY_UNUSABLE["anthropic-401"][0]
    failing = [NEVER_SWITCH[case]] * FAST_POLICY.attempts
    primary = _Adapter("claude", [dead, *failing], default=_ok("claude"))
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup, clock=clock, retry_after=3600)
    assert _complete(gateway).served_by == "openai"
    clock.now += 3600
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai", "same call served by backup"
    assert (primary.calls, backup.calls) == (1 + FAST_POLICY.attempts, 2)
    assert gateway.active_slot == "backup"
    lines = _failover_lines(caplog)
    assert lines == [
        f"llm-failover: primary_retry_failed from=primary(claude) to=backup(openai) {logged}"
    ]
    assert [r.levelno for r in caplog.records if r.getMessage() in lines] == [logging.WARNING]
    clock.now += 3599
    assert _complete(gateway).served_by == "openai"
    assert primary.calls == 1 + FAST_POLICY.attempts, "window restarted at the failed retry"
    clock.now += 1
    assert _complete(gateway).served_by == "claude", "healthy primary takes back over"
    assert gateway.active_slot == "primary"


def test_only_one_request_retries_primary_at_a_time(tmp_path):
    clock = _Clock()
    primary = _Adapter("claude", default=KEY_UNUSABLE["anthropic-401"][0])
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup, clock=clock, retry_after=60)
    _complete(gateway)
    clock.now += 61
    first, first_probe = gateway._plan()
    second, second_probe = gateway._plan()
    assert (first, first_probe) == ([0, 1], True)
    assert (second, second_probe) == ([1, 0], False)


def test_concurrent_requests_all_served_by_backup(tmp_path, caplog):
    primary = _Adapter("claude", default=KEY_UNUSABLE["anthropic-401"][0])
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(_complete(gateway).served_by)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with caplog.at_level(logging.WARNING, logger=log.name):
        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == [] and results == ["openai"] * 16
    assert backup.calls == 16 and 1 <= primary.calls <= 16
    assert len(_failover_lines(caplog)) == primary.calls
    assert gateway.active_slot == "backup"


def test_usage_ledger_names_serving_slot_and_provider(tmp_path):
    primary = _Adapter("claude", default=KEY_UNUSABLE["openai-401"][0])
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _failover(tmp_path, primary, backup)
    _complete(gateway)
    rows = [
        json.loads(line)
        for line in (tmp_path / "deepseek_usage.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == 2
    assert (row["key_slot"], row["provider"], row["model"]) == (
        "backup",
        "openai",
        "openai-model",
    )
    assert row["conservative_cost_usd"] == pytest.approx((10 * 2.0 + 5 * 3.0) / 1e6)


def test_legacy_retry_loop_still_retries_quota_429():
    """Without a backup slot, a quota 429 keeps today's rate-limit retries."""
    quota = KEY_UNUSABLE["openai-429-quota"][0]
    adapter = _Adapter("deepseek", default=quota)
    dependencies = CompletionDependencies(
        adapter=adapter,
        concurrency_semaphore=threading.BoundedSemaphore(1),
        policy=FAST_POLICY,
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
        retry_delay=lambda exc, attempt: extract_retry_delay(exc, attempt, FAST_POLICY),
        sleep=lambda seconds: None,
    )
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        create_chat_completion(
            messages=MESSAGES,
            max_tokens=10,
            temperature=0.3,
            model="m",
            dependencies=dependencies,
        )
    assert adapter.calls == FAST_POLICY.attempts
