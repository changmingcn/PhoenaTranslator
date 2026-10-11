"""LLM network adapters: DeepSeek (OpenAI-compatible SDK) and Claude (anthropic SDK)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


class LLMConfigurationError(RuntimeError):
    """Raised when semantic translation is requested without runtime credentials."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    # Retry policy is owned exclusively by ``api_runtime``; the SDK's built-in
    # retries would multiply attempts and hide 429s from burst accounting.
    max_retries: int = 0


class DeepSeekTranslationAdapter:
    """Lazy adapter whose sole public operation is semantic translation."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client_factory: Callable[..., Any] = OpenAI,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self._client_lock = threading.Lock()

    def _client(self) -> Any:
        if not self._settings.api_key:
            raise LLMConfigurationError(
                "DEEPSEEK_API_KEY is required when semantic translation is requested"
            )
        if self._client_instance is None:
            with self._client_lock:
                if self._client_instance is None:
                    self._client_instance = self._client_factory(
                        api_key=self._settings.api_key,
                        base_url=self._settings.base_url,
                        timeout=self._settings.timeout_seconds,
                        max_retries=self._settings.max_retries,
                    )
        return self._client_instance

    def translate(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ) -> Any:
        """Generate a semantic translation through the configured provider."""
        return self._client().chat.completions.create(
            model=model or self._settings.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )



# ---------------------------------------------------------------------------
# Native Claude backend (official ``anthropic`` SDK, Messages API)
# ---------------------------------------------------------------------------
#
# Selected with ``TRANSLATOR_LLM_PROVIDER=claude``. The adapter keeps the
# ``translate(messages=, max_tokens=, temperature=, model=)`` signature and
# returns a small OpenAI-shaped result (``choices[0].message.content`` and a
# ``usage`` dict) so ``pdf.translation`` and ``api_runtime`` keep a single
# contract. Request rules (current API):
#
# * ``temperature``/``top_p`` are never sent (400 on all 5.5 models); the
#   caller's ``temperature`` argument is accepted and ignored.
# * claude-opus-5-5: thinking cannot be disabled -> omit ``thinking`` and send
#   ``output_config.effort`` (default ``low``).
# * claude-sonnet-5-5: thinking off = ``{"type": "between_tools"}`` (no other
#   field; effort ``high`` or below).
# * claude-haiku-5-5: thinking off = ``{"type": "disabled"}`` (effort ``high``
#   or below); no server-side refusal fallback.
# * Opus/Sonnet 5.5 opt into ``fallbacks="default"`` (beta
#   ``server-side-fallback-2026-07-01``) unless disabled by configuration.
# * The SDK's own retries stay off (``max_retries=0``): ``api_runtime`` is the
#   single retry owner and classifies Anthropic exception classes itself.

CLAUDE_OPUS_5_5 = "claude-opus-5-5"
CLAUDE_SONNET_5_5 = "claude-sonnet-5-5"
CLAUDE_HAIKU_5_5 = "claude-haiku-5-5"
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLAUDE_FALLBACK_FAMILIES = frozenset({CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5})
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# ``between_tools`` / ``disabled`` thinking is a 400 above effort ``high``.
CLAUDE_THINKING_OFF_EFFORTS = frozenset({"low", "medium", "high"})
# Thinking tokens count toward ``max_tokens``; never send less than this.
CLAUDE_MIN_MAX_TOKENS = 16_000
CLAUDE_MAX_OUTPUT_TOKENS = 128_000
CLAUDE_OK_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})


class ClaudeResponseError(RuntimeError):
    """A completed Claude call whose output must not be used as a translation.

    Deliberately not transient for ``api_runtime`` (no retry there); the
    translation ladder in ``pdf.translation`` treats it like any other failed
    attempt and the page-level fallback owns the final outcome.
    """

    def __init__(self, message: str, *, stop_reason: str | None, model: str | None):
        super().__init__(message)
        self.stop_reason = stop_reason
        self.model = model


def claude_model_family(model: str) -> str:
    """Return the known 5.5 family id for ``model`` (or ``""`` if unknown)."""
    for family in (CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5, CLAUDE_HAIKU_5_5):
        if model == family or model.startswith(family + "-"):
            return family
    return ""


@dataclass(frozen=True)
class ClaudeSettings:
    api_key: str = field(repr=False)
    model: str = CLAUDE_OPUS_5_5
    effort: str | None = None
    fallbacks: bool = True
    prompt_cache: bool = True
    timeout_seconds: float = 300.0
    # See LLMSettings.max_retries: api_runtime owns retries.
    max_retries: int = 0
    min_max_tokens: int = CLAUDE_MIN_MAX_TOKENS


def build_claude_request(
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    effort: str | None = None,
    prompt_cache: bool = True,
    min_max_tokens: int = CLAUDE_MIN_MAX_TOKENS,
    logger: Any = None,
) -> dict:
    """Map an OpenAI-style message list to Messages API kwargs (no network)."""
    system_parts = [
        str(message["content"])
        for message in messages
        if message.get("role") == "system" and message.get("content")
    ]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": min(
            max(int(max_tokens), int(min_max_tokens)), CLAUDE_MAX_OUTPUT_TOKENS
        ),
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message.get("role") != "system"
        ],
    }
    if system_parts:
        system_text = "\n\n".join(system_parts)
        if prompt_cache:
            # Below the model's minimum cacheable length this marker is a
            # silent no-op; above it, repeated pages of one document read the
            # shared system prompt (incl. glossary) from cache.
            params["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            params["system"] = system_text

    family = claude_model_family(model)
    if family == CLAUDE_OPUS_5_5:
        params["output_config"] = {"effort": effort or "low"}
    elif family in {CLAUDE_SONNET_5_5, CLAUDE_HAIKU_5_5}:
        if effort is None or effort in CLAUDE_THINKING_OFF_EFFORTS:
            params["thinking"] = (
                {"type": "between_tools"}
                if family == CLAUDE_SONNET_5_5
                else {"type": "disabled"}
            )
            if effort:
                params["output_config"] = {"effort": effort}
        else:
            if logger is not None:
                logger.warning(
                    "CLAUDE_EFFORT=%s cannot run with thinking off on %s; "
                    "using adaptive thinking",
                    effort,
                    model,
                )
            params["output_config"] = {"effort": effort}
    elif effort:
        params["output_config"] = {"effort": effort}
    return params


@dataclass(frozen=True)
class CompletionMessage:
    content: str
    role: str = "assistant"


@dataclass(frozen=True)
class CompletionChoice:
    message: CompletionMessage
    finish_reason: str | None
    index: int = 0


@dataclass(frozen=True)
class ClaudeCompletion:
    """OpenAI-shaped view of one Claude Messages API response.

    Exposes exactly what callers read: ``choices[0].message.content`` and a
    ``usage`` dict in the ledger's field names. ``ledger_fields`` are merged
    into the usage ledger record (schema 2).
    """

    choices: list[CompletionChoice]
    usage: dict
    model: str
    requested_model: str
    stop_reason: str | None
    fallback_served: bool = False
    id: str | None = None
    provider: str = "claude"

    @property
    def ledger_fields(self) -> dict:
        fields: dict[str, Any] = {"provider": self.provider}
        if self.model and self.model != self.requested_model:
            fields["served_model"] = self.model
        if self.fallback_served:
            fields["fallback_served"] = True
        return fields


def _claude_usage_dict(usage: Any) -> dict:
    def tokens(name: str) -> int:
        try:
            return max(int(getattr(usage, name, 0) or 0), 0)
        except (TypeError, ValueError):
            return 0

    input_tokens = tokens("input_tokens")
    output_tokens = tokens("output_tokens")
    cache_write = tokens("cache_creation_input_tokens")
    cache_read = tokens("cache_read_input_tokens")
    prompt_tokens = input_tokens + cache_write + cache_read
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_cache_hit_tokens": cache_read,
        "prompt_cache_miss_tokens": input_tokens + cache_write,
        "cache_creation_input_tokens": cache_write,
    }


def _default_anthropic_factory(**kwargs: Any) -> Any:
    # Imported lazily so DeepSeek-only deployments never load the SDK.
    import anthropic

    return anthropic.Anthropic(**kwargs)


class ClaudeTranslationAdapter:
    """Lazy Claude adapter with the same ``translate`` contract as DeepSeek."""

    provider = "claude"

    def __init__(
        self,
        settings: ClaudeSettings,
        *,
        client_factory: Callable[..., Any] = _default_anthropic_factory,
        logger: Any = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self._client_lock = threading.Lock()
        self._logger = logger

    def _client(self) -> Any:
        if not self._settings.api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is required when TRANSLATOR_LLM_PROVIDER=claude"
            )
        if self._client_instance is None:
            with self._client_lock:
                if self._client_instance is None:
                    self._client_instance = self._client_factory(
                        api_key=self._settings.api_key,
                        timeout=self._settings.timeout_seconds,
                        max_retries=self._settings.max_retries,
                    )
        return self._client_instance

    def build_request(self, *, messages: list[dict], max_tokens: int, model: str) -> dict:
        return build_claude_request(
            messages,
            model=model,
            max_tokens=max_tokens,
            effort=self._settings.effort,
            prompt_cache=self._settings.prompt_cache,
            min_max_tokens=self._settings.min_max_tokens,
            logger=self._logger,
        )

    def uses_fallbacks(self, model: str) -> bool:
        return (
            self._settings.fallbacks
            and claude_model_family(model) in CLAUDE_FALLBACK_FAMILIES
        )

    def translate(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ) -> ClaudeCompletion:
        """Generate a semantic translation through Claude.

        ``temperature`` is accepted for contract compatibility and never sent.
        """
        del temperature
        requested_model = model or self._settings.model
        params = self.build_request(
            messages=messages, max_tokens=max_tokens, model=requested_model
        )
        client = self._client()
        if self.uses_fallbacks(requested_model):
            response = client.beta.messages.create(
                betas=[CLAUDE_FALLBACK_BETA],
                fallbacks="default",
                **params,
            )
        else:
            response = client.messages.create(**params)
        return self._normalise(response, requested_model, params["max_tokens"])

    def _normalise(
        self, response: Any, requested_model: str, max_tokens: int
    ) -> ClaudeCompletion:
        served_model = getattr(response, "model", None) or requested_model
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise ClaudeResponseError(
                f"Claude refusal (stop_reason=refusal, category={category}, "
                f"model={served_model})",
                stop_reason=stop_reason,
                model=served_model,
            )
        if stop_reason == "max_tokens":
            raise ClaudeResponseError(
                f"Claude output truncated (stop_reason=max_tokens, "
                f"max_tokens={max_tokens}, model={served_model})",
                stop_reason=stop_reason,
                model=served_model,
            )
        if stop_reason not in CLAUDE_OK_STOP_REASONS:
            raise ClaudeResponseError(
                f"Claude unexpected stop_reason={stop_reason} (model={served_model})",
                stop_reason=stop_reason,
                model=served_model,
            )
        blocks = list(getattr(response, "content", None) or [])
        fallback_served = any(getattr(b, "type", "") == "fallback" for b in blocks)
        if fallback_served and self._logger is not None:
            self._logger.warning(
                "Claude refusal fallback served request: requested=%s served=%s",
                requested_model,
                served_model,
            )
        text = "".join(
            getattr(block, "text", "") or ""
            for block in blocks
            if getattr(block, "type", "") == "text"
        )
        return ClaudeCompletion(
            choices=[
                CompletionChoice(
                    message=CompletionMessage(content=text),
                    finish_reason=stop_reason,
                )
            ],
            usage=_claude_usage_dict(getattr(response, "usage", None)),
            model=served_model,
            requested_model=requested_model,
            stop_reason=stop_reason,
            fallback_served=fallback_served,
            id=getattr(response, "id", None),
        )
