"""Deterministic provider-gateway policy, accounting, and retry orchestration."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class APIRetryPolicy:
    attempts: int
    monitor_window_seconds: int
    rate_limit_burst_threshold: int
    rate_limit_base_delay: float
    rate_limit_max_delay: float
    transient_error_max_delay: float


DEFAULT_RETRY_POLICY = APIRetryPolicy(
    attempts=6,
    monitor_window_seconds=60,
    rate_limit_burst_threshold=5,
    rate_limit_base_delay=12,
    rate_limit_max_delay=90,
    transient_error_max_delay=30,
)


@dataclass(frozen=True)
class APIUsagePrices:
    cache_hit_usd_per_million: float
    cache_miss_usd_per_million: float
    output_usd_per_million: float
    # Providers that bill prompt-cache writes separately (Claude: 1.25x input)
    # set this; ``None`` keeps the original miss-priced accounting.
    cache_write_usd_per_million: float | None = None


@dataclass
class APIRuntimeState:
    """Mutable process-local state owned by one provider gateway."""

    concurrency_semaphore: threading.BoundedSemaphore
    rate_limit_lock: threading.Lock = field(default_factory=threading.Lock)
    monitor_lock: threading.Lock = field(default_factory=threading.Lock)
    usage_lock: threading.Lock = field(default_factory=threading.Lock)
    resume_after: float = 0.0
    rate_limit_events: deque[float] = field(default_factory=deque)
    error_events: deque[float] = field(default_factory=deque)
    success_events: deque[float] = field(default_factory=deque)

    @classmethod
    def create(cls, max_concurrency: int) -> APIRuntimeState:
        return cls(threading.BoundedSemaphore(max_concurrency))


@dataclass(frozen=True)
class CompletionDependencies:
    adapter: Any
    concurrency_semaphore: Any
    policy: APIRetryPolicy
    logger: logging.Logger
    wait_for_slot: Callable[[], None]
    record_usage: Callable[[Any, str], dict | None]
    record_event: Callable[[str], dict]
    record_rate_limit: Callable[[float, Exception], None]
    is_rate_limit_error: Callable[[Exception], bool]
    is_retryable_error: Callable[[Exception], bool]
    retry_delay: Callable[[Exception, int], float]
    sleep: Callable[[float], None] = time.sleep
    # Key failover: errors that must reach the failover layer unretried
    # (e.g. a quota 429). Without a backup slot nothing is raised early.
    raise_immediately: Callable[[Exception], bool] = lambda exc: False


class LLMRetryBudgetExhausted(RuntimeError):
    """``create_chat_completion`` spent every attempt; ``last_error`` says why."""

    def __init__(self, message: str, *, last_error: Exception | None):
        super().__init__(message)
        self.last_error = last_error


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # A truncated response may never close the tag; the unclosed remainder is
    # reasoning noise, not translation output.
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)
    return text.strip()


def _anthropic_error_kind(exc: Exception) -> str | None:
    """Classify an ``anthropic`` SDK exception by type and status code.

    Returns ``"rate_limit"``, ``"transient"``, ``"fatal"`` or ``None`` when
    ``exc`` is not an Anthropic error. The SDK is looked up in
    ``sys.modules`` only: if it was never imported, no Anthropic exception can
    exist, so DeepSeek-only processes never load it.
    """
    anthropic = sys.modules.get("anthropic")
    if anthropic is None:
        return None
    if isinstance(exc, anthropic.APIConnectionError):  # incl. APITimeoutError
        return "transient"
    if isinstance(exc, anthropic.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return "rate_limit"
        if isinstance(status_code, int) and (
            status_code >= 500 or status_code in {408, 409, 425}
        ):
            return "transient"  # 500 api_error, 529 overloaded_error, ...
        return "fatal"  # 400/401/402/403/404/413/422: never retried
    return None


def is_rate_limit_error(exc: Exception) -> bool:
    anthropic_kind = _anthropic_error_kind(exc)
    if anthropic_kind is not None:
        return anthropic_kind == "rate_limit"
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    return (
        status_code == 429
        or re.search(r"(?<!\d)429(?!\d)", text) is not None
        or "rate_limit" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def extract_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    text = str(exc)
    patterns = (
        r"error code:\s*(\d+)",
        r"http/\d(?:\.\d)?\s+(\d+)",
        r"'http_code':\s*'(\d+)'",
        r'"http_code":\s*"(\d+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def is_retryable_api_error(exc: Exception) -> bool:
    anthropic_kind = _anthropic_error_kind(exc)
    if anthropic_kind is not None:
        return anthropic_kind == "transient"
    text = str(exc).lower()
    status_code = extract_status_code(exc)
    return (
        status_code in {408, 409, 425, 500, 502, 503, 504, 520, 521, 522, 524, 529}
        or "timeout" in text
        or "temporar" in text
        or "connection" in text
        or "service unavailable" in text
        or "server error" in text
        or "unknown status code" in text
    )


def _error_identifiers(exc: Exception) -> list[str]:
    """Provider error ``code``/``type`` strings (openai: attributes; anthropic: body)."""
    values: list[Any] = [getattr(exc, "code", None), getattr(exc, "type", None)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            values.extend((error.get("code"), error.get("type")))
    found: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in found:
            found.append(value)
    return found


def provider_error_code(exc: Exception) -> str:
    identifiers = _error_identifiers(exc)
    return identifiers[0] if identifiers else "-"


def key_unusable_reason(exc: Exception) -> tuple[int | None, str] | None:
    """``(status, code)`` when ``exc`` says this key cannot serve, else ``None``.

    Key-unusable: 401/402/403; a 400 about credit balance, billing or usage
    limits (Anthropic spend limit); a 429 whose code/type names quota or
    billing; an ordinary 429 that outlived the retry budget. Network errors,
    timeouts, 5xx, other 4xx and unusable output are never key-unusable.
    """
    if isinstance(exc, LLMRetryBudgetExhausted):
        last = exc.last_error
        if last is not None and is_rate_limit_error(last):
            return extract_status_code(last) or 429, "rate_limit_exhausted"
        return None
    status = extract_status_code(exc)
    identifiers = " ".join(_error_identifiers(exc)).lower()
    if status in {401, 402, 403}:
        return status, provider_error_code(exc)
    if status == 400:
        text = str(exc).lower()
        if (
            "credit balance" in text
            or "billing" in text
            or "usage limit" in text
            or "billing" in identifiers
        ):
            return status, provider_error_code(exc)
    if status == 429 and ("quota" in identifiers or "billing" in identifiers):
        return status, provider_error_code(exc)
    return None


def raise_key_unusable_immediately(exc: Exception) -> bool:
    return key_unusable_reason(exc) is not None


def prune_api_events(state: APIRuntimeState, now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    for events in (
        state.rate_limit_events,
        state.error_events,
        state.success_events,
    ):
        while events and events[0] < cutoff:
            events.popleft()


def record_api_event(
    state: APIRuntimeState,
    kind: str,
    window_seconds: int,
    *,
    now: float | None = None,
) -> dict:
    timestamp = time.time() if now is None else now
    with state.monitor_lock:
        prune_api_events(state, timestamp, window_seconds)
        event_streams = {
            "rate_limit": state.rate_limit_events,
            "error": state.error_events,
            "success": state.success_events,
        }
        events = event_streams.get(kind)
        if events is not None:
            events.append(timestamp)
        return {
            "rate_limit_count": len(state.rate_limit_events),
            "error_count": len(state.error_events),
            "success_count": len(state.success_events),
            "window_seconds": window_seconds,
        }


def extract_retry_delay(
    exc: Exception,
    attempt: int,
    policy: APIRetryPolicy,
) -> float:
    if _anthropic_error_kind(exc) is not None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        raw_after = headers.get("retry-after") if headers is not None else None
        if raw_after is not None:
            try:
                return min(max(float(raw_after), 0.0) + 1.0, policy.rate_limit_max_delay)
            except (TypeError, ValueError):
                pass
    text = str(exc)
    patterns = (
        r"retry after ([0-9]+(?:\.[0-9]+)?)s",
        r"try again in ([0-9]+(?:\.[0-9]+)?)s",
        r"after ([0-9]+(?:\.[0-9]+)?) seconds?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            return min(float(match.group(1)) + 1.0, policy.rate_limit_max_delay)
        except (TypeError, ValueError):
            continue
    return min(
        policy.rate_limit_base_delay * (2**attempt),
        policy.rate_limit_max_delay,
    )


def wait_for_api_slot(
    state: APIRuntimeState,
    logger: logging.Logger,
    *,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        with state.rate_limit_lock:
            resume_after = state.resume_after
        delay = resume_after - now()
        if delay <= 0:
            return
        sleep_for = min(delay, 5.0)
        logger.info("LLM API cooling down for %.1fs after rate limiting", sleep_for)
        sleep(sleep_for)


def record_rate_limit(
    state: APIRuntimeState,
    delay: float,
    exc: Exception,
    logger: logging.Logger,
    *,
    now: Callable[[], float] = time.time,
) -> None:
    resume_after = now() + delay
    with state.rate_limit_lock:
        state.resume_after = max(state.resume_after, resume_after)
    logger.warning(
        "LLM API rate limited; pausing new requests for %.1fs: %s",
        delay,
        exc,
    )


def usage_as_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    data = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "prompt_tokens_details",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            data[key] = value
    return data


def nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def completion_usage_record(
    response: Any,
    model: str,
    prices: APIUsagePrices,
    *,
    timestamp: float | None = None,
    extra_fields: dict | None = None,
) -> dict | None:
    raw = usage_as_dict(getattr(response, "usage", None))
    if not raw:
        return None

    prompt_tokens = nonnegative_int(raw.get("prompt_tokens"))
    completion_tokens = nonnegative_int(raw.get("completion_tokens"))
    total_tokens = nonnegative_int(
        raw.get("total_tokens"), prompt_tokens + completion_tokens
    )
    details = raw.get("prompt_tokens_details") or {}
    if not isinstance(details, dict) and hasattr(details, "model_dump"):
        try:
            details = details.model_dump()
        except Exception:
            details = {}
    if not isinstance(details, dict):
        details = {}

    cache_hit_tokens = nonnegative_int(
        raw.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))
    )
    explicit_miss = raw.get("prompt_cache_miss_tokens")
    cache_miss_tokens = (
        max(prompt_tokens - cache_hit_tokens, 0)
        if explicit_miss is None
        else nonnegative_int(explicit_miss)
    )
    cache_write_tokens = nonnegative_int(raw.get("cache_creation_input_tokens"))
    if prices.cache_write_usd_per_million is None:
        conservative_cost_usd = (
            cache_hit_tokens * prices.cache_hit_usd_per_million
            + cache_miss_tokens * prices.cache_miss_usd_per_million
            + completion_tokens * prices.output_usd_per_million
        ) / 1_000_000
    else:
        uncached_tokens = max(cache_miss_tokens - cache_write_tokens, 0)
        conservative_cost_usd = (
            cache_hit_tokens * prices.cache_hit_usd_per_million
            + uncached_tokens * prices.cache_miss_usd_per_million
            + cache_write_tokens * prices.cache_write_usd_per_million
            + completion_tokens * prices.output_usd_per_million
        ) / 1_000_000
    record = {
        "schema": 1,
        "ts": time.time() if timestamp is None else timestamp,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "conservative_cost_usd": round(conservative_cost_usd, 10),
    }
    # Schema 2 = schema 1 fields (same meaning) + provider metadata. Claude:
    # prompt_tokens = input + cache_write + cache_read; prompt_cache_hit_tokens
    # = cache reads; prompt_cache_miss_tokens = input + cache writes, of which
    # cache_creation_input_tokens were billed at the cache-write rate.
    # Slot mode adds ``key_slot`` (primary/backup) and the serving provider.
    ledger_fields = getattr(response, "ledger_fields", None)
    merged = dict(ledger_fields) if isinstance(ledger_fields, dict) else {}
    merged.update(extra_fields or {})
    if merged:
        record["schema"] = 2
        record["cache_creation_input_tokens"] = cache_write_tokens
        record.update(merged)
    return record


def record_completion_usage(
    response: Any,
    model: str,
    prices: APIUsagePrices,
    path: Path,
    state: APIRuntimeState,
    logger: logging.Logger,
    *,
    extra_fields: dict | None = None,
) -> dict | None:
    record = completion_usage_record(response, model, prices, extra_fields=extra_fields)
    if record is None:
        return None
    try:
        with state.usage_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Unable to record LLM token usage: %s", exc)
    return record


def create_chat_completion(
    *,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    model: str,
    dependencies: CompletionDependencies,
):
    policy = dependencies.policy
    last_error: Exception | None = None
    for attempt in range(policy.attempts):
        dependencies.wait_for_slot()
        retry_delay: float | None = None
        with dependencies.concurrency_semaphore:
            dependencies.wait_for_slot()
            try:
                response = dependencies.adapter.translate(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                dependencies.record_usage(response, model)
                stats = dependencies.record_event("success")
                if attempt > 0:
                    dependencies.logger.info(
                        "LLM API request recovered after retry; recent API window "
                        "stats: %s rate-limit, %s transient errors, %s successes "
                        "in %ss",
                        stats["rate_limit_count"],
                        stats["error_count"],
                        stats["success_count"],
                        stats["window_seconds"],
                    )
                return response
            except Exception as exc:
                last_error = exc
                if dependencies.raise_immediately(exc):
                    raise
                if dependencies.is_rate_limit_error(exc):
                    stats = dependencies.record_event("rate_limit")
                    delay = dependencies.retry_delay(exc, attempt)
                    if stats["rate_limit_count"] >= policy.rate_limit_burst_threshold:
                        dependencies.record_rate_limit(delay, exc)
                        dependencies.logger.warning(
                            "LLM API rate limit burst detected (%s times in %ss); "
                            "entering slow mode",
                            stats["rate_limit_count"],
                            stats["window_seconds"],
                        )
                    else:
                        dependencies.logger.warning(
                            "LLM API rate limit detected but not bursting yet "
                            "(%s/%s in %ss); retrying",
                            stats["rate_limit_count"],
                            policy.rate_limit_burst_threshold,
                            stats["window_seconds"],
                        )
                    retry_delay = min(delay, policy.transient_error_max_delay)
                elif dependencies.is_retryable_error(exc):
                    stats = dependencies.record_event("error")
                    retry_delay = min(
                        3 * (2**attempt), policy.transient_error_max_delay
                    )
                    dependencies.logger.warning(
                        "LLM API transient error; retrying "
                        "(attempt %s/%s, %s transient errors in %ss): %s",
                        attempt + 1,
                        policy.attempts,
                        stats["error_count"],
                        stats["window_seconds"],
                        exc,
                    )
                else:
                    raise
        if retry_delay is not None and attempt < policy.attempts - 1:
            dependencies.sleep(retry_delay)

    raise LLMRetryBudgetExhausted(
        f"LLM API request failed after {policy.attempts} attempts: {last_error}",
        last_error=last_error,
    )


class LLMGateway:
    """One provider adapter with its process-local pacing, retry and usage ledger.

    Key slots: ``state`` lets slots share the concurrency cap and ledger lock
    while keeping separate rate-limit pacing; ``ledger_fields`` tags every
    usage record; ``raise_immediately`` hands key-unusable errors to the
    failover layer unretried.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        model: str,
        prices: APIUsagePrices,
        usage_log: Path,
        max_concurrency: int,
        logger: logging.Logger,
        policy: APIRetryPolicy = DEFAULT_RETRY_POLICY,
        state: APIRuntimeState | None = None,
        ledger_fields: dict | None = None,
        raise_immediately: Callable[[Exception], bool] | None = None,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.prices = prices
        self.usage_log = Path(usage_log)
        self.policy = policy
        self.state = state or APIRuntimeState.create(max_concurrency)
        self.logger = logger
        self.ledger_fields = dict(ledger_fields or {})
        optional = {} if raise_immediately is None else {"raise_immediately": raise_immediately}
        self._dependencies = CompletionDependencies(
            adapter=adapter,
            concurrency_semaphore=self.state.concurrency_semaphore,
            policy=policy,
            logger=logger,
            wait_for_slot=partial(wait_for_api_slot, self.state, logger),
            record_usage=self.record_usage,
            record_event=partial(
                record_api_event,
                self.state,
                window_seconds=policy.monitor_window_seconds,
            ),
            record_rate_limit=partial(record_rate_limit, self.state, logger=logger),
            is_rate_limit_error=is_rate_limit_error,
            is_retryable_error=is_retryable_api_error,
            retry_delay=partial(extract_retry_delay, policy=policy),
            **optional,
        )

    def record_usage(self, response: Any, model: str) -> dict | None:
        return record_completion_usage(
            response,
            model,
            self.prices,
            self.usage_log,
            self.state,
            self.logger,
            extra_fields=self.ledger_fields,
        )

    def complete(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ):
        return create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=self.model if model is None else model,
            dependencies=self._dependencies,
        )


# ---------------------------------------------------------------------------
# Primary/backup key failover
# ---------------------------------------------------------------------------


class LLMKeysUnusableError(RuntimeError):
    """Every key slot answered key-unusable for one request.

    Not chained to the provider errors: OpenAI's 401 text echoes a masked key
    fragment, and tracebacks print chained causes. ``errors`` keeps them.
    """

    def __init__(self, message: str, *, errors: list[Exception]):
        super().__init__(message)
        self.errors = errors


@dataclass(frozen=True)
class LLMSlot:
    name: str  # "primary" | "backup"
    provider: str  # "claude" | "openai" | "deepseek"
    gateway: LLMGateway


class FailoverLLMGateway:
    """Key slots in order; each slot's ``LLMGateway`` stays the retry owner.

    A request runs on the active slot through the full ``api_runtime`` retry
    loop. Only a key-unusable outcome (:func:`key_unusable_reason`) re-issues
    the same request once on the next slot; each slot is tried at most once
    per request. After a switch the service stays on the new slot; once
    ``retry_primary_seconds`` have passed, the next real request (one at a
    time) goes to primary first. If primary does not serve that request, for
    any reason, the next slot serves it and the window restarts. No synthetic
    probe calls are made.

    With an ``alerter`` (``notify(kind, text)``), each request outcome is
    checked against the last alerted service state, so one change of state
    alerts once however many requests saw it: ``switched`` when a backup slot
    serves while the active slot is not primary, ``restored`` when primary
    serves while it is the active slot, ``all_dead`` when every slot answered
    key-unusable for one request. Both keys dead therefore never alerts
    ``switched``: no backup ever serves. A state counts as alerted only once
    ``notify`` handed its message on; while the throttle holds it back
    (``notify`` returns False), later requests in that state offer it again
    with ``repeat=True``, so it goes out once the interval allows.
    """

    _ALERT_KINDS = {"backup": "switched", "primary": "restored", "all_dead": "all_dead"}
    _SLOT_LABELS = {"primary": "主力", "backup": "备用"}

    def __init__(
        self,
        slots: list[LLMSlot],
        *,
        retry_primary_seconds: float,
        logger: logging.Logger,
        clock: Callable[[], float] = time.monotonic,
        alerter: Any | None = None,
    ) -> None:
        if len(slots) < 2:
            raise ValueError("FailoverLLMGateway needs at least two key slots")
        self.slots = tuple(slots)
        self.retry_primary_seconds = retry_primary_seconds
        self.logger = logger
        self.alerter = alerter
        self._clock = clock
        self._lock = threading.Lock()
        self._active = 0
        self._switched_at = 0.0
        self._probe_in_flight = False
        self._primary_reason: tuple[int | None, str] | None = None
        self._alert_state = "primary"
        self._alert_unsent = False
        self._alert_claim = 0

    @property
    def active_slot(self) -> str:
        with self._lock:
            return self.slots[self._active].name

    def _plan(self) -> tuple[list[int], bool]:
        """Slot order for one request, and whether it is the primary retry."""
        with self._lock:
            probe = (
                self._active != 0
                and not self._probe_in_flight
                and self._clock() - self._switched_at >= self.retry_primary_seconds
            )
            if probe:
                self._probe_in_flight = True
            first = 0 if probe else self._active
            order = [first] + [i for i in range(len(self.slots)) if i != first]
            return order, probe

    def _log_switch(self, source: int, target: int, status: Any, code: str) -> None:
        old, new = self.slots[source], self.slots[target]
        self.logger.warning(
            "llm-failover: from=%s(%s) to=%s(%s) status=%s code=%s",
            old.name,
            old.provider,
            new.name,
            new.provider,
            status if status is not None else "-",
            code,
        )

    def _switch(self, source: int, target: int, reason: tuple[int | None, str]) -> None:
        self._log_switch(source, target, *reason)
        with self._lock:
            self._active = target
            self._switched_at = self._clock()
            if source == 0:
                self._primary_reason = reason

    def _primary_retry_failed(self, target: int, exc: Exception) -> str:
        """Log a non-key failure of the primary retry and restart the window.

        Logs status, error code and exception class only, never provider text.
        Returns the failure entry for the all-slots error message.
        """
        cause = exc
        error = type(exc).__name__
        if isinstance(exc, LLMRetryBudgetExhausted) and exc.last_error is not None:
            cause = exc.last_error
            error = f"{error}/{type(cause).__name__}"
        status = extract_status_code(cause)
        code = provider_error_code(cause)
        old, new = self.slots[0], self.slots[target]
        self.logger.warning(
            "llm-failover: primary_retry_failed from=%s(%s) to=%s(%s) status=%s code=%s error=%s",
            old.name,
            old.provider,
            new.name,
            new.provider,
            status if status is not None else "-",
            code,
            error,
        )
        with self._lock:
            self._switched_at = self._clock()
        return f"{old.name}({old.provider}) retry failed HTTP {status or '-'} {code} {error}"

    def _restore_primary(self) -> None:
        with self._lock:
            source = self._active
            if source == 0:
                return
            self._active = 0
        self._log_switch(source, 0, 200, "primary_retry_ok")

    def _alert_after(
        self,
        served: int | None,
        key_failures: list[tuple[int, int | None, str]],
    ) -> None:
        """Alert once per change of service state; never raises or blocks.

        ``served`` is the slot that answered, ``None`` when every slot was
        key-unusable (``key_failures``: slot index, status, code). The state
        is decided and claimed under the lock; ``notify`` runs after it is
        released. A message not handed on leaves the state unsent.
        """
        if self.alerter is None:
            return
        handed = False
        claim = None
        try:
            with self._lock:
                if served is None:
                    state = "all_dead"
                elif served == 0:
                    state = "primary" if self._active == 0 else None
                else:
                    state = "backup" if self._active != 0 else None
                if state is None:
                    return
                repeat = state == self._alert_state
                if repeat and not self._alert_unsent:
                    return
                # This request carries the message; others in the state wait.
                self._alert_state = state
                self._alert_unsent = False
                self._alert_claim += 1
                claim = self._alert_claim
                active = self._active
                primary_reason = self._primary_reason
            kind = self._ALERT_KINDS[state]
            text = self._alert_text(kind, active, primary_reason, key_failures)
            if repeat:
                handed = self.alerter.notify(kind, text, repeat=True) is not False
            else:
                handed = self.alerter.notify(kind, text) is not False
        except Exception as exc:
            self.logger.warning("llm-alert: skipped error=%s", type(exc).__name__)
        finally:
            if claim is not None and not handed:
                # Held back or failed: the next request in this state offers it again.
                with self._lock:
                    if self._alert_claim == claim:
                        self._alert_unsent = True

    def _alert_text(
        self,
        kind: str,
        active: int,
        primary_reason: tuple[int | None, str] | None,
        key_failures: list[tuple[int, int | None, str]],
    ) -> str:
        """Alert body: slot providers, HTTP status and error code, no provider text."""
        primary = self.slots[0]
        if kind == "switched":
            status, code = primary_reason or (None, "-")
            return (
                f"主力 key（{primary.provider}）不可用（HTTP {status or '-'} {code}），"
                f"已改用备用（{self.slots[active].provider}）。"
                "翻译继续，请检查主力 key 或额度。"
            )
        if kind == "restored":
            return f"主力 key（{primary.provider}）已恢复，已切回主力。"
        named = []
        for index, status, code in sorted(key_failures, key=lambda item: item[0]):
            slot = self.slots[index]
            label = self._SLOT_LABELS.get(slot.name, slot.name)
            named.append((label, slot.provider, status, code))
        count = "两" if len(named) == 2 else str(len(named))
        lines = [
            "和".join(f"{label}（{provider}）" for label, provider, _, _ in named)
            + f"{count}把 key 都不可用，翻译请求失败。"
        ]
        lines.extend(
            f"{label}（{provider}）：HTTP {status or '-'} {code}"
            for label, provider, status, code in named
        )
        return "\n".join(lines)

    def complete(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ):
        order, probe = self._plan()
        failures: list[str] = []
        errors: list[Exception] = []
        key_failures: list[tuple[int, int | None, str]] = []
        try:
            for position, index in enumerate(order):
                slot = self.slots[index]
                try:
                    response = slot.gateway.complete(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model=model,
                    )
                except Exception as exc:
                    reason = key_unusable_reason(exc)
                    if reason is None:
                        # Only a return to primary falls through on a non-key error.
                        if not (probe and index == 0 and position + 1 < len(order)):
                            raise
                        failures.append(self._primary_retry_failed(order[position + 1], exc))
                        errors.append(exc)
                        continue
                    status, code = reason
                    failures.append(
                        f"{slot.name}({slot.provider}) HTTP {status or '-'} {code}"
                    )
                    errors.append(exc)
                    key_failures.append((index, status, code))
                    if position + 1 < len(order):
                        self._switch(index, order[position + 1], reason)
                    continue
                if probe and index == 0:
                    self._restore_primary()
                self._alert_after(index, key_failures)
                return response
        finally:
            if probe:
                with self._lock:
                    self._probe_in_flight = False
        if len(key_failures) == len(order):
            self._alert_after(None, key_failures)
        raise LLMKeysUnusableError(
            "all LLM key slots unusable: " + "; ".join(failures), errors=errors
        )
