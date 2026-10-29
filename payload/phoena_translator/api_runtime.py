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
    ledger_fields = getattr(response, "ledger_fields", None)
    if isinstance(ledger_fields, dict) and ledger_fields:
        record["schema"] = 2
        record["cache_creation_input_tokens"] = cache_write_tokens
        record.update(ledger_fields)
    return record


def record_completion_usage(
    response: Any,
    model: str,
    prices: APIUsagePrices,
    path: Path,
    state: APIRuntimeState,
    logger: logging.Logger,
) -> dict | None:
    record = completion_usage_record(response, model, prices)
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

    raise RuntimeError(
        f"LLM API request failed after {policy.attempts} attempts: {last_error}"
    )
