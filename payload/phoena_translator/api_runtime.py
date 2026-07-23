"""Deterministic provider-gateway policy, accounting, and retry orchestration."""

from __future__ import annotations

import json
import logging
import re
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


def is_rate_limit_error(exc: Exception) -> bool:
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
        logger.info("DeepSeek cooling down for %.1fs after rate limiting", sleep_for)
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
        "DeepSeek rate limited; pausing new requests for %.1fs: %s",
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
    conservative_cost_usd = (
        cache_hit_tokens * prices.cache_hit_usd_per_million
        + cache_miss_tokens * prices.cache_miss_usd_per_million
        + completion_tokens * prices.output_usd_per_million
    ) / 1_000_000
    return {
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
        logger.warning("Unable to record DeepSeek token usage: %s", exc)
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
                        "DeepSeek request recovered after retry; recent API window "
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
                            "DeepSeek rate limit burst detected (%s times in %ss); "
                            "entering slow mode",
                            stats["rate_limit_count"],
                            stats["window_seconds"],
                        )
                    else:
                        dependencies.logger.warning(
                            "DeepSeek rate limit detected but not bursting yet "
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
                        "DeepSeek transient API error; retrying "
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
        f"DeepSeek API request failed after {policy.attempts} attempts: {last_error}"
    )
