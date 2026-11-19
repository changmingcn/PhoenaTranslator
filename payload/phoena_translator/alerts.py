"""Telegram messages: fire-and-forget, token-safe; key-failover alerts throttled.

``notify`` decides under a lock whether a kind may be sent (at most once per
``min_interval_seconds``; suppressed alerts are counted and reported in the
next sent message of that kind), then hands delivery to a daemon thread and
returns whether it did. ``send`` hands a finished message to the same delivery path with no
throttle and no label prefix (task-finished notices). Delivery is one HTTPS
attempt with a bounded timeout and no retry. Nothing here raises into the
caller.

The Telegram URL contains the bot token: log lines name the alert kind, the
HTTP status and the exception class only, never a URL or exception text.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Callable

TELEGRAM_TIMEOUT_SECONDS = 10.0


def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    *,
    timeout: float = TELEGRAM_TIMEOUT_SECONDS,
) -> None:
    """One ``sendMessage`` call; raises on any failure (callers log the class only)."""
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(
            {"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


class TelegramAlerter:
    """Per-kind throttled alerts and unthrottled sends; ``sender(text)`` raises on failure."""

    def __init__(
        self,
        *,
        sender: Callable[[str], None],
        label: str,
        min_interval_seconds: float,
        logger: logging.Logger,
        clock: Callable[[], float] = time.monotonic,
        log_prefix: str = "llm-alert",
    ) -> None:
        self._sender = sender
        self.label = label
        self.min_interval_seconds = min_interval_seconds
        self.logger = logger
        self._clock = clock
        self._log_prefix = log_prefix
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}
        self._suppressed: Counter[str] = Counter()

    @classmethod
    def for_telegram(
        cls,
        *,
        token: str,
        chat_id: str,
        label: str,
        min_interval_seconds: float,
        logger: logging.Logger,
        log_prefix: str = "llm-alert",
    ) -> TelegramAlerter:
        def sender(text: str) -> None:
            send_telegram_message(token, chat_id, text)

        return cls(
            sender=sender,
            label=label,
            min_interval_seconds=min_interval_seconds,
            logger=logger,
            log_prefix=log_prefix,
        )

    def notify(self, kind: str, text: str, *, repeat: bool = False) -> bool:
        """Queue ``【label】text`` unless ``kind`` is throttled; returns at once.

        Returns True when the message was handed to delivery, False when the
        throttle held it back or it could not be queued. ``repeat=True``
        offers again a message held back earlier: held back once more it is
        not counted again, and once sent it is not counted among the merged.
        """
        try:
            with self._lock:
                now = self._clock()
                last = self._last_sent.get(kind)
                if last is not None and now - last < self.min_interval_seconds:
                    if not repeat:
                        self._suppressed[kind] += 1
                    pending = self._suppressed[kind]
                    message = None
                else:
                    self._last_sent[kind] = now
                    pending = self._suppressed.pop(kind, 0)
                    if repeat and pending:
                        pending -= 1
                    message = f"【{self.label}】{text}"
                    if pending:
                        message += f"\n（上次提醒后同类事件另有 {pending} 次，已合并未发）"
            if message is None:
                if not repeat:
                    self.logger.info(
                        "%s: kind=%s throttled suppressed=%s",
                        self._log_prefix,
                        kind,
                        pending,
                    )
                return False
            self._start(kind, message, pending)
            return True
        except Exception as exc:
            self._log_failure(kind, exc)
            return False

    def send(self, kind: str, message: str) -> None:
        """Queue ``message`` exactly as given: no throttle, no label; returns at once."""
        try:
            self._start(kind, message, None)
        except Exception as exc:
            self._log_failure(kind, exc)

    def _start(self, kind: str, message: str, merged: int | None) -> None:
        threading.Thread(
            target=self._deliver,
            args=(kind, message, merged),
            name=f"{self._log_prefix}-{kind}",
            daemon=True,
        ).start()

    def _deliver(self, kind: str, message: str, merged: int | None) -> None:
        try:
            self._sender(message)
        except Exception as exc:
            self._log_failure(kind, exc)
            return
        try:
            if merged is None:
                self.logger.info("%s: kind=%s sent", self._log_prefix, kind)
            else:
                self.logger.info(
                    "%s: kind=%s sent merged=%s", self._log_prefix, kind, merged
                )
        except Exception:
            pass

    def _log_failure(self, kind: str, exc: Exception) -> None:
        status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        if isinstance(exc, urllib.error.HTTPError):
            try:
                exc.close()
            except Exception:
                pass
        try:
            self.logger.warning(
                "%s: kind=%s send failed status=%s error=%s",
                self._log_prefix,
                kind,
                status if status is not None else "-",
                type(exc).__name__,
            )
        except Exception:
            pass
