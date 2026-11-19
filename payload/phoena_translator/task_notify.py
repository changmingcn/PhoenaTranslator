"""One Telegram message when a translation task ends (opt-in, never in the way).

``TaskNotifier.task_finished`` turns the saved final task record into a plain
text message and hands it to :class:`TelegramAlerter` ``send`` (background
thread, bounded timeout, no retry, no throttle). Formatting and queueing are
wrapped so that nothing here raises into the task runner, and nothing here
writes the task record.

A succeeded message names the file and what the record already holds (pages,
pages kept in the source language, elapsed time). A failed message names the
file and the reason: a short plain statement of the failure class when the
code can tell it apart, then the recorded error, cut to 300 characters with
key and token shapes removed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from typing import Any, Iterable, Mapping

from phoena_translator.alerts import TelegramAlerter
from phoena_translator.api_runtime import LLMKeysUnusableError
from phoena_translator.epub.archive import UnsafeEPUBArchive
from phoena_translator.task_runtime import FINAL_STATUSES

TELEGRAM_MAX_CHARS = 4096
MAX_FILENAME_CHARS = 120
MAX_REASON_CHARS = 300
MAX_LISTED_PAGES = 20
REDACTED = "[已隐去]"

# Key and token shapes that may ride along in a provider error text: OpenAI,
# Anthropic and DeepSeek keys (masked ones too), Telegram bot tokens, bearer
# headers, and the Telegram API URL that embeds the bot token.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_*.\-]{4,}"),
    re.compile(r"\bbot\d{5,}:[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b\d{5,}:[A-Za-z0-9_\-]{30,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]+=*"),
)
_TRACEBACK_MARK = "Traceback (most recent call last)"

_KEYS_DEAD = "主力和备用模型 key 都不可用"
_ALL_SOURCE = "所有页都没能译出，退回原文，未交付译文"
_UNPARSABLE = "文件无法解析"
_INTERRUPTED = "任务被中断"
_INTERNAL = "程序内部错误"

_KEYS_DEAD_MARK = "all llm key slots unusable"
_ALL_SOURCE_MARK = "every planned page fell back to the english source"
_SEGMENT_BUDGET_MARK = "did not pass validation"
_INTERRUPTED_MARKS = (
    "upload file lost after restart",
    "interpreter shutdown",
)
_INTERNAL_MARK = "unhandled worker error"
_UNPARSABLE_MARKS = (
    "failed to open",
    "cannot open",
    "filedataerror",
    "no objects found",
    "broken document",
    "not a pdf",
    "pdf密码错误",
    "epub contains no xhtml",
    "file is not a zip file",
    "epub mimetype",
    "archive member",
    "archive path",
)


def _scrub(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _single_line(text: str) -> str:
    """Control characters out (no forged lines), whitespace runs collapsed."""
    cleaned = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in str(text)
    )
    return " ".join(cleaned.split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def display_filename(record: Mapping[str, Any]) -> str:
    """The uploaded name on one line, middle-cut so the extension stays visible."""
    name = _single_line(record.get("filename") or "") or "（未记录文件名）"
    if len(name) <= MAX_FILENAME_CHARS:
        return name
    tail = 30
    return name[: MAX_FILENAME_CHARS - tail - 1] + "…" + name[-tail:]


def _texts(record: Mapping[str, Any]) -> list[str]:
    """Recorded failure texts besides ``error``: page fallbacks, segment reasons."""
    texts = []
    for field in ("source_page_fallbacks", "failure_reasons"):
        for entry in record.get(field) or []:
            if isinstance(entry, Mapping):
                texts.append(str(entry.get("reason") or ""))
                texts.append(str(entry.get("error_type") or ""))
    return texts


def failure_class(
    record: Mapping[str, Any], error: BaseException | None = None
) -> str | None:
    """The plain statement of a failure class the code can tell apart, if any."""
    message = str(record.get("error") or "")
    if error is not None:
        message += " " + str(error) + " " + type(error).__name__
    lowered = message.lower()
    # Dead keys first: a worker crash can carry the keys-unusable text too.
    if isinstance(error, LLMKeysUnusableError) or _KEYS_DEAD_MARK in lowered:
        return _KEYS_DEAD
    # A service stop or restart mid-task, before the generic crash label.
    if any(mark in lowered for mark in _INTERRUPTED_MARKS):
        return _INTERRUPTED
    if _INTERNAL_MARK in lowered:
        return _INTERNAL
    # Pages or segments that fell back carry their own reasons: dead keys
    # there are the root cause of an all-pages or over-budget failure.
    keys_dead_below = any(
        _KEYS_DEAD_MARK in text.lower() or text == "LLMKeysUnusableError"
        for text in _texts(record)
    )
    if _ALL_SOURCE_MARK in lowered:
        return _KEYS_DEAD if keys_dead_below else _ALL_SOURCE
    if _SEGMENT_BUDGET_MARK in lowered and keys_dead_below:
        return _KEYS_DEAD
    if (
        isinstance(error, (UnsafeEPUBArchive, zipfile.BadZipFile))
        or (record.get("type") == "epub" and record.get("failed_files"))
        or any(mark in lowered for mark in _UNPARSABLE_MARKS)
    ):
        return _UNPARSABLE
    return None


def failure_reason(
    record: Mapping[str, Any],
    error: BaseException | None = None,
    *,
    secrets: Iterable[str] = (),
) -> str:
    """Class statement, then the recorded message; 300 characters at most."""
    recorded = str(record.get("error") or "")
    if not recorded.strip() and error is not None:
        recorded = str(error)
    recorded = recorded.split(_TRACEBACK_MARK, 1)[0]
    recorded = _single_line(_scrub(recorded, secrets))
    statement = failure_class(record, error)
    if statement and recorded:
        reason = f"{statement}。{recorded}"
    else:
        reason = statement or recorded or "未记录"
    return _cut(reason, MAX_REASON_CHARS)


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _elapsed(record: Mapping[str, Any]) -> str | None:
    try:
        seconds = float(record["updated_at"]) - float(record["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        text = f"{hours}小时{minutes}分"
    elif minutes:
        text = f"{minutes}分{secs}秒"
    else:
        text = f"{secs}秒"
    return f"用时：{text}（自提交起，含排队）"


def _pages_text(pages: list[int]) -> str:
    listed = "、".join(str(page) for page in pages[:MAX_LISTED_PAGES])
    more = "等" if len(pages) > MAX_LISTED_PAGES else ""
    return f"第 {listed}{more} 页"


def _success_details(record: Mapping[str, Any]) -> list[str]:
    lines = []
    kind = str(record.get("type") or "")
    total = _count(record.get("total"))
    if kind == "pdf":
        if total:
            lines.append(f"页数：{total}")
        pages = sorted(
            {
                _count(entry.get("page"))
                for entry in record.get("source_page_fallbacks") or []
                if isinstance(entry, Mapping) and _count(entry.get("page")) > 0
            }
        )
        if pages:
            lines.append(f"保留原文的页：{len(pages)} 页（{_pages_text(pages)}）")
        elements = len(record.get("element_source_fallbacks") or [])
        if elements:
            lines.append(f"另有 {elements} 处元素保留原文")
    elif kind == "epub":
        if total:
            lines.append(f"章节文件：{total} 个")
        segments = _count(record.get("total_segments"))
        if segments:
            translated = _count(record.get("translated_segments"))
            lines.append(f"段落：共 {segments} 段，译出 {translated} 段")
        fallback = _count(record.get("fallback_segment_count"))
        if fallback:
            lines.append(f"保留原文段：{fallback} 段")
        lenient = _count(record.get("lenient_segment_count"))
        if lenient:
            lines.append(f"质量存疑段：{lenient} 段")
    elif total:
        lines.append(f"进度：{_count(record.get('progress'))}/{total}")
    elapsed = _elapsed(record)
    if elapsed:
        lines.append(elapsed)
    return lines


def format_task_message(
    label: str,
    record: Mapping[str, Any],
    error: BaseException | None = None,
    *,
    secrets: Iterable[str] = (),
) -> str:
    """Plain-text message for a final record (no ``parse_mode``)."""
    name = display_filename(record)
    if record.get("status") == "completed":
        lines = [f"✅【{label}】翻译成功：{name}", *_success_details(record)]
    else:
        lines = [
            f"❌【{label}】翻译失败：{name}",
            f"失败原因：{failure_reason(record, error, secrets=secrets)}",
        ]
    return _cut("\n".join(lines), TELEGRAM_MAX_CHARS)


class TaskNotifier:
    """Sends one message per finished task through an alerter's ``send``."""

    def __init__(
        self,
        *,
        alerter: TelegramAlerter,
        label: str,
        logger: logging.Logger,
        secrets: Iterable[str] = (),
    ) -> None:
        self.alerter = alerter
        self.label = label
        self.logger = logger
        self._secrets = tuple(secret for secret in secrets if secret)

    def task_finished(
        self,
        task_id: str,
        record: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> None:
        """Queue the message for a final record; never raises, never blocks."""
        try:
            status = record.get("status")
            if status not in FINAL_STATUSES:
                return
            kind = "task_done" if status == "completed" else "task_failed"
            message = format_task_message(
                self.label, record, error, secrets=self._secrets
            )
            self.alerter.send(kind, message)
            self.logger.info("task-notify: [%s] kind=%s queued", task_id, kind)
        except Exception as exc:
            try:
                self.logger.warning(
                    "task-notify: [%s] skipped error=%s", task_id, type(exc).__name__
                )
            except Exception:
                pass


def build_task_notifier(config: Any, logger: logging.Logger) -> TaskNotifier | None:
    """A notifier when the channel is set and NOTIFY_DONE is on, else ``None``."""
    if not config.notify_done_enabled:
        return None
    alerter = TelegramAlerter.for_telegram(
        token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        label=config.telegram_label,
        min_interval_seconds=0.0,
        logger=logger,
        log_prefix="task-notify",
    )
    logger.info(
        "Task finished notifications: telegram on, label=%s", config.telegram_label
    )
    return TaskNotifier(
        alerter=alerter,
        label=config.telegram_label,
        logger=logger,
        secrets=(
            config.telegram_bot_token,
            config.llm_primary_api_key,
            config.llm_backup_api_key,
            config.anthropic_api_key,
            config.deepseek_api_key,
        ),
    )


__all__ = [
    "FINAL_STATUSES",
    "TaskNotifier",
    "build_task_notifier",
    "display_filename",
    "failure_class",
    "failure_reason",
    "format_task_message",
]
