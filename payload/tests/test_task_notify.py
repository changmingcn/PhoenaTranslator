"""Telegram message per finished task: one per task, opt-in, never in the way.

No network: message and runner tests use a recording ``send``, alerter tests a
stub sender, and the HTTPS path a patched ``urlopen``.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import io
import logging
import threading
import time
import urllib.error

import pytest

from phoena_translator.alerts import TelegramAlerter
from phoena_translator.config import AppConfig
from phoena_translator.runtime import Runtime
from phoena_translator.task_notify import (
    MAX_FILENAME_CHARS,
    TELEGRAM_MAX_CHARS,
    TaskNotifier,
    build_task_notifier,
    failure_reason,
    format_task_message,
)
from phoena_translator.task_runtime import (
    TaskRecoveryDependencies,
    TranslationJobDependencies,
    resume_interrupted_tasks,
    run_translation_job,
)
from phoena_translator.task_store import TaskRepository, TaskStore
from phoena_translator.workers import TaskJob

log = logging.getLogger("test.task_notify")

# Filler values in token shape, never real credentials.
FAKE_TOKEN = "123456789:AAfiller-not-a-real-bot-token-000000"
FAKE_CHAT = "test-chat-filler"
FAKE_LLM_KEY = "sk-filler-not-a-real-llm-key-0000000000"
CHANNEL = {
    "TRANSLATOR_TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
    "TRANSLATOR_TELEGRAM_CHAT_ID": FAKE_CHAT,
}
NOTIFY_ON = {**CHANNEL, "TRANSLATOR_TELEGRAM_NOTIFY_DONE": "1"}
ODD_NAME = "a_b*c[d](e)_*[]().pdf"
T0 = 1_700_000_000.0


class _Sent:
    """Synchronous ``send`` stand-in: records every call."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def send(self, kind, message):
        self.calls.append((kind, message))


def _notifier(sent=None, label="PhoenaTranslator"):
    sent = sent or _Sent()
    return TaskNotifier(alerter=sent, label=label, logger=log, secrets=(FAKE_TOKEN,)), sent


def _repo(tmp_path):
    store = TaskStore(tmp_path / "progress", tmp_path / "uploads", tmp_path / "outputs")
    return TaskRepository(store)


def _job(task_id="t1", extension="pdf"):
    return TaskJob(task_id=task_id, extension=extension, source_path="in", output_path="out")


def _deps(repo, notifier, *, pdf=None, epub=None):
    def unused(*args, **kwargs):
        raise AssertionError("translation must not run")

    return TranslationJobDependencies(
        logger=log,
        load_progress=repo.load,
        translate_epub=epub or unused,
        translate_pdf=pdf or unused,
        task_finished=notifier.task_finished if notifier else None,
    )


def _saver(repo, task_id, final):
    def translate(*args, **kwargs):
        repo.save(task_id, {**(repo.load(task_id) or {}), "status": "translating"})
        repo.save(task_id, {**(repo.load(task_id) or {}), **final})

    return translate


def _pdf_done(**extra):
    return {
        "status": "completed",
        "type": "pdf",
        "filename": "report.pdf",
        "total": 12,
        "progress": 12,
        "created_at": T0,
        "updated_at": T0 + 125,
        **extra,
    }


def _failed(error, **extra):
    return {"status": "failed", "type": "pdf", "filename": "report.pdf", "error": error, **extra}


def _all_messages(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Configuration and wiring
# ---------------------------------------------------------------------------


def test_notify_done_is_off_by_default_and_on_by_config(tmp_path):
    assert not AppConfig.from_env(CHANNEL, home=tmp_path).notify_done_enabled
    config = AppConfig.from_env(NOTIFY_ON, home=tmp_path)
    assert config.notify_done_enabled and config.key_alerts_enabled
    assert FAKE_TOKEN not in repr(config) and FAKE_CHAT not in repr(config)


@pytest.mark.parametrize(
    "env",
    [
        CHANNEL,  # switch off
        {"TRANSLATOR_TELEGRAM_NOTIFY_DONE": "1"},  # channel unset
        {"TRANSLATOR_TELEGRAM_NOTIFY_DONE": "1", "TRANSLATOR_TELEGRAM_BOT_TOKEN": FAKE_TOKEN},
        {"TRANSLATOR_TELEGRAM_NOTIFY_DONE": "1", "TRANSLATOR_TELEGRAM_CHAT_ID": FAKE_CHAT},
        {**NOTIFY_ON, "TRANSLATOR_TELEGRAM_DISABLE": "1"},
    ],
)
def test_no_notifier_when_off_unset_or_disabled(tmp_path, env):
    config = AppConfig.from_env(env, home=tmp_path)
    assert build_task_notifier(config, log) is None
    assert Runtime(config).task_notifier is None


def test_runtime_builds_notifier_and_logs_no_secret(tmp_path, caplog):
    config = AppConfig.from_env(
        {**NOTIFY_ON, "TRANSLATOR_TELEGRAM_LABEL": "soft", "LLM_PRIMARY_API_KEY": FAKE_LLM_KEY},
        home=tmp_path,
    )
    with caplog.at_level(logging.DEBUG):
        runtime = Runtime(config)
    assert isinstance(runtime.task_notifier, TaskNotifier)
    assert runtime.task_notifier.label == "soft"
    text = _all_messages(caplog)
    assert "Task finished notifications: telegram on, label=soft" in text
    assert FAKE_TOKEN not in text and FAKE_CHAT not in text


def test_key_alerts_switch_does_not_touch_done_notifications(tmp_path):
    config = AppConfig.from_env(
        {**NOTIFY_ON, "TRANSLATOR_TELEGRAM_KEY_ALERTS": "0"}, home=tmp_path
    )
    assert not config.key_alerts_enabled and config.notify_done_enabled


# ---------------------------------------------------------------------------
# Message text
# ---------------------------------------------------------------------------


def test_completed_pdf_message():
    message = format_task_message("PhoenaTranslator", _pdf_done())
    assert message == (
        "✅【PhoenaTranslator】翻译成功：report.pdf\n"
        "页数：12\n"
        "用时：2分5秒（自提交起，含排队）"
    )


def test_completed_pdf_with_source_pages_and_elements():
    record = _pdf_done(
        source_page_fallbacks=[{"page": 7, "stage": "render"}, {"page": 3, "stage": "translation"}],
        element_source_fallbacks=[{"page": 2}, {"page": 4}],
        created_at=T0,
        updated_at=T0 + 3725,
    )
    assert format_task_message("soft", record).splitlines() == [
        "✅【soft】翻译成功：report.pdf",
        "页数：12",
        "保留原文的页：2 页（第 3、7 页）",
        "另有 2 处元素保留原文",
        "用时：1小时2分（自提交起，含排队）",
    ]


def test_completed_epub_message_uses_recorded_counts_only():
    record = {
        "status": "completed",
        "type": "epub",
        "filename": "book.epub",
        "total": 9,
        "total_segments": 400,
        "translated_segments": 395,
        "fallback_segment_count": 3,
        "lenient_segment_count": 0,
    }
    assert format_task_message("soft", record).splitlines() == [
        "✅【soft】翻译成功：book.epub",
        "章节文件：9 个",
        "段落：共 400 段，译出 395 段",
        "保留原文段：3 段",
    ]


def test_completed_without_details_is_one_line():
    assert format_task_message("soft", {"status": "completed", "filename": "x.pdf"}) == (
        "✅【soft】翻译成功：x.pdf"
    )


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (
            _failed(
                "Translation failed after 4 retries: all LLM key slots unusable: "
                "primary(claude) HTTP 401 authentication_error; backup(openai) HTTP 401 invalid_api_key"
            ),
            "主力和备用模型 key 都不可用。Translation failed after 4 retries: all LLM key slots "
            "unusable: primary(claude) HTTP 401 authentication_error; backup(openai) HTTP 401 "
            "invalid_api_key",
        ),
        (
            _failed(
                "PDF page translation failed after one clean retry (1 page(s)): page 1: every "
                "planned page fell back to the English source; refusing to mark an untranslated "
                "copy as completed",
                source_page_fallbacks=[
                    {"page": 1, "reason": "Translation failed after 4 retries: all LLM key slots unusable: x"}
                ],
            ),
            "主力和备用模型 key 都不可用。PDF page translation failed after one clean retry "
            "(1 page(s)): page 1: every planned page fell back to the English source; refusing to "
            "mark an untranslated copy as completed",
        ),
        (
            _failed(
                "PDF page translation failed after one clean retry (1 page(s)): page 1: every "
                "planned page fell back to the English source; refusing to mark an untranslated "
                "copy as completed",
                source_page_fallbacks=[{"page": 1, "reason": "Translation failed: timeout"}],
            ),
            "所有页都没能译出，退回原文，未交付译文。PDF page translation failed after one clean "
            "retry (1 page(s)): page 1: every planned page fell back to the English source; "
            "refusing to mark an untranslated copy as completed",
        ),
        (_failed("Failed to open file 'x'."), "文件无法解析。Failed to open file 'x'."),
        (_failed("PDF密码错误或未提供密码，无法打开加密PDF"), "文件无法解析。PDF密码错误或未提供密码，无法打开加密PDF"),
        (
            {"status": "failed", "type": "epub", "failed_files": ["c1.xhtml"], "error": "c1.xhtml: bad tag"},
            "文件无法解析。c1.xhtml: bad tag",
        ),
        (_failed("Upload file lost after restart"), "任务被中断。Upload file lost after restart"),
        (_failed("Unhandled worker error: boom"), "程序内部错误。Unhandled worker error: boom"),
        (
            _failed("Unhandled worker error: all LLM key slots unusable: primary(claude) HTTP 401 x"),
            "主力和备用模型 key 都不可用。Unhandled worker error: all LLM key slots unusable: "
            "primary(claude) HTTP 401 x",
        ),
        (
            _failed("Unhandled worker error: cannot schedule new futures after interpreter shutdown"),
            "任务被中断。Unhandled worker error: cannot schedule new futures after interpreter shutdown",
        ),
        (_failed("PDF structure validation failed (2 warning(s): overlap)"), "PDF structure validation failed (2 warning(s): overlap)"),
        (_failed(""), "未记录"),
        ({"status": "failed"}, "未记录"),
    ],
)
def test_failure_reason_classes(record, reason):
    message = format_task_message("PhoenaTranslator", record)
    first, second = message.split("\n")
    assert first.startswith("❌【PhoenaTranslator】翻译失败：")
    assert second == f"失败原因：{reason[:299] + '…' if len(reason) > 300 else reason}"


def test_failure_reason_is_cut_scrubbed_and_traceback_free():
    error = (
        f"Error code: 401 - Incorrect API key provided: sk-proj-****abcd. bot{FAKE_TOKEN} "
        f"Bearer abc.def {FAKE_LLM_KEY}\nTraceback (most recent call last):\n  File x\n"
    )
    reason = failure_reason(_failed(error), secrets=(FAKE_LLM_KEY,))
    assert "sk-" not in reason and FAKE_TOKEN not in reason and "abc.def" not in reason
    assert "Traceback" not in reason and "\n" not in reason
    long = failure_reason(_failed("x" * 1000))
    assert len(long) == 300 and long.endswith("…")


def test_failure_reason_falls_back_to_raised_error():
    assert failure_reason({"status": "failed"}, RuntimeError("disk full")) == "disk full"


def test_dead_keys_raised_through_a_crash_keep_their_class():
    from phoena_translator.api_runtime import LLMKeysUnusableError

    error = LLMKeysUnusableError("x", errors=[])
    assert failure_reason(_failed("Unhandled worker error: x"), error) == (
        "主力和备用模型 key 都不可用。Unhandled worker error: x"
    )


def test_file_names_with_markdown_characters_and_long_names():
    message = format_task_message("soft", _pdf_done(filename=ODD_NAME))
    assert message.splitlines()[0] == f"✅【soft】翻译成功：{ODD_NAME}"
    long_name = "第" * 500 + "_end.pdf"
    first = format_task_message("soft", _pdf_done(filename=long_name)).splitlines()[0]
    shown = first.split("翻译成功：", 1)[1]
    assert len(shown) == MAX_FILENAME_CHARS and shown.endswith("_end.pdf") and "…" in shown
    forged = format_task_message("soft", _pdf_done(filename="a\nb\r✅ fake.pdf"))
    assert forged.splitlines()[0] == "✅【soft】翻译成功：a b ✅ fake.pdf"


def test_message_never_exceeds_telegram_limit():
    record = _failed("y" * 10000, filename="n" * 10000)
    assert len(format_task_message("L" * 5000, record)) <= TELEGRAM_MAX_CHARS
    assert len(format_task_message("L" * 5000, _pdf_done(filename="n" * 10000))) <= TELEGRAM_MAX_CHARS


# ---------------------------------------------------------------------------
# Runner: one message per finished task, after the final save
# ---------------------------------------------------------------------------


def test_completed_job_sends_one_message_after_final_save(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": ODD_NAME})
    notifier, sent = _notifier()
    seen_status = []
    sent.send = lambda kind, message: (
        seen_status.append(repo.load("t1")["status"]),
        sent.calls.append((kind, message)),
    )
    run_translation_job(_job(), 1, _deps(repo, notifier, pdf=_saver(repo, "t1", {"status": "completed", "total": 3})))
    assert [kind for kind, _ in sent.calls] == ["task_done"]
    assert sent.calls[0][1].startswith(f"✅【PhoenaTranslator】翻译成功：{ODD_NAME}\n页数：3")
    assert seen_status == ["completed"]


def test_failed_job_sends_one_message(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "epub", "filename": "book.epub"})
    notifier, sent = _notifier()
    final = {"status": "failed", "error": "EPUB contains no XHTML documents"}
    run_translation_job(_job(extension="epub"), 1, _deps(repo, notifier, epub=_saver(repo, "t1", final)))
    assert sent.calls == [
        ("task_failed", "❌【PhoenaTranslator】翻译失败：book.epub\n失败原因：文件无法解析。EPUB contains no XHTML documents")
    ]


def test_job_ending_non_final_sends_nothing(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})
    notifier, sent = _notifier()
    run_translation_job(_job(), 1, _deps(repo, notifier, pdf=_saver(repo, "t1", {"status": "queued"})))
    assert sent.calls == []


def test_raising_job_reports_the_raised_error_once(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})
    notifier, sent = _notifier()

    def translate(*args, **kwargs):
        repo.save("t1", {**repo.load("t1"), "status": "failed", "error": ""})
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        run_translation_job(_job(), 1, _deps(repo, notifier, pdf=translate))
    assert [message.splitlines()[1] for _, message in sent.calls] == ["失败原因：disk full"]


def test_switch_off_means_no_message_and_no_hook(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})
    runtime = Runtime(AppConfig.from_env(CHANNEL, home=tmp_path))
    assert runtime.task_notifier is None
    deps = _deps(repo, None, pdf=_saver(repo, "t1", {"status": "completed"}))
    run_translation_job(_job(), 1, deps)
    runtime._task_finished("t1", repo.load("t1"))  # no notifier: a no-op
    assert repo.load("t1")["status"] == "completed"


def test_restart_with_final_task_sends_nothing(tmp_path):
    repo = _repo(tmp_path)
    notifier, sent = _notifier()
    repo.save("done", {"status": "completed", "type": "pdf", "filename": "a.pdf"})
    repo.save("bad", {"status": "failed", "type": "pdf", "filename": "b.pdf", "error": "x"})
    # A reload plus resume, then a duplicate job for an already-final task.
    reloaded = TaskRepository(repo.store)
    reloaded.load_all()
    resume_interrupted_tasks(
        TaskRecoveryDependencies(
            tasks=reloaded,
            upload_dir=tmp_path / "uploads",
            output_dir=tmp_path / "outputs",
            logger=log,
            enqueue_task=lambda *args: pytest.fail("final tasks are not requeued"),
            source_sha256=lambda path: "",
            load_cached_page_indices=lambda task_id, digest: set(),
            task_finished=notifier.task_finished,
        )
    )
    run_translation_job(_job("done"), 1, _deps(reloaded, notifier))
    run_translation_job(_job("bad"), 1, _deps(reloaded, notifier))
    assert sent.calls == []


def test_restart_that_fails_an_interrupted_task_sends_one(tmp_path):
    repo = _repo(tmp_path)
    notifier, sent = _notifier()
    repo.save("t1", {"status": "translating", "type": "pdf", "filename": "a.pdf"})
    dependencies = TaskRecoveryDependencies(
        tasks=repo,
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        logger=log,
        enqueue_task=lambda *args: pytest.fail("a lost upload is not requeued"),
        source_sha256=lambda path: "",
        load_cached_page_indices=lambda task_id, digest: set(),
        task_finished=notifier.task_finished,
    )
    resume_interrupted_tasks(dependencies)
    resume_interrupted_tasks(dependencies)  # a second restart: already final
    assert [message.splitlines()[1] for _, message in sent.calls] == [
        "失败原因：任务被中断。Upload file lost after restart"
    ]


def test_crashed_job_marked_failed_sends_one(tmp_path):
    runtime = Runtime(AppConfig.from_env(NOTIFY_ON, home=tmp_path))
    sent = _Sent()
    runtime.task_notifier = TaskNotifier(alerter=sent, label="soft", logger=log)
    runtime.tasks.save("t1", {"status": "translating", "type": "pdf", "filename": "a.pdf"})
    runtime._mark_failed_after_crash("t1", "boom")
    runtime._mark_failed_after_crash("t1", "again")  # already failed: no second message
    assert sent.calls == [
        ("task_failed", "❌【soft】翻译失败：a.pdf\n失败原因：程序内部错误。Unhandled worker error: boom")
    ]


def test_runtime_job_path_reaches_the_notifier(tmp_path):
    runtime = Runtime(AppConfig.from_env(NOTIFY_ON, home=tmp_path))
    sent = _Sent()
    runtime.task_notifier = TaskNotifier(alerter=sent, label="soft", logger=log)
    runtime.tasks.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})
    import dataclasses

    runtime._jobs = dataclasses.replace(
        runtime._jobs, translate_pdf=_saver(runtime.tasks, "t1", {"status": "completed"})
    )
    runtime._run_job(_job(), 1)
    assert [kind for kind, _ in sent.calls] == ["task_done"]


# ---------------------------------------------------------------------------
# Never in the way: sender failures, slow senders, misbehaving hooks
# ---------------------------------------------------------------------------


def _real_notifier(sender, logger=log):
    alerter = TelegramAlerter(
        sender=sender, label="soft", min_interval_seconds=0.0, logger=logger, log_prefix="task-notify"
    )
    return TaskNotifier(alerter=alerter, label="soft", logger=logger, secrets=(FAKE_TOKEN,))


def _wait_for_notify_threads():
    for thread in threading.enumerate():
        if thread.name.startswith("task-notify-"):
            thread.join(timeout=5)


def test_sender_raising_leaves_task_untouched(tmp_path, caplog):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})

    def sender(text):
        raise urllib.error.HTTPError(
            f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage", 401, FAKE_TOKEN, {}, io.BytesIO()
        )

    with caplog.at_level(logging.DEBUG):
        run_translation_job(
            _job(), 1, _deps(repo, _real_notifier(sender), pdf=_saver(repo, "t1", {"status": "completed"}))
        )
        _wait_for_notify_threads()
    record = repo.load("t1")
    assert record["status"] == "completed" and "error" not in record
    text = _all_messages(caplog)
    assert "task-notify: kind=task_done send failed status=401 error=HTTPError" in text
    assert FAKE_TOKEN not in text


def test_slow_sender_does_not_delay_the_task(tmp_path):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})
    release = threading.Event()
    delivered = []

    def sender(text):
        release.wait(timeout=5)
        delivered.append(text)

    started = time.monotonic()
    run_translation_job(
        _job(), 1, _deps(repo, _real_notifier(sender), pdf=_saver(repo, "t1", {"status": "failed", "error": "x"}))
    )
    assert time.monotonic() - started < 1.0
    assert repo.load("t1")["status"] == "failed" and delivered == []
    release.set()
    _wait_for_notify_threads()
    assert len(delivered) == 1 and delivered[0].startswith("❌【soft】翻译失败：a.pdf")


def test_hook_or_formatting_failure_never_reaches_the_task(tmp_path, caplog):
    repo = _repo(tmp_path)
    repo.save("t1", {"status": "queued", "type": "pdf", "filename": "a.pdf"})

    class _Broken:
        def send(self, kind, message):
            raise RuntimeError(FAKE_TOKEN)

    notifier = TaskNotifier(alerter=_Broken(), label="soft", logger=log)

    def exploding_hook(task_id, record, error):
        raise ValueError(FAKE_TOKEN)

    with caplog.at_level(logging.DEBUG):
        run_translation_job(_job(), 1, _deps(repo, notifier, pdf=_saver(repo, "t1", {"status": "completed"})))
        deps = TranslationJobDependencies(
            logger=log,
            load_progress=repo.load,
            translate_epub=None,
            translate_pdf=_saver(repo, "t1", {"status": "completed"}),
            task_finished=exploding_hook,
        )
        repo.save("t1", {**repo.load("t1"), "status": "queued"})
        run_translation_job(_job(), 1, deps)
    assert repo.load("t1")["status"] == "completed"
    text = _all_messages(caplog)
    assert "task-notify: [t1] skipped error=RuntimeError" in text
    assert "task-notify: [t1] skipped error=ValueError" in text
    assert FAKE_TOKEN not in text


def test_https_path_logs_no_token(tmp_path, caplog, monkeypatch):
    from phoena_translator import alerts

    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        raise urllib.error.URLError(f"unreachable {request.full_url}")

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    config = AppConfig.from_env(NOTIFY_ON, home=tmp_path)
    with caplog.at_level(logging.DEBUG):
        notifier = build_task_notifier(config, log)
        notifier.task_finished("t1", _pdf_done(filename="a.pdf"))
        _wait_for_notify_threads()
    assert len(seen) == 1 and seen[0][1] == alerts.TELEGRAM_TIMEOUT_SECONDS
    text = _all_messages(caplog)
    assert "task-notify: kind=task_done send failed status=- error=URLError" in text
    assert FAKE_TOKEN not in text and FAKE_CHAT not in text
