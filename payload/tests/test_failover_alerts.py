"""Telegram alerts on key failover: once per state change, throttled, never in the way.

No network: gateway tests use a recording alerter, alerter tests a stub sender,
and the HTTPS sender is exercised against a patched ``urlopen``.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import urllib.error

import pytest
from test_llm_failover import (
    KEY_UNUSABLE,
    NEVER_SWITCH,
    FAST_POLICY,
    _Adapter,
    _Clock,
    _complete,
    _failover,
    _failover_lines,
    _ok,
    CLAUDE_KEY,
    OPENAI_KEY,
    DEEPSEEK_KEY,
    log,
)

from phoena_translator import alerts
from phoena_translator.alerts import TelegramAlerter, send_telegram_message
from phoena_translator.api_runtime import LLMGateway, LLMKeysUnusableError
from phoena_translator.config import AppConfig
from phoena_translator.runtime import build_llm_gateway

# Filler values in token shape, never real credentials.
FAKE_TOKEN = "123456789:AAfiller-not-a-real-bot-token-000000"
FAKE_CHAT = "test-chat-filler"
ALERT_ENV = {
    "TRANSLATOR_TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
    "TRANSLATOR_TELEGRAM_CHAT_ID": FAKE_CHAT,
}
SLOT_ENV = {"LLM_PRIMARY_API_KEY": CLAUDE_KEY, "LLM_BACKUP_API_KEY": OPENAI_KEY}

DEAD_PRIMARY = KEY_UNUSABLE["anthropic-401"][0]
DEAD_BACKUP = KEY_UNUSABLE["openai-401"][0]
SWITCHED_TEXT = (
    "主力 key（claude）不可用（HTTP 401 authentication_error），已改用备用（openai）。"
    "翻译继续，请检查主力 key 或额度。"
)
RESTORED_TEXT = "主力 key（claude）已恢复，已切回主力。"
ALL_DEAD_TEXT = (
    "主力（claude）和备用（openai）两把 key 都不可用，翻译请求失败。\n"
    "主力（claude）：HTTP 401 authentication_error\n"
    "备用（openai）：HTTP 401 invalid_api_key"
)


class _Notes:
    """Synchronous alerter stand-in: records every ``notify`` call."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def notify(self, kind, text):
        with self.lock:
            self.calls.append((kind, text))

    @property
    def kinds(self):
        return [kind for kind, _ in self.calls]


class _Sent:
    """Stub sender: records texts; ``wait_for(n)`` blocks until n arrived."""

    def __init__(self, behaviour=None):
        self.texts: list[str] = []
        self.cond = threading.Condition()
        self.behaviour = behaviour

    def __call__(self, text):
        if self.behaviour is not None:
            self.behaviour()
        with self.cond:
            self.texts.append(text)
            self.cond.notify_all()

    def wait_for(self, count, timeout=5.0):
        with self.cond:
            assert self.cond.wait_for(lambda: len(self.texts) >= count, timeout)


def _join_alert_threads(timeout=5.0):
    for thread in threading.enumerate():
        if thread.name.startswith("llm-alert-"):
            thread.join(timeout)
            assert not thread.is_alive()


def _gateway(tmp_path, primary, backup, alerter, **kwargs):
    gateway = _failover(tmp_path, primary, backup, **kwargs)
    gateway.alerter = alerter
    return gateway


def _run_concurrently(gateway, count=16):
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(_complete(gateway).served_by)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


# ---------------------------------------------------------------------------
# Configuration and wiring
# ---------------------------------------------------------------------------


def test_alert_config_defaults_and_secrets_out_of_repr():
    config = AppConfig.from_env({**SLOT_ENV, **ALERT_ENV})
    assert config.key_alerts_enabled
    assert (config.telegram_label, config.telegram_min_interval_seconds) == ("PhoenaTranslator", 1800.0)
    assert FAKE_TOKEN not in repr(config) and FAKE_CHAT not in repr(config)
    tuned = AppConfig.from_env(
        {**ALERT_ENV, "TRANSLATOR_TELEGRAM_LABEL": " soft ", "TRANSLATOR_TELEGRAM_MIN_INTERVAL_SECONDS": "60"}
    )
    assert (tuned.telegram_label, tuned.telegram_min_interval_seconds) == ("soft", 60.0)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"TRANSLATOR_TELEGRAM_BOT_TOKEN": FAKE_TOKEN},
        {"TRANSLATOR_TELEGRAM_CHAT_ID": FAKE_CHAT},
        {**ALERT_ENV, "TRANSLATOR_TELEGRAM_DISABLE": "1"},
        {**ALERT_ENV, "TRANSLATOR_TELEGRAM_KEY_ALERTS": "0"},
    ],
)
def test_alerts_off_without_both_settings_or_when_disabled(tmp_path, env):
    config = AppConfig.from_env({**SLOT_ENV, **env}, home=tmp_path)
    assert not config.key_alerts_enabled
    assert build_llm_gateway(config, log).alerter is None


def test_failover_gateway_gets_alerter_and_logs_no_secret(tmp_path, caplog):
    config = AppConfig.from_env(
        {**SLOT_ENV, **ALERT_ENV, "TRANSLATOR_TELEGRAM_LABEL": "soft"}, home=tmp_path
    )
    with caplog.at_level(logging.INFO, logger=log.name):
        gateway = build_llm_gateway(config, log)
    assert isinstance(gateway.alerter, TelegramAlerter)
    assert (gateway.alerter.label, gateway.alerter.min_interval_seconds) == ("soft", 1800.0)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLM failover alerts: telegram on, label=soft" in text
    assert FAKE_TOKEN not in text and FAKE_CHAT not in text


@pytest.mark.parametrize(
    "env",
    [
        {"DEEPSEEK_API_KEY": DEEPSEEK_KEY},  # legacy mode
        {"LLM_PRIMARY_API_KEY": OPENAI_KEY},  # one slot: nothing to fail over to
    ],
)
def test_no_failover_means_no_alerter(tmp_path, env):
    gateway = build_llm_gateway(AppConfig.from_env({**env, **ALERT_ENV}, home=tmp_path), log)
    assert type(gateway) is LLMGateway and not hasattr(gateway, "alerter")


# ---------------------------------------------------------------------------
# Gateway: one alert per change of state
# ---------------------------------------------------------------------------


def test_switched_alerts_once_with_concurrent_requests(tmp_path, caplog):
    notes = _Notes()
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, notes)
    with caplog.at_level(logging.WARNING, logger=log.name):
        results, errors = _run_concurrently(gateway)
        results.append(_complete(gateway).served_by)
    assert errors == [] and results == ["openai"] * 17
    assert notes.calls == [("switched", SWITCHED_TEXT)]
    # The failover log is unchanged: one line per request that met the dead key.
    assert len(_failover_lines(caplog)) == primary.calls


def test_both_dead_alerts_all_dead_once_and_never_switched(tmp_path, caplog):
    notes = _Notes()
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=DEAD_BACKUP)
    gateway = _gateway(tmp_path, primary, backup, notes)
    for _ in range(6):  # the active slot flips on every request
        with pytest.raises(LLMKeysUnusableError):
            _complete(gateway)
    results, errors = _run_concurrently(gateway)
    assert results == [] and len(errors) == 16
    assert all(isinstance(e, LLMKeysUnusableError) for e in errors)
    assert notes.calls == [("all_dead", ALL_DEAD_TEXT)]


def test_recovery_from_both_dead_alerts_each_change(tmp_path):
    notes = _Notes()
    primary = _Adapter("claude", [DEAD_PRIMARY, DEAD_PRIMARY], default=_ok("claude"))
    backup = _Adapter("openai", [DEAD_BACKUP], default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, notes)
    with pytest.raises(LLMKeysUnusableError):
        _complete(gateway)  # primary 401 -> backup 401
    assert _complete(gateway).served_by == "openai"  # backup topped up, primary dead
    assert _complete(gateway).served_by == "openai"
    assert notes.kinds == ["all_dead", "switched"]
    assert notes.calls[1][1] == SWITCHED_TEXT


def test_restored_alerts_once_and_late_backup_answer_is_silent(tmp_path):
    notes = _Notes()
    clock = _Clock()
    entered, gate = threading.Event(), threading.Event()

    def slow_backup():
        entered.set()
        gate.wait(5)
        return _ok("openai")

    primary = _Adapter("claude", [DEAD_PRIMARY], default=_ok("claude"))
    backup = _Adapter("openai", [_ok("openai"), slow_backup], default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, notes, clock=clock, retry_after=60)
    assert _complete(gateway).served_by == "openai"
    late: list = []
    straggler = threading.Thread(target=lambda: late.append(_complete(gateway).served_by))
    straggler.start()  # planned on backup, answers after primary is back
    assert entered.wait(5)
    clock.now += 60
    assert _complete(gateway).served_by == "claude"  # the primary retry succeeds
    gate.set()
    straggler.join(5)
    assert late == ["openai"] and gateway.active_slot == "primary"
    assert notes.calls == [("switched", SWITCHED_TEXT), ("restored", RESTORED_TEXT)]


def test_failed_primary_retry_sends_nothing(tmp_path):
    notes = _Notes()
    clock = _Clock()
    failing = [NEVER_SWITCH["anthropic-529"]] * FAST_POLICY.attempts
    primary = _Adapter("claude", [DEAD_PRIMARY, *failing], default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, notes, clock=clock, retry_after=60)
    _complete(gateway)
    clock.now += 60
    assert _complete(gateway).served_by == "openai"  # primary_retry_failed
    clock.now += 60
    assert _complete(gateway).served_by == "openai"  # primary still key-dead
    assert notes.kinds == ["switched"]


def test_backup_dying_back_to_primary_alerts_restored(tmp_path):
    notes = _Notes()
    primary = _Adapter("claude", [DEAD_PRIMARY], default=_ok("claude"))
    backup = _Adapter("openai", [_ok("openai")], default=KEY_UNUSABLE["openai-429-quota"][0])
    gateway = _gateway(tmp_path, primary, backup, notes)
    assert _complete(gateway).served_by == "openai"
    assert _complete(gateway).served_by == "claude"
    assert _complete(gateway).served_by == "claude"
    assert notes.kinds == ["switched", "restored"]


def test_no_alerter_behaves_as_before(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):  # pragma: no cover - asserted by not being hit
        raise AssertionError("no alert may be sent")

    monkeypatch.setattr(alerts.urllib.request, "urlopen", forbidden)
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=DEAD_BACKUP)
    gateway = _failover(tmp_path, primary, backup)
    assert gateway.alerter is None
    with pytest.raises(LLMKeysUnusableError):
        _complete(gateway)


class _BrokenAlerter:
    def notify(self, kind, text):
        raise RuntimeError("alert backend exploded")


def test_alert_failure_never_changes_the_translation(tmp_path, caplog):
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, _BrokenAlerter())
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai"
    assert "llm-alert: skipped error=RuntimeError" in [r.getMessage() for r in caplog.records]
    both_dead = _gateway(
        tmp_path, _Adapter("claude", default=DEAD_PRIMARY), _Adapter("openai", default=DEAD_BACKUP), _BrokenAlerter()
    )
    with pytest.raises(LLMKeysUnusableError):
        _complete(both_dead)


# ---------------------------------------------------------------------------
# Alerter: label, throttle, suppressed count, failures, no blocking
# ---------------------------------------------------------------------------


def _alerter(sender, clock=None, interval=1800.0):
    kwargs = {} if clock is None else {"clock": clock}
    return TelegramAlerter(
        sender=sender, label="soft", min_interval_seconds=interval, logger=log, **kwargs
    )


def test_throttle_per_kind_and_suppressed_count(caplog):
    clock = _Clock()
    sent = _Sent()
    alerter = _alerter(sent, clock)
    with caplog.at_level(logging.INFO, logger=log.name):
        alerter.notify("switched", "a")
        sent.wait_for(1)
        clock.now += 10
        alerter.notify("switched", "b")
        alerter.notify("switched", "c")
        alerter.notify("restored", "d")  # another kind is not throttled
        sent.wait_for(2)
        clock.now += 1790
        alerter.notify("switched", "e")
        sent.wait_for(3)
        clock.now += 10
        alerter.notify("switched", "f")
        _join_alert_threads()
    assert sent.texts == [
        "【soft】a",
        "【soft】d",
        "【soft】e\n（上次提醒后同类事件另有 2 次，已合并未发）",
    ]
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("llm-alert: kind=switched sent merged=0") == 1
    assert "llm-alert: kind=switched sent merged=2" in messages
    assert "llm-alert: kind=switched throttled suppressed=1" in messages
    assert [r.levelno for r in caplog.records if "llm-alert" in r.getMessage()] == [logging.INFO] * 6


def test_sender_failure_logs_class_only(caplog):
    def boom(text):
        raise OSError(f"cannot reach https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage")

    with caplog.at_level(logging.INFO, logger=log.name):
        assert _alerter(boom).notify("all_dead", "x") is True  # handed on; delivery fails
        _join_alert_threads()
    failures = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in failures] == [
        "llm-alert: kind=all_dead send failed status=- error=OSError"
    ]


def test_blocked_sender_does_not_delay_translation(tmp_path, caplog):
    gate = threading.Event()
    sent = _Sent(behaviour=lambda: gate.wait(5))
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=_ok("openai"))
    gateway = _gateway(tmp_path, primary, backup, _alerter(sent))
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=log.name):
        assert _complete(gateway).served_by == "openai"
    assert time.monotonic() - started < 1.0, "the alert never waits on Telegram"
    assert gateway.active_slot == "backup" and len(_failover_lines(caplog)) == 1
    gate.set()
    sent.wait_for(1)
    assert sent.texts == [f"【soft】{SWITCHED_TEXT}"]
    _join_alert_threads()


def test_raising_sender_leaves_result_unchanged(tmp_path):
    def boom(text):
        raise TimeoutError("timed out")

    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", default=_ok("openai", text="原样"))
    gateway = _gateway(tmp_path, primary, backup, _alerter(boom))
    result = _complete(gateway)
    assert result.served_by == "openai" and result.choices[0].message.content == "原样"
    _join_alert_threads()


def test_held_back_state_is_sent_once_the_interval_allows(tmp_path, caplog):
    """Both dead -> switched -> both dead inside one interval, then both stay dead."""
    clock = _Clock()
    sent = _Sent()
    primary = _Adapter("claude", default=DEAD_PRIMARY)
    backup = _Adapter("openai", [DEAD_BACKUP, _ok("openai")], default=DEAD_BACKUP)
    gateway = _gateway(tmp_path, primary, backup, _alerter(sent, clock), clock=clock, retry_after=600)
    with caplog.at_level(logging.INFO, logger=log.name):
        with pytest.raises(LLMKeysUnusableError):
            _complete(gateway)  # t=0
        sent.wait_for(1)
        clock.now += 60
        assert _complete(gateway).served_by == "openai"  # t=60, backup topped up
        sent.wait_for(2)
        clock.now += 60
        for _ in range(6):  # t=120 .. 1620, one request every 5 minutes
            with pytest.raises(LLMKeysUnusableError):
                _complete(gateway)
            clock.now += 300
        _join_alert_threads()
        assert len(sent.texts) == 2, "held back inside the interval"
        with pytest.raises(LLMKeysUnusableError):
            _complete(gateway)  # t=1920: the interval since t=0 has passed
        sent.wait_for(3)
        for _ in range(6):
            clock.now += 300
            with pytest.raises(LLMKeysUnusableError):
                _complete(gateway)
        _join_alert_threads()
    assert sent.texts == [
        f"【soft】{ALL_DEAD_TEXT}",
        f"【soft】{SWITCHED_TEXT}",
        f"【soft】{ALL_DEAD_TEXT}",
    ]
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("llm-alert: kind=all_dead throttled suppressed=1") == 1
    assert messages.count("llm-alert: kind=all_dead sent merged=0") == 2


def test_repeat_offer_is_not_counted_twice():
    clock = _Clock()
    sent = _Sent()
    alerter = _alerter(sent, clock)
    assert alerter.notify("all_dead", "a") is True
    sent.wait_for(1)
    clock.now += 10
    assert alerter.notify("all_dead", "b") is False
    assert alerter.notify("all_dead", "b", repeat=True) is False
    assert alerter.notify("all_dead", "c") is False  # a second held-back event
    clock.now += 1790
    assert alerter.notify("all_dead", "c", repeat=True) is True
    sent.wait_for(2)
    assert alerter.notify("all_dead", "c", repeat=True) is False
    _join_alert_threads()
    assert sent.texts == ["【soft】a", "【soft】c\n（上次提醒后同类事件另有 1 次，已合并未发）"]


# ---------------------------------------------------------------------------
# HTTPS sender: request shape and token-free failure logs (patched urlopen)
# ---------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_send_telegram_message_request_shape(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(
            url=request.full_url,
            method=request.get_method(),
            body=json.loads(request.data),
            timeout=timeout,
        )
        return _Response(b'{"ok":true}')

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    send_telegram_message(FAKE_TOKEN, FAKE_CHAT, "【soft】主力 key 已恢复")
    assert captured == {
        "url": f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage",
        "method": "POST",
        "body": {
            "chat_id": FAKE_CHAT,
            "text": "【soft】主力 key 已恢复",
            "disable_web_page_preview": True,
        },
        "timeout": 10.0,
    }


@pytest.mark.parametrize("failure", ["http-401", "url-error", "timeout", "success"])
def test_no_token_in_any_log_record(tmp_path, monkeypatch, caplog, capsys, failure):
    url = f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"

    def fake_urlopen(request, timeout):
        if failure == "http-401":
            raise urllib.error.HTTPError(url, 401, f"Unauthorized {url}", {}, io.BytesIO(b"{}"))
        if failure == "url-error":
            raise urllib.error.URLError(f"connect failed for {url}")
        if failure == "timeout":
            raise TimeoutError(f"timed out: {url}")
        return _Response(b'{"ok":true}')

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    caplog.set_level(logging.DEBUG)
    alerter = TelegramAlerter.for_telegram(
        token=FAKE_TOKEN, chat_id=FAKE_CHAT, label="soft", min_interval_seconds=0, logger=log
    )
    primary = _Adapter("claude", [DEAD_PRIMARY], default=_ok("claude"))
    backup = _Adapter("openai", [_ok("openai")], default=DEAD_BACKUP)
    gateway = _gateway(tmp_path, primary, backup, alerter)
    assert _complete(gateway).served_by == "openai"  # switched
    assert _complete(gateway).served_by == "claude"  # restored
    _join_alert_threads()
    alert_lines = [r for r in caplog.records if r.getMessage().startswith("llm-alert:")]
    assert len(alert_lines) == 2
    if failure == "success":
        assert all(r.levelno == logging.INFO for r in alert_lines)
    else:
        assert all(r.levelno == logging.WARNING for r in alert_lines)
    if failure == "http-401":
        assert "status=401 error=HTTPError" in alert_lines[0].getMessage()
    for record in caplog.records:
        rendered = " ".join(
            [record.getMessage(), str(record.args), str(record.exc_text or "")]
        )
        assert FAKE_TOKEN not in rendered and "api.telegram.org" not in rendered
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.out + captured.err
