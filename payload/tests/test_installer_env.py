"""install.sh: the translator.env lines a reinstall keeps, and the Telegram option.

The installer itself needs root, apt and systemd, so these tests lift the
pure env-writing functions out of ``install.sh`` and run them in bash against
env files in a temp dir. The prompts read ``/dev/tty`` and are not run here.

Run from the ``payload`` directory:  python -m pytest tests -q
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"
FUNCTIONS = (
    "preserved_env_lines",
    "preserved_env_value",
    "render_optional_env_lines",
    "yes_no_flag",
)
# Filler values in token shape, never real credentials.
FAKE_TOKEN = "123456789:AAfiller-not-a-real-bot-token-000000"
FAKE_CHAT = "-1000000000000"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not INSTALLER.is_file(),
    reason="bash and the repository's install.sh are needed",
)


def _functions() -> str:
    text = INSTALLER.read_text(encoding="utf-8")
    bodies = []
    for name in FUNCTIONS:
        match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", text, flags=re.S | re.M)
        assert match, f"{name} not found in install.sh"
        bodies.append(match.group(0))
    return "\n".join(bodies)


def _bash(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", "set -Eeuo pipefail\nIFS=$'\\n\\t'\n" + _functions() + script, "bash", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _render(env_file: Path, token="", chat="", notify="") -> list[str]:
    result = _bash('\nrender_optional_env_lines "$1" "$2" "$3" "$4"\n', str(env_file), token, chat, notify)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


EXISTING = (
    "DEEPSEEK_API_KEY=old-filler\n"
    "# and LLM_BACKUP_API_KEY (comment lines are not carried over)\n"
    "LLM_PRIMARY_API_KEY=filler-primary\n"
    "LLM_BACKUP_API_KEY=filler-backup\n"
    "LLM_FAILOVER_RETRY_PRIMARY_SECONDS=600\n"
    f"TRANSLATOR_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}\n"
    f"TRANSLATOR_TELEGRAM_CHAT_ID={FAKE_CHAT}\n"
    "TRANSLATOR_TELEGRAM_LABEL=soft\n"
    "TRANSLATOR_TELEGRAM_NOTIFY_DONE=0\n"
    "TRANSLATOR_UPLOAD_DIR=/var/lib/phoena-translator/uploads\n"
)


def test_fresh_install_without_answers_writes_no_optional_lines(tmp_path):
    assert _render(tmp_path / "absent.env") == []


def test_fresh_install_with_channel_and_switch(tmp_path):
    assert _render(tmp_path / "absent.env", FAKE_TOKEN, FAKE_CHAT, "1") == [
        f"TRANSLATOR_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}",
        f"TRANSLATOR_TELEGRAM_CHAT_ID={FAKE_CHAT}",
        "TRANSLATOR_TELEGRAM_NOTIFY_DONE=1",
    ]


def test_reinstall_keeps_llm_and_telegram_lines_and_sets_the_switch(tmp_path):
    env = tmp_path / "translator.env"
    env.write_text(EXISTING, encoding="utf-8")
    assert _render(env, notify="1") == [
        "LLM_PRIMARY_API_KEY=filler-primary",
        "LLM_BACKUP_API_KEY=filler-backup",
        "LLM_FAILOVER_RETRY_PRIMARY_SECONDS=600",
        f"TRANSLATOR_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}",
        f"TRANSLATOR_TELEGRAM_CHAT_ID={FAKE_CHAT}",
        "TRANSLATOR_TELEGRAM_LABEL=soft",
        "TRANSLATOR_TELEGRAM_NOTIFY_DONE=1",
    ]


def test_reinstall_with_no_telegram_answer_keeps_lines_verbatim(tmp_path):
    env = tmp_path / "translator.env"
    env.write_text(EXISTING, encoding="utf-8")
    kept = [line for line in EXISTING.splitlines() if line.startswith(("LLM_", "TRANSLATOR_TELEGRAM_"))]
    assert _render(env) == kept


PROVIDER_LINES = (
    "TRANSLATOR_LLM_PROVIDER=claude\n"
    "ANTHROPIC_API_KEY=filler-anthropic\n"
    "CLAUDE_MODEL=claude-filler\n"
    "CLAUDE_EFFORT=high\n"
    "OPENAI_MODEL=gpt-filler\n"
    "OPENAI_PRICE_INPUT_USD_PER_M=1.5\n"
)
NOT_KEPT = "TRANSLATOR_LLM_PROVIDER_EXTRA=x\nDEEPSEEK_MODEL=old-model\nopenai_model=lower\n"


def test_reinstall_keeps_provider_lines_and_writes_no_variable_twice(tmp_path):
    env = tmp_path / "translator.env"
    env.write_text(EXISTING + PROVIDER_LINES + NOT_KEPT, encoding="utf-8")
    kept = [line for line in EXISTING.splitlines() if line.startswith(("LLM_", "TRANSLATOR_TELEGRAM_"))]
    assert _render(env) == kept + PROVIDER_LINES.splitlines()
    # The whole new translator.env as the installer writes it, prompts answered.
    text = INSTALLER.read_text(encoding="utf-8")
    block = text.split('ENV_NEW="$ROLLBACK_ROOT/new-translator.env"\n{', 1)[1].split('} > "$ENV_NEW"', 1)[0]
    for token, chat, notify in (("", "", ""), ("", "", "1"), (FAKE_TOKEN, FAKE_CHAT, "0")):
        result = _bash(
            '\nENV_FILE="$1" API_KEY=filler-key API_BASE_URL=https://example.invalid API_MODEL=m'
            ' TELEGRAM_TOKEN="$2" TELEGRAM_CHAT_ID="$3" TELEGRAM_NOTIFY_DONE="$4"'
            " DATA_ROOT=/data LOG_ROOT=/logs\n{" + block + "}\n",
            str(env), token, chat, notify,
        )
        assert result.returncode == 0, result.stderr
        names = [line.split("=", 1)[0] for line in result.stdout.splitlines() if not line.startswith("#")]
        assert len(names) == len(set(names)), names
        assert set(PROVIDER_LINES.replace("=", "\n").splitlines()[::2]) <= set(names)
        assert "TRANSLATOR_LLM_PROVIDER_EXTRA" not in names and "openai_model" not in names


def test_new_channel_replaces_only_the_channel_lines(tmp_path):
    env = tmp_path / "translator.env"
    env.write_text("LLM_PRIMARY_API_KEY=filler\nTRANSLATOR_TELEGRAM_CHAT_ID=1\n", encoding="utf-8")
    assert _render(env, FAKE_TOKEN, FAKE_CHAT, "0") == [
        "LLM_PRIMARY_API_KEY=filler",
        f"TRANSLATOR_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}",
        f"TRANSLATOR_TELEGRAM_CHAT_ID={FAKE_CHAT}",
        "TRANSLATOR_TELEGRAM_NOTIFY_DONE=0",
    ]


def test_preserved_value_and_yes_no(tmp_path):
    env = tmp_path / "translator.env"
    env.write_text(EXISTING, encoding="utf-8")
    result = _bash(
        '\npreserved_env_value "$1" TRANSLATOR_TELEGRAM_LABEL'
        '\nfor answer in y YES No n; do yes_no_flag "$answer"; done'
        '\nyes_no_flag maybe || printf " rejected"\n',
        str(env),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "soft\n1100 rejected"


def test_installer_writes_the_optional_lines_into_the_new_env_file():
    text = INSTALLER.read_text(encoding="utf-8")
    block = text.split('ENV_NEW="$ROLLBACK_ROOT/new-translator.env"', 1)[1].split('} > "$ENV_NEW"', 1)[0]
    assert (
        'render_optional_env_lines "$ENV_FILE" "$TELEGRAM_TOKEN" "$TELEGRAM_CHAT_ID" '
        '"$TELEGRAM_NOTIFY_DONE"'
    ) in block
    # The optional prompts treat end of input as "skip": no new required input.
    for name in ("read_optional_secret", "read_optional_value"):
        body = re.search(rf"^{name}\(\) {{\n.*?^}}\n", text, flags=re.S | re.M).group(0)
        assert '</dev/tty || result=""' in body
