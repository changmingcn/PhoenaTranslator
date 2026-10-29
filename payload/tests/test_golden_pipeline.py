"""End-to-end characterization of the PDF pipeline on a synthetic corpus.

Each corpus document is translated in a fresh interpreter with a deterministic
stand-in translator.  The published artefacts are compared with the snapshot
in ``tests/golden``.  Refactors must keep them byte-for-byte identical; an
intentional behaviour change regenerates them with::

    PHOENA_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_pipeline.py
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import golden_support

TESTS_DIR = Path(__file__).resolve().parent
PAYLOAD_DIR = TESTS_DIR.parent
GOLDEN_DIR = TESTS_DIR / "golden"


def _run(name: str, workdir: Path, font: Path) -> dict:
    out = workdir / "snapshot.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(PAYLOAD_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(TESTS_DIR / "golden_support.py"), name, str(workdir), str(font), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _diff(expected: str, actual: str) -> str:
    lines = difflib.unified_diff(
        expected.splitlines(), actual.splitlines(), "golden", "actual", lineterm="", n=2
    )
    return "\n".join(list(lines)[:200])


@pytest.mark.parametrize("name", sorted(golden_support.CORPUS))
def test_pdf_pipeline_matches_golden_snapshot(name, tmp_path):
    font = golden_support.build_test_font(tmp_path)
    if font is None:
        pytest.skip("no glyf-outline CJK system font available for the golden corpus")
    font_path, font_id = font
    workdir = tmp_path / "work"
    workdir.mkdir()
    stable, font_dependent = golden_support.split_snapshot(
        _run(name, workdir, font_path), str(workdir)
    )
    assert (stable.get("task") or {}).get("status") == "completed", stable.get("task")

    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("PHOENA_UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(
            golden_support.dumps(
                {"font": font_id, "stable": stable, "font_dependent": font_dependent}
            ),
            encoding="utf-8",
        )
        pytest.skip(f"regenerated {golden_path.name}")

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    expected, actual = golden_support.dumps(golden["stable"]), golden_support.dumps(stable)
    assert actual == expected, _diff(expected, actual)
    if golden.get("font") == font_id:
        expected = golden_support.dumps(golden["font_dependent"])
        actual = golden_support.dumps(font_dependent)
        assert actual == expected, _diff(expected, actual)
