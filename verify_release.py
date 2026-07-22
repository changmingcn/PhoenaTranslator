#!/usr/bin/env python3
"""Deterministically validate a PhoenaTranslator recovery release."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath


EXPECTED_SOURCE_COUNT = 51
MANIFEST_NAME = "MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".first-secretary",
    ".git",
    ".pytest_cache",
    ".workflow",
    "__pycache__",
    "audit",
    "backups",
    "caches",
    "logs",
    "outputs",
    "progress",
    "uploads",
    "venv",
}
EXCLUDED_SUFFIXES = {
    ".bak",
    ".env",
    ".key",
    ".log",
    ".pem",
    ".pyc",
    ".pyo",
    ".tmp",
}
SECRET_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bxox(?:b|a|p|r|s)-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bBearer[ \t]+[A-Za-z0-9._~+/-]{20,}={0,2}\b", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)\b(?:password|passwd)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"),
)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def relative_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            fail(f"symlink is forbidden: {relative}")
        mode = path.lstat().st_mode
        if path.is_dir():
            if mode & 0o022:
                fail(f"group/world-writable directory: {relative}")
            continue
        if not stat.S_ISREG(mode):
            fail(f"non-regular file is forbidden: {relative}")
        if mode & 0o022:
            fail(f"group/world-writable file: {relative}")
        result[relative] = path
    return result


def safe_manifest_path(value: str) -> bool:
    pure = PurePosixPath(value)
    return (
        value != ""
        and "\\" not in value
        and not pure.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if not match:
            fail(f"invalid manifest line {number}")
        digest, relative = match.groups()
        if not safe_manifest_path(relative):
            fail(f"unsafe manifest path: {relative!r}")
        if relative == MANIFEST_NAME:
            fail("manifest must not list itself")
        if relative in entries:
            fail(f"duplicate manifest path: {relative}")
        entries[relative] = digest
    if not entries:
        fail("manifest is empty")
    return entries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_exclusions(files: dict[str, Path]) -> None:
    for relative, path in files.items():
        pure = PurePosixPath(relative)
        lower_parts = {part.lower() for part in pure.parts}
        if lower_parts & EXCLUDED_PARTS:
            fail(f"excluded state path present: {relative}")
        if pure.suffix.lower() in EXCLUDED_SUFFIXES:
            fail(f"excluded state suffix present: {relative}")
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                fail(f"credential-shaped value found in: {relative}")


def check_python(files: dict[str, Path]) -> int:
    count = 0
    for relative, path in files.items():
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (SyntaxError, UnicodeDecodeError) as exc:
            fail(f"Python parse failed for {relative}: {exc}")
        count += 1
    return count


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent).resolve()
    if not root.is_dir():
        fail(f"release directory not found: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        fail(f"missing {MANIFEST_NAME}")

    files = relative_files(root)
    manifest = read_manifest(manifest_path)
    actual_manifest_files = set(files) - {MANIFEST_NAME}
    if set(manifest) != actual_manifest_files:
        missing = sorted(actual_manifest_files - set(manifest))
        extra = sorted(set(manifest) - actual_manifest_files)
        fail(f"manifest closure mismatch; missing={missing}, extra={extra}")
    for relative, expected in manifest.items():
        actual = sha256(files[relative])
        if actual != expected:
            fail(f"hash mismatch: {relative}")

    allowlist_path = root / "SOURCE_ALLOWLIST.txt"
    allowlist = [line for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line]
    if len(allowlist) != EXPECTED_SOURCE_COUNT or len(set(allowlist)) != EXPECTED_SOURCE_COUNT:
        fail(f"source allowlist must contain {EXPECTED_SOURCE_COUNT} unique paths")
    if any(not safe_manifest_path(item) for item in allowlist):
        fail("source allowlist contains an unsafe path")
    payload_files = {
        relative.removeprefix("payload/")
        for relative in files
        if relative.startswith("payload/")
    }
    if payload_files != set(allowlist):
        missing = sorted(set(allowlist) - payload_files)
        extra = sorted(payload_files - set(allowlist))
        fail(f"payload/allowlist mismatch; missing={missing}, extra={extra}")

    check_exclusions(files)
    python_count = check_python(files)
    install_mode = stat.S_IMODE((root / "install.sh").stat().st_mode)
    if install_mode != 0o755:
        fail(f"install.sh mode must be 0755, got {install_mode:04o}")
    if sum(path.stat().st_size for path in files.values()) > 10 * 1024 * 1024:
        fail("release unexpectedly exceeds 10 MiB before compression")

    print(
        f"OK: {len(allowlist)} payload files, {len(files)} total files, "
        f"{python_count} Python files; manifest, topology, exclusions and syntax passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
