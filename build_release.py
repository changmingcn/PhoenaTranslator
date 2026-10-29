#!/usr/bin/env python3
"""Refresh the release metadata, or build the reinstall package.

    python3 build_release.py              rewrite SOURCE_ALLOWLIST.txt, the payload
                                          fields of SOURCE_IDENTITY.json, the count in
                                          verify_release.py and MANIFEST.sha256
    python3 build_release.py --dist DIR   check that metadata is current, then write
                                          DIR/<package>.tar.gz, its .sha256 and
                                          release-notes.md

The package holds every tracked file except development-only ones: the tests,
their requirements, this script and the GitHub workflow.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ONLY = ("payload/tests/", "payload/requirements-test.txt", "build_release.py", ".github/")
MANIFEST = "MANIFEST.sha256"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_files() -> list[str]:
    listing = subprocess.run(
        ["git", "-c", "core.quotepath=off", "ls-files", "-z"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted(
        path for path in listing.split("\0")
        if path and not path.startswith(DEVELOPMENT_ONLY) and path != MANIFEST
    )


def metadata(files: list[str]) -> dict[str, bytes]:
    """The four metadata files as they must read for ``files``."""
    payload = [path.removeprefix("payload/") for path in files if path.startswith("payload/")]
    payload_manifest = "".join(
        f"{sha256((ROOT / 'payload' / path).read_bytes())}  {path}\n" for path in payload
    )
    identity = json.loads((ROOT / "SOURCE_IDENTITY.json").read_text(encoding="utf-8"))
    identity.update({
        "payload_file_count": len(payload),
        "payload_total_bytes": sum((ROOT / "payload" / path).stat().st_size for path in payload),
        "payload_manifest_sha256": sha256(payload_manifest.encode()),
    })
    verify = re.sub(
        r"^EXPECTED_SOURCE_COUNT = \d+$",
        f"EXPECTED_SOURCE_COUNT = {len(payload)}",
        (ROOT / "verify_release.py").read_text(encoding="utf-8"),
        flags=re.M,
    )
    contents = {
        "SOURCE_ALLOWLIST.txt": ("\n".join(payload) + "\n").encode(),
        "SOURCE_IDENTITY.json": (json.dumps(identity, ensure_ascii=False, indent=2) + "\n").encode(),
        "verify_release.py": verify.encode(),
    }
    contents[MANIFEST] = "".join(
        f"{sha256(contents.get(path) or (ROOT / path).read_bytes())}  {path}\n" for path in files
    ).encode()
    return contents


def release_notes(version: str) -> str:
    guide = (ROOT / "README_安装说明.md").read_text(encoding="utf-8")
    changes = guide.split("## 相对上一版的变化", 1)[1].split("\n## ", 1)[0].strip()
    return (
        f"PhoenaTranslator {version}\n\n## 相对上一版的变化\n\n{changes}\n\n"
        "安装、校验与升级步骤见压缩包内的 `README_安装说明.md`。\n"
    )


def build(dist: Path, files: list[str]) -> None:
    identity = json.loads((ROOT / "SOURCE_IDENTITY.json").read_text(encoding="utf-8"))
    name = identity["package_name"]
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if identity["version"] != version:
        sys.exit(f"SOURCE_IDENTITY version {identity['version']} != VERSION {version}")
    mtime = int(subprocess.run(
        ["git", "log", "-1", "--format=%ct"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout)
    with tempfile.TemporaryDirectory() as staging:
        package = Path(staging) / name
        for path in [*files, MANIFEST]:
            target = package / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
            target.chmod(0o755 if path == "install.sh" else 0o644)
        for directory in [package, *package.rglob("*")]:
            if directory.is_dir():
                directory.chmod(0o755)
        subprocess.run([sys.executable, str(package / "verify_release.py"), str(package)], check=True)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(package.parent.rglob("*")):
                info = tar.gettarinfo(str(path), arcname=path.relative_to(package.parent).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = mtime
                if path.is_file():
                    with path.open("rb") as source:
                        tar.addfile(info, source)
                else:
                    tar.addfile(info)
    # A fixed gzip timestamp makes the same commit always yield the same bytes.
    compressed = gzip.compress(archive.getvalue(), mtime=mtime)
    dist.mkdir(parents=True, exist_ok=True)
    tarball = dist / f"{name}.tar.gz"
    tarball.write_bytes(compressed)
    (dist / f"{tarball.name}.sha256").write_text(f"{sha256(compressed)}  {tarball.name}\n")
    (dist / "release-notes.md").write_text(release_notes(version), encoding="utf-8")
    print(f"built {tarball} ({len(files) + 1} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, help="build the package into this directory")
    args = parser.parse_args()
    files = package_files()
    expected = metadata(files)
    if args.dist is None:
        for path, content in expected.items():
            (ROOT / path).write_bytes(content)
        print(f"metadata refreshed for {len(files) + 1} package files")
        return
    stale = [path for path, content in expected.items() if (ROOT / path).read_bytes() != content]
    if stale:
        sys.exit(f"release metadata is stale ({', '.join(stale)}); run build_release.py first")
    build(args.dist, files)


if __name__ == "__main__":
    main()
