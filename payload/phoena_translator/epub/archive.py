"""Bounded, traversal-safe EPUB extraction and deterministic packaging."""

from __future__ import annotations

import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


EPUB_MIMETYPE = b"application/epub+zip"
_COPY_BLOCK_SIZE = 1024 * 1024
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class UnsafeEPUBArchive(ValueError):
    """Raised when an EPUB member or archive exceeds the safety contract."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 10_000
    max_expanded_bytes: int = 1024 * 1024 * 1024
    max_member_bytes: int = 256 * 1024 * 1024
    max_compression_ratio: float = 1000.0

    def __post_init__(self) -> None:
        if self.max_members < 1:
            raise ValueError("max_members must be positive")
        if self.max_expanded_bytes < 1 or self.max_member_bytes < 1:
            raise ValueError("archive byte limits must be positive")
        if self.max_compression_ratio < 1:
            raise ValueError("max_compression_ratio must be at least one")


@dataclass(frozen=True)
class ValidatedMember:
    info: zipfile.ZipInfo
    name: str
    is_directory: bool


def _safe_member_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name:
        raise UnsafeEPUBArchive("empty or NUL-containing archive member")
    if "\\" in raw_name:
        raise UnsafeEPUBArchive(f"backslash archive path rejected: {raw_name!r}")
    if raw_name.startswith(("/", "//")) or _WINDOWS_DRIVE_RE.match(raw_name):
        raise UnsafeEPUBArchive(f"absolute archive path rejected: {raw_name!r}")
    path = PurePosixPath(raw_name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeEPUBArchive(f"non-canonical archive path rejected: {raw_name!r}")
    name = path.as_posix()
    if name in {"", "."}:
        raise UnsafeEPUBArchive(f"invalid archive member: {raw_name!r}")
    return name


def _member_kind(info: zipfile.ZipInfo) -> tuple[bool, bool]:
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    is_directory = info.is_dir() or kind == stat.S_IFDIR
    is_regular = kind in {0, stat.S_IFREG}
    return is_directory, is_regular


def validate_epub_members(
    archive: zipfile.ZipFile,
    limits: ArchiveLimits,
) -> list[ValidatedMember]:
    infos = archive.infolist()
    if len(infos) > limits.max_members:
        raise UnsafeEPUBArchive(
            f"archive member count {len(infos)} exceeds limit {limits.max_members}"
        )

    members: list[ValidatedMember] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    total_size = 0
    file_names: set[str] = set()
    directory_names: set[str] = set()

    for info in infos:
        name = _safe_member_name(info.filename)
        folded = unicodedata.normalize("NFC", name).casefold()
        if name in seen or folded in seen_casefolded:
            raise UnsafeEPUBArchive(f"duplicate archive member rejected: {name!r}")
        seen.add(name)
        seen_casefolded.add(folded)
        if info.flag_bits & 0x1:
            raise UnsafeEPUBArchive(f"encrypted archive member rejected: {name!r}")

        is_directory, is_regular = _member_kind(info)
        if not is_directory and not is_regular:
            raise UnsafeEPUBArchive(f"link or special archive member rejected: {name!r}")
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeEPUBArchive(f"invalid archive member size: {name!r}")
        if info.file_size > limits.max_member_bytes:
            raise UnsafeEPUBArchive(
                f"archive member {name!r} exceeds {limits.max_member_bytes} bytes"
            )
        total_size += info.file_size
        if total_size > limits.max_expanded_bytes:
            raise UnsafeEPUBArchive(
                f"expanded archive exceeds {limits.max_expanded_bytes} bytes"
            )
        if info.file_size:
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > limits.max_compression_ratio:
                raise UnsafeEPUBArchive(
                    f"archive member {name!r} compression ratio {ratio:.1f} exceeds limit"
                )
        if is_directory:
            directory_names.add(name.rstrip("/"))
        else:
            file_names.add(name)
        members.append(ValidatedMember(info, name, is_directory))

    folded_file_names = {
        unicodedata.normalize("NFC", name).casefold() for name in file_names
    }
    folded_directory_names = {
        unicodedata.normalize("NFC", name).casefold() for name in directory_names
    }
    if folded_file_names & folded_directory_names:
        collision = sorted(folded_file_names & folded_directory_names)[0]
        raise UnsafeEPUBArchive(f"file/directory collision rejected: {collision!r}")
    for file_name in file_names:
        if file_name in directory_names:
            raise UnsafeEPUBArchive(f"file/directory collision rejected: {file_name!r}")
        for parent in PurePosixPath(file_name).parents:
            parent_name = parent.as_posix()
            folded_parent = unicodedata.normalize("NFC", parent_name).casefold()
            if parent_name != "." and folded_parent in folded_file_names:
                raise UnsafeEPUBArchive(
                    f"archive member nested below file rejected: {file_name!r}"
                )
    return members


def _contained(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def safe_extract_epub(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    limits: ArchiveLimits = ArchiveLimits(),
) -> list[Path]:
    """Extract a validated EPUB without unsafe bulk extraction helpers."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    observed_total = 0
    with zipfile.ZipFile(source, "r") as archive:
        members = validate_epub_members(archive, limits)
        for member in members:
            target = root.joinpath(*PurePosixPath(member.name).parts)
            resolved_target = target.resolve(strict=False)
            if not _contained(root, resolved_target):
                raise UnsafeEPUBArchive(
                    f"archive member escapes extraction root: {member.name!r}"
                )
            if member.is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise UnsafeEPUBArchive(
                    f"archive member would overwrite an existing path: {member.name!r}"
                )
            observed_member = 0
            target_created = False
            try:
                with archive.open(member.info, "r") as source_handle:
                    with target.open("xb") as output:
                        target_created = True
                        while block := source_handle.read(_COPY_BLOCK_SIZE):
                            observed_member += len(block)
                            observed_total += len(block)
                            if observed_member > limits.max_member_bytes:
                                raise UnsafeEPUBArchive(
                                    f"archive member expanded past limit: {member.name!r}"
                                )
                            if observed_total > limits.max_expanded_bytes:
                                raise UnsafeEPUBArchive(
                                    "archive expanded past total byte limit"
                                )
                            output.write(block)
                if observed_member != member.info.file_size:
                    raise UnsafeEPUBArchive(
                        f"archive member size mismatch: {member.name!r}"
                    )
            except BaseException:
                if target_created:
                    target.unlink(missing_ok=True)
                raise
            extracted.append(target)
    return extracted


def package_epub(
    source_directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Path:
    """Build an EPUB atomically with the mimetype first and uncompressed."""
    root = Path(source_directory).resolve()
    output = Path(output_path)
    mimetype = root / "mimetype"
    if not mimetype.is_file() or mimetype.is_symlink():
        raise UnsafeEPUBArchive("EPUB mimetype file is missing or unsafe")
    if mimetype.read_bytes() != EPUB_MIMETYPE:
        raise UnsafeEPUBArchive("EPUB mimetype content is invalid")

    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    files: list[Path] = []
    expanded = 0
    for path in paths:
        if path.is_symlink():
            raise UnsafeEPUBArchive(
                f"symlink in packaging tree rejected: {path.relative_to(root)}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise UnsafeEPUBArchive(
                f"special path in packaging tree rejected: {path.relative_to(root)}"
            )
        files.append(path)
        size = path.stat().st_size
        if size > limits.max_member_bytes:
            raise UnsafeEPUBArchive(f"packaging member exceeds byte limit: {path}")
        expanded += size
        if expanded > limits.max_expanded_bytes:
            raise UnsafeEPUBArchive("packaging tree exceeds expanded byte limit")
    if len(files) > limits.max_members:
        raise UnsafeEPUBArchive("packaging tree exceeds member limit")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as raw_output:
            with zipfile.ZipFile(raw_output, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
                for path in files:
                    if path == mimetype:
                        continue
                    archive.write(path, path.relative_to(root).as_posix())
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return output


__all__ = [
    "ArchiveLimits",
    "EPUB_MIMETYPE",
    "UnsafeEPUBArchive",
    "package_epub",
    "safe_extract_epub",
    "validate_epub_members",
]
