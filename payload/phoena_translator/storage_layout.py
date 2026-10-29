"""File names of persisted task state: the single owner of the on-disk layout.

All task state lives under the progress directory except the EPUB segment
checkpoint, which sits next to the output it belongs to.  Callers pass the
directory; this module never reads configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

PDF_TASK_SIDECAR_SUFFIXES = ("_layout.json", "_audit.json")


def task_record_path(progress_dir: str | os.PathLike, task_id: str) -> str:
    return os.path.join(progress_dir, f"{task_id}.json")


def pdf_layout_cache_path(progress_dir: str | os.PathLike, task_id: str) -> str:
    return os.path.join(progress_dir, f"{task_id}_layout.json")


def pdf_audit_path(progress_dir: str | os.PathLike, task_id: str) -> str:
    return os.path.join(progress_dir, f"{task_id}_audit.json")


def pdf_page_cache_dir(progress_dir: str | os.PathLike, task_id: str) -> str:
    return os.path.join(progress_dir, f"{task_id}_pages")


def pdf_page_cache_path(progress_dir: str | os.PathLike, task_id: str, page_index: int) -> str:
    return os.path.join(pdf_page_cache_dir(progress_dir, task_id), f"page_{page_index}.json")


def epub_checkpoint_path(output_path: str | os.PathLike) -> Path:
    return Path(f"{os.fspath(output_path)}.segments.json")


def is_pdf_task_sidecar(name: str) -> bool:
    """Whether a progress-directory JSON file is PDF state, not a task record."""
    return name.endswith(PDF_TASK_SIDECAR_SUFFIXES)


__all__ = [
    "PDF_TASK_SIDECAR_SUFFIXES",
    "epub_checkpoint_path",
    "is_pdf_task_sidecar",
    "pdf_audit_path",
    "pdf_layout_cache_path",
    "pdf_page_cache_dir",
    "pdf_page_cache_path",
    "task_record_path",
]
