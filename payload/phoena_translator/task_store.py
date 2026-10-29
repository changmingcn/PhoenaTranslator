"""Root-contained, backward-compatible JSON task persistence and its in-memory index."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .storage_layout import (
    epub_checkpoint_path,
    is_pdf_task_sidecar,
    pdf_audit_path,
    pdf_layout_cache_path,
    pdf_page_cache_dir,
    task_record_path,
)


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PRESERVED_FIELDS = ("filename", "out_filename", "type")


class InvalidTaskId(ValueError):
    pass


def _validate_task_id(task_id: str) -> str:
    if (
        not isinstance(task_id, str)
        or not _TASK_ID_RE.fullmatch(task_id)
        or ".." in task_id
    ):
        raise InvalidTaskId("task_id contains unsupported path characters")
    return task_id


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


class TaskStore:
    def __init__(self, progress_dir: Path, upload_dir: Path, output_dir: Path) -> None:
        self.progress_dir = Path(progress_dir).expanduser().resolve(strict=False)
        self.upload_dir = Path(upload_dir).expanduser().resolve(strict=False)
        self.output_dir = Path(output_dir).expanduser().resolve(strict=False)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def task_lock(self, task_id: str) -> threading.RLock:
        """The re-entrant lock that serialises every read and write of one task."""
        task_id = _validate_task_id(task_id)
        with self._locks_guard:
            return self._locks.setdefault(task_id, threading.RLock())

    def progress_path(self, task_id: str) -> Path:
        return Path(task_record_path(self.progress_dir, _validate_task_id(task_id)))

    def layout_cache_path(self, task_id: str) -> Path:
        return Path(pdf_layout_cache_path(self.progress_dir, _validate_task_id(task_id)))

    def audit_path(self, task_id: str) -> Path:
        return Path(pdf_audit_path(self.progress_dir, _validate_task_id(task_id)))

    def page_cache_dir(self, task_id: str) -> Path:
        return Path(pdf_page_cache_dir(self.progress_dir, _validate_task_id(task_id)))

    @staticmethod
    def _timestamp(value: Any) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def infer_created_at(self, task_id: str, data: Mapping[str, Any]) -> float:
        ext = str(data.get("type") or "")
        candidates = [self.progress_path(task_id)]
        if ext:
            candidates.extend(
                [
                    self.upload_dir / f"{task_id}.{ext}",
                    self.output_dir / f"{task_id}.{ext}",
                ]
            )
        timestamps: list[float] = []
        for path in candidates:
            try:
                if path.exists():
                    timestamps.append(path.stat().st_mtime)
            except OSError:
                continue
        return min(timestamps) if timestamps else time.time()

    def normalize(self, task_id: str, data: Mapping[str, Any] | None) -> dict | None:
        if not data:
            return None
        _validate_task_id(task_id)
        record = dict(data)
        created = self._timestamp(record.get("created_at"))
        if created is None:
            created = self.infer_created_at(task_id, record)
        updated = self._timestamp(record.get("updated_at"))
        record["created_at"] = created
        record["updated_at"] = updated if updated is not None else created
        return record

    def load(self, task_id: str) -> dict | None:
        path = self.progress_path(task_id)
        with self.task_lock(task_id):
            try:
                content = path.read_text(encoding="utf-8")
                payload = json.loads(content) if content.strip() else None
            except (OSError, json.JSONDecodeError):
                return None
        return self.normalize(task_id, payload)

    def save(
        self,
        task_id: str,
        data: Mapping[str, Any],
        *,
        existing: Mapping[str, Any] | None = None,
    ) -> dict:
        path = self.progress_path(task_id)
        with self.task_lock(task_id):
            prior = dict(existing or self.load(task_id) or {})
            merged = dict(data)
            for key in _PRESERVED_FIELDS:
                if key not in merged and prior.get(key) is not None:
                    merged[key] = prior[key]
            if "created_at" not in merged and prior.get("created_at") is not None:
                merged["created_at"] = prior["created_at"]
            merged = self.normalize(task_id, merged) or {}
            merged["updated_at"] = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_name = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary_name = handle.name
                    json.dump(merged, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, path)
                temporary_name = ""
            finally:
                if temporary_name:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
            return merged

    def load_all(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if not self.progress_dir.is_dir():
            return result
        for path in sorted(self.progress_dir.glob("*.json")):
            if is_pdf_task_sidecar(path.name):
                continue
            task_id = path.stem
            try:
                record = self.load(task_id)
            except InvalidTaskId:
                continue
            if record:
                result[task_id] = record
        return result

    def resolve_output_path(self, task_id: str, record: Mapping[str, Any]) -> Path | None:
        ext = str(record.get("type") or "epub")
        raw = record.get("output")
        candidate = Path(str(raw)) if raw else self.output_dir / f"{_validate_task_id(task_id)}.{ext}"
        if not candidate.is_absolute():
            candidate = self.output_dir / candidate
        candidate = candidate.expanduser().resolve(strict=False)
        if not _contains(self.output_dir, candidate):
            return None
        return candidate

    def artifact_paths(self, task_id: str, record: Mapping[str, Any]) -> list[Path]:
        task_id = _validate_task_id(task_id)
        ext = str(record.get("type") or "epub")
        paths = [
            self.upload_dir / f"{task_id}.{ext}",
            self.progress_path(task_id),
            self.layout_cache_path(task_id),
            self.audit_path(task_id),
            self.page_cache_dir(task_id),
        ]
        output = self.resolve_output_path(task_id, record)
        if output is not None:
            paths.insert(1, output)
            if ext == "epub":
                checkpoint = epub_checkpoint_path(output)
                paths.append(checkpoint)
                paths.append(Path(f"{checkpoint}.tmp"))
        return paths


class TaskRepository:
    """The in-memory task index, kept in step with the records in its TaskStore.

    A save holds the task's store lock across the disk write and the index
    update, so concurrent saves of one task cannot leave the index older than
    the file.  Reads return copies; the repository is the index's only writer.
    """

    def __init__(self, store: TaskStore, records: dict[str, dict] | None = None) -> None:
        self.store = store
        self._records = {} if records is None else records
        self._lock = threading.Lock()

    def get(self, task_id: str) -> dict | None:
        """Return the indexed record, falling back to the persisted one."""
        with self._lock:
            record = self._records.get(task_id)
        return dict(record) if record else self.store.load(task_id)

    def load(self, task_id: str) -> dict | None:
        """Return the persisted record, bypassing the index."""
        return self.store.load(task_id)

    def save(self, task_id: str, data: Mapping[str, Any]) -> dict:
        with self.store.task_lock(task_id):
            with self._lock:
                existing = self._records.get(task_id)
            merged = self.store.save(task_id, data, existing=existing)
            with self._lock:
                self._records[task_id] = merged
        return dict(merged)

    def remove(self, task_id: str) -> None:
        """Drop the task from the index and delete its persisted record."""
        with self.store.task_lock(task_id):
            self.store.progress_path(task_id).unlink(missing_ok=True)
            with self._lock:
                self._records.pop(task_id, None)

    def load_all(self) -> None:
        records = self.store.load_all()
        with self._lock:
            self._records.update(records)

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {task_id: dict(record) for task_id, record in self._records.items()}
