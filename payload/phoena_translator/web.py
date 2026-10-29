"""Flask app factory and public HTTP route layer."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Callable, MutableMapping

from flask import Flask, jsonify, request, send_file

from .config import AppConfig
from .task_store import InvalidTaskId, TaskStore


@dataclass(frozen=True)
class WebDependencies:
    tasks: Callable[[], MutableMapping[str, dict]]
    tasks_lock: Callable[[], threading.Lock]
    ensure_runtime: Callable[[], None]
    save_progress: Callable[[str, dict], None]
    load_progress: Callable[[str], dict | None]
    normalize_task: Callable[[str, dict | None], dict | None]
    enqueue_task: Callable[[str, str, str, str, str, int], bool]
    task_store: Callable[[], TaskStore]


QUEUE_FULL = ({"error": "Translation queue is at capacity"}, 503)
NOT_FOUND = ({"error": "Task not found"}, 404)
UNSAFE_OUTPUT = ({"error": "Unsafe output path in task record"}, 409)
UPLOAD_LOST = ({"error": "Upload file lost after restart"}, 409)
_HIDDEN_FIELDS = frozenset({"output"})


def _failed_segment_count(data: dict) -> int:
    try:
        return int(data.get("failed_segment_count") or 0)
    except (TypeError, ValueError):
        return 0


def _safe_download_name(value: str, fallback: str) -> str:
    value = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    value = "".join(char for char in value if ord(char) >= 32 and char not in "\r\n")
    return value[:240] or fallback


def _public_view(data: dict) -> dict:
    return {key: value for key, value in data.items() if key not in _HIDDEN_FIELDS}


class _TaskRoutes:
    """Request handlers; each returns ``(json_body, status)`` or a Flask response."""

    def __init__(self, config: AppConfig, dependencies: WebDependencies) -> None:
        self.config = config
        self.deps = dependencies

    # -- shared task-registry helpers ------------------------------------

    def record(self, task_id: str) -> dict | None:
        try:
            with self.deps.tasks_lock():
                data = self.deps.tasks().get(task_id)
            return data or self.deps.load_progress(task_id)
        except InvalidTaskId:
            return None

    def publish(self, task_id: str, record: dict) -> None:
        self.deps.save_progress(task_id, record)
        with self.deps.tasks_lock():
            self.deps.tasks()[task_id] = record

    def queue_or_restore(
        self, task_id: str, queued: dict, restore: dict, ext: str, password: str
    ) -> bool:
        """Publish ``queued`` and enqueue the stored source; on refusal publish ``restore``."""
        source_path = self.config.upload_dir / f"{task_id}.{ext}"
        output_path = self.config.output_dir / f"{task_id}.{ext}"
        self.publish(task_id, queued)
        accepted = self.deps.enqueue_task(
            task_id, ext, str(source_path), str(output_path), password,
            source_path.stat().st_size,
        )
        if not accepted:
            self.publish(task_id, restore)
        return accepted

    # -- routes ------------------------------------------------------------

    def upload(self):
        if "file" not in request.files:
            return {"error": "No file provided"}, 400
        uploaded = request.files["file"]
        if not uploaded.filename:
            return {"error": "Empty filename"}, 400
        filename = uploaded.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in self.config.allowed_extensions:
            allowed = ", ".join(sorted(self.config.allowed_extensions))
            return {"error": f"Unsupported format. Allowed: {allowed}"}, 400

        task_id = str(uuid.uuid4())
        source_path = self.config.upload_dir / f"{task_id}.{ext}"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(source_path)
        file_size = source_path.stat().st_size
        if file_size > self.config.max_file_size:
            source_path.unlink(missing_ok=True)
            return {"error": "File size cannot exceed configured limit"}, 413
        if ext == "epub" and not zipfile.is_zipfile(source_path):
            source_path.unlink(missing_ok=True)
            return {"error": "Invalid EPUB file (not a valid zip archive)"}, 400

        pdf_password = request.form.get("password", "") or ""
        created_at = time.time()
        record = {
            "status": "queued",
            "progress": 0,
            "total": 0,
            "current_file": "",
            "completed_files": [],
            "filename": filename,
            "out_filename": filename if filename.startswith("zh_") else f"zh_{filename}",
            "type": ext,
            "created_at": created_at,
            "updated_at": created_at,
            "requires_password": bool(pdf_password),
        }
        with self.deps.tasks_lock():
            self.deps.tasks()[task_id] = record
        self.deps.save_progress(task_id, record)
        accepted = self.deps.enqueue_task(
            task_id,
            ext,
            str(source_path),
            str(self.config.output_dir / f"{task_id}.{ext}"),
            pdf_password,
            file_size,
        )
        if not accepted:
            source_path.unlink(missing_ok=True)
            self.deps.task_store().progress_path(task_id).unlink(missing_ok=True)
            with self.deps.tasks_lock():
                self.deps.tasks().pop(task_id, None)
            return QUEUE_FULL
        return {"task_id": task_id, "filename": filename}, 200

    def status(self, task_id: str):
        data = self.record(task_id)
        if not data:
            return NOT_FOUND
        return _public_view(data), 200

    def download(self, task_id: str):
        data = self.record(task_id)
        if not data:
            return NOT_FOUND
        if data.get("status") != "completed":
            return {"error": "Translation not yet completed"}, 400
        output_path = self.deps.task_store().resolve_output_path(task_id, data)
        if output_path is None:
            return UNSAFE_OUTPUT
        if not output_path.is_file():
            return {"error": "Output file not found"}, 404
        download_name = _safe_download_name(
            data.get("out_filename") or f"zh_{data.get('filename', '')}",
            f"zh_translated.{data.get('type', 'epub')}",
        )
        return send_file(output_path, as_attachment=True, download_name=download_name)

    def delete(self, task_id: str):
        data = self.record(task_id)
        if not data:
            return NOT_FOUND
        if data.get("status") in {"queued", "translating"}:
            return {"error": "Cannot delete an active task"}, 400
        store = self.deps.task_store()
        if data.get("output") and store.resolve_output_path(task_id, data) is None:
            return UNSAFE_OUTPUT
        try:
            paths = store.artifact_paths(task_id, data)
        except InvalidTaskId:
            return NOT_FOUND
        for path in paths:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        with self.deps.tasks_lock():
            self.deps.tasks().pop(task_id, None)
        return {"ok": True}, 200

    def resume(self, task_id: str):
        data = self.record(task_id)
        if not data:
            return NOT_FOUND
        # A completed EPUB that published fallback or lenient segments can be
        # resumed too: the checkpoint retries only the segments that failed.
        if data.get("type") == "epub" and (
            data.get("status") == "failed"
            or (data.get("status") == "completed" and _failed_segment_count(data) > 0)
        ):
            return self._resume_epub(task_id, data)
        if data.get("status") != "needs_password":
            return {"error": "Task does not require password resumption"}, 400
        return self._resume_encrypted_pdf(task_id, data)

    def _resume_epub(self, task_id: str, data: dict):
        if not (self.config.upload_dir / f"{task_id}.epub").is_file():
            return UPLOAD_LOST
        queued = dict(data)
        queued.update({"status": "queued", "current_file": "正在恢复已通过校验的译文...",
                       "error": None, "output": None, "warning": None})
        if not self.queue_or_restore(task_id, queued, data, "epub", ""):
            return QUEUE_FULL
        return {"ok": True}, 200

    def _resume_encrypted_pdf(self, task_id: str, data: dict):
        payload = request.get_json(silent=True)
        password = payload.get("password", "") if isinstance(payload, dict) else ""
        if not password:
            return {"error": "PDF password is required"}, 400
        if not (self.config.upload_dir / f"{task_id}.pdf").is_file():
            return UPLOAD_LOST
        queued = dict(data)
        queued.update({"status": "queued", "current_file": "等待任务队列执行..."})
        restore = {**queued, "status": "needs_password", "current_file": ""}
        if not self.queue_or_restore(task_id, queued, restore, "pdf", str(password)):
            return QUEUE_FULL
        return {"ok": True}, 200

    def list_all(self):
        with self.deps.tasks_lock():
            entries = [
                (task_id, self.deps.normalize_task(task_id, data) or {})
                for task_id, data in self.deps.tasks().items()
            ]
        # Flask serialises objects with sorted keys; the UI orders by created_at.
        return {task_id: _public_view(data) for task_id, data in entries}, 200


def _json_route(handler):
    def view(*args, **kwargs):
        result = handler(*args, **kwargs)
        if isinstance(result, tuple):
            body, status = result
            return jsonify(body), status
        return result

    view.__name__ = handler.__name__
    return view


def create_app(
    config: AppConfig,
    dependencies: WebDependencies | None = None,
) -> Flask:
    """Create an offline-importable WSGI app; runtime work is request-lifecycle gated."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_request_size

    @app.before_request
    def _initialize_api_runtime():
        if (
            dependencies is not None
            and request.path.startswith("/api/")
            and request.path != "/api/health"
        ):
            dependencies.ensure_runtime()
        return None

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True}), 200

    if dependencies is None:
        return app

    routes = _TaskRoutes(config, dependencies)
    app.add_url_rule("/api/upload", "upload_file", _json_route(routes.upload), methods=["POST"])
    app.add_url_rule("/api/status/<task_id>", "task_status", _json_route(routes.status), methods=["GET"])
    app.add_url_rule("/api/download/<task_id>", "download_file", _json_route(routes.download), methods=["GET"])
    app.add_url_rule("/api/task/<task_id>", "delete_task", _json_route(routes.delete), methods=["DELETE"])
    app.add_url_rule("/api/task/<task_id>/resume", "resume_task", _json_route(routes.resume), methods=["POST"])
    app.add_url_rule("/api/tasks", "list_tasks", _json_route(routes.list_all), methods=["GET"])
    return app
