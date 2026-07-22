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


def _safe_download_name(value: str, fallback: str) -> str:
    value = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    value = "".join(char for char in value if ord(char) >= 32 and char not in "\r\n")
    return value[:240] or fallback


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

    def _record(task_id: str) -> dict | None:
        try:
            with dependencies.tasks_lock():
                data = dependencies.tasks().get(task_id)
            return data or dependencies.load_progress(task_id)
        except InvalidTaskId:
            return None

    @app.post("/api/upload")
    def upload_file():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Empty filename"}), 400
        ext = uploaded.filename.rsplit(".", 1)[-1].lower() if "." in uploaded.filename else ""
        if ext not in config.allowed_extensions:
            allowed = ", ".join(sorted(config.allowed_extensions))
            return jsonify({"error": f"Unsupported format. Allowed: {allowed}"}), 400

        task_id = str(uuid.uuid4())
        filename = uploaded.filename
        source_path = config.upload_dir / f"{task_id}.{ext}"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(source_path)
        file_size = source_path.stat().st_size
        if file_size > config.max_file_size:
            source_path.unlink(missing_ok=True)
            return jsonify({"error": "File size cannot exceed configured limit"}), 413
        if ext == "epub" and not zipfile.is_zipfile(source_path):
            source_path.unlink(missing_ok=True)
            return jsonify({"error": "Invalid EPUB file (not a valid zip archive)"}), 400

        pdf_password = request.form.get("password", "") or ""
        output_filename = filename if filename.startswith("zh_") else f"zh_{filename}"
        output_path = config.output_dir / f"{task_id}.{ext}"
        created_at = time.time()
        record = {
            "status": "queued",
            "progress": 0,
            "total": 0,
            "current_file": "",
            "completed_files": [],
            "filename": filename,
            "out_filename": output_filename,
            "type": ext,
            "created_at": created_at,
            "updated_at": created_at,
            "requires_password": bool(pdf_password),
        }
        with dependencies.tasks_lock():
            dependencies.tasks()[task_id] = record
        dependencies.save_progress(task_id, record)
        accepted = dependencies.enqueue_task(
            task_id,
            ext,
            str(source_path),
            str(output_path),
            pdf_password,
            file_size,
        )
        if not accepted:
            source_path.unlink(missing_ok=True)
            dependencies.task_store().progress_path(task_id).unlink(missing_ok=True)
            with dependencies.tasks_lock():
                dependencies.tasks().pop(task_id, None)
            return jsonify({"error": "Translation queue is at capacity"}), 503
        return jsonify({"task_id": task_id, "filename": filename}), 200

    @app.get("/api/status/<task_id>")
    def task_status(task_id: str):
        data = _record(task_id)
        if not data:
            return jsonify({"error": "Task not found"}), 404
        safe = {key: value for key, value in data.items() if key not in {"output"}}
        return jsonify(safe), 200

    @app.get("/api/download/<task_id>")
    def download_file(task_id: str):
        data = _record(task_id)
        if not data:
            return jsonify({"error": "Task not found"}), 404
        if data.get("status") != "completed":
            return jsonify({"error": "Translation not yet completed"}), 400
        output_path = dependencies.task_store().resolve_output_path(task_id, data)
        if output_path is None:
            return jsonify({"error": "Unsafe output path in task record"}), 409
        if not output_path.is_file():
            return jsonify({"error": "Output file not found"}), 404
        fallback = f"zh_translated.{data.get('type', 'epub')}"
        download_name = _safe_download_name(
            data.get("out_filename") or f"zh_{data.get('filename', '')}",
            fallback,
        )
        return send_file(output_path, as_attachment=True, download_name=download_name)

    @app.delete("/api/task/<task_id>")
    def delete_task(task_id: str):
        data = _record(task_id)
        if not data:
            return jsonify({"error": "Task not found"}), 404
        if data.get("status") in {"queued", "translating"}:
            return jsonify({"error": "Cannot delete an active task"}), 400
        store = dependencies.task_store()
        if data.get("output") and store.resolve_output_path(task_id, data) is None:
            return jsonify({"error": "Unsafe output path in task record"}), 409
        try:
            paths = store.artifact_paths(task_id, data)
        except InvalidTaskId:
            return jsonify({"error": "Task not found"}), 404
        for path in paths:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        with dependencies.tasks_lock():
            dependencies.tasks().pop(task_id, None)
        return jsonify({"ok": True}), 200

    @app.post("/api/task/<task_id>/resume")
    def resume_task(task_id: str):
        data = _record(task_id)
        if not data:
            return jsonify({"error": "Task not found"}), 404
        if data.get("status") != "needs_password":
            return jsonify({"error": "Task does not require password resumption"}), 400
        payload = request.get_json(silent=True)
        password = payload.get("password", "") if isinstance(payload, dict) else ""
        if not password:
            return jsonify({"error": "PDF password is required"}), 400
        source_path = config.upload_dir / f"{task_id}.pdf"
        output_path = config.output_dir / f"{task_id}.pdf"
        if not source_path.is_file():
            return jsonify({"error": "Upload file lost after restart"}), 409
        queued = dict(data)
        queued.update({"status": "queued", "current_file": "等待任务队列执行..."})
        dependencies.save_progress(task_id, queued)
        with dependencies.tasks_lock():
            dependencies.tasks()[task_id] = queued
        accepted = dependencies.enqueue_task(
            task_id,
            "pdf",
            str(source_path),
            str(output_path),
            str(password),
            source_path.stat().st_size,
        )
        if not accepted:
            queued.update({"status": "needs_password", "current_file": ""})
            dependencies.save_progress(task_id, queued)
            with dependencies.tasks_lock():
                dependencies.tasks()[task_id] = queued
            return jsonify({"error": "Translation queue is at capacity"}), 503
        return jsonify({"ok": True}), 200

    @app.get("/api/tasks")
    def list_tasks():
        with dependencies.tasks_lock():
            entries = [
                (task_id, dependencies.normalize_task(task_id, data) or {})
                for task_id, data in dependencies.tasks().items()
            ]
        entries.sort(
            key=lambda item: (
                float(item[1].get("created_at") or 0.0),
                float(item[1].get("updated_at") or 0.0),
                item[0],
            ),
            reverse=True,
        )
        result = {
            task_id: {
                key: value
                for key, value in data.items()
                if key not in {"output"}
            }
            for task_id, data in entries
        }
        return jsonify(result), 200

    return app
