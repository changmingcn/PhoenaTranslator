"""HTTP contract of the Flask route layer, pinned before refactoring it."""

from __future__ import annotations

import io
import threading
import zipfile

import pytest

from phoena_translator.config import AppConfig
from phoena_translator.task_store import TaskStore
from phoena_translator.web import WebDependencies, create_app


class Harness:
    def __init__(self, tmp_path, *, accept=True, max_file_size=1024 * 1024):
        self.uploads, self.outputs, self.progress = (
            tmp_path / name for name in ("uploads", "outputs", "progress")
        )
        for directory in (self.uploads, self.outputs, self.progress):
            directory.mkdir()
        self.config = AppConfig(
            upload_dir=self.uploads,
            output_dir=self.outputs,
            progress_dir=self.progress,
            log_file=tmp_path / "log",
            max_file_size=max_file_size,
            max_request_size=max_file_size * 2,
        )
        self.store = TaskStore(self.progress, self.uploads, self.outputs)
        self.tasks: dict[str, dict] = {}
        self.saved: list[tuple[str, dict]] = []
        self.queued: list[tuple] = []
        self.runtime_calls = 0
        self.accept = accept
        lock = threading.Lock()

        def save(task_id, data):
            self.saved.append((task_id, dict(data)))
            self.store.save(task_id, data)
            self.tasks[task_id] = dict(data)

        def enqueue(*args):
            self.queued.append(args)
            return self.accept

        def ensure_runtime():
            self.runtime_calls += 1

        self.client = create_app(
            self.config,
            WebDependencies(
                tasks=lambda: self.tasks,
                tasks_lock=lambda: lock,
                ensure_runtime=ensure_runtime,
                save_progress=save,
                load_progress=lambda task_id: self.store.load(task_id),
                normalize_task=lambda task_id, data: self.store.normalize(task_id, data),
                enqueue_task=enqueue,
                task_store=lambda: self.store,
            ),
        ).test_client()


def _epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
    return buffer.getvalue()


def _upload(harness, name, payload, **form):
    return harness.client.post(
        "/api/upload",
        data={"file": (io.BytesIO(payload), name), **form},
        content_type="multipart/form-data",
    )


def test_health_does_not_initialize_runtime(tmp_path):
    harness = Harness(tmp_path)
    assert harness.client.get("/api/health").get_json() == {"ok": True}
    assert harness.runtime_calls == 0
    harness.client.get("/api/tasks")
    assert harness.runtime_calls == 1


def test_app_without_dependencies_only_serves_health(tmp_path):
    client = create_app(Harness(tmp_path).config).test_client()
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/tasks").status_code == 404


@pytest.mark.parametrize(
    ("data", "status", "error"),
    [
        ({}, 400, "No file provided"),
        ({"file": (io.BytesIO(b"x"), "")}, 400, "Empty filename"),
        ({"file": (io.BytesIO(b"x"), "notes.txt")}, 400, "Unsupported format. Allowed: epub, pdf"),
        ({"file": (io.BytesIO(b"not a zip"), "book.epub")}, 400, "Invalid EPUB file (not a valid zip archive)"),
    ],
)
def test_upload_rejections(tmp_path, data, status, error):
    harness = Harness(tmp_path)
    response = harness.client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == status
    assert response.get_json() == {"error": error}
    assert harness.queued == [] and harness.tasks == {}
    assert list(harness.uploads.iterdir()) == []


def test_upload_rejects_oversized_file(tmp_path):
    harness = Harness(tmp_path, max_file_size=2048)
    response = _upload(harness, "big.pdf", b"x" * 4096)
    assert response.status_code == 413
    assert list(harness.uploads.iterdir()) == []


def test_upload_records_and_enqueues_task(tmp_path):
    harness = Harness(tmp_path)
    response = _upload(harness, "paper.pdf", b"%PDF-1.4 body", password="secret")
    assert response.status_code == 200
    body = response.get_json()
    task_id = body["task_id"]
    assert body["filename"] == "paper.pdf"
    record = harness.tasks[task_id]
    assert record["status"] == "queued" and record["out_filename"] == "zh_paper.pdf"
    assert record["requires_password"] is True and "secret" not in str(record)
    assert harness.queued == [(
        task_id,
        "pdf",
        str(harness.uploads / f"{task_id}.pdf"),
        str(harness.outputs / f"{task_id}.pdf"),
        "secret",
        len(b"%PDF-1.4 body"),
    )]
    assert harness.store.load(task_id)["filename"] == "paper.pdf"


def test_upload_keeps_existing_zh_prefix_and_accepts_epub(tmp_path):
    harness = Harness(tmp_path)
    response = _upload(harness, "zh_book.epub", _epub_bytes())
    task_id = response.get_json()["task_id"]
    assert harness.tasks[task_id]["out_filename"] == "zh_book.epub"
    assert harness.tasks[task_id]["requires_password"] is False


def test_upload_rolls_back_when_queue_is_full(tmp_path):
    harness = Harness(tmp_path, accept=False)
    response = _upload(harness, "paper.pdf", b"%PDF-1.4")
    assert response.status_code == 503
    assert response.get_json() == {"error": "Translation queue is at capacity"}
    assert harness.tasks == {}
    assert list(harness.uploads.iterdir()) == []
    assert list(harness.progress.glob("*.json")) == []


def _seed(harness, task_id, record, *, source=None, output=None):
    harness.tasks[task_id] = dict(record)
    harness.store.save(task_id, record)
    ext = record.get("type", "pdf")
    if source is not None:
        (harness.uploads / f"{task_id}.{ext}").write_bytes(source)
    if output is not None:
        (harness.outputs / f"{task_id}.{ext}").write_bytes(output)


def test_status_hides_output_path_and_reports_missing_task(tmp_path):
    harness = Harness(tmp_path)
    _seed(harness, "t1", {"type": "pdf", "status": "completed", "output": "/secret/path.pdf", "filename": "a.pdf"})
    body = harness.client.get("/api/status/t1").get_json()
    assert "output" not in body and body["status"] == "completed"
    assert harness.client.get("/api/status/missing").status_code == 404
    assert harness.client.get("/api/status/..").status_code == 404


def test_status_falls_back_to_persisted_record(tmp_path):
    harness = Harness(tmp_path)
    harness.store.save("t2", {"type": "epub", "status": "failed", "filename": "b.epub"})
    assert harness.client.get("/api/status/t2").get_json()["status"] == "failed"


def test_download_rules(tmp_path):
    harness = Harness(tmp_path)
    _seed(harness, "run", {"type": "pdf", "status": "translating"})
    assert harness.client.get("/api/download/run").status_code == 400
    assert harness.client.get("/api/download/none").status_code == 404
    _seed(harness, "bad", {"type": "pdf", "status": "completed", "output": "/etc/passwd"})
    assert harness.client.get("/api/download/bad").status_code == 409
    _seed(harness, "gone", {"type": "pdf", "status": "completed"})
    assert harness.client.get("/api/download/gone").get_json() == {"error": "Output file not found"}
    _seed(
        harness,
        "ok",
        {"type": "pdf", "status": "completed", "filename": "paper.pdf", "out_filename": "zh_\r\npa/per.pdf"},
        output=b"%PDF translated",
    )
    response = harness.client.get("/api/download/ok")
    assert response.status_code == 200 and response.data == b"%PDF translated"
    assert "per.pdf" in response.headers["Content-Disposition"]
    assert "\r" not in response.headers["Content-Disposition"]


def test_delete_rules(tmp_path):
    harness = Harness(tmp_path)
    assert harness.client.delete("/api/task/none").status_code == 404
    _seed(harness, "busy", {"type": "pdf", "status": "queued"})
    assert harness.client.delete("/api/task/busy").status_code == 400
    _seed(harness, "bad", {"type": "pdf", "status": "failed", "output": "/etc/passwd"})
    assert harness.client.delete("/api/task/bad").status_code == 409
    _seed(harness, "done", {"type": "pdf", "status": "completed"}, source=b"src", output=b"out")
    pages = harness.store.page_cache_dir("done")
    pages.mkdir()
    (pages / "page_0.json").write_text("{}")
    assert harness.client.delete("/api/task/done").get_json() == {"ok": True}
    assert "done" not in harness.tasks
    assert not (harness.uploads / "done.pdf").exists()
    assert not (harness.outputs / "done.pdf").exists()
    assert not pages.exists()
    assert not harness.store.progress_path("done").exists()


def test_pdf_password_resume(tmp_path):
    harness = Harness(tmp_path)
    _seed(harness, "enc", {"type": "pdf", "status": "needs_password"}, source=b"%PDF")
    assert harness.client.post("/api/task/enc/resume", json={}).status_code == 400
    response = harness.client.post("/api/task/enc/resume", json={"password": "pw"})
    assert response.get_json() == {"ok": True}
    assert harness.tasks["enc"]["status"] == "queued"
    assert harness.queued[-1] == ("enc", "pdf", str(harness.uploads / "enc.pdf"), str(harness.outputs / "enc.pdf"), "pw", 4)


def test_pdf_password_resume_rejections(tmp_path):
    harness = Harness(tmp_path, accept=False)
    _seed(harness, "plain", {"type": "pdf", "status": "failed"})
    assert harness.client.post("/api/task/plain/resume", json={"password": "pw"}).status_code == 400
    _seed(harness, "lost", {"type": "pdf", "status": "needs_password"})
    assert harness.client.post("/api/task/lost/resume", json={"password": "pw"}).status_code == 409
    _seed(harness, "full", {"type": "pdf", "status": "needs_password"}, source=b"%PDF")
    response = harness.client.post("/api/task/full/resume", json={"password": "pw"})
    assert response.status_code == 503
    assert harness.tasks["full"]["status"] == "needs_password"
    assert harness.tasks["full"]["current_file"] == ""


def test_epub_resume_rolls_back_when_queue_is_full(tmp_path):
    harness = Harness(tmp_path, accept=False)
    record = {"type": "epub", "status": "failed", "filename": "b.epub", "error": "boom"}
    _seed(harness, "book", record, source=b"zip")
    response = harness.client.post("/api/task/book/resume", json={})
    assert response.status_code == 503
    assert harness.tasks["book"]["status"] == "failed"
    assert harness.tasks["book"]["error"] == "boom"
    _seed(harness, "lost", {"type": "epub", "status": "failed"})
    assert harness.client.post("/api/task/lost/resume", json={}).status_code == 409


def test_list_tasks_returns_every_task_without_output(tmp_path):
    # Flask serialises with sorted keys; the UI orders tasks itself.
    harness = Harness(tmp_path)
    harness.tasks.update({
        "old": {"type": "pdf", "status": "completed", "created_at": 10.0, "output": "/x"},
        "new": {"type": "epub", "status": "queued", "created_at": 20.0},
        "tie": {"type": "pdf", "status": "failed", "created_at": 10.0, "updated_at": 30.0},
    })
    body = harness.client.get("/api/tasks").get_json()
    assert sorted(body) == ["new", "old", "tie"]
    assert body["tie"]["updated_at"] == 30.0
    assert all("output" not in item for item in body.values())
