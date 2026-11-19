"""Restart recovery and translation-queue orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from phoena_translator.task_store import TaskRepository
from phoena_translator.workers import TaskJob, TaskWorkerPool

FINAL_STATUSES = ("completed", "failed")
# Called with (task_id, final record, raised error or None) after the final
# record is saved; it must not raise or block (see task_notify.TaskNotifier).
TaskFinishedHook = Callable[[str, dict, BaseException | None], None]


@dataclass(frozen=True)
class TaskRecoveryDependencies:
    tasks: TaskRepository
    upload_dir: Path
    output_dir: Path
    logger: logging.Logger
    enqueue_task: Callable[[str, str, str, str], bool]
    source_sha256: Callable[[str], str]
    load_cached_page_indices: Callable[[str, str], set[int]]
    task_finished: TaskFinishedHook | None = None


@dataclass(frozen=True)
class TranslationJobDependencies:
    logger: logging.Logger
    load_progress: Callable[[str], dict | None]
    translate_epub: Callable[[str, str, str], Any]
    translate_pdf: Callable[..., Any]
    task_finished: TaskFinishedHook | None = None


def _report_finished(
    hook: TaskFinishedHook | None,
    task_id: str,
    record: dict | None,
    error: BaseException | None,
    logger: logging.Logger,
) -> None:
    """Hand a final record to the hook; a hook failure never reaches the task."""
    if hook is None or not record or record.get("status") not in FINAL_STATUSES:
        return
    try:
        hook(task_id, record, error)
    except Exception as exc:
        logger.warning("task-notify: [%s] skipped error=%s", task_id, type(exc).__name__)


def resume_interrupted_tasks(dependencies: TaskRecoveryDependencies) -> None:
    """Requeue interrupted work without retaining PDF passwords in state."""

    to_resume = [
        (task_id, data)
        for task_id, data in dependencies.tasks.snapshot().items()
        if data.get("status") in ("translating", "queued")
    ]

    for task_id, data in to_resume:
        extension = data.get("type", "epub")
        source_path = dependencies.upload_dir / f"{task_id}.{extension}"
        output_path = dependencies.output_dir / f"{task_id}.{extension}"
        if not source_path.exists():
            dependencies.logger.warning(
                "[%s] Cannot resume: upload file missing", task_id
            )
            saved = dependencies.tasks.save(
                task_id,
                {
                    **data,
                    "status": "failed",
                    "error": "Upload file lost after restart",
                },
            )
            _report_finished(
                dependencies.task_finished, task_id, saved, None, dependencies.logger
            )
            continue

        if extension == "pdf":
            if data.get("requires_password"):
                data.update(
                    status="needs_password",
                    current_file="",
                    error="PDF password must be re-entered after restart",
                )
                dependencies.tasks.save(task_id, data)
                dependencies.logger.info(
                    "[%s] Encrypted PDF awaits password re-entry after restart",
                    task_id,
                )
                continue
            source_hash = dependencies.source_sha256(str(source_path))
            cached_indices = dependencies.load_cached_page_indices(
                task_id,
                source_hash,
            )
            if cached_indices:
                total = data.get("total", 0)
                dependencies.logger.info(
                    "[%s] Resuming interrupted pdf translation from cache "
                    "(%s/%s pages cached)",
                    task_id,
                    len(cached_indices),
                    total or "?",
                )
                data["completed_files"] = [
                    str(index) for index in sorted(cached_indices)
                ]
                data["progress"] = len(cached_indices)
                data["current_file"] = (
                    f"准备续跑 ({len(cached_indices)}/{total or '?'})"
                )
            else:
                dependencies.logger.info(
                    "[%s] Resuming interrupted pdf translation with no cached pages",
                    task_id,
                )
                data["completed_files"] = []
                data["progress"] = 0
        else:
            dependencies.logger.info(
                "[%s] Resuming interrupted %s translation (starting fresh)",
                task_id,
                extension,
            )
            data["completed_files"] = []
            data["progress"] = 0

        data["status"] = "queued"
        data["current_file"] = "等待任务队列执行..."
        dependencies.tasks.save(task_id, data)
        accepted = dependencies.enqueue_task(
            task_id,
            extension,
            str(source_path),
            str(output_path),
        )
        if not accepted:
            dependencies.tasks.save(
                task_id,
                {
                    **data,
                    "status": "paused",
                    "current_file": "",
                    "error": "Translation queue capacity reached during recovery",
                },
            )


def run_translation_job(
    job: TaskJob,
    worker_id: int,
    dependencies: TranslationJobDependencies,
) -> None:
    data = dependencies.load_progress(job.task_id) or {}
    if data.get("status") in FINAL_STATUSES:
        dependencies.logger.info(
            "[%s] Worker %s: task already %s, skipping",
            job.task_id,
            worker_id,
            data.get("status"),
        )
        return
    dependencies.logger.info(
        "[%s] Worker %s: starting %s translation",
        job.task_id,
        worker_id,
        job.extension,
    )
    raised: BaseException | None = None
    try:
        if job.extension == "epub":
            dependencies.translate_epub(
                job.task_id,
                job.source_path,
                job.output_path,
            )
            return
        dependencies.translate_pdf(
            job.task_id,
            job.source_path,
            job.output_path,
            job.pdf_password,
            _forced_source_page_fallbacks=data.get("source_page_fallbacks"),
        )
    except BaseException as error:
        raised = error
        raise
    finally:
        # The pipelines save their own final record. A job that raised before
        # saving one is recorded (and reported) by the worker pool's
        # mark_failed; a job that ends non-final (queued for restart) is not.
        if dependencies.task_finished is not None:
            try:
                record = dependencies.load_progress(job.task_id)
            except Exception:
                record = None
            _report_finished(
                dependencies.task_finished,
                job.task_id,
                record,
                raised,
                dependencies.logger,
            )


def enqueue_translation_task(
    task_id: str,
    extension: str,
    source_path: str,
    output_path: str,
    pdf_password: str,
    file_size: int | None,
    pool: TaskWorkerPool,
    task_max_concurrency: int,
    logger: logging.Logger,
) -> bool:
    if file_size is None:
        try:
            file_size = Path(source_path).stat().st_size
        except OSError:
            file_size = 0
    accepted = pool.enqueue(
        TaskJob(
            task_id=task_id,
            extension=extension,
            source_path=source_path,
            output_path=output_path,
            pdf_password=pdf_password,
        ),
        size_bytes=file_size,
    )
    if accepted:
        active_count, queued_count = pool.counts()
        logger.info(
            "[%s] Queued %s translation job "
            "(active=%s, queued=%s, task_concurrency=%s)",
            task_id,
            extension,
            active_count,
            queued_count,
            task_max_concurrency,
        )
    return accepted
