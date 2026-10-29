"""Restart recovery and translation-queue orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from phoena_translator.task_store import TaskRepository
from phoena_translator.workers import TaskJob, TaskWorkerPool


@dataclass(frozen=True)
class TaskRecoveryDependencies:
    tasks: TaskRepository
    upload_dir: Path
    output_dir: Path
    logger: logging.Logger
    enqueue_task: Callable[[str, str, str, str], bool]
    source_sha256: Callable[[str], str]
    load_cached_page_indices: Callable[[str, str], set[int]]


@dataclass(frozen=True)
class TranslationJobDependencies:
    logger: logging.Logger
    load_progress: Callable[[str], dict | None]
    translate_epub: Callable[[str, str, str], Any]
    translate_pdf: Callable[..., Any]


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
            dependencies.tasks.save(
                task_id,
                {
                    **data,
                    "status": "failed",
                    "error": "Upload file lost after restart",
                },
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
    if data.get("status") in ("completed", "failed"):
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
