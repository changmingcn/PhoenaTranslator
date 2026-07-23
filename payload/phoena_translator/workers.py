"""Deterministic admission controls shared by task producers and workers."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AdmissionSnapshot:
    task_count: int
    pending_bytes: int
    max_tasks: int
    max_pending_bytes: int


class AdmissionController:
    def __init__(self, *, max_tasks: int, max_pending_bytes: int) -> None:
        if max_tasks < 1 or max_pending_bytes < 1:
            raise ValueError("admission limits must be positive")
        self._max_tasks = max_tasks
        self._max_pending_bytes = max_pending_bytes
        self._reservations: dict[str, int] = {}
        self._lock = threading.Lock()

    def try_reserve(self, task_id: str, size_bytes: int) -> bool:
        if size_bytes < 0:
            return False
        with self._lock:
            if task_id in self._reservations:
                return True
            if len(self._reservations) >= self._max_tasks:
                return False
            if sum(self._reservations.values()) + size_bytes > self._max_pending_bytes:
                return False
            self._reservations[task_id] = size_bytes
            return True

    def release(self, task_id: str) -> None:
        with self._lock:
            self._reservations.pop(task_id, None)

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            return AdmissionSnapshot(
                task_count=len(self._reservations),
                pending_bytes=sum(self._reservations.values()),
                max_tasks=self._max_tasks,
                max_pending_bytes=self._max_pending_bytes,
            )


@dataclass(frozen=True)
class TaskJob:
    task_id: str
    extension: str
    source_path: str
    output_path: str
    pdf_password: str = ""


class TaskWorkerPool:
    """Bounded worker pool with duplicate suppression and byte admission."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_queue: int,
        max_pending_bytes: int,
        handler: Callable[[TaskJob, int], None],
        logger: logging.Logger,
        mark_failed: Callable[[str, str], None] | None = None,
    ) -> None:
        if max_workers < 1 or max_queue < 1:
            raise ValueError("worker and queue limits must be positive")
        self._max_workers = max_workers
        self._mark_failed = mark_failed
        self._queue: queue.Queue[TaskJob] = queue.Queue(maxsize=max_queue)
        self._admission = AdmissionController(
            max_tasks=max_queue + max_workers,
            max_pending_bytes=max_pending_bytes,
        )
        self._handler = handler
        self._logger = logger
        self._lock = threading.Lock()
        self._queued_ids: set[str] = set()
        self._active_ids: set[str] = set()
        self._started = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        for index in range(self._max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(index + 1,),
                daemon=True,
                name=f"phoena-translation-{index + 1}",
            )
            thread.start()
        self._logger.info("Started %s translation task worker(s)", self._max_workers)

    def enqueue(self, job: TaskJob, *, size_bytes: int) -> bool:
        if not self._admission.try_reserve(job.task_id, size_bytes):
            self._logger.warning(
                "[%s] Translation admission rejected by task/byte limits",
                job.task_id,
            )
            return False
        self.start()
        with self._lock:
            if job.task_id in self._queued_ids or job.task_id in self._active_ids:
                return True
            self._queued_ids.add(job.task_id)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._queued_ids.discard(job.task_id)
            self._admission.release(job.task_id)
            return False
        return True

    def _worker_loop(self, worker_id: int) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                self._queued_ids.discard(job.task_id)
                self._active_ids.add(job.task_id)
            try:
                self._handler(job, worker_id)
            except Exception as handler_error:
                self._logger.exception(
                    "[%s] Worker %s: unhandled translation error",
                    job.task_id,
                    worker_id,
                )
                # The pipelines record their own terminal states; this is the
                # last resort so a crashed job can never stay "translating"
                # forever (which would also make it undeletable).
                if self._mark_failed is not None:
                    try:
                        self._mark_failed(job.task_id, str(handler_error))
                    except Exception:
                        self._logger.exception(
                            "[%s] Worker %s: could not record crash failure",
                            job.task_id,
                            worker_id,
                        )
            finally:
                with self._lock:
                    self._active_ids.discard(job.task_id)
                self._admission.release(job.task_id)
                self._queue.task_done()

    def counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._active_ids), len(self._queued_ids)
