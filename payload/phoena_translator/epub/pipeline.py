"""Dependency-injected EPUB translation orchestration."""

from __future__ import annotations

import logging
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archive import ArchiveLimits, package_epub, safe_extract_epub
from .segments import (
    BatchValidation,
    TextSegment,
    batch_segments,
    extract_segments,
    make_payload,
    parse_translation_response,
    validate_segment_translation,
)
from ..prompts import PROMPT_RULE_BLOCKS, SEGMENT_SYSTEM_PROMPT
from ..storage_layout import epub_checkpoint_path
from .xhtml import DEFAULT_CHUNK_MAX_BYTES


TranslateText = Callable[..., str]
BuildGlossary = Callable[[str, str], str]
MakePrompt = Callable[[str, str], str]
SaveProgress = Callable[[str, dict[str, Any]], None]
LoadTask = Callable[[str], Mapping[str, Any] | None]


@dataclass(frozen=True)
class EPUBPipelineDependencies:
    translate_text: TranslateText
    build_glossary: BuildGlossary
    make_prompt: MakePrompt
    save_progress: SaveProgress
    load_task: LoadTask
    logger: logging.Logger
    workers: int
    api_concurrency: int
    archive_limits: ArchiveLimits = ArchiveLimits()
    chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES
    max_chunk_rounds: int = 10
    sleep: Callable[[float], None] = time.sleep
    translate_payload: TranslateText | None = None
    # Tolerance budget for segments still failing after the last attempt, with
    # N requested segments: min(max(max_fallback_segments,
    # int(max_fallback_ratio * N)), int(max_fallback_share * N)). The share cap
    # keeps a mostly untranslated book (provider down, tiny book) failed.
    max_fallback_segments: int = 20
    max_fallback_ratio: float = 0.05
    max_fallback_share: float = 0.2

    def __post_init__(self) -> None:
        if self.workers < 1 or self.api_concurrency < 1:
            raise ValueError("EPUB worker and API concurrency must be positive")
        if self.chunk_max_bytes < 1 or self.max_chunk_rounds < 1:
            raise ValueError("EPUB chunk and retry limits must be positive")
        if self.max_fallback_segments < 0 or not self.max_fallback_ratio >= 0:
            raise ValueError("EPUB fallback tolerance must not be negative")
        if not 0 <= self.max_fallback_share <= 1:
            raise ValueError("EPUB fallback share must be between 0 and 1")


# After the last attempt, at most min(max(max_fallback_segments,
# max_fallback_ratio x N), max_fallback_share x N) of the N requested segments
# may still fail. Within that budget a segment whose last structurally valid
# response failed only the long-English rule, with every long English run
# copied from its source (a kept article title), is published leniently; any
# other failure keeps its exact original markup. Structural rules are never
# waived, the task records every such segment plus a warning, the checkpoint
# keeps only accepted translations (so a resume retries them), and exceeding
# the budget still fails the task without publishing output.


@dataclass
class _EPUBRun:
    """Mutable state of translating one extracted EPUB."""

    task_id: str
    root: Path
    total_files: int
    documents: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    completed_files: set[str] = field(default_factory=set)
    skipped_files: list[str] = field(default_factory=list)
    results: dict[str, dict[str, str]] = field(default_factory=dict)
    failures: dict[tuple[str, str], str] = field(default_factory=dict)
    last_quality_reject: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    total_segments: int = 0
    preserved_segments: int = 0
    request_segments: int = 0
    request_count: int = 0
    accepted_count: int = 0
    prompt: str = ""
    signature: str = ""


class EPUBPipeline:
    """Translate one EPUB using bounded archive and injected application services."""

    def __init__(self, dependencies: EPUBPipelineDependencies) -> None:
        self.dependencies = dependencies
        self._coverage: dict[str, Any] = {}
        self._last_progress: dict[str, Any] = {}
        self._checkpoint_path: Path | None = None
        self._segment_sources: dict[str, dict[str, str]] = {}

    def _checkpoint(self, task_id: str, signature: str, results: dict) -> None:
        if self._checkpoint_path is None:
            return
        target = self._checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps({"schema": 2, "signature": signature,
                                        "sources": self._segment_sources,
                                        "results": results}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)

    def _task(self, task_id: str) -> dict[str, Any]:
        return dict(self.dependencies.load_task(task_id) or {})

    def _save(
        self,
        task_id: str,
        *,
        status: str,
        progress: int,
        total: int,
        current_file: str,
        chunk_progress: int,
        chunk_total: int,
        completed_files: set[str] | list[str],
        **extra: Any,
    ) -> None:
        task = self._task(task_id)
        payload: dict[str, Any] = {
            "status": status,
            "progress": progress,
            "total": total,
            "current_file": current_file,
            "chunk_progress": chunk_progress,
            "chunk_total": chunk_total,
            "completed_files": sorted(completed_files),
            "filename": task.get("filename", ""),
            "out_filename": task.get("out_filename", ""),
            "type": "epub",
        }
        payload.update(self._coverage)
        payload.update(extra)
        self._last_progress = payload
        self.dependencies.save_progress(task_id, payload)

    def _translate_extracted(self, task_id: str, root: Path) -> tuple[int, int, set[str]]:
        files = sorted(path for path in root.rglob("*") if path.is_file()
                       and path.suffix.lower() in {".xhtml", ".html", ".htm"})
        if not files:
            raise ValueError("EPUB contains no XHTML documents")
        run = _EPUBRun(task_id=task_id, root=root, total_files=len(files))
        self._coverage = {
            "total_segments": 0, "translated_segments": 0,
            "preserved_segments": 0, "untranslated_segments": 0,
            "failed_segments": [], "failed_segment_count": 0,
            "failure_reasons": [], "skipped_files": [], "coverage": 0.0,
            "source_validation_complete": False, "failed_files": [],
            "fallback_segments": [], "fallback_segment_count": 0,
            "lenient_segments": [], "lenient_segment_count": 0,
            "repaired_segment_count": 0, "warning": None,
        }
        self._save_run(run, "正在提取并校验 XHTML 段落...", error=None, output=None)
        self._extract_documents(run, files)
        # Indexes and references are no longer omitted wholesale: their prose
        # and titles are translated, while names, URLs and numbers can survive.
        self._save_run(run, "正在构建术语表...")
        self._prepare_prompt(run)
        self._reuse_checkpoint(run)
        self._coverage["reused_segments"] = run.accepted_count
        work_queue = self._initial_work_queue(run)
        self._coverage["initial_batches"] = len(work_queue)
        max_attempts = min(self.dependencies.max_chunk_rounds, 4)
        for attempt in range(max_attempts):
            if not work_queue:
                break
            work_queue = self._run_attempt(run, work_queue, attempt)
        if work_queue:
            self._publish_within_tolerance(run, max_attempts)
        self._save_run(run)
        return run.total_files, run.request_segments, run.completed_files

    def _save_run(self, run: _EPUBRun, current_file: str = "", *, status: str = "translating",
                  **extra: Any) -> None:
        pending = run.total_segments - run.preserved_segments - run.accepted_count
        failures = run.failures
        self._coverage.update({
            "total_segments": run.total_segments,
            "translated_segments": run.accepted_count,
            "preserved_segments": run.preserved_segments,
            "untranslated_segments": max(0, pending),
            "failed_segment_count": len(failures),
            "failed_segments": [f"{relative}#{sid}" for relative, sid in sorted(failures)[:250]],
            "failed_segments_truncated": len(failures) > 250,
            "failure_reasons": [
                {"file": relative, "segment": sid, "reason": failures[(relative, sid)][:240]}
                for relative, sid in sorted(failures)[:100]
            ],
            "skipped_files": sorted(run.skipped_files),
            "skipped_file_count": len(run.skipped_files),
            "coverage": ((run.preserved_segments + run.accepted_count) / run.total_segments
                         if run.total_segments else float(len(run.documents) == run.total_files)),
            "source_validation_complete": len(run.documents) == run.total_files,
            "translation_requests": run.request_count,
        })
        self._save(
            run.task_id, status=status,
            progress=len(run.completed_files) + len(run.skipped_files), total=run.total_files,
            current_file=current_file, chunk_progress=run.accepted_count,
            chunk_total=run.request_segments, completed_files=run.completed_files, **extra,
        )

    def _extract_documents(self, run: _EPUBRun, files: list[Path]) -> None:
        """Parse every XHTML file into segments; any invalid file fails the task."""
        for path in files:
            relative = path.relative_to(run.root).as_posix()
            try:
                document = extract_segments(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError, OSError) as exc:
                self._coverage["failed_files"] = [relative]
                self._save_run(run, status="failed", failed_files=[relative],
                               error=f"{relative}: {str(exc)[:500]}", output=None)
                raise ValueError(f"{relative}: {str(exc)[:500]}") from exc
            run.documents[relative], run.paths[relative] = document, path
            run.total_segments += len(document.segments)
            run.results[relative] = {
                segment.id: segment.source for segment in document.segments
                if not segment.needs_request
            }
            run.preserved_segments += len(run.results[relative])
            run.request_segments += sum(segment.needs_request for segment in document.segments)
            if not document.segments:
                run.skipped_files.append(relative)
            elif len(run.results[relative]) == len(document.segments):
                run.completed_files.add(relative)
            self.dependencies.logger.info(
                "[%s] %s: %s semantic segments", run.task_id, relative, len(document.segments))

    def _prepare_prompt(self, run: _EPUBRun) -> None:
        """Build the glossary prompt and the checkpoint signature it implies."""
        dependencies = self.dependencies
        # has_words (not needs_request) keeps the glossary input, and so the
        # checkpoint signature, identical to releases before the single-letter
        # exemption.
        glossary_source = "\n".join(
            segment.plain_text for document in run.documents.values()
            for segment in document.segments if segment.has_words
        )
        glossary = (dependencies.build_glossary(glossary_source, run.task_id)
                    if run.request_segments else "")
        run.prompt = dependencies.make_prompt(SEGMENT_SYSTEM_PROMPT + PROMPT_RULE_BLOCKS, glossary)
        run.signature = hashlib.sha256((run.prompt + "\n" + "\n".join(
            relative + ":" + hashlib.sha256(document.source.encode()).hexdigest()
            for relative, document in run.documents.items()
        )).encode()).hexdigest()
        self._segment_sources = {
            relative: {segment.id: hashlib.sha256((segment.source + "\0" + segment.original).encode()).hexdigest()
                       for segment in document.segments}
            for relative, document in run.documents.items()
        }

    def _reuse_checkpoint(self, run: _EPUBRun) -> None:
        """Adopt still-valid translations from a checkpoint with the same signature."""
        if self._checkpoint_path is None or not self._checkpoint_path.is_file():
            return
        logger = self.dependencies.logger
        try:
            checkpoint = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("schema") == 2 and checkpoint.get("signature") == run.signature:
                for relative, document in run.documents.items():
                    cached = checkpoint.get("results", {}).get(relative, {})
                    for segment in document.segments:
                        if not segment.needs_request or segment.id not in cached:
                            continue
                        if (checkpoint.get("sources", {}).get(relative, {}).get(segment.id)
                                != self._segment_sources[relative][segment.id]):
                            continue
                        try:
                            validate_segment_translation(segment, cached[segment.id])
                        except (TypeError, ValueError):
                            continue
                        run.results[relative][segment.id] = cached[segment.id]
                        run.accepted_count += 1
                    if len(run.results[relative]) == len(document.segments) and document.segments:
                        run.paths[relative].write_text(document.render(run.results[relative]), encoding="utf-8")
                        run.completed_files.add(relative)
                logger.info("[%s] Reused %s validated EPUB segments", run.task_id, run.accepted_count)
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("[%s] Could not read EPUB checkpoint: %s", run.task_id, exc)

    def _initial_work_queue(self, run: _EPUBRun) -> list[tuple[str, tuple[TextSegment, ...]]]:
        work_queue: list[tuple[str, tuple[TextSegment, ...]]] = []
        for relative, document in run.documents.items():
            requested = [segment for segment in document.segments
                         if segment.needs_request and segment.id not in run.results[relative]]
            work_queue.extend((relative, batch) for batch in
                              batch_segments(requested, min(self.dependencies.chunk_max_bytes, 12_000)))
        return work_queue

    def _translate_batch(self, run: _EPUBRun, relative: str, batch: tuple[TextSegment, ...],
                         attempt: int) -> BatchValidation:
        dependencies = self.dependencies
        logger = dependencies.logger
        translator = dependencies.translate_payload or dependencies.translate_text
        logger.info("[%s] %s: translating %s segments (attempt %s)",
                    run.task_id, relative, len(batch), attempt + 1)
        retry_hint = ("\n上次响应未通过校验。请完整翻译这一段，严格保留占位符，"
                      "返回且只返回指定 ID 的 JSON 对象。" if attempt else "")
        if attempt:
            retry_hint += "\n校验失败原因：" + run.failures.get((relative, batch[0].id), "未通过校验")
        try:
            response = translator(make_payload(batch), system_prompt=run.prompt + retry_hint)
            return parse_translation_response(response, batch)
        except Exception as exc:
            # Transport retries belong to the injected gateway; this layer
            # has a small bounded attempt budget and never sleeps for a
            # semantic rejection.
            reason = f"translation adapter failed: {str(exc)[:200]}"
            logger.warning("[%s] %s: %s", run.task_id, relative, reason)
            return BatchValidation(failures={segment.id: reason for segment in batch})

    def _run_attempt(self, run: _EPUBRun, work_queue: list, attempt: int) -> list:
        """Translate every queued batch once; return single-segment retries."""
        dependencies = self.dependencies
        retry_queue: list[tuple[str, tuple[TextSegment, ...]]] = []
        with ThreadPoolExecutor(max_workers=min(dependencies.workers, dependencies.api_concurrency)) as executor:
            pending = {
                executor.submit(self._translate_batch, run, relative, batch, attempt): (relative, batch)
                for relative, batch in work_queue
            }
            run.request_count += len(pending)
            for future in as_completed(pending):
                relative, batch = pending[future]
                verdict = future.result()
                run.results[relative].update(verdict.translations)
                run.accepted_count += len(verdict.translations)
                self._coverage["repaired_segment_count"] += len(
                    verdict.repaired.intersection(verdict.translations))
                self._checkpoint(run.task_id, run.signature, run.results)
                for segment in batch:
                    key = (relative, segment.id)
                    if segment.id in verdict.translations:
                        run.failures.pop(key, None)
                        run.last_quality_reject.pop(key, None)
                    else:
                        run.failures[key] = verdict.failures.get(segment.id, "missing validated result")
                        if segment.id in verdict.quality_rejects:
                            run.last_quality_reject[key] = verdict.quality_rejects[segment.id]
                        retry_queue.append((relative, (segment,)))
                document = run.documents[relative]
                if (relative not in run.completed_files
                        and len(run.results[relative]) == len(document.segments)):
                    rendered = document.render(run.results[relative])
                    run.paths[relative].write_text(rendered, encoding="utf-8")
                    run.completed_files.add(relative)
                self._save_run(run, f"翻译中 ({run.accepted_count}/{run.request_segments} 段落，"
                               f"{len(run.completed_files) + len(run.skipped_files)}/{run.total_files} 文件)")
        # Preserve accepted segments. Every remaining retry is one complete
        # paragraph, so a bad batch cannot make unrelated work run again.
        return retry_queue

    def _publish_within_tolerance(self, run: _EPUBRun, max_attempts: int) -> None:
        """Publish leftover failures as fallbacks, or fail beyond the budget."""
        dependencies = self.dependencies
        failures = run.failures
        floor = dependencies.max_fallback_segments
        ratio_term = int(dependencies.max_fallback_ratio * run.request_segments)
        share_cap = int(dependencies.max_fallback_share * run.request_segments)
        budget = min(max(floor, ratio_term), share_cap)
        if len(failures) > budget:
            message = (f"EPUB translation failed: {len(failures)} segment(s) did not pass "
                       f"validation after {max_attempts} attempts (tolerance budget {budget} = "
                       f"min(max({floor}, {ratio_term}), {share_cap}) for {run.request_segments} "
                       "requested segments); no output published")
            self._save_run(run, status="failed", error=message, output=None)
            raise ValueError(message)
        # Within budget. Nothing below reaches the checkpoint: it keeps
        # only accepted translations, so a resume retries these segments.
        by_id = {relative: {segment.id: segment for segment in document.segments}
                 for relative, document in run.documents.items()}
        tolerated: dict[str, set[str]] = {}
        lenient: list[dict[str, str]] = []
        fallback: list[dict[str, str]] = []
        for relative, sid in sorted(failures):
            key = (relative, sid)
            source_text = by_id[relative][sid].source
            text, reason = run.last_quality_reject.get(key, (source_text, failures[key]))
            run.results[relative][sid] = text
            # An echoed source is reported as kept original, not as a translation.
            (fallback if text == source_text else lenient).append(
                {"segment": f"{relative}#{sid}", "reason": reason[:240]})
            tolerated.setdefault(relative, set()).add(sid)
        for relative, document in run.documents.items():
            if relative in run.completed_files or not document.segments:
                continue
            rendered = document.render(run.results[relative], lenient_ids=tolerated.get(relative, ()))
            run.paths[relative].write_text(rendered, encoding="utf-8")
            run.completed_files.add(relative)
        parts = [f"{len(fallback)}段保留原文"] if fallback else []
        if lenient:
            parts.append(f"{len(lenient)}段接受质量存疑译文")
        warning = (f"{len(failures)}段未通过校验：" + "，".join(parts)
                   + "；可用“重试未通过段落”再试")
        self._coverage.update({
            "fallback_segments": fallback[:250], "fallback_segment_count": len(fallback),
            "lenient_segments": lenient[:250], "lenient_segment_count": len(lenient),
            "warning": warning,
        })
        dependencies.logger.warning(
            "[%s] EPUB published within tolerance budget %s after %s attempts: "
            "source fallback %s; lenient quality %s", run.task_id, budget, max_attempts,
            "; ".join(f"{item['segment']} ({item['reason']})" for item in fallback) or "none",
            "; ".join(f"{item['segment']} ({item['reason']})" for item in lenient) or "none",
        )

    def _update_opf_language(self, task_id: str, root: Path) -> None:
        opf_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".opf"
        )
        for path in opf_paths:
            try:
                content = path.read_text(encoding="utf-8")
                updated = re.sub(
                    r"<dc:language>[^<]*</dc:language>",
                    "<dc:language>zh</dc:language>",
                    content,
                )
                path.write_text(updated, encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                self.dependencies.logger.warning(
                    "[%s] Could not update OPF language in %s: %s",
                    task_id,
                    path,
                    exc,
                )

    def translate(self, task_id: str, source_path: str, output_path: str) -> bool:
        dependencies = self.dependencies
        self._coverage = {}
        self._last_progress = {}
        self._checkpoint_path = epub_checkpoint_path(output_path)
        temporary_root = Path(tempfile.mkdtemp(prefix="epub_"))
        try:
            dependencies.logger.info(
                "[%s] Starting EPUB translation (workers=%s, api_concurrency=%s)",
                task_id,
                dependencies.workers,
                dependencies.api_concurrency,
            )
            safe_extract_epub(
                source_path,
                temporary_root,
                limits=dependencies.archive_limits,
            )
            total_files, total_chunks, completed_files = self._translate_extracted(
                task_id, temporary_root
            )
            self._update_opf_language(task_id, temporary_root)
            package_epub(
                temporary_root,
                output_path,
                limits=dependencies.archive_limits,
            )
            self._save(
                task_id,
                status="completed",
                progress=total_files,
                total=total_files,
                current_file="",
                chunk_progress=total_chunks,
                chunk_total=total_chunks,
                completed_files=completed_files,
                output=output_path,
            )
            dependencies.logger.info("[%s] EPUB translation completed", task_id)
            return True
        except Exception as exc:
            dependencies.logger.error(
                "[%s] EPUB translation error: %s", task_id, traceback.format_exc()
            )
            task = {**self._task(task_id), **self._last_progress}
            self._save(
                task_id,
                status="failed",
                progress=int(task.get("progress", 0) or 0),
                total=int(task.get("total", 0) or 0),
                current_file="",
                chunk_progress=int(task.get("chunk_progress", 0) or 0),
                chunk_total=int(task.get("chunk_total", 0) or 0),
                completed_files=list(task.get("completed_files", []) or []),
                error=str(exc)[:1000],
                output=None,
            )
            return False
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = [
    "EPUBPipeline",
    "EPUBPipelineDependencies",
]
