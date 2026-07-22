"""Dependency-injected EPUB translation orchestration."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import ArchiveLimits, package_epub, safe_extract_epub
from .math_protection import (
    protect_xhtml_math_fragments,
    restore_xhtml_math_fragments,
)
from .xhtml import (
    DEFAULT_CHUNK_MAX_BYTES,
    fix_xhtml_entities,
    fix_xhtml_tags,
    is_appendix_xhtml,
    markup_tokens,
    merge_translated_chunks,
    split_xhtml_to_chunks,
)


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
    system_prompt_xhtml: str
    workers: int
    api_concurrency: int
    archive_limits: ArchiveLimits = ArchiveLimits()
    chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES
    max_chunk_rounds: int = 10
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.workers < 1 or self.api_concurrency < 1:
            raise ValueError("EPUB worker and API concurrency must be positive")
        if self.chunk_max_bytes < 1 or self.max_chunk_rounds < 1:
            raise ValueError("EPUB chunk and retry limits must be positive")


def translate_single_chunk(
    task_id: str,
    relative_path: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    xhtml_prompt: str,
    *,
    translate_text: TranslateText,
    logger: logging.Logger,
) -> tuple[str, int, str | None]:
    """Translate one XHTML chunk and fail closed to source on math corruption."""
    try:
        protected_chunk, math_records = protect_xhtml_math_fragments(chunk_text)
        logger.info(
            "[%s] %s chunk %s/%s (%s UTF-8 bytes) — sending to translation adapter",
            task_id,
            relative_path,
            chunk_index + 1,
            total_chunks,
            len(chunk_text.encode("utf-8")),
        )
        translated = translate_text(
            "请将以下XHTML文件内容翻译为中文，保留所有HTML标签结构不变，只翻译文本内容："
            f"\n\n{protected_chunk}",
            system_prompt=xhtml_prompt,
        )
        translated = fix_xhtml_entities(translated)
        translated = fix_xhtml_tags(translated)
        restored = restore_xhtml_math_fragments(translated, math_records)
        if restored is None:
            logger.error(
                "[%s] %s chunk %s/%s — math placeholder integrity failed; using source",
                task_id,
                relative_path,
                chunk_index + 1,
                total_chunks,
            )
            return relative_path, chunk_index, chunk_text
        if markup_tokens(restored) != markup_tokens(chunk_text):
            logger.error(
                "[%s] %s chunk %s/%s — XHTML markup integrity failed; using source",
                task_id,
                relative_path,
                chunk_index + 1,
                total_chunks,
            )
            return relative_path, chunk_index, chunk_text
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", restored))
        logger.info(
            "[%s] %s chunk %s/%s — OK, %s Chinese chars",
            task_id,
            relative_path,
            chunk_index + 1,
            total_chunks,
            chinese_count,
        )
        return relative_path, chunk_index, restored
    except Exception as exc:
        logger.error(
            "[%s] %s chunk %s/%s — failed: %s",
            task_id,
            relative_path,
            chunk_index + 1,
            total_chunks,
            exc,
        )
        return relative_path, chunk_index, None


class EPUBPipeline:
    """Translate one EPUB using bounded archive and injected application services."""

    def __init__(self, dependencies: EPUBPipelineDependencies) -> None:
        self.dependencies = dependencies

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
        payload.update(extra)
        self.dependencies.save_progress(task_id, payload)

    def _translate_extracted(self, task_id: str, root: Path) -> tuple[int, int, set[str]]:
        dependencies = self.dependencies
        logger = dependencies.logger
        xhtml_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".xhtml", ".html", ".htm"}
        )
        total_files = len(xhtml_files)
        logger.info("[%s] Found %s XHTML files", task_id, total_files)
        self._save(
            task_id,
            status="translating",
            progress=0,
            total=total_files,
            current_file="正在构建术语表...",
            chunk_progress=0,
            chunk_total=0,
            completed_files=[],
        )

        file_info: dict[str, dict[str, Any]] = {}
        source_text: list[str] = []
        for path in xhtml_files:
            relative = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8")
            text_only = re.sub(r"<[^>]+>", "", content).strip()
            skip = is_appendix_xhtml(relative, content) or len(text_only) <= 20
            chunks = (
                []
                if skip
                else split_xhtml_to_chunks(
                    content,
                    dependencies.chunk_max_bytes,
                    logger=logger,
                )
            )
            file_info[relative] = {
                "path": path,
                "original": content,
                "chunks": chunks,
                "skip": skip,
            }
            if skip:
                logger.info("[%s] Skipping non-translatable XHTML: %s", task_id, relative)
            else:
                source_text.append(text_only)
                logger.info(
                    "[%s] %s: %s chunks", task_id, relative, len(chunks)
                )

        glossary = dependencies.build_glossary("\n".join(source_text), task_id)
        xhtml_prompt = dependencies.make_prompt(
            dependencies.system_prompt_xhtml, glossary
        )
        chunk_results: dict[tuple[str, int], str] = {}
        chunk_attempts: dict[tuple[str, int], int] = {}
        completed_files = {
            relative for relative, info in file_info.items() if info["skip"]
        }
        work_queue: list[tuple[str, int, str]] = []
        for relative, info in file_info.items():
            for chunk_index, chunk in enumerate(info["chunks"]):
                work_queue.append((relative, chunk_index, chunk))
                chunk_attempts[(relative, chunk_index)] = 0
        total_chunks = len(work_queue)

        self._save(
            task_id,
            status="translating",
            progress=len(completed_files),
            total=total_files,
            current_file=(
                f"翻译中 (0/{total_chunks} chunks, 线程 {dependencies.workers}/"
                f"API {dependencies.api_concurrency})"
            ),
            chunk_progress=0,
            chunk_total=total_chunks,
            completed_files=completed_files,
        )

        def try_merge_file(relative: str) -> bool:
            info = file_info[relative]
            results = [
                chunk_results.get((relative, index))
                for index in range(len(info["chunks"]))
            ]
            if any(result is None for result in results):
                return False
            translated = merge_translated_chunks(results)  # type: ignore[arg-type]
            info["path"].write_text(translated, encoding="utf-8")
            completed_files.add(relative)
            logger.info("[%s] File merged: %s", task_id, relative)
            return True

        def update_progress() -> None:
            self._save(
                task_id,
                status="translating",
                progress=len(completed_files),
                total=total_files,
                current_file=(
                    f"翻译中 ({len(chunk_results)}/{total_chunks} chunks, "
                    f"{len(completed_files)}/{total_files} 文件)"
                ),
                chunk_progress=len(chunk_results),
                chunk_total=total_chunks,
                completed_files=completed_files,
            )

        for round_number in range(dependencies.max_chunk_rounds):
            if not work_queue:
                break
            if round_number:
                delay = min(30 * round_number, 120)
                logger.info(
                    "[%s] Chunk retry round %s: %s chunks, waiting %ss",
                    task_id,
                    round_number,
                    len(work_queue),
                    delay,
                )
                dependencies.sleep(delay)
            failed: list[tuple[str, int, str]] = []
            with ThreadPoolExecutor(max_workers=dependencies.workers) as executor:
                future_to_item = {}
                for relative, chunk_index, chunk_text in work_queue:
                    total_for_file = len(file_info[relative]["chunks"])
                    key = (relative, chunk_index)
                    chunk_attempts[key] = chunk_attempts.get(key, 0) + 1
                    future = executor.submit(
                        translate_single_chunk,
                        task_id,
                        relative,
                        chunk_index,
                        total_for_file,
                        chunk_text,
                        xhtml_prompt,
                        translate_text=dependencies.translate_text,
                        logger=logger,
                    )
                    future_to_item[future] = (relative, chunk_index, chunk_text)
                for future in as_completed(future_to_item):
                    relative, chunk_index, chunk_text = future_to_item[future]
                    try:
                        _relative, _index, translated = future.result()
                    except Exception as exc:
                        logger.error(
                            "[%s] %s chunk %s future failed: %s",
                            task_id,
                            relative,
                            chunk_index + 1,
                            exc,
                        )
                        translated = None
                    if translated is None:
                        failed.append((relative, chunk_index, chunk_text))
                    else:
                        chunk_results[(relative, chunk_index)] = translated
                        try_merge_file(relative)
                    update_progress()
            work_queue = [
                item
                for item in failed
                if chunk_attempts[(item[0], item[1])]
                < dependencies.max_chunk_rounds
            ]
            for relative, chunk_index, chunk_text in failed:
                key = (relative, chunk_index)
                if (
                    chunk_attempts[key] >= dependencies.max_chunk_rounds
                    and key not in chunk_results
                ):
                    logger.error(
                        "[%s] %s chunk %s exhausted retries; using source",
                        task_id,
                        relative,
                        chunk_index + 1,
                    )
                    chunk_results[key] = chunk_text
                    try_merge_file(relative)

        for relative, info in file_info.items():
            if relative in completed_files or info["skip"]:
                continue
            for chunk_index, chunk in enumerate(info["chunks"]):
                chunk_results.setdefault((relative, chunk_index), chunk)
            try_merge_file(relative)
        update_progress()
        return total_files, total_chunks, completed_files

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
            task = self._task(task_id)
            self._save(
                task_id,
                status="failed",
                progress=int(task.get("progress", 0) or 0),
                total=int(task.get("total", 0) or 0),
                current_file="",
                chunk_progress=int(task.get("chunk_progress", 0) or 0),
                chunk_total=int(task.get("chunk_total", 0) or 0),
                completed_files=list(task.get("completed_files", []) or []),
                error=str(exc),
            )
            return False
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = [
    "EPUBPipeline",
    "EPUBPipelineDependencies",
    "translate_single_chunk",
]
