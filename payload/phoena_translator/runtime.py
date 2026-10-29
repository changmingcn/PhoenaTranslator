"""Composition root: every process-wide service, built once from one AppConfig.

Constructing a :class:`Runtime` performs no I/O: provider clients connect
lazily and worker threads start with the first enqueued job.  Directory
creation, logging, task loading and restart recovery happen in
:meth:`Runtime.initialize`, which the web layer calls before the first API
request.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from functools import partial

from phoena_translator.api_runtime import APIUsagePrices, LLMGateway, strip_think_tags
from phoena_translator.config import AppConfig
from phoena_translator.epub.archive import ArchiveLimits
from phoena_translator.epub.pipeline import EPUBPipeline, EPUBPipelineDependencies
from phoena_translator.epub.xhtml import DEFAULT_CHUNK_MAX_BYTES
from phoena_translator.glossary import build_glossary_text, load_explicit_glossary
from phoena_translator.llm import (
    ClaudeSettings,
    ClaudeTranslationAdapter,
    DeepSeekTranslationAdapter,
    LLMSettings,
)
from phoena_translator.logging_setup import configure_logging
from phoena_translator.pdf import pipeline as pdf_pipeline
from phoena_translator.pdf.cache import (
    _load_pdf_cached_page_indices,
    _pdf_source_sha256,
    configure_pdf_cache_progress_dir,
)
from phoena_translator.pdf.extract.page import _extract_page_elements
from phoena_translator.pdf.geometry import _filter_pdf_formula_safe_rects
from phoena_translator.pdf.progress import _save_pdf_translation_progress
from phoena_translator.pdf.render.fonts import pdf_font_subsetting_available
from phoena_translator.pdf.semantics.formula_policy import _summarize_pdf_formula_protection
from phoena_translator.pdf.translate import translator as text_translator
from phoena_translator.pdf.verify.output import (
    PDFAuditDependencies,
    _check_pdf_output_structure,
    _check_pdf_output_structure_serialized,
)
from phoena_translator.prompts import SYSTEM_PROMPT_TEXT, with_glossary
from phoena_translator.task_runtime import (
    TaskRecoveryDependencies,
    TranslationJobDependencies,
    enqueue_translation_task,
    resume_interrupted_tasks,
    run_translation_job,
)
from phoena_translator.task_store import TaskRepository, TaskStore
from phoena_translator.web import WebDependencies, create_app
from phoena_translator.workers import TaskJob, TaskWorkerPool

log = logging.getLogger("translator")

# The ledger kept its original name when Claude became a second provider.
USAGE_LEDGER_NAME = "deepseek_usage.jsonl"


def trim_process_memory() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_llm_gateway(config: AppConfig, logger: logging.Logger) -> LLMGateway:
    """Select the configured provider: OpenAI-compatible DeepSeek (default) or Claude."""
    if config.llm_provider == "claude":
        adapter = ClaudeTranslationAdapter(
            ClaudeSettings(
                api_key=config.anthropic_api_key,
                model=config.claude_model,
                effort=config.claude_effort,
                fallbacks=config.claude_fallbacks,
                prompt_cache=config.claude_prompt_cache,
                timeout_seconds=config.claude_timeout_seconds,
            ),
            logger=logger,
        )
        model = config.claude_model
        prices = APIUsagePrices(
            cache_hit_usd_per_million=config.claude_price_cache_read_usd_per_m,
            cache_miss_usd_per_million=config.claude_price_input_usd_per_m,
            output_usd_per_million=config.claude_price_output_usd_per_m,
            cache_write_usd_per_million=config.claude_price_cache_write_usd_per_m,
        )
    else:
        adapter = DeepSeekTranslationAdapter(
            LLMSettings(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                model=config.deepseek_model,
            )
        )
        model = config.deepseek_model
        prices = APIUsagePrices(
            cache_hit_usd_per_million=config.deepseek_price_cache_hit_usd_per_m,
            cache_miss_usd_per_million=config.deepseek_price_cache_miss_usd_per_m,
            output_usd_per_million=config.deepseek_price_output_usd_per_m,
        )
    return LLMGateway(
        adapter,
        model=model,
        prices=prices,
        usage_log=config.progress_dir / USAGE_LEDGER_NAME,
        max_concurrency=config.api_max_concurrency,
        logger=logger,
    )


class Runtime:
    """LLM gateway, task repository, worker pool and pipeline wiring for one process.

    The ``text_translation``, ``epub`` and ``pdf`` dependency bundles are built
    once; tests substitute a collaborator with ``dataclasses.replace`` on the
    relevant bundle instead of monkeypatching module globals.
    """

    def __init__(self, config: AppConfig, *, gateway: LLMGateway | None = None) -> None:
        self.config = config
        self.gateway = gateway or build_llm_gateway(config, log)
        self.tasks = TaskRepository(
            TaskStore(config.progress_dir, config.upload_dir, config.output_dir)
        )
        self.worker_pool = TaskWorkerPool(
            max_workers=config.task_max_concurrency,
            max_queue=config.task_queue_max,
            max_pending_bytes=config.pending_bytes_max,
            handler=self._run_job,
            logger=log,
            mark_failed=self._mark_failed_after_crash,
        )
        self._init_lock = threading.Lock()
        self._initialized = False

        extraction_semaphore = threading.BoundedSemaphore(
            config.pdf_extraction_max_concurrency
        )
        audit = PDFAuditDependencies(
            extraction_max_concurrency=config.pdf_extraction_max_concurrency,
            extraction_semaphore=extraction_semaphore,
            check_output_structure=_check_pdf_output_structure,
            trim_process_memory=trim_process_memory,
            save_progress=self.tasks.save,
        )
        self.text_translation = text_translator.PDFTranslationDependencies(
            create_chat_completion=self.gateway.complete,
            strip_think_tags=strip_think_tags,
            system_prompt_text=SYSTEM_PROMPT_TEXT,
            logger=log,
        )
        self.epub = EPUBPipelineDependencies(
            translate_text=self.translate_text,
            translate_payload=self._translate_epub_payload,
            build_glossary=self.build_glossary,
            make_prompt=with_glossary,
            save_progress=self.tasks.save,
            load_task=self.tasks.load,
            logger=log,
            workers=config.translation_workers,
            api_concurrency=config.api_max_concurrency,
            archive_limits=ArchiveLimits(
                max_members=config.epub_max_members,
                max_expanded_bytes=config.epub_max_expanded_bytes,
                max_member_bytes=config.epub_max_member_bytes,
                max_compression_ratio=config.epub_max_compression_ratio,
            ),
            chunk_max_bytes=DEFAULT_CHUNK_MAX_BYTES,
        )
        self.pdf = pdf_pipeline.PDFPipelineDependencies(
            api_max_concurrency=config.api_max_concurrency,
            assembly_max_concurrency=config.pdf_assembly_max_concurrency,
            extraction_max_concurrency=config.pdf_extraction_max_concurrency,
            fail_open_to_source_page=config.pdf_fail_open_to_source_page,
            preserve_tables=config.pdf_preserve_tables,
            minimum_htmlbox_scale=config.pdf_min_acceptable_htmlbox_scale,
            save_clean=config.pdf_save_clean,
            save_garbage=config.pdf_save_garbage,
            use_htmlbox=config.pdf_use_htmlbox,
            progress_dir=str(config.progress_dir),
            system_prompt_text=SYSTEM_PROMPT_TEXT,
            translation_workers=config.translation_workers,
            assembly_semaphore=threading.BoundedSemaphore(
                config.pdf_assembly_max_concurrency
            ),
            extraction_semaphore=extraction_semaphore,
            logger=log,
            build_glossary=self.build_glossary,
            make_prompt_with_glossary=with_glossary,
            load_progress=self.tasks.load,
            save_progress=self.tasks.save,
            translate_text=self.translate_text,
            trim_process_memory=trim_process_memory,
            extract_page_elements=_extract_page_elements,
            filter_formula_safe_rects=_filter_pdf_formula_safe_rects,
            summarize_formula_protection=_summarize_pdf_formula_protection,
            check_output_structure_serialized=partial(
                _check_pdf_output_structure_serialized, dependencies=audit
            ),
            save_translation_progress=partial(
                _save_pdf_translation_progress, dependencies=audit
            ),
            font_subsetting_available=pdf_font_subsetting_available(),
            font_regular_override=(
                str(config.pdf_font_regular) if config.pdf_font_regular else None
            ),
            font_bold_override=(
                str(config.pdf_font_bold) if config.pdf_font_bold else None
            ),
        )
        self._jobs = TranslationJobDependencies(
            logger=log,
            load_progress=self.tasks.load,
            translate_epub=self.translate_epub,
            translate_pdf=self.translate_pdf,
        )
        self._recovery = TaskRecoveryDependencies(
            tasks=self.tasks,
            upload_dir=config.upload_dir,
            output_dir=config.output_dir,
            logger=log,
            enqueue_task=self.enqueue,
            source_sha256=_pdf_source_sha256,
            load_cached_page_indices=_load_pdf_cached_page_indices,
        )

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Create directories, start logging, load tasks and resume them, once."""
        with self._init_lock:
            if self._initialized:
                return
            self.config.ensure_runtime_directories()
            configure_logging(self.config.log_file)
            configure_pdf_cache_progress_dir(lambda: str(self.config.progress_dir))
            self.tasks.load_all()
            if not self.config.disable_auto_resume:
                self.resume_interrupted_tasks()
            self._initialized = True

    def create_app(self):
        """Create the WSGI application without performing runtime I/O."""
        return create_app(
            self.config,
            WebDependencies(
                tasks=self.tasks,
                ensure_runtime=self.initialize,
                enqueue_task=self.enqueue,
            ),
        )

    # -- translation services ----------------------------------------------

    def build_glossary(self, all_text: str, task_id: str = "") -> str:
        """Select explicit JSON glossary entries using deterministic Python rules."""
        try:
            mapping = load_explicit_glossary(self.config.glossary_json)
            glossary = build_glossary_text(all_text, mapping)
            line_count = len(glossary.splitlines()) if glossary else 0
            log.info(
                f"[{task_id}] Deterministic glossary selected {line_count} explicit terms"
            )
            return glossary
        except (OSError, ValueError) as exc:
            log.error(f"[{task_id}] Explicit glossary configuration rejected: {exc}")
            return ""

    def translate_text(
        self, text: str, system_prompt: str | None = None, retries: int = 4
    ) -> str:
        return text_translator.translate_text(
            text, system_prompt, retries, dependencies=self.text_translation
        )

    def _translate_epub_payload(self, text: str, system_prompt: str) -> str:
        """Let the EPUB segment validator own JSON and per-paragraph integrity."""
        response = self.text_translation.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=65536,
            temperature=0.3,
        )
        return strip_think_tags(response.choices[0].message.content or "")

    def translate_epub(self, task_id: str, src_path: str, out_path: str):
        return EPUBPipeline(self.epub).translate(task_id, src_path, out_path)

    def translate_pdf(
        self,
        task_id: str,
        src_path: str,
        out_path: str,
        pdf_password: str = "",
        _structure_retry: int = 0,
        _forced_source_page_fallbacks: list[dict] | None = None,
    ):
        return pdf_pipeline.translate_pdf(
            task_id,
            src_path,
            out_path,
            pdf_password,
            _structure_retry,
            _forced_source_page_fallbacks,
            dependencies=self.pdf,
        )

    # -- task queue --------------------------------------------------------

    def enqueue(
        self,
        task_id: str,
        ext: str,
        src_path: str,
        out_path: str,
        pdf_password: str = "",
        file_size: int | None = None,
    ) -> bool:
        return enqueue_translation_task(
            task_id,
            ext,
            src_path,
            out_path,
            pdf_password,
            file_size,
            self.worker_pool,
            self.config.task_max_concurrency,
            log,
        )

    def resume_interrupted_tasks(self) -> None:
        resume_interrupted_tasks(self._recovery)

    def _run_job(self, job: TaskJob, worker_id: int) -> None:
        run_translation_job(job, worker_id, self._jobs)

    def _mark_failed_after_crash(self, task_id: str, error: str) -> None:
        """Last-resort terminal state for a job whose handler crashed."""
        record = self.tasks.load(task_id) or {}
        if record.get("status") in ("completed", "failed"):
            return
        record.update(
            {
                "status": "failed",
                "current_file": "",
                "error": f"Unhandled worker error: {error}"[:500],
            }
        )
        self.tasks.save(task_id, record)
