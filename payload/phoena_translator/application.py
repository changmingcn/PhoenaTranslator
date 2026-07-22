"""Application runtime and WSGI construction for PhoenaTranslator."""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path

from phoena_translator.api_runtime import (
    APIRetryPolicy,
    APIRuntimeState,
    APIUsagePrices,
    CompletionDependencies,
    completion_usage_record as _completion_usage_record_impl,
    create_chat_completion as _create_chat_completion_impl,
    extract_retry_delay as _extract_retry_delay_impl,
    extract_status_code as _extract_status_code,  # noqa: F401 - legacy API
    is_rate_limit_error as _is_rate_limit_error,
    is_retryable_api_error as _is_retryable_api_error,
    nonnegative_int as _nonnegative_int,  # noqa: F401 - legacy API
    record_api_event as _record_api_event_impl,
    record_completion_usage as _record_completion_usage_impl,
    record_rate_limit as _record_rate_limit_impl,
    strip_think_tags,
    usage_as_dict as _usage_as_dict,  # noqa: F401 - legacy API
    wait_for_api_slot as _wait_for_api_slot_impl,
)
from phoena_translator.config import AppConfig
from phoena_translator.epub.archive import ArchiveLimits
from phoena_translator.epub.pipeline import (
    EPUBPipeline,
    EPUBPipelineDependencies,
    translate_single_chunk as _translate_epub_single_chunk,
)
from phoena_translator.epub.xhtml import DEFAULT_CHUNK_MAX_BYTES as CHUNK_MAX_BYTES
from phoena_translator.glossary import build_glossary_text, load_explicit_glossary
from phoena_translator.llm import DeepSeekTranslationAdapter, LLMSettings
from phoena_translator.logging_setup import configure_logging
from phoena_translator.pdf.audit import (
    PDFAuditDependencies,
    _check_pdf_output_structure,
    _check_pdf_output_structure_serialized as _check_pdf_output_structure_serialized_impl,
    _save_pdf_translation_progress as _save_pdf_translation_progress_impl,
    _summarize_pdf_formula_protection,
)
from phoena_translator.pdf.cache import (
    _load_pdf_cached_page_indices,
    _pdf_source_sha256,
    configure_pdf_cache_progress_dir,
)
from phoena_translator.pdf.fonts import pdf_font_subsetting_available
from phoena_translator.pdf.math_detection import _filter_pdf_formula_safe_rects
from phoena_translator.pdf.pipeline import (
    PDFPipelineDependencies,
    translate_pdf as _translate_pdf_impl,
)
from phoena_translator.pdf.targets import _extract_page_elements
from phoena_translator.pdf.translation import (
    PDFTranslationDependencies,
    translate_text as _translate_text_impl,
)
from phoena_translator.pdf.types import PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE
from phoena_translator.prompts import SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_XHTML
from phoena_translator.task_store import TaskStore
from phoena_translator.task_runtime import (
    TaskPersistenceDependencies,
    TaskRecoveryDependencies,
    TranslationJobDependencies,
    enqueue_translation_task as _enqueue_translation_task_impl,
    load_all_tasks as _load_all_tasks_impl,
    load_progress as _load_progress_impl,
    normalize_task_record as _normalize_task_record_impl,
    resume_interrupted_tasks as _resume_interrupted_tasks_impl,
    run_translation_job as _run_translation_job_impl,
    save_progress as _save_progress_impl,
)
from phoena_translator.web import WebDependencies, create_app as create_web_app
from phoena_translator.workers import TaskJob, TaskWorkerPool


def __getattr__(name: str):
    """Resolve deprecated private facade names only when callers request them."""

    from phoena_translator import legacy_exports

    if name not in legacy_exports.exported_names():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(legacy_exports, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    from phoena_translator import legacy_exports

    return sorted({*globals(), *legacy_exports.exported_names()})


APP_CONFIG = AppConfig.from_env()
LOG_FILE = str(APP_CONFIG.log_file)
log = logging.getLogger("translator")
if not log.handlers:
    log.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = str(APP_CONFIG.upload_dir)
OUTPUT_DIR = str(APP_CONFIG.output_dir)
PROGRESS_DIR = str(APP_CONFIG.progress_dir)
configure_pdf_cache_progress_dir(lambda: PROGRESS_DIR)
MAX_FILE_SIZE = APP_CONFIG.max_file_size
MAX_REQUEST_SIZE = APP_CONFIG.max_request_size
ALLOWED_EXTENSIONS = set(APP_CONFIG.allowed_extensions)

DEEPSEEK_API_KEY = APP_CONFIG.deepseek_api_key
DEEPSEEK_BASE_URL = APP_CONFIG.deepseek_base_url
DEEPSEEK_MODEL = APP_CONFIG.deepseek_model

tasks: dict = {}
tasks_lock = threading.Lock()
_task_store_lock = threading.Lock()
_task_store_instance: TaskStore | None = None
_task_store_signature: tuple[str, str, str] | None = None
_runtime_init_lock = threading.Lock()
_runtime_initialized = False

# ---------------------------------------------------------------------------
# OpenAI-compatible client for DeepSeek
# ---------------------------------------------------------------------------

_llm_adapter = DeepSeekTranslationAdapter(
    LLMSettings(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
    )
)


API_MAX_CONCURRENCY = APP_CONFIG.api_max_concurrency
PDF_EXTRACTION_MAX_CONCURRENCY = APP_CONFIG.pdf_extraction_max_concurrency
PDF_ASSEMBLY_MAX_CONCURRENCY = APP_CONFIG.pdf_assembly_max_concurrency
PDF_SAVE_GARBAGE = APP_CONFIG.pdf_save_garbage
PDF_SAVE_CLEAN = APP_CONFIG.pdf_save_clean
PDF_USE_HTMLBOX = APP_CONFIG.pdf_use_htmlbox
PDF_FAIL_OPEN_TO_SOURCE_PAGE = APP_CONFIG.pdf_fail_open_to_source_page
# v14 preserves prose-dominant inline formula tokens while translating their
# surrounding sentence, keeps shallow formula-adjacent prose translatable,
# and redraws vertical chart labels at their native rotation. Earlier geometry
# identities remain safe to attempt to rebind: the identity migration drops
# split/merged cells, while active per-page validation rejects semantically
# incomplete survivors and retranslates only the affected page.
API_RATE_LIMIT_RETRIES = 6
API_MONITOR_WINDOW_SECONDS = 60
API_RATE_LIMIT_BURST_THRESHOLD = 5
API_RATE_LIMIT_BASE_DELAY = 12
API_RATE_LIMIT_MAX_DELAY = 90
API_TRANSIENT_ERROR_MAX_DELAY = 30
DEEPSEEK_PRICE_CACHE_HIT_USD_PER_M = APP_CONFIG.deepseek_price_cache_hit_usd_per_m
DEEPSEEK_PRICE_CACHE_MISS_USD_PER_M = APP_CONFIG.deepseek_price_cache_miss_usd_per_m
DEEPSEEK_PRICE_OUTPUT_USD_PER_M = APP_CONFIG.deepseek_price_output_usd_per_m

_api_state = APIRuntimeState.create(API_MAX_CONCURRENCY)
_api_concurrency_semaphore = _api_state.concurrency_semaphore
_pdf_extraction_semaphore = threading.BoundedSemaphore(PDF_EXTRACTION_MAX_CONCURRENCY)
_pdf_assembly_semaphore = threading.BoundedSemaphore(PDF_ASSEMBLY_MAX_CONCURRENCY)


def _trim_process_memory():
    gc.collect()
    if os.name != "posix":
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _record_api_event(kind: str) -> dict:
    return _record_api_event_impl(
        _api_state,
        kind,
        API_MONITOR_WINDOW_SECONDS,
    )


def _current_api_policy() -> APIRetryPolicy:
    return APIRetryPolicy(
        attempts=API_RATE_LIMIT_RETRIES,
        monitor_window_seconds=API_MONITOR_WINDOW_SECONDS,
        rate_limit_burst_threshold=API_RATE_LIMIT_BURST_THRESHOLD,
        rate_limit_base_delay=API_RATE_LIMIT_BASE_DELAY,
        rate_limit_max_delay=API_RATE_LIMIT_MAX_DELAY,
        transient_error_max_delay=API_TRANSIENT_ERROR_MAX_DELAY,
    )


def _extract_retry_delay(exc: Exception, attempt: int) -> float:
    return _extract_retry_delay_impl(exc, attempt, _current_api_policy())


def _wait_for_api_slot() -> None:
    _wait_for_api_slot_impl(_api_state, log)


def _record_rate_limit(delay: float, exc: Exception) -> None:
    _record_rate_limit_impl(_api_state, delay, exc, log)


def _current_api_prices() -> APIUsagePrices:
    return APIUsagePrices(
        cache_hit_usd_per_million=DEEPSEEK_PRICE_CACHE_HIT_USD_PER_M,
        cache_miss_usd_per_million=DEEPSEEK_PRICE_CACHE_MISS_USD_PER_M,
        output_usd_per_million=DEEPSEEK_PRICE_OUTPUT_USD_PER_M,
    )


def _completion_usage_record(resp, model: str) -> dict | None:
    return _completion_usage_record_impl(
        resp,
        model,
        _current_api_prices(),
    )


def _api_usage_log_path() -> str:
    return str(Path(PROGRESS_DIR) / "deepseek_usage.jsonl")


def _record_completion_usage(resp, model: str) -> dict | None:
    return _record_completion_usage_impl(
        resp,
        model,
        _current_api_prices(),
        Path(_api_usage_log_path()),
        _api_state,
        log,
    )


def _create_chat_completion(
    *,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    model: str = DEEPSEEK_MODEL,
):
    return _create_chat_completion_impl(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        dependencies=CompletionDependencies(
            adapter=_llm_adapter,
            concurrency_semaphore=_api_concurrency_semaphore,
            policy=_current_api_policy(),
            logger=log,
            wait_for_slot=_wait_for_api_slot,
            record_usage=_record_completion_usage,
            record_event=_record_api_event,
            record_rate_limit=_record_rate_limit,
            is_rate_limit_error=_is_rate_limit_error,
            is_retryable_error=_is_retryable_api_error,
            retry_delay=_extract_retry_delay,
        ),
    )


def build_glossary(all_text: str, task_id: str = "") -> str:
    """Select explicit JSON glossary entries using deterministic Python rules."""
    try:
        mapping = load_explicit_glossary(APP_CONFIG.glossary_json)
        glossary = build_glossary_text(all_text, mapping)
        line_count = len(glossary.splitlines()) if glossary else 0
        log.info(
            f"[{task_id}] Deterministic glossary selected {line_count} explicit terms"
        )
        return glossary
    except (OSError, ValueError) as exc:
        log.error(f"[{task_id}] Explicit glossary configuration rejected: {exc}")
        return ""


def _make_prompt_with_glossary(base_prompt: str, glossary: str) -> str:
    """Inject glossary into a system prompt."""
    if not glossary:
        return base_prompt
    return (
        base_prompt
        + f"\n\n以下是本书的统一术语表，请严格按此翻译专有名词：\n{glossary}"
    )


def translate_text(text: str, system_prompt: str = None, retries: int = 4) -> str:
    """Compatibility facade over the dependency-injected text translator."""
    return _translate_text_impl(
        text,
        system_prompt,
        retries,
        dependencies=PDFTranslationDependencies(
            create_chat_completion=_create_chat_completion,
            strip_think_tags=strip_think_tags,
            system_prompt_text=SYSTEM_PROMPT_TEXT,
            logger=log,
        ),
    )


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _get_task_store() -> TaskStore:
    global _task_store_instance, _task_store_signature
    signature = (str(PROGRESS_DIR), str(UPLOAD_DIR), str(OUTPUT_DIR))
    with _task_store_lock:
        if _task_store_instance is None or _task_store_signature != signature:
            _task_store_instance = TaskStore(
                progress_dir=Path(PROGRESS_DIR),
                upload_dir=Path(UPLOAD_DIR),
                output_dir=Path(OUTPUT_DIR),
            )
            _task_store_signature = signature
        return _task_store_instance


def _task_persistence_dependencies() -> TaskPersistenceDependencies:
    return TaskPersistenceDependencies(
        tasks=tasks,
        tasks_lock=tasks_lock,
        task_store=_get_task_store,
    )


def _normalize_task_record(task_id: str, data: dict | None) -> dict | None:
    return _normalize_task_record_impl(
        task_id,
        data,
        _task_persistence_dependencies(),
    )


def save_progress(task_id: str, data: dict) -> None:
    _save_progress_impl(
        task_id,
        data,
        _task_persistence_dependencies(),
    )


def load_progress(task_id: str) -> dict | None:
    return _load_progress_impl(
        task_id,
        _task_persistence_dependencies(),
    )


def load_all_tasks() -> None:
    _load_all_tasks_impl(_task_persistence_dependencies())


def resume_interrupted_tasks() -> None:
    _resume_interrupted_tasks_impl(
        TaskRecoveryDependencies(
            tasks=tasks,
            tasks_lock=tasks_lock,
            upload_dir=Path(UPLOAD_DIR),
            output_dir=Path(OUTPUT_DIR),
            logger=log,
            save_progress=save_progress,
            enqueue_task=enqueue_translation_task,
            source_sha256=_pdf_source_sha256,
            load_cached_page_indices=_load_pdf_cached_page_indices,
        )
    )


# ---------------------------------------------------------------------------
# EPUB helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EPUB chunking helpers
# ---------------------------------------------------------------------------

# Max bytes per chunk sent to API (~20KB; 40KB still causes frequent timeouts on large chapters)

# Number of per-document translation worker threads used to prepare and dispatch
# page/chunk work. Actual DeepSeek request concurrency is capped separately.
TRANSLATION_WORKERS = APP_CONFIG.translation_workers

# Number of documents processed at once. Keep this low on small VPS instances:
# each PDF translation holds extracted page layout and rendered output in memory.
TASK_MAX_CONCURRENCY = APP_CONFIG.task_max_concurrency
TASK_QUEUE_MAX = APP_CONFIG.task_queue_max
PENDING_BYTES_MAX = APP_CONFIG.pending_bytes_max
_task_worker_pool_lock = threading.Lock()
_task_worker_pool: TaskWorkerPool | None = None


def _run_translation_job(job: TaskJob, worker_id: int) -> None:
    _run_translation_job_impl(
        job,
        worker_id,
        TranslationJobDependencies(
            logger=log,
            load_progress=load_progress,
            translate_epub=translate_epub,
            translate_pdf=translate_pdf,
        ),
    )


def _get_task_worker_pool() -> TaskWorkerPool:
    global _task_worker_pool
    with _task_worker_pool_lock:
        if _task_worker_pool is None:
            _task_worker_pool = TaskWorkerPool(
                max_workers=TASK_MAX_CONCURRENCY,
                max_queue=TASK_QUEUE_MAX,
                max_pending_bytes=PENDING_BYTES_MAX,
                handler=_run_translation_job,
                logger=log,
            )
        return _task_worker_pool


def enqueue_translation_task(
    task_id: str,
    ext: str,
    src_path: str,
    out_path: str,
    pdf_password: str = "",
    file_size: int | None = None,
) -> bool:
    return _enqueue_translation_task_impl(
        task_id,
        ext,
        src_path,
        out_path,
        pdf_password,
        file_size,
        _get_task_worker_pool(),
        TASK_MAX_CONCURRENCY,
        log,
    )


# ---------------------------------------------------------------------------
# EPUB translation
# ---------------------------------------------------------------------------


def _translate_single_chunk(
    task_id, rel, chunk_idx, total_chunks, chunk_text, xhtml_prompt
):
    """Compatibility facade over the dependency-injected EPUB chunk translator."""
    return _translate_epub_single_chunk(
        task_id,
        rel,
        chunk_idx,
        total_chunks,
        chunk_text,
        xhtml_prompt,
        translate_text=translate_text,
        logger=log,
    )


def translate_epub(task_id: str, src_path: str, out_path: str):
    """Run the modular, bounded EPUB pipeline through legacy application services."""
    dependencies = EPUBPipelineDependencies(
        translate_text=translate_text,
        build_glossary=build_glossary,
        make_prompt=_make_prompt_with_glossary,
        save_progress=save_progress,
        load_task=load_progress,
        logger=log,
        system_prompt_xhtml=SYSTEM_PROMPT_XHTML,
        workers=TRANSLATION_WORKERS,
        api_concurrency=API_MAX_CONCURRENCY,
        archive_limits=ArchiveLimits(
            max_members=APP_CONFIG.epub_max_members,
            max_expanded_bytes=APP_CONFIG.epub_max_expanded_bytes,
            max_member_bytes=APP_CONFIG.epub_max_member_bytes,
            max_compression_ratio=APP_CONFIG.epub_max_compression_ratio,
        ),
        chunk_max_bytes=CHUNK_MAX_BYTES,
    )
    return EPUBPipeline(dependencies).translate(task_id, src_path, out_path)


# ---------------------------------------------------------------------------
# PDF translation
# ---------------------------------------------------------------------------


def _pdf_audit_dependencies() -> PDFAuditDependencies:
    return PDFAuditDependencies(
        extraction_max_concurrency=PDF_EXTRACTION_MAX_CONCURRENCY,
        extraction_semaphore=_pdf_extraction_semaphore,
        check_output_structure=_check_pdf_output_structure,
        trim_process_memory=_trim_process_memory,
        save_progress=save_progress,
        tasks=tasks,
    )


def _check_pdf_output_structure_serialized(
    out_path: str,
    superscript_expectations: list[dict],
    merge_decisions: list[dict] | None = None,
    formula_expectations: list[dict] | None = None,
    vector_ocr_expectations: list[dict] | None = None,
    source_path: str | None = None,
    source_password: str = "",
    page_extractions: dict | None = None,
) -> dict:
    """Compatibility facade that preserves runtime monkeypatch boundaries."""
    return _check_pdf_output_structure_serialized_impl(
        out_path,
        superscript_expectations,
        merge_decisions,
        formula_expectations,
        vector_ocr_expectations,
        source_path,
        source_password,
        page_extractions,
        dependencies=_pdf_audit_dependencies(),
    )


def _save_pdf_translation_progress(
    task_id: str,
    completed_indices: set[int],
    total_pages: int,
    current_file: str,
    source_page_fallbacks: list[dict] | None = None,
):
    """Compatibility facade for task-state persistence from the PDF pipeline."""
    return _save_pdf_translation_progress_impl(
        task_id,
        completed_indices,
        total_pages,
        current_file,
        source_page_fallbacks,
        dependencies=_pdf_audit_dependencies(),
    )


def translate_pdf(
    task_id: str,
    src_path: str,
    out_path: str,
    pdf_password: str = "",
    _structure_retry: int = 0,
    _forced_source_page_fallbacks: list[dict] | None = None,
):
    """Run the modular PDF pipeline with current application dependencies."""
    dependencies = PDFPipelineDependencies(
        api_max_concurrency=API_MAX_CONCURRENCY,
        assembly_max_concurrency=PDF_ASSEMBLY_MAX_CONCURRENCY,
        extraction_max_concurrency=PDF_EXTRACTION_MAX_CONCURRENCY,
        fail_open_to_source_page=PDF_FAIL_OPEN_TO_SOURCE_PAGE,
        minimum_htmlbox_scale=PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE,
        save_clean=PDF_SAVE_CLEAN,
        save_garbage=PDF_SAVE_GARBAGE,
        use_htmlbox=PDF_USE_HTMLBOX,
        progress_dir=PROGRESS_DIR,
        system_prompt_text=SYSTEM_PROMPT_TEXT,
        translation_workers=TRANSLATION_WORKERS,
        assembly_semaphore=_pdf_assembly_semaphore,
        extraction_semaphore=_pdf_extraction_semaphore,
        logger=log,
        tasks=tasks,
        build_glossary=build_glossary,
        make_prompt_with_glossary=_make_prompt_with_glossary,
        load_progress=load_progress,
        save_progress=save_progress,
        translate_text=translate_text,
        retry_translate_pdf=translate_pdf,
        trim_process_memory=_trim_process_memory,
        extract_page_elements=_extract_page_elements,
        filter_formula_safe_rects=_filter_pdf_formula_safe_rects,
        summarize_formula_protection=_summarize_pdf_formula_protection,
        check_output_structure_serialized=_check_pdf_output_structure_serialized,
        save_translation_progress=_save_pdf_translation_progress,
        font_subsetting_available=pdf_font_subsetting_available(),
    )
    return _translate_pdf_impl(
        task_id,
        src_path,
        out_path,
        pdf_password,
        _structure_retry,
        _forced_source_page_fallbacks,
        dependencies=dependencies,
    )


def initialize_runtime() -> None:
    """Initialize writable/runtime state exactly once, outside module import."""
    global _runtime_initialized, log
    with _runtime_init_lock:
        if _runtime_initialized:
            return
        for directory in (UPLOAD_DIR, OUTPUT_DIR, PROGRESS_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)
        log = configure_logging(Path(LOG_FILE))
        load_all_tasks()
        if not APP_CONFIG.disable_auto_resume:
            resume_interrupted_tasks()
        _runtime_initialized = True


def _legacy_web_dependencies() -> WebDependencies:
    return WebDependencies(
        tasks=lambda: tasks,
        tasks_lock=lambda: tasks_lock,
        ensure_runtime=initialize_runtime,
        save_progress=lambda task_id, data: save_progress(task_id, data),
        load_progress=lambda task_id: load_progress(task_id),
        normalize_task=lambda task_id, data: _normalize_task_record(task_id, data),
        enqueue_task=lambda task_id, ext, src, out, password, size: (
            enqueue_translation_task(
                task_id,
                ext,
                src,
                out,
                password,
                size,
            )
        ),
        task_store=_get_task_store,
    )


def create_app(
    config: AppConfig | None = None,
    dependencies: WebDependencies | None = None,
):
    """Create the WSGI application without performing runtime I/O."""
    return create_web_app(
        config or APP_CONFIG,
        dependencies or _legacy_web_dependencies(),
    )


app = create_app()

# Transitional names preserve direct imports while route ownership lives in web.py.
upload_file = app.view_functions["upload_file"]
task_status = app.view_functions["task_status"]
download_file = app.view_functions["download_file"]
delete_task = app.view_functions["delete_task"]
list_tasks = app.view_functions["list_tasks"]

# Preserve the former module's non-private wildcard-import surface.  Private
# compatibility symbols remain available lazily through ``__getattr__``.
from phoena_translator.legacy_exports import exported_names as _legacy_exported_names

__all__ = tuple(
    sorted(
        name
        for name in {*globals(), *_legacy_exported_names()}
        if not name.startswith("_")
    )
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
