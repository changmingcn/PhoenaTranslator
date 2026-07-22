"""Typed state and dependency boundaries for the PDF translation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PDFPipelineDependencies:
    """Runtime dependencies injected by the application bootstrap layer."""

    api_max_concurrency: int
    assembly_max_concurrency: int
    extraction_max_concurrency: int
    fail_open_to_source_page: bool
    minimum_htmlbox_scale: float
    save_clean: bool
    save_garbage: int
    use_htmlbox: bool
    progress_dir: str
    system_prompt_text: str
    translation_workers: int
    assembly_semaphore: Any
    extraction_semaphore: Any
    logger: logging.Logger
    tasks: dict[str, dict]
    build_glossary: Callable[..., str]
    make_prompt_with_glossary: Callable[[str, str], str]
    load_progress: Callable[[str], dict | None]
    save_progress: Callable[[str, dict], None]
    translate_text: Callable[..., str]
    # Kept for the frozen Stage 1 construction contract.  Recovery is now
    # iterative and deliberately does not call this compatibility callback.
    retry_translate_pdf: Callable[..., Any]
    trim_process_memory: Callable[[], None]
    extract_page_elements: Callable[..., list[dict]]
    filter_formula_safe_rects: Callable[..., list[Any]]
    summarize_formula_protection: Callable[..., dict]
    check_output_structure_serialized: Callable[..., dict]
    save_translation_progress: Callable[..., None]
    font_subsetting_available: bool


@dataclass(frozen=True)
class PDFPageTranslationContext:
    """Small immutable boundary used by concurrent page translators."""

    task_id: str
    prompt: str
    logger: logging.Logger
    translate_text: Callable[..., str]


__all__ = [
    "PDFPageTranslationContext",
    "PDFPipelineDependencies",
]
