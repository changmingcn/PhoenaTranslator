"""Typed environment configuration without import-time side effects."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    try:
        value = int(env.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    try:
        value = float(env.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


@dataclass(frozen=True)
class AppConfig:
    upload_dir: Path
    output_dir: Path
    progress_dir: Path
    log_file: Path
    max_file_size: int = 100 * 1024 * 1024
    max_request_size: int = 101 * 1024 * 1024
    allowed_extensions: frozenset[str] = frozenset({"epub", "pdf"})
    task_queue_max: int = 8
    pending_bytes_max: int = 500 * 1024 * 1024
    epub_max_members: int = 10_000
    epub_max_expanded_bytes: int = 1024 * 1024 * 1024
    epub_max_member_bytes: int = 256 * 1024 * 1024
    epub_max_compression_ratio: int = 1000
    api_max_concurrency: int = 2
    translation_workers: int = 4
    task_max_concurrency: int = 1
    pdf_extraction_max_concurrency: int = 1
    pdf_assembly_max_concurrency: int = 1
    pdf_save_garbage: int = 1
    pdf_save_clean: bool = False
    pdf_use_htmlbox: bool = True
    pdf_fail_open_to_source_page: bool = True
    pdf_preserve_tables: bool = True
    pdf_min_acceptable_htmlbox_scale: float = 0.60
    pdf_vector_ocr_dpi: int = 216
    pdf_vector_ocr_min_alpha_words: int = 60
    pdf_font_regular: Path | None = None
    pdf_font_bold: Path | None = None
    glossary_json: Path | None = None
    deepseek_api_key: str = field(default="", repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_price_cache_hit_usd_per_m: float = 0.0028
    deepseek_price_cache_miss_usd_per_m: float = 0.14
    deepseek_price_output_usd_per_m: float = 0.28
    disable_auto_resume: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        home: Path | None = None,
    ) -> "AppConfig":
        env = os.environ if environ is None else environ
        base = Path.home() if home is None else Path(home)
        max_file_size = _env_int(
            env,
            "TRANSLATOR_MAX_FILE_SIZE",
            100 * 1024 * 1024,
            minimum=1024,
        )
        glossary_raw = env.get("TRANSLATOR_GLOSSARY_JSON", "").strip()
        font_regular_raw = env.get("TRANSLATOR_PDF_FONT_REGULAR", "").strip()
        font_bold_raw = env.get("TRANSLATOR_PDF_FONT_BOLD", "").strip()
        return cls(
            upload_dir=Path(env.get("TRANSLATOR_UPLOAD_DIR", base / "translator_uploads")).expanduser(),
            output_dir=Path(env.get("TRANSLATOR_OUTPUT_DIR", base / "translator_outputs")).expanduser(),
            progress_dir=Path(env.get("TRANSLATOR_PROGRESS_DIR", base / "translator_progress")).expanduser(),
            log_file=Path(env.get("TRANSLATOR_LOG_FILE", base / "translator_app.log")).expanduser(),
            max_file_size=max_file_size,
            max_request_size=_env_int(
                env,
                "TRANSLATOR_MAX_REQUEST_SIZE",
                max_file_size + 1024 * 1024,
                minimum=max_file_size,
            ),
            task_queue_max=_env_int(
                env,
                "TRANSLATOR_TASK_QUEUE_MAX",
                8,
                minimum=1,
            ),
            pending_bytes_max=_env_int(
                env,
                "TRANSLATOR_PENDING_BYTES_MAX",
                500 * 1024 * 1024,
                minimum=max_file_size,
            ),
            epub_max_members=_env_int(
                env,
                "TRANSLATOR_EPUB_MAX_MEMBERS",
                10_000,
                minimum=1,
            ),
            epub_max_expanded_bytes=_env_int(
                env,
                "TRANSLATOR_EPUB_MAX_EXPANDED_BYTES",
                1024 * 1024 * 1024,
                minimum=max_file_size,
            ),
            epub_max_member_bytes=_env_int(
                env,
                "TRANSLATOR_EPUB_MAX_MEMBER_BYTES",
                256 * 1024 * 1024,
                minimum=max_file_size,
            ),
            epub_max_compression_ratio=_env_int(
                env,
                "TRANSLATOR_EPUB_MAX_COMPRESSION_RATIO",
                1000,
                minimum=1,
            ),
            api_max_concurrency=_env_int(
                env,
                "TRANSLATOR_API_MAX_CONCURRENCY",
                2,
                minimum=1,
            ),
            translation_workers=_env_int(
                env,
                "TRANSLATOR_TRANSLATION_WORKERS",
                4,
                minimum=1,
            ),
            task_max_concurrency=_env_int(
                env,
                "TRANSLATOR_TASK_MAX_CONCURRENCY",
                1,
                minimum=1,
            ),
            pdf_extraction_max_concurrency=_env_int(
                env,
                "TRANSLATOR_PDF_EXTRACTION_MAX_CONCURRENCY",
                1,
                minimum=1,
            ),
            pdf_assembly_max_concurrency=_env_int(
                env,
                "TRANSLATOR_PDF_ASSEMBLY_MAX_CONCURRENCY",
                1,
                minimum=1,
            ),
            pdf_save_garbage=_env_int(
                env,
                "TRANSLATOR_PDF_SAVE_GARBAGE",
                1,
                minimum=0,
            ),
            pdf_save_clean=_env_bool(
                env,
                "TRANSLATOR_PDF_SAVE_CLEAN",
                False,
            ),
            pdf_use_htmlbox=_env_bool(
                env,
                "TRANSLATOR_PDF_USE_HTMLBOX",
                True,
            ),
            pdf_fail_open_to_source_page=_env_bool(
                env,
                "TRANSLATOR_PDF_FAIL_OPEN_TO_SOURCE_PAGE",
                True,
            ),
            pdf_preserve_tables=_env_bool(
                env,
                "TRANSLATOR_PDF_PRESERVE_TABLES",
                True,
            ),
            pdf_min_acceptable_htmlbox_scale=_env_float(
                env,
                "TRANSLATOR_PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE",
                0.60,
                minimum=0.60,
            ),
            pdf_vector_ocr_dpi=_env_int(
                env,
                "TRANSLATOR_PDF_VECTOR_OCR_DPI",
                216,
                minimum=144,
            ),
            pdf_vector_ocr_min_alpha_words=_env_int(
                env,
                "TRANSLATOR_PDF_VECTOR_OCR_MIN_ALPHA_WORDS",
                60,
                minimum=20,
            ),
            pdf_font_regular=(
                Path(font_regular_raw).expanduser() if font_regular_raw else None
            ),
            pdf_font_bold=Path(font_bold_raw).expanduser() if font_bold_raw else None,
            glossary_json=Path(glossary_raw).expanduser() if glossary_raw else None,
            deepseek_api_key=env.get("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=env.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_price_cache_hit_usd_per_m=_env_float(
                env,
                "DEEPSEEK_PRICE_CACHE_HIT_USD_PER_M",
                0.0028,
            ),
            deepseek_price_cache_miss_usd_per_m=_env_float(
                env,
                "DEEPSEEK_PRICE_CACHE_MISS_USD_PER_M",
                0.14,
            ),
            deepseek_price_output_usd_per_m=_env_float(
                env,
                "DEEPSEEK_PRICE_OUTPUT_USD_PER_M",
                0.28,
            ),
            disable_auto_resume=_env_bool(
                env,
                "TRANSLATOR_DISABLE_AUTO_RESUME",
                False,
            ),
        )

    def ensure_runtime_directories(self) -> None:
        for directory in (self.upload_dir, self.output_dir, self.progress_dir):
            directory.mkdir(parents=True, exist_ok=True)


_shared_config: AppConfig | None = None
_shared_config_lock = threading.Lock()


def get_app_config() -> AppConfig:
    """Return the process-wide AppConfig, reading the environment exactly once.

    Every module that needs configuration at import time must use this
    accessor so the whole process observes one consistent snapshot.  To embed
    the package with a custom configuration, call :func:`set_app_config`
    before importing any ``phoena_translator.pdf`` module.
    """
    global _shared_config
    if _shared_config is None:
        with _shared_config_lock:
            if _shared_config is None:
                _shared_config = AppConfig.from_env()
    return _shared_config


def set_app_config(config: AppConfig | None) -> None:
    """Install (or with ``None`` reset) the process-wide AppConfig."""
    global _shared_config
    with _shared_config_lock:
        _shared_config = config
