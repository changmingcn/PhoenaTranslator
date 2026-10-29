"""Process wiring for PhoenaTranslator: the shared Runtime and its WSGI app."""

from __future__ import annotations

import logging
from operator import attrgetter
from pathlib import Path

from phoena_translator.config import get_app_config
from phoena_translator.runtime import Runtime

APP_CONFIG = get_app_config()
log = logging.getLogger("translator")
if not log.handlers:
    log.addHandler(logging.NullHandler())

runtime = Runtime(APP_CONFIG)
app = runtime.create_app()
initialize_runtime = runtime.initialize

__all__ = ["APP_CONFIG", "app", "initialize_runtime", "runtime"]

# Former module-level services and where they now live on ``runtime``.
_FORMER_SERVICES = {
    "build_glossary": "build_glossary",
    "translate_text": "translate_text",
    "translate_epub": "translate_epub",
    "translate_pdf": "translate_pdf",
    "save_progress": "tasks.save",
    "load_progress": "tasks.load",
    "load_all_tasks": "tasks.load_all",
    "resume_interrupted_tasks": "resume_interrupted_tasks",
    "enqueue_translation_task": "enqueue",
}


def _former_global(name: str):
    """Resolve a former module global, or raise ``AttributeError``."""
    if name in _FORMER_SERVICES:
        return attrgetter(_FORMER_SERVICES[name])(runtime)
    if name.isupper() and hasattr(APP_CONFIG, name.lower()):
        value = getattr(APP_CONFIG, name.lower())
        # The directory constants were strings.
        return str(value) if isinstance(value, Path) else value
    from phoena_translator import legacy_exports

    if name in legacy_exports.exported_names():
        return getattr(legacy_exports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str):
    return _former_global(name)
