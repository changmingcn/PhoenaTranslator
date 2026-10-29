"""Process wiring for PhoenaTranslator: the shared Runtime and its WSGI app."""

from __future__ import annotations

import logging
import warnings
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

# Former module globals, served with a DeprecationWarning for one release:
# services map onto ``runtime``, upper-case constants onto ``APP_CONFIG``.
# The private compatibility names of the old legacy_exports facade are gone.
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
_reported: set[str] = set()


def __getattr__(name: str):
    if name in _FORMER_SERVICES:
        replacement = f"runtime.{_FORMER_SERVICES[name]}"
        value = attrgetter(_FORMER_SERVICES[name])(runtime)
    elif name.isupper() and hasattr(APP_CONFIG, name.lower()):
        replacement = f"APP_CONFIG.{name.lower()}"
        value = getattr(APP_CONFIG, name.lower())
        # The directory constants were strings.
        value = str(value) if isinstance(value, Path) else value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    message = f"{__name__}.{name} is deprecated; use {replacement}"
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    if name not in _reported:
        _reported.add(name)
        log.warning(message)
    return value
