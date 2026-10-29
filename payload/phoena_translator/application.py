"""Process wiring for PhoenaTranslator: the shared Runtime and its WSGI app."""

from __future__ import annotations

import logging

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
