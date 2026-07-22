#!/usr/bin/env python3
"""Thin WSGI and direct-execution entry point for PhoenaTranslator."""

from __future__ import annotations

import sys

from phoena_translator import application as _application


if __name__ == "__main__":
    _application.initialize_runtime()
    _application.app.run(
        host=_application.APP_CONFIG.host,
        port=_application.APP_CONFIG.port,
        debug=_application.APP_CONFIG.debug,
    )
else:
    # Preserve legacy monkey-patching/import behavior without copying hundreds
    # of compatibility symbols into this root file.
    sys.modules[__name__] = _application
