#!/usr/bin/env python3
"""WSGI (``gunicorn app:app``) and direct-execution entry point for PhoenaTranslator."""

from __future__ import annotations

from phoena_translator.application import APP_CONFIG, app, initialize_runtime


if __name__ == "__main__":
    initialize_runtime()
    app.run(host=APP_CONFIG.host, port=APP_CONFIG.port, debug=APP_CONFIG.debug)
