"""PDF settings that low-level helpers read at import time.

One shared configuration snapshot for the whole process (see
``config.get_app_config``): the composition root and these module constants
must never disagree about the same knob.  Code with an injected context
should prefer the context's value.
"""

from __future__ import annotations

from phoena_translator.config import get_app_config

_PDF_CONFIG = get_app_config()
PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE = _PDF_CONFIG.pdf_min_acceptable_htmlbox_scale
PDF_VECTOR_OCR_DPI = _PDF_CONFIG.pdf_vector_ocr_dpi
PDF_VECTOR_OCR_MIN_ALPHA_WORDS = _PDF_CONFIG.pdf_vector_ocr_min_alpha_words
del _PDF_CONFIG
