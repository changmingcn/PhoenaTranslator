"""Fonts for deterministic PDF processing."""

from __future__ import annotations


import os
from pathlib import Path
from phoena_translator.pdf.math_detection import (
    _normalize_pdf_translation,
    _plain_text,
)

try:
    from fontTools import subset as fonttools_subset
except ImportError:  # Optional: full fonts remain usable without subsetting.
    fonttools_subset = None


def pdf_font_subsetting_available() -> bool:
    return fonttools_subset is not None


# HTML-box insertion embeds a fresh copy of the font per call; only final
# compaction deduplicates them.  An unsubsettable font beyond this size grows
# the incremental work file by gigabytes and has been observed to silently
# lose the last pages' assembly, so it must be rejected up front.
PDF_HTMLBOX_MAX_EMBED_FONT_BYTES = 4 * 1024 * 1024


def pdf_font_is_safely_embeddable(font_path: str) -> bool:
    """Return whether a font can be used for per-insert HTML-box embedding."""
    try:
        size = os.path.getsize(font_path)
    except OSError:
        return False
    if size <= PDF_HTMLBOX_MAX_EMBED_FONT_BYTES:
        return True
    # Large fonts are fine only when the subsetting path can shrink them.
    return pdf_font_subsetting_available() and not _pdf_font_is_cff(font_path)

def _get_chinese_font_path(bold: bool = False, *, override: str | None = None) -> str:
    """Return the filesystem path to a Chinese font for fitz insert_textbox().
    An explicit configured override wins; otherwise search the known system
    locations. If bold=True, try to find a bold variant first."""
    if override:
        override_path = os.path.expanduser(str(override))
        if os.path.exists(override_path):
            return override_path
        raise RuntimeError(
            f"Configured Chinese font does not exist: {override_path}"
        )
    if bold:
        bold_candidates = [
            os.path.expanduser("~/fonts/NotoSansSC-Bold.ttf"),
            os.path.expanduser("~/fonts/SarasaGothicSC-Bold.ttf"),
        ]
        for fpath in bold_candidates:
            if os.path.exists(fpath):
                return fpath
    candidate_paths = [
        os.path.expanduser("~/fonts/SarasaGothicSC-Regular.ttf"),
        os.path.expanduser("~/fonts/NotoSansSC-Regular.ttf"),
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    ]
    for fpath in candidate_paths:
        if os.path.exists(fpath):
            return fpath
    raise RuntimeError("No Chinese font file found for PDF output")


def _collect_pdf_font_chars(page_extractions, translated_texts) -> str:
    chars = {chr(i) for i in range(32, 127)}
    chars.update({" ", "\n", "\t"})

    for page_num, info in page_extractions.items():
        if "cached" in info:
            page_values = info["cached"].values()
        else:
            page_values = translated_texts.get(page_num, {}).values()
        for text in page_values:
            chars.update(_plain_text(_normalize_pdf_translation(text)))
        # Element-granular source restores re-insert original text with this
        # font; include source characters so a restored fragment (including
        # Greek letters and math symbols) never renders as .notdef boxes.
        for elem in info.get("elements", []):
            if elem.get("type") == "text":
                chars.update(_plain_text(elem.get("content") or ""))

    return "".join(sorted(chars))


def _pdf_font_is_cff(source_path: str) -> bool:
    """Return whether the font file is CFF-flavored OpenType (``OTTO``)."""
    try:
        with open(source_path, "rb") as handle:
            return handle.read(4) == b"OTTO"
    except OSError:
        return False


def _subset_pdf_font(source_path: str, text_chars: str, output_dir: str, suffix: str) -> str:
    """Create a font subset containing only characters used by this PDF."""
    if not fonttools_subset:
        return source_path
    if _pdf_font_is_cff(source_path):
        # MuPDF's HTML-box @font-face loader mis-renders fontTools-subset
        # CFF outlines (CJK codepoints resolve to wrong or empty glyphs)
        # regardless of subsetter options.  Embed the complete font instead;
        # save-time garbage collection deduplicates the single copy.
        return source_path

    subset_path = os.path.join(output_dir, f"{Path(source_path).stem}-{suffix}{Path(source_path).suffix}")
    options = fonttools_subset.Options()
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.name_languages = ["*"]
    # Sarasa/Nerd fonts can contain OpenType private/layout tables that
    # fontTools cannot subset reliably. PDF htmlbox rendering only needs
    # glyph outlines and cmap data here, so drop layout tables for a small,
    # stable embeddable subset.
    options.layout_features = []
    options.drop_tables += ["GSUB", "GPOS", "GDEF", "PfEd"]
    options.notdef_outline = True
    options.recalc_average_width = True
    options.recalc_bounds = True
    options.canonical_order = True
    options.hinting = True
    options.desubroutinize = False

    font = fonttools_subset.load_font(source_path, options)
    try:
        subsetter = fonttools_subset.Subsetter(options=options)
        subsetter.populate(text=text_chars)
        subsetter.subset(font)
        fonttools_subset.save_font(font, subset_path, options)
    finally:
        font.close()
    return subset_path
