"""Fonts for deterministic PDF processing."""

from __future__ import annotations

import html
import fitz
import os
from pathlib import Path
from phoena_translator.pdf.inline_markup import _normalize_pdf_translation
from phoena_translator.pdf.text_shape import _plain_text
from phoena_translator.pdf.model import ElementType


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
            if elem.get("type") == ElementType.TEXT:
                chars.update(_plain_text(elem.get("content") or ""))

    return "".join(sorted(chars))


def _pdf_font_is_cff(source_path: str) -> bool:
    """Return whether the font is CFF-flavored OpenType or a collection.

    ``ttcf`` collections (e.g. NotoSansCJK ``.ttc``) are grouped with CFF:
    fontTools subsetting needs a font number for them, and the common CJK
    collections are CFF-flavored anyway, so treating them as unsubsettable
    routes them to the same complete-embed/size-gate policy."""
    try:
        with open(source_path, "rb") as handle:
            return handle.read(4) in (b"OTTO", b"ttcf")
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


def resolve_output_fonts(
    regular_override: str | None = None,
    bold_override: str | None = None,
) -> tuple[str, str]:
    """Fail fast on missing render prerequisites before provider work begins."""

    regular = _get_chinese_font_path(override=regular_override)
    bold = _get_chinese_font_path(bold=True, override=bold_override or regular_override)
    for font_path in {regular, bold}:
        if not pdf_font_is_safely_embeddable(font_path):
            raise RuntimeError(
                f"Chinese font {font_path} is too large for per-insert HTML-box "
                "embedding and cannot be subset (CFF-flavored or fontTools "
                "missing); configure a TrueType (.ttf) font via "
                "TRANSLATOR_PDF_FONT_REGULAR / TRANSLATOR_PDF_FONT_BOLD"
            )
    return regular, bold


def _htmlbox_font_ink(fontfile: str, sample: str) -> int:
    """Rasterize a short sample through the HTML-box @font-face path."""
    document = fitz.open()
    try:
        page = document.new_page(width=240, height=80)
        css = (
            f'@font-face{{font-family:probe;src:url("{os.path.basename(fontfile)}");}}'
            "*{font-family:probe;}"
        )
        archive = fitz.Archive(os.path.dirname(fontfile))
        page.insert_htmlbox(
            fitz.Rect(5, 5, 235, 75),
            f'<p style="font-size:22px">{html.escape(sample)}</p>',
            css=css,
            archive=archive,
        )
        pix = page.get_pixmap()
        return sum(
            1 for index in range(0, len(pix.samples), pix.n)
            if pix.samples[index] < 128
        )
    finally:
        document.close()


def _subset_font_renders_like_original(
    subset_path: str,
    original_path: str,
    used_chars: str,
) -> bool:
    """Fail closed if the subset draws materially less ink than the original.

    A structurally plausible subset can still resolve CJK codepoints to wrong
    or empty glyphs in MuPDF's @font-face loader; comparing rendered ink for a
    small CJK sample catches that before a whole document is assembled with an
    unreadable font.
    """
    sample = "".join(
        char for char in used_chars if ord(char) >= 0x2E80
    )[:6]
    if not sample:
        return True
    try:
        subset_ink = _htmlbox_font_ink(subset_path, sample)
        original_ink = _htmlbox_font_ink(original_path, sample)
    except Exception:
        return False
    if original_ink <= 0:
        return True
    return subset_ink >= original_ink * 0.5
