"""Types for deterministic PDF processing."""

from __future__ import annotations

import re

from phoena_translator.config import get_app_config

SKIP_SECTION_HEADING_RE = re.compile(
    r"(?i)^(?:section\s+)?(?:\d+(?:\.\d+)*[.):]?\s*)?"
    r"(?:bibliography|references|index|endnotes?|glossary|acronyms|abbreviations)"
    r"(?:\s*(?:&|and)\s*(?:further\s+reading|notes|references))?$"
)

PDF_STRONG_MATH_FONT_TOKENS = (
    "cmex", "cmmi", "cmsy", "msam", "msbm", "eufm", "eurm",
    "math", "symbol", "euclid", "mt-extra", "mtextra", "mathematicalpi",
)

PDF_MATH_ITALIC_FONT_TOKENS = (
    "cmssi", "mathitalic", "mtmi", "texmathitalic",
)

PDF_FORMULA_GUARD_PADDING = 0.5

PDF_SIGNATURE_GEOMETRY_QUANTUM = 0.25

# Some generators (observed: Apache FOP 2.7) stroke every line of body text
# twice at the same coordinates.  The two copies are never bit-identical --
# measured drift is under a twentieth of a point -- so duplicate detection
# needs a tolerance rather than an equality test on the bbox.
PDF_DUPLICATE_GLYPH_POSITION_TOLERANCE = 0.25

# A citation's work title is the quoted run inside a bibliography entry.  Its
# presence separates "this entry has something to translate" from "this entry
# is only authors, initials and a year", where returning the source unchanged
# is the correct translation rather than a failure.
_PDF_QUOTED_TITLE_RE = re.compile(r"[“\"]([^”\"]{2,})[”\"]?")

_PDF_WRAPPED_MATH_DANGLING_TAIL_RE = re.compile(r"[(\[][A-Za-z]{0,3}$")

_PDF_WRAPPED_MATH_HYPHEN_TAIL_RE = re.compile(r"[A-Za-z]-$")

_PDF_WRAPPED_MATH_SCRIPT_LEAD_RE = re.compile(r"^[a-z]{1,3}[)\],]")

_PDF_WRAPPED_MATH_PROSE_WORD_RE = re.compile(r"[A-Za-z][a-z]{2,}")

DISCLAIMER_PATTERNS = re.compile(
    r'(?i)^(\s*\*?\s*(disclaimer|copyright|all rights reserved|legal notice|terms of use|'
    r'this (document|report|paper|publication) (is|was|has been)|'
    r'the views expressed|the opinions expressed|'
    r'for informational purposes only|not intended as|'
    r'source:|sources?:|\[\d+\]|footnote|endnote))',
    re.MULTILINE
)

_PDF_TRANSLATABLE_DISCLAIMER_LABELS = {
    "this report is intended for",
}

_PDF_TRANSLATABLE_REPORT_STRUCTURE_RE = re.compile(
    r"(?is)^\s*this\s+report\s+is\s+"
    r"(?:comprised|composed|organised|organized|structured|divided|arranged|made\s+up)\s+"
    r"(?:of|into)\b"
)

WATERMARK_PATTERNS = re.compile(
    r'(?i)\b(draft|confidential|sample|watermark|do not distribute|internal use|'
    r'for review only|preliminary|intended for|copy)\b'
)

PDF_BATCH_SEGMENT_RE = re.compile(r"PHOENA_SEG_\d{4}_[A-F0-9]{16}")

_PDF_PARAGRAPH_MARKER_ATOM = (
    r"(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdmIVXLCDM]{1,8})"
)

_PDF_NUMBERED_PARAGRAPH_LEAD_RE = re.compile(
    rf"^\s*(?P<marker>(?:{_PDF_PARAGRAPH_MARKER_ATOM}[.)]|"
    rf"\({_PDF_PARAGRAPH_MARKER_ATOM}\)))\s+(?=\S)"
)

_PDF_BULLETED_PARAGRAPH_LEAD_RE = re.compile(
    r"^\s*[\u2022\u25aa\u25cf\u25e6\u2043\u2219\uf0b7\-*\u00b7]+\s+(?=\S)"
)

_PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE = re.compile(
    r"^(?:[A-Za-z]|\d{1,3}|[*\u2020\u2021\u00a7\u00b6]{1,3})$"
)

_PDF_FOOTNOTE_MARKER_RE = re.compile(r"^(?:\d{1,3}|[*\u2020\u2021\u00a7\u00b6]{1,3})$")

_PDF_CITATION_PUBLISHER_CONNECTORS = frozenset({
    "and", "at", "by", "for", "in", "of", "on", "the", "to",
})

_PDF_SEMANTIC_TERMINAL_RE = re.compile(
    r"[.!?。！？:：;；][\"'’”)]*$"
)

_PDF_BARE_FOOTNOTE_LEAD_RE = re.compile(r"^\d{1,3}\s+(?=[A-Z])")

_PRESERVED_URI_BODY = r"[A-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]+"

_PRESERVED_URI_LAYOUT_WRAPS = (
    rf"(?:"
    rf"(?<=[/_?&=%\-])\s+{_PRESERVED_URI_BODY}"
    rf"|(?<=\.)\s+(?=[A-Z0-9._~\-]+[/?:#]){_PRESERVED_URI_BODY}"
    rf")*"
)

_PRESERVED_IDENTIFIER_RE = re.compile(
    rf"(?i)(?:"
    rf"(?:https?|ftp)://{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf"|www\.{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf"|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{{2,}}"
    rf"|(?:doi:\s*)?10\.\d{{4,9}}/{_PRESERVED_URI_BODY}{_PRESERVED_URI_LAYOUT_WRAPS}"
    rf")"
)

_PRESERVED_STRUCTURED_CODE_RE = re.compile(
    r"(?i)(?=[A-Z0-9/_\-]*\d)(?=(?:.*[/_\-]){2,})"
    r"[A-Z0-9]+(?:[/_\-][A-Z0-9]+){2,}[/_\-]?"
)

_SHORT_CITATION_YEAR_RE = re.compile(r"\((?:18|19|20)\d{2}[a-z]?\)", re.IGNORECASE)

_SHORT_CITATION_WORD_RE = re.compile(
    r"[^\W\d_]+(?:['’\-][^\W\d_]+)*\.?",
    re.UNICODE,
)

_SHORT_CITATION_CONNECTORS = {
    "al", "and", "da", "de", "del", "der", "di", "dos", "du", "et",
    "jr", "la", "le", "st", "van", "von",
}

_PDF_TRANSLATABLE_SHORT_LABELS = {
    "abstract",
    "acknowledgement",
    "acknowledgements",
    "annex",
    "annexure",
    "appendix",
    "background",
    "bibliography",
    "chapter",
    "chart",
    "conclusion",
    "conclusions",
    "contents",
    "definition",
    "definitions",
    "discussion",
    "graphical analysis",
    "executive summary",
    "figure",
    "findings",
    "foreword",
    "glossary",
    "implication",
    "implications",
    "index",
    "intern",
    "introduction",
    "interim order",
    "methodology",
    "model",
    "models",
    "note",
    "overview",
    "part",
    "preface",
    "pfutp regulations",
    "project co-ordinator",
    "project coordinator",
    "project leader",
    "project manager",
    "project researcher",
    "recommendation",
    "recommendations",
    "reference",
    "references",
    "ratio",
    "result",
    "results",
    "section",
    "sebi act",
    "source",
    "table",
    "test",
    "tests",
    "terms",
    "acronyms",
    "chapter title",
    "workshop reports",
    "year",
    "urgency",
    "volatility",
}

_PDF_SHORT_LABEL_CONNECTORS = {
    "and", "at", "by", "da", "de", "der", "di", "for", "from", "in", "of",
    "on", "the", "to", "van", "von", "with",
}

_PDF_PROPER_NAME_SUFFIXES = {
    "advisors", "asset", "bank", "capital", "company", "consulting", "corp",
    "corporation", "exchange", "fund", "group", "holdings", "inc", "investments",
    "limited", "llc", "lp", "ltd", "management", "markets", "partners", "plc",
    "securities", "trust", "university", "institute", "college", "school",
}

_PDF_PERSON_HONORIFICS = {
    "dame", "dr", "mr", "mrs", "ms", "prof", "professor", "sir",
}

_PDF_DISPLAY_IDENTITY_END_WORDS = {
    "advisors", "association", "authority", "bank", "commission", "commissions",
    "committee", "company",
    "consulting", "corporation", "council", "department", "exchange", "fund",
    "government", "group", "holdings", "institute", "management", "ministry",
    "office", "organization", "organisation", "partners", "school", "trust",
    "university",
}

_PDF_PROPER_NAME_CONNECTORS = {
    "and", "at", "by", "de", "del", "der", "di", "dos", "du", "et",
    "for", "in", "jr", "la", "le", "of", "on", "sr", "the", "van",
    "von",
}

PDF_TRANSLATABLE_CITATION_LABELS = (
    (re.compile(r"(?i)\blast\s+accessed\s*:\s*"), "最后访问日期："),
    (re.compile(r"(?i)\baccessed\s*:\s*"), "访问日期："),
    (re.compile(r"(?i)\bretrieved(?:\s+on)?\s*:\s*"), "检索日期："),
    (re.compile(r"(?i)\bavailable\s+at\s*:\s*"), "可获取地址："),
    (re.compile(r"(?i)\bavailable\s*:\s*"), "可获取："),
)

PDF_FIXED_SHORT_LABEL_TRANSLATIONS = {
    "all†": "全部†",
    "day": "日盘",
    "half": "一半",
    "internal": "内部",
    "max": "最大值",
    "min": "最小值",
    "external": "外部",
    "other": "其他",
    "others": "其他",
    "night": "夜间",
    "received by": "接收方",
    "enter*": "输入*",
}

PDF_SUPERSCRIPT_PLACEHOLDER_RE = re.compile(
    r"PHOENA_SUP_\d{4}_[0-9A-F]{16}"
)

PDF_INLINE_MATH_PLACEHOLDER_RE = re.compile(
    r"PHOENA_IMATH_\d{4}_[0-9A-F]{16}"
)

PDF_IDENTIFIER_PLACEHOLDER_RE = re.compile(
    r"PHOENA_ID_\d{4}_[0-9A-F]{16}"
)

PDF_PAGE_CACHE_SCHEMA_VERSION = 3
PDF_LAYOUT_SEMANTICS_VERSION = "2026-08-semantic-paragraphs-v39"
# These older geometry identities remain safe to rebind. The identity
# migration rejects split/merged cells, and active validation retranslates
# semantically incomplete survivors.
PDF_LAYOUT_CACHE_COMPATIBLE_VERSIONS = {
    PDF_LAYOUT_SEMANTICS_VERSION,
    "2026-07-semantic-paragraphs-v29",
    "2026-07-semantic-paragraphs-v28",
    "2026-07-semantic-paragraphs-v27",
    "2026-07-semantic-paragraphs-v26",
    "2026-07-semantic-paragraphs-v22",
    "2026-07-semantic-paragraphs-v21",
    "2026-07-semantic-paragraphs-v20",
    "2026-07-semantic-paragraphs-v19",
    "2026-07-semantic-paragraphs-v13",
    "2026-07-semantic-paragraphs-v12",
    "2026-07-semantic-paragraphs-v11",
}
PDF_BATCH_SEPARATOR = "≡≡≡SPLIT≡≡≡"
PDF_TEXT_NEIGHBOR_GUARD_PADDING = 0.6
PDF_MAX_TRANSLATION_EXPANSION_RATIO = 1.40
PDF_MAX_TRANSLATION_EXPANSION_SLACK = 64
PDF_HTMLBOX_SUPERSCRIPT_CSS = (
    'sup{line-height:0;vertical-align:super;white-space:nowrap;}'
)

_PDF_MIXED_TRANSLATION_ENGLISH_RUN_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z'’\-]*\b(?:\s+|[,;:()–—-]+)){4,}"
    r"\b[A-Za-z][A-Za-z'’\-]*\b"
)
_PDF_WORK_TITLE_QUOTE_CHARS = "'\"‘’“”"
_PDF_UNTRANSLATED_LEAK_WORD_RE = re.compile(
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*"
)
_PDF_UNTRANSLATED_LEAK_RUN_RE = re.compile(
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*"
    r"(?:[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]+"
    r"[A-Za-z][a-z]{2,}(?:['\u2019][a-z]+)*)+"
)
_PDF_UNTRANSLATED_LEAK_LONE_WORD_RE = re.compile(
    r"[a-z]{3,}(?:['\u2019][a-z]+)*"
)
_PDF_UNTRANSLATED_LEAK_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
_PDF_UNTRANSLATED_LEAK_UNIT_WORDS = {
    "bps", "ppm", "pts", "bbl", "mmbtu", "kwh", "mwh", "gwh", "twh",
}
_PDF_UNTRANSLATED_LEAK_SYMBOL_WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega", "rho", "vega",
}
_PDF_UNTRANSLATED_LEAK_NEIGHBOR_BEFORE_RE = re.compile(
    r"[A-Za-z]{3,}[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]*$"
)
_PDF_UNTRANSLATED_LEAK_NEIGHBOR_AFTER_RE = re.compile(
    r"^[\s,;:()\"'\u201c\u201d\u2018\u2019\-\u2013\u2014]*[A-Za-z]{3,}"
)
_PDF_REFERENCE_TRANSLATION_YEAR_RE = re.compile(
    r"(?<!\d)(?:18\d{2}|19\d{2}|20[0-2]\d)[a-z]?(?!\d)",
    re.IGNORECASE,
)
_PDF_TOC_ENTRY_RE = re.compile(
    r"(?P<leader>(?:[.…·]\s*){4,})\s*(?P<page>\d+)\b"
)

# One shared configuration snapshot for the whole process (see
# ``config.get_app_config``): the composition root and these module constants
# must never disagree about the same knob.
_PDF_CONFIG = get_app_config()
PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE = _PDF_CONFIG.pdf_min_acceptable_htmlbox_scale
PDF_VECTOR_OCR_DPI = _PDF_CONFIG.pdf_vector_ocr_dpi
PDF_VECTOR_OCR_MIN_ALPHA_WORDS = _PDF_CONFIG.pdf_vector_ocr_min_alpha_words
del _PDF_CONFIG

class PDFPageRenderError(RuntimeError):
    """A page could not be redrawn safely and may be retried from source."""

    def __init__(
        self,
        page: int,
        element: int,
        reason: str,
        *,
        retryable: bool = True,
    ):
        self.page = int(page)
        self.element = int(element)
        self.retryable = bool(retryable)
        self.fallback_pages = [self.page] if self.page > 0 else []
        self.repair_pages = (
            [self.page] if self.retryable and self.page > 0 else []
        )
        super().__init__(f"Page {self.page} elem {self.element}: {reason}")


class PDFPageExtractionError(RuntimeError):
    """A source page could not be extracted safely and may be retried once."""

    def __init__(self, page: int, reason: str):
        self.page = int(page)
        self.fallback_pages = [self.page] if self.page > 0 else []
        self.repair_pages = [self.page] if self.page > 0 else []
        super().__init__(f"Page {self.page} extraction failed: {reason}")


class PDFTranslationIntegrityError(RuntimeError):
    """A page contains a required element without a verified translation."""

    def __init__(self, page: int, issues: list[dict], translations: dict | None = None):
        self.page = int(page)
        self.issues = list(issues or [])
        # The surviving per-element candidates let the caller downgrade a
        # page-level failure to element-level source fallbacks instead of
        # discarding every verified translation on the page.
        self.translations = dict(translations or {})
        summary = ", ".join(
            f"elem {issue.get('element')} {issue.get('reason')}"
            for issue in self.issues[:5]
        )
        super().__init__(
            f"PDF translation integrity failed on page {self.page}: "
            f"{summary or 'unknown translation defect'}"
        )


class PDFPageTranslationError(RuntimeError):
    """One or more pages failed both the normal and clean translation pass."""

    def __init__(self, errors: dict[int, Exception]):
        self.errors = dict(errors or {})
        self.pages = sorted(int(page) for page in self.errors)
        summary = "; ".join(
            f"page {page}: {self.errors[page]}" for page in self.pages[:5]
        )
        super().__init__(
            f"PDF page translation failed after one clean retry "
            f"({len(self.pages)} page(s)): {summary}"
        )


class PDFStructureValidationError(RuntimeError):
    """A final PDF audit failure with pages eligible for one clean rebuild."""

    def __init__(self, check: dict):
        self.check = dict(check or {})
        status = str(self.check.get("status", "error"))
        warnings = self.check.get("warnings") or []
        try:
            warning_count = int(self.check.get("warning_count", len(warnings)))
        except (TypeError, ValueError):
            warning_count = max(1, len(warnings))
        warning_types = sorted({
            str(warning.get("type", "unknown"))
            for warning in warnings
            if isinstance(warning, dict)
        })
        summary = ", ".join(warning_types[:5]) or status
        self.repair_pages = sorted({
            int(warning["page"])
            for warning in warnings
            if isinstance(warning, dict)
            and str(warning.get("page", "")).isdigit()
            and int(warning["page"]) > 0
        })
        super().__init__(
            f"PDF structure validation failed ({warning_count} warning(s): {summary})"
        )
