"""Per-task PDF caches: page translations (bound to element identities),
the saved layout, and the audit file."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable

import fitz

from phoena_translator.config import get_app_config
from phoena_translator.pdf.extract.signatures import _pdf_signature_number
from phoena_translator.pdf.geometry import _get_pdf_elem_rect
from phoena_translator.pdf.model import (
    ElementType,
    PageExtraction,
    PdfAudit,
    SkipReason,
)
from phoena_translator.pdf.translate.deterministic_labels import (
    _translate_pdf_deterministic_labels,
)
from phoena_translator.pdf.translate.inline_markup import (
    _pdf_inline_markup_preserved,
    _pdf_inline_math_fragments_preserved,
)
from phoena_translator.pdf.translate.targets import _pdf_element_requires_translation
from phoena_translator.pdf.translate.validation import (
    _pdf_element_translation_needs_retry,
)
from phoena_translator.storage_layout import (
    pdf_audit_path,
    pdf_layout_cache_path,
    pdf_page_cache_dir,
    pdf_page_cache_path,
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


log = logging.getLogger("translator")

_progress_dir_provider: Callable[[], str] | None = None


@dataclass(frozen=True)
class CacheIdentity:
    """Typed in-memory cache key whose persisted form remains its digest."""

    digest: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError(
                "cache identity digest must be 64 lowercase hexadecimal characters"
            )


def configure_pdf_cache_progress_dir(provider: Callable[[], str]) -> None:
    """Bind cache paths to the application runtime without importing ``app``."""
    global _progress_dir_provider
    _progress_dir_provider = provider


def _cache_progress_dir() -> str:
    if _progress_dir_provider is not None:
        return str(_progress_dir_provider())
    return str(get_app_config().progress_dir)

def _write_json_atomically(path: str, payload, **dump_options) -> None:
    """Write UTF-8 JSON beside ``path`` and rename it into place."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, **dump_options)
    os.replace(tmp_path, path)


def _pdf_source_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_cache_path(task_id: str, page_num: int) -> str:
    return pdf_page_cache_path(_cache_progress_dir(), task_id, page_num)


def _discard_pdf_page_translation_cache(task_id: str, page_num: int) -> None:
    """Delete one page's cached translations; a missing file is fine."""
    try:
        os.unlink(_pdf_page_cache_path(task_id, page_num))
    except FileNotFoundError:
        pass


def _load_pdf_page_translation_cache(
    path: str,
    source_sha256: str | None = None,
    elements: list[dict] | None = None,
) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != PDF_PAGE_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("layout_semantics") not in PDF_LAYOUT_CACHE_COMPATIBLE_VERSIONS:
        return None
    cached_source_sha256 = payload.get("source_sha256")
    if not isinstance(cached_source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", cached_source_sha256):
        return None
    if source_sha256 is not None and cached_source_sha256 != source_sha256:
        return None

    translations = payload.get("translations")
    if not isinstance(translations, dict):
        return None
    if not all(
        re.fullmatch(r"\d+", str(key)) and isinstance(value, str)
        for key, value in translations.items()
    ):
        return None

    element_identities = payload.get("element_identities")
    if not isinstance(element_identities, dict):
        return None
    normalized_translations = {str(key): value for key, value in translations.items()}
    if not all(
        key in element_identities
        and isinstance(element_identities[key], str)
        and re.fullmatch(r"[0-9a-f]{64}", element_identities[key])
        for key in normalized_translations
    ):
        return None
    if elements is None:
        return normalized_translations

    # Page element indices are not stable when table/formula extraction is
    # corrected or when a task resumes after an interrupted layout pass. Map
    # cached translations by complete source/layout identity instead of
    # trusting a positional index from an older extraction.
    cached_by_identity: dict[CacheIdentity, list[str]] = {}
    for key, translation in normalized_translations.items():
        cached_identity = CacheIdentity(element_identities[key])
        cached_by_identity.setdefault(cached_identity, []).append(translation)

    migrated: dict[str, str] = {}
    for current_index, elem in enumerate(elements or []):
        if elem.get("type") != ElementType.TEXT:
            continue
        key = str(current_index)
        if not _pdf_element_requires_translation(elem):
            source_text = (
                "" if elem.get("skip_translate_reason") == SkipReason.WATERMARK
                else elem.get("rich_content") or elem.get("content", "")
            )
            migrated[key] = _translate_pdf_deterministic_labels(source_text)
            continue
        identity_variants = []
        for identity in (
            _pdf_cache_identity(elem),
            # A short-lived v28 build wrote explicit false/empty vector-OCR
            # fields into every otherwise unchanged element identity.  Keep
            # those already-paid caches reusable while the canonical identity
            # below remains byte-compatible with v26/v27 for native text.
            _pdf_cache_identity(
                elem,
                _include_empty_vector_fields=True,
            ),
        ):
            if identity and identity not in identity_variants:
                identity_variants.append(identity)
        candidates = [
            candidate
            for identity in identity_variants
            for candidate in cached_by_identity.get(identity, [])
        ]
        # Element-granular fail-open caches deliberately contain the exact
        # English source and were written with this explicit reason in their
        # identity.  On a fresh extraction that in-memory marker is absent;
        # probe the exact fallback identity separately so an intentional
        # fallback is not mistaken for a poisoned source-echo cache and sent
        # through the provider again on every render-recovery round.
        fallback_element = dict(elem)
        fallback_element["skip_translate_reason"] = (
            SkipReason.TRANSLATION_INTEGRITY_FALLBACK
        )
        fallback_identities = []
        for identity in (
            _pdf_cache_identity(fallback_element),
            _pdf_cache_identity(
                fallback_element,
                _include_empty_vector_fields=True,
            ),
        ):
            if identity and identity not in fallback_identities:
                fallback_identities.append(identity)
        fallback_candidates = [
            candidate
            for identity in fallback_identities
            for candidate in cached_by_identity.get(identity, [])
        ]
        if len(candidates) == 1:
            candidate = candidates[0]
            source = elem.get("rich_content") or elem.get("content", "")
            # A classifier correction can turn a previously preserved source
            # block into required prose.  Treat its old identity translation
            # (usually the untouched English source) as missing instead of
            # poisoning the whole otherwise-valid page cache.
            if (
                not _pdf_element_translation_needs_retry(elem, source, candidate)
                and _pdf_inline_markup_preserved(source, candidate)
                and _pdf_inline_math_fragments_preserved(elem, candidate)
            ):
                migrated[key] = candidate
        if key not in migrated and len(fallback_candidates) == 1:
            candidate = fallback_candidates[0]
            source = elem.get("content", "")
            if candidate == source:
                elem["skip_translate_reason"] = (
                    SkipReason.TRANSLATION_INTEGRITY_FALLBACK
                )
                migrated[key] = candidate
    return migrated


def _pdf_cache_identity(
    elem: dict,
    *,
    _include_empty_vector_fields: bool = False,
) -> CacheIdentity | None:
    """Return a source-and-layout identity for explicit legacy cache migration.

    Page element indices are intentionally excluded: formula promotion and
    cross-page sentence merging can change what an index means between layout
    semantics versions.  A migration may reuse a translation only when the
    complete source text, rich markup, paragraph structure, geometry, and
    relevant classification/style metadata still agree.
    """
    if not isinstance(elem, dict) or elem.get("type") != ElementType.TEXT:
        return None

    try:
        bbox = _get_pdf_elem_rect(elem)
        normalized_bbox = [
            _pdf_signature_number(value)
            for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        ]
    except (TypeError, ValueError, AssertionError):
        # PyMuPDF raises AssertionError for Rect(None) in this build; see
        # _safe_bbox_rect in geometry.py for the same guard rationale.
        return None

    paragraphs = [
        {
            "plain": paragraph.get("plain") or "",
            "rich": paragraph.get("rich") or "",
            "text_align": paragraph.get("text_align") or "",
            "nowrap": bool(paragraph.get("nowrap")),
            "margin_left": _pdf_signature_number(float(paragraph.get("margin_left", 0.0))),
            "text_indent": _pdf_signature_number(float(paragraph.get("text_indent", 0.0))),
            "gap_before": _pdf_signature_number(float(paragraph.get("gap_before", 0.0))),
            "first_line_indent": bool(paragraph.get("first_line_indent")),
        }
        for paragraph in (elem.get("paragraphs") or [])
        if isinstance(paragraph, dict)
    ]
    superscript_runs = [
        {
            "text": run.get("text") or "",
            "source": run.get("source") or "",
            "scale": round(float(run.get("scale", 0.0)), 3),
        }
        for run in (elem.get("superscript_runs") or [])
        if isinstance(run, dict)
    ]
    inline_math_fragments = [
        {
            "text": fragment.get("text") or "",
            "font_kind": fragment.get("font_kind") or "",
        }
        for fragment in (elem.get("inline_math_fragments") or [])
        if isinstance(fragment, dict) and (fragment.get("text") or "")
    ]
    identity_payload = {
        "content": elem.get("content") or "",
        "rich_content": elem.get("rich_content") or "",
        "paragraphs": paragraphs,
        "bbox": normalized_bbox,
        "skip_translate_reason": elem.get("skip_translate_reason"),
        "layout_class": elem.get("layout_class"),
        "table_hint": bool(elem.get("table_hint")),
        "single_line_heading": bool(elem.get("single_line_heading")),
        "bold": bool(elem.get("bold")),
        "color": elem.get("color"),
        "fontsize": _pdf_signature_number(float(elem.get("fontsize", 0.0))),
        "non_horizontal": bool(elem.get("non_horizontal")),
        "preserve_source_style": bool(elem.get("preserve_source_style")),
        "superscript_runs": superscript_runs,
        "inline_math_fragments": inline_math_fragments,
    }
    # Do not perturb every native-text identity merely because vector-outline
    # recovery was added.  Vector geometry is identity-bearing only for an
    # actual recovered OCR element.  This preserves safe v26/v27 cache reuse
    # for all unaffected pages while ensuring page 175/177 OCR cells cannot
    # bind to a native-text cache entry.
    if elem.get("vector_ocr") or _include_empty_vector_fields:
        identity_payload.update({
            "vector_ocr": bool(elem.get("vector_ocr")),
            "vector_ocr_source_signature": elem.get(
                "vector_ocr_source_signature"
            ),
            "vector_ocr_fill_rects": elem.get("vector_ocr_fill_rects") or [],
            "vector_ocr_erase_rects": elem.get("vector_ocr_erase_rects") or [],
        })
    encoded = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CacheIdentity(hashlib.sha256(encoded).hexdigest())


def _reconcile_pdf_page_translation_cache(
    elements: list[dict],
    translations: dict,
) -> tuple[dict, int]:
    """Remove stale redraws for elements the current classifier preserves.

    A prior run may have cached rich formula markup as if it were translated
    text. Re-extraction can now classify that element as a formula image, or
    decide that a remaining text element is not translatable. In either case,
    the original PDF object must survive assembly unchanged.
    """
    reconciled = {str(key): value for key, value in (translations or {}).items()}
    repaired = 0

    for elem_index, elem in enumerate(elements or []):
        key = str(elem_index)
        if key not in reconciled:
            continue
        if elem.get("type") != ElementType.TEXT:
            del reconciled[key]
            repaired += 1
            continue

        source_text = elem.get("rich_content") or elem.get("content", "")
        if elem.get("skip_translate_reason") == SkipReason.WATERMARK:
            if reconciled[key] != "":
                reconciled[key] = ""
                repaired += 1
            continue
        if not _pdf_element_requires_translation(elem):
            deterministic_label = _translate_pdf_deterministic_labels(source_text)
            if reconciled[key] != deterministic_label:
                reconciled[key] = deterministic_label
                repaired += 1
            continue
        if _pdf_element_requires_translation(elem) and isinstance(reconciled[key], str):
            normalized_candidate = _translate_pdf_deterministic_labels(
                reconciled[key]
            )
            if normalized_candidate != reconciled[key]:
                reconciled[key] = normalized_candidate
                repaired += 1

    return reconciled, repaired


def _save_pdf_page_translation_cache(
    task_id: str,
    page_num: int,
    translations: dict,
    source_sha256: str,
    elements: list[dict] | None = None,
) -> str:
    page_cache_dir = pdf_page_cache_dir(_cache_progress_dir(), task_id)
    os.makedirs(page_cache_dir, exist_ok=True)
    path = _pdf_page_cache_path(task_id, page_num)
    normalized_translations = {
        str(key): value for key, value in translations.items()
    }
    element_identities = {}
    for key in normalized_translations:
        try:
            elem = (elements or [])[int(key)]
        except (IndexError, TypeError, ValueError):
            continue
        identity = _pdf_cache_identity(elem)
        if identity:
            element_identities[key] = identity.digest
    payload = {
        "schema_version": PDF_PAGE_CACHE_SCHEMA_VERSION,
        "layout_semantics": PDF_LAYOUT_SEMANTICS_VERSION,
        "source_sha256": source_sha256,
        "translations": normalized_translations,
        "element_identities": element_identities,
    }
    _write_json_atomically(path, payload, separators=(",", ":"))
    return path


def _load_pdf_cached_page_indices(
    task_id: str,
    source_sha256: str | None = None,
) -> set[int]:
    page_cache_dir = pdf_page_cache_dir(_cache_progress_dir(), task_id)
    if not os.path.isdir(page_cache_dir):
        return set()

    cached_indices = set()
    invalid_count = 0
    for name in os.listdir(page_cache_dir):
        match = re.fullmatch(r"page_(\d+)\.json", name)
        if not match:
            continue
        path = os.path.join(page_cache_dir, name)
        if _load_pdf_page_translation_cache(path, source_sha256) is not None:
            cached_indices.add(int(match.group(1)))
        else:
            invalid_count += 1
    if invalid_count:
        log.warning(
            f"[{task_id}] Ignoring {invalid_count} incompatible or invalid PDF page cache file(s)"
        )
    return cached_indices


def _pdf_layout_cache_path(task_id: str) -> str:
    return pdf_layout_cache_path(_cache_progress_dir(), task_id)


def _discard_pdf_layout_cache(task_id: str) -> None:
    """Delete the saved layout; a missing file is fine."""
    try:
        os.unlink(_pdf_layout_cache_path(task_id))
    except FileNotFoundError:
        pass


def _pdf_audit_path(task_id: str) -> str:
    return pdf_audit_path(_cache_progress_dir(), task_id)


def _save_pdf_audit(task_id: str, audit: PdfAudit) -> str:
    path = _pdf_audit_path(task_id)
    _write_json_atomically(path, audit, separators=(",", ":"))
    return path


def _json_safe_pdf_layout(value):
    if isinstance(value, fitz.Rect):
        return [value.x0, value.y0, value.x1, value.y1]
    if isinstance(value, dict):
        return {str(k): _json_safe_pdf_layout(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_pdf_layout(v) for v in value]
    return value


def _save_pdf_layout_cache(task_id: str, page_extractions: dict[int, PageExtraction], page_rects: dict):
    payload = {
        "page_extractions": _json_safe_pdf_layout(page_extractions),
        "page_rects": _json_safe_pdf_layout(page_rects),
    }
    path = _pdf_layout_cache_path(task_id)
    _write_json_atomically(path, payload)
    return path


def _load_pdf_layout_cache(task_id: str) -> tuple[dict, dict]:
    path = _pdf_layout_cache_path(task_id)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    page_extractions = {int(k): v for k, v in payload.get("page_extractions", {}).items()}
    page_rects = {
        int(k): fitz.Rect(v)
        for k, v in payload.get("page_rects", {}).items()
    }
    return page_extractions, page_rects


