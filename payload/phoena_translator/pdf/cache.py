"""Cache for deterministic PDF processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable
from phoena_translator.config import get_app_config
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
import fitz
from phoena_translator.pdf.math_detection import (
    _pdf_inline_markup_preserved,
    _pdf_inline_math_fragments_preserved,
    _pdf_signature_number,
    _plain_text,
)
from phoena_translator.pdf.targets import (
    _pdf_element_requires_translation,
    _translate_pdf_deterministic_labels,
)
from phoena_translator.pdf.types import (
    PDF_LAYOUT_CACHE_COMPATIBLE_VERSIONS,
    PDF_LAYOUT_SEMANTICS_VERSION,
    PDF_PAGE_CACHE_SCHEMA_VERSION,
)
from phoena_translator.pdf.translation import (
    _pdf_element_translation_needs_retry,
)

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

def _pdf_source_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_cache_path(task_id: str, page_num: int) -> str:
    return os.path.join(_cache_progress_dir(), f"{task_id}_pages", f"page_{page_num}.json")


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
        if elem.get("type") != "text":
            continue
        key = str(current_index)
        if not _pdf_element_requires_translation(elem):
            source_text = (
                "" if elem.get("skip_translate_reason") == "watermark"
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
    if not isinstance(elem, dict) or elem.get("type") != "text":
        return None

    try:
        bbox = fitz.Rect(elem.get("bbox", elem.get("rect")))
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


def _pdf_cache_element_identity(
    elem: dict,
    *,
    _include_empty_vector_fields: bool = False,
) -> str | None:
    """Compatibility facade returning the historical 64-character digest."""

    identity = _pdf_cache_identity(
        elem,
        _include_empty_vector_fields=_include_empty_vector_fields,
    )
    return identity.digest if identity is not None else None


def _migrate_pdf_page_translation_cache_by_identity(
    legacy_elements: list[dict],
    current_elements: list[dict],
    legacy_translations: dict,
) -> tuple[dict, dict]:
    """Fail-closed migration from a legacy page cache to current semantics.

    The caller must explicitly supply the legacy layout alongside the legacy
    translations.  Every current API-translatable element must have one, and
    only one, source/layout-identical legacy element with a cached translation.
    Non-translatable text is reconstructed deterministically; formulas and
    other immutable objects never enter the migrated cache.
    """
    legacy_by_identity: dict[CacheIdentity, list[tuple[int, str]]] = {}
    valid_legacy_entries = 0
    for raw_index, translation in (legacy_translations or {}).items():
        if not re.fullmatch(r"\d+", str(raw_index)) or not isinstance(translation, str):
            continue
        old_index = int(raw_index)
        if old_index < 0 or old_index >= len(legacy_elements or []):
            continue
        old_elem = legacy_elements[old_index]
        if not _pdf_element_requires_translation(old_elem):
            continue
        identity = _pdf_cache_identity(old_elem)
        if identity is None:
            continue
        legacy_by_identity.setdefault(identity, []).append((old_index, translation))
        valid_legacy_entries += 1

    current_identity_counts: dict[CacheIdentity, int] = {}
    for elem in current_elements or []:
        if not _pdf_element_requires_translation(elem):
            continue
        identity = _pdf_cache_identity(elem)
        if identity is not None:
            current_identity_counts[identity] = current_identity_counts.get(identity, 0) + 1

    migrated: dict[str, str] = {}
    missing_indices: list[int] = []
    ambiguous_indices: list[int] = []
    reused_legacy_indices: set[int] = set()
    required_count = 0

    for current_index, elem in enumerate(current_elements or []):
        if elem.get("type") != "text":
            continue
        key = str(current_index)
        if not _pdf_element_requires_translation(elem):
            source_text = (
                "" if elem.get("skip_translate_reason") == "watermark"
                else elem.get("rich_content") or elem.get("content", "")
            )
            migrated[key] = _translate_pdf_deterministic_labels(source_text)
            continue

        required_count += 1
        identity = _pdf_cache_identity(elem)
        candidates = legacy_by_identity.get(identity, []) if identity else []
        if identity and current_identity_counts.get(identity) == 1 and len(candidates) == 1:
            old_index, translation = candidates[0]
            migrated[key] = translation
            reused_legacy_indices.add(old_index)
        elif len(candidates) > 1 or (identity and current_identity_counts.get(identity, 0) > 1):
            ambiguous_indices.append(current_index)
        else:
            missing_indices.append(current_index)

    report = {
        "complete": not missing_indices and not ambiguous_indices,
        "required_count": required_count,
        "reused_count": len(reused_legacy_indices),
        "missing_indices": missing_indices,
        "ambiguous_indices": ambiguous_indices,
        "ignored_legacy_entries": max(valid_legacy_entries - len(reused_legacy_indices), 0),
    }
    return migrated, report


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
        if elem.get("type") != "text":
            del reconciled[key]
            repaired += 1
            continue

        source_text = elem.get("rich_content") or elem.get("content", "")
        if elem.get("skip_translate_reason") == "watermark":
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
    page_cache_dir = os.path.join(_cache_progress_dir(), f"{task_id}_pages")
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
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)
    return path


def _load_pdf_cached_page_indices(
    task_id: str,
    source_sha256: str | None = None,
) -> set[int]:
    page_cache_dir = os.path.join(_cache_progress_dir(), f"{task_id}_pages")
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
    return os.path.join(_cache_progress_dir(), f"{task_id}_layout.json")


def _pdf_audit_path(task_id: str) -> str:
    return os.path.join(_cache_progress_dir(), f"{task_id}_audit.json")


def _save_pdf_audit(task_id: str, audit: dict) -> str:
    path = _pdf_audit_path(task_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)
    return path


def _json_safe_pdf_layout(value):
    if isinstance(value, fitz.Rect):
        return [value.x0, value.y0, value.x1, value.y1]
    if isinstance(value, dict):
        return {str(k): _json_safe_pdf_layout(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_pdf_layout(v) for v in value]
    return value


def _save_pdf_layout_cache(task_id: str, page_extractions: dict, page_rects: dict):
    payload = {
        "page_extractions": _json_safe_pdf_layout(page_extractions),
        "page_rects": _json_safe_pdf_layout(page_rects),
    }
    path = _pdf_layout_cache_path(task_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)
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


def _pdf_source_page_fallback_entry(
    page: int,
    stage: str,
    reason: str,
    *,
    error_type: str = "",
) -> dict:
    """Return the public audit record for one exact source-page fallback."""
    entry = {
        "page": int(page),
        "stage": str(stage or "unknown"),
        "reason": str(reason or "unspecified page failure")[:2000],
    }
    if error_type:
        entry["error_type"] = str(error_type)
    return entry


def _append_pdf_source_page_fallback(entries: list[dict], entry: dict) -> dict:
    """Append a fallback once per page while retaining the first root cause."""
    page = int(entry.get("page", 0) or 0)
    for existing in entries:
        if int(existing.get("page", 0) or 0) == page:
            return existing
    entries.append(dict(entry))
    entries.sort(key=lambda item: int(item.get("page", 0) or 0))
    return entry


def _exclude_pdf_audit_expectations_for_source_pages(
    audit: dict,
    page_numbers: set[int] | list[int],
) -> None:
    """Exclude knowingly untranslated source pages from translation-only checks."""
    pages = {int(page) for page in page_numbers if int(page) > 0}
    if not pages:
        return

    excluded = audit.setdefault("source_page_fallback_expectation_exclusions", {})
    for key in (
        "superscript_expectations",
        "formula_expectations",
        "vector_ocr_expectations",
    ):
        values = list(audit.get(key) or [])
        retained = [
            value
            for value in values
            if int(value.get("page", 0) or 0) not in pages
        ]
        removed = len(values) - len(retained)
        if removed:
            excluded[key] = int(excluded.get(key, 0) or 0) + removed
            audit[key] = retained


def _record_pdf_element_source_fallback(
    audit: dict,
    page_number: int,
    element_index: int,
    stage: str,
    reason: str,
    source_text: str = "",
) -> dict:
    """Record that one element keeps its exact source text inside a translated page.

    This is the element-granular sibling of ``source_page_fallbacks``: the page
    itself still translates and assembles normally, only this element's original
    ink (or restored source text) is delivered untranslated.
    """
    entry = {
        "page": int(page_number),
        "element": int(element_index),
        "stage": str(stage or "unknown"),
        "reason": str(reason or "unspecified element failure")[:500],
    }
    if source_text:
        entry["source_text"] = re.sub(r"\s+", " ", _plain_text(source_text)).strip()[:160]
    audit.setdefault("element_source_fallbacks", []).append(entry)
    return entry


def _exclude_pdf_audit_expectations_for_element(
    audit: dict,
    page_number: int,
    element_index: int,
) -> None:
    """Exclude one knowingly untranslated element from translation-only checks."""
    page_number = int(page_number)
    element_index = int(element_index)
    excluded = audit.setdefault("element_source_fallback_expectation_exclusions", {})
    for key in (
        "superscript_expectations",
        "vector_ocr_expectations",
    ):
        values = list(audit.get(key) or [])
        retained = [
            value
            for value in values
            if not (
                int(value.get("page", 0) or 0) == page_number
                and int(value.get("element", -1)) == element_index
            )
        ]
        removed = len(values) - len(retained)
        if removed:
            excluded[key] = int(excluded.get(key, 0) or 0) + removed
            audit[key] = retained


def _replace_pdf_pages_with_source(
    output_path: str,
    source_path: str,
    page_numbers: set[int] | list[int],
    *,
    source_password: str = "",
    task_id: str = "",
) -> list[int]:
    """Atomically replace selected one-based output pages with exact source pages."""
    pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not pages:
        return []

    output_doc = None
    source_doc = None
    tmp_path = None
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    original_mode = stat.S_IMODE(os.stat(output_path).st_mode)
    try:
        output_doc = fitz.open(output_path)
        source_doc = fitz.open(source_path)
        if output_doc.is_encrypted and not output_doc.authenticate(source_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法修复输出PDF")
        if source_doc.is_encrypted and not source_doc.authenticate(source_password):
            raise RuntimeError("PDF密码错误或未提供密码，无法读取源PDF回退页")
        if len(output_doc) != len(source_doc):
            raise RuntimeError(
                "source-page fallback refused because page counts differ: "
                f"output={len(output_doc)}, source={len(source_doc)}"
            )
        invalid = [page for page in pages if page > len(source_doc)]
        if invalid:
            raise RuntimeError(f"source-page fallback pages out of range: {invalid}")

        for page in pages:
            page_index = page - 1
            output_doc.delete_page(page_index)
            output_doc.insert_pdf(
                source_doc,
                from_page=page_index,
                to_page=page_index,
                start_at=page_index,
                links=1,
                annots=1,
                widgets=1,
                final=1,
            )

        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{Path(output_path).stem}_source_fallback_",
            suffix=".pdf",
            dir=directory,
        )
        os.close(fd)
        output_doc.save(tmp_path, deflate=True, garbage=4, clean=True)
        output_doc.close()
        output_doc = None
        source_doc.close()
        source_doc = None
        os.chmod(tmp_path, original_mode or 0o644)
        os.replace(tmp_path, output_path)
        tmp_path = None
        log.warning(
            f"[{task_id}] Restored exact source page(s) {pages} into the output PDF"
        )
        return pages
    finally:
        if output_doc is not None:
            output_doc.close()
        if source_doc is not None:
            source_doc.close()
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
