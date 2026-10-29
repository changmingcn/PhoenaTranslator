"""The batch envelope: several page paragraphs sent in one request, each
bound to an opaque content-hashed segment id that must round-trip."""

from __future__ import annotations

import hashlib
from phoena_translator.pdf.placeholders import PDF_BATCH_SEGMENT_RE


PDF_BATCH_SEPARATOR = "≡≡≡SPLIT≡≡≡"


def _build_pdf_batch_segment_envelope(
    targets: list[tuple[int, str]],
) -> tuple[str, list[dict]]:
    """Bind every batch paragraph to an opaque, content-hashed segment id."""
    records = []
    parts = []
    for ordinal, (element_index, source) in enumerate(targets):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest().upper()
        token = f"PHOENA_SEG_{ordinal:04d}_{digest[:16]}"
        records.append({
            "token": token,
            "element_index": int(element_index),
            "source": source,
            "sha256": digest.lower(),
        })
        parts.append(f"{token}\n{source}")
    separator = f"\n{PDF_BATCH_SEPARATOR}\n"
    return separator.join(parts), records


def _pdf_batch_segment_records_from_source(source_text: str) -> list[dict] | None:
    parts = [part.strip() for part in source_text.split(PDF_BATCH_SEPARATOR)]
    records = []
    for ordinal, part in enumerate(parts):
        match = PDF_BATCH_SEGMENT_RE.match(part)
        if not match or match.start() != 0:
            return None
        token = match.group(0)
        expected_prefix = f"PHOENA_SEG_{ordinal:04d}_"
        if not token.startswith(expected_prefix):
            return None
        source = part[match.end():].lstrip(" \t\r\n")
        if not source:
            return None
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if token.rsplit("_", 1)[-1] != digest[:16].upper():
            return None
        records.append({
            "token": token,
            "element_index": ordinal,
            "source": source,
            "sha256": digest,
        })
    return records or None


def _parse_pdf_batch_segment_envelope(
    translated_text: str,
    records: list[dict],
) -> list[str] | None:
    """Return translations only when every exact id remains in order."""
    expected_tokens = [record["token"] for record in records]
    observed_tokens = PDF_BATCH_SEGMENT_RE.findall(translated_text or "")
    if observed_tokens != expected_tokens:
        return None
    parts = [part.strip() for part in (translated_text or "").split(PDF_BATCH_SEPARATOR)]
    if len(parts) != len(records):
        return None
    translated_parts = []
    for part, record in zip(parts, records):
        token = record["token"]
        if not part.startswith(token):
            return None
        if hashlib.sha256(record["source"].encode("utf-8")).hexdigest() != record["sha256"]:
            return None
        candidate = part[len(token):]
        if candidate and not candidate[:1].isspace():
            return None
        candidate = candidate.strip()
        if not candidate or PDF_BATCH_SEGMENT_RE.search(candidate):
            return None
        translated_parts.append(candidate)
    return translated_parts
