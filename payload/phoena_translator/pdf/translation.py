"""Translation for deterministic PDF processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable
import re
import time
from phoena_translator.pdf.text_shape import _plain_text
from phoena_translator.pdf.translation_targets import (
    _pdf_translation_language_views,
)
from phoena_translator.pdf.batch_codec import (
    PDF_BATCH_SEPARATOR,
    _pdf_batch_segment_records_from_source,
)
from phoena_translator.pdf.toc import _pdf_toc_entries
from phoena_translator.pdf.translation_validation import (
    _looks_like_pdf_toc_leader_block,
    _pdf_batch_translation_candidate_is_valid,
    _pdf_toc_translation_needs_retry,
    _short_translation_needs_retry,
)


@dataclass(frozen=True)
class PDFTranslationDependencies:
    """Only the semantic translation gateway and its local response helpers."""

    create_chat_completion: Callable[..., Any]
    strip_think_tags: Callable[[str], str]
    system_prompt_text: str
    logger: logging.Logger

def translate_text(
    text: str,
    system_prompt: str | None = None,
    retries: int = 4,
    *,
    dependencies: PDFTranslationDependencies,
) -> str:
    log = dependencies.logger
    if not text or not text.strip():
        return text
    if system_prompt is None:
        system_prompt = dependencies.system_prompt_text

    # Backoff applies only to transport-level exceptions (the provider's rate
    # limiting is owned by ``api_runtime``).  Semantic verification failures
    # retry immediately: the same stateless request gains nothing from
    # waiting, and the system prompt is already hardened on attempt > 0.
    # Measured 2026-08-31, these sleeps alone added 60-120s per doomed
    # element on deal-table pages.
    backoff = [10, 20, 30, 45, 60]
    last_err = None
    best_result = None      # Track best result across all attempts
    best_cn_chars = -1      # Chinese char count of best result
    for attempt in range(retries):
        try:
            prompt_for_attempt = system_prompt
            if attempt > 0:
                prompt_for_attempt += (
                    "\n\n上一次输出未充分翻译。若输入是英文正文、标题或图注，"
                    "这次必须输出中文译文，不得直接重复英文原文。"
                )
            resp = dependencies.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt_for_attempt},
                    {"role": "user", "content": text},
                ],
                max_tokens=65536,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            result = dependencies.strip_think_tags(content)
            # Strip markdown code fences if present
            if result.startswith("```"):
                result = re.sub(r'^```[a-zA-Z]*\n?', '', result)
                result = re.sub(r'\n?```\s*$', '', result)
            # Verify translation quality
            text_only = _plain_text(result)
            input_text = _plain_text(text)
            input_language, result_language, identifiers_preserved = (
                _pdf_translation_language_views(input_text, text_only)
            )
            cn_chars = len(re.findall(r'[\u4e00-\u9fff]', result_language))
            total_chars = len(result_language.strip())
            input_len = len(input_language.strip())
            cn_ratio = cn_chars / max(total_chars, 1)
            len_ratio = total_chars / max(input_len, 1)

            # Always track the best result so far
            if cn_chars > best_cn_chars:
                best_cn_chars = cn_chars
                best_result = result

            if _pdf_batch_translation_candidate_is_valid(text, result):
                log.info(
                    "Translation verified at PDF batch element boundaries: "
                    f"{len(text.split(PDF_BATCH_SEPARATOR))} parts"
                )
                return result
            if _pdf_batch_segment_records_from_source(text):
                # The page-level caller owns the safe fallback.  Return the
                # first response immediately so it can retry only the affected
                # elements individually instead of spending several delayed
                # attempts on an unsafe positional batch.
                log.warning(
                    "PDF batch segment envelope did not validate; handing the "
                    "response to the element-level fallback"
                )
                return result

            if not identifiers_preserved:
                log.warning(
                    f"Translation attempt {attempt+1}: protected URL/e-mail/DOI "
                    "identifier mismatch, retrying"
                )
                last_err = RuntimeError(
                    "Protected URL/e-mail/DOI identifier mismatch"
                )
                continue

            if PDF_BATCH_SEPARATOR not in text and _looks_like_pdf_toc_leader_block(text):
                if _pdf_toc_translation_needs_retry(text, result):
                    log.warning(
                        f"Translation attempt {attempt+1}: dotted TOC labels or page references are incomplete, retrying"
                    )
                    last_err = RuntimeError("Dotted TOC translation is incomplete")
                    continue
                log.info(
                    f"Translation verified as dotted TOC ({len(_pdf_toc_entries(text))} entries)"
                )
                return result

            # Skip strict verification for very small inputs (headers, nav, etc.)
            if input_len < 200:
                if _short_translation_needs_retry(text, result):
                    log.warning(
                        f"Translation attempt {attempt+1}: short input appears untranslated, retrying"
                    )
                    last_err = RuntimeError("Short input appears untranslated")
                    continue
                log.info(f"Translation verified (small input, {input_len} chars): {cn_chars} CN chars")
                return result

            if cn_chars < 20:
                log.warning(f"Translation attempt {attempt+1}: only {cn_chars} Chinese chars in {total_chars} total chars, retrying")
                last_err = RuntimeError(f"Translation has only {cn_chars} Chinese chars")
                continue
            if cn_ratio < 0.1:
                log.warning(f"Translation attempt {attempt+1}: Chinese ratio too low ({cn_ratio:.1%}), retrying")
                last_err = RuntimeError(f"Chinese ratio {cn_ratio:.1%} too low")
                continue
            if len_ratio < 0.15:
                log.warning(f"Translation attempt {attempt+1}: output too short ({total_chars} vs input {input_len}, ratio {len_ratio:.1%}), retrying")
                last_err = RuntimeError(f"Output length ratio {len_ratio:.1%} too low")
                continue
            log.info(f"Translation verified: {cn_chars} CN chars, ratio={cn_ratio:.0%}, len_ratio={len_ratio:.0%}")
            return result
        except Exception as exc:
            last_err = exc
            log.warning(f"Translation attempt {attempt+1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(backoff[attempt])
    # The PDF caller performs a stricter element-level gate before caching this
    # candidate. Non-PDF callers retain their existing best-effort behavior.
    if best_result is not None:
        log.warning(
            f"All {retries} retries failed aggregate verification; returning best candidate "
            f"({best_cn_chars} CN chars) for caller-level validation"
        )
        return best_result
    raise RuntimeError(f"Translation failed after {retries} retries: {last_err}")


