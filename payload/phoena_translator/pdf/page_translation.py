"""Concurrent page-translation stage for the PDF pipeline.

Only natural-language translation crosses the injected ``translate_text``
boundary.  Queue admission, deterministic labels, token protection, retries,
batch parsing, and integrity checks remain local Python operations.
"""

from __future__ import annotations

import re

from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.math_detection import (
    _normalize_pdf_translation,
    _pdf_inline_markup_preserved,
    _pdf_inline_math_fragments_preserved,
    _plain_text,
    _protect_pdf_identifier_fragments,
    _protect_pdf_inline_math_fragments,
    _protect_pdf_superscript_fragments,
    _restore_pdf_identifier_fragments,
    _restore_pdf_inline_math_fragments,
    _restore_pdf_superscript_fragments,
    _restore_pdf_superscript_markup,
)
from phoena_translator.pdf.semantics import _is_disclaimer_block
from phoena_translator.pdf.targets import (
    _dedupe_pdf_translation_targets,
    _looks_like_translatable_english,
    _pdf_element_requires_translation,
    _pdf_translation_language_views,
    _translate_pdf_deterministic_labels,
)
from phoena_translator.pdf.translation import (
    _build_pdf_batch_segment_envelope,
    _looks_like_pdf_toc_leader_block,
    _parse_pdf_batch_segment_envelope,
    _pdf_element_translation_needs_retry,
    _pdf_mixed_translation_has_untranslated_english_run,
    _pdf_vector_ocr_translation_hint,
    _split_pdf_reference_entry_translation_parts,
    _validate_pdf_page_translations,
)


def _clean_translated_text(
    source_text: str,
    candidate: str,
    superscript_records: list | tuple,
    inline_math_records: list | tuple,
    identifier_records: list | tuple,
) -> tuple[str, tuple[str, ...]]:
    """Restore protected tokens and normalize one provider response."""

    restoration_failures: list[str] = []
    if identifier_records:
        restored = _restore_pdf_identifier_fragments(candidate, identifier_records)
        if restored is not None:
            candidate = restored
        else:
            restoration_failures.append("identifier-token-restore")
    if inline_math_records:
        restored = _restore_pdf_inline_math_fragments(candidate, inline_math_records)
        if restored is not None:
            candidate = restored
        else:
            restoration_failures.append("inline-math-token-restore")
    if superscript_records:
        restored = _restore_pdf_superscript_fragments(candidate, superscript_records)
        if restored is not None:
            candidate = restored
        else:
            restoration_failures.append("superscript-token-restore")
    # Oversized candidates are rejected by the expansion gate in
    # ``_translation_part_needs_retry`` and retried; truncating here could cut
    # a just-restored URL or tag in half and manufacture new integrity
    # failures.
    candidate = re.sub(r"^#+\s*", "", candidate, flags=re.MULTILINE)
    candidate = re.sub(r"^\*+\s*", "", candidate, flags=re.MULTILINE)
    candidate = _normalize_pdf_translation(candidate)
    candidate = _translate_pdf_deterministic_labels(candidate)
    return (
        _restore_pdf_superscript_markup(source_text, candidate),
        tuple(restoration_failures),
    )


def _translation_integrity_failures(
    source_text: str,
    element: dict,
    candidate: str,
    restoration_failures: tuple[str, ...],
) -> tuple[str, ...]:
    """Return deterministic reasons why a restored translation is unsafe."""

    failures = list(restoration_failures)
    if not _pdf_inline_markup_preserved(source_text, candidate):
        failures.append("superscript-integrity")
    if not _pdf_inline_math_fragments_preserved(element, candidate):
        failures.append("inline-math-integrity")
    if not _pdf_translation_language_views(source_text, candidate)[2]:
        failures.append("identifier-integrity")
    return tuple(dict.fromkeys(failures))


def _translate_one(
    context: PDFPageTranslationContext,
    page_num: int,
    text: str,
    element: dict | None = None,
    extra_hint: str = "",
) -> str:
    """Translate one text unit with protected-token validation and one retry."""

    element = element or {}
    prompt = context.prompt + extra_hint if extra_hint else context.prompt
    protected_text, superscript_records = _protect_pdf_superscript_fragments(text)
    protected_text, inline_math_records = _protect_pdf_inline_math_fragments(
        protected_text,
        element.get("inline_math_fragments") or [],
    )
    protected_text, identifier_records = _protect_pdf_identifier_fragments(
        protected_text
    )
    if superscript_records:
        prompt += (
            "\n\n输入中的 PHOENA_SUP_0000_XXXXXXXXXXXXXXXX 形态标记"
            "是不可改写的上标占位符；必须逐字、逐个、按原顺序保留。"
        )
    if inline_math_records:
        prompt += (
            "\n\n输入中的 PHOENA_IMATH_0000_XXXXXXXXXXXXXXXX 形态标记"
            "是不可改写的行内公式占位符；必须逐字、逐个、按原顺序保留。"
        )
    if identifier_records:
        prompt += (
            "\n\n输入中的 PHOENA_ID_0000_XXXXXXXXXXXXXXXX 形态标记"
            "是不可改写的网址、邮箱或DOI占位符；必须逐字、逐个、按原顺序保留。"
        )

    translated, restoration_failures = _clean_translated_text(
        text,
        context.translate_text(protected_text, system_prompt=prompt),
        superscript_records,
        inline_math_records,
        identifier_records,
    )
    integrity_failures = _translation_integrity_failures(
        text,
        element,
        translated,
        restoration_failures,
    )
    if not integrity_failures:
        return translated

    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1}: protected-token integrity "
        f"failed ({', '.join(integrity_failures)}); retrying with a strict "
        "marker hint"
    )
    retry_prompt = (
        prompt + "\n\n输入中的每一个 PHOENA_SUP_、PHOENA_IMATH_ 或 PHOENA_ID_ 占位符"
        "都代表不可改写的源上标、行内公式或标识符；必须逐字、逐个、按原顺序"
        "保留，不得翻译或删除。"
    )
    retried, retry_restoration_failures = _clean_translated_text(
        text,
        context.translate_text(protected_text, system_prompt=retry_prompt),
        superscript_records,
        inline_math_records,
        identifier_records,
    )
    retry_integrity_failures = _translation_integrity_failures(
        text,
        element,
        retried,
        retry_restoration_failures,
    )
    if retry_integrity_failures:
        context.logger.error(
            f"[{context.task_id}] Page {page_num + 1}: protected-token integrity "
            f"still failed after retry ({', '.join(retry_integrity_failures)})"
        )
    return retried


def _strict_retry_hint(element: dict, text: str, translated: str) -> str:
    if element.get("reference_entry_hint"):
        return (
            "\n\n这是参考文献中必须翻译的作品标题和说明部分。"
            "必须输出包含中文的完整译文；期刊/出版社、数字、URL和DOI保持原样。"
            "不得翻译或补写作者，不得把英文标题原样返回。"
        )
    if element.get("vector_ocr"):
        return (
            _pdf_vector_ocr_translation_hint(text)
            + "\n\n上一版译文未通过完整性或术语校验。必须重新输出整个语义单元，"
            "并严格遵守上述段落结构和术语要求。"
        )
    if _looks_like_pdf_toc_leader_block(text):
        return (
            "\n\n这是必须翻译的英文目录块。必须逐项翻译目录标签为中文，"
            "并逐项保留原有点状引导符和页码；不得把任何英文目录项原样返回。"
        )
    if _pdf_mixed_translation_has_untranslated_english_run(text, translated):
        return (
            "\n\n程序检测到上一版译文仍连续保留了英文作品标题或说明文字。"
            "必须返回这整个输入的完整译文，并把其中每个英文句子、作品标题和"
            "说明文字翻译成中文；作者姓名、机构名、期刊/出版社名称、数字、"
            "URL和DOI保持原样。不得只翻译脚注或文献条目的外围说明。"
        )
    return (
        "\n\n这段输入已经由程序确认是必须翻译的英文标题或正文，"
        "不是引用文献、网址、专有名词列表、免责声明或法律声明。"
        "必须输出完整的中文译文；专有名词、数字和公式可以保留原文，"
        "但不得把整段英文原样返回，也不得遗漏句子。"
    )


def _translate_element(
    context: PDFPageTranslationContext,
    page_num: int,
    elements: list[dict],
    index: int,
    text: str,
) -> str:
    """Translate and, when needed, retry one semantic page element."""

    element = elements[index]
    reference_parts = (
        _split_pdf_reference_entry_translation_parts(text)
        if element.get("reference_entry_hint")
        else None
    )
    if reference_parts is not None:
        reference_prefix, reference_tail = reference_parts
        translated = (
            reference_prefix
            + _translate_one(
                context,
                page_num,
                reference_tail,
                element=element,
                extra_hint=(
                    "\n\n这是完整参考文献条目中作者和年份之后的部分。"
                    "程序已单独保护作者与年份；必须把作品标题及说明文字翻译成中文。"
                    "期刊/出版社名称、卷期页码、URL和DOI保持原样，不得把这部分英文整段原样返回。"
                ),
            ).lstrip()
        )
    elif element.get("vector_ocr"):
        translated = _translate_one(
            context,
            page_num,
            text,
            element=element,
            extra_hint=_pdf_vector_ocr_translation_hint(text),
        )
    else:
        translated = _translate_one(context, page_num, text, element=element)
    if not _pdf_element_translation_needs_retry(element, text, translated):
        return translated

    context.logger.warning(
        f"[{context.task_id}] Page {page_num + 1} elem {index}: individual "
        "translation still looks untranslated; retrying with an explicit "
        "prose/title instruction"
    )
    strict_hint = _strict_retry_hint(element, text, translated)
    if reference_parts is not None:
        reference_prefix, reference_tail = reference_parts
        translated = (
            reference_prefix
            + _translate_one(
                context,
                page_num,
                reference_tail,
                element=element,
                extra_hint=strict_hint,
            ).lstrip()
        )
    else:
        translated = _translate_one(
            context,
            page_num,
            text,
            element=element,
            extra_hint=strict_hint,
        )
    if not _pdf_element_translation_needs_retry(element, text, translated):
        return translated

    paragraphs = element.get("paragraphs") or []
    translatable_parts = [
        (paragraph.get("rich") or paragraph.get("plain") or "").strip()
        for paragraph in paragraphs
        if (paragraph.get("rich") or paragraph.get("plain") or "").strip()
    ]
    if len(paragraphs) > 1 and translatable_parts and not element.get("vector_ocr"):
        context.logger.warning(
            f"[{context.task_id}] Page {page_num + 1} elem {index}: whole block "
            "still looks untranslated, retrying paragraph-by-paragraph"
        )
        translated_parts = []
        for paragraph_text in translatable_parts:
            if not _looks_like_translatable_english(paragraph_text):
                translated_parts.append(paragraph_text)
                continue
            translated_parts.append(
                _translate_one(
                    context,
                    page_num,
                    paragraph_text,
                    element=element,
                    extra_hint=(
                        "\n\n如果输入是项目符号、编号列表或多段正文，"
                        "必须完整翻译成中文，并保留段落边界。"
                    ),
                )
            )
        paragraph_translation = "\n\n".join(
            part for part in translated_parts if part.strip()
        ).strip()
        if paragraph_translation:
            return paragraph_translation
    return translated


def _collect_translation_targets(
    context: PDFPageTranslationContext,
    page_num: int,
    elements: list[dict],
    translations: dict[int, str],
) -> list[tuple[int, str]]:
    """Apply deterministic skip/label rules and return provider-bound targets."""

    targets: list[tuple[int, str]] = []
    for index, element in enumerate(elements):
        if element["type"] != "text" or index in translations:
            continue
        text = element["content"]
        if _is_disclaimer_block(text):
            context.logger.info(
                f"[{context.task_id}] Page {page_num + 1} elem {index}: "
                f"disclaimer/citation, keeping original "
                f"(y={element.get('y', 0):.0f}, {len(text)} chars)"
            )
            translations[index] = _translate_pdf_deterministic_labels(
                element.get("rich_content") or text
            )
            continue
        skip_reason = element.get("skip_translate_reason")
        if skip_reason == "watermark":
            context.logger.info(
                f"[{context.task_id}] Page {page_num + 1} elem {index}: "
                "watermark detected, removing from output"
            )
            translations[index] = ""
            continue
        if skip_reason:
            context.logger.info(
                f"[{context.task_id}] Page {page_num + 1} elem {index}: "
                f"{skip_reason} detected, keeping original"
            )
            translations[index] = text
            continue
        plain = _plain_text(text).strip()
        if not plain:
            translations[index] = text
            continue
        if not _looks_like_translatable_english(text):
            translations[index] = _translate_pdf_deterministic_labels(
                element.get("rich_content") or text
            )
            continue
        if not _pdf_element_requires_translation(element):
            rich_source = element.get("rich_content") or text
            deterministic_label = _translate_pdf_deterministic_labels(rich_source)
            translations[index] = (
                deterministic_label
                if _plain_text(deterministic_label) != _plain_text(text)
                else text
            )
            continue
        send_text = element.get("rich_content") or text
        targets.append((index, send_text))
        context.logger.info(
            f"[{context.task_id}] Page {page_num + 1} elem {index}: queued for "
            f"translation (y={element.get('y', 0):.0f}, {len(text)} chars)"
        )
    return targets


def _restore_batch_part(
    context: PDFPageTranslationContext,
    page_num: int,
    element: dict,
    index: int,
    source_text: str,
    translated: str,
    protection: tuple,
) -> str | None:
    """Restore one batch segment, returning ``None`` when individual retry is safer."""

    superscript_records, inline_math_records, identifier_records = protection
    for records, restore, label in (
        (identifier_records, _restore_pdf_identifier_fragments, "identifier"),
        (inline_math_records, _restore_pdf_inline_math_fragments, "inline-math"),
        (superscript_records, _restore_pdf_superscript_fragments, "superscript"),
    ):
        restored = restore(translated, records)
        if restored is not None:
            translated = restored
        elif records:
            context.logger.warning(
                f"[{context.task_id}] Page {page_num + 1} elem {index}: batch "
                f"{label} marker changed; retrying individually"
            )
            return None
    translated = re.sub(r"^#+\s*", "", translated, flags=re.MULTILINE)
    translated = re.sub(r"^\*+\s*", "", translated, flags=re.MULTILINE)
    translated = _normalize_pdf_translation(translated)
    translated = _translate_pdf_deterministic_labels(translated)
    translated = _restore_pdf_superscript_markup(source_text, translated)
    if (
        _pdf_element_translation_needs_retry(element, source_text, translated)
        or not _pdf_inline_markup_preserved(source_text, translated)
        or not _pdf_inline_math_fragments_preserved(element, translated)
    ):
        return None
    return translated


def _translate_batch(
    context: PDFPageTranslationContext,
    page_num: int,
    elements: list[dict],
    targets: list[tuple[int, str]],
) -> dict[int, str]:
    """Translate multiple unique elements in one provider request."""

    protected_targets = []
    protections = {}
    for index, source_text in targets:
        protected_source, superscript_records = _protect_pdf_superscript_fragments(
            source_text
        )
        protected_source, inline_math_records = _protect_pdf_inline_math_fragments(
            protected_source,
            elements[index].get("inline_math_fragments") or [],
        )
        protected_source, identifier_records = _protect_pdf_identifier_fragments(
            protected_source
        )
        protected_targets.append((index, protected_source))
        protections[index] = (
            superscript_records,
            inline_math_records,
            identifier_records,
        )
    batch_text, segment_records = _build_pdf_batch_segment_envelope(protected_targets)
    batch_prompt = (
        context.prompt + "\n\n特别注意：输入文本由 ≡≡≡SPLIT≡≡≡ 分隔为多个独立段落。"
        "请分别翻译每个段落，翻译结果之间也用 ≡≡≡SPLIT≡≡≡ 分隔。"
        "段落数量必须与输入完全一致，不要合并或拆分段落。"
        "每段开头的 PHOENA_SEG_0000_XXXXXXXXXXXXXXXX 是该段不可改写的编号；"
        "输出时必须在对应译文前逐字保留同一编号，编号数量和顺序不得改变。"
        "PHOENA_SUP_、PHOENA_IMATH_ 和 PHOENA_ID_ 形态的标记分别是不可改写的"
        "上标、行内公式与网址/邮箱/DOI占位符，也必须逐字、逐个、按原顺序保留。"
        "若某段是完整参考文献条目，作者与年份前缀必须逐字保留；"
        "只翻译其后的作品标题和说明，期刊/出版社、卷期页码、URL和DOI保持原样。"
    )
    if any(elements[index].get("vector_ocr") for index, _ in targets):
        batch_prompt += (
            "\n\n其中有些编号段是从一个完整图框恢复出的语义单元。"
            "每个编号段必须作为整体一次翻译，不得按视觉行拆分；"
            "编号段内部的空行分段数量和项目符号必须保持不变。"
        )
    result = context.translate_text(batch_text, system_prompt=batch_prompt)
    parts = _parse_pdf_batch_segment_envelope(result, segment_records)
    if parts is None:
        context.logger.warning(
            f"[{context.task_id}] Page {page_num + 1}: batch segment ids did not "
            "round-trip exactly; falling back to individual translation"
        )
        return {
            index: _translate_element(context, page_num, elements, index, text)
            for index, text in targets
        }

    translations = {}
    for (index, source_text), translated in zip(targets, parts):
        restored = _restore_batch_part(
            context,
            page_num,
            elements[index],
            index,
            source_text,
            translated,
            protections[index],
        )
        translations[index] = (
            restored
            if restored is not None
            else _translate_element(context, page_num, elements, index, source_text)
        )
    context.logger.info(
        f"[{context.task_id}] Page {page_num + 1}: batch translated "
        f"{len(targets)} unique elements in 1 API call"
    )
    return translations


def translate_page(
    context: PDFPageTranslationContext,
    page_num: int,
    page_info: dict,
) -> tuple[int, dict[int, str]]:
    """Translate and validate one page; safe for ``ThreadPoolExecutor`` use."""

    elements = page_info["elements"]
    translations = {
        int(key): value for key, value in (page_info.get("cached_seed") or {}).items()
    }
    targets = _collect_translation_targets(
        context,
        page_num,
        elements,
        translations,
    )
    if not targets:
        _validate_pdf_page_translations(page_num + 1, elements, translations)
        return page_num, translations

    unique_targets, duplicate_indices = _dedupe_pdf_translation_targets(targets)
    if len(unique_targets) < len(targets):
        context.logger.info(
            f"[{context.task_id}] Page {page_num + 1}: collapsed {len(targets)} "
            f"translation slots to {len(unique_targets)} unique API input(s)"
        )
    if len(unique_targets) == 1:
        index, text = unique_targets[0]
        translations[index] = _translate_element(
            context,
            page_num,
            elements,
            index,
            text,
        )
    else:
        translations.update(
            _translate_batch(context, page_num, elements, unique_targets)
        )
    for primary_index, copies in duplicate_indices.items():
        for index in copies:
            translations[index] = translations[primary_index]
    _validate_pdf_page_translations(page_num + 1, elements, translations)
    return page_num, translations


__all__ = ["translate_page"]
