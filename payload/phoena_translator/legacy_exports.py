"""Legacy public symbol exports kept separate from application runtime."""

from __future__ import annotations

from phoena_translator.epub.archive import ArchiveLimits  # noqa: F401 - legacy facade
from phoena_translator.epub.math_protection import (
    XHTML_MATH_PLACEHOLDER_RE,  # noqa: F401 - legacy facade
    find_xhtml_math_intervals as _find_xhtml_math_intervals,  # noqa: F401 - legacy facade
    looks_like_latex_dollar_content as _looks_like_latex_dollar_content,  # noqa: F401 - legacy facade
    protect_xhtml_math_fragments as _protect_xhtml_math_fragments,  # noqa: F401 - legacy facade
    restore_xhtml_math_fragments as _restore_xhtml_math_fragments,  # noqa: F401 - legacy facade
)
from phoena_translator.epub.pipeline import (
    EPUBPipeline,  # noqa: F401 - legacy facade
    EPUBPipelineDependencies,  # noqa: F401 - legacy facade
    translate_single_chunk as _translate_epub_single_chunk,  # noqa: F401 - legacy facade
)
from phoena_translator.epub.xhtml import (
    DEFAULT_CHUNK_MAX_BYTES as CHUNK_MAX_BYTES,  # noqa: F401 - legacy facade
    SKIP_SECTION_PATTERNS,  # noqa: F401 - legacy facade
    fix_xhtml_entities as _fix_xhtml_entities,  # noqa: F401 - legacy facade
    fix_xhtml_tags as _fix_xhtml_tags,  # noqa: F401 - legacy facade
    is_appendix_xhtml as _is_appendix_xhtml,  # noqa: F401 - legacy facade
    merge_translated_chunks as _merge_translated_chunks,  # noqa: F401 - legacy facade
    split_raw as _split_raw,  # noqa: F401 - legacy facade
    split_xhtml_to_chunks as _split_xhtml_to_chunks,  # noqa: F401 - legacy facade
    visible_text_ranges as _xhtml_visible_text_ranges,  # noqa: F401 - legacy facade
)
from phoena_translator.math_text import (
    MATH_FUNCTION_WORDS,  # noqa: F401 - legacy facade
    MATH_LETTERLIKE_SYMBOLS,  # noqa: F401 - legacy facade
    MATH_SYMBOLS,  # noqa: F401 - legacy facade
    SHORT_PROSE_WORDS,  # noqa: F401 - legacy facade
    is_math_block as _is_math_block,  # noqa: F401 - legacy facade
    unicode_math_symbol_count as _unicode_math_symbol_count,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.types import (
    DISCLAIMER_PATTERNS,  # noqa: F401 - legacy facade
    PDFPageExtractionError,  # noqa: F401 - legacy facade
    PDFPageRenderError,  # noqa: F401 - legacy facade
    PDFPageTranslationError,  # noqa: F401 - legacy facade
    PDFStructureValidationError,  # noqa: F401 - legacy facade
    PDFTranslationIntegrityError,  # noqa: F401 - legacy facade
    PDF_BATCH_SEGMENT_RE,  # noqa: F401 - legacy facade
    PDF_FIXED_SHORT_LABEL_TRANSLATIONS,  # noqa: F401 - legacy facade
    PDF_FORMULA_GUARD_PADDING,  # noqa: F401 - legacy facade
    PDF_IDENTIFIER_PLACEHOLDER_RE,  # noqa: F401 - legacy facade
    PDF_INLINE_MATH_PLACEHOLDER_RE,  # noqa: F401 - legacy facade
    PDF_MATH_ITALIC_FONT_TOKENS,  # noqa: F401 - legacy facade
    PDF_SIGNATURE_GEOMETRY_QUANTUM,  # noqa: F401 - legacy facade
    PDF_STRONG_MATH_FONT_TOKENS,  # noqa: F401 - legacy facade
    PDF_SUPERSCRIPT_PLACEHOLDER_RE,  # noqa: F401 - legacy facade
    PDF_TRANSLATABLE_CITATION_LABELS,  # noqa: F401 - legacy facade
    PDF_VECTOR_OCR_DPI,  # noqa: F401 - legacy facade
    PDF_VECTOR_OCR_MIN_ALPHA_WORDS,  # noqa: F401 - legacy facade
    SKIP_SECTION_HEADING_RE,  # noqa: F401 - legacy facade
    WATERMARK_PATTERNS,  # noqa: F401 - legacy facade
    _PDF_BARE_FOOTNOTE_LEAD_RE,  # noqa: F401 - legacy facade
    _PDF_BULLETED_PARAGRAPH_LEAD_RE,  # noqa: F401 - legacy facade
    _PDF_CITATION_PUBLISHER_CONNECTORS,  # noqa: F401 - legacy facade
    _PDF_DISPLAY_IDENTITY_END_WORDS,  # noqa: F401 - legacy facade
    _PDF_FOOTNOTE_MARKER_RE,  # noqa: F401 - legacy facade
    _PDF_LEADING_SUPERSCRIPT_FOOTNOTE_MARKER_RE,  # noqa: F401 - legacy facade
    _PDF_NUMBERED_PARAGRAPH_LEAD_RE,  # noqa: F401 - legacy facade
    _PDF_PARAGRAPH_MARKER_ATOM,  # noqa: F401 - legacy facade
    _PDF_PERSON_HONORIFICS,  # noqa: F401 - legacy facade
    _PDF_PROPER_NAME_CONNECTORS,  # noqa: F401 - legacy facade
    _PDF_PROPER_NAME_SUFFIXES,  # noqa: F401 - legacy facade
    _PDF_SEMANTIC_TERMINAL_RE,  # noqa: F401 - legacy facade
    _PDF_SHORT_LABEL_CONNECTORS,  # noqa: F401 - legacy facade
    _PDF_TRANSLATABLE_DISCLAIMER_LABELS,  # noqa: F401 - legacy facade
    _PDF_TRANSLATABLE_REPORT_STRUCTURE_RE,  # noqa: F401 - legacy facade
    _PDF_TRANSLATABLE_SHORT_LABELS,  # noqa: F401 - legacy facade
    _PDF_WRAPPED_MATH_DANGLING_TAIL_RE,  # noqa: F401 - legacy facade
    _PDF_WRAPPED_MATH_HYPHEN_TAIL_RE,  # noqa: F401 - legacy facade
    _PDF_WRAPPED_MATH_PROSE_WORD_RE,  # noqa: F401 - legacy facade
    _PDF_WRAPPED_MATH_SCRIPT_LEAD_RE,  # noqa: F401 - legacy facade
    _PRESERVED_IDENTIFIER_RE,  # noqa: F401 - legacy facade
    _PRESERVED_STRUCTURED_CODE_RE,  # noqa: F401 - legacy facade
    _PRESERVED_URI_BODY,  # noqa: F401 - legacy facade
    _PRESERVED_URI_LAYOUT_WRAPS,  # noqa: F401 - legacy facade
    _SHORT_CITATION_CONNECTORS,  # noqa: F401 - legacy facade
    _SHORT_CITATION_WORD_RE,  # noqa: F401 - legacy facade
    _SHORT_CITATION_YEAR_RE,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.geometry import (
    _cluster_pdf_line_spans,  # noqa: F401 - legacy facade
    _detect_pdf_text_align,  # noqa: F401 - legacy facade
    _get_pdf_elem_rect,  # noqa: F401 - legacy facade
    _get_pdf_primary_paragraph_rect,  # noqa: F401 - legacy facade
    _get_pdf_render_bbox,  # noqa: F401 - legacy facade
    _get_pdf_source_ink_rect,  # noqa: F401 - legacy facade
    _line_overlaps_pdf_table_rects,  # noqa: F401 - legacy facade
    _pdf_elem_last_source_line_rect,  # noqa: F401 - legacy facade
    _pdf_rect_intersects_protected,  # noqa: F401 - legacy facade
    _pdf_rotation_from_direction,  # noqa: F401 - legacy facade
    _split_pdf_line_layout_cells,  # noqa: F401 - legacy facade
    _subtract_pdf_protected_rects,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.math_detection import (
    _demote_pdf_wrapped_identifier_math_lines,  # noqa: F401 - legacy facade
    _filter_pdf_formula_safe_rects,  # noqa: F401 - legacy facade
    _find_pdf_fragment_outside_control_tokens,  # noqa: F401 - legacy facade
    _looks_like_compact_pdf_identifier,  # noqa: F401 - legacy facade
    _looks_like_pdf_identifier_line,  # noqa: F401 - legacy facade
    _looks_like_pdf_list_marker,  # noqa: F401 - legacy facade
    _looks_like_pdf_math_italic_run,  # noqa: F401 - legacy facade
    _looks_like_pdf_wrapped_identifier_query_tail,  # noqa: F401 - legacy facade
    _make_pdf_formula_element_from_lines,  # noqa: F401 - legacy facade
    _mark_pdf_superscript_spans,  # noqa: F401 - legacy facade
    _normalize_pdf_font_name,  # noqa: F401 - legacy facade
    _normalize_pdf_translation,  # noqa: F401 - legacy facade
    _pdf_element_leading_superscript_footnote_marker,  # noqa: F401 - legacy facade
    _pdf_exact_source_native_superscript_matches,  # noqa: F401 - legacy facade
    _pdf_formula_region_signature,  # noqa: F401 - legacy facade
    _pdf_inline_markup_preserved,  # noqa: F401 - legacy facade
    _pdf_inline_math_fragments_preserved,  # noqa: F401 - legacy facade
    _pdf_inline_math_span_fragments,  # noqa: F401 - legacy facade
    _pdf_is_semantic_superscript_marker,  # noqa: F401 - legacy facade
    _pdf_line_math_evidence,  # noqa: F401 - legacy facade
    _pdf_marker_pattern,  # noqa: F401 - legacy facade
    _pdf_math_font_kind,  # noqa: F401 - legacy facade
    _pdf_native_superscript_fingerprints,  # noqa: F401 - legacy facade
    _pdf_nearby_superscript_reference,  # noqa: F401 - legacy facade
    _pdf_signature_number,  # noqa: F401 - legacy facade
    _pdf_source_line_leading_superscript_footnote_marker,  # noqa: F401 - legacy facade
    _pdf_span_has_lexical_text,  # noqa: F401 - legacy facade
    _pdf_span_origin_y,  # noqa: F401 - legacy facade
    _pdf_superscript_signature,  # noqa: F401 - legacy facade
    _plain_text,  # noqa: F401 - legacy facade
    _propagate_pdf_math_line_context,  # noqa: F401 - legacy facade
    _protect_pdf_identifier_fragments,  # noqa: F401 - legacy facade
    _protect_pdf_inline_math_fragments,  # noqa: F401 - legacy facade
    _protect_pdf_superscript_fragments,  # noqa: F401 - legacy facade
    _restore_pdf_identifier_fragments,  # noqa: F401 - legacy facade
    _restore_pdf_inline_math_fragments,  # noqa: F401 - legacy facade
    _restore_pdf_superscript_fragments,  # noqa: F401 - legacy facade
    _restore_pdf_superscript_markup,  # noqa: F401 - legacy facade
    _select_pdf_inline_math_fragments_for_text,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.extraction import (
    _extract_pdf_vector_ocr_elements,  # noqa: F401 - legacy facade
    _find_pdf_table_rects,  # noqa: F401 - legacy facade
    _find_pdf_table_regions,  # noqa: F401 - legacy facade
    _has_compact_open_pdf_table_rules,  # noqa: F401 - legacy facade
    _is_compact_open_pdf_table,  # noqa: F401 - legacy facade
    _is_skip_page,  # noqa: F401 - legacy facade
    _join_pdf_vector_ocr_lines,  # noqa: F401 - legacy facade
    _looks_like_pdf_body_text_false_table,  # noqa: F401 - legacy facade
    _looks_like_pdf_hidden_text_artifact_span,  # noqa: F401 - legacy facade
    _make_pdf_vector_ocr_element,  # noqa: F401 - legacy facade
    _mark_pdf_embedded_thumbnail_text_elements,  # noqa: F401 - legacy facade
    _normalize_pdf_vector_ocr_text,  # noqa: F401 - legacy facade
    _pdf_table_cell_rects,  # noqa: F401 - legacy facade
    _pdf_table_list,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_candidate,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_detect_regions,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_dominant_fill,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_existing_line_match,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_line_is_artifact,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_normalized_match_text,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_outside_paragraphs,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_parse_tsv,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_semantic_paragraphs,  # noqa: F401 - legacy facade
    _sanitize_pdf_hidden_text_artifact_spans,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.semantics import (
    _bucket_pdf_fontsize,  # noqa: F401 - legacy facade
    _build_pdf_paragraphs,  # noqa: F401 - legacy facade
    _carry_fragment_is_prose,  # noqa: F401 - legacy facade
    _color_int_to_rgb,  # noqa: F401 - legacy facade
    _complete_pdf_citation_separator_inline_math_fragments,  # noqa: F401 - legacy facade
    _cross_page_dominant_fontsize,  # noqa: F401 - legacy facade
    _cross_page_element_rejection,  # noqa: F401 - legacy facade
    _cross_page_geometry_rejection,  # noqa: F401 - legacy facade
    _derive_pdf_page_layout_styles,  # noqa: F401 - legacy facade
    _detect_pdf_drop_cap_span,  # noqa: F401 - legacy facade
    _ends_with_sentence_boundary,  # noqa: F401 - legacy facade
    _first_cross_page_lexical_char,  # noqa: F401 - legacy facade
    _is_cross_page_body_elem,  # noqa: F401 - legacy facade
    _is_cross_page_heading_elem,  # noqa: F401 - legacy facade
    _is_disclaimer_block,  # noqa: F401 - legacy facade
    _is_heading_like_elem,  # noqa: F401 - legacy facade
    _is_new_pdf_paragraph,  # noqa: F401 - legacy facade
    _is_pdf_first_line_indent_continuation,  # noqa: F401 - legacy facade
    _is_pdf_hanging_list_continuation,  # noqa: F401 - legacy facade
    _is_pdf_single_visual_line_element,  # noqa: F401 - legacy facade
    _is_pdf_translatable_disclaimer_label,  # noqa: F401 - legacy facade
    _join_pdf_line_fragments,  # noqa: F401 - legacy facade
    _looks_light_gray,  # noqa: F401 - legacy facade
    _looks_like_heading_text,  # noqa: F401 - legacy facade
    _looks_like_pdf_citation_publisher,  # noqa: F401 - legacy facade
    _looks_like_pdf_drop_cap_token,  # noqa: F401 - legacy facade
    _looks_like_pdf_paragraph_lead,  # noqa: F401 - legacy facade
    _looks_like_pdf_reference_entry_lead,  # noqa: F401 - legacy facade
    _looks_like_right_aligned_signoff,  # noqa: F401 - legacy facade
    _looks_like_split_layout_line,  # noqa: F401 - legacy facade
    _make_pdf_glossary_column_fragment,  # noqa: F401 - legacy facade
    _make_pdf_reference_fragment_from_source_lines,  # noqa: F401 - legacy facade
    _make_pdf_space_span,  # noqa: F401 - legacy facade
    _make_pdf_text_element_from_lines,  # noqa: F401 - legacy facade
    _mark_pdf_reference_entry_elements,  # noqa: F401 - legacy facade
    _mark_watermark_elements,  # noqa: F401 - legacy facade
    _merge_adjacent_heading_elements,  # noqa: F401 - legacy facade
    _merge_cross_page_sentences,  # noqa: F401 - legacy facade
    _merge_pdf_detached_list_marker_elements,  # noqa: F401 - legacy facade
    _merge_pdf_glossary_cell_group,  # noqa: F401 - legacy facade
    _merge_pdf_list_marker_clusters,  # noqa: F401 - legacy facade
    _merge_pdf_reference_entry_elements,  # noqa: F401 - legacy facade
    _merge_pdf_semantic_continuation_elements,  # noqa: F401 - legacy facade
    _merge_pdf_semantic_table_cells,  # noqa: F401 - legacy facade
    _merge_pdf_strong_continuation_elements,  # noqa: F401 - legacy facade
    _merge_pdf_table_cell_group,  # noqa: F401 - legacy facade
    _merge_pdf_wrapped_inline_math_continuation_fragments,  # noqa: F401 - legacy facade
    _merge_pdf_wrapped_line_elements,  # noqa: F401 - legacy facade
    _merge_pdf_wrapped_math_continuation_pair,  # noqa: F401 - legacy facade
    _normalize_pdf_drop_cap_lines,  # noqa: F401 - legacy facade
    _normalize_pdf_footnote_alignment,  # noqa: F401 - legacy facade
    _normalize_pdf_glossary_cells,  # noqa: F401 - legacy facade
    _normalize_watermark_text,  # noqa: F401 - legacy facade
    _parse_pdf_toc_leader,  # noqa: F401 - legacy facade
    _pdf_citation_separator_pipe_count,  # noqa: F401 - legacy facade
    _pdf_compact_open_table_cell_groups,  # noqa: F401 - legacy facade
    _pdf_detached_list_marker_target_score,  # noqa: F401 - legacy facade
    _pdf_element_footnote_definition_marker,  # noqa: F401 - legacy facade
    _pdf_element_prefers_single_line_heading,  # noqa: F401 - legacy facade
    _pdf_internal_paragraph_lead_residuals,  # noqa: F401 - legacy facade
    _pdf_reference_continuation_residual_pairs,  # noqa: F401 - legacy facade
    _pdf_reference_entry_elements_can_merge,  # noqa: F401 - legacy facade
    _pdf_reference_entry_lead_counts,  # noqa: F401 - legacy facade
    _pdf_reference_entry_structure_residuals,  # noqa: F401 - legacy facade
    _pdf_semantic_continuation_elements_can_merge,  # noqa: F401 - legacy facade
    _pdf_source_line_footnote_definition_marker,  # noqa: F401 - legacy facade
    _pdf_strong_continuation_element_is_eligible,  # noqa: F401 - legacy facade
    _pdf_strong_continuation_elements_can_merge,  # noqa: F401 - legacy facade
    _pdf_strong_continuation_residual_pairs,  # noqa: F401 - legacy facade
    _pdf_table_visual_line_bands,  # noqa: F401 - legacy facade
    _pdf_text_colors_semantically_compatible,  # noqa: F401 - legacy facade
    _pdf_visual_line_info_from_element,  # noqa: F401 - legacy facade
    _pdf_wrapped_line_elements_can_merge,  # noqa: F401 - legacy facade
    _pdf_wrapped_line_group_can_start,  # noqa: F401 - legacy facade
    _pdf_wrapped_math_continuation_pair_can_merge,  # noqa: F401 - legacy facade
    _pdf_wrapped_math_fragment_prose_words,  # noqa: F401 - legacy facade
    _pick_pdf_dominant_value,  # noqa: F401 - legacy facade
    _prepare_pdf_glossary_column_geometry,  # noqa: F401 - legacy facade
    _promote_pdf_formula_overlaps,  # noqa: F401 - legacy facade
    _rebase_pdf_split_paragraph_layout,  # noqa: F401 - legacy facade
    _should_keep_pdf_lines_separate,  # noqa: F401 - legacy facade
    _split_pdf_disjoint_line_segments,  # noqa: F401 - legacy facade
    _split_pdf_formula_adjacent_paragraph_elements,  # noqa: F401 - legacy facade
    _split_pdf_reference_entry_elements,  # noqa: F401 - legacy facade
    _split_pdf_text_elements_into_semantic_fragments,  # noqa: F401 - legacy facade
    _split_trailing_fragment_for_merge,  # noqa: F401 - legacy facade
    _strip_pdf_toc_label,  # noqa: F401 - legacy facade
    _translate_pdf_citation_labels,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.targets import (
    _classify_pdf_page_text_elements,  # noqa: F401 - legacy facade
    _dedupe_pdf_translation_targets,  # noqa: F401 - legacy facade
    _extract_page_elements,  # noqa: F401 - legacy facade
    _is_preserved_nonlinguistic_text,  # noqa: F401 - legacy facade
    _is_preserved_short_citation,  # noqa: F401 - legacy facade
    _looks_like_pdf_entity_directory_row,  # noqa: F401 - legacy facade
    _looks_like_pdf_proper_name_label,  # noqa: F401 - legacy facade
    _looks_like_pdf_translatable_short_label,  # noqa: F401 - legacy facade
    _looks_like_tiny_pdf_label,  # noqa: F401 - legacy facade
    _looks_like_translatable_english,  # noqa: F401 - legacy facade
    _mark_pdf_entity_directory_elements,  # noqa: F401 - legacy facade
    _mark_pdf_glossary_term_elements,  # noqa: F401 - legacy facade
    _pdf_element_is_multiline_display_title,  # noqa: F401 - legacy facade
    _pdf_element_requires_translation,  # noqa: F401 - legacy facade
    _pdf_translation_language_views,  # noqa: F401 - legacy facade
    _translate_pdf_deterministic_labels,  # noqa: F401 - legacy facade
    _translate_pdf_fixed_short_label,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.types import (
    PDF_BATCH_SEPARATOR,  # noqa: F401 - legacy facade
    PDF_HTMLBOX_SUPERSCRIPT_CSS,  # noqa: F401 - legacy facade
    PDF_LAYOUT_CACHE_COMPATIBLE_VERSIONS,  # noqa: F401 - legacy facade
    PDF_LAYOUT_SEMANTICS_VERSION,  # noqa: F401 - legacy facade
    PDF_MAX_TRANSLATION_EXPANSION_RATIO,  # noqa: F401 - legacy facade
    PDF_MAX_TRANSLATION_EXPANSION_SLACK,  # noqa: F401 - legacy facade
    PDF_MIN_ACCEPTABLE_HTMLBOX_SCALE,  # noqa: F401 - legacy facade
    PDF_PAGE_CACHE_SCHEMA_VERSION,  # noqa: F401 - legacy facade
    PDF_TEXT_NEIGHBOR_GUARD_PADDING,  # noqa: F401 - legacy facade
    _PDF_MIXED_TRANSLATION_ENGLISH_RUN_RE,  # noqa: F401 - legacy facade
    _PDF_REFERENCE_TRANSLATION_YEAR_RE,  # noqa: F401 - legacy facade
    _PDF_TOC_ENTRY_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_CJK_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_LONE_WORD_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_NEIGHBOR_AFTER_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_NEIGHBOR_BEFORE_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_RUN_RE,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_SYMBOL_WORDS,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_UNIT_WORDS,  # noqa: F401 - legacy facade
    _PDF_UNTRANSLATED_LEAK_WORD_RE,  # noqa: F401 - legacy facade
    _PDF_WORK_TITLE_QUOTE_CHARS,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.translation import (
    _build_pdf_batch_segment_envelope,  # noqa: F401 - legacy facade
    _has_chinese,  # noqa: F401 - legacy facade
    _looks_like_pdf_toc_leader_block,  # noqa: F401 - legacy facade
    _parse_pdf_batch_segment_envelope,  # noqa: F401 - legacy facade
    _pdf_batch_segment_records_from_source,  # noqa: F401 - legacy facade
    _pdf_batch_translation_candidate_is_valid,  # noqa: F401 - legacy facade
    _pdf_collect_untranslated_english_leaks,  # noqa: F401 - legacy facade
    _pdf_element_translation_needs_retry,  # noqa: F401 - legacy facade
    _pdf_english_run_looks_like_proper_name,  # noqa: F401 - legacy facade
    _pdf_has_untranslated_citation_label,  # noqa: F401 - legacy facade
    _pdf_mixed_translation_has_untranslated_english_run,  # noqa: F401 - legacy facade
    _pdf_number_anchor_plain_text,  # noqa: F401 - legacy facade
    _pdf_reference_entry_translation_needs_retry,  # noqa: F401 - legacy facade
    _pdf_semantic_number_anchor_preserved,  # noqa: F401 - legacy facade
    _pdf_toc_entries,  # noqa: F401 - legacy facade
    _pdf_toc_trailing_text,  # noqa: F401 - legacy facade
    _pdf_toc_translation_needs_retry,  # noqa: F401 - legacy facade
    _pdf_untranslated_english_leak_runs,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_translation_hint,  # noqa: F401 - legacy facade
    _short_translation_needs_retry,  # noqa: F401 - legacy facade
    _split_pdf_reference_entry_translation_parts,  # noqa: F401 - legacy facade
    _split_pdf_translated_paragraphs,  # noqa: F401 - legacy facade
    _translation_part_needs_retry,  # noqa: F401 - legacy facade
    _validate_pdf_page_translations,  # noqa: F401 - legacy facade
    translate_text as _translate_text_impl,  # noqa: F401 - legacy facade
    PDFTranslationDependencies,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.cache import (
    _append_pdf_source_page_fallback,  # noqa: F401 - legacy facade
    _exclude_pdf_audit_expectations_for_element,  # noqa: F401 - legacy facade
    _exclude_pdf_audit_expectations_for_source_pages,  # noqa: F401 - legacy facade
    _json_safe_pdf_layout,  # noqa: F401 - legacy facade
    _load_pdf_cached_page_indices,  # noqa: F401 - legacy facade
    _load_pdf_layout_cache,  # noqa: F401 - legacy facade
    _load_pdf_page_translation_cache,  # noqa: F401 - legacy facade
    _migrate_pdf_page_translation_cache_by_identity,  # noqa: F401 - legacy facade
    _pdf_audit_path,  # noqa: F401 - legacy facade
    _pdf_cache_element_identity,  # noqa: F401 - legacy facade
    _pdf_layout_cache_path,  # noqa: F401 - legacy facade
    _pdf_page_cache_path,  # noqa: F401 - legacy facade
    _pdf_source_page_fallback_entry,  # noqa: F401 - legacy facade
    _pdf_source_sha256,  # noqa: F401 - legacy facade
    _reconcile_pdf_page_translation_cache,  # noqa: F401 - legacy facade
    _record_pdf_element_source_fallback,  # noqa: F401 - legacy facade
    _replace_pdf_pages_with_source,  # noqa: F401 - legacy facade
    _save_pdf_audit,  # noqa: F401 - legacy facade
    _save_pdf_layout_cache,  # noqa: F401 - legacy facade
    _save_pdf_page_translation_cache,  # noqa: F401 - legacy facade
    configure_pdf_cache_progress_dir,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.fonts import (
    _collect_pdf_font_chars,  # noqa: F401 - legacy facade
    _get_chinese_font_path,  # noqa: F401 - legacy facade
    _subset_pdf_font,  # noqa: F401 - legacy facade
    pdf_font_subsetting_available,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.rendering import (
    _PDFInlineHTMLParser,  # noqa: F401 - legacy facade
    _build_htmlbox_rect_ladder,  # noqa: F401 - legacy facade
    _build_pdf_rotated_text_rects,  # noqa: F401 - legacy facade
    _collect_pdf_overlap_redraw_indices,  # noqa: F401 - legacy facade
    _compact_pdf_file_in_place,  # noqa: F401 - legacy facade
    _constrain_pdf_htmlbox_rect_ladder_to_neighbors,  # noqa: F401 - legacy facade
    _expand_pdf_htmlbox_rect,  # noqa: F401 - legacy facade
    _expand_pdf_textbox_fit_rect,  # noqa: F401 - legacy facade
    _insert_pdf_htmlbox_at_readable_scale,  # noqa: F401 - legacy facade
    _insert_pdf_rotated_textbox_with_fit,  # noqa: F401 - legacy facade
    _insert_pdf_textbox_with_fit,  # noqa: F401 - legacy facade
    _paint_pdf_vector_ocr_element_background,  # noqa: F401 - legacy facade
    _pdf_plain_text_to_html,  # noqa: F401 - legacy facade
    _pdf_text_to_html,  # noqa: F401 - legacy facade
    _resolve_pdf_render_style,  # noqa: F401 - legacy facade
    _restore_pdf_source_text_element,  # noqa: F401 - legacy facade
    _sanitize_pdf_text_rect,  # noqa: F401 - legacy facade
    _wrap_pdf_latin_runs,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.audit import (
    _check_pdf_output_structure,  # noqa: F401 - legacy facade
    _check_pdf_output_structure_serialized as _check_pdf_output_structure_serialized_impl,  # noqa: F401 - legacy facade
    _collect_pdf_formula_expectations,  # noqa: F401 - legacy facade
    _collect_pdf_superscript_expectations,  # noqa: F401 - legacy facade
    _collect_pdf_vector_ocr_expectations,  # noqa: F401 - legacy facade
    _expand_pdf_fallback_pages_for_accepted_merges,  # noqa: F401 - legacy facade
    _pdf_formula_protection_fallback_pages,  # noqa: F401 - legacy facade
    _pdf_formula_raster_difference_summary,  # noqa: F401 - legacy facade
    _pdf_glyph_substitution_warnings,  # noqa: F401 - legacy facade
    _pdf_prose_inline_math_symbol_tolerance,  # noqa: F401 - legacy facade
    _pdf_text_element_formula_audit_symbol_count,  # noqa: F401 - legacy facade
    _pdf_text_element_unprotected_math_symbols,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_audit_phrases,  # noqa: F401 - legacy facade
    _pdf_vector_ocr_page_text,  # noqa: F401 - legacy facade
    _preserve_pdf_formula_risk_text_elements,  # noqa: F401 - legacy facade
    _require_clean_pdf_structure_check,  # noqa: F401 - legacy facade
    _save_and_validate_pdf_audit,  # noqa: F401 - legacy facade
    _save_pdf_translation_progress as _save_pdf_translation_progress_impl,  # noqa: F401 - legacy facade
    _summarize_pdf_formula_protection,  # noqa: F401 - legacy facade
    _validate_pdf_merge_audit,  # noqa: F401 - legacy facade
    PDFAuditDependencies,  # noqa: F401 - legacy facade
)
from phoena_translator.pdf.pipeline import (
    translate_pdf as _translate_pdf_impl,  # noqa: F401 - legacy facade
    PDFPipelineDependencies,  # noqa: F401 - legacy facade
)


def exported_names() -> tuple[str, ...]:
    """Return stable legacy names without exposing module bookkeeping."""

    return tuple(
        sorted(
            name
            for name in globals()
            if not name.startswith("__") and name != "exported_names"
        )
    )


__all__ = ["exported_names"]
