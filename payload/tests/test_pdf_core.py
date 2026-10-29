"""PDF pipeline core: context, cache identities, recovery and shared geometry/model helpers.

Split from the former test_review_fixes.py; run from ``payload``:
``python -m pytest tests -q``.
"""

from __future__ import annotations

import copy
import logging
import threading

import pytest
from review_support import (
    _policy_context,
    _policy_result,
    _policy_test_elements,
    _single_formula_page,
)

log = logging.getLogger("test")


def _without_layout_mutation(stage):
    replacements = {
        "_mark_watermark_elements": lambda *args, **kwargs: None,
        "_classify_pdf_page_text_elements": lambda *args, **kwargs: None,
        "_merge_cross_page_sentences": lambda *args, **kwargs: 0,
        "_preserve_pdf_formula_mid_sentence_neighbors": (
            lambda *args, **kwargs: []
        ),
    }
    originals = {name: getattr(stage, name) for name in replacements}
    for name, replacement in replacements.items():
        setattr(stage, name, replacement)
    return originals


def _restore_stage_functions(stage, originals):
    for name, original in originals.items():
        setattr(stage, name, original)


# ---------------------------------------------------------------------------
# 2026-07-25 Advanced-Core: typed contracts and edge ownership
# ---------------------------------------------------------------------------


def _advanced_native_cache_element():
    return {
        "type": "text",
        "content": "Ordinary prose sentence for a stable cache identity.",
        "rich_content": "",
        "paragraphs": [
            {
                "plain": "Ordinary prose sentence for a stable cache identity.",
                "rich": "",
                "text_align": "left",
                "nowrap": False,
                "margin_left": 0.0,
                "text_indent": 0.0,
                "gap_before": 0.0,
                "first_line_indent": False,
            }
        ],
        "bbox": [72.0, 100.0, 510.0, 132.0],
        "layout_class": "body",
        "fontsize": 11.0,
        "bold": False,
        "color": 0,
        "non_horizontal": False,
        "preserve_source_style": False,
    }


def test_prepare_fonts_subsets_distinct_regular_and_bold(
    tmp_path,
    monkeypatch,
):
    import shutil
    from types import SimpleNamespace

    from phoena_translator.pdf.render import stage as output_stage
    regular = tmp_path / "regular.ttf"
    bold = tmp_path / "bold.ttf"
    regular.write_bytes(b"regular")
    bold.write_bytes(b"bold")
    calls = []

    def fake_subset(source, chars, output_dir, role):
        calls.append((source, chars, role))
        subset = tmp_path / f"{role}-subset.ttf"
        subset.write_bytes(role.encode())
        return str(subset)

    class FakeArchive:
        def __init__(self, path):
            self.paths = [path]

        def add(self, path):
            self.paths.append(path)

    monkeypatch.setattr(
        output_stage,
        "_collect_pdf_font_chars",
        lambda *_args: "ABC测试",
    )
    monkeypatch.setattr(output_stage, "_subset_pdf_font", fake_subset)
    monkeypatch.setattr(
        output_stage,
        "_subset_font_renders_like_original",
        lambda *_args: True,
    )
    monkeypatch.setattr(output_stage.fitz, "Archive", FakeArchive)
    context = SimpleNamespace(
        task_id="font-test",
        font_path=str(regular),
        bold_font_path=str(bold),
        font_subsetting_available=True,
        logger=logging.getLogger("font-test"),
    )

    subset_dir = None
    try:
        subset_regular, subset_bold, distinct, _archive, subset_dir = (
            output_stage._prepare_fonts(context, {})
        )

        assert distinct is True
        assert subset_regular.endswith("regular-subset.ttf")
        assert subset_bold.endswith("bold-subset.ttf")
        assert [call[2] for call in calls] == ["regular", "bold"]
        assert subset_dir is not None
    finally:
        if subset_dir:
            shutil.rmtree(subset_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Production crash 2026-07-23: Rect(None) raises AssertionError in PyMuPDF
# ---------------------------------------------------------------------------


class TestSourceInkRectRobustness:
    def test_paragraph_without_any_geometry_falls_back_to_elem_rect(self):
        import fitz

        from phoena_translator.pdf.geometry import _get_pdf_source_ink_rects

        elem = {
            "type": "text",
            "bbox": [10.0, 20.0, 200.0, 40.0],
            # The exact shape that crashed assembly on the SEC document:
            # a paragraph carrying no source_lines, no source_line_bboxes
            # and no source_bbox.
            "paragraphs": [{"plain": "cell text", "text_align": "left"}],
        }
        rects = _get_pdf_source_ink_rects(elem)
        assert rects == [fitz.Rect(10.0, 20.0, 200.0, 40.0)]

    def test_malformed_bbox_values_are_skipped(self):
        import fitz

        from phoena_translator.pdf.geometry import (
            _get_pdf_source_ink_rects,
            _safe_bbox_rect,
        )

        assert _safe_bbox_rect(None) is None
        assert _safe_bbox_rect("garbage") is None
        assert _safe_bbox_rect([1, 2]) is None
        assert _safe_bbox_rect([0, 0, 0, 0]) is None  # empty rect
        assert _safe_bbox_rect([1, 2, 3, 4]) == fitz.Rect(1, 2, 3, 4)

        elem = {
            "type": "text",
            "bbox": [0.0, 0.0, 100.0, 10.0],
            "paragraphs": [
                {"plain": "x", "source_bbox": None},
                {"plain": "y", "source_line_bboxes": [None, "bad", [5, 5, 50, 15]]},
            ],
        }
        rects = _get_pdf_source_ink_rects(elem)
        assert rects == [fitz.Rect(5, 5, 50, 15)]


def test_table_overlap_does_not_capture_first_body_line_below_chart():
    import fitz

    from phoena_translator.pdf.geometry import _line_overlaps_pdf_table_rects

    table_rect = fitz.Rect(72.15, 98.55, 552.26, 368.06)
    body_line = [72.00, 379.92, 542.25, 392.75]
    edge_cell_line = [90.00, 367.00, 300.00, 378.00]

    assert not _line_overlaps_pdf_table_rects(body_line, [table_rect])
    assert _line_overlaps_pdf_table_rects(edge_cell_line, [table_rect])


def test_warm_cache_policy_transition_reconciles_before_completion(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf.extract import stage
    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "a" * 64
    off_elements = _policy_test_elements()
    translations = {
        **{
            str(index): (
                f"这是合约类别 {index + 1} 每日交易量为 123 单位的完整中文译文。"
            )
            for index in range(5)
        },
        "5": "这是需要翻译的普通正文句子的完整中文译文。",
    }
    cache._save_pdf_page_translation_cache(
        "policy-test",
        0,
        translations,
        source_sha256,
        elements=off_elements,
    )
    result = _policy_result(
        copy.deepcopy(off_elements),
        source_sha256,
        cached=True,
    )
    context = _policy_context(tmp_path, True)
    originals = _without_layout_mutation(stage)
    try:
        stage.reconcile_extraction(context, result)
        # Retain the historical call so this regression follows both the
        # pre-fix and post-fix production surfaces.
        stage.enforce_table_preservation(context, result)
    finally:
        _restore_stage_functions(stage, originals)

    elements = result.page_extractions[0]["elements"]
    cached = result.page_extractions[0]["cached"]
    assert all(
        element.get("skip_translate_reason") == "table_preserved"
        for element in elements[:5]
    )
    assert all(cached[str(index)] == elements[index]["content"] for index in range(5))
    assert cached["5"] == "这是需要翻译的普通正文句子的完整中文译文。"
    assert result.completed_indices == {0}


def test_table_cache_policy_transition_matrix(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf.semantics.table_preservation import (
        _mark_pdf_table_preserved_elements,
    )

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "b" * 64

    def elements_for(policy):
        elements = _policy_test_elements()
        if policy:
            assert len(_mark_pdf_table_preserved_elements({0: {"elements": elements}})) == 5
        return elements

    def save(policy, task_id):
        elements = elements_for(policy)
        translations = {
            str(index): (
                element["content"]
                if element.get("skip_translate_reason")
                else (
                    "这是需要翻译的普通正文句子的完整中文译文。"
                    if index == 5
                    else (
                        f"这是合约类别 {index + 1} 每日交易量为 123 "
                        "单位的完整中文译文。"
                    )
                )
            )
            for index, element in enumerate(elements)
        }
        path = cache._save_pdf_page_translation_cache(
            task_id,
            0,
            translations,
            source_sha256,
            elements=elements,
        )
        return path, translations

    off_path, off_translations = save(False, "cache-off")
    on_path, on_translations = save(True, "cache-on")

    off_off = cache._load_pdf_page_translation_cache(
        off_path, source_sha256, elements_for(False)
    )
    off_on_elements = elements_for(True)
    off_on = cache._load_pdf_page_translation_cache(
        off_path, source_sha256, off_on_elements
    )
    on_on_first = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(True)
    )
    on_on_second = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(True)
    )
    on_off = cache._load_pdf_page_translation_cache(
        on_path, source_sha256, elements_for(False)
    )

    assert off_off == off_translations
    assert all(
        off_on[str(index)] == off_on_elements[index]["content"]
        for index in range(5)
    )
    assert off_on["5"] == "这是需要翻译的普通正文句子的完整中文译文。"
    assert on_on_first == on_translations
    assert on_on_second == on_translations
    assert on_off == {"5": "这是需要翻译的普通正文句子的完整中文译文。"}
    assert (
        cache._load_pdf_page_translation_cache(
            off_path, "c" * 64, elements_for(False)
        )
        is None
    )


def test_table_formula_overlap_preserves_existing_precedence(tmp_path):
    from phoena_translator.pdf import cache
    from phoena_translator.pdf.extract import stage
    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    elements = _policy_test_elements()
    elements[0]["content"] = (
        "Regression alpha α plus beta β equals gamma γ under limit δ."
    )
    result = _policy_result(elements)
    context = _policy_context(tmp_path, True)
    originals = _without_layout_mutation(stage)
    original_bind = stage._bind_and_reconcile_caches
    stage._bind_and_reconcile_caches = lambda *args, **kwargs: None
    try:
        stage.reconcile_extraction(context, result)
        stage.enforce_formula_protection(context, result)
        stage.enforce_table_preservation(context, result)
    finally:
        stage._bind_and_reconcile_caches = original_bind
        _restore_stage_functions(stage, originals)

    target = result.page_extractions[0]["elements"][0]
    assert target["skip_translate_reason"] == "formula_risk_preserved"
    assert context.pdf_audit["formula_risk_preserved_elements"]
    assert all(
        entry["element"] != 0
        for entry in context.pdf_audit["table_preserved_elements"]
    )


def test_cache_identity_golden_is_unchanged_before_typed_hardening():
    from phoena_translator.pdf.cache import _pdf_cache_element_identity

    elem = _advanced_native_cache_element()
    assert (
        _pdf_cache_element_identity(elem)
        == "bae0777a3fafe1f4df4b694efc7a73fe5921c475cf9fb228963689246bf423ea"
    )
    assert (
        _pdf_cache_element_identity(elem, _include_empty_vector_fields=True)
        == "2346d344b4cdf203267bfba8ea611261231238a6db3c867b6db185540a76562a"
    )


def test_cache_identity_typed_core_preserves_legacy_digest():
    from dataclasses import FrozenInstanceError

    from phoena_translator.pdf.cache import (
        CacheIdentity,
        _pdf_cache_element_identity,
        _pdf_cache_identity,
    )

    elem = _advanced_native_cache_element()
    identity = _pdf_cache_identity(elem)
    expected = "bae0777a3fafe1f4df4b694efc7a73fe5921c475cf9fb228963689246bf423ea"

    assert isinstance(identity, CacheIdentity)
    assert identity.digest == expected
    assert _pdf_cache_element_identity(elem) == expected
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        CacheIdentity("not-a-cache-digest")
    with pytest.raises(FrozenInstanceError):
        identity.digest = "0" * 64


def test_translation_integrity_source_fallback_cache_rebinds_once(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "d" * 64
    source_element = _advanced_native_cache_element()
    fallback_element = copy.deepcopy(source_element)
    fallback_element["skip_translate_reason"] = "translation_integrity_fallback"
    path = cache._save_pdf_page_translation_cache(
        "fallback-cache",
        0,
        {"0": source_element["content"]},
        source_sha256,
        elements=[fallback_element],
    )

    resumed_element = copy.deepcopy(source_element)
    loaded = cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    )

    assert loaded == {"0": source_element["content"]}
    assert resumed_element["skip_translate_reason"] == "translation_integrity_fallback"


def test_unmarked_source_echo_cache_is_still_rejected(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "e" * 64
    source_element = _advanced_native_cache_element()
    path = cache._save_pdf_page_translation_cache(
        "source-echo-cache",
        0,
        {"0": source_element["content"]},
        source_sha256,
        elements=[source_element],
    )

    resumed_element = copy.deepcopy(source_element)
    assert cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    ) == {}
    assert "skip_translate_reason" not in resumed_element


def test_fallback_cache_is_checked_after_a_normal_source_echo_is_rejected(tmp_path):
    from phoena_translator.pdf import cache

    cache.configure_pdf_cache_progress_dir(lambda: str(tmp_path))
    source_sha256 = "f" * 64
    source_element = _advanced_native_cache_element()
    fallback_element = copy.deepcopy(source_element)
    fallback_element["skip_translate_reason"] = "translation_integrity_fallback"
    path = cache._save_pdf_page_translation_cache(
        "mixed-source-echo-cache",
        0,
        {
            "0": source_element["content"],
            "1": source_element["content"],
        },
        source_sha256,
        elements=[source_element, fallback_element],
    )

    resumed_element = copy.deepcopy(source_element)
    loaded = cache._load_pdf_page_translation_cache(
        path,
        source_sha256,
        elements=[resumed_element],
    )

    assert loaded == {"0": source_element["content"]}
    assert resumed_element["skip_translate_reason"] == "translation_integrity_fallback"


def test_clean_page_retries_use_the_configured_worker_pool(monkeypatch):
    import time
    from types import SimpleNamespace

    from phoena_translator.pdf.translate import stage
    lock = threading.Lock()
    active = 0
    max_active = 0
    committed: list[int] = []

    def fake_translate_page(_page_context, page_num, _page_info):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return page_num, {0: "译文"}

    monkeypatch.setattr(
        stage,
        "_run_first_pass",
        lambda *_args, **_kwargs: {page: RuntimeError("retry") for page in range(3)},
    )
    monkeypatch.setattr(stage, "translate_page", fake_translate_page)
    monkeypatch.setattr(stage, "_validate_pdf_page_translations", lambda *_args: None)
    monkeypatch.setattr(
        stage,
        "_commit_translated_page",
        lambda _context, page_num, _translations, _lock: committed.append(page_num),
    )
    monkeypatch.setattr(stage, "_apply_page_source_fallbacks", lambda *_args: None)
    monkeypatch.setattr(stage, "_require_translated_page_progress", lambda *_args: None)

    context = SimpleNamespace(
        workers=3,
        page_extractions={page: {"elements": [{}]} for page in range(3)},
        logger=log,
        task_id="parallel-clean-retry",
    )
    stage.translate_pages(context, SimpleNamespace(), [0, 1, 2])

    assert max_active == 3
    assert sorted(committed) == [0, 1, 2]


def test_clean_page_retries_keep_worker_errors_bound_to_scheduled_pages(monkeypatch):
    from types import SimpleNamespace

    from phoena_translator.pdf.translate import stage
    captured_errors = {}
    committed = []

    def fake_translate_page(_page_context, page_num, _page_info):
        if page_num == 0:
            return 2, {0: "wrong page"}
        raise TimeoutError("provider retry timed out")

    monkeypatch.setattr(
        stage,
        "_run_first_pass",
        lambda *_args, **_kwargs: {0: RuntimeError("retry"), 1: RuntimeError("retry")},
    )
    monkeypatch.setattr(stage, "translate_page", fake_translate_page)
    monkeypatch.setattr(stage, "_validate_pdf_page_translations", lambda *_args: None)
    monkeypatch.setattr(
        stage,
        "_commit_translated_page",
        lambda _context, page_num, _translations, _lock: committed.append(page_num),
    )
    monkeypatch.setattr(
        stage,
        "_apply_page_source_fallbacks",
        lambda _context, errors: captured_errors.update(errors),
    )
    monkeypatch.setattr(stage, "_require_translated_page_progress", lambda *_args: None)

    context = SimpleNamespace(
        workers=2,
        page_extractions={page: {"elements": [{}]} for page in range(3)},
        logger=log,
        task_id="retry-page-binding",
    )
    stage.translate_pages(context, SimpleNamespace(), [0, 1])

    assert committed == []
    assert sorted(captured_errors) == [1, 2]
    assert "returned page 3 for scheduled page 1" in str(captured_errors[1])
    assert isinstance(captured_errors[2], TimeoutError)


def test_render_failure_restores_source_pages_without_restarting_pipeline(
    tmp_path,
    monkeypatch,
):
    import shutil
    from types import SimpleNamespace

    import fitz

    from phoena_translator.pdf.errors import PDFPageRenderError
    from phoena_translator.pdf.render import assembly

    source_path = tmp_path / "source.pdf"
    work_path = tmp_path / "assembly-work.pdf"
    source_document = fitz.open()
    for page_number in range(1, 4):
        page = source_document.new_page()
        page.insert_text((72, 72), f"SOURCE PAGE {page_number}")
    source_document.save(source_path)
    source_document.close()
    shutil.copy2(source_path, work_path)

    work_document = fitz.open(work_path)
    work_document[0].insert_text((72, 100), "CHECKPOINTED TRANSLATION")
    work_document.saveIncr()
    # Simulate the dirty current page at the instant its renderer raises. This
    # mutation must disappear when the document closes without another save.
    work_document[1].insert_text((72, 100), "UNSAFE PARTIAL RENDER")

    audit = {
        "source_page_fallbacks": [],
        "merge_decisions": [],
        "superscript_expectations": [{"page": 1}, {"page": 2}, {"page": 3}],
        "formula_expectations": [],
        "vector_ocr_expectations": [],
    }
    context = SimpleNamespace(
        task_id="inline-render-recovery",
        total_pages=3,
        pdf_password="",
        fail_open_to_source_page=True,
        assembly_work_path=str(work_path),
        src_path=str(source_path),
        out_doc=work_document,
        page_extractions={0: {}, 1: {}, 2: {}},
        pdf_audit=audit,
        logger=log,
    )
    monkeypatch.setattr(
        assembly,
        "_expand_pdf_fallback_pages_for_accepted_merges",
        lambda *_args, **_kwargs: [1, 2],
    )

    recovered = assembly._continue_assembly_with_source_pages(
        context,
        1,
        PDFPageRenderError(2, 7, "no safe rectangle"),
    )
    try:
        assert recovered is True
        assert "CHECKPOINTED TRANSLATION" not in context.out_doc[0].get_text()
        assert "UNSAFE PARTIAL RENDER" not in context.out_doc[1].get_text()
        assert [
            entry["page"] for entry in audit["source_page_fallbacks"]
        ] == [1, 2]
        assert context.page_extractions[0]["source_page_fallback"]["page"] == 1
        assert context.page_extractions[1]["source_page_fallback"]["page"] == 2
        assert audit["superscript_expectations"] == [{"page": 3}]
    finally:
        context.out_doc.close()


def test_assembly_continues_after_an_inline_render_source_fallback(monkeypatch):
    from types import SimpleNamespace

    from phoena_translator.pdf.errors import PDFPageRenderError
    from phoena_translator.pdf.render import assembly

    attempted = []

    def fake_assemble_page(_context, page_num):
        attempted.append(page_num)
        if page_num == 0:
            raise PDFPageRenderError(1, 2, "render failure")

    monkeypatch.setattr(assembly, "_assemble_page", fake_assemble_page)
    monkeypatch.setattr(
        assembly,
        "_continue_assembly_with_source_pages",
        lambda *_args, **_kwargs: True,
    )

    assembly.assemble_document_pages(SimpleNamespace(total_pages=3))

    assert attempted == [0, 1, 2]


def test_pipeline_dependencies_freeze_one_element_policy(tmp_path):
    from dataclasses import FrozenInstanceError
    from inspect import signature

    from phoena_translator.pdf.context import (
        ImmutableElementPolicy,
        PDFPipelineDependencies,
    )

    policy = ImmutableElementPolicy(preserve_tables=True)
    assert policy.preserve_tables is True
    with pytest.raises(FrozenInstanceError):
        policy.preserve_tables = False

    dependency_field = PDFPipelineDependencies.__dataclass_fields__[
        "element_policy"
    ]
    assert dependency_field.init is False
    constructor_parameters = signature(PDFPipelineDependencies).parameters
    assert "preserve_tables" in constructor_parameters
    assert "element_policy" not in constructor_parameters

    context = _policy_context(tmp_path, True)
    assert context.element_policy == policy
    assert context.preserve_tables is True
    assert "preserve_tables" not in context.__dataclass_fields__


def test_pipeline_dependencies_preserve_legacy_bool_constructor():
    from phoena_translator.pdf import pipeline
    from phoena_translator.pdf.context import (
        ImmutableElementPolicy,
        PDFPipelineDependencies,
    )

    no_op = lambda *args, **kwargs: None
    dependencies = PDFPipelineDependencies(
        api_max_concurrency=1,
        assembly_max_concurrency=1,
        extraction_max_concurrency=1,
        fail_open_to_source_page=False,
        preserve_tables=True,
        minimum_htmlbox_scale=0.5,
        save_clean=False,
        save_garbage=0,
        use_htmlbox=True,
        progress_dir="/tmp/advanced-policy-test",
        system_prompt_text="test",
        translation_workers=1,
        assembly_semaphore=threading.Semaphore(1),
        extraction_semaphore=threading.Semaphore(1),
        logger=log,
        build_glossary=no_op,
        make_prompt_with_glossary=no_op,
        load_progress=no_op,
        save_progress=no_op,
        translate_text=no_op,
        trim_process_memory=no_op,
        extract_page_elements=no_op,
        filter_formula_safe_rects=no_op,
        summarize_formula_protection=no_op,
        check_output_structure_serialized=no_op,
        save_translation_progress=no_op,
        font_subsetting_available=False,
    )

    assert dependencies.preserve_tables is True
    assert isinstance(dependencies.element_policy, ImmutableElementPolicy)
    assert dependencies.element_policy.preserve_tables is True

    class _ContextCaptured(RuntimeError):
        pass

    captured = {}
    original_context = pipeline.PDFExtractionStageContext
    original_resolve_fonts = pipeline.resolve_output_fonts

    def capture_context(**kwargs):
        captured.update(kwargs)
        raise _ContextCaptured

    pipeline.PDFExtractionStageContext = capture_context
    pipeline.resolve_output_fonts = lambda *args, **kwargs: (None, None)
    try:
        with pytest.raises(_ContextCaptured):
            pipeline._run_pdf_attempt(
                "policy-test",
                "/tmp/source.pdf",
                "/tmp/output.pdf",
                "",
                {},
                dependencies=dependencies,
            )
    finally:
        pipeline.PDFExtractionStageContext = original_context
        pipeline.resolve_output_fonts = original_resolve_fonts

    assert captured["element_policy"] is dependencies.element_policy
    assert "preserve_tables" not in captured


def test_preservation_is_decided_before_cross_page_merging():
    import inspect

    from phoena_translator.pdf.extract import stage as extraction_stage
    body = inspect.getsource(extraction_stage.reconcile_extraction)
    merge_at = body.index("_merge_cross_page_sentences")
    for stamper in (
        "_preserve_pdf_formula_mid_sentence_neighbors",
        "_preserve_formula_risk_before_cache_binding",
        "enforce_table_preservation",
    ):
        assert body.index(stamper) < merge_at, stamper
    # ...and every stamp still lands before cache binding.
    assert merge_at < body.index("_bind_and_reconcile_caches")


def test_one_unsignable_formula_does_not_strip_its_page_siblings(monkeypatch):
    from phoena_translator.pdf.extract import page as page_extraction
    attempted = []

    def flaky_signature(page, bbox, page_dict=None):
        attempted.append(list(bbox))
        if len(attempted) == 1:
            raise ValueError("empty formula region")
        return {"signature_sha256": "b" * 64, "span_count": 1}

    monkeypatch.setattr(
        page_extraction,
        "_pdf_formula_region_signature",
        flaky_signature,
    )
    doc, page = _single_formula_page(0)
    elements = [
        {"type": "formula_image", "bbox": [10.0, 10.0, 40.0, 30.0]},
        {"type": "formula_image", "bbox": [10.0, 50.0, 40.0, 70.0]},
        {"type": "formula_image", "bbox": [10.0, 90.0, 40.0, 110.0]},
    ]

    enriched = page_extraction._enrich_pdf_page_elements(page, elements, [])

    assert len(attempted) == 3, "every formula must be attempted"
    assert "math_signature" not in enriched[0]
    assert enriched[1]["math_signature"]["signature_sha256"] == "b" * 64
    assert enriched[2]["math_signature"]["signature_sha256"] == "b" * 64
    doc.close()


class TestRotatedPageCoordinateSpace:
    """Extraction geometry lives in the unrotated box, not ``page.rect``."""

    @staticmethod
    def _rotated_page(rotation):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(
            fitz.Rect(72, 640, 540, 760),
            "Regulators responded with circuit breakers and minimum quoting "
            "obligations, but the evidence on their effectiveness is mixed.",
            fontsize=11,
            fontname="tiro",
        )
        if rotation:
            page.set_rotation(rotation)
        return doc, page

    def test_extraction_rect_is_a_noop_without_rotation(self):

        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect

        doc, page = self._rotated_page(0)
        assert tuple(_pdf_page_extraction_rect(page)) == tuple(page.rect)
        doc.close()

    def test_extraction_rect_derotates_the_display_box(self):
        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect

        doc, page = self._rotated_page(90)
        assert tuple(page.rect) == (0.0, 0.0, 792.0, 612.0)
        assert tuple(_pdf_page_extraction_rect(page)) == (0.0, 0.0, 612.0, 792.0)
        doc.close()

    def test_clamping_against_the_display_box_destroys_the_rect(self):
        import fitz

        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect
        from phoena_translator.pdf.render.primitives import _sanitize_pdf_text_rect

        doc, page = self._rotated_page(90)
        elem = fitz.Rect(100.0, 700.0, 300.0, 714.0)

        crushed = _sanitize_pdf_text_rect(fitz.Rect(elem), fitz.Rect(page.rect))
        intact = _sanitize_pdf_text_rect(
            fitz.Rect(elem), _pdf_page_extraction_rect(page)
        )

        assert crushed is not None and crushed.height <= 2.0
        assert intact is not None and abs(intact.height - elem.height) < 0.01
        doc.close()

    def test_body_text_in_the_far_band_is_not_demoted_to_scattered(self):
        import fitz

        from phoena_translator.pdf.extract.page import _extract_page_elements
        from phoena_translator.pdf.geometry import _pdf_page_extraction_rect
        from phoena_translator.pdf.semantics.layout_classifier import (
            _classify_pdf_page_text_elements,
        )

        doc, page = self._rotated_page(90)
        elements = _extract_page_elements(page, doc)
        assert elements, "probe page produced no elements"

        display = [dict(e) for e in elements]
        _classify_pdf_page_text_elements(display, fitz.Rect(page.rect))
        extraction = [dict(e) for e in elements]
        _classify_pdf_page_text_elements(
            extraction, _pdf_page_extraction_rect(page)
        )

        assert any(e.get("layout_class") == "scattered" for e in display)
        assert all(
            e.get("layout_class") == "body"
            for e in extraction
            if e.get("type") == "text"
        )
        doc.close()
