"""Regression coverage for local XHTML markup and JSON paragraph translation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from phoena_translator.epub.pipeline import EPUBPipeline, EPUBPipelineDependencies
from phoena_translator.epub.segments import (
    SEGMENT_SYSTEM_PROMPT,
    TOKEN_RE,
    SegmentValidationError,
    batch_segments,
    extract_segments,
    make_payload,
    parse_translation_response,
    repair_invisible_anchors,
    validate_segment_translation,
)
from phoena_translator.epub.xhtml import markup_tokens


PROSE = "This is a complete English sentence that must be translated into Chinese."
CHINESE = "这是一段已经完整翻译成中文的正文，内容涵盖原文所表达的全部信息。"


def xhtml(body: str) -> str:
    return '<html xmlns="http://www.w3.org/1999/xhtml"><body>' + body + "</body></html>"


def chinese(value: str) -> str:
    return CHINESE * max(1, len(value.split()) // 25) + "".join(TOKEN_RE.findall(value))


def dependencies(adapter, progress, **extra):
    state = {}

    def save(task_id, data):
        state.update(data)
        progress.append(dict(data))

    return EPUBPipelineDependencies(
        translate_text=adapter,
        build_glossary=lambda text, task_id: "术语表",
        make_prompt=lambda prompt, glossary: prompt + "\n" + glossary,
        save_progress=save,
        load_task=lambda task_id: state,
        logger=logging.getLogger(__name__),
        workers=1, api_concurrency=1,
        sleep=lambda seconds: pytest.fail("semantic retries must not sleep"),
        **extra,
    )


def write_epub(path: Path, content: str | bytes):
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("OPS/chapter.xhtml", content)
        archive.writestr("OPS/book.opf", '<package xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:language>en</dc:language></package>')


def test_verified_segments_resume_without_api_calls(tmp_path):
    source, output = tmp_path / "input.epub", tmp_path / "output.epub"
    write_epub(source, xhtml(f"<p>{PROSE}</p>"))
    progress = []
    def translate(payload, **kwargs):
        return json.dumps({key: chinese(value) for key, value in json.loads(payload).items()})
    assert EPUBPipeline(dependencies(translate, progress)).translate("resume", str(source), str(output))
    assert output.is_file()
    output.unlink()
    def no_api(*args, **kwargs):
        pytest.fail("validated checkpoint should avoid a provider call")
    assert EPUBPipeline(dependencies(no_api, progress)).translate("resume", str(source), str(output))
    assert progress[-1]["reused_segments"] == 1
    assert progress[-1]["translation_requests"] == 0
    with ZipFile(output) as archive:
        assert CHINESE in archive.read("OPS/chapter.xhtml").decode()


@pytest.mark.parametrize("heading", ["CLOUD", "POWER", "DATA", "MIX"])
def test_uppercase_word_headings_cannot_echo(heading):
    segment = extract_segments(xhtml(f"<h1>{heading}</h1>")).segments[0]
    with pytest.raises(SegmentValidationError):
        validate_segment_translation(segment, heading)


@pytest.mark.parametrize("name", ["John A. Smith", "AT&T"])
def test_middle_initials_and_company_names_can_remain(name):
    import html
    segment = extract_segments(xhtml(f"<p>{html.escape(name)}</p>")).segments[0]
    validate_segment_translation(segment, name)


def test_decorative_latin_initial_is_not_left_before_chinese_translation():
    segment = extract_segments(xhtml('<p><span class="colb">B</span>y the end of 2022, residents were angry.</p>')).segments[0]
    opening, closing = [part.token for part in segment.fragments]
    response = json.dumps({segment.id: opening + 'B' + closing + '到2022年底，居民们已经非常愤怒。'})
    verdict = parse_translation_response(response, [segment])
    assert verdict.translations[segment.id] == opening + '到' + closing + '2022年底，居民们已经非常愤怒。'


@pytest.mark.parametrize("value", ["EC1A 7JQ", "Design LLC, ix–x,", "iPhone, –", "OVHcloud, xiv"])
def test_postal_identifiers_and_index_brand_entries_can_remain(value):
    segment = extract_segments(xhtml('<li>' + value + '</li>')).segments[0]
    validate_segment_translation(segment, value)


def test_query_string_url_is_one_immutable_fragment():
    source = xhtml('<p>A video title at https://www.youtube.com/watch?v=x7qPAY9JqE4.</p>')
    segment = extract_segments(source).segments[0]
    assert [item.original for item in segment.fragments] == ['https://www.youtube.com/watch?v=x7qPAY9JqE4']
    assert 'youtube' not in segment.source


def test_semantic_units_keep_dropcaps_italics_and_footnotes_together():
    source = xhtml(
        '<h1>A chapter heading</h1><p><span class="drop">B</span>y the end of the year, '
        'the <i>company</i> had changed.<a href="#fn1"><sup>1</sup></a></p>'
        '<ul><li>Parent text <i>with context</i><ul><li>Nested text</li></ul>'
        'A trailing explanation.</li></ul>'
    )
    document = extract_segments(source)
    assert len(document.segments) == 5
    paragraph = document.segments[1]
    assert paragraph.plain_text.startswith("By the end of the year")
    assert "company had changed" in paragraph.plain_text
    assert len(paragraph.fragments) == 5
    assert any(fragment.original == '<a href="#fn1"><sup>1</sup></a>'
               for fragment in paragraph.fragments)
    assert document.segments[2].plain_text == "Parent text with context"
    assert document.segments[3].plain_text == "Nested text"
    assert document.segments[4].plain_text == "A trailing explanation."
    assert document.render({s.id: s.source for s in document.segments}, check_quality=False) == source


def test_render_escapes_text_and_preserves_exact_tags_links_comments_and_entities():
    source = xhtml('<p data-note="a &gt; b">Hello <a href="chapter.xhtml#id">world</a><!--keep--></p>')
    document = extract_segments(source)
    segment = document.segments[0]
    tokens = [fragment.token for fragment in segment.fragments]
    translated = "中文 & < 5 " + tokens[0] + "链接" + tokens[1] + tokens[2]
    rendered = document.render({segment.id: translated})
    assert "中文 &amp; &lt; 5" in rendered
    assert markup_tokens(rendered) == markup_tokens(source)
    assert ET.fromstring(rendered) is not None


def test_math_urls_and_email_are_immutable():
    source = xhtml('<p>The formula $x^2 + y^2$ is explained at '
                   '<a href="https://example.com/?a=1&amp;b=2">https://example.com/?a=1&amp;b=2</a> '
                   'and reader@example.com. <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math></p>')
    document = extract_segments(source)
    segment = document.segments[0]
    assert "$x^2" not in segment.source
    assert "example.com" not in segment.source
    originals = [fragment.original for fragment in segment.fragments]
    assert "$x^2 + y^2$" in originals
    assert "https://example.com/?a=1&amp;b=2" in originals
    assert "reader@example.com" in originals
    rendered = document.render({segment.id: chinese(segment.source)})
    assert markup_tokens(rendered) == markup_tokens(source)
    assert "$x^2 + y^2$" in rendered
    assert "https://example.com/?a=1&amp;b=2" in rendered


def test_cdata_is_translated_without_double_escaping():
    document = extract_segments(xhtml('<p><![CDATA[English text & symbols]]></p>'))
    segment = document.segments[0]
    first, last = [fragment.token for fragment in segment.fragments]
    rendered = document.render({segment.id: first + "中文 & 符号" + last})
    assert "<![CDATA[中文 & 符号]]>" in rendered


@pytest.mark.parametrize("kind", ["trailing", "fence", "extra", "duplicate", "list", "nested"])
def test_malformed_or_explanatory_json_is_rejected(kind):
    segment = extract_segments(xhtml(f"<p>{PROSE}</p>")).segments[0]
    good = json.dumps({segment.id: CHINESE}, ensure_ascii=False)
    responses = {
        "trailing": good + "\n翻译完成。",
        "fence": "```json\n" + good + "\n```",
        "extra": json.dumps({segment.id: CHINESE, "explanation": "翻译完成。"}),
        "duplicate": '{"' + segment.id + '": "中文", "' + segment.id + '": "重复"}',
        "list": json.dumps([CHINESE]),
        "nested": json.dumps({segment.id: {"translation": CHINESE}}),
    }
    verdict = parse_translation_response(responses[kind], [segment])
    assert verdict.translations == {}
    assert segment.id in verdict.failures


def test_missing_id_keeps_other_validated_segments():
    segments = extract_segments(xhtml(f"<p>{PROSE}</p><p>{PROSE}</p>")).segments
    verdict = parse_translation_response(json.dumps({segments[0].id: CHINESE}), segments)
    assert verdict.translations == {segments[0].id: CHINESE}
    assert verdict.failures == {segments[1].id: "missing segment ID"}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered", "altered", "invented"])
def test_placeholder_count_and_order_are_checked(kind):
    segment = extract_segments(xhtml('<p>This <i>important</i> sentence needs translation.</p>')).segments[0]
    first, last = [fragment.token for fragment in segment.fragments]
    translations = {
        "missing": CHINESE + first,
        "duplicate": CHINESE + first + first + last,
        "reordered": CHINESE + last + first,
        "altered": CHINESE + first.replace("PH_", "ph_") + last,
        "invented": CHINESE + first + last + "[[PH_0002_DEADBEEF]]",
    }
    with pytest.raises(SegmentValidationError, match="placeholder"):
        validate_segment_translation(segment, translations[kind])


@pytest.mark.parametrize("translation", [PROSE, "中文。" + PROSE, "以下是翻译：" + CHINESE, "中文\x00"])
def test_english_echo_partial_echo_explanation_and_invalid_xml_are_rejected(translation):
    segment = extract_segments(xhtml(f"<p>{PROSE}</p>")).segments[0]
    with pytest.raises(SegmentValidationError):
        validate_segment_translation(segment, translation)


def test_tiny_chinese_prefix_cannot_hide_missing_paragraph():
    segment = extract_segments(xhtml(f"<p>{PROSE * 5}</p>")).segments[0]
    with pytest.raises(SegmentValidationError, match="coverage"):
        validate_segment_translation(segment, "中文")


@pytest.mark.parametrize("text", [
    "John Smith", "Will Smith", "Tim O’Reilly", "New York Times", "The New York Times",
    "University of California", "Center for American Progress", "123–456", "IBM",
    "John Smith, Mary Jones, James Brown, Elizabeth Smith, Charles Jones and Robert Brown",
    "Smith, John, and Mary Jones. 2020.",
])
def test_names_numbers_acronyms_and_citation_apparatus_may_remain(text):
    segment = extract_segments(xhtml(f"<p>{text}</p>")).segments[0]
    validate_segment_translation(segment, segment.source)


def test_book_titles_and_reference_descriptions_need_translation():
    for body in ('<h1>Hyperscale</h1>', '<h1>Civil</h1>', '<h1>The World of Big Tech</h1>',
                 '<li>Smith, John. “The environmental footprint of cloud computing.” 2020.</li>'):
        segment = extract_segments(xhtml(body)).segments[0]
        with pytest.raises(SegmentValidationError):
            validate_segment_translation(segment, segment.source)


def test_pure_url_and_empty_structure_need_no_request():
    source = xhtml('<section><p><a href="https://example.com">https://example.com</a></p><hr/></section>')
    document = extract_segments(source)
    assert document.segments == ()
    assert document.render({}) == source


def test_batches_are_bounded_without_cutting_a_paragraph():
    document = extract_segments(xhtml("".join(f"<p>{PROSE}</p>" for _ in range(5))))
    batches = batch_segments(document.segments, 200)
    assert sum(len(batch) for batch in batches) == 5
    assert all(len(make_payload(batch).encode()) <= 200 for batch in batches)
    assert len(batch_segments(document.segments[:1], 20)[0]) == 1


def test_original_book_all_twenty_files_reconstruct_exactly_with_nested_index_and_notes():
    book = Path(__file__).resolve().parents[2] / "Hyperscale.epub"
    if not book.exists():
        pytest.skip("local source book is not part of the test distribution")
    documents = {}
    with ZipFile(book) as archive:
        for name in archive.namelist():
            if name.endswith(".xhtml"):
                source = archive.read(name).decode("utf-8")
                document = extract_segments(source)
                reconstructed = document.render({s.id: s.source for s in document.segments}, check_quality=False)
                assert reconstructed == source, name
                assert markup_tokens(reconstructed) == markup_tokens(source), name
                documents[name] = document
    assert len(documents) == 20
    intro = documents["ops/xhtml/intro.xhtml"]
    dropcap = next(s for s in intro.segments if s.plain_text.startswith("By the end of 2022"))
    assert "Residents wanted to know" in dropcap.plain_text
    assert len(dropcap.fragments) == 2
    index = documents["ops/xhtml/index.xhtml"]
    assert any(s.plain_text.startswith("AI (artificial intelligence).") for s in index.segments)
    assert any(s.plain_text.startswith("origin of term") for s in index.segments)
    notes = documents["ops/xhtml/notes.xhtml"]
    note = next(s for s in notes.segments if s.plain_text.startswith("Google’s data centers had"))
    assert "Mike Rogoway" in note.plain_text and "soaring" in note.plain_text
    assert any(f.original.startswith("https://") for f in note.fragments)


def test_pipeline_retries_only_failed_segments_and_prefers_raw_payload_adapter(tmp_path):
    progress, calls, prompts = [], [], []
    source = xhtml("".join(f"<p>{PROSE}</p>" for _ in range(3)))
    source_epub, output = tmp_path / "source.epub", tmp_path / "translated.epub"
    write_epub(source_epub, source)

    def adapter(payload, system_prompt):
        values = json.loads(payload)
        calls.append(values)
        prompts.append(system_prompt)
        if len(calls) == 1:
            return json.dumps({sid: (value if index == 1 else chinese(value))
                               for index, (sid, value) in enumerate(values.items())})
        return json.dumps({sid: chinese(value) for sid, value in values.items()})

    deps = dependencies(lambda *a, **k: pytest.fail("legacy adapter called"), progress,
                        translate_payload=adapter)
    assert EPUBPipeline(deps).translate("t", str(source_epub), str(output))
    assert [len(call) for call in calls] == [3, 1]
    assert list(calls[1]) == [list(calls[0])[1]]
    assert prompts[0].startswith(SEGMENT_SYSTEM_PROMPT) and "术语表" in prompts[0]
    final = progress[-1]
    assert final["status"] == "completed"
    assert final["translated_segments"] == final["total_segments"] == 3
    assert final["untranslated_segments"] == final["failed_segment_count"] == 0
    assert final["coverage"] == 1.0
    with ZipFile(output) as archive:
        result = archive.read("OPS/chapter.xhtml").decode()
        assert markup_tokens(result) == markup_tokens(source)
        assert PROSE not in result


def test_original_book_offline_translation_preserves_all_twenty_markup_streams(tmp_path):
    book = Path(__file__).resolve().parents[2] / "Hyperscale.epub"
    if not book.exists():
        pytest.skip("local source book is not part of the test distribution")
    progress = []
    output = tmp_path / "offline-translated.epub"

    def adapter(payload, system_prompt):
        return json.dumps({sid: chinese(value) for sid, value in json.loads(payload).items()})

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(book), str(output))
    checked = 0
    with ZipFile(book) as source_archive, ZipFile(output) as translated_archive:
        for name in source_archive.namelist():
            if name.endswith(".xhtml"):
                source = source_archive.read(name).decode()
                rendered = translated_archive.read(name).decode()
                assert markup_tokens(rendered) == markup_tokens(source), name
                ET.fromstring(rendered)
                checked += 1
    assert checked == 20
    assert progress[-1]["untranslated_segments"] == 0
    assert progress[-1]["total_segments"] > 2000
    assert progress[-1]["skipped_files"] == []


FAILING = '<p class="note">An invented <i>second</i> sentence keeps failing every check.</p>'
ANCHORED = ('<p>The small harbour was quiet that morning<span id="n1"/> and every boat '
            'stayed near the old stone wall.</p>')
# Four good paragraphs first: the share cap needs at least 5 requested
# segments before a single failure fits the budget (int(0.2 * 5) == 1).
GOOD4 = "".join(f"<p>{PROSE}</p>" for _ in range(4))
CITATION = ('<a href="ch01.xhtml#n1"><i>quiet harbour</i></a> Jane Doe, “Small Boats Gather '
            'Near the Old Northern Harbour Wall Today,” Example Daily, May 2, 2020, '
            '<a href="https://example.com/news/boats">https://example.com/news/boats</a>.')
KEPT_TITLE = "Small Boats Gather Near the Old Northern Harbour Wall Today"


def kept_title(value):
    """Translate the invented citation but keep its English title (straight quotes)."""
    tokens = TOKEN_RE.findall(value)
    return (tokens[0] + tokens[1] + "安静的港口" + tokens[2] + tokens[3]
            + f' Jane Doe，"{KEPT_TITLE}"，Example Daily，2020年5月2日，' + "".join(tokens[4:]) + "。")


def chapter(tmp_path, body):
    source_epub, output = tmp_path / "source.epub", tmp_path / "translated.epub"
    write_epub(source_epub, xhtml(body))
    return source_epub, output


def published(output):
    with ZipFile(output) as archive:
        return archive.read("OPS/chapter.xhtml").decode()


def checkpoint_results(output):
    data = json.loads(Path(str(output) + ".segments.json").read_text(encoding="utf-8"))
    return data["results"]["OPS/chapter.xhtml"]


@pytest.mark.parametrize("mode", ["echo", "invalid_json", "exception"])
def test_pipeline_publishes_failed_segment_as_original_markup_within_budget(tmp_path, mode):
    progress, calls = [], []
    source_epub, output = chapter(tmp_path, GOOD4 + FAILING)

    def adapter(payload, system_prompt):
        values = json.loads(payload)
        calls.append(list(values))
        if "s000005" in values and mode == "exception":
            raise RuntimeError("provider down")
        if "s000005" in values and mode == "invalid_json":
            return "译文在这里，但没有 JSON。"
        return json.dumps({sid: (value if sid == "s000005" else chinese(value))
                           for sid, value in values.items()})

    assert EPUBPipeline(dependencies(adapter, progress, max_chunk_rounds=10)).translate(
        "t", str(source_epub), str(output))
    assert sum("s000005" in call for call in calls) == 4
    final = progress[-1]
    assert final["status"] == "completed"
    result = published(output)
    assert FAILING in result and f"<p>{PROSE}</p>" not in result
    assert markup_tokens(result) == markup_tokens(xhtml(GOOD4 + FAILING))
    assert [item["segment"] for item in final["fallback_segments"]] == ["OPS/chapter.xhtml#s000005"]
    assert final["fallback_segment_count"] == 1 and final["lenient_segment_count"] == 0
    assert final["warning"].startswith("1段未通过校验：1段保留原文")
    assert final["failed_segments"] == ["OPS/chapter.xhtml#s000005"]
    assert final["failed_segment_count"] == final["untranslated_segments"] == 1
    assert final["translated_segments"] == 4 and final["coverage"] == 0.8
    assert "s000005" not in checkpoint_results(output)


def test_pipeline_without_tolerance_budget_never_publishes_failed_source(tmp_path):
    progress, calls = [], []
    source_epub, output = chapter(tmp_path, f"<p>{PROSE}</p>")

    def adapter(payload, system_prompt):
        calls.append(payload)
        return payload

    deps = dependencies(adapter, progress, max_chunk_rounds=10,
                        max_fallback_segments=0, max_fallback_ratio=0)
    assert not EPUBPipeline(deps).translate("t", str(source_epub), str(output))
    assert len(calls) == 4
    assert not output.exists()
    assert all(p["status"] != "completed" for p in progress)
    assert progress[-1]["status"] == "failed"
    assert "(tolerance budget 0 = " in progress[-1]["error"]
    assert progress[-1]["untranslated_segments"] == progress[-1]["failed_segment_count"] == 1
    assert progress[-1]["failed_segments"] == ["OPS/chapter.xhtml#s000001"]
    assert progress[-1]["failure_reasons"]
    assert progress[-1]["coverage"] == 0
    assert progress[-1]["warning"] is None


@pytest.mark.parametrize("requested, failing, budget_text", [
    (500, 21, None),  # ratio term: budget max(20, int(0.05 * 500)) = 25
    (500, 26, "(tolerance budget 25 = min(max(20, 25), 100) for 500 requested segments)"),
    (10, 1, None),    # share cap: budget int(0.2 * 10) = 2
    (10, 3, "(tolerance budget 2 = min(max(20, 0), 2) for 10 requested segments)"),
    (4, 1, "(tolerance budget 0 = min(max(20, 0), 0) for 4 requested segments)"),
])
def test_tolerance_budget_formula(tmp_path, requested, failing, budget_text):
    progress = []
    source_epub, output = chapter(tmp_path, "".join(f"<p>{PROSE}</p>" for _ in range(requested)))
    bad = {f"s{index:06d}" for index in range(1, failing + 1)}

    def adapter(payload, **kwargs):
        return json.dumps({sid: (value if sid in bad else chinese(value))
                           for sid, value in json.loads(payload).items()})

    publishes = budget_text is None
    assert EPUBPipeline(dependencies(adapter, progress)).translate(
        "t", str(source_epub), str(output)) is publishes
    assert output.exists() is publishes
    final = progress[-1]
    assert final["failed_segment_count"] == failing
    if publishes:
        assert final["status"] == "completed" and final["fallback_segment_count"] == failing
    else:
        assert final["status"] == "failed" and budget_text in final["error"]


@pytest.mark.parametrize("extra", [{"max_fallback_segments": -1}, {"max_fallback_ratio": -0.1},
                                   {"max_fallback_share": -0.1}, {"max_fallback_share": 1.5}])
def test_invalid_fallback_budget_is_rejected(extra):
    with pytest.raises(ValueError):
        dependencies(lambda *a, **k: "", [], **extra)


def test_pipeline_accepts_kept_english_title_leniently_on_last_attempt(tmp_path):
    progress, calls = [], []
    source_epub, output = chapter(tmp_path, GOOD4 + f"<ol><li>{CITATION}</li></ol>")

    def adapter(payload, system_prompt):
        values = json.loads(payload)
        calls.append(list(values))
        return json.dumps({sid: (kept_title(value) if sid == "s000005" else chinese(value))
                           for sid, value in values.items()}, ensure_ascii=False)

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(source_epub), str(output))
    assert sum("s000005" in call for call in calls) == 4
    final = progress[-1]
    assert final["status"] == "completed"
    result = published(output)
    assert "安静的港口" in result and KEPT_TITLE in result
    assert '<a href="https://example.com/news/boats">https://example.com/news/boats</a>' in result
    assert final["lenient_segments"] == [{"segment": "OPS/chapter.xhtml#s000005",
                                          "reason": "untranslated long English passage"}]
    assert final["lenient_segment_count"] == 1 and final["fallback_segment_count"] == 0
    assert "1段接受质量存疑译文" in final["warning"]
    assert final["translated_segments"] == 4 and final["coverage"] == 0.8
    assert "s000005" not in checkpoint_results(output)


def test_quality_reject_from_first_attempt_survives_later_adapter_errors(tmp_path):
    progress, seen = [], []
    source_epub, output = chapter(tmp_path, GOOD4 + f"<ol><li>{CITATION}</li></ol>")

    def adapter(payload, system_prompt):
        values = json.loads(payload)
        if "s000005" in values:
            seen.append(1)
            if len(seen) > 1:
                raise RuntimeError("provider timeout")
        return json.dumps({sid: (kept_title(value) if sid == "s000005" else chinese(value))
                           for sid, value in values.items()}, ensure_ascii=False)

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(source_epub), str(output))
    assert len(seen) == 4
    final = progress[-1]
    assert [item["segment"] for item in final["lenient_segments"]] == ["OPS/chapter.xhtml#s000005"]
    assert final["lenient_segments"][0]["reason"] == "untranslated long English passage"
    assert final["failure_reasons"][0]["reason"].startswith("translation adapter failed")
    assert "安静的港口" in published(output)


def test_coverage_failure_falls_back_to_source_not_lenient(tmp_path):
    progress = []
    long_paragraph = "<p>" + PROSE * 5 + "</p>"
    source_epub, output = chapter(tmp_path, GOOD4 + long_paragraph)

    def adapter(payload, system_prompt):
        return json.dumps({sid: ("好的。" if sid == "s000005" else chinese(value))
                           for sid, value in json.loads(payload).items()})

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(source_epub), str(output))
    final = progress[-1]
    result = published(output)
    assert long_paragraph in result and "好的" not in result
    assert final["lenient_segment_count"] == 0
    assert final["fallback_segments"][0]["reason"].startswith("insufficient Chinese coverage")


def test_assistant_explanation_is_never_accepted_leniently(tmp_path):
    progress = []
    source_epub, output = chapter(tmp_path, GOOD4 + f"<p>{PROSE} Again.</p>")

    def adapter(payload, system_prompt):
        values = json.loads(payload)
        return json.dumps({sid: ("以下是翻译：" + CHINESE if sid == "s000005" else chinese(value))
                           for sid, value in values.items()})

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(source_epub), str(output))
    final = progress[-1]
    result = published(output)
    assert "以下是翻译" not in result
    assert f"<p>{PROSE} Again.</p>" in result
    assert final["lenient_segment_count"] == 0
    assert [item["segment"] for item in final["fallback_segments"]] == ["OPS/chapter.xhtml#s000005"]
    assert final["fallback_segments"][0]["reason"] == "assistant explanation in translation"


def test_repaired_anchor_is_accepted_and_counted_end_to_end(tmp_path):
    progress = []
    source_epub, output = chapter(tmp_path, GOOD4 + ANCHORED)

    def adapter(payload, system_prompt):
        return json.dumps({sid: ("那天早上小港口很安静，每条船都停在旧石墙附近。" if sid == "s000005"
                                 else chinese(value)) for sid, value in json.loads(payload).items()})

    assert EPUBPipeline(dependencies(adapter, progress)).translate("t", str(source_epub), str(output))
    final = progress[-1]
    assert final["repaired_segment_count"] == 1
    assert final["failed_segment_count"] == 0 and final["warning"] is None
    assert final["translated_segments"] == 5 and final["coverage"] == 1.0
    assert published(output).count('<span id="n1"/>') == 1


def test_identity_text_in_checkpoint_is_retried_and_clears_warning(tmp_path):
    progress = []
    source_epub, output = chapter(tmp_path, GOOD4 + FAILING)
    echo_last = lambda payload, **kwargs: json.dumps(
        {sid: (value if sid == "s000005" else chinese(value)) for sid, value in json.loads(payload).items()})
    assert EPUBPipeline(dependencies(echo_last, progress)).translate("t", str(source_epub), str(output))
    assert progress[-1]["warning"]
    # Even a checkpoint that holds the source text for a requested segment
    # (older release, manual edit) cannot turn the fallback into a result.
    checkpoint_path = Path(str(output) + ".segments.json")
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    segment = extract_segments(xhtml(GOOD4 + FAILING)).segments[4]
    data["results"]["OPS/chapter.xhtml"][segment.id] = segment.source
    checkpoint_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    calls = []

    def good(payload, **kwargs):
        values = json.loads(payload)
        calls.append(list(values))
        return json.dumps({sid: chinese(value) for sid, value in values.items()})

    assert EPUBPipeline(dependencies(good, progress)).translate("t", str(source_epub), str(output))
    assert calls == [["s000005"]]
    final = progress[-1]
    assert final["reused_segments"] == 4
    assert final["warning"] is None and final["failed_segment_count"] == 0
    assert final["fallback_segments"] == [] and final["coverage"] == 1.0
    assert FAILING not in published(output)


@pytest.mark.parametrize("content", [
    "<html><body>Unclosed</html>",
    xhtml(f"<p>{PROSE}</p>") + "说明：翻译完成",
    xhtml("<p>Non UTF-8 source</p>").encode("utf-16"),
])
def test_pipeline_rejects_invalid_or_non_utf8_sources_before_translation(tmp_path, content):
    progress = []
    source_epub, output = tmp_path / "source.epub", tmp_path / "translated.epub"
    write_epub(source_epub, content)
    deps = dependencies(lambda *a, **k: pytest.fail("invalid source was sent"), progress)
    assert not EPUBPipeline(deps).translate("t", str(source_epub), str(output))
    assert progress[-1]["status"] == "failed"
    assert "OPS/chapter.xhtml" in progress[-1]["error"]
    assert not output.exists()


def test_pipeline_empty_and_numeric_files_are_recorded_without_api_calls(tmp_path):
    progress = []
    (tmp_path / "empty.xhtml").write_text(xhtml("<section><hr/></section>"))
    (tmp_path / "numbers.xhtml").write_text(xhtml("<p>123</p>"))
    deps = dependencies(lambda *a, **k: pytest.fail("unneeded request"), progress)
    total, requests, completed = EPUBPipeline(deps)._translate_extracted("t", tmp_path)
    assert total == 2 and requests == 0
    assert completed == {"numbers.xhtml"}
    assert progress[-1]["skipped_files"] == ["empty.xhtml"]
    assert progress[-1]["translated_segments"] == 0
    assert progress[-1]["preserved_segments"] == 1
    assert progress[-1]["untranslated_segments"] == 0


@pytest.mark.parametrize("body", ["<h2>E</h2>", "<p>A.</p>", '<h2 class="letter">M</h2>'])
def test_single_letter_segments_are_preserved_without_request(tmp_path, body):
    source = xhtml(f"<section>{body}</section>")
    document = extract_segments(source)
    assert len(document.segments) == 1 and not document.segments[0].needs_request
    assert document.render({s.id: s.source for s in document.segments}) == source
    (tmp_path / "index.xhtml").write_text(source, encoding="utf-8")
    progress = []
    deps = dependencies(lambda *a, **k: pytest.fail("single letters need no request"), progress)
    total, requests, completed = EPUBPipeline(deps)._translate_extracted("t", tmp_path)
    assert (total, requests, completed) == (1, 0, {"index.xhtml"})
    assert (tmp_path / "index.xhtml").read_text(encoding="utf-8") == source
    assert progress[-1]["preserved_segments"] == 1


def test_glossary_input_still_includes_single_letter_headings(tmp_path):
    import dataclasses
    (tmp_path / "index.xhtml").write_text(xhtml(f"<h2>E</h2><p>{PROSE}</p>"), encoding="utf-8")
    seen = []
    deps = dataclasses.replace(
        dependencies(lambda payload, **k: json.dumps(
            {sid: chinese(value) for sid, value in json.loads(payload).items()}), []),
        build_glossary=lambda text, task_id: seen.append(text) or "")
    total, requests, completed = EPUBPipeline(deps)._translate_extracted("t", tmp_path)
    assert seen == ["E\n" + PROSE]
    # The preserved letter renders next to translated entries in the same file.
    assert (requests, completed) == (1, {"index.xhtml"})
    rendered = (tmp_path / "index.xhtml").read_text(encoding="utf-8")
    assert "<h2>E</h2>" in rendered and CHINESE in rendered and PROSE not in rendered


def test_dropped_invisible_anchor_is_reinserted_once():
    document = extract_segments(xhtml(ANCHORED))
    segment = document.segments[0]
    verdict = parse_translation_response(
        json.dumps({segment.id: "那天早上小港口很安静，每条船都停在旧石墙附近。"}), [segment])
    assert verdict.repaired == {segment.id}
    translated = verdict.translations[segment.id]
    assert TOKEN_RE.findall(translated) == [segment.fragments[0].token]
    rendered = document.render({segment.id: translated})
    assert rendered.count('<span id="n1"/>') == 1
    assert markup_tokens(rendered) == markup_tokens(xhtml(ANCHORED))


def test_swapped_invisible_anchors_are_put_back_in_order():
    segment = extract_segments(xhtml(
        '<p>The first invented clause ends here<a id="p1"/> and a second one '
        'follows<span id="n2"/> before the end.</p>')).segments[0]
    first, second = [fragment.token for fragment in segment.fragments]
    repaired = repair_invisible_anchors(segment, f"第一个虚构分句到此{second}结束，第二个{first}随后。")
    assert TOKEN_RE.findall(repaired) == [first, second]
    validate_segment_translation(segment, repaired)


def test_dropped_visible_fragment_is_not_repaired():
    segment = extract_segments(xhtml(
        '<p>The boats stayed <i>close</i> to the wall<span id="n1"/> all through the night.</p>')).segments[0]
    opening, closing, anchor = [fragment.token for fragment in segment.fragments]
    response = f"那些船整夜都{anchor}紧靠着墙停泊。"
    assert repair_invisible_anchors(segment, response) == response
    verdict = parse_translation_response(json.dumps({segment.id: response}), [segment])
    assert verdict.translations == {} and verdict.repaired == set()
    assert verdict.failures[segment.id] == "placeholder count/order changed"
    assert verdict.quality_rejects == {}


def test_anchor_with_href_is_not_invisible():
    segment = extract_segments(xhtml(
        '<p>Read the invented note here<a href="#n" id="x"/> before you continue reading.</p>')).segments[0]
    response = "继续阅读之前，请先读这里的虚构注释。"
    assert repair_invisible_anchors(segment, response) == response
    with pytest.raises(SegmentValidationError, match="placeholder"):
        validate_segment_translation(segment, response)


def test_empty_anchor_pair_is_reinserted_together():
    segment = extract_segments(xhtml(
        '<p>The quiet invented story begins<a id="x"></a> on a cold winter evening.</p>')).segments[0]
    opening, closing = [fragment.token for fragment in segment.fragments]
    repaired = repair_invisible_anchors(segment, f"这个安静的虚构故事{closing}开始于一个寒冷的冬夜。")
    assert TOKEN_RE.findall(repaired) == [opening, closing]
    assert opening + closing in repaired


def test_invisible_anchor_placement_is_proportional_between_neighbours():
    segment = extract_segments(xhtml(
        '<p>Alpha words here<span id="n1"/> then <i>beta</i> and more gamma words'
        '<a id="p2"/> at the end.</p>')).segments[0]
    n1, opening, closing, p2 = [fragment.token for fragment in segment.fragments]
    repaired = repair_invisible_anchors(segment, f"阿尔法词在这里，然后{opening}贝塔{closing}和更多伽马词在最后。")
    assert repaired == f"阿尔法词在这里{n1}，然后{opening}贝塔{closing}和更多伽马词{p2}在最后。"


def test_citation_keeping_english_title_is_a_quality_reject_only():
    segment = extract_segments(xhtml(
        '<ol><li><a href="ch01.xhtml#n1"><i>quiet harbour</i></a> Jane Doe, “Small Boats Gather '
        'Near the Old Northern Harbour Wall Today,” Example Daily, May 2, 2020, '
        '<a href="https://example.com/news/boats">https://example.com/news/boats</a>.</li></ol>')).segments[0]
    tokens = [fragment.token for fragment in segment.fragments]
    response = (tokens[0] + tokens[1] + "安静的港口" + tokens[2] + tokens[3]
                + " Jane Doe，“Small Boats Gather Near the Old Northern Harbour Wall Today”，"
                + "Example Daily，2020年5月2日，" + "".join(tokens[4:]) + "。")
    verdict = parse_translation_response(json.dumps({segment.id: response}, ensure_ascii=False), [segment])
    assert verdict.failures[segment.id] == "untranslated long English passage"
    assert verdict.quality_rejects == {segment.id: (response, "untranslated long English passage")}
    document = extract_segments(xhtml(
        '<ol><li><a href="ch01.xhtml#n1"><i>quiet harbour</i></a> Jane Doe, “Small Boats Gather '
        'Near the Old Northern Harbour Wall Today,” Example Daily, May 2, 2020, '
        '<a href="https://example.com/news/boats">https://example.com/news/boats</a>.</li></ol>'))
    with pytest.raises(SegmentValidationError):
        document.render({segment.id: response})
    assert "https://example.com/news/boats</a>" in document.render(
        {segment.id: response}, lenient_ids={segment.id})


@pytest.mark.parametrize("response", [
    "I'm sorry, but I can't help with translating this passage.",
    "好的。",
    CHINESE + " Note: I kept some words in English because they could not be translated well.",
    CHINESE + "（Here is the translation of the sentence you asked me for.）",
])
def test_refusals_truncations_and_added_english_are_never_quality_rejects(response):
    segment = extract_segments(xhtml("<p>" + PROSE * 5 + "</p>")).segments[0]
    verdict = parse_translation_response(json.dumps({segment.id: response}, ensure_ascii=False), [segment])
    assert verdict.translations == {} and verdict.quality_rejects == {}
    assert segment.id in verdict.failures


def test_lenient_render_still_rejects_placeholder_change():
    document = extract_segments(xhtml(f"<ol><li>{CITATION}</li></ol>"))
    segment = document.segments[0]
    broken = kept_title(segment.source).replace(segment.fragments[1].token, "", 1)
    with pytest.raises(SegmentValidationError, match="placeholder"):
        document.render({segment.id: broken}, lenient_ids={segment.id})


def test_styled_empty_element_is_not_invisible_but_class_alone_is():
    styled = extract_segments(xhtml(
        '<p>The invented story pauses here<span style="display:inline-block" id="s1"/> for a short '
        'while.</p>')).segments[0]
    response = "这个虚构的故事在这里短暂停顿。"
    assert repair_invisible_anchors(styled, response) == response
    classed = extract_segments(xhtml(
        '<p>The invented story pauses here<span class="marker"/> for a short while.</p>')).segments[0]
    assert TOKEN_RE.findall(repair_invisible_anchors(classed, response)) == [classed.fragments[0].token]
