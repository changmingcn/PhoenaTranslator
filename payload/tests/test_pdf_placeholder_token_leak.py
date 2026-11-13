"""Raw internal placeholder tokens never reach a delivered page.

A model answer can drop or mangle one of an element's tokens, or a batch
answer can move a neighbour's tokens into another element's segment (that
element has no records of its own, so nothing is restored).  Translation
validation rejects such a candidate for its element; assembly never draws a
token that still arrives (an older page cache).

Synthetic fixtures only.  Run from ``payload``: ``python -m pytest tests -q``.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from test_pdf_figure_table_fallback import (
    ZH,
    Document,
    _body,
    _squash,
    _text,
    font_path,  # noqa: F401  (module fixture)
)
from test_pdf_render_element_fallback import _assert_kept_alone, _kept_records

from phoena_translator.pdf.context import PDFPageTranslationContext
from phoena_translator.pdf.errors import PDFTranslationIntegrityError
from phoena_translator.pdf.extract import stage as extract_stage
from phoena_translator.pdf.model import SkipReason
from phoena_translator.pdf.pipeline import _new_pdf_audit
from phoena_translator.pdf.translate import stage as translate_stage
from phoena_translator.pdf.translate.batch_codec import _build_pdf_batch_segment_envelope
from phoena_translator.pdf.translate.placeholders import (
    PDF_PLACEHOLDER_KINDS,
    PDF_PLACEHOLDER_LEAK_REASON,
    _protect_pdf_identifier_fragments,
    _protect_pdf_inline_math_fragments,
    _protect_pdf_superscript_fragments,
    pdf_internal_placeholder_leaks,
    pdf_placeholder_token,
)
from phoena_translator.pdf.translate.validation import _validate_pdf_page_translations

log = logging.getLogger("test-placeholder-leak")
TOKEN_RE = re.compile(r"PHOENA", re.IGNORECASE)


def _minted_tokens() -> dict[str, str]:
    """One token of every kind the code mints, taken from the minting code."""
    _, sup = _protect_pdf_superscript_fragments("Revenue rose<sup>1</sup> in the quarter")
    _, imath = _protect_pdf_inline_math_fragments(
        "priced ~ 40 units", [{"text": "~", "font_kind": "unicode"}]
    )
    _, ident = _protect_pdf_identifier_fragments("see https://example.org/report for details")
    _, seg = _build_pdf_batch_segment_envelope([(0, "First paragraph"), (1, "Second one")])
    return {
        "SUP": sup[0]["token"],
        "IMATH": imath[0]["token"],
        "ID": ident[0]["token"],
        "SEG": seg[1]["token"],
    }


@pytest.mark.parametrize("kind", ["SUP", "IMATH", "ID", "SEG"])
def test_every_kind_of_minted_token_is_a_leak(kind):
    token = _minted_tokens()[kind]
    assert token.startswith(f"PHOENA_{kind}_")
    assert kind in PDF_PLACEHOLDER_KINDS
    assert pdf_internal_placeholder_leaks("Plain English source", f"中文译文{token}其余") == [token]


def test_a_kind_outside_the_generators_set_cannot_be_minted():
    # The guard recognises exactly the generator's kinds, so a new kind must
    # join that set before any token of it exists.
    with pytest.raises(ValueError):
        pdf_placeholder_token("LATER", 7, "0123456789ABCDEF")


@pytest.mark.parametrize(
    "mangled",
    [
        "phoena_imath_0000_cbae14bec289ee7a",  # case changed
        "PHOENA IMATH 0001 5A3D",  # spaces, truncated digest
        "PHOENAIMATH0002",  # separators dropped
    ],
)
def test_a_mangled_token_is_still_a_leak(mangled):
    assert pdf_internal_placeholder_leaks("Plain English source", f"译文{mangled}译文")


def test_clean_text_and_source_owned_text_are_not_leaks():
    token = _minted_tokens()["IMATH"]
    assert pdf_internal_placeholder_leaks("Plain English source", "干净的中文译文") == []
    # The source itself contains the string: copying it is not a leak.
    assert pdf_internal_placeholder_leaks(f"Field {token} is reserved", f"字段{token}保留") == []
    # The brand word alone is not a token.
    assert pdf_internal_placeholder_leaks("Plain source", "Phoena 翻译器") == []


# ---------------------------------------------------------------------------
# Layer 1: translation validation
# ---------------------------------------------------------------------------

BULLET_A = "Farmers planted ~ 40 hectares of new orchards across three valleys this spring"
BULLET_B = "showing that local growers still have room to expand their orchards in a dry year"
ZH_A_BODY = ("今年春天农户在三个山谷里新种了", "40公顷的果园")
ZH_B = "显示即便在干旱的年份里，当地种植户仍然有扩大果园的余地"


def _page_elements():
    return [
        {
            "type": "text",
            "layout_class": "body",
            "content": BULLET_A,
            "bbox": [72.0, 100.0, 540.0, 112.0],
            "fontsize": 10.0,
            "inline_math_fragments": [{"text": "~", "font_kind": "unicode"}],
        },
        {
            "type": "text",
            "layout_class": "body",
            "content": BULLET_B,
            "bbox": [72.0, 112.0, 540.0, 124.0],
            "fontsize": 10.0,
        },
    ]


def _leaking_provider(calls):
    """A batch answer moves element 0's formula token into element 1's segment;
    individual answers keep echoing that token for element 1."""
    seen = {}

    def translate_text(text, system_prompt=""):
        calls.append(text)
        match = re.search(r"PHOENA_IMATH_\d{4}_[0-9A-F]{16}", text)
        if match:
            seen["imath"] = match.group(0)
        token = seen.get("imath", "PHOENA_IMATH_0000_0000000000000000")
        if "PHOENA_SEG_" in text:
            seg = re.findall(r"PHOENA_SEG_\d{4}_[0-9A-F]{16}", text)
            return (
                f"{seg[0]}\n{ZH_A_BODY[0]}{ZH_A_BODY[1]}\n≡≡≡SPLIT≡≡≡\n"
                f"{seg[1]}\n{ZH_B}{token}。"
            )
        if match:  # element 0 alone: a correct answer
            return f"{ZH_A_BODY[0]}{token}{ZH_A_BODY[1]}。"
        return f"{ZH_B}{token}。"

    return translate_text


def _translation_stage(elements, merge_decisions=()):
    audit = _new_pdf_audit("t", 0.6, [])
    audit["merge_decisions"] = list(merge_decisions)
    # Pages 2 and 3 were translated earlier (cached); only page 1 is sent.
    return translate_stage.PDFTranslationStageContext(
        task_id="t",
        total_pages=3,
        source_sha256="0" * 64,
        workers=1,
        fail_open_to_source_page=True,
        logger=log,
        page_extractions={
            0: {"elements": elements},
            1: {"elements": [], "cached": {}},
            2: {"elements": [], "cached": {}},
        },
        pdf_audit=audit,
        completed_indices=set(),
        save_translation_progress=lambda *args, **kwargs: None,
    )


def _run_translation(monkeypatch, merge_decisions=()):
    saved = {}
    monkeypatch.setattr(
        translate_stage,
        "_save_pdf_page_translation_cache",
        lambda _task, page, translations, _sha, elements=None: saved.update(
            {page: dict(translations)}
        ),
    )
    elements = _page_elements()
    context = _translation_stage(elements, merge_decisions)
    calls = []
    page_context = PDFPageTranslationContext(
        task_id="t", prompt="翻译", logger=log, translate_text=_leaking_provider(calls)
    )
    translate_stage.translate_pages(context, page_context, [0])
    return context, elements, saved, calls


def test_a_batch_answer_that_moves_a_token_keeps_that_element_in_source(monkeypatch):
    context, elements, saved, calls = _run_translation(monkeypatch)

    committed = saved[0]
    assert not any(TOKEN_RE.search(str(value)) for value in committed.values())
    # Element 0 restored its own token; element 1 is the exact source text.
    assert committed[0] == f"{ZH_A_BODY[0]}~{ZH_A_BODY[1]}。"
    assert committed[1] == BULLET_B
    assert elements[1]["skip_translate_reason"] == SkipReason.TRANSLATION_INTEGRITY_FALLBACK
    (entry,) = context.pdf_audit["element_source_fallbacks"]
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "translation")
    assert entry["reason"] == PDF_PLACEHOLDER_LEAK_REASON
    assert context.pdf_audit["source_page_fallbacks"] == []
    # Retried as today: individually after the batch, then one clean page retry.
    assert len(calls) > 2


def test_a_leaking_merge_endpoint_keeps_the_endpoint_rule(monkeypatch):
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 0,
    }
    context, _elements, saved, _calls = _run_translation(monkeypatch, [merge])

    assert saved == {}
    assert [entry["page"] for entry in context.pdf_audit["source_page_fallbacks"]] == [1, 2]
    assert context.pdf_audit.get("element_source_fallbacks", []) == []


def test_a_batch_segment_holding_a_neighbours_token_is_retried_individually():
    from phoena_translator.pdf.translate import page

    element = _page_elements()[1]
    token = _minted_tokens()["IMATH"]
    context = SimpleNamespace(task_id="t", logger=log)
    # No records of its own: every restorer returns the text unchanged.
    assert page._restore_batch_part(
        context, 0, element, 1, BULLET_B, f"{ZH_B}{token}。", ((), (), ())
    ) is None
    assert page._restore_batch_part(
        context, 0, element, 1, BULLET_B, f"{ZH_B}。", ((), (), ())
    ) == f"{ZH_B}。"


def test_an_individual_answer_holding_a_token_gets_the_strict_marker_retry():
    from phoena_translator.pdf.translate import page

    element = _page_elements()[1]
    token = _minted_tokens()["ID"]
    failures = page._translation_integrity_failures(BULLET_B, element, f"{ZH_B}{token}。", ())
    assert PDF_PLACEHOLDER_LEAK_REASON in failures
    assert page._translation_integrity_failures(BULLET_B, element, f"{ZH_B}。", ()) == ()


def test_page_validation_names_the_leak_and_the_cache_rebind_defers_it():
    elements = _page_elements()[1:]
    leaking = {0: f"{ZH_B}{_minted_tokens()['ID']}。"}
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, elements, leaking)
    assert raised.value.issues == [{"element": 1, "reason": PDF_PLACEHOLDER_LEAK_REASON}]
    _validate_pdf_page_translations(1, elements, leaking, reject_placeholder_leaks=False)


def test_an_older_page_cache_with_a_token_is_kept_for_assembly(monkeypatch, tmp_path):
    elements = _page_elements()
    leaking = {"0": f"{ZH_A_BODY[0]}~{ZH_A_BODY[1]}。", "1": f"{ZH_B}{_minted_tokens()['IMATH']}。"}
    cache_file = tmp_path / "page_0.json"
    cache_file.write_text("{}")
    discarded = []
    monkeypatch.setattr(extract_stage, "_pdf_page_cache_path", lambda _task, _page: str(cache_file))
    monkeypatch.setattr(
        extract_stage, "_load_pdf_page_translation_cache", lambda *_a, **_k: dict(leaking)
    )
    monkeypatch.setattr(
        extract_stage, "_discard_pdf_page_translation_cache", lambda *a: discarded.append(a)
    )
    monkeypatch.setattr(extract_stage, "_save_pdf_page_translation_cache", lambda *a, **k: None)
    result = SimpleNamespace(
        completed_indices={0},
        cached_indices={0},
        page_extractions={0: {"elements": elements}},
        source_sha256="0" * 64,
        total_pages=1,
    )
    extract_stage._bind_and_reconcile_caches(SimpleNamespace(task_id="t", logger=log), result)

    assert discarded == []
    assert result.completed_indices == {0}
    assert result.page_extractions[0]["cached"]["1"] == leaking["1"]


# ---------------------------------------------------------------------------
# Layer 2: assembly safety net
# ---------------------------------------------------------------------------


def _leaking_cached_page():
    kept = _text([72, 200, 540, 212], "Showing that growers have room to expand", layout_class="body")
    return [_body(100), kept, _body(400)], kept


def test_a_cached_token_is_never_drawn_and_the_element_keeps_its_source_ink(tmp_path, font_path):
    elements, kept = _leaking_cached_page()
    token = _minted_tokens()["IMATH"]
    document = Document(
        tmp_path, font_path, [(elements, {0: ZH[:30], 1: f"显示{token}余地", 2: ZH[5:30]})]
    )

    (text,) = document.assemble()

    assert not TOKEN_RE.search(text)
    assert document.fallback_pages() == []
    assert ZH[:30] in _squash(text) and ZH[5:30] in _squash(text)
    assert text.count("Showing that growers") == 1
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert (entry["page"], entry["element"], entry["stage"]) == (1, 1, "render")
    assert entry["reason"].startswith(PDF_PLACEHOLDER_LEAK_REASON)
    assert "translation withdrawn" in entry["reason"]
    assert kept["skip_translate_reason"] == SkipReason.RENDER_ELEMENT_SOURCE


def test_a_cached_token_on_a_merge_endpoint_takes_the_merge_closure(tmp_path, font_path):
    elements, _kept = _leaking_cached_page()
    token = _minted_tokens()["SUP"]
    merge = {
        "decision": "accepted",
        "source_page": 1,
        "source_element": 1,
        "destination_page": 2,
        "destination_element": 0,
    }
    document = Document(
        tmp_path,
        font_path,
        [
            (elements, {0: ZH[:30], 1: f"显示{token}余地", 2: ZH[5:30]}),
            ([_body(100), _body(300)], {0: ZH[6:30], 1: ZH[9:30]}),
        ],
        merge_decisions=[merge],
    )

    first, second = document.assemble()

    assert document.fallback_pages() == [1, 2]
    assert not TOKEN_RE.search(first + second)
    assert _kept_records(document) == []


# ---------------------------------------------------------------------------
# Review round 1 (R2): the guard reads the text as it will be drawn
# ---------------------------------------------------------------------------

SOURCE_TOKEN = "PHOENA_IMATH_0000_AAAAAAAAAAAAAAAA"
SOURCE_OWNING = f"The diagnostic field {SOURCE_TOKEN} is documented here for reference"
ORDINARY_SOURCE = "The second version of the Phoena Model is now available for use"
ORDINARY_ZH = "Phoena Model 2 已经发布，并且现在可以正常使用。"

# (a) spellings a model produces and (b) markup or an entity hiding a token.
HIDDEN_TOKENS = {
    "letter-spaced": "P HOENA_IMATH_0000_AB",
    "hyphenated": "PHOENA-IMATH-0000-ABCD",
    "full-width-underscore": "PHOENA＿IMATH＿0000",
    "full-width-prefix": "ＰＨＯＥＮＡ_IMATH_0000",
    "zero-width": "PHOENA_​IMATH_0000",
    "spaced-kind": "PHOENA_I MATH_0000",
    "cut-after-kind": "PHOENA_IMATH",
    "no-index": "PHOENA_IMATH_ABCDEF",
    "cut-after-separator": "PHOENA_SUP_",
    "bold-markup": "PHOENA_<b>IMATH</b>_0000_0123456789ABCDEF",
    "entity": "PHOENA&#95;IMATH_0000_0123456789ABCDEF",
}
# (c) the source owns one token; the candidate keeps it and adds another.
FOREIGN_NEXT_TO_OWNED = f"诊断字段{SOURCE_TOKEN}与PHOENA_IMATH_0001_BBBBBBBBBBBBBBBB在此说明"
OWNED_ONLY = f"诊断字段{SOURCE_TOKEN}在此说明，仅供参考"

# Review round 2 (T1): the class is every candidate whose drawn text holds a
# recognisable token, whole or cut off.  The guard reads the text the
# renderer draws (``_PDFInlineHTMLParser`` drops comments, processing
# instructions and declarations; entities are decoded once), and only a
# following Latin letter makes a kind part of a word.
DRAWN_TOKENS = {
    "cut-kind-before-cjk": f"{ZH_B}PHOENA_IMATH其余说明。",
    "cut-id-before-cjk": f"{ZH_B}PHOENA_ID其余说明。",
    "cut-index-before-cjk": f"{ZH_B}PHOENA_SUP_00其余说明。",
    "cut-digest-before-cjk": f"{ZH_B}PHOENA_SEG_0001_AB其余说明。",
    "cut-before-comma": f"{ZH_B}PHOENA_ID，其余说明。",
    "cut-before-bracket": f"{ZH_B}（PHOENA_IMATH）其余说明。",
    "cut-at-end": f"{ZH_B}PHOENA_ID",
    "comment": f"{ZH_B}PHOE<!-- -->NA_IMATH_0000_0123456789ABCDEF。",
    "processing-instruction": f"{ZH_B}PHOENA<?x?>_IMATH_0000_0123456789ABCDEF。",
    "declaration": f"{ZH_B}PHOENA_<!x>IMATH_0000_0123456789ABCDEF。",
    "nested-markup": f"{ZH_B}PHOE<b>NA_<sup>IM</sup>ATH</b>_0000_0123456789ABCDEF。",
    "hex-entity": f"{ZH_B}PHOENA&#x5F;IMATH&#95;0000。",
    "named-entity": f"{ZH_B}PHOENA&lowbar;SEG&lowbar;0001。",
    "comment-markup-cut-before-cjk": f"{ZH_B}PHOE<!-- c -->NA<b>_IM</b>ATH其余说明。",
    "pi-entity-cut-before-cjk": f"{ZH_B}PHOENA<?p?>&#95;ID其余说明。",
    "comment-entity-nested-cut-before-comma": f"{ZH_B}P<!---->HOENA&#95;<b>S<sup>U</sup>P</b>，其余。",
}
# Words that merely start like a kind stay ordinary text.
WORD_SOURCE = "The Phoena IDE plugin and the Phoena Supply desk are now available"
WORD_ZH = "Phoena IDE 插件与 Phoena Supply 柜台现已可以正常使用。"

# Review round 2 (T2): a token is exempt only when the identical complete
# token (as the generator writes it) occurs in the element's own source.
# Each case below is a different token that a truncated or substring
# comparison with the owned token exempts.
OWNED_FIVE_DIGIT = pdf_placeholder_token("IMATH", 10000, "A" * 16)
OTHER_FIVE_DIGIT = pdf_placeholder_token("IMATH", 10000, "B" * 16)
SOURCE_OWNING_FIVE_DIGIT = f"The diagnostic field {OWNED_FIVE_DIGIT} is documented here for reference"
SPLIT_DIGEST = "PHOENA_IMATH_0000_AAAAAAAA_AAAAAAAA"
SOURCE_OWNING_SPLIT = f"The diagnostic field {SPLIT_DIGEST} is documented here for reference"
NOT_IDENTICAL = {
    "owned-token-cut-off": (SOURCE_OWNING, "诊断字段PHOENA_IMATH_0000在此说明，仅供参考"),
    "five-digit-index-other-digest": (
        SOURCE_OWNING_FIVE_DIGIT, f"诊断字段{OTHER_FIVE_DIGIT}在此说明，仅供参考"
    ),
    "split-digest-other-tail": (
        SOURCE_OWNING_SPLIT, "诊断字段PHOENA_IMATH_0000_AAAAAAAA_AAAAAAAB在此说明，仅供参考"
    ),
}

LEAK_CASES = [
    *[(BULLET_B, f"{ZH_B}{spelling}。") for spelling in HIDDEN_TOKENS.values()],
    (SOURCE_OWNING, FOREIGN_NEXT_TO_OWNED),
    *[(BULLET_B, candidate) for candidate in DRAWN_TOKENS.values()],
    *NOT_IDENTICAL.values(),
]
LEAK_IDS = [*HIDDEN_TOKENS, "foreign-next-to-owned", *DRAWN_TOKENS, *NOT_IDENTICAL]
CLEAN_CASES = [
    (ORDINARY_SOURCE, ORDINARY_ZH),
    (SOURCE_OWNING, OWNED_ONLY),
    (WORD_SOURCE, WORD_ZH),
    (SOURCE_OWNING_FIVE_DIGIT, f"诊断字段{OWNED_FIVE_DIGIT}在此说明，仅供参考"),
]
CLEAN_IDS = [
    "product-and-version",
    "owned-token-only",
    "words-starting-like-a-kind",
    "owned-five-digit-token",
]


def _single(source):
    return [{"type": "text", "layout_class": "body", "content": source,
             "bbox": [72.0, 112.0, 540.0, 124.0], "fontsize": 10.0}]


@pytest.mark.parametrize(("source", "candidate"), LEAK_CASES, ids=LEAK_IDS)
def test_a_hidden_or_foreign_token_is_a_leak(source, candidate):
    assert pdf_internal_placeholder_leaks(source, candidate)


@pytest.mark.parametrize(("source", "candidate"), CLEAN_CASES, ids=CLEAN_IDS)
def test_ordinary_text_and_an_owned_token_are_not_leaks(source, candidate):
    assert pdf_internal_placeholder_leaks(source, candidate) == []


@pytest.mark.parametrize(("source", "candidate"), LEAK_CASES, ids=LEAK_IDS)
def test_validation_rejects_a_hidden_or_foreign_token(source, candidate):
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, _single(source), {0: candidate})
    assert raised.value.issues == [{"element": 1, "reason": PDF_PLACEHOLDER_LEAK_REASON}]


@pytest.mark.parametrize(("source", "candidate"), CLEAN_CASES, ids=CLEAN_IDS)
def test_validation_accepts_ordinary_text_and_an_owned_token(source, candidate):
    _validate_pdf_page_translations(1, _single(source), {0: candidate})


@pytest.mark.parametrize(("source", "candidate"), LEAK_CASES, ids=LEAK_IDS)
def test_the_batch_part_net_rejects_a_hidden_or_foreign_token(source, candidate):
    from phoena_translator.pdf.translate import page

    context = SimpleNamespace(task_id="t", logger=log)
    element = _single(source)[0]
    assert page._restore_batch_part(context, 0, element, 1, source, candidate, ((), (), ())) is None
    assert PDF_PLACEHOLDER_LEAK_REASON in page._translation_integrity_failures(
        source, element, candidate, ()
    )


@pytest.mark.parametrize(("source", "candidate"), CLEAN_CASES, ids=CLEAN_IDS)
def test_the_batch_part_net_accepts_ordinary_text_and_an_owned_token(source, candidate):
    from phoena_translator.pdf.translate import page

    context = SimpleNamespace(task_id="t", logger=log)
    element = _single(source)[0]
    assert page._restore_batch_part(
        context, 0, element, 1, source, candidate, ((), (), ())
    ) == candidate
    assert page._translation_integrity_failures(source, element, candidate, ()) == ()


def _drawn_tokens(text):
    import unicodedata

    visible = re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text))
    return re.findall(r"PHOENA(?:IMATH|SUP|SEG|ID)[0-9A-F]*", visible)


@pytest.mark.parametrize(("source", "candidate"), LEAK_CASES, ids=LEAK_IDS)
def test_assembly_never_draws_a_hidden_or_foreign_token(tmp_path, font_path, source, candidate):
    kept = _text([72, 200, 540, 212], source, layout_class="body")
    document = Document(
        tmp_path, font_path, [([_body(100), kept, _body(400)], {0: ZH[:30], 1: candidate, 2: ZH[5:30]})]
    )

    (text,) = document.assemble()

    assert _drawn_tokens(text) == _drawn_tokens(source)
    assert document.fallback_pages() == []
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert entry["reason"].startswith(PDF_PLACEHOLDER_LEAK_REASON)


@pytest.mark.parametrize(("source", "candidate"), CLEAN_CASES, ids=CLEAN_IDS)
def test_assembly_draws_ordinary_text_and_an_owned_token(tmp_path, font_path, source, candidate):
    element = _text([72, 200, 540, 212], source, layout_class="body")
    document = Document(
        tmp_path, font_path, [([_body(100), element, _body(400)], {0: ZH[:30], 1: candidate, 2: ZH[5:30]})]
    )

    (text,) = document.assemble()

    assert _kept_records(document) == []
    assert document.fallback_pages() == []
    assert any(words in _squash(text) for words in ("已经发布", "在此说明", "插件与"))


# ---------------------------------------------------------------------------
# Review leftover F2 (B5): whitespace a drawing path collapses.
# ---------------------------------------------------------------------------
# The class: any string a drawing path shows as one space or as nothing, but
# that the guard's drawn form keeps as several characters, must not separate
# the letters of a token.  The guard reads a flattened form (no whitespace at
# all) besides the visible forms, rather than imitating the renderer.
COLLAPSING_SEPARATORS = {
    "decimal-ref": "&#32;" * 4,
    "hex-ref": "&#x20;" * 4,
    "decimal-tab-ref": "&#9;" * 4,
    "named-tab": "&Tab;" * 4,
    "named-newline": "&NewLine;" * 4,
    "named-nbsp": "&nbsp;" * 4,
    "literal-and-refs": " &#32; &#x20;",
    "bold-spaces": "<b> </b>" * 4,
    "sup-spaces": "<sup> </sup>" * 4,
    "comment-spaces": " <!--a--> <!--b--> <!--c--> <!--d-->",
    "pi-spaces": " <?a?> <?b?> <?c?> <?d?>",
    "refs-markup-unicode": "\u2002&#32;<b> </b>\u3000",
    **{
        f"U+{code:04X}": chr(code) * 4
        for code in (0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000)
    },
}
# Where the separator falls: inside the prefix, inside the kind, before the
# index, before the digest.
SEPARATOR_PLACES = {
    "prefix": ("PHOE", "NA_IMATH_0001_0123456789ABCDEF"),
    "kind": ("PHOENA_IM", "ATH_0001_0123456789ABCDEF"),
    "index": ("PHOENA_IMATH", "0001_0123456789ABCDEF"),
    "digest": ("PHOENA_IMATH_0001", "0123456789ABCDEF"),
}
COLLAPSED_TOKENS = {
    f"{name}-in-{place}": head + separator + tail
    for name, separator in COLLAPSING_SEPARATORS.items()
    for place, (head, tail) in SEPARATOR_PLACES.items()
}
# One separator of each family in every place, for the slower layers.
_LAYER_SEPARATORS = ("decimal-ref", "bold-spaces", "comment-spaces", "U+2002", "U+3000")
LAYER_TOKENS = {
    key: value
    for key, value in COLLAPSED_TOKENS.items()
    if any(key.startswith(f"{name}-in-") for name in _LAYER_SEPARATORS)
}
SPACED_OWNER = f"The diagnostic field {SOURCE_TOKEN}    Notes apply here for reference"
COLLAPSED_CLEAN = {
    "owned-token-then-refs-and-word": (
        SPACED_OWNER, f"诊断字段{SOURCE_TOKEN}&#32;&#32;&#32;&#32;Notes在此说明，仅供参考"
    ),
    "owned-token-in-unicode-spaces": (
        SOURCE_OWNING, f"诊断字段\u3000\u3000{SOURCE_TOKEN}\u2002\u2002\u2002\u2002在此说明"
    ),
    "words-after-collapsed-runs": (
        WORD_SOURCE,
        "Phoena <!--a--> <!--b--> <!--c--> <!--d-->IDE 插件与 Phoena\u2002\u2002\u2002\u2002Supply 柜台"
        "与 Phoena<b> </b><b> </b><b> </b><b> </b>IDs 现已可以正常使用。",
    ),
    "product-after-collapsed-run": (
        # A digit-free source: character references would hand condition M
        # their digits, so this run is written in Unicode spaces.
        ORDINARY_SOURCE, "Phoena\u3000\u3000\u3000\u3000Model 2 已经发布，并且现在可以正常使用。"
    ),
}


def _collapsed_candidate(token):
    return f"{ZH_B}{token}。"


@pytest.mark.parametrize("token", COLLAPSED_TOKENS.values(), ids=COLLAPSED_TOKENS.keys())
def test_f2_whitespace_a_renderer_collapses_does_not_hide_a_token(token):
    assert pdf_internal_placeholder_leaks(BULLET_B, _collapsed_candidate(token))


@pytest.mark.parametrize(("source", "candidate"), COLLAPSED_CLEAN.values(), ids=COLLAPSED_CLEAN.keys())
def test_f2_owned_tokens_and_words_stay_clean_across_collapsed_runs(source, candidate):
    assert pdf_internal_placeholder_leaks(source, candidate) == []


@pytest.mark.parametrize("token", LAYER_TOKENS.values(), ids=LAYER_TOKENS.keys())
def test_f2_validation_and_the_batch_net_reject_a_collapsed_token(token):
    from phoena_translator.pdf.translate import page

    candidate = _collapsed_candidate(token)
    with pytest.raises(PDFTranslationIntegrityError) as raised:
        _validate_pdf_page_translations(1, _single(BULLET_B), {0: candidate})
    assert raised.value.issues == [{"element": 1, "reason": PDF_PLACEHOLDER_LEAK_REASON}]
    context = SimpleNamespace(task_id="t", logger=log)
    element = _single(BULLET_B)[0]
    assert page._restore_batch_part(context, 0, element, 1, BULLET_B, candidate, ((), (), ())) is None


@pytest.mark.parametrize("token", LAYER_TOKENS.values(), ids=LAYER_TOKENS.keys())
def test_f2_assembly_never_draws_a_collapsed_token(tmp_path, font_path, token):
    kept = _text([72, 200, 540, 212], BULLET_B, layout_class="body")
    document = Document(
        tmp_path,
        font_path,
        [([_body(100), kept, _body(400)], {0: ZH[:30], 1: _collapsed_candidate(token), 2: ZH[5:30]})],
    )

    (text,) = document.assemble()

    assert _drawn_tokens(text) == []
    assert document.fallback_pages() == []
    _assert_kept_alone(document, kept)
    (entry,) = _kept_records(document)
    assert entry["reason"].startswith(PDF_PLACEHOLDER_LEAK_REASON)


@pytest.mark.parametrize(("source", "candidate"), COLLAPSED_CLEAN.values(), ids=COLLAPSED_CLEAN.keys())
def test_f2_assembly_draws_the_clean_near_misses(tmp_path, font_path, source, candidate):
    element = _text([72, 200, 540, 212], source, layout_class="body")
    document = Document(
        tmp_path, font_path, [([_body(100), element, _body(400)], {0: ZH[:30], 1: candidate, 2: ZH[5:30]})]
    )

    document.assemble()

    assert _kept_records(document) == []
    assert document.fallback_pages() == []


def _htmlbox_round_trip(candidate):
    import fitz

    from phoena_translator.pdf.render.paragraph_html import _pdf_text_to_html

    with fitz.open() as document:
        page = document.new_page()
        page.insert_htmlbox(fitz.Rect(36, 36, 576, 400), _pdf_text_to_html(candidate))
        return page.get_text()


@pytest.mark.parametrize("token", COLLAPSED_TOKENS.values(), ids=COLLAPSED_TOKENS.keys())
def test_f2_round_trip_whatever_is_drawn_was_caught(token):
    # Draw, read the text back, run the check: a token the reader can see in
    # the drawn text was already caught in the candidate.
    candidate = f"Revenue rose {token} in the quarter."
    drawn = _htmlbox_round_trip(candidate)
    if pdf_internal_placeholder_leaks("", drawn):
        assert pdf_internal_placeholder_leaks("", candidate), drawn


@pytest.mark.parametrize("name", ["decimal-ref", "hex-ref", "named-tab", "bold-spaces", "comment-spaces"])
def test_f2_round_trip_the_renderer_does_collapse_these(name):
    # The round trip has teeth: the htmlbox draws one space for each of
    # these runs, so the drawn text holds a token the old drawn form missed.
    candidate = f"Revenue rose PHOE{COLLAPSING_SEPARATORS[name]}NA_IMATH_0001_0123456789ABCDEF in the quarter."
    assert pdf_internal_placeholder_leaks("", _htmlbox_round_trip(candidate))
