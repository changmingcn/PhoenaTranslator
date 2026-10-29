"""Deterministic fixtures for the end-to-end PDF golden snapshot tests.

The corpus is generated with PyMuPDF at test time (no binary fixtures), the
translator is a pure function of its input, and the snapshot only reads the
files the pipeline publishes (layout cache, page caches, audit, task record
and the output PDF).  It therefore pins observable behaviour without
depending on where any helper lives inside ``phoena_translator``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import fitz

# ---------------------------------------------------------------------------
# Deterministic stand-in translator
# ---------------------------------------------------------------------------

_CJK_ALPHABET = (
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可"
    "主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家"
    "电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合"
    "还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明"
)
_PROTECTED_RE = re.compile(
    r"PHOENA_[A-Z]+_\d{4}_[0-9A-F]{16}"
    r"|≡≡≡SPLIT≡≡≡"
    r"|<sup>.*?</sup>"
    r"|</?[A-Za-z]+>"
    r"|https?://\S+"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def _fake_word(word: str) -> str:
    digest = hashlib.sha256(word.lower().encode("utf-8")).digest()
    return _CJK_ALPHABET[digest[0] % len(_CJK_ALPHABET)] + _CJK_ALPHABET[
        digest[1] % len(_CJK_ALPHABET)
    ]


def _translate_plain(text: str) -> str:
    return _WORD_RE.sub(lambda match: _fake_word(match.group(0)), text)


def fake_translate_text(text: str, system_prompt: str | None = None, retries: int = 4) -> str:
    """Replace every English word with two CJK characters, keeping tokens."""
    del system_prompt, retries
    parts: list[str] = []
    cursor = 0
    for match in _PROTECTED_RE.finditer(text):
        parts.append(_translate_plain(text[cursor:match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_translate_plain(text[cursor:]))
    return "".join(parts)


FAKE_FONT_CHARS = "".join(sorted(set(_CJK_ALPHABET)))

# ---------------------------------------------------------------------------
# Test font: a TrueType subset cut from a system CJK font
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = (
    ("~/fonts/NotoSansSC-Regular.ttf", None),
    ("~/fonts/SarasaGothicSC-Regular.ttf", None),
    ("/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf", None),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
)


def build_test_font(output_dir: Path) -> tuple[Path, str] | None:
    """Return a small glyf-outline TTF covering the corpus, plus its source id."""
    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    for raw_path, font_number in _FONT_CANDIDATES:
        path = Path(os.path.expanduser(raw_path))
        if not path.is_file():
            continue
        try:
            font = TTFont(str(path), fontNumber=font_number or 0, lazy=False)
        except Exception:
            continue
        if "glyf" not in font:
            font.close()
            continue
        options = subset.Options()
        options.layout_features = []
        options.drop_tables += ["GSUB", "GPOS", "GDEF"]
        options.notdef_outline = True
        options.name_IDs = ["*"]
        subsetter = subset.Subsetter(options=options)
        text = (
            "".join(chr(code) for code in range(32, 127))
            + "".join(chr(code) for code in range(0xA0, 0x180))
            + "".join(chr(code) for code in range(0x391, 0x3CA))
            + "–—‘’“”•…≤≥≠≈±×÷∑∫√∞∂∆∇∈∀∃→←"
            + FAKE_FONT_CHARS
        )
        subsetter.populate(text=text)
        subsetter.subset(font)
        target = output_dir / "golden-test-font.ttf"
        font.save(str(target))
        font.close()
        source_id = f"{path.name}#{font_number or 0}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
        return target, source_id
    return None


# ---------------------------------------------------------------------------
# Synthetic source documents
# ---------------------------------------------------------------------------

LETTER = fitz.Rect(0, 0, 612, 792)
_BODY = (
    "Market makers provide liquidity across the trading day, and their quotes "
    "respond to inventory risk, funding costs and the volatility of the "
    "underlying securities. When volatility rises, spreads widen and depth "
    "declines, which in turn affects the execution quality observed by "
    "institutional investors."
)


def _paragraph(page, y, text, *, size=10.5, font="helv", x0=72, x1=540, indent=0.0):
    rect = fitz.Rect(x0 + indent, y, x1, y + 400)
    shortfall = page.insert_textbox(rect, text, fontsize=size, fontname=font, align=0)
    used = 400 - shortfall
    return y + used + size * 0.9


def _superscript(page, point, text, size=6.5):
    page.insert_text(fitz.Point(point.x, point.y - 3.6), text, fontsize=size, fontname="helv")


def build_report_pdf(path: Path) -> None:
    """A three-page report exercising headings, lists, footnotes, references,
    cross-page sentences, TOC leaders, formulas and a ruled table."""
    doc = fitz.open()

    # Page 1: title, TOC, body, bullets, footnote.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 90), "Liquidity in Fixed Income Markets", fontsize=20, fontname="hebo")
    page.insert_text((72, 112), "Working Paper Series", fontsize=11, fontname="helv")
    y = 140
    for label, number in (("Introduction", 1), ("Market Structure", 2), ("References", 3)):
        page.insert_text((72, y), f"{label} " + "." * 60 + f" {number}", fontsize=10, fontname="helv")
        y += 14
    y += 10
    page.insert_text((72, y), "1 Introduction", fontsize=14, fontname="hebo")
    y = _paragraph(page, y + 8, _BODY)
    y = _paragraph(page, y, "Dealers balance three competing objectives:")
    for item in (
        "maintaining a competitive two-sided market for clients;",
        "keeping inventory within the limits set by the risk committee;",
        "minimising the balance-sheet cost of holding long positions.",
    ):
        page.insert_text((80, y + 9), "•", fontsize=10.5, fontname="helv")
        y = _paragraph(page, y, item, x0=92)
    y = _paragraph(
        page,
        y,
        "Recent studies document a persistent decline in dealer capacity since "
        "the introduction of the leverage ratio.",
    )
    _superscript(page, fitz.Point(72 + 4.7 * 70, y - 9.5), "1")
    page.draw_line((72, 700), (220, 700), width=0.5)
    page.insert_text((72, 714), "1", fontsize=6.5, fontname="helv")
    page.insert_text((78, 717), "See the supplementary appendix for the full sample description.", fontsize=8, fontname="helv")
    page.insert_text((300, 760), "1", fontsize=9, fontname="helv")
    # A body sentence that runs over the page boundary.
    _paragraph(
        page,
        620,
        "The remainder of this paper describes how the quoting behaviour of the "
        "largest dealers changed during the stress episode, and why the",
    )

    # Page 2: continuation, formula, table.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    y = _paragraph(
        page,
        72,
        "subsequent recovery in market depth was uneven across maturities. "
        "Short-dated bonds recovered first, while long-dated bonds lagged.",
    )
    page.insert_text((72, y + 6), "2 Market Structure", fontsize=14, fontname="hebo")
    y = _paragraph(page, y + 14, "We model the bid-ask spread as a function of volatility and inventory:")
    page.insert_text((200, y + 14), "s = α + β σ + γ |I| ≤ ∑ w x", fontsize=12, fontname="symb")
    page.insert_text((470, y + 14), "(1)", fontsize=10.5, fontname="helv")
    y = _paragraph(page, y + 34, "where sigma denotes realised volatility and I the dealer inventory.")
    top = y + 10
    rows = (("Maturity", "Spread (bp)", "Depth ($m)"), ("2-year", "0.8", "250"), ("10-year", "1.6", "120"), ("30-year", "3.1", "45"))
    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row):
            page.insert_text(
                (82 + col_index * 150, top + 16 + row_index * 20),
                cell,
                fontsize=10,
                fontname="hebo" if row_index == 0 else "helv",
            )
    for row_index in range(len(rows) + 1):
        page.draw_line((72, top + row_index * 20 + 2), (522, top + row_index * 20 + 2), width=0.6)
    for col_index in range(4):
        page.draw_line((72 + col_index * 150, top + 2), (72 + col_index * 150, top + 82), width=0.6)
    _paragraph(page, top + 100, "Table 1 reports average spreads and depth by maturity bucket.", size=9)
    page.insert_text((300, 760), "2", fontsize=9, fontname="helv")

    # Page 3: references with hanging indents and a disclaimer.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 80), "References", fontsize=14, fontname="hebo")
    y = 100
    for entry in (
        "Adrian, T., Fleming, M. and Vogt, E. (2017). Market liquidity after the "
        "financial crisis. Annual Review of Financial Economics, 9, 43-83.",
        "Duffie, D. (2020). Still the world's safe haven? Redesigning the U.S. "
        "Treasury market after the COVID-19 crisis. Hutchins Center Working Paper 62.",
        "He, Z., Nagel, S. and Song, Z. (2022). Treasury inconvenience yields during "
        "the COVID-19 crisis. Journal of Financial Economics, 143(1), 57-79.",
    ):
        y = _paragraph(page, y, entry, size=10)
    _paragraph(
        page,
        y + 30,
        "Disclaimer: The views expressed in this paper are those of the authors "
        "and do not necessarily reflect the position of the institution.",
        size=8,
    )
    page.insert_text((300, 760), "3", fontsize=9, fontname="helv")
    doc.save(str(path), garbage=0, deflate=False, no_new_id=True)
    doc.close()


def _lines(page, x, y, rows, *, size=10.5, font="helv", lead=12.6):
    """Insert one text object per visual line, as typeset PDFs usually do."""
    for index, row in enumerate(rows):
        offset, text = row if isinstance(row, tuple) else (0, row)
        page.insert_text((x + offset, y + index * lead), text, fontsize=size, fontname=font)
    return y + len(rows) * lead


def build_structures_pdf(path: Path) -> None:
    """Line-by-line layouts that drive the semantic merge and split passes."""
    doc = fitz.open()

    # Page 1: numbered items, hanging references, detached bullets.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 60), "Recommendations", fontsize=13, fontname="hebo")
    y = _lines(page, 72, 84, [
        "1. The first numbered recommendation asks dealers to publish",
        (14, "their quoting policies so that clients can compare execution"),
        (14, "quality across venues and time."),
    ])
    y = _lines(page, 72, y + 10, [
        "2. The second recommendation concerns central clearing of",
        (14, "cash Treasury transactions and of repo agreements."),
    ])
    y = _lines(page, 72, y + 18, [
        "Liquidity conditions deteriorated sharply in March as investors sold",
        "Treasury securities to raise cash, and dealers struggled to absorb the",
        "flows because their balance sheets were already constrained by",
        "regulation and by the size of their existing inventories.",
    ])
    page.insert_text((74, y + 22), "•", fontsize=13, fontname="helv")
    y = _lines(page, 92, y + 20, [
        "Primary dealers increased their holdings of off-the-run securities",
        "during the first two weeks of the episode.",
    ])
    page.insert_text((74, y + 14), "•", fontsize=13, fontname="helv")
    y = _lines(page, 92, y + 12, ["Hedge funds unwound basis trades at scale."])
    y = _lines(page, 72, y + 16, [
        "The episode also showed that the market depends on a small number of",
    ])
    page.insert_text((72, y + 1), "large intermediaries whose capacity", fontsize=10.5, fontname="heit")
    y = _lines(page, 72, y + 13.6, [
        "to warehouse risk is limited during periods of broad deleveraging.",
    ])
    page.insert_text((72, y + 16), "References", fontsize=12, fontname="hebo")
    y = _lines(page, 72, y + 34, [
        "Smith, J. and Jones, K. (2019). Dealer balance sheets and pricing.",
        (18, "Journal of Finance, 74(3), 1201-1245."),
        "Taylor, R. (2021). Repo market stress and the central bank.",
        (18, "Review of Financial Studies, 34(2), 501-540."),
        "Wang, L. (2018). Search frictions in bond markets.",
        (18, "Econometrica, 86(1), 1-40."),
    ], size=9.5, lead=11.5)
    page.insert_text((72, y + 20), "3", fontsize=6, fontname="helv")
    _lines(page, 76, y + 24, [
        "The data cover the period from January 2015 to December 2021 and",
        "exclude transactions with the central bank.",
    ], size=8, lead=9.6)

    # Page 2: an open term/meaning glossary.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 70), "Glossary", fontsize=13, fontname="hebo")
    page.insert_text((72, 100), "Term", fontsize=10, fontname="hebo")
    page.insert_text((220, 100), "Meaning", fontsize=10, fontname="hebo")
    for row, (term, meaning) in enumerate((
        ("spread", "The difference between the bid and the ask price."),
        ("depth", "The quantity available at the best quoted prices."),
        ("pinging", "Sending small orders to discover hidden liquidity."),
        ("profiling", "Classifying counterparties by their trading behaviour."),
        ("inventory", "The net position a dealer holds in a security."),
        ("tick size", "The minimum price increment on a trading venue."),
        ("haircut", "The discount applied to collateral in a repo."),
        ("tenor", "The remaining time until a contract matures."),
        ("basis", "The price gap between a bond and its future."),
    )):
        page.insert_text((72, 124 + row * 22), term, fontsize=10, fontname="helv")
        page.insert_text((220, 124 + row * 22), meaning, fontsize=10, fontname="helv")

    # Page 3: an entity directory and a two-column body.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    _lines(page, 72, 72, [
        "The list of firms that responded to the survey is presented in",
        "alphabetical order below.",
    ])
    for row, firm in enumerate((
        "Alpha Capital Markets",
        "Blue River Securities",
        "Cedar Point Trading",
        "Delta Bond Partners",
        "Evergreen Fixed Income",
        "Fairview Liquidity Group",
    )):
        page.insert_text((90, 120 + row * 20), firm, fontsize=9.5, fontname="helv")
    left = (
        "Electronic trading has grown steadily in the dealer-to-client segment, "
        "where request-for-quote protocols now account for most volume."
    )
    right = (
        "In the interdealer segment, central limit order books dominate the "
        "on-the-run market, while voice trading persists off the run."
    )
    page.insert_textbox(fitz.Rect(72, 260, 296, 360), left, fontsize=10, fontname="helv")
    page.insert_textbox(fitz.Rect(316, 260, 540, 360), right, fontsize=10, fontname="helv")
    _lines(page, 72, 420, [
        "Brunnermeier, M. and Pedersen, L. (2009). Market liquidity and funding",
        (18, "liquidity in a model of margin spirals and dealer"),
        (18, "constraints. Review of Financial Studies, 22, 2201-2238."),
        "Fleming, M. (2003). Measuring Treasury market liquidity and the",
        (18, "role of the primary dealer system. Economic Policy Review, 9, 83-108."),
    ], size=9.5, lead=11.5)
    page.insert_text((560, 500), "Basis points", fontsize=8, fontname="helv", rotate=90)

    # Page 4: display and inline mathematics next to prose.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    y = _lines(page, 72, 80, [
        "The optimal quote solves a dynamic programming problem in which the",
        "dealer trades off expected profit against inventory risk, so that",
    ])
    page.insert_text((180, y + 10), "V(q) = max  E[ ∑ β (π − λ q²) ]", fontsize=12, fontname="symb")
    page.insert_text((470, y + 10), "(2)", fontsize=10.5, fontname="helv")
    y = _lines(page, 72, y + 34, [
        "where the discount factor and the penalty parameter are estimated",
        "from the cross-section of dealer inventories.",
    ])
    page.insert_text((72, y + 8), "The estimate of", fontsize=10.5, fontname="helv")
    page.insert_text((150, y + 8), "λ ≈ 0.42", fontsize=10.5, fontname="symb")
    page.insert_text((200, y + 8), "implies a strong inventory effect on quoted spreads.", fontsize=10.5, fontname="helv")

    # Page 5: wrapped citations whose lines do not end a sentence.
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    _lines(page, 72, 80, [
        "Smith, J. and Jones, K. (2019). Dealer balance sheets and the pricing",
        (18, "of liquidity in government bond markets. Journal of Finance, 74(3),"),
        (18, "1201-1245."),
        "Taylor, R. (2021). Repo market stress and the role of the central bank.",
        (18, "Review of Financial Studies, 34(2), 501-540."),
    ], size=9.5, lead=11.5)
    doc.save(str(path), garbage=0, deflate=False, no_new_id=True)
    doc.close()


CORPUS = {
    "report": build_report_pdf,
    "structures": build_structures_pdf,
}

# ---------------------------------------------------------------------------
# Snapshot normalisation
# ---------------------------------------------------------------------------

_VOLATILE_KEYS = {"ts"}


def _is_volatile(key: str, value) -> bool:
    """Wall-clock timestamps (``*_at`` numbers and ``ts``) are not behaviour."""
    if key in _VOLATILE_KEYS:
        return True
    return key.endswith("_at") and isinstance(value, (int, float))
# Font metrics decide how a translation is fitted; these depend on the CJK
# font available on the host and are compared only when the font matches.
FONT_DEPENDENT_AUDIT_KEYS = {
    "htmlbox_placement_events",
    "render_scale_events",
    "rejected_render_scale_events",
    "textbox_fallback_events",
}


def normalise(value, tmp_root: str):
    if isinstance(value, dict):
        return {
            str(key): normalise(item, tmp_root)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_volatile(str(key), item)
        }
    if isinstance(value, (list, tuple)):
        return [normalise(item, tmp_root) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        return value.replace(tmp_root, "<TMP>")
    return value


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Pipeline runner (executed in a fresh interpreter by the golden test)
# ---------------------------------------------------------------------------

TASK_ID = "golden0000000000"


def run_pipeline(name: str, workdir: Path, font_path: Path) -> dict:
    """Translate one corpus document end to end and return its raw snapshot."""
    for key, sub in (
        ("TRANSLATOR_UPLOAD_DIR", "uploads"),
        ("TRANSLATOR_OUTPUT_DIR", "outputs"),
        ("TRANSLATOR_PROGRESS_DIR", "progress"),
    ):
        os.environ[key] = str(workdir / sub)
    os.environ.update(
        {
            "TRANSLATOR_LOG_FILE": str(workdir / "translator.log"),
            "TRANSLATOR_TRANSLATION_WORKERS": "1",
            "TRANSLATOR_API_MAX_CONCURRENCY": "1",
            "TRANSLATOR_DISABLE_AUTO_RESUME": "1",
            "TRANSLATOR_PDF_FONT_REGULAR": str(font_path),
            "TRANSLATOR_PDF_FONT_BOLD": str(font_path),
            "TRANSLATOR_LLM_PROVIDER": "deepseek",
        }
    )
    from phoena_translator import application

    application.initialize_runtime()
    requests: list[str] = []

    def translate(text, system_prompt=None, retries=4):
        requests.append(text)
        return fake_translate_text(text, system_prompt, retries)

    application.translate_text = translate
    source = workdir / "uploads" / f"{TASK_ID}.pdf"
    CORPUS[name](source)
    output = workdir / "outputs" / f"{TASK_ID}.pdf"
    application.save_progress(
        TASK_ID,
        {
            "status": "queued",
            "progress": 0,
            "total": 0,
            "current_file": "",
            "completed_files": [],
            "filename": f"{name}.pdf",
            "out_filename": f"zh_{name}.pdf",
            "type": "pdf",
        },
    )
    application.translate_pdf(TASK_ID, str(source), str(output))

    progress_dir = workdir / "progress"

    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    pages_dir = progress_dir / f"{TASK_ID}_pages"
    page_caches = {
        item.name: read_json(item)
        for item in sorted(pages_dir.glob("*.json"))
    } if pages_dir.is_dir() else {}
    output_text = []
    if output.is_file():
        with fitz.open(str(output)) as doc:
            output_text = [
                re.sub(r"\s+", "", page.get_text("text")) for page in doc
            ]
    return {
        "task": read_json(progress_dir / f"{TASK_ID}.json"),
        "layout": read_json(progress_dir / f"{TASK_ID}_layout.json"),
        "audit": read_json(progress_dir / f"{TASK_ID}_audit.json"),
        "page_caches": page_caches,
        "requests": sorted(requests),
        "output_text": output_text,
    }


def split_snapshot(raw: dict, tmp_root: str) -> tuple[dict, dict]:
    """Return (font-independent, font-dependent) normalised snapshots."""
    snapshot = normalise(raw, tmp_root)
    audit = snapshot.get("audit") or {}
    font_dependent = {
        key: audit.pop(key) for key in sorted(FONT_DEPENDENT_AUDIT_KEYS) if key in audit
    }
    font_dependent["output_text"] = snapshot.pop("output_text")
    return snapshot, font_dependent


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    import sys

    _name, _workdir, _font, _out = sys.argv[1:5]
    _raw = run_pipeline(_name, Path(_workdir), Path(_font))
    Path(_out).write_text(json.dumps(_raw, ensure_ascii=False), encoding="utf-8")
