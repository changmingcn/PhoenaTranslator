"""The typed PDF vocabulary must stay wire-compatible with plain strings."""

from __future__ import annotations

import json

from phoena_translator.pdf.model import (
    CrossPageMergeDecision,
    ElementType,
    LayoutClass,
    SkipReason,
)


def test_enum_members_behave_like_their_strings():
    elem = {"type": ElementType.TEXT, "layout_class": LayoutClass.BODY,
            "skip_translate_reason": SkipReason.WATERMARK}
    assert elem["type"] == "text" and "body" == elem["layout_class"]
    assert elem["skip_translate_reason"] in {"watermark"}
    assert json.dumps(elem, sort_keys=True) == json.dumps(
        {"type": "text", "layout_class": "body", "skip_translate_reason": "watermark"},
        sort_keys=True,
    )
    assert f"{SkipReason.TABLE_PRESERVED}" == "table_preserved"


def test_cross_page_decision_serialises_the_historical_entry():
    decision = CrossPageMergeDecision(
        source_page=0,
        source_element=4,
        destination_element=0,
        source_continuation_reason="dangling",
        carried_tail="traders.",
        source_layout="body",
        destination_layout="scattered",
    )
    assert list(decision.to_audit().items()) == [
        ("kind", "cross-page-orphan-tail"),
        ("source_page", 1),
        ("destination_page", 2),
        ("source_element", 4),
        ("destination_element", 0),
        ("decision", "accepted"),
        ("reason", "accepted"),
        ("source_continuation_reason", "dangling"),
        ("carried_tail", "traders."),
        ("source_layout", "body"),
        ("destination_layout", "scattered"),
    ]
    flagged = CrossPageMergeDecision(0, 1, 2, "complete", "x.", None, None, tail_unterminated=True)
    assert flagged.to_audit()["tail_unterminated"] is True
