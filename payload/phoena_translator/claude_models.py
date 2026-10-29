"""Claude 5.5 model identities, effort levels and list prices.

Shared by configuration (defaults and pricing) and the Claude adapter
(request shaping) so both recognise models the same way.
"""

from __future__ import annotations

CLAUDE_OPUS_5_5 = "claude-opus-5-5"
CLAUDE_SONNET_5_5 = "claude-sonnet-5-5"
CLAUDE_HAIKU_5_5 = "claude-haiku-5-5"
CLAUDE_MODEL_FAMILIES = (CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5, CLAUDE_HAIKU_5_5)
CLAUDE_DEFAULT_MODEL = CLAUDE_OPUS_5_5
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# USD per 1M tokens: (input, output, cache read). 5-minute cache writes are
# billed at 1.25x input. Unknown Claude models are priced as Opus 5.5 so the
# ledger's ``conservative_cost_usd`` never under-reports.
CLAUDE_DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    CLAUDE_OPUS_5_5: (4.0, 20.0, 0.20),
    CLAUDE_SONNET_5_5: (2.0, 10.0, 0.20),
    CLAUDE_HAIKU_5_5: (0.10, 0.50, 0.01),
}
CLAUDE_CACHE_WRITE_MULTIPLIER = 1.25


def claude_model_family(model: str) -> str:
    """Return the known 5.5 family id for ``model`` (or ``""`` if unknown)."""
    for family in CLAUDE_MODEL_FAMILIES:
        if model == family or model.startswith(family + "-"):
            return family
    return ""


def claude_default_prices(model: str) -> tuple[float, float, float]:
    return CLAUDE_DEFAULT_PRICES[claude_model_family(model) or CLAUDE_DEFAULT_MODEL]


__all__ = [
    "CLAUDE_CACHE_WRITE_MULTIPLIER",
    "CLAUDE_DEFAULT_MODEL",
    "CLAUDE_DEFAULT_PRICES",
    "CLAUDE_EFFORT_LEVELS",
    "CLAUDE_HAIKU_5_5",
    "CLAUDE_MODEL_FAMILIES",
    "CLAUDE_OPUS_5_5",
    "CLAUDE_SONNET_5_5",
    "claude_default_prices",
    "claude_model_family",
]
