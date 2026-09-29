"""Minimal request-owned composition conflict detection.

This shared foundation deliberately contains only the deterministic composition
ownership check. Memory-specific filtering belongs to a later feature layer.
"""

from __future__ import annotations

from dataclasses import dataclass


_COMPOSITION_CUES = (
    "全身",
    "上半身",
    "バストアップ",
    "ポートレート",
    "構図",
    "full body",
    "full-body",
    "upper body",
    "portrait",
    "close-up",
    "close up",
    "cowboy shot",
    "medium shot",
)


@dataclass(frozen=True)
class ExplicitVisualConstraints:
    """The request-owned visual categories needed by shared composition helpers."""

    controls_composition: bool = False


def analyze_explicit_visual_constraints(request_text: str) -> ExplicitVisualConstraints:
    text = str(request_text or "").casefold()
    return ExplicitVisualConstraints(
        controls_composition=any(cue.casefold() in text for cue in _COMPOSITION_CUES),
    )
