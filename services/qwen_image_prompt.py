"""Small natural-language prompt boundary for Qwen-Image.

This module intentionally does not import the Animagine/Danbooru prompt
pipeline.  The first Qwen adapter accepts the user's natural-language layers
and joins them without adding model-specific quality or tag vocabulary.
"""

from __future__ import annotations

from typing import Any


class QwenPromptError(ValueError):
    """Raised when a Qwen prompt cannot be built from the request layers."""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def build_qwen_prompt(request: Any) -> str:
    """Build a bounded natural-language prompt without SDXL tag injection."""

    values: list[str] = []
    for value in (
        getattr(request, "character_preset", None),
        getattr(request, "freeform_character", None),
        getattr(request, "request_text", None),
        getattr(request, "appearance_text", None),
    ):
        clean = _clean(value)
        if clean:
            values.append(clean)

    # Explicit values are user input, not a request to run the Danbooru
    # resolver.  Preserve them literally so the Qwen path stays transparent.
    for value in getattr(request, "explicit_tags", ()) or ():
        clean = _clean(value)
        if clean:
            values.append(clean)

    if not values:
        raise QwenPromptError("Qwen-Image用の自然文promptを指定してください。")
    return ", ".join(dict.fromkeys(values))
