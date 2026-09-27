"""Small, conservative cleanup for Japanese Animagine prompt output.

This is intentionally narrower than a general prompt sanitizer.  It runs only
after the authoritative core/general/character resolvers have been applied and
removes tokens that are either clearly invalid or are obvious aliases of a
resolver-selected canonical tag.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_STANDALONE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_TAG_SEPARATOR_RE = re.compile(r"[_\-]+")

# These are deliberately only high-confidence equivalences.  The canonical
# side is the prompt spelling used by the core/general resolver, not a free
# form LLM output.
_CANONICAL_ALIASES: dict[str, frozenset[str]] = {
    "smile": frozenset({"smiley face", "smiling", "smiling face"}),
    "peace sign": frozenset({"v", "v sign", "victory sign"}),
}


@dataclass(frozen=True)
class PositivePromptCleanupResult:
    positive: str
    removed_tags: tuple[str, ...] = ()
    removed_reasons: tuple[tuple[str, str], ...] = ()


def normalize_prompt_tag_key(value: str) -> str:
    """Normalize only presentation differences, such as ``long_hair``."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = _TAG_SEPARATOR_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _explicit_canonical_aliases(user_input: str) -> set[str]:
    """Return tiny canonical mappings supported by an explicit Japanese cue.

    The core dictionary currently does not own hand-gesture tags.  ``ピース``
    is nevertheless an explicit, unambiguous cue in the Japanese route, so a
    resolver-produced ``v`` can be represented by the stable prompt spelling
    ``peace sign``.  No such promotion is done for English/direct input.
    """

    text = unicodedata.normalize("NFKC", str(user_input or "")).casefold()
    if any(cue in text for cue in ("ピース", "ピースサイン", "peace sign")):
        return {"peace sign"}
    return set()


def cleanup_positive_prompt(
    positive: str,
    *,
    canonical_prompt_tags: Iterable[str] = (),
    user_input: str = "",
) -> PositivePromptCleanupResult:
    """Remove invalid tokens and aliases after resolver application.

    Only standalone numeric tokens are treated as invalid.  Tokens such as
    ``1girl`` remain valid.  Semantic aliases are removed only when their
    canonical counterpart is authoritative (or is the tiny explicit ``ピース``
    mapping above), so unrelated general tags are not collapsed together.
    """

    canonical_tags = tuple(
        str(tag).strip()
        for tag in canonical_prompt_tags
        if str(tag or "").strip()
    )
    canonical_resolved_keys = {
        normalize_prompt_tag_key(tag)
        for tag in canonical_tags
    }
    canonical_keys = set(canonical_resolved_keys)
    canonical_keys.update(_explicit_canonical_aliases(user_input))

    alias_to_canonical = {
        alias: canonical
        for canonical, aliases in _CANONICAL_ALIASES.items()
        if canonical in canonical_keys
        for alias in aliases
    }

    kept: list[str] = []
    removed: list[str] = []
    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    explicit_canonical_keys = _explicit_canonical_aliases(user_input)
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = normalize_prompt_tag_key(token)
        if _STANDALONE_NUMBER_RE.fullmatch(key):
            removed.append(token)
            reasons.append((token, "invalid_standalone_number"))
            continue
        if key in alias_to_canonical:
            canonical = alias_to_canonical[key]
            removed.append(token)
            reasons.append((token, f"duplicate_alias_of:{canonical}"))
            if canonical not in seen and (
                canonical in explicit_canonical_keys
                or canonical in canonical_resolved_keys
            ):
                kept.append(canonical)
                seen.add(canonical)
            continue
        if key in seen:
            removed.append(token)
            reasons.append((token, "duplicate_normalized_tag"))
            continue
        kept.append(token)
        seen.add(key)

    return PositivePromptCleanupResult(
        positive=", ".join(kept),
        removed_tags=tuple(removed),
        removed_reasons=tuple(reasons),
    )
