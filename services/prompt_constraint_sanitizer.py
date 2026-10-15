"""Request-owned prompt constraints applied after Prompt Designer output.

The Prompt Designer may suggest useful tags, but request fields such as rating,
framing, and a released interaction profile are stronger constraints.  This
module removes only lower-priority Prompt Designer tags that are unambiguously
in conflict with those constraints; user/request and interaction-profile tags
are preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from services.prompt_conflict_resolver import (
    PromptTagRecord,
    SuppressedPromptTag,
    classify_tag,
    normalize_prompt_constraint,
)


SAFE_BLOCKED_TAGS = frozenset(
    {
        "sex", "explicit", "sexual", "intercourse", "penetration", "vaginal",
        "anal sex", "oral sex", "missionary", "reverse cowgirl", "cowgirl",
        "doggystyle", "man on top", "woman on top", "topless", "nude", "nudity",
        "cum", "ejaculation", "fellatio", "cunnilingus", "clothed sex",
    }
)
_GENERIC_INTERACTION_TAGS = frozenset({"hetero", "together"})
_STANDING_CONFLICT_TAGS = frozenset(
    {"lying", "lying on back", "reclining", "reclined", "sitting", "seated", "sitting together", "lap pillow"}
)


@dataclass(frozen=True)
class PromptConstraintResult:
    positive: str
    provenance: tuple[PromptTagRecord, ...]
    removed: tuple[SuppressedPromptTag, ...] = ()


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(
        clean
        for raw in str(value or "").split(",")
        if (clean := " ".join(raw.strip().split()))
    )


def _request_tag_keys(request: object) -> frozenset[str]:
    values: list[str] = []
    for name in ("explicit_tags", "request_text", "character_preset", "freeform_character", "appearance_text"):
        values.extend(_tokens(getattr(request, name, "")))
    return frozenset(normalize_prompt_constraint(value) for value in values if value)


def _request_text_contains(request: object, key: str) -> bool:
    text = " ".join(
        str(getattr(request, name, "") or "")
        for name in ("request_text", "character_preset", "freeform_character", "appearance_text")
    )
    normalized = normalize_prompt_constraint(text)
    if not normalized or not key:
        return False
    if any(ord(char) > 127 for char in key):
        return key in normalized
    return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalized) is not None


def _profile_tag_keys(profile: object) -> frozenset[str]:
    return frozenset(
        normalize_prompt_constraint(value)
        for value in (getattr(profile, "prompt_additions", ()) or ())
        if value
    )


def _source_for(tag: str, provenance: Mapping[str, PromptTagRecord]) -> str:
    item = provenance.get(normalize_prompt_constraint(tag))
    if item is None:
        return "prompt_designer"
    source = str(item.source or "").strip()
    return "prompt_designer" if source == "request_resolver" else (source or "prompt_designer")


def _is_prompt_designer_source(source: str) -> bool:
    return source in {"prompt_designer", "request_resolver"}


def apply_request_prompt_constraints(
    positive: object,
    *,
    request: object,
    profile: object,
    provenance: Iterable[PromptTagRecord] = (),
) -> PromptConstraintResult:
    """Sanitize only lower-priority Prompt Designer tags."""

    source_by_key: dict[str, PromptTagRecord] = {}
    for item in provenance:
        if isinstance(item, PromptTagRecord):
            source_by_key.setdefault(normalize_prompt_constraint(item.tag), item)

    request_keys = _request_tag_keys(request)
    profile_keys = _profile_tag_keys(profile)
    requested_rating = normalize_prompt_constraint(getattr(request, "rating", "auto"))
    interaction_id = str(getattr(profile, "id", "") or "").strip()
    standing_profile = "standing together" in profile_keys
    result: list[str] = []
    kept_records: list[PromptTagRecord] = []
    removed: list[SuppressedPromptTag] = []
    seen: set[str] = set()

    for tag in _tokens(positive):
        key = normalize_prompt_constraint(tag)
        if not key or key in seen:
            continue
        source = _source_for(tag, source_by_key)
        protected = (
            key in request_keys
            or _request_text_contains(request, key)
            or key in profile_keys
            or not _is_prompt_designer_source(source)
        )
        reason = ""
        category = ""
        conflicting_tag = ""
        conflicting_concept = ""

        if not protected and requested_rating == "safe" and key in SAFE_BLOCKED_TAGS:
            reason = "rating_safe_conflict"
            category = "safety"
            conflicting_concept = "safe_rating"
        elif not protected:
            categories = classify_tag(tag)
            if "interaction" in categories and key not in _GENERIC_INTERACTION_TAGS and key not in profile_keys:
                reason = "interaction_conflict"
                category = "interaction"
                conflicting_tag = next(iter(profile_keys), interaction_id)
                conflicting_concept = "released_interaction"
            elif standing_profile and key in _STANDING_CONFLICT_TAGS:
                reason = "composition_conflict"
                category = "basic_pose"
                conflicting_tag = "standing together"
                conflicting_concept = "standing_profile"

        if reason:
            removed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason=reason,
                    category=category,
                    conflicting_tag=conflicting_tag,
                    conflicting_concept=conflicting_concept,
                    source_interaction=interaction_id,
                )
            )
            continue

        result.append(tag)
        seen.add(key)
        known = source_by_key.get(key)
        kept_records.append(PromptTagRecord(tag, source, known.category if known else ""))

    return PromptConstraintResult(
        positive=", ".join(result),
        provenance=tuple(kept_records),
        removed=tuple(dict.fromkeys(removed)),
    )


__all__ = ["PromptConstraintResult", "SAFE_BLOCKED_TAGS", "apply_request_prompt_constraints"]
