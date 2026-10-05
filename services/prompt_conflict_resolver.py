"""Deterministic prompt-tag conflict resolution.

The Prompt Designer remains responsible for interpreting natural language.  This
module is deliberately local and conservative: it only recognizes categories
already used by the variation planner plus a small set of well-known structural
constraints.  User-provided constraints are never removed; lower-priority
automatic additions are the only tags that may be suppressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from services.prompt_tag_dictionary import (
    DEFAULT_PROMPT_TAG_PROFILE,
    PromptTagDictionary,
    load_prompt_tag_dictionary,
)


SOURCE_PRIORITY: Mapping[str, int] = {
    "user_explicit": 100,
    "user_request": 95,
    "character_preset": 90,
    "character_confirmed": 90,
    "appearance_resolver": 85,
    "reference_lock": 80,
    "variation_auto": 60,
    "prompt_designer": 50,
    "request_resolver": 50,
    "inference": 40,
    "system": 20,
    "finalizer_default": 20,
    "quality_default": 10,
}

_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "people_count": (
        "solo",
        "1girl",
        "1boy",
        "1other",
        "2girls",
        "2boys",
        "1girl 1boy",
        "multiple girls",
        "multiple boys",
        "multiple people",
        "group",
        "couple",
        "two girls",
        "two boys",
        "three girls",
        "three boys",
    ),
    "interaction": (
        "hugging",
        "holding hands",
        "handshake",
        "dancing together",
        "dancing",
        "kissing",
        "kiss",
        "embrace",
        "embracing",
        "together",
    ),
    "expression": (
        "smile",
        "smiling",
        "grin",
        "serious",
        "surprised",
        "blush",
        "closed eyes",
        "open mouth",
        "closed mouth",
        "neutral expression",
    ),
    "basic_pose": (
        "standing",
        "sitting",
        "kneeling",
        "squatting",
        "lying",
        "walking",
        "running",
        "jumping",
        "crouching",
    ),
    "framing": (
        "portrait",
        "close-up",
        "close up",
        "upper body",
        "cowboy shot",
        "thigh up",
        "full body",
        "medium shot",
    ),
    "camera_horizontal": (
        "straight-on",
        "straight on",
        "front view",
        "front-facing",
        "front facing",
        "three-quarter view",
        "three quarter view",
        "from side",
        "side view",
        "from behind",
        "back view",
    ),
    "gaze": (
        "looking at viewer",
        "looking away",
        "looking back",
        "looking to the side",
    ),
}

_CATEGORY_ALIASES_NORMALIZED = {
    category: tuple(" ".join(alias.casefold().replace("_", " ").split()) for alias in aliases)
    for category, aliases in _CATEGORY_ALIASES.items()
}


@dataclass(frozen=True)
class PromptTagRecord:
    tag: str
    source: str
    category: str = ""


@dataclass(frozen=True)
class SuppressedPromptTag:
    tag: str
    source: str
    reason: str
    category: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "tag": self.tag,
            "source": self.source,
            "reason": self.reason,
            "category": self.category,
        }


@dataclass(frozen=True)
class PromptConflict:
    category: str
    tags: tuple[str, ...]
    sources: tuple[str, ...]
    reason: str = "explicit conflict"

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "tags": list(self.tags),
            "sources": list(self.sources),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PromptConflictResolution:
    positive: str
    provenance: tuple[PromptTagRecord, ...]
    suppressed_tags: tuple[SuppressedPromptTag, ...] = ()
    prompt_conflicts: tuple[PromptConflict, ...] = ()
    locked_categories: frozenset[str] = frozenset()


def normalize_prompt_constraint(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


def _dictionary_category(category: str) -> str:
    if category == "gaze_direction":
        return "gaze"
    return category


def classify_tag(
    value: object,
    *,
    dictionary: PromptTagDictionary | None = None,
) -> frozenset[str]:
    """Return known semantic categories without making a new tag."""

    normalized = normalize_prompt_constraint(value)
    if not normalized:
        return frozenset()
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    entry = prompt_dictionary.prompt_token_entry(str(value or ""))
    if entry is not None:
        return frozenset({_dictionary_category(entry.category)})
    return frozenset(
        category
        for category, aliases in _CATEGORY_ALIASES_NORMALIZED.items()
        if normalized in aliases
    )


def classify_text_constraints(
    text: object,
    *,
    dictionary: PromptTagDictionary | None = None,
) -> dict[str, tuple[str, ...]]:
    """Extract only bounded, recognizable constraints from user text."""

    raw = str(text or "").strip()
    normalized = normalize_prompt_constraint(raw)
    if not normalized:
        return {}
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    result: dict[str, list[str]] = {}
    for category, aliases in _CATEGORY_ALIASES_NORMALIZED.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if not alias:
                continue
            if any(ord(char) > 127 for char in alias):
                found = alias in normalized
            else:
                found = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) is not None
            if found:
                result.setdefault(category, []).append(alias)
    # Also honor aliases defined in the existing verified dictionary.  This is
    # useful for Japanese aliases while keeping the list bounded by the core
    # dictionary rather than inventing a parallel taxonomy.
    for entry in prompt_dictionary.production_entries:
        category = _dictionary_category(entry.category)
        if category not in {"framing", "camera_horizontal", "expression", "basic_pose", "gaze"}:
            continue
        for alias in (entry.prompt_tag, entry.danbooru_tag, *entry.output_aliases, *entry.ja_aliases):
            candidate = normalize_prompt_constraint(alias)
            if candidate and candidate in normalized and candidate not in result.setdefault(category, []):
                result[category].append(candidate)
    return {category: tuple(values) for category, values in result.items()}


def _tokens(value: object) -> tuple[str, ...]:
    values = []
    for raw in str(value or "").split(","):
        clean = " ".join(raw.strip().split())
        if clean:
            values.append(clean)
    return tuple(values)


def _source_priority(source: str) -> int:
    return SOURCE_PRIORITY.get(source, 50)


def _base_source(source: object) -> str:
    normalized = str(source or "").strip()
    return "prompt_designer" if normalized == "request_resolver" else (normalized or "prompt_designer")


def _people_count_values_conflict(values: Sequence[str]) -> bool:
    """Treat complementary one-person tags as a valid multi-person request."""

    normalized = {
        normalize_prompt_constraint(value)
        for value in values
        if normalize_prompt_constraint(value)
    }
    if not normalized:
        return False
    # 1girl + 1boy (or 1other) describes distinct people, not contradictory
    # counts.  solo plus one singular subject is likewise compatible.
    singular_subjects = {"solo", "1girl", "1boy", "1other"}
    return not normalized.issubset(singular_subjects) and len(normalized) > 1


def resolve_prompt_tag_conflicts(
    positive: object,
    *,
    explicit_tags: Sequence[str] = (),
    request_text: str = "",
    character_text: str = "",
    base_provenance: Iterable[PromptTagRecord] = (),
    variation_tags: Sequence[str] = (),
    dictionary: PromptTagDictionary | None = None,
) -> PromptConflictResolution:
    """Keep user intent and suppress only lower-priority conflicting tags."""

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    explicit_values = _tokens(", ".join(str(value or "") for value in explicit_tags))
    explicit_keys = {normalize_prompt_constraint(value) for value in explicit_values}
    text_constraints = classify_text_constraints(
        " ".join(part for part in (request_text, character_text) if str(part or "").strip()),
        dictionary=prompt_dictionary,
    )
    explicit_categories: set[str] = set()
    for value in explicit_values:
        explicit_categories.update(classify_tag(value, dictionary=prompt_dictionary))
    explicit_categories.update(text_constraints)
    context_values = {value for values in text_constraints.values() for value in values}

    known_sources: dict[str, PromptTagRecord] = {}
    for item in base_provenance:
        if not isinstance(item, PromptTagRecord):
            continue
        known_sources.setdefault(normalize_prompt_constraint(item.tag), item)
    variation_keys = {normalize_prompt_constraint(value) for value in variation_tags}

    records: list[PromptTagRecord] = []
    suppressed: list[SuppressedPromptTag] = []
    kept: list[str] = []
    seen: set[str] = set()

    def explicit_text_match(tag: str) -> bool:
        key = normalize_prompt_constraint(tag)
        return key in explicit_keys or key in context_values

    for tag in _tokens(positive):
        key = normalize_prompt_constraint(tag)
        if not key or key in seen:
            continue
        categories = classify_tag(tag, dictionary=prompt_dictionary)
        known = known_sources.get(key)
        if key in explicit_keys:
            source = "user_explicit"
        elif key in variation_keys:
            source = "variation_auto"
        elif explicit_text_match(tag):
            source = "user_request"
        else:
            source = _base_source(known.source if known else "prompt_designer")
        category = next(iter(categories), known.category if known else "")
        locked = bool(categories.intersection(explicit_categories))
        # User-explicit tags and request-derived constraints are retained even
        # when the user's own input contains a contradiction.
        is_user_constraint = source in {"user_explicit", "user_request"}
        non_solo_constraint = any(
            category_name in {"interaction", "people_count"}
            and normalize_prompt_constraint(tag) != "solo"
            for category_name in categories
        )
        if (
            normalize_prompt_constraint(tag) == "solo"
            and non_solo_constraint is False
            and any(
                category_name in {"interaction", "people_count"}
                for category_name in explicit_categories
            )
            and not is_user_constraint
        ):
            suppressed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason="複数人物またはinteraction制約が明示されているため除外",
                    category="people_count",
                )
            )
            continue
        if locked and not is_user_constraint and _source_priority(source) < SOURCE_PRIORITY["user_request"]:
            locked_category = next(iter(categories.intersection(explicit_categories)), category)
            explicit_hint = ", ".join(text_constraints.get(locked_category, ())) or ", ".join(
                value for value in explicit_values if locked_category in classify_tag(value, dictionary=prompt_dictionary)
            )
            suppressed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason=f"{locked_category} is locked by user constraint: {explicit_hint or 'explicit tag'}",
                    category=locked_category,
                )
            )
            continue
        kept.append(tag)
        seen.add(key)
        records.append(PromptTagRecord(tag=tag, source=source, category=category))

    # Explicit-vs-explicit conflicts are warnings only.  Never silently remove
    # one of the user's own tags.
    explicit_by_category: dict[str, list[str]] = {}
    explicit_sources: dict[str, list[str]] = {}
    for value in explicit_values:
        for category in classify_tag(value, dictionary=prompt_dictionary):
            explicit_by_category.setdefault(category, []).append(value)
            explicit_sources.setdefault(category, []).append("user_explicit")
    for category, values in text_constraints.items():
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) > 1:
            explicit_by_category.setdefault(category, []).extend(unique_values)
            explicit_sources.setdefault(category, []).extend("user_request" for _ in unique_values)
    conflicts = tuple(
        PromptConflict(
            category=category,
            tags=tuple(dict.fromkeys(values)),
            sources=tuple(dict.fromkeys(explicit_sources.get(category, ("user_explicit",)))),
        )
        for category, values in explicit_by_category.items()
        if (
            _people_count_values_conflict(values)
            if category == "people_count"
            else len(dict.fromkeys(values)) > 1
        )
    )
    return PromptConflictResolution(
        positive=", ".join(kept),
        provenance=tuple(records),
        suppressed_tags=tuple(dict.fromkeys(suppressed)),
        prompt_conflicts=conflicts,
        locked_categories=frozenset(explicit_categories),
    )


def has_non_solo_constraint(
    values: Sequence[str] = (),
    *,
    context_text: str = "",
    dictionary: PromptTagDictionary | None = None,
) -> bool:
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    one_person_tags = {"1girl", "1boy", "1other"}
    if any(
        category in {"interaction", "people_count"}
        and (
            category == "interaction"
            or normalize_prompt_constraint(value) not in one_person_tags | {"solo"}
        )
        for value in values
        for category in classify_tag(value, dictionary=prompt_dictionary)
    ):
        return True
    constraints = classify_text_constraints(context_text, dictionary=prompt_dictionary)
    return bool(constraints.get("interaction") or any(
        normalize_prompt_constraint(value) not in one_person_tags | {"solo"}
        for value in constraints.get("people_count", ())
    ))


__all__ = [
    "PromptConflict",
    "PromptConflictResolution",
    "PromptTagRecord",
    "SOURCE_PRIORITY",
    "SuppressedPromptTag",
    "classify_tag",
    "classify_text_constraints",
    "has_non_solo_constraint",
    "normalize_prompt_constraint",
    "resolve_prompt_tag_conflicts",
]
