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
    "user_ui": 94,
    "variation_auto": 60,
    "prompt_designer": 50,
    "framing_auto": 55,
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
        "carrying",
        "clothed sex",
        "hetero",
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
        "face close-up",
        "face close up",
        "face focus",
        "upper body",
        "cowboy shot",
        "thigh up",
        "full body",
        "medium shot",
        "wide shot",
        "long shot",
        "distant view",
        "from far away",
    ),
    "body_visibility": (
        "feet visible",
        "feet shown",
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

# This is intentionally narrower than the interaction category above.  Some
# interaction-shaped words (for example ``dancing``) can describe one person;
# only high-confidence relationship tags participate in the people-count
# invariant.
_MULTI_PERSON_INTERACTIONS: Mapping[str, str] = {
    "hugging": "hugging",
    "holding hands": "holding hands",
    "handshake": "handshake",
    "dancing together": "dancing together",
    "kissing": "kissing",
    "kiss": "kiss",
    "embrace": "embrace",
    "embracing": "embracing",
    "carrying": "carrying",
    "clothed sex": "clothed_sex",
    "hetero": "hetero",
}
_MIXED_SEX_INTERACTIONS = frozenset({"hetero"})
_MULTI_PERSON_COUNT_TAGS = frozenset(
    {
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
    }
)
_FEMALE_COUNT_TAGS = frozenset({"1girl", "2girls", "multiple girls", "two girls", "three girls"})
_MALE_COUNT_TAGS = frozenset({"1boy", "2boys", "multiple boys", "two boys", "three boys"})
_SINGLE_SUBJECT_FRAMING = frozenset(
    {
        "portrait",
        "close-up",
        "close up",
        "face close-up",
        "face close up",
        "face focus",
        # For a multi-person request, an automatically inferred upper-body
        # framing is just as likely to crop the partner as a close-up.  Keep
        # explicit/user-requested ``upper body`` intact; this set is only
        # consulted for lower-priority generated framing.
        "upper body",
    }
)


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
    status: str = "suppressed"
    conflicting_tag: str = ""
    conflicting_concept: str = ""
    source_interaction: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "tag": self.tag,
            "source": self.source,
            "reason": self.reason,
            "category": self.category,
            "status": self.status,
            "conflicting_tag": self.conflicting_tag,
            "conflicting_concept": self.conflicting_concept,
            "source_interaction": self.source_interaction,
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
class PromptPeopleSemantics:
    """Conservative people/relationship facts derived from known tags."""

    requires_multiple_people: bool = False
    implies_mixed_sex: bool = False
    explicit_people_count: tuple[str, ...] = ()
    visible_partner_preferred: bool = False
    source_interactions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "requires_multiple_people": self.requires_multiple_people,
            "implies_mixed_sex": self.implies_mixed_sex,
            "explicit_people_count": list(self.explicit_people_count),
            "visible_partner_preferred": self.visible_partner_preferred,
            "source_interactions": list(self.source_interactions),
        }


@dataclass(frozen=True)
class PromptFramingSemantics:
    """Framing tags requested, effective, and suppressed by finalization."""

    requested: tuple[str, ...] = ()
    effective: tuple[str, ...] = ()
    source: tuple[tuple[str, str], ...] = ()
    suppressed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": list(self.requested),
            "effective": list(self.effective),
            "source": [
                {"tag": tag, "source": source}
                for tag, source in self.source
            ],
            "suppressed": list(self.suppressed),
        }


@dataclass(frozen=True)
class PromptConflictResolution:
    positive: str
    provenance: tuple[PromptTagRecord, ...]
    suppressed_tags: tuple[SuppressedPromptTag, ...] = ()
    prompt_conflicts: tuple[PromptConflict, ...] = ()
    locked_categories: frozenset[str] = frozenset()
    people_semantics: PromptPeopleSemantics = PromptPeopleSemantics()
    framing_semantics: PromptFramingSemantics = PromptFramingSemantics()


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
    raw_values = value if isinstance(value, (tuple, list, set, frozenset)) else (value,)
    for raw_value in raw_values:
        for raw in str(raw_value or "").split(","):
            clean = " ".join(raw.strip().split())
            if clean:
                values.append(clean)
    return tuple(values)


def _people_count_requires_multiple(values: Sequence[str]) -> bool:
    normalized = {
        normalize_prompt_constraint(value)
        for value in values
        if normalize_prompt_constraint(value)
    }
    if normalized.intersection(_MULTI_PERSON_COUNT_TAGS):
        return True
    return bool(normalized.intersection(_FEMALE_COUNT_TAGS) and normalized.intersection(_MALE_COUNT_TAGS))


def _canonical_interaction(value: object) -> str:
    return _MULTI_PERSON_INTERACTIONS.get(normalize_prompt_constraint(value), "")


def _single_subject_framing(value: object) -> str:
    normalized = normalize_prompt_constraint(value)
    return normalized if normalized in _SINGLE_SUBJECT_FRAMING else ""


def merge_prompt_framing_semantics(
    *values: PromptFramingSemantics,
) -> PromptFramingSemantics:
    """Merge additive framing traces across the multiple finalization passes."""

    requested: list[str] = []
    effective: list[str] = []
    source: list[tuple[str, str]] = []
    suppressed: list[str] = []
    for value in values:
        for tag in value.requested:
            if tag not in requested:
                requested.append(tag)
        for tag in value.effective:
            if tag not in effective:
                effective.append(tag)
        for item in value.source:
            if item not in source:
                source.append(item)
        for tag in value.suppressed:
            if tag not in suppressed:
                suppressed.append(tag)
    return PromptFramingSemantics(
        requested=tuple(requested),
        effective=tuple(effective),
        source=tuple(source),
        suppressed=tuple(suppressed),
    )


def detect_people_semantics(
    values: Sequence[str] = (),
    *,
    context_text: str = "",
    explicit_values: Sequence[str] = (),
    dictionary: PromptTagDictionary | None = None,
) -> PromptPeopleSemantics:
    """Derive only high-confidence people/relationship constraints.

    ``values`` may include the current Positive tokens so a late default or
    Prompt Designer tag cannot evade the final safety pass.  ``explicit_values``
    is kept separate so provenance can distinguish user counts from observed
    generated tags.
    """

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    observed_values = _tokens(values)
    context_constraints = classify_text_constraints(context_text, dictionary=prompt_dictionary)
    observed_values = (*observed_values, *context_constraints.get("interaction", ()))
    observed_counts = [
        value
        for value in (*_tokens(explicit_values), *observed_values)
        if "people_count" in classify_tag(value, dictionary=prompt_dictionary)
    ]
    source_interactions = tuple(
        dict.fromkeys(
            canonical
            for value in observed_values
            if (canonical := _canonical_interaction(value))
        )
    )
    explicit_counts = tuple(
        dict.fromkeys(
            normalize_prompt_constraint(value)
            for value in (*_tokens(explicit_values), *context_constraints.get("people_count", ()))
            if "people_count" in classify_tag(value, dictionary=prompt_dictionary)
        )
    )
    observed_count_keys = tuple(normalize_prompt_constraint(value) for value in observed_counts)
    mixed_by_count = bool(
        set(observed_count_keys).intersection(_FEMALE_COUNT_TAGS)
        and set(observed_count_keys).intersection(_MALE_COUNT_TAGS)
    )
    requires_multiple = bool(source_interactions) or _people_count_requires_multiple(observed_count_keys)
    mixed = mixed_by_count or bool(set(source_interactions).intersection(_MIXED_SEX_INTERACTIONS))
    return PromptPeopleSemantics(
        requires_multiple_people=requires_multiple,
        implies_mixed_sex=mixed,
        explicit_people_count=explicit_counts,
        visible_partner_preferred=requires_multiple,
        source_interactions=source_interactions,
    )


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
    preferred_framing_tag: str = "",
    protected_framing_tags: Sequence[str] = (),
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
    people_semantics = detect_people_semantics(
        (*explicit_values, *_tokens(positive), *context_values),
        context_text=" ".join(part for part in (request_text, character_text) if str(part or "").strip()),
        explicit_values=explicit_values,
        dictionary=prompt_dictionary,
    )

    known_sources: dict[str, PromptTagRecord] = {}
    for item in base_provenance:
        if not isinstance(item, PromptTagRecord):
            continue
        known_sources.setdefault(normalize_prompt_constraint(item.tag), item)
    variation_keys = {normalize_prompt_constraint(value) for value in variation_tags}
    preferred_framing_key = normalize_prompt_constraint(preferred_framing_tag)
    protected_framing_keys = {
        normalize_prompt_constraint(value)
        for value in protected_framing_tags
        if normalize_prompt_constraint(value)
    }

    records: list[PromptTagRecord] = []
    suppressed: list[SuppressedPromptTag] = []
    kept: list[str] = []
    seen: set[str] = set()
    framing_requested: list[str] = []
    framing_effective: list[str] = []
    framing_sources: list[tuple[str, str]] = []
    framing_suppressed: list[str] = []

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
        framing_key = _single_subject_framing(tag) if "framing" in categories else ""
        if "framing" in categories:
            if tag not in framing_requested:
                framing_requested.append(tag)
            if (tag, source) not in framing_sources:
                framing_sources.append((tag, source))
        locked = bool(categories.intersection(explicit_categories))
        # User-explicit tags and request-derived constraints are retained even
        # when the user's own input contains a contradiction.
        is_user_constraint = source in {"user_explicit", "user_request", "user_ui"}
        tag_semantics = detect_people_semantics((tag,), dictionary=prompt_dictionary)
        non_solo_constraint = tag_semantics.requires_multiple_people
        if (
            preferred_framing_key
            and "framing" in categories
            and key != preferred_framing_key
            and key not in protected_framing_keys
            and not is_user_constraint
        ):
            suppressed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason="overridden_by_multi_person_framing_selector",
                    category="framing",
                    conflicting_tag=preferred_framing_tag,
                    conflicting_concept="multi_person_framing",
                )
            )
            if tag not in framing_suppressed:
                framing_suppressed.append(tag)
            continue
        if (
            framing_key
            and people_semantics.requires_multiple_people
            and framing_key not in protected_framing_keys
            and not is_user_constraint
        ):
            source_interaction = people_semantics.source_interactions[0] if people_semantics.source_interactions else ""
            suppressed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason="conflicts_with_multi_person_framing",
                    category="framing",
                    conflicting_tag=source_interaction,
                    conflicting_concept="requires_multiple_people",
                    source_interaction=source_interaction,
                )
            )
            if tag not in framing_suppressed:
                framing_suppressed.append(tag)
            continue
        if (
            normalize_prompt_constraint(tag) == "solo"
            and non_solo_constraint is False
            and people_semantics.requires_multiple_people
            and not is_user_constraint
        ):
            source_interaction = people_semantics.source_interactions[0] if people_semantics.source_interactions else ""
            conflicting_concept = (
                "requires_multiple_people"
                if source_interaction
                else "explicit_people_count"
            )
            suppressed.append(
                SuppressedPromptTag(
                    tag=tag,
                    source=source,
                    reason=(
                        "conflicts_with_multi_person_interaction"
                        if source_interaction
                        else "conflicts_with_multiple_people_count"
                    ),
                    category="people_count",
                    conflicting_tag=source_interaction,
                    conflicting_concept=conflicting_concept,
                    source_interaction=source_interaction,
                )
            )
            continue
        if (
            locked
            and key not in protected_framing_keys
            and not is_user_constraint
            and _source_priority(source) < SOURCE_PRIORITY["user_request"]
        ):
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
        if "framing" in categories and tag not in framing_effective:
            framing_effective.append(tag)

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
        people_semantics=people_semantics,
        framing_semantics=PromptFramingSemantics(
            requested=tuple(framing_requested),
            effective=tuple(framing_effective),
            source=tuple(framing_sources),
            suppressed=tuple(framing_suppressed),
        ),
    )


def has_non_solo_constraint(
    values: Sequence[str] = (),
    *,
    context_text: str = "",
    dictionary: PromptTagDictionary | None = None,
) -> bool:
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    return detect_people_semantics(
        values,
        context_text=context_text,
        dictionary=prompt_dictionary,
    ).requires_multiple_people


__all__ = [
    "PromptConflict",
    "PromptConflictResolution",
    "PromptFramingSemantics",
    "PromptPeopleSemantics",
    "PromptTagRecord",
    "SOURCE_PRIORITY",
    "SuppressedPromptTag",
    "classify_tag",
    "classify_text_constraints",
    "detect_people_semantics",
    "has_non_solo_constraint",
    "merge_prompt_framing_semantics",
    "normalize_prompt_constraint",
    "resolve_prompt_tag_conflicts",
]
