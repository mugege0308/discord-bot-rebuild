"""Deterministic, core-dictionary-only variation planning.

This module intentionally does not call the Prompt Designer or the generic
Danbooru searcher.  It creates a small set of verified composition tags after
the common request has already been resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from services.prompt_tag_dictionary import (
    DEFAULT_PROMPT_TAG_PROFILE,
    PromptTagDictionary,
    PromptTagEntry,
    load_prompt_tag_dictionary,
)
from services.prompt_conflict_resolver import classify_text_constraints
MAX_GENERATION_COUNT = 8
VARIATION_OFF = "off"
VARIATION_AUTO = "auto"
VARIATION_MODES = frozenset({VARIATION_OFF, VARIATION_AUTO})

# Keep the planner deliberately narrow.  Every value comes from the verified
# Animagine core dictionary; no English tag is invented here.
_PLANNER_IDS: dict[str, tuple[str, ...]] = {
    "framing": (
        "framing.portrait",
        "framing.upper_body",
        "framing.cowboy_shot",
        "framing.full_body",
    ),
    "camera_horizontal": (
        "camera.horizontal.straight_on",
        "camera.horizontal.three_quarter_view",
        "camera.horizontal.from_side",
        "camera.horizontal.from_behind",
    ),
    "expression": (
        "expression.smile",
        "expression.serious",
        "expression.surprised",
        "expression.blush",
    ),
    "basic_pose": (
        "basic_pose.standing",
        "basic_pose.sitting",
        "basic_pose.walking",
        "basic_pose.kneeling",
    ),
}

_EXPLICIT_CATEGORY_HINTS = {
    "closed mouth": "expression",
    "open mouth": "expression",
    "closed eyes": "expression",
    "looking at viewer": "gaze_direction",
    "looking away": "gaze_direction",
    "looking back": "gaze_direction",
    "looking to the side": "gaze_direction",
}


@dataclass(frozen=True)
class VariationSlot:
    index: int
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"variation_index": self.index, "tags": list(self.tags)}


def _key(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def _entry_for_token(dictionary: PromptTagDictionary, token: object) -> PromptTagEntry | None:
    return dictionary.prompt_token_entry(str(token or ""))


def locked_categories(
    explicit_tags: Sequence[str] = (),
    resolved_prompt_tags: Sequence[str] = (),
    resolved_danbooru_tags: Sequence[str] = (),
    request_text: str = "",
    *,
    dictionary: PromptTagDictionary | None = None,
) -> frozenset[str]:
    """Return categories fixed by user/resolver input.

    The fallback hints cover common explicit general tags that are outside the
    small core JSON dictionary, notably ``closed mouth``.
    """

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    values = (*explicit_tags, *resolved_prompt_tags, *resolved_danbooru_tags)
    categories: set[str] = set()
    for value in values:
        entry = _entry_for_token(prompt_dictionary, value)
        if entry is not None:
            categories.add(entry.category)
            continue
        category = _EXPLICIT_CATEGORY_HINTS.get(_key(value))
        if category:
            categories.add(category)
    categories.update(
        category
        for category in classify_text_constraints(request_text, dictionary=prompt_dictionary)
        if category in {"framing", "camera_horizontal", "expression", "basic_pose", "gaze"}
    )
    return frozenset(categories)


def plan_variation_slots(
    count: int,
    mode: str,
    *,
    start_index: int = 0,
    explicit_tags: Sequence[str] = (),
    resolved_prompt_tags: Sequence[str] = (),
    resolved_danbooru_tags: Sequence[str] = (),
    request_text: str = "",
    reference_mode: str = "none",
    dictionary: PromptTagDictionary | None = None,
) -> tuple[VariationSlot, ...]:
    """Build bounded slots without changing identity, appearance, or LoRAs."""

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_GENERATION_COUNT:
        raise ValueError(f"countは1～{MAX_GENERATION_COUNT}枚で指定してください。")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index < MAX_GENERATION_COUNT:
        raise ValueError("variationの開始位置が不正です。")
    if start_index + count > MAX_GENERATION_COUNT:
        raise ValueError("variationの開始位置と枚数の組み合わせが不正です。")
    normalized_mode = _key(mode or VARIATION_OFF)
    if normalized_mode not in VARIATION_MODES:
        raise ValueError("variationはoffまたはautoで指定してください。")
    if count == 1 or normalized_mode == VARIATION_OFF:
        return tuple(
            VariationSlot(index=index)
            for index in range(start_index + 1, start_index + count + 1)
        )

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    locked = locked_categories(
        explicit_tags,
        resolved_prompt_tags,
        resolved_danbooru_tags,
        request_text,
        dictionary=prompt_dictionary,
    )
    normalized_reference_mode = _key(reference_mode or "none")
    if normalized_reference_mode == "pose":
        # The pose reference owns the pose and its framing.  Expression may
        # still vary, but auto variation must not replace the supplied pose.
        locked = frozenset((*locked, "basic_pose", "framing"))
    elif normalized_reference_mode == "composition":
        # IP-Adapter composition owns the camera/framing relationship.  Keep
        # the reference stable and vary only the remaining safe dimensions.
        locked = frozenset((*locked, "framing", "camera_horizontal", "basic_pose"))
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for category, ids in _PLANNER_IDS.items():
        if category in locked:
            continue
        values = tuple(
            prompt_dictionary.entry_by_id[tag_id].prompt_tag
            for tag_id in ids
            if tag_id in prompt_dictionary.entry_by_id
            and prompt_dictionary.entry_by_id[tag_id].status == "verified"
        )
        if values:
            candidates.append((category, values))

    slots: list[VariationSlot] = []
    for local_index in range(count):
        index = start_index + local_index
        tags: list[str] = []
        for category_index, (_category, values) in enumerate(candidates):
            # Keep slots 1-4 byte-for-byte compatible with the original
            # planner.  For slots 5-8, rotate each category independently so
            # the second group is not a repetition of slots 1-4.
            cycle = index // 4
            rotation = cycle * (category_index + 1)
            tags.append(values[(index + rotation) % len(values)])
        slots.append(VariationSlot(index=index + 1, tags=tuple(tags)))
    return tuple(slots)


def merge_variation_tags(
    positive: str,
    tags: Iterable[str],
    *,
    dictionary: PromptTagDictionary | None = None,
) -> str:
    """Replace only unlocked core exclusive groups and preserve all other tags."""

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    variation_entries = tuple(
        entry
        for tag in tags
        if (entry := _entry_for_token(prompt_dictionary, tag)) is not None
    )
    groups = {entry.exclusive_group for entry in variation_entries if entry.exclusive_group}
    tokens = [token.strip() for token in str(positive or "").split(",") if token.strip()]
    kept: list[str] = []
    for token in tokens:
        entry = _entry_for_token(prompt_dictionary, token)
        if entry is not None and entry.exclusive_group in groups:
            continue
        kept.append(token)
    seen = {_key(token) for token in kept}
    for tag in tags:
        clean = str(tag or "").strip()
        if clean and _key(clean) not in seen:
            kept.append(clean)
            seen.add(_key(clean))
    return ", ".join(kept)


__all__ = [
    "MAX_GENERATION_COUNT",
    "VARIATION_AUTO",
    "VARIATION_MODES",
    "VARIATION_OFF",
    "VariationSlot",
    "locked_categories",
    "merge_variation_tags",
    "plan_variation_slots",
]
