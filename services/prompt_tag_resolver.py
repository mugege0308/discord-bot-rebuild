"""Resolve loose Japanese composition requests into verified prompt tags."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from services.prompt_tag_dictionary import (
    DEFAULT_PROMPT_TAG_PROFILE,
    PromptTagDictionary,
    PromptTagEntry,
    load_prompt_tag_dictionary,
)


_COMPOSITION_PRESET_TAGS = {
    "face_closeup": "framing.portrait",
    "portrait": "framing.portrait",
    "upper_body": "framing.upper_body",
    "cowboy_shot": "framing.cowboy_shot",
    "medium_shot": "framing.medium_shot",
    "full_body": "framing.full_body",
    "full_body_tight": "framing.full_body",
}
_COMPOSITION_CUES = (
    "構図", "ポーズ", "アングル", "向き", "視線", "目線", "表情", "振り返", "振り向",
    "座", "立", "膝", "しゃが", "横たわ", "寝そべ", "歩", "走", "笑", "驚",
    "照れ", "赤面", "動き", "ダイナミック", "上から", "下から", "正面", "横から", "後ろ",
)
_QUALITY_SUFFIX_TAGS = {
    "masterpiece",
    "best quality",
    "high quality",
    "high score",
    "great score",
    "absurdres",
    "newest",
    "very aesthetic",
    "anime style",
}


@dataclass(frozen=True)
class CompositionTagApplication:
    """Result of applying an explicit composition preset to a final prompt."""

    positive: str
    composition_preset: str
    composition_tag_applied: str | None = None
    conflict_removed_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptTagMatch:
    entry: PromptTagEntry
    matched_alias: str
    start: int
    source: str = "user"


@dataclass(frozen=True)
class PromptTagResolution:
    dictionary_version: str
    profile: str
    matches: tuple[PromptTagMatch, ...] = ()
    conflicts: tuple[str, ...] = ()
    unresolved_japanese_phrases: tuple[str, ...] = ()

    @property
    def entries(self) -> tuple[PromptTagEntry, ...]:
        return tuple(match.entry for match in self.matches)

    @property
    def resolved_tag_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    @property
    def resolved_prompt_tags(self) -> tuple[str, ...]:
        return tuple(entry.prompt_tag for entry in self.entries)

    @property
    def has_resolved_tags(self) -> bool:
        return bool(self.matches)


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _split_top_level_prompt_tags(value: str) -> tuple[str, ...]:
    """Split prompt tags without breaking commas nested in prompt syntax."""

    source = str(value or "")
    tokens: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    escaped = False
    for char in source:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in depths:
            depths[char] += 1
            current.append(char)
            continue
        if char in closing:
            opener = closing[char]
            if depths[opener] > 0:
                depths[opener] -= 1
            current.append(char)
            continue
        if char == "," and not any(depths.values()):
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current.clear()
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tuple(tokens)


def _composition_entry_for_token(
    token: str,
    dictionary: PromptTagDictionary,
) -> PromptTagEntry | None:
    entry = dictionary.prompt_token_entry(token)
    if entry is not None:
        return entry

    # Treat a weighted known framing tag as the same dictionary entry for
    # conflict removal, while leaving all unknown weighted tags untouched.
    stripped = token.strip()
    if stripped.startswith("(") and stripped.endswith(")") and ":" in stripped:
        inner, _, weight = stripped[1:-1].rpartition(":")
        if inner.strip() and weight.strip():
            return dictionary.prompt_token_entry(inner.strip())
    return None


def apply_composition_preset_to_positive(
    positive: str,
    composition_preset: str | None,
    *,
    dictionary: PromptTagDictionary | None = None,
) -> CompositionTagApplication:
    """Apply the selected composition tag at the final prompt boundary.

    Only verified dictionary entries in the same exclusive group as the
    selected composition are removed.  Unknown imported/user tags are kept.
    The selected canonical tag is inserted immediately before trailing quality
    tags and appears at most once.
    """

    preset = _normalise(composition_preset or "none").replace(" ", "_")
    preset_id = _COMPOSITION_PRESET_TAGS.get(preset)
    if preset_id is None:
        return CompositionTagApplication(
            positive=positive,
            composition_preset=preset or "none",
        )

    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    selected_entry = prompt_dictionary.entry_by_id.get(preset_id)
    if selected_entry is None or selected_entry.status != "verified":
        return CompositionTagApplication(
            positive=positive,
            composition_preset=preset,
        )

    kept: list[str] = []
    removed: list[str] = []
    for token in _split_top_level_prompt_tags(positive):
        entry = _composition_entry_for_token(token, prompt_dictionary)
        if entry is not None and entry.id == selected_entry.id:
            # Reinsert the canonical unweighted tag once below.
            continue
        if (
            entry is not None
            and selected_entry.exclusive_group
            and entry.exclusive_group == selected_entry.exclusive_group
        ):
            removed.append(token)
            continue
        kept.append(token)

    quality_start = len(kept)
    while (
        quality_start > 0
        and _normalise(kept[quality_start - 1]) in _QUALITY_SUFFIX_TAGS
    ):
        quality_start -= 1
    kept.insert(quality_start, selected_entry.prompt_tag)
    return CompositionTagApplication(
        positive=", ".join(kept),
        composition_preset=preset,
        composition_tag_applied=selected_entry.prompt_tag,
        conflict_removed_tags=tuple(removed),
    )


def _alias_match(text: str, alias: str) -> int | None:
    normalized_alias = _normalise(alias)
    if not normalized_alias:
        return None
    if any(ord(char) > 127 for char in normalized_alias):
        start = text.find(normalized_alias)
        return start if start >= 0 else None
    pattern = rf"(?<![\w-]){re.escape(normalized_alias)}(?![\w-])"
    match = re.search(pattern, text)
    return match.start() if match is not None else None


def _best_alias_match(text: str, entry: PromptTagEntry) -> tuple[str, int] | None:
    candidates: list[tuple[int, int, str]] = []
    for alias in entry.ja_aliases:
        start = _alias_match(text, alias)
        if start is not None:
            candidates.append((start, -len(_normalise(alias)), alias))
    if not candidates:
        return None
    start, _negative_length, alias = min(candidates)
    return alias, start


def _split_phrases(text: str) -> Iterable[str]:
    for phrase in re.split(r"[、,，。.!！?？\n]+", text):
        cleaned = phrase.strip()
        if cleaned:
            yield cleaned


def _unresolved_phrases(text: str, matched_phrases: Iterable[str]) -> tuple[str, ...]:
    matched = set(matched_phrases)
    unresolved: list[str] = []
    for phrase in _split_phrases(text):
        if phrase in matched:
            continue
        if any(cue in phrase for cue in _COMPOSITION_CUES):
            unresolved.append(phrase[:120])
    return tuple(dict.fromkeys(unresolved))


def _select_exclusive_matches(matches: Iterable[PromptTagMatch]) -> tuple[tuple[PromptTagMatch, ...], tuple[str, ...]]:
    grouped: dict[str, list[PromptTagMatch]] = {}
    without_group: list[PromptTagMatch] = []
    for match in matches:
        group = match.entry.exclusive_group
        if group:
            grouped.setdefault(group, []).append(match)
        else:
            without_group.append(match)

    selected = list(without_group)
    conflicts: list[str] = []
    for group, group_matches in grouped.items():
        winner = min(
            group_matches,
            key=lambda item: (item.start, -len(_normalise(item.matched_alias)), item.entry.id),
        )
        selected.append(winner)
        for loser in group_matches:
            if loser.entry.id != winner.entry.id:
                conflicts.append(f"{group}:{loser.entry.id}->{winner.entry.id}")

    selected_ids = {match.entry.id for match in selected}
    changed = True
    while changed:
        changed = False
        for match in tuple(selected):
            for excluded_id in match.entry.excludes:
                if excluded_id in selected_ids:
                    selected = [candidate for candidate in selected if candidate.entry.id != excluded_id]
                    selected_ids.remove(excluded_id)
                    conflicts.append(f"{match.entry.id}:{excluded_id}")
                    changed = True
    selected.sort(key=lambda item: (item.entry.category, item.start, item.entry.id))
    return tuple(selected), tuple(dict.fromkeys(conflicts))


def resolve_prompt_tags(
    text: str,
    *,
    composition_preset: str = "none",
    dictionary: PromptTagDictionary | None = None,
) -> PromptTagResolution:
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(DEFAULT_PROMPT_TAG_PROFILE)
    normalized_text = _normalise(str(text or ""))
    matches: list[PromptTagMatch] = []
    matched_phrases: list[str] = []
    for entry in prompt_dictionary.production_entries:
        found = _best_alias_match(normalized_text, entry)
        if found is None:
            continue
        alias, start = found
        matches.append(PromptTagMatch(entry=entry, matched_alias=alias, start=start))
        matched_phrases.append(next((phrase for phrase in _split_phrases(str(text or "")) if alias in phrase), ""))

    user_categories = {match.entry.category for match in matches}
    if composition_preset in _COMPOSITION_PRESET_TAGS and "framing" not in user_categories:
        preset_id = _COMPOSITION_PRESET_TAGS[composition_preset]
        preset_entry = prompt_dictionary.entry_by_id.get(preset_id)
        if preset_entry is not None and preset_entry.status == "verified":
            matches.append(
                PromptTagMatch(
                    entry=preset_entry,
                    matched_alias=f"composition_preset:{composition_preset}",
                    start=10**9,
                    source="composition_preset",
                )
            )

    selected, conflicts = _select_exclusive_matches(matches)
    entry_order = {entry.id: index for index, entry in enumerate(prompt_dictionary.entries)}
    selected = tuple(sorted(selected, key=lambda item: entry_order.get(item.entry.id, 10**9)))
    return PromptTagResolution(
        dictionary_version=prompt_dictionary.version,
        profile=prompt_dictionary.profile,
        matches=selected,
        conflicts=conflicts,
        unresolved_japanese_phrases=_unresolved_phrases(str(text or ""), matched_phrases),
    )


def build_prompt_tag_instruction(resolution: PromptTagResolution) -> str:
    if not resolution.has_resolved_tags and not resolution.unresolved_japanese_phrases:
        return ""
    lines = [
        "## Deterministic prompt-tag resolution (authoritative)",
        f"- dictionary version: {resolution.dictionary_version}",
        "- These are verified Animagine XL 4.0 prompt tags resolved from the request or the selected composition preset.",
        "- Keep every resolved prompt tag exactly as written in Positive; do not translate, replace, or invent a competing tag in the same exclusive group.",
        "- Do not infer unrequested objects or actions from these tags. In particular, sitting does not imply chair, floor, or crossed legs; dynamic pose does not imply running or jumping.",
    ]
    if resolution.resolved_prompt_tags:
        lines.append("- resolved prompt tags: " + ", ".join(resolution.resolved_prompt_tags))
        lines.append("- resolved tag IDs: " + ", ".join(resolution.resolved_tag_ids))
    if resolution.conflicts:
        lines.append("- conflicting user cues were resolved safely: " + ", ".join(resolution.conflicts))
    if resolution.unresolved_japanese_phrases:
        lines.append(
            "- unresolved composition phrases may be interpreted conservatively by the existing LLM, but must not be converted into invented dictionary tags: "
            + " / ".join(resolution.unresolved_japanese_phrases)
        )
    return "\n".join(lines)


def apply_prompt_tag_resolution(
    positive: str,
    resolution: PromptTagResolution,
    *,
    dictionary: PromptTagDictionary | None = None,
) -> str:
    """Canonicalize controlled tags and append the authoritative resolutions.

    Only tags known to this small dictionary are filtered. All unrelated LLM
    output (characters, outfits, backgrounds, props, and general tags) stays
    untouched.
    """

    if not resolution.has_resolved_tags:
        return positive.strip()
    prompt_dictionary = dictionary or load_prompt_tag_dictionary(resolution.profile)
    selected_ids = set(resolution.resolved_tag_ids)
    suppress_unrequested_basic_pose = (
        "dynamic_composition.dynamic_pose" in selected_ids
        and not any(entry.category == "basic_pose" for entry in resolution.entries)
    )
    kept: list[str] = []
    seen: set[str] = set()
    for raw_token in positive.split(","):
        token = raw_token.strip()
        if not token:
            continue
        entry = prompt_dictionary.prompt_token_entry(token)
        if entry is not None:
            if entry.id in selected_ids:
                continue
            if entry.exclusive_group and any(
                selected.exclusive_group == entry.exclusive_group
                for selected in resolution.entries
            ):
                continue
            if suppress_unrequested_basic_pose and entry.category == "basic_pose":
                continue
        key = token.casefold()
        if key in seen:
            continue
        kept.append(token)
        seen.add(key)

    resolved_tokens: list[str] = []
    for entry in resolution.entries:
        key = entry.prompt_tag.casefold()
        if key not in seen:
            resolved_tokens.append(entry.prompt_tag)
            seen.add(key)
    insert_at = len(kept)
    while insert_at > 0 and kept[insert_at - 1].casefold() in _QUALITY_SUFFIX_TAGS:
        insert_at -= 1
    kept[insert_at:insert_at] = resolved_tokens
    return ", ".join(kept)
