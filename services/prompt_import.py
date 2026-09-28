"""Deterministic Prompt Import normalization for ``/generate_ai``.

Prompt Import is intentionally different from both the Japanese Prompt
Designer path and the existing English direct-input path.  It does not call
an LLM and it does not apply a semantic allowlist: tags which are not in the
small verified dictionary remain imported content.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from services.character_tag_dictionary import resolve_all_character_tags
from services.direct_prompt_parser import PROMPT_IMPORT_INPUT_MODE, parse_direct_prompt
from services.prompt_generator import PromptGeneratorError
from services.prompt_positive_cleanup import normalize_prompt_tag_key
from services.prompt_tag_dictionary import (
    PromptTagDictionary,
    PromptTagDictionaryError,
    PromptTagEntry,
    load_prompt_tag_dictionary,
)
from services.sd_control import get_negative_prompt
from services.shogou_prompt_designer import extract_prompt_rating, validate_requested_prompt_rating


QUALITY_SUFFIX = ("masterpiece", "high score", "great score", "absurdres")
_QUALITY_KEYS = {
    normalize_prompt_tag_key(tag)
    for tag in (
        *QUALITY_SUFFIX,
        "best quality",
        "high quality",
        "very aesthetic",
        "newest",
    )
}
_RATING_KEYS = {"safe", "sensitive", "nsfw", "explicit", "rating: safe", "rating: sensitive", "rating: nsfw", "rating: explicit"}
_WEIGHTED_TAG_RE = re.compile(
    r"^\((?P<tag>[^(),]+):(?P<weight>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\)$"
)
_COUNT_TAG_RE = re.compile(r"^\d+(?:girl|girls|boy|boys|other)$", re.IGNORECASE)


@dataclass(frozen=True)
class PromptImportResult:
    positive: str
    negative: str
    raw_imported_positive: str
    raw_imported_negative: str
    parsed_imported_tags: tuple[str, ...] = ()
    canonicalized_tags: tuple[str, ...] = ()
    imported_prompt_tags: tuple[str, ...] = ()
    conflict_removed_tags: tuple[str, ...] = ()
    character_resolution: tuple[str, ...] = ()
    framing_resolution: tuple[str, ...] = ()
    tag_provenance: tuple[str, ...] = ()
    llm_added_tags: tuple[str, ...] = ()
    removed_llm_only_tags: tuple[str, ...] = ()


def tokenize_prompt(value: str) -> tuple[str, ...]:
    """Split only commas at nesting depth zero.

    Civitai/Danbooru prompts commonly contain weighted tags such as
    ``(tag:1.2)``.  This small scanner deliberately preserves all characters
    inside (), [] and {} instead of attempting to interpret prompt syntax.
    """

    source = str(value or "")
    tokens: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{",
    }
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
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tuple(tokens)


def _weighted_inner(token: str) -> str | None:
    match = _WEIGHTED_TAG_RE.fullmatch(token.strip())
    return match.group("tag").strip() if match is not None else None


def _comparison_text(token: str) -> str:
    return _weighted_inner(token) or token


def _display_token(token: str) -> str:
    """Normalize simple presentation differences without rewriting weights."""

    token = unicodedata.normalize("NFKC", str(token or "")).strip()
    if not token:
        return ""
    weighted_inner = _weighted_inner(token)
    if weighted_inner is not None:
        normalized_inner = " ".join(weighted_inner.replace("_", " ").split())
        weight = token.rsplit(":", 1)[-1][:-1]
        return f"({normalized_inner}:{weight})"
    return " ".join(token.replace("_", " ").split())


def _tag_key(token: str) -> str:
    return normalize_prompt_tag_key(_comparison_text(token))


def _entry_for_token(token: str, dictionary: PromptTagDictionary) -> PromptTagEntry | None:
    key = _tag_key(token)
    if not key:
        return None
    for entry in dictionary.production_entries:
        forms = (
            entry.prompt_tag,
            entry.danbooru_tag,
            entry.danbooru_tag.replace("_", " "),
            *entry.output_aliases,
        )
        if any(normalize_prompt_tag_key(form) == key for form in forms):
            return entry
    return None


def _dedupe_tokens(tokens: Iterable[str], removed: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_token in tokens:
        token = _display_token(raw_token)
        key = _tag_key(token)
        if not token or not key:
            continue
        if key in seen:
            removed.append(token)
            continue
        output.append(token)
        seen.add(key)
    return output


def _canonicalize_characters(
    tokens: list[str],
) -> tuple[list[str], tuple[str, ...], tuple[str, ...], set[str]]:
    """Merge only verified character + series identity components."""

    canonicalized: list[str] = []
    resolutions: list[str] = []
    identity_keys: set[str] = set()
    working = list(tokens)
    for resolution in resolve_all_character_tags(", ".join(working)):
        character_key = normalize_prompt_tag_key(resolution.character_tag)
        series_key = normalize_prompt_tag_key(resolution.series_tag)
        if not character_key:
            continue
        atomic = resolution.character_tag
        if resolution.series_tag and "(" not in atomic:
            atomic = f"{resolution.character_tag} ({resolution.series_tag})"
        atomic_key = normalize_prompt_tag_key(atomic)
        if atomic_key in {_tag_key(token) for token in working}:
            identity_keys.add(atomic_key)
            continue
        indexes = [
            index
            for index, token in enumerate(working)
            if _tag_key(token) in {character_key, series_key}
        ]
        if not indexes:
            continue
        first_index = min(indexes)
        removed_components = [working[index] for index in indexes]
        working = [token for index, token in enumerate(working) if index not in set(indexes)]
        working.insert(first_index, atomic)
        identity_keys.add(atomic_key)
        resolutions.append(
            f"{', '.join(removed_components)} -> {atomic}"
        )
        canonicalized.append(f"{', '.join(removed_components)} -> {atomic}")
    return working, tuple(canonicalized), tuple(resolutions), identity_keys


def _resolve_framing_conflicts(
    tokens: list[str],
    dictionary: PromptTagDictionary,
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """Keep the first imported tag in each verified exclusive group."""

    selected_by_group: dict[str, str] = {}
    removed: list[str] = []
    resolutions: list[str] = []
    output: list[str] = []
    for token in tokens:
        entry = _entry_for_token(token, dictionary)
        group = entry.exclusive_group if entry is not None else ""
        if not group:
            output.append(entry.prompt_tag if entry is not None and _weighted_inner(token) is None else token)
            continue
        selected = selected_by_group.get(group)
        if selected is None:
            selected_by_group[group] = entry.id
            output.append(entry.prompt_tag)
            continue
        if selected == entry.id:
            removed.append(token)
            continue
        removed.append(token)
        resolutions.append(f"{group}: {selected} kept; {entry.id} removed")
    return output, tuple(removed), tuple(resolutions)


def _is_rating(token: str) -> bool:
    return _tag_key(token) in _RATING_KEYS or _tag_key(token) in {
        "safe",
        "sensitive",
        "nsfw",
        "explicit",
    }


def _is_quality(token: str) -> bool:
    return _tag_key(token) in _QUALITY_KEYS


def _choose_rating(
    tokens: list[str],
    requested_rating: str | None,
    *,
    is_adult: bool,
) -> tuple[str, list[str]]:
    imported_positive = ", ".join(tokens)
    imported_rating = extract_prompt_rating(imported_positive)
    panel_rating = validate_requested_prompt_rating(requested_rating, is_adult=is_adult)
    if panel_rating != "auto":
        return panel_rating, [token for token in tokens if _is_rating(token)]
    if imported_rating is not None:
        if not is_adult and imported_rating != "safe":
            return "safe", [token for token in tokens if _is_rating(token)]
        validate_requested_prompt_rating(imported_rating, is_adult=is_adult)
        return imported_rating, [token for token in tokens if _is_rating(token)]
    return "safe", [token for token in tokens if _is_rating(token)]


def _insert_rating(
    tokens: list[str],
    rating: str,
    identity_keys: set[str],
) -> list[str]:
    cleaned = [token for token in tokens if not _is_rating(token)]
    insert_at = 0
    if cleaned and _COUNT_TAG_RE.fullmatch(_tag_key(cleaned[0])):
        insert_at = 1
        while insert_at < len(cleaned) and _tag_key(cleaned[insert_at]) in identity_keys:
            insert_at += 1
    cleaned.insert(insert_at, rating)
    return cleaned


def _normalize_quality_suffix(tokens: list[str]) -> list[str]:
    body = [token for token in tokens if not _is_quality(token)]
    body.extend(QUALITY_SUFFIX)
    return body


def _normalize_negative_tokens(value: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_token in tokenize_prompt(value):
        token = _display_token(raw_token)
        if not token:
            continue
        if _tag_key(token) == "signature watermark":
            candidates = ("signature", "watermark")
        else:
            candidates = (token,)
        for candidate in candidates:
            key = _tag_key(candidate)
            if key and key not in seen:
                output.append(candidate)
                seen.add(key)
    return output


def _merge_negative(external: str, standard: str) -> str:
    return ", ".join(_normalize_negative_tokens(external) + [
        token
        for token in _normalize_negative_tokens(standard)
        if _tag_key(token) not in {_tag_key(item) for item in _normalize_negative_tokens(external)}
    ])


def resolve_prompt_import(
    text: str,
    *,
    requested_rating: str | None = "auto",
    is_adult: bool = False,
    dictionary: PromptTagDictionary | None = None,
) -> PromptImportResult:
    """Normalize one imported prompt without invoking an LLM."""

    parsed = parse_direct_prompt(str(text or ""), "")
    raw_positive = parsed.positive.strip()
    raw_negative = parsed.negative.strip()
    if not raw_positive:
        raise PromptGeneratorError("Positiveが空です。外部Promptを入力してください。")

    try:
        prompt_dictionary = dictionary or load_prompt_tag_dictionary()
    except PromptTagDictionaryError as exc:
        raise PromptGeneratorError("Prompt辞書を読み込めないため、外部Promptを整理できません。") from exc
    parsed_tags = tokenize_prompt(raw_positive)
    dedupe_removed: list[str] = []
    tokens = _dedupe_tokens(parsed_tags, dedupe_removed)
    tokens, character_changes, character_resolution, identity_keys = _canonicalize_characters(tokens)
    tokens, conflict_removed, framing_resolution = _resolve_framing_conflicts(tokens, prompt_dictionary)

    rating, rating_tags = _choose_rating(tokens, requested_rating, is_adult=is_adult)
    canonicalized = list(character_changes)
    canonicalized.extend(f"duplicate removed: {tag}" for tag in dedupe_removed)
    canonicalized.append(f"rating normalized: {', '.join(rating_tags) or 'none'} -> {rating}")
    tokens = [token for token in tokens if not _is_rating(token)]
    tokens = _insert_rating(tokens, rating, identity_keys)
    quality_removed = [token for token in tokens if _is_quality(token)]
    if quality_removed:
        canonicalized.append("quality suffix normalized: " + ", ".join(quality_removed))
    tokens = _normalize_quality_suffix(tokens)

    final_positive = ", ".join(tokens)
    standard_negative = get_negative_prompt(None)
    final_negative = _merge_negative(raw_negative, standard_negative)
    imported_prompt_tags = tuple(
        token
        for token in tokens
        if not _is_rating(token) and not _is_quality(token)
    )
    tag_provenance = tuple(f"{token} -> imported_prompt" for token in imported_prompt_tags)
    return PromptImportResult(
        positive=final_positive,
        negative=final_negative,
        raw_imported_positive=raw_positive,
        raw_imported_negative=raw_negative,
        parsed_imported_tags=tuple(_display_token(token) for token in parsed_tags),
        canonicalized_tags=tuple(canonicalized),
        imported_prompt_tags=imported_prompt_tags,
        conflict_removed_tags=conflict_removed,
        character_resolution=character_resolution,
        framing_resolution=framing_resolution,
        tag_provenance=tag_provenance,
    )


__all__ = [
    "PROMPT_IMPORT_INPUT_MODE",
    "PromptImportResult",
    "QUALITY_SUFFIX",
    "resolve_prompt_import",
    "tokenize_prompt",
]
