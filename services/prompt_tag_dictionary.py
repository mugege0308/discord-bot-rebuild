"""Load the versioned prompt-tag data used by Animagine's Japanese path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from services.prompt_tag_validator import validate_prompt_tag_dictionary_document


PROMPT_TAG_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "prompt_tags"
DEFAULT_PROMPT_TAG_PROFILE = "animagine_xl_4"
DEFAULT_PROMPT_TAG_DICTIONARY_PATH = PROMPT_TAG_DATA_DIR / "animagine_xl_4.json"


class PromptTagDictionaryError(ValueError):
    """Raised when a prompt-tag dictionary cannot be safely loaded."""


@dataclass(frozen=True)
class PromptTagEntry:
    id: str
    category: str
    danbooru_tag: str
    prompt_tag: str
    ja_aliases: tuple[str, ...]
    excludes: tuple[str, ...]
    output_aliases: tuple[str, ...]
    exclusive_group: str = ""
    status: str = "verified"
    profile: str = DEFAULT_PROMPT_TAG_PROFILE


@dataclass(frozen=True)
class PromptTagDictionary:
    version: str
    profile: str
    entries: tuple[PromptTagEntry, ...]

    @property
    def entry_by_id(self) -> dict[str, PromptTagEntry]:
        return {entry.id: entry for entry in self.entries}

    @property
    def production_entries(self) -> tuple[PromptTagEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "verified")

    @property
    def entry_by_danbooru_tag(self) -> dict[str, PromptTagEntry]:
        return {entry.danbooru_tag.casefold(): entry for entry in self.entries}

    def prompt_token_entry(self, token: str) -> PromptTagEntry | None:
        normalized = " ".join(str(token).casefold().replace("_", " ").split())
        if not normalized:
            return None
        for entry in self.entries:
            forms = {
                entry.prompt_tag.casefold(),
                entry.danbooru_tag.casefold().replace("_", " "),
                *(alias.casefold() for alias in entry.output_aliases),
            }
            if normalized in forms:
                return entry
        return None


def _entry_from_document(raw_entry: Mapping[str, Any]) -> PromptTagEntry:
    return PromptTagEntry(
        id=str(raw_entry["id"]),
        category=str(raw_entry["category"]),
        danbooru_tag=str(raw_entry["danbooru_tag"]),
        prompt_tag=str(raw_entry["prompt_tag"]),
        ja_aliases=tuple(str(value) for value in raw_entry.get("ja_aliases", ())),
        excludes=tuple(str(value) for value in raw_entry.get("excludes", ())),
        output_aliases=tuple(str(value) for value in raw_entry.get("output_aliases", ())),
        exclusive_group=str(raw_entry.get("exclusive_group", "")),
        status=str(raw_entry.get("status", "verified")),
        profile=str(raw_entry.get("profile", DEFAULT_PROMPT_TAG_PROFILE)),
    )


@lru_cache(maxsize=4)
def load_prompt_tag_dictionary(profile: str = DEFAULT_PROMPT_TAG_PROFILE) -> PromptTagDictionary:
    if profile != DEFAULT_PROMPT_TAG_PROFILE:
        raise PromptTagDictionaryError(f"未対応のprompt tag profileです: {profile}")
    path = PROMPT_TAG_DATA_DIR / f"{profile}.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptTagDictionaryError(f"prompt tag辞書を読み込めません: {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise PromptTagDictionaryError("prompt tag辞書のルートはobjectである必要があります")
    report = validate_prompt_tag_dictionary_document(document, expected_profile=profile)
    if not report.is_usable:
        raise PromptTagDictionaryError("prompt tag辞書の検証に失敗しました: " + "; ".join(report.errors))
    raw_entries = document.get("entries")
    assert isinstance(raw_entries, list)
    entries = tuple(_entry_from_document(raw_entry) for raw_entry in raw_entries if isinstance(raw_entry, Mapping))
    return PromptTagDictionary(
        version=str(document["version"]),
        profile=str(document["profile"]),
        entries=entries,
    )
