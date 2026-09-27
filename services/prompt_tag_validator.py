"""Validation helpers for the small, model-specific prompt tag dictionaries."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_DANBOORU_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_'\-]*$")
_ALLOWED_STATUSES = {"verified", "candidate", "deprecated"}


@dataclass(frozen=True)
class PromptTagValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return not self.errors


def _normalise_alias(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def validate_prompt_tag_dictionary_document(
    document: Mapping[str, Any],
    *,
    expected_profile: str | None = "animagine_xl_4",
) -> PromptTagValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    version = document.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append("version must be a non-empty string")

    profile = document.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        errors.append("profile must be a non-empty string")
    elif expected_profile and profile != expected_profile:
        errors.append(f"profile must be {expected_profile!r}, got {profile!r}")

    entries = document.get("entries")
    if not isinstance(entries, list):
        errors.append("entries must be an array")
        return PromptTagValidationReport(tuple(errors), tuple(warnings))

    if len(entries) < 30:
        errors.append(f"entries must contain at least 30 items, got {len(entries)}")

    ids: set[str] = set()
    danbooru_tags: set[str] = set()
    aliases: dict[str, str] = {}
    exclusive_groups: dict[str, list[str]] = {}

    for index, raw_entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(raw_entry, Mapping):
            errors.append(f"{prefix} must be an object")
            continue

        required = ("id", "category", "danbooru_tag", "prompt_tag", "ja_aliases", "excludes", "status", "profile")
        for key in required:
            if key not in raw_entry:
                errors.append(f"{prefix} missing {key}")

        entry_id = raw_entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
            entry_id = f"<invalid:{index}>"
        elif entry_id in ids:
            errors.append(f"duplicate entry id: {entry_id}")
        ids.add(entry_id)

        category = raw_entry.get("category")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"{prefix}.category must be a non-empty string")

        danbooru_tag = raw_entry.get("danbooru_tag")
        if not isinstance(danbooru_tag, str) or not _DANBOORU_TAG_RE.fullmatch(danbooru_tag):
            errors.append(f"{prefix}.danbooru_tag is not a valid Danbooru tag: {danbooru_tag!r}")
        elif danbooru_tag in danbooru_tags:
            errors.append(f"duplicate danbooru_tag: {danbooru_tag}")
        else:
            danbooru_tags.add(danbooru_tag)

        prompt_tag = raw_entry.get("prompt_tag")
        if not isinstance(prompt_tag, str) or not prompt_tag.strip() or "," in prompt_tag:
            errors.append(f"{prefix}.prompt_tag must be one non-empty tag without commas")

        raw_aliases = raw_entry.get("ja_aliases")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            errors.append(f"{prefix}.ja_aliases must be a non-empty array")
        else:
            for alias in raw_aliases:
                normalised = _normalise_alias(alias)
                if not normalised:
                    errors.append(f"{prefix}.ja_aliases contains an empty alias")
                    continue
                previous = aliases.get(normalised)
                if previous is not None and previous != entry_id:
                    errors.append(f"alias collision: {alias!r} maps to both {previous} and {entry_id}")
                else:
                    aliases[normalised] = entry_id

        status = raw_entry.get("status")
        if status not in _ALLOWED_STATUSES:
            errors.append(f"{prefix}.status must be one of {sorted(_ALLOWED_STATUSES)}")
        elif status != "verified":
            warnings.append(f"{entry_id} has non-production status {status}")

        entry_profile = raw_entry.get("profile")
        if entry_profile != profile:
            errors.append(f"{prefix}.profile must match document profile {profile!r}")

        exclusive_group = raw_entry.get("exclusive_group", "")
        if exclusive_group and isinstance(exclusive_group, str):
            exclusive_groups.setdefault(exclusive_group, []).append(entry_id)

        for key in ("excludes", "output_aliases"):
            values = raw_entry.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                errors.append(f"{prefix}.{key} must be an array of non-empty strings")

    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        entry_id = raw_entry.get("id")
        for excluded in raw_entry.get("excludes", []) if isinstance(raw_entry.get("excludes", []), list) else []:
            if excluded not in ids:
                errors.append(f"{entry_id} excludes unknown entry {excluded}")

    return PromptTagValidationReport(tuple(errors), tuple(warnings))


def validate_prompt_tag_dictionary_file(
    path: str | Path,
    *,
    expected_profile: str | None = "animagine_xl_4",
) -> PromptTagValidationReport:
    file_path = Path(path)
    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PromptTagValidationReport((f"could not read {file_path}: {exc}",), ())
    if not isinstance(document, Mapping):
        return PromptTagValidationReport(("dictionary root must be an object",), ())
    return validate_prompt_tag_dictionary_document(document, expected_profile=expected_profile)
