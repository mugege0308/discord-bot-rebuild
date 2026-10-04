"""Optional local Japanese dictionary for canonical Danbooru tag search.

The upstream CSV is runtime data, not a replacement for the canonical SQLite
snapshot.  Entries are only returned after their canonical tag is found in the
existing :class:`DanbooruTagStore`.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable

from services.danbooru_tag_store import (
    DanbooruTagCandidate,
    DanbooruTagStore,
    normalize_search_text,
    normalize_tag_name,
)


DANBOORU_JA_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "danbooru_ja"
DEFAULT_DICTIONARY_PATH = DANBOORU_JA_DATA_DIR / "danbooru-ja.csv"
DEFAULT_META_PATH = DANBOORU_JA_DATA_DIR / "meta.json"
UPSTREAM_REPOSITORY = "https://github.com/PYU224/tagdb-updater"
UPSTREAM_CSV_URL = "https://raw.githubusercontent.com/PYU224/tagdb-updater/main/dist/danbooru-ja.csv"
UPSTREAM_META_URL = "https://raw.githubusercontent.com/PYU224/tagdb-updater/main/dist/meta.json"

UPSTREAM_CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}
SUPPORTED_SEARCH_CATEGORIES = frozenset({"all", "general", "character", "copyright"})


class DictionaryFormatError(ValueError):
    """Raised when an upstream dictionary cannot be safely imported."""


@dataclass(frozen=True)
class JapaneseDictionaryEntry:
    tag: str
    normalized_tag: str
    category: str
    post_count: int
    aliases: tuple[str, ...]
    translation_ja: str | None


@dataclass(frozen=True)
class JapaneseDictionaryMatch:
    candidate: DanbooruTagCandidate
    translation_ja: str | None
    match_type: str


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip())


def _upstream_category(value: object) -> str:
    try:
        return UPSTREAM_CATEGORY_NAMES[int(str(value).strip())]
    except (KeyError, TypeError, ValueError) as exc:
        raise DictionaryFormatError(f"unsupported category: {value!r}") from exc


def _split_aliases(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(item) for item in re.split(r"\s*,\s*", value) if _clean_text(item)))


def parse_dictionary_rows(rows: Iterable[list[str]]) -> tuple[JapaneseDictionaryEntry, ...]:
    entries: list[JapaneseDictionaryEntry] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if not row or not any(str(value).strip() for value in row):
            continue
        if len(row) != 5:
            raise DictionaryFormatError(f"line {line_number}: expected 5 columns")
        tag = _clean_text(row[0])
        if not tag:
            raise DictionaryFormatError(f"line {line_number}: tag is empty")
        if tag.casefold() in {"tag", "name"} and str(row[1]).strip().casefold() in {"category", "category_id"}:
            raise DictionaryFormatError("header row is not supported; expected headerless 5-column CSV")
        try:
            category = _upstream_category(row[1])
            post_count = int(str(row[2]).strip())
        except ValueError as exc:
            raise DictionaryFormatError(f"line {line_number}: invalid category or post count") from exc
        if post_count < 0:
            raise DictionaryFormatError(f"line {line_number}: post count is negative")
        normalized_tag = normalize_tag_name(tag)
        if not normalized_tag or normalized_tag in seen:
            continue
        seen.add(normalized_tag)
        aliases = _split_aliases(row[3])
        translation = _clean_text(row[4]) or None
        entries.append(
            JapaneseDictionaryEntry(
                tag=tag,
                normalized_tag=normalized_tag,
                category=category,
                post_count=post_count,
                aliases=aliases,
                translation_ja=translation,
            )
        )
    if not entries:
        raise DictionaryFormatError("dictionary contains no rows")
    return tuple(entries)


def parse_dictionary_file(path: Path | str) -> tuple[JapaneseDictionaryEntry, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return parse_dictionary_rows(csv.reader(handle))


class DanbooruJapaneseDictionary:
    """Lazily loaded Japanese lookup layer backed by the upstream CSV."""

    def __init__(self, path: Path | str = DEFAULT_DICTIONARY_PATH) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._entries: tuple[JapaneseDictionaryEntry, ...] | None = None
        self._by_tag: dict[str, JapaneseDictionaryEntry] = {}
        self._exact_ja: dict[str, tuple[JapaneseDictionaryEntry, ...]] = {}

    @property
    def available(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    def _ensure_loaded(self) -> tuple[JapaneseDictionaryEntry, ...]:
        with self._lock:
            if self._entries is None:
                if not self.available:
                    self._entries = ()
                else:
                    self._entries = parse_dictionary_file(self.path)
                    self._by_tag = {entry.normalized_tag: entry for entry in self._entries}
                    exact: dict[str, list[JapaneseDictionaryEntry]] = {}
                    for entry in self._entries:
                        if entry.translation_ja:
                            exact.setdefault(normalize_search_text(entry.translation_ja), []).append(entry)
                        for alias in entry.aliases:
                            if any(ord(char) > 127 for char in alias):
                                exact.setdefault(normalize_search_text(alias), []).append(entry)
                    self._exact_ja = {key: tuple(value) for key, value in exact.items()}
            return self._entries

    @property
    def entry_count(self) -> int:
        return len(self._ensure_loaded())

    def translation_for(self, canonical_tag: str) -> str | None:
        entry = self._by_tag.get(normalize_tag_name(canonical_tag))
        if entry is None:
            self._ensure_loaded()
            entry = self._by_tag.get(normalize_tag_name(canonical_tag))
        return entry.translation_ja if entry else None

    def _entry_matches_query(self, entry: JapaneseDictionaryEntry, query: str) -> tuple[int, str] | None:
        query_text = normalize_search_text(query)
        query_tag = normalize_tag_name(query)
        if query_text and entry.translation_ja and normalize_search_text(entry.translation_ja) == query_text:
            return 1000, "dictionary_exact"
        if query_text and any(normalize_search_text(alias) == query_text for alias in entry.aliases):
            return 990, "dictionary_alias_exact"
        if query_tag == entry.normalized_tag:
            return 980, "dictionary_canonical_exact"
        if query_text and entry.translation_ja and normalize_search_text(entry.translation_ja).startswith(query_text):
            return 850, "dictionary_prefix"
        if query_text and any(normalize_search_text(alias).startswith(query_text) for alias in entry.aliases):
            return 840, "dictionary_alias_prefix"
        if query_text and entry.translation_ja and query_text in normalize_search_text(entry.translation_ja):
            return 700, "dictionary_substring"
        if query_text and any(query_text in normalize_search_text(alias) for alias in entry.aliases):
            return 690, "dictionary_alias_substring"
        return None

    def search(
        self,
        query: str,
        *,
        category: str,
        store: DanbooruTagStore,
        limit: int,
    ) -> tuple[JapaneseDictionaryMatch, ...]:
        selected_category = str(category or "all").strip().casefold()
        if selected_category not in SUPPORTED_SEARCH_CATEGORIES:
            return ()
        entries = self._ensure_loaded()
        if not entries:
            return ()
        matches: list[tuple[int, JapaneseDictionaryEntry, DanbooruTagCandidate]] = []
        for entry in entries:
            if selected_category != "all" and entry.category != selected_category:
                continue
            match = self._entry_matches_query(entry, query)
            if match is None:
                continue
            score, _match_type = match
            candidate = store.find_canonical(entry.tag, allowed_categories=(entry.category,))
            if candidate is None:
                continue
            matches.append((score, entry, candidate))
        matches.sort(key=lambda item: (-item[0], -item[2].post_count, item[2].name))
        return tuple(
            JapaneseDictionaryMatch(
                candidate=candidate,
                translation_ja=entry.translation_ja,
                match_type=self._entry_matches_query(entry, query)[1],
            )
            for _score, entry, candidate in matches[:limit]
        )


_default_dictionary: DanbooruJapaneseDictionary | None = None
_default_dictionary_lock = RLock()


def get_default_dictionary() -> DanbooruJapaneseDictionary:
    global _default_dictionary
    configured = os.environ.get("DANBOORU_JA_DICTIONARY_PATH", "").strip()
    path = Path(configured) if configured else DEFAULT_DICTIONARY_PATH
    with _default_dictionary_lock:
        if _default_dictionary is None or _default_dictionary.path != path:
            _default_dictionary = DanbooruJapaneseDictionary(path)
        return _default_dictionary


def clear_dictionary_cache() -> None:
    global _default_dictionary
    with _default_dictionary_lock:
        _default_dictionary = None


def dictionary_metadata(path: Path | str = DEFAULT_META_PATH) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "DANBOORU_JA_DATA_DIR",
    "DEFAULT_DICTIONARY_PATH",
    "DEFAULT_META_PATH",
    "UPSTREAM_CSV_URL",
    "UPSTREAM_META_URL",
    "UPSTREAM_REPOSITORY",
    "DictionaryFormatError",
    "DanbooruJapaneseDictionary",
    "JapaneseDictionaryEntry",
    "JapaneseDictionaryMatch",
    "clear_dictionary_cache",
    "dictionary_metadata",
    "get_default_dictionary",
    "parse_dictionary_file",
    "parse_dictionary_rows",
]
