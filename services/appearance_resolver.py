"""Deterministic appearance-text resolution through the Danbooru tag DB.

Appearance text is intentionally narrower than the normal request resolver.
It recognizes only explicit hair, eye, skin, and stable appearance phrases,
then accepts a canonical active ``general`` tag from the existing SQLite
store.  Unknown prose is ignored rather than translated or invented.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from services.danbooru_tag_store import (
    DEFAULT_DANBOORU_DB_PATH,
    DanbooruTagStore,
)


@dataclass(frozen=True)
class AppearanceResolution:
    input_text: str
    tags: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    resolver: str = "appearance_resolver"
    queries: tuple[str, ...] = ()
    matched_fragments: tuple[str, ...] = ()

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "input_text": self.input_text,
            "tags": list(self.tags),
            "unresolved": list(self.unresolved),
            "resolver": self.resolver,
            "queries": list(self.queries),
            "matched_fragments": list(self.matched_fragments),
        }


@dataclass(frozen=True)
class _AppearanceHint:
    category: str
    fragments: tuple[str, ...]
    queries: tuple[str, ...]


_COLOR_WORDS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "blonde",
        "golden",
        "gray",
        "grey",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "violet",
        "white",
        "yellow",
    }
)


# Longer phrases are listed first so a phrase such as 黒髪 is not treated as
# generic prose before its exact appearance meaning is recognized.
_HINTS: tuple[_AppearanceHint, ...] = (
    _AppearanceHint("hair_color", ("black hair", "黒髪", "黒い髪", "髪が黒い", "髪が黒"), ("black hair",)),
    _AppearanceHint("hair_color", ("blue hair", "青髪", "青い髪", "髪が青い", "髪が青"), ("blue hair",)),
    _AppearanceHint("hair_color", ("brown hair", "茶髪", "茶色の髪", "髪が茶色"), ("brown hair",)),
    _AppearanceHint("hair_color", ("blonde hair", "金髪", "金色の髪", "髪が金色"), ("blonde hair",)),
    _AppearanceHint("hair_color", ("silver hair", "銀髪", "シルバー髪", "銀色の髪"), ("silver hair", "grey hair", "gray hair")),
    _AppearanceHint("hair_color", ("white hair", "白髪", "白い髪", "髪が白い", "髪が白"), ("white hair",)),
    _AppearanceHint("hair_color", ("red hair", "赤髪", "赤い髪", "髪が赤い", "髪が赤"), ("red hair",)),
    _AppearanceHint("hair_color", ("pink hair", "桃髪", "ピンク髪", "ピンクの髪"), ("pink hair",)),
    _AppearanceHint("hair_color", ("purple hair", "紫髪", "紫の髪", "髪が紫"), ("purple hair",)),
    _AppearanceHint("hair_color", ("green hair", "緑髪", "緑の髪", "髪が緑"), ("green hair",)),
    _AppearanceHint("hair_style", ("long hair", "長髪", "ロングヘア", "長い髪"), ("long hair",)),
    _AppearanceHint("hair_style", ("short hair", "短髪", "ショートヘア", "短い髪"), ("short hair",)),
    _AppearanceHint("hair_style", ("bob cut", "ボブカット", "ボブ"), ("bob cut",)),
    _AppearanceHint("hair_style", ("ponytail", "ポニーテール"), ("ponytail",)),
    _AppearanceHint("hair_style", ("twintails", "ツインテール"), ("twintails", "twin tails")),
    _AppearanceHint("hair_style", ("braid", "三つ編み", "みつあみ"), ("braid",)),
    _AppearanceHint("eye_color", ("black eyes", "黒目", "黒い目", "目が黒い"), ("black eyes",)),
    _AppearanceHint("eye_color", ("blue eyes", "青目", "青い目", "青い瞳", "目が青い", "瞳が青い"), ("blue eyes",)),
    _AppearanceHint("eye_color", ("brown eyes", "茶目", "茶色の目", "目が茶色"), ("brown eyes",)),
    _AppearanceHint("eye_color", ("green eyes", "緑目", "緑の目", "目が緑"), ("green eyes",)),
    _AppearanceHint("eye_color", ("purple eyes", "紫目", "紫の目", "紫の瞳", "目が紫"), ("purple eyes",)),
    _AppearanceHint("eye_color", ("red eyes", "赤目", "赤い目", "赤い瞳", "目が赤い"), ("red eyes",)),
    _AppearanceHint("eye_color", ("pink eyes", "ピンクの目", "目がピンク"), ("pink eyes",)),
    _AppearanceHint("eye_color", ("golden eyes", "金目", "金色の目", "目が金色"), ("golden eyes",)),
    _AppearanceHint("skin_tone", ("pale skin", "色白", "白い肌", "肌が白い"), ("pale skin",)),
    _AppearanceHint("stable_appearance", ("freckles", "そばかす"), ("freckles",)),
    _AppearanceHint("stable_appearance", ("mole", "ほくろ"), ("mole",)),
    _AppearanceHint("stable_appearance", ("pointy ears", "尖った耳", "とがった耳"), ("pointy ears",)),
    _AppearanceHint("stable_appearance", ("horns", "角がある", "角"), ("horns",)),
    _AppearanceHint("stable_appearance", ("tail", "尻尾", "しっぽ"), ("tail",)),
    _AppearanceHint("constant_accessory", ("glasses", "眼鏡", "メガネ"), ("glasses",)),
    _AppearanceHint("constant_accessory", ("eyepatch", "眼帯"), ("eyepatch",)),
    _AppearanceHint("constant_accessory", ("hair ribbon", "髪リボン", "髪飾り", "リボン"), ("hair ribbon",)),
)


def _normalise_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _contains_fragment(text: str, fragment: str) -> bool:
    normalized_text = text.casefold()
    normalized_fragment = fragment.casefold()
    if any(ord(char) > 127 for char in normalized_fragment):
        return normalized_fragment in normalized_text
    return re.search(rf"(?<![a-z0-9_-]){re.escape(normalized_fragment)}(?![a-z0-9_-])", normalized_text) is not None


def appearance_tag_category(tag: str) -> str | None:
    """Classify a canonical appearance tag for precedence handling."""

    key = " ".join(str(tag or "").casefold().replace("_", " ").split())
    words = set(key.split())
    if "hair" in words:
        if words & _COLOR_WORDS:
            return "hair_color"
        return "hair_style"
    if "eyes" in words or "eye" in words:
        if words & _COLOR_WORDS:
            return "eye_color"
        return "eye_shape"
    if "skin" in words:
        return "skin_tone"
    if any(marker in key for marker in ("freckles", "mole", "pointy ears", "horn", "tail")):
        return "stable_appearance"
    if any(marker in key for marker in ("glasses", "eyepatch", "hair ribbon", "hairband", "hairpin")):
        return "constant_accessory"
    return None


def _default_store() -> DanbooruTagStore:
    # This optional runtime override keeps generated catalogs outside the
    # repository usable without changing the checked-in code or .env file.
    configured_path = os.getenv("DANBOORU_TAG_DB_PATH", "").strip()
    return DanbooruTagStore(configured_path or DEFAULT_DANBOORU_DB_PATH)


def resolve_appearance_text(
    text: str,
    *,
    store: DanbooruTagStore | None = None,
) -> AppearanceResolution:
    """Resolve explicit appearance fragments to active general DB tags only."""

    clean = _normalise_text(text)
    if not clean:
        return AppearanceResolution(input_text="")

    search_store = store or _default_store()
    if not search_store.available:
        return AppearanceResolution(
            input_text=clean,
            unresolved=(clean,),
            resolver="appearance_resolver_unavailable",
        )

    tags: list[str] = []
    unresolved: list[str] = []
    queries: list[str] = []
    matched_fragments: list[str] = []
    seen_categories: set[str] = set()

    for hint in _HINTS:
        matched = next((fragment for fragment in hint.fragments if _contains_fragment(clean, fragment)), None)
        if matched is None:
            continue
        matched_fragments.append(matched)
        category = hint.category
        if category in seen_categories:
            continue
        seen_categories.add(category)
        selected = None
        for query in hint.queries:
            queries.append(query)
            candidates = search_store.search(query, limit=8, allowed_categories=("general",))
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.category == "general"
                    and not candidate.is_deprecated
                    and candidate.match_type in {"exact", "alias_exact"}
                    and candidate.prompt_tag.strip()
                ),
                None,
            )
            if selected is not None:
                break
        if selected is None:
            unresolved.append(matched)
            continue
        tags.append(selected.prompt_tag.strip())

    return AppearanceResolution(
        input_text=clean,
        tags=tuple(dict.fromkeys(tags)),
        unresolved=tuple(dict.fromkeys(unresolved)),
        queries=tuple(dict.fromkeys(queries)),
        matched_fragments=tuple(dict.fromkeys(matched_fragments)),
    )


__all__ = [
    "AppearanceResolution",
    "appearance_tag_category",
    "resolve_appearance_text",
]
