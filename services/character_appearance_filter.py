"""Suppress unrequested identity-appearance completions for known characters.

When a Danbooru canonical character tag is authoritative, an LLM-generated
hair/eye/accessory description can silently change the character's identity.
This module removes only appearance-like tags unless the request explicitly
mentions that appearance or a verified resolver selected the tag.

Scene tags such as pose, expression, clothing, background, and lighting are
deliberately outside this filter.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_COLOR_WORDS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "cyan",
        "gold",
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
_HAIR_STYLE_WORDS = frozenset(
    {
        "ahoge",
        "bangs",
        "bob",
        "braid",
        "braids",
        "curly",
        "fluffy",
        "long",
        "medium",
        "messy",
        "ponytail",
        "short",
        "straight",
        "swept",
        "tousled",
        "twintails",
        "voluminous",
        "wavy",
    }
)
_EYE_SHAPE_WORDS = frozenset(
    {"big", "clear", "droopy", "large", "narrow", "upturned"}
)
_ACCESSORY_MARKERS = (
    "hairband",
    "hair ribbon",
    "hairpin",
    "hair ornament",
    "headband",
    "ribbon",
    "bow",
    "glasses",
    "eyepatch",
    "earrings",
    "piercing",
    "choker",
    "necklace",
)
_STABLE_APPEARANCE_MARKERS = (
    "freckles",
    "mole",
    "dark skin",
    "pale skin",
    "pointy ears",
    "fang",
    "horn",
    "tail",
    "heterochromia",
)
_TRANSIENT_HAIR_TAGS = frozenset(
    {
        "messy hair",
        "disheveled hair",
        "wet hair",
        "damp hair",
        "windblown hair",
        "hair blowing in wind",
        "tousled hair",
    }
)
_TAG_SEPARATOR_RE = re.compile(r"[_\-]+")


@dataclass(frozen=True)
class CharacterAppearanceFilterResult:
    positive: str
    removed_tags: tuple[str, ...] = ()
    removed_reasons: tuple[tuple[str, str], ...] = ()


def _tag_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = _TAG_SEPARATOR_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _has_phrase(key: str, phrase: str) -> bool:
    phrase_words = tuple(phrase.split())
    words = tuple(key.split())
    if not phrase_words:
        return False
    width = len(phrase_words)
    return any(words[index : index + width] == phrase_words for index in range(len(words) - width + 1))


def _japanese_appearance_scopes(user_input: str) -> set[str]:
    """Return appearance categories explicitly mentioned in Japanese input."""

    text = unicodedata.normalize("NFKC", str(user_input or "")).casefold()
    scopes: set[str] = set()

    # A bare mention of 髪 (for example 白髪) does not authorize an unrelated
    # length/style completion.  Style requires a style/length/arrangement cue.
    if any(
        cue in text
        for cue in (
            "髪型",
            "長い髪",
            "短い髪",
            "髪の長さ",
            "髪を長",
            "髪を短",
            "髪ふわ",
            "髪がふわ",
            "髪を流",
            "横に流",
            "ロングヘア",
            "ショートヘア",
            "ボブ",
            "ポニーテール",
            "ツインテール",
            "三つ編み",
            "みつあみ",
            "前髪",
            "long hair",
            "short hair",
            "ponytail",
            "twintails",
            "braid",
            "bob cut",
        )
    ):
        scopes.add("hair_style")
    hair_color_pattern = re.compile(
        r"(?:白|黒|茶|金|銀|赤|青|紫|緑|桃|ピンク|灰|グレー|黄|黄色|オレンジ)"
        r"(?:い|色|髪)?髪|髪(?:の色|色|を[^、。！？]{0,6}(?:白|黒|茶|金|銀|赤|青|紫|緑|桃|ピンク|灰|グレー|黄|黄色|オレンジ))"
    )
    if hair_color_pattern.search(text) or any(
        phrase in text
        for phrase in (
            "white hair",
            "black hair",
            "blue hair",
            "red hair",
            "pink hair",
            "髪が白",
            "髪が黒",
            "髪が青",
            "髪が赤",
            "髪がピンク",
            "髪を白",
            "髪を黒",
            "髪を青",
            "髪を赤",
            "髪をピンク",
        )
    ):
        scopes.add("hair_color")

    eye_color_pattern = re.compile(
        r"(?:赤|青|緑|紫|茶|金|黄|桃|ピンク|灰|グレー|銀|水色|オレンジ)"
        r"(?:い|色)?(?:の)?(?:目|瞳)|(?:目|瞳)(?:の色|色)"
    )
    if eye_color_pattern.search(text) or any(
        phrase in text
        for phrase in (
            "red eyes",
            "blue eyes",
            "green eyes",
            "purple eyes",
            "pink eyes",
            "brown eyes",
            "eye color",
        )
    ):
        scopes.add("eye_color")
    if any(
        cue in text
        for cue in (
            "大きい目",
            "大きな目",
            "たれ目",
            "垂れ目",
            "つり目",
            "吊り目",
            "細い目",
            "目が大き",
            "目を大き",
            "big eyes",
            "large eyes",
            "droopy eyes",
            "upturned eyes",
            "narrow eyes",
        )
    ):
        scopes.add("eye_shape")

    if any(
        cue in text
        for cue in (
            "髪飾り",
            "髪留め",
            "ヘアピン",
            "ヘアバンド",
            "カチューシャ",
            "リボン",
            "眼鏡",
            "メガネ",
            "イヤリング",
            "ピアス",
            "ネックレス",
            "チョーカー",
            "hairpin",
            "hairband",
            "ribbon",
            "glasses",
        )
    ):
        scopes.add("constant_accessory")

    if any(cue in text for cue in ("そばかす", "ほくろ", "肌", "角", "牙", "尻尾")):
        scopes.add("stable_appearance")
    return scopes


def _appearance_categories(tag: str) -> frozenset[str]:
    key = _tag_key(tag)
    if not key or key in _TRANSIENT_HAIR_TAGS:
        return frozenset()

    categories: set[str] = set()
    words = set(key.split())

    if any(_has_phrase(key, marker) for marker in _ACCESSORY_MARKERS):
        categories.add("constant_accessory")

    if any(_has_phrase(key, marker) for marker in _STABLE_APPEARANCE_MARKERS):
        categories.add("stable_appearance")

    if key.endswith(" eyes") or key.endswith(" eye color") or "heterochromia" in key:
        if words.intersection(_COLOR_WORDS) or "heterochromia" in key:
            categories.add("eye_color")
        elif words.intersection(_EYE_SHAPE_WORDS):
            categories.add("eye_shape")

    if "hair" in words or "bangs" in words:
        if (key.endswith(" hair") and words.intersection(_COLOR_WORDS)) or any(
            _has_phrase(key, marker)
            for marker in ("two-tone hair", "gradient hair", "colored tips", "highlights")
        ):
            categories.add("hair_color")
        if words.intersection(_HAIR_STYLE_WORDS) or any(
            _has_phrase(key, marker)
            for marker in (
                "hair between eyes",
                "face-framing",
                "strands around face",
                "side-swept hair",
                "swept to the side",
            )
        ):
            categories.add("hair_style")

    return frozenset(categories)


def _is_authorized_appearance(
    categories: frozenset[str],
    *,
    explicit_scopes: set[str],
) -> bool:
    return bool(categories) and categories.issubset(explicit_scopes)


def filter_unrequested_character_appearance(
    positive: str,
    *,
    user_input: str,
    character_resolution: object | None,
    verified_prompt_tags: Iterable[str] = (),
) -> CharacterAppearanceFilterResult:
    """Remove LLM identity-appearance tags not supported by the request.

    ``verified_prompt_tags`` is the union of core/general resolver outputs.
    Such tags are already Python-side decisions and are therefore preserved.
    """

    if character_resolution is None or not bool(
        getattr(character_resolution, "has_selected_tag", False)
    ):
        return CharacterAppearanceFilterResult(str(positive or "").strip())

    explicit_scopes = _japanese_appearance_scopes(user_input)
    verified_keys = {_tag_key(tag) for tag in verified_prompt_tags if str(tag).strip()}
    kept: list[str] = []
    removed: list[str] = []
    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = _tag_key(token)
        if key in seen:
            continue
        categories = _appearance_categories(token)
        if categories and key not in verified_keys and not _is_authorized_appearance(
            categories,
            explicit_scopes=explicit_scopes,
        ):
            removed.append(token)
            reasons.append((token, "unrequested_character_appearance:" + "+".join(sorted(categories))))
            continue
        kept.append(token)
        seen.add(key)

    return CharacterAppearanceFilterResult(
        positive=", ".join(kept),
        removed_tags=tuple(removed),
        removed_reasons=tuple(reasons),
    )


def enforce_confirmed_character_appearance(
    positive: str,
    confirmed_tags: Iterable[str],
    *,
    user_input: str = "",
    explicit_prompt_tags: Iterable[str] = (),
    identity_tags: Iterable[str] = (),
) -> CharacterAppearanceFilterResult:
    """Keep curated appearance tags authoritative through final cleanup.

    Confirmed character appearance is stronger than an unsupported LLM tag,
    but an explicitly requested appearance remains the user's choice.  When
    no explicit appearance category is present, another appearance tag in
    the same category is removed before the confirmed tag is inserted.  This
    prevents a character's curated hair/eye features from being silently
    dropped or contradicted by the model response.
    """

    confirmed: list[str] = []
    confirmed_by_key: dict[str, str] = {}
    for raw_tag in confirmed_tags:
        tag = str(raw_tag or "").strip()
        key = _tag_key(tag)
        if not tag or key in confirmed_by_key:
            continue
        confirmed.append(tag)
        confirmed_by_key[key] = tag
    if not confirmed:
        return CharacterAppearanceFilterResult(str(positive or "").strip())

    explicit_keys = {
        _tag_key(tag)
        for tag in explicit_prompt_tags
        if str(tag or "").strip()
    }
    explicit_scopes = _japanese_appearance_scopes(user_input)
    explicit_categories = {
        category
        for tag in explicit_prompt_tags
        for category in _appearance_categories(str(tag))
        if str(tag or "").strip()
    }
    confirmed_categories = {
        category
        for tag in confirmed
        for category in _appearance_categories(tag)
    }

    kept: list[str] = []
    removed: list[str] = []
    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = _tag_key(token)
        if key in seen:
            removed.append(token)
            reasons.append((token, "duplicate_normalized_tag"))
            continue
        categories = _appearance_categories(token)
        if categories and categories.intersection(explicit_categories or explicit_scopes):
            if key in explicit_keys:
                kept.append(token)
                seen.add(key)
                continue
            if key in confirmed_by_key:
                removed.append(token)
                reasons.append((token, "confirmed_appearance_overridden_by_explicit_request"))
                continue
            # Explicit user appearance takes precedence over another model
            # completion in the same category.
            kept.append(token)
            seen.add(key)
            continue
        if key in confirmed_by_key:
            kept.append(token)
            seen.add(key)
            continue
        if categories and categories.intersection(confirmed_categories):
            removed.append(token)
            reasons.append((token, "conflicts_with_confirmed_appearance"))
            continue
        kept.append(token)
        seen.add(key)

    eligible_confirmed = [
        tag
        for tag in confirmed
        if not _appearance_categories(tag).intersection(explicit_categories or explicit_scopes)
    ]
    missing = [tag for tag in eligible_confirmed if _tag_key(tag) not in seen]
    if missing:
        identity_keys = {
            _tag_key(tag)
            for tag in identity_tags
            if str(tag or "").strip()
        }
        insert_at = 1 if kept and re.fullmatch(r"\d+(?:girl|girls|boy|boys|other)", kept[0].casefold()) else 0
        while insert_at < len(kept) and _tag_key(kept[insert_at]) in identity_keys:
            insert_at += 1
        kept[insert_at:insert_at] = missing

    return CharacterAppearanceFilterResult(
        positive=", ".join(kept),
        removed_tags=tuple(removed),
        removed_reasons=tuple(reasons),
    )
