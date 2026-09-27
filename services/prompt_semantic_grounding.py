"""Keep Japanese Prompt Designer output grounded in the user's request.

The character appearance filter protects an identified character's identity.
This module has a narrower, separate responsibility: when the Japanese
Animagine route receives an LLM positive, remove a small set of recognizable
scene/pose/appearance completions that have no support in the request.  It is
deliberately conservative for tags outside the known semantic families so it
does not become a second Danbooru allowlist.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

from services.prompt_positive_cleanup import normalize_prompt_tag_key


@dataclass(frozen=True)
class PromptSemanticGroundingResult:
    positive: str
    user_grounded_tags: tuple[str, ...] = ()
    removed_tags: tuple[str, ...] = ()
    removed_reasons: tuple[tuple[str, str], ...] = ()
    pose_normalizations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    removed_assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SemanticRule:
    canonical: str
    category: str
    cues: tuple[str, ...]
    tag_aliases: tuple[str, ...] = ()


# These rules are intentionally small and high-confidence.  They are not a
# translation dictionary for all Danbooru; they only identify common semantic
# additions that must not be hallucinated by the Prompt Designer.
_RULES: tuple[_SemanticRule, ...] = (
    _SemanticRule("cute", "expression.cute", ("かわいい", "可愛い", "キュート", "cute")),
    _SemanticRule(
        "smile",
        "expression.smile",
        ("笑顔", "笑って", "微笑", "smile", "smiling"),
        tag_aliases=("smiley face",),
    ),
    _SemanticRule(
        "anime style",
        "style.anime",
        ("アニメ風", "アニメ調", "アニメスタイル", "anime style"),
    ),
    _SemanticRule("grin", "expression.grin", ("ニヤリ", "にやり", "grin", "grinning")),
    _SemanticRule("serious", "expression.serious", ("真顔", "真剣な表情", "serious")),
    _SemanticRule("surprised", "expression.surprised", ("驚い", "びっくり", "surprised")),
    _SemanticRule("blush", "expression.blush", ("照れ", "赤面", "blush")),
    _SemanticRule("closed eyes", "eye_state.closed", ("目を閉じ", "目閉じ", "closed eyes", "eyes closed")),
    _SemanticRule("looking at viewer", "gaze.viewer", ("カメラ目線", "こちらを見る", "こっちを見る", "looking at viewer")),
    _SemanticRule("looking away", "gaze.away", ("目をそら", "視線をそら", "looking away")),
    _SemanticRule("looking back", "gaze.back", ("振り返", "振り向", "見返", "looking back")),
    _SemanticRule("standing", "pose.standing", ("立つ", "立って", "立ち姿", "standing")),
    _SemanticRule("sitting", "pose.sitting", ("座る", "座って", "座り", "着席", "sitting")),
    _SemanticRule("kneeling", "pose.kneeling", ("ひざまず", "膝立ち", "跪く", "kneeling")),
    _SemanticRule("squatting", "pose.squatting", ("しゃが", "squatting")),
    _SemanticRule("lying", "pose.lying", ("横たわ", "寝そべ", "寝転", "lying")),
    _SemanticRule("walking", "pose.walking", ("歩く", "歩いて", "歩き", "walking")),
    _SemanticRule("running", "pose.running", ("走る", "走って", "走り", "running")),
    _SemanticRule("jumping", "pose.jumping", ("跳ぶ", "飛び跳ね", "ジャンプ", "jumping")),
    _SemanticRule("dynamic pose", "pose.dynamic", ("動きのある", "ダイナミック", "dynamic pose")),
    _SemanticRule(
        "hands behind back",
        "pose.hands_behind_back",
        ("後ろで手を組", "手を後ろで組", "背中で手を組", "背中側で手を組", "hands behind back"),
    ),
    _SemanticRule("peace sign", "pose.hand_gesture", ("ピース", "ピースサイン", "peace sign")),
    _SemanticRule(
        "leaning forward",
        "pose.leaning_forward",
        ("前かがみ", "前屈み", "前にかが", "身を乗り出", "leaning forward"),
    ),
    _SemanticRule(
        "slouched posture",
        "pose.slouched",
        ("猫背", "だらっとした姿勢", "姿勢が悪", "slouched posture"),
    ),
    _SemanticRule("school uniform", "clothing.school_uniform", ("制服", "school uniform")),
    _SemanticRule("hoodie", "clothing.hoodie", ("パーカー", "フーディー", "hoodie")),
    _SemanticRule("jacket", "clothing.jacket", ("ジャケット", "上着", "jacket")),
    _SemanticRule("coat", "clothing.coat", ("コート", "coat")),
    _SemanticRule("dress", "clothing.dress", ("ワンピース", "ドレス", "dress")),
    _SemanticRule("skirt", "clothing.skirt", ("スカート", "skirt")),
    _SemanticRule("long sleeves", "clothing.long_sleeves", ("長袖", "袖が長", "long sleeves")),
    _SemanticRule("short sleeves", "clothing.short_sleeves", ("半袖", "袖が短", "short sleeves")),
    _SemanticRule("classroom", "background.classroom", ("教室", "classroom", "classroom background")),
    _SemanticRule("school", "background.school", ("学校", "school")),
    _SemanticRule("outdoors", "background.outdoors", ("屋外", "外で", "outdoor", "outdoors")),
    _SemanticRule("natural lighting", "lighting.natural", ("自然光", "自然な光", "自然な照明", "natural lighting")),
    _SemanticRule("soft lighting", "lighting.soft", ("柔らかい光", "柔らかな光", "ソフトな光", "soft lighting")),
    _SemanticRule("warm lighting", "lighting.warm", ("暖かい光", "暖かな光", "暖色の光", "warm lighting")),
    _SemanticRule("sunlight", "lighting.sunlight", ("日光", "陽光", "sunlight")),
    _SemanticRule("balanced color palette", "color_palette.balanced", ("balanced color palette",)),
    _SemanticRule(
        "blue tones",
        "color_palette.blue",
        ("青系", "青み", "青い色", "blue tones", "blue tone"),
    ),
    _SemanticRule(
        "stylish outfit",
        "clothing.style",
        ("おしゃれな服", "おしゃれな衣装", "stylish outfit"),
    ),
    _SemanticRule("umbrella", "prop.umbrella", ("傘", "umbrella")),
    _SemanticRule("book", "prop.book", ("本を", "本が", "本の", "本読む", "book")),
    _SemanticRule("sword", "prop.sword", ("剣", "刀", "sword")),
    _SemanticRule("hair accessory", "accessory.hair", ("髪飾り", "髪留め", "ヘアピン", "リボン", "hair ornament", "hairpin", "ribbon")),
    _SemanticRule("white hair", "appearance.hair_color", ("白髪", "白い髪", "髪が白", "white hair")),
    _SemanticRule("black hair", "appearance.hair_color", ("黒髪", "黒い髪", "髪が黒", "black hair")),
    _SemanticRule("blue hair", "appearance.hair_color", ("青髪", "青い髪", "髪が青", "blue hair")),
    _SemanticRule("green hair", "appearance.hair_color", ("緑髪", "緑の髪", "髪が緑", "green hair")),
    _SemanticRule("long hair", "appearance.hair_style", ("長い髪", "髪が長", "ロングヘア", "long hair", "long_hair")),
    _SemanticRule("short hair", "appearance.hair_style", ("短い髪", "髪が短", "ショートヘア", "short hair", "short_hair")),
    _SemanticRule("wavy hair", "appearance.hair_style", ("ウェーブヘア", "髪がウェーブ", "wavy hair")),
    _SemanticRule("blue eyes", "appearance.eye_color", ("青い目", "青い瞳", "目が青", "blue eyes")),
    _SemanticRule("green eyes", "appearance.eye_color", ("緑の目", "緑の瞳", "目が緑", "green eyes")),
    _SemanticRule("red eyes", "appearance.eye_color", ("赤い目", "赤い瞳", "目が赤", "red eyes")),
    _SemanticRule("pointy ears", "appearance.body", ("尖った耳", "とがった耳", "獣耳", "pointy ears")),
    _SemanticRule("horns", "appearance.body", ("角が", "角を", "角の", "horns")),
    _SemanticRule("tail", "appearance.body", ("尻尾", "しっぽ", "tail")),
    _SemanticRule("freckles", "appearance.body", ("そばかす", "freckles")),
    _SemanticRule("glasses", "accessory.glasses", ("眼鏡", "メガネ", "glasses")),
)

_QUALITY_OR_STRUCTURE = frozenset(
    {
        "safe",
        "sensitive",
        "nsfw",
        "explicit",
        "masterpiece",
        "best quality",
        "high quality",
        "high score",
        "great score",
        "absurdres",
        "1girl",
        "1boy",
        "2girls",
        "2boys",
        "solo",
    }
)
_QUALITY_SUFFIX = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "high score",
        "great score",
        "absurdres",
        "newest",
        "very aesthetic",
    }
)

_COLOR_WORDS = frozenset({"black", "blue", "brown", "cyan", "gold", "green", "grey", "gray", "orange", "pink", "purple", "red", "silver", "violet", "white", "yellow"})
_HAIR_STYLE_WORDS = frozenset({"long", "short", "medium", "curly", "fluffy", "messy", "straight", "wavy", "ponytail", "braid", "braids", "twintails", "tousled", "voluminous"})
_KNOWN_BACKGROUND_WORDS = frozenset({"classroom", "classroom background", "school", "outdoor", "outdoors", "street", "bedroom", "office", "forest", "beach", "city", "indoors"})
_KNOWN_CLOTHING_WORDS = frozenset({"uniform", "school uniform", "hoodie", "jacket", "coat", "dress", "skirt", "shirt", "sweater", "long sleeves", "short sleeves", "oversized clothes", "baggy clothes"})
_KNOWN_LIGHTING_WORDS = frozenset({"natural lighting", "soft lighting", "warm lighting", "cool lighting", "neutral lighting", "sunlight", "moonlight", "rim light", "backlighting", "balanced color palette"})
_KNOWN_POSE_WORDS = frozenset({"standing", "sitting", "kneeling", "squatting", "lying", "walking", "running", "jumping", "dynamic pose", "hands behind back", "leaning forward", "slouched posture"})
_KNOWN_EXPRESSION_WORDS = frozenset({"cute", "smile", "smiling", "smiley face", "grin", "serious", "surprised", "blush", "closed eyes", "eyes closed"})
_KNOWN_GAZE_WORDS = frozenset({"looking at viewer", "facing viewer", "looking away", "looking back", "looking to the side", "front view"})
_KNOWN_PROP_WORDS = frozenset({"umbrella", "book", "sword", "bag", "microphone", "flower", "phone", "weapon"})


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _rule_hits(text: str, rule: _SemanticRule) -> bool:
    return any(cue.casefold() in text for cue in rule.cues)


def _explicit_rules(user_input: str) -> tuple[_SemanticRule, ...]:
    text = _normalized_text(user_input)
    return tuple(rule for rule in _RULES if _rule_hits(text, rule))


def _tag_categories(tag: str) -> frozenset[str]:
    key = normalize_prompt_tag_key(tag)
    if not key or key in _QUALITY_OR_STRUCTURE:
        return frozenset()
    exact = {rule.category for rule in _RULES if key == normalize_prompt_tag_key(rule.canonical)}
    exact.update(
        rule.category
        for rule in _RULES
        if key in {
            normalize_prompt_tag_key(alias)
            for alias in (*rule.tag_aliases, *rule.cues)
            if alias.isascii()
        }
    )
    if exact:
        return frozenset(exact)

    categories: set[str] = set()
    words = set(key.split())
    if words.intersection(_COLOR_WORDS) and "hair" in words:
        categories.add("appearance.hair_color")
    if "hair" in words or "bangs" in words:
        if words.intersection(_HAIR_STYLE_WORDS):
            categories.add("appearance.hair_style")
    if "eye" in words or "eyes" in words:
        if words.intersection(_COLOR_WORDS):
            categories.add("appearance.eye_color")
        else:
            categories.add("appearance.eye_shape")
    if any(marker in key for marker in ("pointy ears", "animal ears", "horns", "tail", "freckles", "fang", "mole", "dark skin", "pale skin")):
        categories.add("appearance.body")
    if any(word in key for word in ("hairband", "hair ribbon", "hairpin", "hair ornament", "headband", "glasses", "eyepatch", "choker", "necklace")):
        categories.add("accessory.constant")
    if key in _KNOWN_BACKGROUND_WORDS or key.endswith(" background"):
        categories.add("background.generic")
    if key in _KNOWN_CLOTHING_WORDS:
        categories.add("clothing.generic")
    if key in _KNOWN_LIGHTING_WORDS or key.endswith(" lighting") or key.endswith(" light"):
        categories.add("lighting.generic")
    if key in _KNOWN_POSE_WORDS:
        categories.add("pose.generic")
    if key in _KNOWN_EXPRESSION_WORDS:
        categories.add("expression.generic")
    if key in _KNOWN_GAZE_WORDS:
        categories.add("gaze.generic")
    if key in _KNOWN_PROP_WORDS:
        categories.add("prop.generic")
    return frozenset(categories)


def _insert_before_quality(tags: list[str], additions: Iterable[str]) -> list[str]:
    new_tags = list(tags)
    seen = {normalize_prompt_tag_key(tag) for tag in new_tags}
    additions_to_insert = [
        tag for tag in additions if normalize_prompt_tag_key(tag) not in seen
    ]
    if not additions_to_insert:
        return new_tags
    insert_at = len(new_tags)
    for index, token in enumerate(new_tags):
        key = normalize_prompt_tag_key(token)
        if key in _QUALITY_SUFFIX:
            insert_at = index
            break
    new_tags[insert_at:insert_at] = additions_to_insert
    return new_tags


def _filter_assumptions(
    assumptions: Iterable[str],
    *,
    explicit_categories: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kept: list[str] = []
    removed: list[str] = []
    for assumption in assumptions:
        text = str(assumption or "").strip()
        if not text:
            continue
        categories: set[str] = set()
        key = normalize_prompt_tag_key(text)
        for rule in _RULES:
            if _rule_hits(_normalized_text(text), rule):
                categories.add(rule.category)
        categories.update(_tag_categories(key))
        if categories and not categories.issubset(explicit_categories):
            removed.append(text)
            continue
        kept.append(text)
    return tuple(kept), tuple(removed)


def ground_prompt_semantics(
    positive: str,
    *,
    user_input: str,
    authoritative_prompt_tags: Iterable[str] = (),
    assumptions: Iterable[str] = (),
    defer_character_appearance: bool = False,
) -> PromptSemanticGroundingResult:
    """Filter known unrequested semantic tags from an Animagine LLM positive.

    Unknown tags are intentionally preserved.  The function is a guardrail for
    common semantic families, not a replacement for the core or Danbooru
    resolvers.  Authoritative resolver tags always win over this filter.
    """

    rules = _explicit_rules(user_input)
    explicit_categories = {rule.category for rule in rules}
    explicit_canonicals = [rule.canonical for rule in rules]
    authoritative = tuple(
        str(tag).strip() for tag in authoritative_prompt_tags if str(tag or "").strip()
    )
    authoritative_keys = {normalize_prompt_tag_key(tag) for tag in authoritative}
    grounded: list[str] = []
    for tag in (*authoritative, *explicit_canonicals):
        if normalize_prompt_tag_key(tag) not in {normalize_prompt_tag_key(item) for item in grounded}:
            grounded.append(tag)

    kept: list[str] = []
    removed: list[str] = []
    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = normalize_prompt_tag_key(token)
        if not key or key in seen:
            continue
        categories = _tag_categories(token)
        if defer_character_appearance:
            categories = frozenset(
                category
                for category in categories
                if not (
                    category.startswith("appearance.")
                    or category.startswith("accessory.")
                )
            )
        authorized = key in authoritative_keys or bool(categories) and categories.issubset(explicit_categories)
        if categories and not authorized:
            removed.append(token)
            reasons.append((token, "unrequested_semantic:" + "+".join(sorted(categories))))
            continue
        kept.append(token)
        seen.add(key)

    pose_normalizations: list[str] = []
    canonical_pose_additions: list[str] = []
    if any(rule.category == "pose.leaning_forward" for rule in rules):
        canonical_pose_additions.append("leaning forward")
        pose_normalizations.append("前かがみ -> leaning forward")
    if any(rule.category == "pose.hands_behind_back" for rule in rules):
        canonical_pose_additions.append("hands behind back")
        pose_normalizations.append("後ろで手を組む -> hands behind back")
    kept = _insert_before_quality(kept, canonical_pose_additions)
    for tag in canonical_pose_additions:
        if normalize_prompt_tag_key(tag) not in {normalize_prompt_tag_key(item) for item in grounded}:
            grounded.append(tag)

    filtered_assumptions, removed_assumptions = _filter_assumptions(
        assumptions,
        explicit_categories=explicit_categories,
    )
    return PromptSemanticGroundingResult(
        positive=", ".join(kept),
        user_grounded_tags=tuple(grounded),
        removed_tags=tuple(removed),
        removed_reasons=tuple(reasons),
        pose_normalizations=tuple(pose_normalizations),
        assumptions=filtered_assumptions,
        removed_assumptions=removed_assumptions,
    )
