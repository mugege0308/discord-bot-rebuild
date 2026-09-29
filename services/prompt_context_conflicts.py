"""Request-aware filtering for weak Prompt Designer context.

Preference and memory hints are intentionally weaker than the current
request.  This module turns that policy into a deterministic pre-send filter
for the visual constraints that are most harmful when they conflict:
framing/composition and pose intensity.

It does not rewrite the user's request or the Prompt Designer output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_WIDE_FRAMING_CUES = (
    "全身",
    "頭から足先",
    "頭からつま先",
    "頭から足元",
    "足先まで",
    "顔から太もも",
    "顔〜太もも",
    "顔～太もも",
    "顔から腿",
    "顔〜腿",
    "顔～腿",
    "腰から太もも",
    "腰〜太もも",
    "腰～太もも",
    "太ももまで",
    "太腿まで",
    "腿まで",
    "下半身",
    "中距離",
    "ミディアムショット",
    "少し引",
    "引いた構図",
    "引きの構図",
    "full body",
    "head to toe",
    "head-to-toe",
    "thighs visible",
    "lower body",
    "medium shot",
    "cowboy shot",
    "from face to thighs",
    "waist to thighs",
)

_CLOSE_FRAMING_CUES = (
    "顔のアップ",
    "顔アップ",
    "顔を大きく",
    "もう少し寄",
    "寄った構図",
    "クローズアップ",
    "バストアップ",
    "ポートレート",
    "上半身だけ",
    "上半身のみ",
    "上半身",
    "close-up of torso",
    "close up of torso",
    "close-up",
    "close up",
    "closer framing",
    "tight framing",
    "upper body visible",
    "upper body only",
    "upper body",
    "portrait",
    "bust shot",
    "bust-up",
    "bust up",
)

_ACTIVE_POSE_CUES = (
    "動きのある",
    "躍動感",
    "走って",
    "走る",
    "走り",
    "歩いて",
    "歩く",
    "ジャンプ",
    "飛び跳ね",
    "戦闘",
    "バトル",
    "戦う",
    "アクション",
    "踊って",
    "踊る",
    "ダンス",
    "active pose",
    "dynamic pose",
    "action pose",
    "running",
    "walking",
    "jumping",
    "fighting",
    "dancing",
)

_CALM_POSE_CUES = (
    "落ち着いたポーズ",
    "静かなポーズ",
    "立ち姿",
    "立っている",
    "座り姿",
    "座っている",
    "calm pose",
    "natural pose",
    "natural standing pose",
    "standing",
    "sitting",
)

_PROHIBITION_MARKERS = (
    "禁止",
    "不可",
    "避けて",
    "避ける",
    "しないで",
    "しないこと",
    "入れない",
    "含めない",
    "なし",
    "不要",
    "no ",
    "not ",
    "without ",
    "avoid ",
    "do not ",
    "don't ",
    "must not ",
)

_CLAUSE_SEPARATORS = ("。", "、", "，", ",", ".", "!", "！", "?", "？", ";", "；", "\n")

_MEDIUM_FRAMING_REQUEST_CUES = (
    "中距離",
    "ミディアムショット",
    "少し引",
    "引いた構図",
    "medium shot",
    "medium distance",
)
_THIGH_FRAMING_REQUEST_CUES = (
    "顔から太もも",
    "顔〜太もも",
    "顔～太もも",
    "腰から太もも",
    "腰〜太もも",
    "腰～太もも",
    "太ももまで",
    "太腿まで",
    "腿まで",
    "thighs visible",
    "from face to thighs",
    "waist to thighs",
)
_LOWER_BODY_REQUEST_CUES = (
    "下半身",
    "顔から太もも",
    "顔〜太もも",
    "顔～太もも",
    "腰から太もも",
    "腰〜太もも",
    "腰～太もも",
    "lower body",
    "from face to thighs",
    "waist to thighs",
)
_PROHIBITED_UPPER_ONLY_CUES = (
    "上半身だけ",
    "上半身のみ",
    "upper body only",
)

_EXPLICIT_BACKGROUND_ABSENCE_CUES = (
    "背景は指定しない",
    "背景を指定しない",
    "背景なし",
    "背景はなし",
    "背景不要",
    "背景は不要",
    "背景を入れない",
    "背景を描かない",
    "no background",
    "without a background",
)
_CONCRETE_BACKGROUND_OUTPUT_CUES = (
    "classroom",
    "school background",
    "schoolyard",
    "outdoor",
    "outdoors",
    "indoor",
    "indoors",
    "gymnasium",
    "shrine",
    "cafe",
    "coffee shop",
    "street",
    "city",
    "forest",
    "park",
    "beach",
    "bedroom",
    "living room",
    "office",
    "station",
    "train",
    "sea",
    "window",
)

_FACE_VISIBLE_REQUEST_RE = re.compile(
    r"(?:"
    r"顔[^。！？\n]{0,60}(?:画面内|フレーム内|写す|映す|見える|含め|入れる|収める)"
    r"|(?:画面内|フレーム内)[^。！？\n]{0,60}顔"
    r"|\b(?:face|head)\s+(?:must\s+be\s+)?(?:visible|in\s+(?:the\s+)?frame)\b"
    r"|\b(?:include|show|keep)\b[^.\n]{0,40}\b(?:face|head)\b"
    r")",
    re.IGNORECASE,
)
_FACE_HIDDEN_OUTPUT_CUES = (
    "face not visible",
    "faceless",
    "head out of frame",
    "face out of frame",
    "cropped head",
    "head cropped",
    "no face",
)

_MEDIUM_FRAMING_OUTPUT_CUES = (
    "medium shot",
    "medium long shot",
    "medium full shot",
    "medium distance",
    "mid shot",
)
_THIGH_FRAMING_OUTPUT_CUES = (
    "thighs visible",
    "cowboy shot",
    "knees up",
    "from face to thighs",
    "waist to thighs",
)
_LOWER_BODY_OUTPUT_CUES = (
    "lower body visible",
    "lower body",
    "hips visible",
    "waist and thighs visible",
    "from face to thighs",
    "waist to thighs",
)
_CLOSE_FRAMING_OUTPUT_CUES = (
    "close-up",
    "close up",
    "close-up of torso",
    "close up of torso",
    "upper body",
    "upper body visible",
    "upper body only",
    "portrait",
    "bust shot",
    "bust-up",
    "bust up",
)

_EXPLICIT_LIGHTING_MARKERS = (
    "照明",
    "光源",
    "ライティング",
    "逆光",
    "月明かり",
    "日差し",
    "木漏れ日",
    "lighting",
    " light",
    "light ",
    "backlight",
    "moonlight",
    "sunlight",
    "自然光",
    "自然な光",
    "中立な光",
    "ニュートラルな光",
)
_LIGHTING_CONCEPT_CUES = {
    "natural": ("自然光", "自然な光", "natural light", "natural lighting"),
    "neutral": (
        "中立な光",
        "ニュートラルな光",
        "neutral light",
        "neutral lighting",
        "balanced color palette",
    ),
    "soft": ("柔らか", "やわらか", "soft light", "soft lighting"),
    "warm": ("暖色", "温かい光", "暖かい光", "warm light", "warm lighting"),
    "indirect": ("間接照明", "indirect light", "indirect lighting", "ambient lighting"),
    "firefly": ("蛍", "ほたる", "firefly", "fireflies"),
    "moonlight": ("月明かり", "月光", "moonlight"),
    "sunlight": ("日差し", "太陽光", "sunlight", "daylight"),
    "sunset": ("夕焼け", "夕暮れの光", "sunset light", "sunset lighting"),
    "neon": ("ネオン", "neon light", "neon lighting"),
    "candle": ("ろうそく", "蝋燭", "キャンドル", "candlelight"),
    "rim": ("リムライト", "rim light", "rim lighting"),
    "backlight": ("逆光", "backlight", "backlighting"),
    "dramatic": ("ドラマチックな照明", "dramatic light", "dramatic lighting"),
    "cinematic": ("映画的な照明", "cinematic light", "cinematic lighting"),
    "volumetric": ("ボリューメトリック", "volumetric light", "volumetric lighting"),
    "studio": ("スタジオ照明", "studio light", "studio lighting"),
}
_LIGHTING_OUTPUT_CUES = {
    "natural": (
        "natural light",
        "natural lighting",
        "bright natural lighting",
    ),
    "neutral": (
        "neutral light",
        "neutral lighting",
        "balanced color palette",
    ),
    "soft": (
        "soft light",
        "soft lighting",
        "soft warm indirect light",
        "diffused light",
        "diffuse light",
    ),
    "warm": (
        "warm light",
        "warm lighting",
        "warm indirect light",
        "soft warm indirect light",
        "warm tones",
        "warm tone",
    ),
    "indirect": (
        "indirect light",
        "indirect lighting",
        "warm indirect light",
        "soft warm indirect light",
        "ambient light",
        "ambient lighting",
        "bounce light",
        "bounced light",
        "diffused light",
        "diffuse light",
    ),
    "firefly": ("firefly", "fireflies"),
    "moonlight": ("moonlight",),
    "sunlight": ("sunlight", "daylight"),
    "sunset": ("sunset light", "sunset lighting"),
    "neon": ("neon light", "neon lighting"),
    "candle": ("candlelight", "candle light"),
    "rim": ("rim light", "rim lighting"),
    "backlight": ("backlight", "backlighting"),
    "dramatic": ("dramatic light", "dramatic lighting"),
    "cinematic": ("cinematic light", "cinematic lighting"),
    "volumetric": ("volumetric light", "volumetric lighting"),
    "studio": ("studio light", "studio lighting"),
}
_LIGHTING_CORRECTION_TAGS = {
    "natural": "natural lighting",
    "neutral": "neutral lighting",
    "soft": "soft lighting",
    "warm": "warm lighting",
    "indirect": "indirect lighting",
    "firefly": "firefly lighting",
    "moonlight": "moonlight",
    "sunlight": "sunlight",
    "sunset": "sunset lighting",
    "neon": "neon lighting",
    "candle": "candlelight",
    "rim": "rim lighting",
    "backlight": "backlighting",
    "dramatic": "dramatic lighting",
    "cinematic": "cinematic lighting",
    "volumetric": "volumetric lighting",
    "studio": "studio lighting",
}
_LIGHTING_ASSUMPTION_RE = re.compile(
    r"(?:照明|光源|色温度|ライティング|逆光|月明かり|日差し|木漏れ日|"
    r"\blighting\b|\blight source\b|\bbacklight(?:ing)?\b|\bmoonlight\b|"
    r"\bsunlight\b|\bwarm tones?\b|\bcool tones?\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitVisualConstraints:
    wide_framing: bool = False
    close_framing: bool = False
    prohibit_wide_framing: bool = False
    prohibit_close_framing: bool = False
    active_pose: bool = False
    calm_pose: bool = False
    prohibit_active_pose: bool = False
    prohibit_calm_pose: bool = False
    require_medium_framing: bool = False
    require_thigh_framing: bool = False
    require_lower_body: bool = False
    require_face_visible: bool = False
    prohibit_concrete_background: bool = False

    @property
    def controls_composition(self) -> bool:
        return any((
            self.wide_framing,
            self.close_framing,
            self.prohibit_wide_framing,
            self.prohibit_close_framing,
        ))

    @property
    def controls_pose(self) -> bool:
        return any((
            self.active_pose,
            self.calm_pose,
            self.prohibit_active_pose,
            self.prohibit_calm_pose,
        ))


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _cue_is_prohibited(text: str, cue: str) -> bool:
    """Match prohibition words near a cue, not anywhere in the request."""
    start = 0
    while True:
        index = text.find(cue, start)
        if index < 0:
            return False
        previous_separators = [text.rfind(separator, 0, index) for separator in _CLAUSE_SEPARATORS]
        window_start = max(max(previous_separators) + 1, index - 24)
        next_separators = [
            position
            for separator in _CLAUSE_SEPARATORS
            if (position := text.find(separator, index + len(cue))) >= 0
        ]
        next_separator = min(next_separators) if next_separators else len(text)
        window_end = min(next_separator, index + len(cue) + 24)
        window = text[window_start:window_end]
        if _contains_any(window, _PROHIBITION_MARKERS):
            return True
        start = index + len(cue)


def _cue_state(text: str, cues: tuple[str, ...]) -> tuple[bool, bool]:
    present = False
    prohibited = False
    for cue in cues:
        if cue not in text:
            continue
        present = True
        prohibited = prohibited or _cue_is_prohibited(text, cue)
    return present and not prohibited, prohibited


def analyze_explicit_visual_constraints(request_text: str) -> ExplicitVisualConstraints:
    text = unicodedata.normalize("NFKC", request_text or "").casefold()
    wide, prohibit_wide = _cue_state(text, _WIDE_FRAMING_CUES)
    close, prohibit_close = _cue_state(text, _CLOSE_FRAMING_CUES)
    active, prohibit_active = _cue_state(text, _ACTIVE_POSE_CUES)
    calm, prohibit_calm = _cue_state(text, _CALM_POSE_CUES)
    medium_framing, _prohibit_medium_framing = _cue_state(
        text, _MEDIUM_FRAMING_REQUEST_CUES
    )
    thigh_framing, _prohibit_thigh_framing = _cue_state(
        text, _THIGH_FRAMING_REQUEST_CUES
    )
    lower_body, _prohibit_lower_body = _cue_state(
        text, _LOWER_BODY_REQUEST_CUES
    )
    prohibit_upper_only = any(
        cue in text and _cue_is_prohibited(text, cue)
        for cue in _PROHIBITED_UPPER_ONLY_CUES
    )
    require_face_visible = bool(_FACE_VISIBLE_REQUEST_RE.search(text))
    prohibit_concrete_background = _contains_any(
        text, _EXPLICIT_BACKGROUND_ABSENCE_CUES
    )
    return ExplicitVisualConstraints(
        wide_framing=wide,
        close_framing=close,
        prohibit_wide_framing=prohibit_wide,
        prohibit_close_framing=prohibit_close,
        active_pose=active,
        calm_pose=calm,
        prohibit_active_pose=prohibit_active,
        prohibit_calm_pose=prohibit_calm,
        require_medium_framing=medium_framing,
        require_thigh_framing=thigh_framing,
        require_lower_body=(lower_body or prohibit_upper_only),
        require_face_visible=require_face_visible,
        prohibit_concrete_background=prohibit_concrete_background,
    )


@dataclass(frozen=True)
class PromptOutputConstraintValidation:
    applicable: bool = False
    missing_requirements: tuple[str, ...] = ()
    missing_tags: tuple[str, ...] = ()
    conflicting_tags: tuple[str, ...] = ()
    assumption_violations: tuple[str, ...] = ()
    preserved_tags: tuple[str, ...] = ()
    requested_lighting: tuple[str, ...] = ()
    prohibited_framing_tags: tuple[str, ...] = ()
    framing_missing_tags: tuple[str, ...] = ()
    framing_conflicting_tags: tuple[str, ...] = ()
    background_conflicting_tags: tuple[str, ...] = ()
    lighting_missing_tags: tuple[str, ...] = ()
    lighting_conflicting_tags: tuple[str, ...] = ()
    face_visibility_required: bool = False
    close_framing_prohibited: bool = False
    face_hidden_tags: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not (
            self.missing_requirements
            or self.conflicting_tags
            or self.assumption_violations
        )


def _normalized_tag(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _positive_tags(positive: str) -> tuple[str, ...]:
    return tuple(_normalized_tag(tag) for tag in positive.split(",") if tag.strip())


def _matching_tags(tags: tuple[str, ...], cues: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(tag for tag in tags if any(cue in tag for cue in cues))


def request_has_explicit_lighting(request_text: str) -> bool:
    text = unicodedata.normalize("NFKC", request_text or "").casefold()
    return _contains_any(text, _EXPLICIT_LIGHTING_MARKERS)


def requested_lighting_concepts(request_text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", request_text or "").casefold()
    return tuple(
        concept
        for concept, cues in _LIGHTING_CONCEPT_CUES.items()
        if _contains_any(text, cues)
    )


def build_explicit_lighting_control_instruction(request_text: str) -> str:
    """Small dynamic rule used only when the current request owns lighting."""

    if not request_has_explicit_lighting(request_text):
        return ""
    concepts = requested_lighting_concepts(request_text)
    concept_text = ", ".join(concepts) if concepts else "the user's stated lighting"
    return (
        "Explicit lighting control: preserve the request's lighting exactly "
        f"({concept_text}). Do not invent a different light source or lighting effect. "
        "Do not repeat lighting or color-direction decisions in assumptions."
    )


def validate_explicit_prompt_output(
    request_text: str,
    positive: str,
    assumptions: tuple[str, ...] = (),
) -> PromptOutputConstraintValidation:
    """Checks only visual constraints explicitly owned by the current request."""

    constraints = analyze_explicit_visual_constraints(request_text)
    tags = _positive_tags(positive)
    missing: list[str] = []
    missing_tags: list[str] = []
    preserved: list[str] = []

    def validate_required_tag(
        required: bool,
        cues: tuple[str, ...],
        requirement: str,
        correction_tag: str,
    ) -> None:
        if not required:
            return
        matched = _matching_tags(tags, cues)
        if matched:
            preserved.extend(matched)
            return
        missing.append(requirement)
        missing_tags.append(correction_tag)

    validate_required_tag(
        constraints.require_medium_framing,
        _MEDIUM_FRAMING_OUTPUT_CUES,
        "medium framing (medium shot equivalent)",
        "medium shot",
    )
    validate_required_tag(
        constraints.require_thigh_framing,
        _THIGH_FRAMING_OUTPUT_CUES,
        "thigh framing (thighs visible/cowboy shot equivalent)",
        "thighs visible",
    )
    validate_required_tag(
        constraints.require_lower_body,
        _LOWER_BODY_OUTPUT_CUES,
        "lower-body visibility (lower body visible equivalent)",
        "lower body visible",
    )
    framing_missing = tuple(dict.fromkeys(missing_tags))

    conflicting: list[str] = []
    framing_conflicting: list[str] = []
    if constraints.prohibit_close_framing:
        framing_conflicting.extend(
            tag
            for tag in tags
            if any(cue in tag for cue in _CLOSE_FRAMING_OUTPUT_CUES)
        )
    if constraints.require_face_visible:
        framing_conflicting.extend(
            tag
            for tag in tags
            if any(cue in tag for cue in _FACE_HIDDEN_OUTPUT_CUES)
        )
    conflicting.extend(framing_conflicting)

    background_conflicting: list[str] = []
    if constraints.prohibit_concrete_background:
        background_conflicting.extend(
            tag
            for tag in tags
            if any(cue in tag for cue in _CONCRETE_BACKGROUND_OUTPUT_CUES)
        )
    conflicting.extend(background_conflicting)

    assumption_violations: list[str] = []
    lighting_missing: list[str] = []
    lighting_conflicting: list[str] = []
    lighting_is_explicit = request_has_explicit_lighting(request_text)
    lighting_concepts = requested_lighting_concepts(request_text)
    if lighting_is_explicit:
        requested_concepts = set(lighting_concepts)
        matched_lighting: dict[str, tuple[str, ...]] = {
            concept: _matching_tags(tags, _LIGHTING_OUTPUT_CUES.get(concept, cues))
            for concept, cues in _LIGHTING_CONCEPT_CUES.items()
        }
        # Animagine commonly expresses soft warm indirect illumination as
        # soft lighting + warm lighting/tones without the literal tag
        # "indirect lighting".  When all three concepts were requested, that
        # pair is a sufficient visual proxy unless a different source/effect
        # is present (checked below).
        indirect_proxy = bool(
            {"soft", "warm", "indirect"}.issubset(requested_concepts)
            and matched_lighting.get("soft")
            and matched_lighting.get("warm")
        )
        for concept in lighting_concepts:
            matched = matched_lighting.get(concept, ())
            if matched or (concept == "indirect" and indirect_proxy):
                preserved.extend(matched)
                continue
            lighting_missing.append(
                _LIGHTING_CORRECTION_TAGS.get(concept, f"{concept} lighting")
            )
        for concept, cues in _LIGHTING_CONCEPT_CUES.items():
            if concept in requested_concepts:
                continue
            lighting_conflicting.extend(
                tag
                for tag in tags
                if any(
                    cue in tag
                    for cue in _LIGHTING_OUTPUT_CUES.get(concept, cues)
                )
            )
        conflicting.extend(lighting_conflicting)
        assumption_violations.extend(
            assumption
            for assumption in assumptions
            if _LIGHTING_ASSUMPTION_RE.search(assumption)
        )
        for tag in lighting_missing:
            missing.append(f"lighting requirement ({tag})")
            missing_tags.append(tag)

    return PromptOutputConstraintValidation(
        applicable=bool(
            constraints.require_medium_framing
            or constraints.require_thigh_framing
            or constraints.require_lower_body
            or constraints.prohibit_close_framing
            or constraints.require_face_visible
            or constraints.prohibit_concrete_background
            or lighting_is_explicit
        ),
        missing_requirements=tuple(dict.fromkeys(missing)),
        missing_tags=tuple(dict.fromkeys(missing_tags)),
        conflicting_tags=tuple(dict.fromkeys(conflicting)),
        assumption_violations=tuple(dict.fromkeys(assumption_violations)),
        preserved_tags=tuple(dict.fromkeys(preserved)),
        requested_lighting=lighting_concepts,
        prohibited_framing_tags=(
            tuple(dict.fromkeys(_CLOSE_FRAMING_OUTPUT_CUES))
            if constraints.prohibit_close_framing
            else ()
        ),
        framing_missing_tags=framing_missing,
        framing_conflicting_tags=tuple(dict.fromkeys(framing_conflicting)),
        background_conflicting_tags=tuple(dict.fromkeys(background_conflicting)),
        lighting_missing_tags=tuple(dict.fromkeys(lighting_missing)),
        lighting_conflicting_tags=tuple(dict.fromkeys(lighting_conflicting)),
        face_visibility_required=constraints.require_face_visible,
        close_framing_prohibited=constraints.prohibit_close_framing,
        face_hidden_tags=_FACE_HIDDEN_OUTPUT_CUES if constraints.require_face_visible else (),
    )


def request_controls_hint_category(category_key: str, request_text: str) -> bool:
    """Whether weak hints in a category must yield to this request.

    Preference categories use a single prose sentence that may describe more
    than the exact matched tag.  Once the user explicitly controls framing or
    pose, the whole corresponding weak category is therefore suppressed.
    """
    constraints = analyze_explicit_visual_constraints(request_text)
    if category_key == "composition":
        return constraints.controls_composition
    if category_key == "action_intensity":
        return constraints.controls_pose
    return False


def weak_visual_hint_conflicts_with_request(hint_text: str, request_text: str) -> bool:
    """Return True when one memory/tag hint opposes an explicit request."""
    constraints = analyze_explicit_visual_constraints(request_text)
    hint = (hint_text or "").lower()
    hint_is_wide = _contains_any(hint, _WIDE_FRAMING_CUES)
    hint_is_close = _contains_any(hint, _CLOSE_FRAMING_CUES)
    hint_is_active = _contains_any(hint, _ACTIVE_POSE_CUES)
    hint_is_calm = _contains_any(hint, _CALM_POSE_CUES)

    if hint_is_close and (constraints.wide_framing or constraints.prohibit_close_framing):
        return True
    if hint_is_wide and (constraints.close_framing or constraints.prohibit_wide_framing):
        return True
    if hint_is_calm and (constraints.active_pose or constraints.prohibit_calm_pose):
        return True
    if hint_is_active and (constraints.calm_pose or constraints.prohibit_active_pose):
        return True
    return False


def filter_conflicting_weak_visual_hints(
    hints: list[str],
    request_text: str,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    excluded: list[str] = []
    for hint in hints:
        if weak_visual_hint_conflicts_with_request(hint, request_text):
            excluded.append(hint)
        else:
            kept.append(hint)
    return kept, excluded
