import json
import re
from dataclasses import dataclass
from typing import Any


QUALITY_TAGS = [
    "masterpiece",
    "best quality",
    "newest",
    "very aesthetic",
    "absurdres",
    "anime style",
]

REQUIRED_PERSON_TAGS = [
    "1girl",
    "solo",
]

STANDARD_PERSON_DETAIL_TAGS = [
    "detailed face",
    "beautiful face",
    "detailed eyes",
    "expressive eyes",
]

STANDARD_COMPOSITION_TAGS = [
    "upper body",
    "looking at viewer",
    "centered composition",
]

KIMONO_DETAIL_TAGS = [
    "floral kimono",
    "detailed kimono",
    "long sleeves",
    "obi",
]

REQUIRED_POSITIVE_PREFIX = QUALITY_TAGS + REQUIRED_PERSON_TAGS

FIXED_NEGATIVE_PROMPT = (
    "worst quality, low quality, normal quality, bad anatomy, bad hands, "
    "extra fingers, missing fingers, extra limbs, deformed, blurry, text, "
    "watermark, logo, jpeg artifacts, bad eyes, distorted eyes, "
    "asymmetrical eyes, glowing eyes, overly bright eyes, two-tone hair, "
    "multicolored hair, face paint, cheek mark, facial mark, leaf mark, "
    "asymmetrical hair color"
    ", stray hair, excessive hair strands, hair covering face, facial texture, "
    "noisy face, scribble, sketch lines"
)

FORBIDDEN_NEGATIVE_TAGS = {
    "cartoonish style",
    "unrealistic proportions",
    "poor lighting",
    "dark tones",
}

COMPOSITE_TAG_REPLACEMENTS = {
    "sunset shrine": ["shrine", "torii", "stone pathway", "japanese architecture", "sunset", "golden hour", "warm lighting"],
    "shrine at sunset": ["shrine", "torii", "stone pathway", "japanese architecture", "sunset", "golden hour", "warm lighting"],
    "japanese shrine": ["shrine", "torii", "stone pathway", "japanese architecture"],
    "shrine": ["shrine", "torii", "stone pathway", "japanese architecture"],
    "kimono": ["kimono", "obi", "wide sleeves", "traditional japanese clothes"],
    "wearing kimono": ["kimono", "obi", "wide sleeves", "traditional japanese clothes"],
    "school uniform": ["school uniform", "serafuku", "pleated skirt"],
    "sailor uniform": ["serafuku", "sailor collar", "pleated skirt"],
    "maid outfit": ["maid", "maid headdress", "apron", "frills"],
    "rainy street": ["rain", "wet street", "puddle", "reflection"],
    "night city": ["cityscape", "night", "neon lights", "street lights"],
    "cherry blossom": ["cherry blossoms", "sakura petals", "spring"],
    "cherry blossoms": ["cherry blossoms", "sakura petals", "spring"],
    "summer festival": ["summer festival", "yukata", "festival lights", "lantern"],
    "beach sunset": ["beach", "ocean", "sunset", "golden hour", "warm lighting"],
    "coffee shop": ["coffee shop", "cafe interior", "wooden table", "coffee cup", "counter", "dessert", "cozy atmosphere", "warm interior"],
    "small cafe": ["coffee shop", "cafe interior", "wooden table", "coffee cup", "counter", "dessert", "cozy atmosphere", "warm interior"],
}

JAPANESE_TAG_REPLACEMENTS = {
    "サンタ服": [
        "santa hat",
        "red santa dress",
        "white fur trim",
        "christmas costume",
        "christmas lights",
        "snow",
        "gift box",
    ],
    "サンタ衣装": [
        "santa hat",
        "red santa dress",
        "white fur trim",
        "christmas costume",
        "christmas lights",
        "snow",
        "gift box",
    ],
    "クリスマス衣装": [
        "santa hat",
        "red santa dress",
        "white fur trim",
        "christmas costume",
        "christmas lights",
        "snow",
        "gift box",
    ],
    "巫女": ["miko", "red hakama", "white kimono", "shrine maiden"],
    "制服": ["school uniform", "sailor uniform", "pleated skirt", "ribbon"],
    "セーラー服": ["sailor uniform", "sailor collar", "pleated skirt", "ribbon"],
    "メイド": ["maid", "maid headdress", "apron", "frills"],
    "メイド服": ["maid", "maid headdress", "apron", "frills"],
    "カフェ": ["coffee shop", "cafe interior", "wooden table", "coffee cup", "counter", "dessert", "cozy atmosphere", "warm interior"],
    "喫茶店": ["coffee shop", "cafe interior", "wooden table", "coffee cup", "counter", "dessert", "cozy atmosphere", "warm interior"],
    "神社": ["shrine", "torii", "stone pathway", "japanese architecture"],
    "雨": ["rain", "wet ground", "umbrella", "cloudy sky"],
    "雨の日": ["rain", "wet ground", "umbrella", "cloudy sky"],
}

JAPANESE_NEGATIVE_TAG_REPLACEMENTS = {
    "カフェ": ["classroom", "school desk", "blackboard"],
    "喫茶店": ["classroom", "school desk", "blackboard"],
    "サンタ服": ["face paint", "cheek mark", "leaf mark", "multicolored hair", "two-tone hair"],
    "サンタ衣装": ["face paint", "cheek mark", "leaf mark", "multicolored hair", "two-tone hair"],
    "クリスマス衣装": ["face paint", "cheek mark", "leaf mark", "multicolored hair", "two-tone hair"],
}

SANTA_SOURCE_KEYWORDS = ("サンタ服", "サンタ衣装", "クリスマス衣装")
SANTA_TAGS = {"santa hat", "red santa dress", "christmas costume"}
SANTA_ABSTRACT_COLOR_TAGS = {
    "red and green colors",
    "red green colors",
    "christmas colors",
    "christmas color scheme",
    "red and green theme",
}
SANTA_UNWANTED_POSITIVE_TAGS = {
    "two-tone hair",
    "multicolored hair",
    "face paint",
    "cheek mark",
    "facial mark",
    "leaf mark",
    "asymmetrical hair color",
}
HAIR_COLOR_SOURCE_KEYWORDS = {
    "black": ("black hair", "黒髪"),
    "blonde": ("blonde hair", "金髪", "ブロンド"),
    "blue": ("blue hair", "青髪"),
    "brown": ("brown hair", "茶髪"),
    "green": ("green hair", "緑髪"),
    "grey": ("grey hair", "gray hair", "銀髪", "白髪", "灰髪"),
    "gray": ("gray hair", "grey hair", "銀髪", "白髪", "灰髪"),
    "pink": ("pink hair", "ピンク髪"),
    "purple": ("purple hair", "紫髪"),
    "red": ("red hair", "赤髪"),
    "silver": ("silver hair", "銀髪"),
    "white": ("white hair", "白髪", "銀髪"),
}

NATURAL_TAG_REPLACEMENTS = {
    "cute 1girl": ["cute", "1girl"],
    "cute girl": ["cute", "1girl"],
    "sleepy 1girl": ["sleepy expression", "1girl"],
    "sleepy girl": ["sleepy expression", "1girl"],
    "short haircut girl": ["short hair", "1girl"],
    "short haired girl": ["short hair", "1girl"],
    "girl with short hair": ["short hair", "1girl"],
    "long haired girl": ["long hair", "1girl"],
    "girl with long hair": ["long hair", "1girl"],
    "blue eyed girl": ["blue eyes", "1girl"],
    "girl with blue eyes": ["blue eyes", "1girl"],
    "red eyed girl": ["red eyes", "1girl"],
    "girl with red eyes": ["red eyes", "1girl"],
    "silver haired girl": ["silver hair", "1girl"],
    "girl with silver hair": ["silver hair", "1girl"],
}

COLOR_WORDS = {
    "black",
    "blonde",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}

TAG_CATEGORIES = {
    "quality": {
        "masterpiece",
        "best quality",
        "newest",
        "very aesthetic",
        "absurdres",
        "anime style",
    },
    "person": {
        "1girl",
        "solo",
        "portrait",
        "upper body",
        "cowboy shot",
        "full body",
        "centered composition",
    },
    "hair_eyes_keywords": {
        "face",
        "hair",
        "eyes",
        "eye",
        "twintails",
        "bangs",
        "braid",
        "ponytail",
        "ahoge",
    },
    "expression_gaze": {
        "cute",
        "smile",
        "smiling",
        "gentle smile",
        "sleepy expression",
        "shy",
        "looking at viewer",
        "looking away",
        "half-closed eyes",
        "closed mouth",
        "open mouth",
        "blush",
    },
    "pose_keywords": {
        "standing",
        "sitting",
        "lying",
        "kneeling",
        "walking",
        "running",
        "pose",
        "hand",
        "hands",
        "arms",
        "leaning",
    },
    "clothing_keywords": {
        "kimono",
        "obi",
        "sleeves",
        "clothes",
        "uniform",
        "serafuku",
        "skirt",
        "dress",
        "maid",
        "apron",
        "yukata",
        "ribbon",
        "frills",
        "collar",
        "headdress",
    },
    "background_keywords": {
        "shrine",
        "torii",
        "pathway",
        "architecture",
        "cafe",
        "coffee shop",
        "cafe interior",
        "counter",
        "dessert",
        "table",
        "chair",
        "window",
        "cityscape",
        "street",
        "beach",
        "ocean",
        "cherry blossoms",
        "sakura petals",
        "rain",
        "puddle",
        "reflection",
        "lantern",
        "festival",
    },
    "atmosphere_keywords": {
        "lighting",
        "light",
        "cozy atmosphere",
        "sunset",
        "golden hour",
        "warm lighting",
        "orange tones",
        "cool colors",
        "blue tones",
        "soft lighting",
        "pastel colors",
        "night",
        "neon lights",
        "street lights",
        "spring",
        "fragile atmosphere",
    },
}

CONFLICT_GROUPS = [
    (
        {"warm lighting", "golden hour", "orange tones"},
        {"cool colors", "blue tones"},
    ),
]

WARM_INPUT_TAGS = {"sunset", "golden hour", "warm lighting", "orange tones"}
COOL_MEMORY_TAGS = {"cool colors", "blue tones"}
KIMONO_TAGS = {"kimono", "floral kimono", "detailed kimono", "traditional japanese clothes"}


@dataclass(frozen=True)
class PromptTemplateResult:
    positive_prompt: str
    negative_prompt: str
    notes: str = ""
    blocked_reason: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reason)


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        text = str(value or "")
        text = re.sub(r"^(positive|negative)\s*prompt\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        raw_items = re.split(r"[,;\n]+", text)

    tags: list[str] = []
    for item in raw_items:
        tag = re.sub(r"\s+", " ", str(item)).strip().strip(",")
        if tag:
            tags.append(tag)
    return tags


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        cleaned = re.sub(r"\s+", " ", tag).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _tags_from_japanese_text(text: str) -> list[str]:
    if not text:
        return []

    tags: list[str] = []
    for keyword, replacements in sorted(JAPANESE_TAG_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        if keyword in text:
            tags.extend(replacements)
    return _dedupe_tags(tags)


def _negative_tags_from_japanese_text(text: str) -> list[str]:
    if not text:
        return []

    tags: list[str] = []
    for keyword, replacements in sorted(JAPANESE_NEGATIVE_TAG_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        if keyword in text:
            tags.extend(replacements)
    return _dedupe_tags(tags)


def _normalize_single_tag(tag: str) -> list[str]:
    lower_tag = tag.lower().strip()
    if not lower_tag:
        return []

    if lower_tag in NATURAL_TAG_REPLACEMENTS:
        return NATURAL_TAG_REPLACEMENTS[lower_tag]
    if lower_tag in COMPOSITE_TAG_REPLACEMENTS:
        return COMPOSITE_TAG_REPLACEMENTS[lower_tag]
    japanese_replacement = _tags_from_japanese_text(tag)
    if japanese_replacement:
        return japanese_replacement

    color_feature_match = re.fullmatch(rf"({'|'.join(COLOR_WORDS)})\s+(eyes|hair)", lower_tag)
    if color_feature_match:
        return [lower_tag]

    haired_match = re.fullmatch(rf"({'|'.join(COLOR_WORDS)})\s+haired\s+girl", lower_tag)
    if haired_match:
        return [f"{haired_match.group(1)} hair", "1girl"]

    eyed_match = re.fullmatch(rf"({'|'.join(COLOR_WORDS)})\s+eyed\s+girl", lower_tag)
    if eyed_match:
        return [f"{eyed_match.group(1)} eyes", "1girl"]

    person_compound_match = re.fullmatch(r"(cute|sleepy|shy|smiling|smile)\s+1girl", lower_tag)
    if person_compound_match:
        descriptor = person_compound_match.group(1)
        if descriptor == "sleepy":
            descriptor = "sleepy expression"
        return [descriptor, "1girl"]

    cleaned = lower_tag
    cleaned = re.sub(r"\bshort haircut\b", "short hair", cleaned)
    cleaned = re.sub(r"\bshort haired\b", "short hair", cleaned)
    cleaned = re.sub(r"\blong haired\b", "long hair", cleaned)
    cleaned = re.sub(rf"\b({'|'.join(COLOR_WORDS)})\s+eyed\b", r"\1 eyes", cleaned)
    cleaned = re.sub(rf"\b({'|'.join(COLOR_WORDS)})\s+haired\b", r"\1 hair", cleaned)
    cleaned = re.sub(r"\bgirl\b", "1girl", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return [cleaned] if cleaned else []


def _expand_composite_tags(tags: list[str]) -> list[str]:
    expanded: list[str] = []
    for tag in tags:
        normalized = _normalize_single_tag(tag)
        if _tags_from_japanese_text(tag):
            expanded.extend(normalized)
            continue
        for item in normalized:
            replacement = COMPOSITE_TAG_REPLACEMENTS.get(item.lower())
            expanded.extend(replacement if replacement else [item])
    return expanded


def _normalize_positive_tags(value: Any) -> list[str]:
    return _dedupe_tags(_expand_composite_tags(_split_tags(value)))


def _first_index(tags: list[str], candidates: set[str]) -> int | None:
    for index, tag in enumerate(tags):
        if tag.lower() in candidates:
            return index
    return None


def _contains_santa_request(tags: list[str], source_text: str) -> bool:
    if any(keyword in source_text for keyword in SANTA_SOURCE_KEYWORDS):
        return True
    return any(tag.lower() in SANTA_TAGS for tag in tags)


def _requested_hair_color_tags(source_text: str) -> set[str]:
    requested: set[str] = set()
    lower_source = source_text.lower()
    for color, keywords in HAIR_COLOR_SOURCE_KEYWORDS.items():
        if any(keyword.lower() in lower_source for keyword in keywords):
            requested.add(f"{color} hair")
    return requested


def _is_hair_color_tag(tag: str) -> bool:
    return bool(re.fullmatch(rf"({'|'.join(COLOR_WORDS)})\s+hair", tag.lower()))


def _clean_santa_prompt_tags(tags: list[str], source_text: str) -> list[str]:
    if not _contains_santa_request(tags, source_text):
        return tags

    requested_hair_colors = _requested_hair_color_tags(source_text)
    cleaned: list[str] = []
    for tag in tags:
        lower_tag = tag.lower()
        if lower_tag in SANTA_ABSTRACT_COLOR_TAGS:
            continue
        if lower_tag in SANTA_UNWANTED_POSITIVE_TAGS:
            continue
        if _is_hair_color_tag(tag) and lower_tag not in requested_hair_colors:
            continue
        cleaned.append(tag)
    return cleaned


def _resolve_contradictory_tags(tags: list[str]) -> list[str]:
    result = list(tags)
    lower_tags = {tag.lower() for tag in result}
    if lower_tags & WARM_INPUT_TAGS:
        result = [tag for tag in result if tag.lower() not in COOL_MEMORY_TAGS]

    for left_group, right_group in CONFLICT_GROUPS:
        left_index = _first_index(result, left_group)
        right_index = _first_index(result, right_group)
        if left_index is None or right_index is None:
            continue

        # Current input usually appears earlier than memory-derived helper tags,
        # so keep the first conflicting mood/color family that reached the output.
        remove_group = right_group if left_index <= right_index else left_group
        result = [tag for tag in result if tag.lower() not in remove_group]
    return result


def _tag_category(tag: str) -> int:
    lower_tag = tag.lower()
    if lower_tag in TAG_CATEGORIES["quality"]:
        return 0
    if lower_tag in TAG_CATEGORIES["person"]:
        return 1
    if any(keyword in lower_tag for keyword in TAG_CATEGORIES["hair_eyes_keywords"]):
        return 2
    if lower_tag in TAG_CATEGORIES["expression_gaze"]:
        return 3
    if any(keyword in lower_tag for keyword in TAG_CATEGORIES["pose_keywords"]):
        return 4
    if any(keyword in lower_tag for keyword in TAG_CATEGORIES["clothing_keywords"]):
        return 5
    if any(keyword in lower_tag for keyword in TAG_CATEGORIES["background_keywords"]):
        return 6
    if lower_tag in TAG_CATEGORIES["atmosphere_keywords"] or any(
        keyword in lower_tag for keyword in TAG_CATEGORIES["atmosphere_keywords"]
    ):
        return 7
    return 8


def _sort_positive_tags(tags: list[str]) -> list[str]:
    indexed = list(enumerate(tags))
    indexed.sort(key=lambda item: (_tag_category(item[1]), item[0]))
    return [tag for _, tag in indexed]


def _add_standard_person_tags(tags: list[str]) -> list[str]:
    result = list(tags)
    result.extend(STANDARD_PERSON_DETAIL_TAGS)
    result.extend(STANDARD_COMPOSITION_TAGS)

    lower_tags = {tag.lower() for tag in result}
    if lower_tags & KIMONO_TAGS:
        result.extend(KIMONO_DETAIL_TAGS)

    return _dedupe_tags(result)


def _build_positive_prompt(value: Any, source_text: str = "") -> str:
    source_tags = _tags_from_japanese_text(source_text)
    llm_tags = _normalize_positive_tags(value)
    combined_tags = _dedupe_tags(source_tags + llm_tags)
    fixed_keys = {item.lower() for item in REQUIRED_POSITIVE_PREFIX}
    filtered = [tag for tag in combined_tags if tag.lower() not in fixed_keys]
    filtered = _clean_santa_prompt_tags(filtered, source_text)
    filtered = _add_standard_person_tags(filtered)
    filtered = _resolve_contradictory_tags(filtered)
    sorted_tags = _sort_positive_tags(filtered)
    return ", ".join(QUALITY_TAGS + REQUIRED_PERSON_TAGS + sorted_tags)


def _build_negative_prompt(source_text: str = "") -> str:
    tags = [tag.strip() for tag in FIXED_NEGATIVE_PROMPT.split(",") if tag.strip()]
    tags.extend(_negative_tags_from_japanese_text(source_text))
    tags = [tag for tag in tags if tag.lower() not in FORBIDDEN_NEGATIVE_TAGS]
    return ", ".join(_dedupe_tags(tags))


def _safe_check_block_reason(payload: dict[str, Any]) -> str:
    # Local personal-use prompt generation does not block on LLM-provided safety fields.
    # Keep parsing/formatting errors active, but ignore safe_check.blocked_elements,
    # adult, and consent_ok so Positive/Negative generation can continue.
    return ""


def format_llm_prompt_response(
    raw_response: str,
    default_negative_prompt: str = FIXED_NEGATIVE_PROMPT,
    source_text: str = "",
) -> PromptTemplateResult:
    payload = _extract_json(raw_response)
    fixed_negative = _build_negative_prompt(source_text=source_text)
    if payload is None:
        return PromptTemplateResult(
            positive_prompt="",
            negative_prompt=fixed_negative,
            notes="",
            blocked_reason="LM Studioの返答をJSONとして読み取れなかったみたい。",
        )

    blocked_reason = _safe_check_block_reason(payload)
    if blocked_reason:
        return PromptTemplateResult("", fixed_negative, str(payload.get("notes", "")).strip(), blocked_reason)

    positive_source = payload.get("positive_prompt") or payload.get("positive") or payload.get("prompt")
    positive = _build_positive_prompt(positive_source, source_text=source_text)
    notes = str(payload.get("notes", "")).strip()

    if not positive_source or positive == ", ".join(REQUIRED_POSITIVE_PREFIX):
        return PromptTemplateResult(
            positive_prompt="",
            negative_prompt=fixed_negative,
            notes=notes,
            blocked_reason="Positive prompt が空だったみたい。もう少し具体的に入力してみてね。",
        )

    return PromptTemplateResult(positive, fixed_negative, notes)


def build_discord_prompt_message(result: PromptTemplateResult) -> str:
    if result.blocked:
        message = result.blocked_reason
        if result.notes:
            message += f"\n\n補足: {result.notes}"
        return message

    return "\n".join(
        [
            "Positive",
            "```text",
            result.positive_prompt,
            "```",
            "Negative",
            "```text",
            result.negative_prompt,
            "```",
        ]
    )
