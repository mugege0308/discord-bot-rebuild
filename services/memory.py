import json
import logging
import re
from datetime import datetime
from pathlib import Path

from services.prompt_context_conflicts import (
    filter_conflicting_weak_visual_hints,
    weak_visual_hint_conflicts_with_request,
)
from services.weather import JST
from services.runtime_paths import project_path


logger = logging.getLogger("secretary-bot.memory_service")

DATA_DIR = project_path("data")
MEMORY_FILE = DATA_DIR / "memory.json"
MAX_MEMORY_TEXT_LENGTH = 500
MAX_CONTEXT_CHARS = 1400

HIGH_PRIORITY_WORDS = ("好き", "すき", "よく使う", "お気に入り", "好み", "推し", "優先")
IMAGE_RELATED_WORDS = (
    "画像",
    "プロンプト",
    "絵",
    "イラスト",
    "生成",
    "stable diffusion",
    "comfyui",
    "モデル",
    "lora",
    "背景",
    "構図",
    "色",
    "画風",
    "タグ",
)

ALWAYS_MEMORY_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # Keep the mood hint, but do not turn it into an additional facial or
    # lighting decision.  The current request (and the LM's explicit
    # expression, if any) owns those details.
    (("可愛い", "かわいい", "cute"), ("cute",)),
    (("人物品質", "顔をきれい", "顔品質", "綺麗な顔"), ("detailed face", "beautiful face")),
    (("自然な目", "目を自然", "目が自然", "綺麗な目"), ("detailed eyes", "expressive eyes")),
    (("暖色", "あたたかい色", "温かい色", "オレンジ"), ("warm lighting", "orange tones")),
    # Color preference may carry a restrained palette hint.  Lighting is a
    # separate visual decision and must not be pulled in by a color memory.
    (("寒色", "青系", "青っぽい", "クール"), ("cool colors", "blue tones")),
    (("淡い", "淡色", "パステル", "柔らかい光", "やわらかい光", "soft light"), ("pastel colors", "soft colors", "gentle lighting")),
    (("暗め", "ダーク"), ("low light", "moody lighting")),
    (("眠たげ", "眠そう"), ("sleepy expression", "half-closed eyes", "soft expression")),
    (("病弱", "儚い"), ("pale skin", "fragile atmosphere", "soft expression")),
    (("銀髪", "シルバー髪"), ("silver hair",)),
    (("赤目", "赤い目"), ("red eyes",)),
    (("青目", "青い目"), ("blue eyes",)),
    (("短髪", "ショートヘア"), ("short hair",)),
    (("ロングヘア", "長髪"), ("long hair",)),
)

CONDITIONAL_MEMORY_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (("夕焼け", "夕日", "sunset"), ("夕焼け", "夕日", "夕方", "sunset", "golden hour"), ("sunset", "golden hour", "warm lighting")),
    (("夜", "night"), ("夜", "夜景", "night"), ("night", "low light", "moody lighting")),
    (("和服", "着物", "kimono"), ("和服", "着物", "kimono"), ("kimono", "obi", "traditional japanese clothes")),
    (("浴衣", "yukata"), ("浴衣", "夏祭り", "yukata"), ("yukata", "obi", "summer festival")),
    (("神社", "鳥居", "shrine", "torii"), ("神社", "鳥居", "shrine", "torii"), ("shrine", "torii", "stone pathway")),
    (("桜", "さくら", "sakura"), ("桜", "さくら", "花見", "sakura"), ("cherry blossoms", "sakura petals", "spring")),
    (("雨", "雨の日"), ("雨", "傘", "rain", "umbrella"), ("rain", "wet street", "puddle", "reflection")),
    (("カフェ", "喫茶店"), ("カフェ", "喫茶店", "cafe", "coffee shop"), ("cafe", "coffee cup", "table", "window")),
    (("制服", "セーラー服"), ("制服", "セーラー服", "学校", "教室", "school uniform"), ("school uniform", "serafuku", "pleated skirt")),
    (("メイド", "maid"), ("メイド", "maid"), ("maid", "maid headdress", "apron", "frills")),
    (
        ("サンタ服", "サンタ衣装", "クリスマス衣装", "サンタ", "christmas"),
        ("サンタ服", "サンタ", "クリスマス", "christmas"),
        ("santa hat", "red santa dress", "white fur trim", "christmas costume", "christmas lights", "snow", "gift box"),
    ),
)

MEMORY_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    ALWAYS_MEMORY_TAG_RULES
    + tuple((memory_keywords, mapped_tags) for memory_keywords, _query_keywords, mapped_tags in CONDITIONAL_MEMORY_TAG_RULES)
)

MAX_PROMPT_MEMORY_TAGS = 5

EXPLICIT_DIFFERENT_BACKGROUND_WORDS = (
    "カフェ",
    "喫茶店",
    "街",
    "市街",
    "都市",
    "学校",
    "教室",
    "部屋",
    "室内",
    "寝室",
    "リビング",
    "オフィス",
    "駅",
    "電車",
    "海",
    "浜辺",
    "森",
    "公園",
)

MEMORY_PERSON_TAGS = {
    "sleepy expression",
    "half-closed eyes",
    "soft expression",
    "pale skin",
    "fragile atmosphere",
    "silver hair",
    "red eyes",
    "blue eyes",
    "short hair",
    "long hair",
}

MEMORY_STYLE_COLOR_TAGS = {
    "warm lighting",
    "golden hour",
    "orange tones",
    "cool colors",
    "blue tones",
    "soft lighting",
    "pastel colors",
    "soft colors",
    "gentle lighting",
    "night",
    "low light",
    "moody lighting",
}

MEMORY_COLOR_TAGS = {
    "cool colors",
    "blue tones",
    "orange tones",
    "pastel colors",
    "soft colors",
}

MEMORY_LIGHTING_TAGS = {
    "warm lighting",
    "golden hour",
    "soft lighting",
    "gentle lighting",
    "night",
    "low light",
    "moody lighting",
}

MEMORY_MOOD_TAGS = {
    "cute",
    "sleepy expression",
    "half-closed eyes",
    "soft expression",
    "fragile atmosphere",
}

MEMORY_APPEARANCE_TAGS = {
    "pale skin",
    "silver hair",
    "red eyes",
    "blue eyes",
    "short hair",
    "long hair",
}

MEMORY_CLOTHING_TAGS = {
    "kimono",
    "obi",
    "traditional japanese clothes",
    "yukata",
    "school uniform",
    "serafuku",
    "pleated skirt",
    "maid",
    "maid headdress",
    "apron",
    "frills",
}

MEMORY_BACKGROUND_TAGS = {
    "summer festival",
    "shrine",
    "torii",
    "stone pathway",
    "cherry blossoms",
    "sakura petals",
    "spring",
    "rain",
    "wet street",
    "puddle",
    "reflection",
    "cafe",
    "coffee cup",
    "table",
    "window",
}

MEMORY_TAGS_SUPPRESSED_BY_EXPLICIT_BACKGROUND = {
    "kimono",
    "obi",
    "traditional japanese clothes",
    "yukata",
    "summer festival",
    "shrine",
    "torii",
    "stone pathway",
}

_PROMPT_MEMORY_CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "color": (
        "色",
        "配色",
        "色味",
        "色合い",
        "カラー",
        "寒色",
        "暖色",
        "青系",
        "赤系",
        "オレンジ",
        "パステル",
        "カラフル",
        "color",
        "palette",
        "tone",
    ),
    "lighting": (
        "光",
        "明かり",
        "照明",
        "ライティング",
        "日差し",
        "夕日",
        "夕焼け",
        "月明かり",
        "夜",
        "lighting",
        "light",
        "sunlight",
        "moonlight",
    ),
    "mood": (
        "表情",
        "雰囲気",
        "ムード",
        "かわいい",
        "可愛い",
        "かっこいい",
        "クール",
        "笑顔",
        "cute",
        "mood",
        "atmosphere",
        "expression",
    ),
    "background": (
        "背景",
        "場所",
        "舞台",
        "風景",
        "神社",
        "鳥居",
        "室内",
        "屋外",
        "カフェ",
        "学校",
        "海",
        "森",
        "background",
        "location",
        "scene",
        "indoor",
        "outdoor",
    ),
    "clothing": (
        "服",
        "服装",
        "衣装",
        "着物",
        "和服",
        "制服",
        "メイド",
        "ドレス",
        "パーカー",
        "outfit",
        "clothing",
        "kimono",
        "dress",
    ),
    "appearance": (
        "髪",
        "髪型",
        "目",
        "瞳",
        "顔",
        "肌",
        "外見",
        "hair",
        "eyes",
        "face",
        "skin",
        "appearance",
    ),
}

_PROMPT_MEMORY_EXPLICIT_USE_CUES: dict[str, tuple[str, ...]] = {
    "color": ("いつもの色味", "いつもの色", "好みの色味", "好みの色"),
    "lighting": ("いつもの光", "好みの光", "いつもの照明", "好みの照明"),
    "mood": ("いつもの雰囲気", "好みの雰囲気", "いつもの感じ"),
    "background": ("いつもの背景", "好みの背景", "いつもの場所", "好みの場所"),
    "clothing": ("いつもの服", "好みの服", "いつもの衣装", "好みの衣装"),
    "all": ("メモリを使って", "メモリを反映", "好みを反映", "好みで生成"),
}

_PROMPT_MEMORY_SPECIFIC_TAG_CUES: dict[str, tuple[str, ...]] = {
    "cool colors": ("寒色", "クール", "cool"),
    "blue tones": ("寒色", "青系", "青み", "青い", "クール", "blue"),
    "orange tones": ("暖色", "オレンジ", "orange", "warm color"),
    "pastel colors": ("淡色", "パステル", "pastel"),
    "soft colors": ("淡色", "柔らかい色", "soft color"),
    "warm lighting": ("暖色の光", "暖かい光", "暖かな光", "warm light"),
    "golden hour": ("夕日", "夕焼け", "夕暮れ", "golden hour"),
    "soft lighting": ("柔らかい光", "柔らかな光", "ソフトな光", "soft light"),
    "gentle lighting": ("柔らかい光", "柔らかな光", "gentle lighting"),
    "night": ("夜", "夜景", "night"),
    "low light": ("暗め", "暗い", "low light"),
    "moody lighting": ("暗め", "ダーク", "moody"),
}


def _load_memory_data() -> dict[str, list[dict[str, object]]]:
    if not MEMORY_FILE.exists():
        return {}
    try:
        value = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load memory file.")
        return {}
    if not isinstance(value, dict):
        return {}

    data: dict[str, list[dict[str, object]]] = {}
    for user_id, items in value.items():
        if isinstance(items, list):
            data[str(user_id)] = [item for item in items if isinstance(item, dict)]
    return data


def _save_memory_data(data: dict[str, list[dict[str, object]]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_id(item: dict[str, object], default: int = -1) -> int:
    try:
        return int(item.get("id", default))
    except (TypeError, ValueError):
        return default


def _next_memory_id(items: list[dict[str, object]]) -> int:
    current_ids = [_memory_id(item, 0) for item in items]
    return (max(current_ids) if current_ids else 0) + 1


def _memory_text(item: dict[str, object]) -> str:
    return str(item.get("text", "")).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def _is_high_priority(text: str) -> bool:
    lower_text = text.lower()
    return any(word.lower() in lower_text for word in HIGH_PRIORITY_WORDS)


def _is_image_related(text: str) -> bool:
    lower_text = text.lower()
    return any(word.lower() in lower_text for word in IMAGE_RELATED_WORDS)


def _query_is_image_related(query: str) -> bool:
    return _is_image_related(query)


def _query_terms(query: str) -> list[str]:
    normalized = query.lower().replace("　", " ")
    return [term for term in re.split(r"\s+", normalized) if len(term) >= 2]


def memory_text_to_tags(text: str) -> list[str]:
    tags: list[str] = []
    lower_text = text.lower()
    for keywords, mapped_tags in MEMORY_TAG_RULES:
        if any(keyword.lower() in lower_text for keyword in keywords):
            tags.extend(mapped_tags)
    return _dedupe(tags)


def _text_has_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _always_memory_tags(text: str) -> list[str]:
    tags: list[str] = []
    for keywords, mapped_tags in ALWAYS_MEMORY_TAG_RULES:
        if _text_has_keywords(text, keywords):
            tags.extend(mapped_tags)
    return _dedupe(tags)


def _conditional_memory_tag_decisions(text: str, query: str) -> tuple[list[str], list[str]]:
    adopted: list[str] = []
    excluded: list[str] = []
    for memory_keywords, query_keywords, mapped_tags in CONDITIONAL_MEMORY_TAG_RULES:
        if not _text_has_keywords(text, memory_keywords):
            continue
        if _text_has_keywords(query, query_keywords):
            adopted.extend(mapped_tags)
        else:
            excluded.extend(mapped_tags)
    return _dedupe(adopted), _dedupe(excluded)


def _prompt_text_has_unrelated_conditional_memory(text: str, query: str) -> bool:
    for memory_keywords, query_keywords, _mapped_tags in CONDITIONAL_MEMORY_TAG_RULES:
        if _text_has_keywords(text, memory_keywords) and not _text_has_keywords(query, query_keywords):
            return True
    return False


def _query_has_explicit_different_background(query: str) -> bool:
    lower_query = query.lower()
    return any(word.lower() in lower_query for word in EXPLICIT_DIFFERENT_BACKGROUND_WORDS)


def _tag_priority(tag: str) -> int:
    lower_tag = tag.lower()
    if lower_tag in MEMORY_PERSON_TAGS:
        return 0
    if lower_tag in MEMORY_STYLE_COLOR_TAGS:
        return 1
    if lower_tag in MEMORY_CLOTHING_TAGS:
        return 2
    if lower_tag in MEMORY_BACKGROUND_TAGS:
        return 3
    return 4


def _prompt_memory_tag_category(tag: str) -> str | None:
    normalized = tag.lower().strip()
    if normalized in MEMORY_COLOR_TAGS:
        return "color"
    if normalized in MEMORY_LIGHTING_TAGS:
        return "lighting"
    if normalized in MEMORY_MOOD_TAGS:
        return "mood"
    if normalized in MEMORY_BACKGROUND_TAGS:
        return "background"
    if normalized in MEMORY_CLOTHING_TAGS:
        return "clothing"
    if normalized in MEMORY_APPEARANCE_TAGS:
        return "appearance"
    return None


def _prompt_memory_categories_for_query(query: str) -> set[str]:
    lower_query = query.lower()
    categories = {
        category
        for category, cues in _PROMPT_MEMORY_CATEGORY_CUES.items()
        if any(cue.lower() in lower_query for cue in cues)
    }
    for category, cues in _PROMPT_MEMORY_EXPLICIT_USE_CUES.items():
        if any(cue.lower() in lower_query for cue in cues):
            categories.add(category)
    if "all" in categories:
        categories.update(_PROMPT_MEMORY_CATEGORY_CUES)
    return categories


def _prompt_memory_tag_allowed(tag: str, query: str, categories: set[str] | None = None) -> bool:
    category = _prompt_memory_tag_category(tag)
    if category is None:
        return True
    allowed_categories = categories if categories is not None else _prompt_memory_categories_for_query(query)
    if category not in allowed_categories:
        return False

    # A specific request such as "寒色系" should not reopen every tag in the
    # broad color category.  Generic category wording such as "色味" still
    # permits all memories in that category.
    lower_query = query.lower()
    category_specific_cues = tuple(
        cue.lower()
        for known_tag, cues in _PROMPT_MEMORY_SPECIFIC_TAG_CUES.items()
        if _prompt_memory_tag_category(known_tag) == category
        for cue in cues
    )
    if not any(cue in lower_query for cue in category_specific_cues):
        return True
    tag_cues = _PROMPT_MEMORY_SPECIFIC_TAG_CUES.get(tag.lower(), ())
    return any(cue.lower() in lower_query for cue in tag_cues)


def _prompt_memory_text_allowed(text: str, query: str, categories: set[str]) -> bool:
    """Keep raw preference text only when every known visual category in it is relevant.

    A memory such as "暖色、神社、和服が好き" must not be passed as raw prose
    merely because the request mentions one of those categories.  The
    category-specific converted tags are filtered separately below.
    """

    tags = memory_text_to_tags(text)
    visual_categories = {
        category
        for tag in tags
        if (category := _prompt_memory_tag_category(tag)) is not None
    }
    lower_text = text.lower()
    visual_categories.update(
        category
        for category, cues in _PROMPT_MEMORY_CATEGORY_CUES.items()
        if any(cue.lower() in lower_text for cue in cues)
    )
    return not visual_categories or visual_categories.issubset(categories)


def _select_prompt_memory_tags(query: str, tags: list[str]) -> list[str]:
    deduped_tags = _dedupe(tags)
    filtered_visual_tags, visual_conflicts = filter_conflicting_weak_visual_hints(deduped_tags, query)
    allowed_categories = _prompt_memory_categories_for_query(query)
    category_suppressed = [
        tag
        for tag in filtered_visual_tags
        if not _prompt_memory_tag_allowed(tag, query, allowed_categories)
    ]
    filtered_visual_tags = [
        tag for tag in filtered_visual_tags
        if _prompt_memory_tag_allowed(tag, query, allowed_categories)
    ]
    has_explicit_background = _query_has_explicit_different_background(query)
    lower_query = query.lower()

    filtered_tags: list[str] = []
    for tag in filtered_visual_tags:
        lower_tag = tag.lower()
        explicitly_requested = lower_tag in lower_query
        if (
            has_explicit_background
            and lower_tag in MEMORY_TAGS_SUPPRESSED_BY_EXPLICIT_BACKGROUND
            and not explicitly_requested
        ):
            continue
        filtered_tags.append(tag)

    indexed_tags = list(enumerate(filtered_tags))
    indexed_tags.sort(key=lambda item: (_tag_priority(item[1]), item[0]))
    selected_tags = [tag for _, tag in indexed_tags[:MAX_PROMPT_MEMORY_TAGS]]
    logger.info(
        "Prompt memory tags selected: query_length=%s explicit_background=%s selected=%s suppressed=%s",
        len(query),
        has_explicit_background,
        selected_tags,
        _dedupe(
            visual_conflicts
            + category_suppressed
            + [tag for tag in deduped_tags if tag not in selected_tags]
        ),
    )
    return selected_tags


def _collect_prompt_memory_tags(query: str, memories: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    always_tags: list[str] = []
    conditional_adopted_tags: list[str] = []
    excluded_tags: list[str] = []
    allowed_categories = _prompt_memory_categories_for_query(query)
    for item in memories:
        text = _memory_text(item)
        if not text:
            continue
        always_tags.extend(_always_memory_tags(text))
        conditional_adopted, conditional_excluded = _conditional_memory_tag_decisions(text, query)
        conditional_adopted_tags.extend(
            tag
            for tag in conditional_adopted
            if _prompt_memory_tag_allowed(tag, query, allowed_categories)
        )
        excluded_tags.extend(conditional_excluded)

    conditional_selected = _dedupe(conditional_adopted_tags)
    remaining_slots = max(MAX_PROMPT_MEMORY_TAGS - len(conditional_selected), 0)
    always_selected = _select_prompt_memory_tags(query, always_tags)
    selected_tags = _dedupe(conditional_selected + always_selected)[:MAX_PROMPT_MEMORY_TAGS]
    if remaining_slots:
        selected_tags = _dedupe(conditional_selected + always_selected[:remaining_slots])[:MAX_PROMPT_MEMORY_TAGS]

    prioritized_tags = _dedupe(conditional_adopted_tags + always_tags)
    unselected_tags = [tag for tag in prioritized_tags if tag not in selected_tags]
    all_excluded_tags = _dedupe(excluded_tags + unselected_tags)
    logger.info(
        "Prompt memory tag classification: query_length=%s adopted=%s excluded=%s",
        len(query),
        selected_tags,
        all_excluded_tags,
    )
    return selected_tags, all_excluded_tags


def _memory_text_conflicts_with_prompt_background(query: str, text: str) -> bool:
    if not _query_has_explicit_different_background(query):
        return False
    tags = memory_text_to_tags(text)
    return any(tag.lower() in MEMORY_TAGS_SUPPRESSED_BY_EXPLICIT_BACKGROUND for tag in tags)


def add_memory(user_id: int, text: str) -> str:
    content = text.strip()
    if not content:
        return "覚える内容を入力してください。"
    if len(content) > MAX_MEMORY_TEXT_LENGTH:
        return f"少し長すぎます。{MAX_MEMORY_TEXT_LENGTH}文字以内にしてください。"

    data = _load_memory_data()
    key = str(user_id)
    items = data.setdefault(key, [])
    memory_id = _next_memory_id(items)
    items.append(
        {
            "id": memory_id,
            "text": content,
            "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        }
    )
    _save_memory_data(data)
    return f"覚えておきました。ID: {memory_id}"


def list_memories(user_id: int) -> str:
    items = _load_memory_data().get(str(user_id), [])
    if not items:
        return "保存済みメモリはまだありません。"

    lines = ["保存済みメモリ:"]
    for item in items:
        text = _memory_text(item)
        if text:
            marker = " ★" if _is_high_priority(text) else ""
            tags = memory_text_to_tags(text)
            tag_text = f" -> {', '.join(tags)}" if tags else ""
            lines.append(f"{_memory_id(item)}. {text}{marker}{tag_text}")
    return "\n".join(lines)


def delete_memory(user_id: int, memory_id: int) -> str:
    data = _load_memory_data()
    key = str(user_id)
    items = data.get(key, [])
    kept = [item for item in items if _memory_id(item) != memory_id]
    if len(kept) == len(items):
        return "そのIDのメモリは見つかりませんでした。"

    if kept:
        data[key] = kept
    else:
        data.pop(key, None)
    _save_memory_data(data)
    return f"メモリID {memory_id} を削除しました。"


def rename_linked_image_memory(user_id: int, memory_id: int, name: str) -> bool:
    data = _load_memory_data()
    items = data.get(str(user_id), [])
    for item in items:
        if _memory_id(item) != int(memory_id):
            continue
        text = _memory_text(item)
        if not text.startswith("画像参考メモ"):
            return False
        suffix = re.sub(r"^画像参考メモ(?:「[^」]*」)?\s*:\s*", "", text)
        item["text"] = f"画像参考メモ「{name}」: {suffix}"
        _save_memory_data(data)
        return True
    return False


def _score_memory(item: dict[str, object], query: str) -> int:
    text = _memory_text(item)
    lower_text = text.lower()
    lower_query = query.lower()
    terms = _query_terms(query)

    score = 0
    if lower_query and lower_query in lower_text:
        score += 8
    score += sum(2 for term in terms if term in lower_text)
    if _is_high_priority(text):
        score += 10
    if _query_is_image_related(query) and (_is_image_related(text) or memory_text_to_tags(text)):
        score += 10
    score += min(_memory_id(item, 0), 1000) // 100
    return score


def search_memories(user_id: int, query: str, limit: int = 7) -> list[dict[str, object]]:
    logger.info("Memory search start: user_id=%s query_length=%s limit=%s", user_id, len(query), limit)
    items = _load_memory_data().get(str(user_id), [])
    if not items:
        logger.info("Memory search end: user_id=%s total=0 returned=0", user_id)
        return []

    scored = [(_score_memory(item, query), item) for item in items if _memory_text(item)]
    scored.sort(key=lambda pair: (pair[0], _memory_id(pair[1], 0)), reverse=True)

    relevant = [item for score, item in scored if score > 0]
    if relevant:
        result = relevant[:limit]
    else:
        high_priority = [item for item in items if _is_high_priority(_memory_text(item))]
        result = high_priority[-min(limit, 5) :] if high_priority else items[-min(limit, 3) :]

    logger.info("Memory search end: user_id=%s total=%s returned=%s", user_id, len(items), len(result))
    return result


def build_memory_summary(
    user_id: int,
    query: str,
    mode: str = "generic",
    *,
    compact: bool = False,
) -> str:
    memories = search_memories(user_id, query)
    if mode == "prompt":
        # Image reference memories are selected from data/image_memory with their
        # own relevance and tag-limit rules. Never feed their copied free text to the LLM.
        memories = [item for item in memories if not _memory_text(item).startswith("画像参考メモ")]
        conflicting_memory_ids = [
            _memory_id(item)
            for item in memories
            if weak_visual_hint_conflicts_with_request(_memory_text(item), query)
        ]
        if conflicting_memory_ids:
            memories = [
                item
                for item in memories
                if not weak_visual_hint_conflicts_with_request(_memory_text(item), query)
            ]
            logger.info(
                "Prompt memory items excluded by explicit visual request: query_length=%s memory_ids=%s",
                len(query),
                conflicting_memory_ids,
            )
    if not memories:
        return ""

    preference_items: list[str] = []
    image_items: list[str] = []
    other_items: list[str] = []
    skipped_conditional_memory_count = 0
    skipped_visual_memory_count = 0
    converted_tags: list[str] = []
    excluded_tags: list[str] = []
    allowed_categories: set[str] = set()
    if mode == "prompt":
        allowed_categories = _prompt_memory_categories_for_query(query)
        converted_tags, excluded_tags = _collect_prompt_memory_tags(query, memories)

    for item in memories:
        text = _memory_text(item)
        if not text:
            continue
        tags = memory_text_to_tags(text)
        if mode == "prompt" and not _prompt_memory_text_allowed(text, query, allowed_categories):
            skipped_visual_memory_count += 1
            continue
        if mode == "prompt" and _prompt_text_has_unrelated_conditional_memory(text, query):
            skipped_conditional_memory_count += 1
            continue
        if _is_high_priority(text):
            preference_items.append(text)
        elif _is_image_related(text) or tags:
            image_items.append(text)
        else:
            other_items.append(text)

    lines = [] if compact else [
        "このユーザーの好み・前提として、以下を優先的に反映してください。",
        "今回の依頼に明示された人物・服装・背景・場所・色・構図を最優先してください。",
        "今回の依頼と矛盾する保存済みメモリは使わないでください。",
    ]
    if preference_items:
        label = "Memory preference: " if compact else "特に優先する好み: "
        lines.append(label + " / ".join(preference_items[:4]))
    if mode == "prompt" or _query_is_image_related(query):
        if image_items:
            label = "Memory image hints: " if compact else "画像生成・プロンプト関連で優先する情報: "
            lines.append(label + " / ".join(image_items[:5]))
        if converted_tags:
            if compact:
                lines.append("Memory tags (use only when relevant, max 5): " + ", ".join(converted_tags))
            else:
                lines.append(
                    "初号ちゃんがメモリからルールベースで変換したタグ。"
                    "今回の依頼に合う場合だけ、補助的に最大5個まで反映してください: "
                    + ", ".join(converted_tags)
                )
    if other_items:
        label = "Memory notes: " if compact else "その他の参考情報: "
        lines.append(label + " / ".join(other_items[:4]))
    if mode == "chat":
        lines.append("回答では、可能なら上記の保存済みメモを自然に引用または活用してください。")
    if mode == "prompt" and not compact:
        lines.append(
            "プロンプト生成では、背景系メモリより人物・画風・色味のメモリを優先してください。"
            "入力文にカフェ・街・学校・部屋など別背景がある場合、神社・鳥居・和服などの背景寄りメモリは混ぜないでください。"
        )
    if skipped_conditional_memory_count:
        logger.info(
            "Prompt memory conditional texts skipped: user_id=%s count=%s query_length=%s excluded_tags=%s",
            user_id,
            skipped_conditional_memory_count,
            len(query),
            excluded_tags,
        )
    if mode == "prompt":
        logger.info(
            "Prompt memory category filter: user_id=%s query_length=%s allowed_categories=%s "
            "skipped_visual_texts=%s resolved_tags=%s",
            user_id,
            len(query),
            sorted(allowed_categories),
            skipped_visual_memory_count,
            converted_tags,
        )

    if compact and lines:
        lines.insert(0, "Memory hints (weak; current request wins; ignore conflicts):")
    summary = "\n".join(lines)
    if len(summary) > MAX_CONTEXT_CHARS:
        summary = summary[: MAX_CONTEXT_CHARS - 20] + "\n...(truncated)"
    logger.info("Memory summary built: user_id=%s mode=%s summary_length=%s", user_id, mode, len(summary))
    return summary


def append_memory_to_system_prompt(
    system_prompt: str,
    user_id: int | None,
    query: str,
    mode: str = "generic",
    *,
    compact: bool = False,
) -> str:
    if user_id is None:
        return system_prompt

    memory_summary = build_memory_summary(user_id, query, mode=mode, compact=compact)
    if not memory_summary:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{memory_summary}"
