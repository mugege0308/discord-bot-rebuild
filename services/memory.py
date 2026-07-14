import json
import logging
import re
from datetime import datetime
from pathlib import Path

from services.weather import JST


logger = logging.getLogger("secretary-bot.memory_service")

DATA_DIR = Path("data")
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
    (("可愛い", "かわいい", "cute"), ("cute", "soft expression")),
    (("人物品質", "顔をきれい", "顔品質", "綺麗な顔"), ("detailed face", "beautiful face")),
    (("自然な目", "目を自然", "目が自然", "綺麗な目"), ("detailed eyes", "expressive eyes")),
    (("暖色", "あたたかい色", "温かい色", "オレンジ"), ("warm lighting", "orange tones")),
    (("寒色", "青系", "青っぽい", "クール"), ("cool colors", "blue tones", "soft lighting")),
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


def _select_prompt_memory_tags(query: str, tags: list[str]) -> list[str]:
    deduped_tags = _dedupe(tags)
    has_explicit_background = _query_has_explicit_different_background(query)
    lower_query = query.lower()

    filtered_tags: list[str] = []
    for tag in deduped_tags:
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
        [tag for tag in deduped_tags if tag not in selected_tags],
    )
    return selected_tags


def _collect_prompt_memory_tags(query: str, memories: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    always_tags: list[str] = []
    conditional_adopted_tags: list[str] = []
    excluded_tags: list[str] = []
    for item in memories:
        text = _memory_text(item)
        if not text:
            continue
        always_tags.extend(_always_memory_tags(text))
        conditional_adopted, conditional_excluded = _conditional_memory_tag_decisions(text, query)
        conditional_adopted_tags.extend(conditional_adopted)
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


def build_memory_summary(user_id: int, query: str, mode: str = "generic") -> str:
    memories = search_memories(user_id, query)
    if mode == "prompt":
        # Image reference memories are selected from data/image_memory with their
        # own relevance and tag-limit rules. Never feed their copied free text to the LLM.
        memories = [item for item in memories if not _memory_text(item).startswith("画像参考メモ")]
    if not memories:
        return ""

    preference_items: list[str] = []
    image_items: list[str] = []
    other_items: list[str] = []
    skipped_conditional_memory_count = 0
    converted_tags: list[str] = []
    excluded_tags: list[str] = []
    if mode == "prompt":
        converted_tags, excluded_tags = _collect_prompt_memory_tags(query, memories)

    for item in memories:
        text = _memory_text(item)
        if not text:
            continue
        tags = memory_text_to_tags(text)
        if mode == "prompt" and _prompt_text_has_unrelated_conditional_memory(text, query):
            skipped_conditional_memory_count += 1
            continue
        if _is_high_priority(text):
            preference_items.append(text)
        elif _is_image_related(text) or tags:
            image_items.append(text)
        else:
            other_items.append(text)

    lines = [
        "このユーザーの好み・前提として、以下を優先的に反映してください。",
        "今回の依頼に明示された人物・服装・背景・場所・色・構図を最優先してください。",
        "今回の依頼と矛盾する保存済みメモリは使わないでください。",
    ]
    if preference_items:
        lines.append("特に優先する好み: " + " / ".join(preference_items[:4]))
    if mode == "prompt" or _query_is_image_related(query):
        if image_items:
            lines.append("画像生成・プロンプト関連で優先する情報: " + " / ".join(image_items[:5]))
        if converted_tags:
            lines.append(
                "初号ちゃんがメモリからルールベースで変換したタグ。"
                "今回の依頼に合う場合だけ、補助的に最大5個まで反映してください: "
                + ", ".join(converted_tags)
            )
    if other_items:
        lines.append("その他の参考情報: " + " / ".join(other_items[:4]))
    if mode == "chat":
        lines.append("回答では、可能なら上記の保存済みメモを自然に引用または活用してください。")
    if mode == "prompt":
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

    summary = "\n".join(lines)
    if len(summary) > MAX_CONTEXT_CHARS:
        summary = summary[: MAX_CONTEXT_CHARS - 20] + "\n...(truncated)"
    logger.info("Memory summary built: user_id=%s mode=%s summary_length=%s", user_id, mode, len(summary))
    return summary


def append_memory_to_system_prompt(system_prompt: str, user_id: int | None, query: str, mode: str = "generic") -> str:
    if user_id is None:
        return system_prompt

    memory_summary = build_memory_summary(user_id, query, mode=mode)
    if not memory_summary:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{memory_summary}"
