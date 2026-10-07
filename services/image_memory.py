import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from services.prompt_context_conflicts import filter_conflicting_weak_visual_hints
from services.runtime_paths import project_path, resolve_project_path


logger = logging.getLogger("secretary-bot.image_memory")

JST = timezone(timedelta(hours=9))
DATA_DIR = project_path("data", "image_memory")
IMAGES_DIR = DATA_DIR / "images"
FILE_PREFIX = "ref_"
IMAGE_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
TAG_KEYS = ("person", "hair", "eyes", "expression", "outfit", "background", "composition", "lighting_mood", "sd_tags")
PRIORITY_TAG_KEYS = ("hair", "eyes", "expression", "person", "outfit", "composition", "background", "lighting_mood", "sd_tags")
MAX_NORMAL_REFERENCE_TAGS = 5
MAX_IMAGE_MEMORY_NAME_LENGTH = 50
AUTO_NAME_TAGS = {
    "black hair": "黒髪", "brown hair": "茶髪", "blonde hair": "金髪", "silver hair": "銀髪",
    "white hair": "白髪", "red hair": "赤髪", "blue hair": "青髪", "pink hair": "ピンク髪",
    "short hair": "ショート", "long hair": "ロング", "bob cut": "ボブ", "ponytail": "ポニーテール",
    "twintails": "ツインテール", "twin tails": "ツインテール", "braid": "三つ編み",
    "red eyes": "赤目", "blue eyes": "青目", "green eyes": "緑目", "purple eyes": "紫目",
    "golden eyes": "金目", "smile": "笑顔", "maid": "メイド", "cheerleader": "チア",
}
QUERY_TAG_HINTS = {
    "金髪": ("blonde hair",), "赤目": ("red eyes",), "赤い目": ("red eyes",),
    "青目": ("blue eyes",), "青い目": ("blue eyes",), "ロング": ("long hair",),
    "長髪": ("long hair",), "ツインテール": ("twintails", "twin tails"),
    "制服": ("school uniform",), "チア": ("cheerleader",), "笑顔": ("smile",),
}

SECTION_KEYS = {
    "人物": "person",
    "髪色・髪型": "hair",
    "目": "eyes",
    "表情": "expression",
    "服装": "outfit",
    "背景": "background",
    "構図": "composition",
    "光・雰囲気": "lighting_mood",
    "Stable Diffusion向けタグ候補": "sd_tags",
}


def _memory_path(memory_id: int) -> Path:
    return DATA_DIR / f"{FILE_PREFIX}{memory_id:04d}.json"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read image memory JSON: path=%s", path)
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _write_image_atomic(memory_id: int, image_bytes: bytes, content_type: str) -> Path:
    extension = IMAGE_EXTENSIONS.get(content_type.lower().split(";", 1)[0].strip())
    if not extension:
        raise ValueError("保存できる画像形式はJPEG / PNG / WebPです。")
    if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
        raise ValueError("保存画像は8MB以内にしてください。")
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{FILE_PREFIX}{memory_id:04d}{extension}"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(image_bytes)
    os.replace(tmp_path, path)
    return path


def resolve_image_memory_image_path(item: dict[str, Any], memory_id: int | None = None) -> Path | None:
    raw_path = item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        item_id = int(item.get("id", memory_id or 0))
    except (TypeError, ValueError):
        return None
    candidate = Path(raw_path.strip())
    candidate = resolve_project_path(candidate, candidate)
    try:
        resolved = candidate.resolve()
        allowed_dir = (DATA_DIR / "images").resolve()
    except OSError:
        return None
    if resolved.parent != allowed_dir:
        logger.warning("Rejected image memory path outside images directory: id=%s path=%s", item_id, raw_path)
        return None
    if not re.fullmatch(rf"{FILE_PREFIX}{item_id:04d}\.(?:jpg|png|webp)", resolved.name, flags=re.IGNORECASE):
        logger.warning("Rejected mismatched image memory filename: id=%s path=%s", item_id, raw_path)
        return None
    return resolved if resolved.is_file() else None


def _existing_ids() -> list[int]:
    if not DATA_DIR.exists():
        return []
    ids: list[int] = []
    for path in DATA_DIR.glob(f"{FILE_PREFIX}*.json"):
        match = re.fullmatch(rf"{FILE_PREFIX}(\d+)\.json", path.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def next_image_memory_id() -> int:
    ids = _existing_ids()
    return (max(ids) if ids else 0) + 1


def sanitize_image_memory_name(value: str, *, allow_empty: bool = False) -> str:
    name = "".join(character for character in str(value) if character.isprintable() and character not in "\r\n")
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > MAX_IMAGE_MEMORY_NAME_LENGTH:
        name = name[:MAX_IMAGE_MEMORY_NAME_LENGTH].rstrip()
    if not name and not allow_empty:
        raise ValueError("名前は1〜50文字で入力してください。")
    return name


def get_image_memory_name(item: dict[str, Any], memory_id: int | None = None) -> str:
    try:
        item_id = int(item.get("id", memory_id or 0))
    except (TypeError, ValueError):
        item_id = memory_id or 0
    raw_name = str(item.get("name") or item.get("title") or "")
    name = sanitize_image_memory_name(raw_name, allow_empty=True)
    return name or f"参考画像 #{item_id}"


def get_image_memory_rating(item: dict[str, Any]) -> int:
    try:
        rating = int(item.get("rating", 0))
    except (TypeError, ValueError):
        return 0
    return rating if 0 <= rating <= 5 else 0


def get_image_memory_use_for_generation(item: dict[str, Any]) -> bool:
    value = item.get("use_for_generation", True)
    return value if isinstance(value, bool) else True


def format_image_memory_rating(rating: int) -> str:
    return "未評価" if rating == 0 else "★" * rating + "☆" * (5 - rating)


def _auto_image_memory_name(structured_tags: dict[str, list[str]] | None, memory_id: int) -> str:
    candidates: list[str] = []
    for key in ("hair", "eyes", "expression", "outfit", "person", "sd_tags"):
        for tag in (structured_tags or {}).get(key, []):
            label = AUTO_NAME_TAGS.get(str(tag).strip().lower())
            if label and label not in candidates:
                candidates.append(label)
            if len(candidates) >= 3:
                return "・".join(candidates)
    return "・".join(candidates) if candidates else f"参考画像 #{memory_id}"


def suggest_image_memory_name(structured_tags: dict[str, list[str]], memory_id: int | None = None) -> str:
    return _auto_image_memory_name(structured_tags, memory_id or next_image_memory_id())


def _extract_sections(analysis_text: str) -> dict[str, str]:
    sections = {key: "" for key in SECTION_KEYS.values()}
    current_key: str | None = None
    buffers: dict[str, list[str]] = {key: [] for key in SECTION_KEYS.values()}

    for raw_line in analysis_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matched_key = None
        for heading, field_name in SECTION_KEYS.items():
            if line.startswith(f"【{heading}】"):
                matched_key = field_name
                remainder = line.replace(f"【{heading}】", "", 1).strip()
                current_key = field_name
                if remainder:
                    buffers[field_name].append(remainder)
                break

        if matched_key:
            continue
        if current_key:
            buffers[current_key].append(line)

    for key, lines in buffers.items():
        sections[key] = "\n".join(lines).strip()
    return sections


def _guess_title(analysis_text: str, note: str) -> str:
    if note.strip():
        return note.strip()[:80]
    first_line = next((line.strip() for line in analysis_text.splitlines() if line.strip()), "")
    return first_line[:80] if first_line else "画像参考メモ"


def save_image_memory(
    user_id: int,
    analysis_text: str,
    note: str = "",
    title: str = "",
    memory_id: int | None = None,
    structured_tags: dict[str, list[str]] | None = None,
    name: str = "",
    image_bytes: bytes | None = None,
    image_content_type: str = "",
) -> dict[str, Any]:
    image_memory_id = next_image_memory_id()
    legacy_sections = _extract_sections(analysis_text)
    sections = {key: list((structured_tags or {}).get(key, [])) for key in TAG_KEYS}
    if not structured_tags:
        sections = {key: ([value] if value else []) for key, value in legacy_sections.items()}
    requested_name = sanitize_image_memory_name(name or title, allow_empty=True)
    display_name = requested_name or _auto_image_memory_name(structured_tags, image_memory_id)
    image_path: Path | None = None
    if image_bytes is not None:
        image_path = _write_image_atomic(image_memory_id, image_bytes, image_content_type)
    payload: dict[str, Any] = {
        "id": image_memory_id,
        "user_id": int(user_id),
        "name": display_name,
        "title": display_name,
        "rating": 0,
        "use_for_generation": True,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "note": note.strip(),
        "memory_id": memory_id,
        "analysis_text": analysis_text.strip(),
        **sections,
    }
    if image_path is not None:
        payload["image_path"] = image_path.as_posix()
    try:
        _write_json_atomic(_memory_path(image_memory_id), payload)
    except Exception:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        raise
    return payload


def rename_image_memory(user_id: int, memory_id: int, name: str) -> dict[str, Any] | None:
    new_name = sanitize_image_memory_name(name)
    payload = load_image_memory(memory_id)
    if not payload:
        return None
    try:
        owner_id = int(payload.get("user_id", 0))
    except (TypeError, ValueError):
        return None
    if owner_id != int(user_id):
        return None
    payload["name"] = new_name
    payload["title"] = new_name
    _write_json_atomic(_memory_path(memory_id), payload)
    return payload


def rate_image_memory(user_id: int, memory_id: int, rating: int) -> dict[str, Any] | None:
    if not 1 <= int(rating) <= 5:
        raise ValueError("評価は1〜5で指定してください。")
    payload = load_image_memory(memory_id)
    if not payload:
        return None
    try:
        owner_id = int(payload.get("user_id", 0))
    except (TypeError, ValueError):
        return None
    if owner_id != int(user_id):
        return None
    payload["rating"] = int(rating)
    payload.setdefault("use_for_generation", True)
    _write_json_atomic(_memory_path(memory_id), payload)
    return payload


def set_image_memory_generation_use(user_id: int, memory_id: int, enabled: bool) -> dict[str, Any] | None:
    payload = load_image_memory(memory_id)
    if not payload:
        return None
    try:
        owner_id = int(payload.get("user_id", 0))
    except (TypeError, ValueError):
        return None
    if owner_id != int(user_id):
        return None
    payload["rating"] = get_image_memory_rating(payload)
    payload["use_for_generation"] = bool(enabled)
    _write_json_atomic(_memory_path(memory_id), payload)
    return payload


def get_owned_image_memory_name(user_id: int, memory_id: int) -> str:
    payload = load_image_memory(memory_id)
    try:
        owner_id = int((payload or {}).get("user_id", 0))
    except (TypeError, ValueError):
        owner_id = 0
    if not payload or owner_id != int(user_id):
        raise ValueError(f"画像参考メモ #{memory_id} は見つからないか、使用できません。")
    return get_image_memory_name(payload, memory_id)


def load_image_memory(memory_id: int) -> dict[str, Any] | None:
    path = _memory_path(memory_id)
    if not path.exists():
        return None
    return _safe_read_json(path)


def list_image_memories(user_id: int) -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob(f"{FILE_PREFIX}*.json")):
        payload = _safe_read_json(path)
        if not payload:
            continue
        try:
            owner_id = int(payload.get("user_id", 0))
        except (TypeError, ValueError):
            continue
        if owner_id == int(user_id):
            items.append(payload)
    items.sort(key=lambda item: int(item.get("id", 0)))
    return items


def _tag_list(value: Any) -> list[str]:
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        values = re.split(r"[,\n]", value.lstrip("：:"))
    else:
        return []
    result: list[str] = []
    for value in values:
        tag = str(value).strip().lower().lstrip("：:")
        if tag and tag not in result:
            result.append(tag)
    return result


def _all_tags(item: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in PRIORITY_TAG_KEYS:
        for tag in _tag_list(item.get(key)):
            if not re.fullmatch(r"[a-z0-9_+()' -]+", tag) or tag.count(" ") > 7:
                continue
            if tag not in result:
                result.append(tag)
    return result


def _is_related_image_memory(item: dict[str, Any], query: str) -> bool:
    lower_query = query.lower()
    searchable = " ".join(str(item.get(key) or "") for key in ("title", "note", "analysis_text")).lower()
    terms = [term for term in re.split(r"[\s、,。]+", lower_query) if len(term) >= 2]
    if any(term in searchable for term in terms):
        return True
    tags = _all_tags(item)
    for keyword, hinted_tags in QUERY_TAG_HINTS.items():
        if keyword in query and any(tag in tags for tag in hinted_tags):
            return True
    return any(tag in lower_query for tag in tags if len(tag) >= 3)


def select_image_memory_tags(
    user_id: int,
    query: str,
    reference_id: int | None = None,
) -> tuple[list[int], list[str]]:
    if reference_id is not None:
        item = load_image_memory(reference_id)
        if not item or int(item.get("user_id", 0)) != int(user_id):
            raise ValueError(f"画像参考メモ #{reference_id} は見つからないか、使用できません。")
        tags = _all_tags(item)
        rating = get_image_memory_rating(item)
        logger.info(
            "Image memory tags adopted: user_id=%s reference_ids=%s ratings=%s explicit=true use_for_generation=%s tags=%s",
            user_id, [reference_id], {reference_id: rating}, get_image_memory_use_for_generation(item), tags,
        )
        return [reference_id], tags

    selected_ids: list[int] = []
    selected_tags: list[str] = []
    candidates = [item for item in list_image_memories(user_id) if get_image_memory_use_for_generation(item)]
    candidates.sort(key=lambda item: (get_image_memory_rating(item), int(item.get("id", 0))), reverse=True)
    selected_ratings: dict[int, int] = {}
    for item in candidates:
        if not _is_related_image_memory(item, query):
            continue
        memory_id = int(item.get("id", 0))
        tag_count_before = len(selected_tags)
        category_tags: list[str] = []
        for key in PRIORITY_TAG_KEYS:
            # Avoid the broad sd_tags bucket until specific categories have had priority.
            for tag in _tag_list(item.get(key)):
                if re.fullmatch(r"[a-z0-9_+()' -]+", tag) and tag.count(" ") <= 7 and tag not in category_tags:
                    category_tags.append(tag)
        category_tags = [tag for _, tag in sorted(enumerate(category_tags), key=lambda pair: (0 if any(word in pair[1] for word in ("hair", "eyes", "eye", "smile", "expression")) else 1, pair[0]))]
        for tag in category_tags:
                if tag not in selected_tags:
                    selected_tags.append(tag)
                if len(selected_tags) >= MAX_NORMAL_REFERENCE_TAGS:
                    break
        if len(selected_tags) > tag_count_before and memory_id not in selected_ids:
            selected_ids.append(memory_id)
            selected_ratings[memory_id] = get_image_memory_rating(item)
        if len(selected_tags) >= MAX_NORMAL_REFERENCE_TAGS:
            break
    logger.info(
        "Image memory tags adopted: user_id=%s reference_ids=%s ratings=%s explicit=false tags=%s",
        user_id, selected_ids, selected_ratings, selected_tags,
    )
    return selected_ids, selected_tags


def build_image_memory_prompt_context(user_id: int, query: str, reference_id: int | None = None) -> str:
    reference_ids, tags = select_image_memory_tags(user_id, query, reference_id)
    tags, excluded_tags = filter_conflicting_weak_visual_hints(tags, query)
    if excluded_tags:
        logger.info(
            "Image memory tags excluded by explicit visual request: user_id=%s reference_ids=%s excluded=%s",
            user_id,
            reference_ids,
            excluded_tags,
        )
    if not tags:
        return ""
    mode = "明示指定のため全タグ" if reference_id is not None else f"関連タグのみ最大{MAX_NORMAL_REFERENCE_TAGS}個"
    return f"画像参考Memory #{', #'.join(map(str, reference_ids))} ({mode}): {', '.join(tags)}"


def delete_image_memory(user_id: int, memory_id: int) -> bool:
    payload = load_image_memory(memory_id)
    if not payload:
        return False
    try:
        owner_id = int(payload.get("user_id", 0))
    except (TypeError, ValueError):
        return False
    if owner_id != int(user_id):
        return False

    image_path = resolve_image_memory_image_path(payload, memory_id)
    if image_path is not None:
        try:
            image_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to delete image memory source image: id=%s user_id=%s", memory_id, user_id)
            return False

    try:
        _memory_path(memory_id).unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Failed to delete image memory: id=%s user_id=%s", memory_id, user_id)
        return False
    return True


def build_image_memory_list_message(user_id: int, page: int = 1, page_size: int = 10) -> str:
    items = list_image_memories(user_id)
    if not items:
        return "画像参考メモはまだありません。"

    page = max(page, 1)
    total_pages = max((len(items) + page_size - 1) // page_size, 1)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    selected = items[start : start + page_size]

    lines = [f"画像参考メモ一覧 {page}/{total_pages}:"]
    for item in selected:
        memory_id = item.get("id", "?")
        title = get_image_memory_name(item, int(memory_id))
        rating = format_image_memory_rating(get_image_memory_rating(item))
        generation_use = "ON" if get_image_memory_use_for_generation(item) else "OFF"
        person = item.get("person")
        summary = (", ".join(person) if isinstance(person, list) else str(person or item.get("analysis_text") or "")).replace("\n", " ")
        if len(summary) > 70:
            summary = summary[:67].rstrip() + "..."
        lines.append(f"ID {memory_id} / {title} / 評価: {rating} / 生成利用: {generation_use} / {summary}")
    return "\n".join(lines)


def get_image_memory_summary(item: dict[str, Any], limit: int = 100) -> str:
    person = item.get("person")
    summary = (", ".join(person) if isinstance(person, list) else str(person or item.get("analysis_text") or "")).replace("\n", " ").strip()
    if len(summary) > limit:
        summary = summary[: max(limit - 3, 0)].rstrip() + "..."
    return summary or "要約なし"


def build_image_memory_detail_message(user_id: int, memory_id: int) -> str:
    item = load_image_memory(memory_id)
    if not item:
        return "そのIDの画像参考メモは見つかりませんでした。"
    try:
        owner_id = int(item.get("user_id", 0))
    except (TypeError, ValueError):
        return "そのIDの画像参考メモは見つかりませんでした。"
    if owner_id != int(user_id):
        return "その画像参考メモは表示できません。"

    title = get_image_memory_name(item, memory_id)
    rating = format_image_memory_rating(get_image_memory_rating(item))
    generation_use = "ON" if get_image_memory_use_for_generation(item) else "OFF"
    analysis_text = str(item.get("analysis_text") or "")
    raw_sd_tags = item.get("sd_tags") or []
    sd_tags = ", ".join(raw_sd_tags) if isinstance(raw_sd_tags, list) else str(raw_sd_tags)
    message = (
        f"ID {memory_id}「{title}」\n"
        f"評価: {rating}\n"
        f"生成利用: {generation_use}\n\n"
        f"{analysis_text}"
    )
    message += "\n\n元画像: あり" if resolve_image_memory_image_path(item, memory_id) else "\n\n元画像なし"
    if sd_tags and "Stable Diffusion向けタグ候補" not in analysis_text:
        message += f"\n\n【Stable Diffusion向けタグ候補】\n{sd_tags}"
    return message.strip()
