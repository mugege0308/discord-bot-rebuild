import json
import random
from pathlib import Path
from typing import Any


OPTIONS_FILE = Path("data/character_template_options.json")
CATEGORY_ORDER = ("hair_color", "hair_style", "eye_color", "expression", "outfit", "background", "atmosphere")
SPECIAL_NONE = "__none__"
SPECIAL_RANDOM = "__random__"


def load_character_template_options() -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(OPTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"キャラクターテンプレート設定を読み込めません: {OPTIONS_FILE}") from exc
    if not isinstance(value, dict):
        raise ValueError("キャラクターテンプレート設定の形式が不正です。")
    for key in CATEGORY_ORDER:
        category = value.get(key)
        if not isinstance(category, dict) or not isinstance(category.get("options"), list):
            raise ValueError(f"キャラクターテンプレート設定に {key} がありません。")
    return value


def select_options(category: dict[str, Any]) -> list[tuple[str, str]]:
    result = [("指定なし", SPECIAL_NONE), ("ランダム", SPECIAL_RANDOM)]
    for item in category["options"]:
        if isinstance(item, dict) and str(item.get("label", "")).strip() and str(item.get("value", "")).strip():
            result.append((str(item["label"]).strip(), str(item["value"]).strip()))
    return result[:25]


def resolve_character_choices(
    selected: dict[str, str],
    options: dict[str, dict[str, Any]],
    rng: random.Random | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    rng = rng or random.SystemRandom()
    resolved: dict[str, str] = {}
    display: dict[str, str] = {}
    for key in CATEGORY_ORDER:
        category = options[key]
        chosen = selected.get(key, SPECIAL_NONE)
        candidates = select_options(category)[2:]
        if chosen == SPECIAL_RANDOM and candidates:
            label, value = rng.choice(candidates)
            resolved[key] = value
            display[key] = f"ランダム → {label}"
        elif chosen == SPECIAL_NONE:
            display[key] = "指定なし"
        else:
            resolved[key] = chosen
            display[key] = next((label for label, value in candidates if value == chosen), chosen)
    return resolved, display


def build_character_instruction(resolved: dict[str, str], options: dict[str, dict[str, Any]]) -> str:
    parts = [resolved[key] for key in CATEGORY_ORDER if resolved.get(key)]
    if not parts:
        return "魅力的な1人のアニメ調キャラクターを、おまかせで生成してください。"
    return "、".join(parts) + "の、1人のアニメ調キャラクターを生成してください。"


def build_selection_summary(display: dict[str, str], options: dict[str, dict[str, Any]]) -> str:
    return "\n".join(f"- {options[key]['label']}: {display.get(key, '指定なし')}" for key in CATEGORY_ORDER)
