from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DevTaskEngine(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"


class DevTaskRoutingError(ValueError):
    """An error whose message is safe to show in Discord."""


# Adjust these keyword lists and the default engine here as routing needs evolve.
DEFAULT_ENGINE = DevTaskEngine.CODEX

LARGE_SCALE_KEYWORDS: tuple[str, ...] = (
    "リファクタ",
    "refactor",
    "再設計",
    "全体的に",
    "アーキテクチャ",
    "architecture",
    "複数ファイル",
    "大規模",
    "移行して",
    "migrate",
    "migration",
    "書き直して",
    "rewrite",
)

SMALL_SCALE_KEYWORDS: tuple[str, ...] = (
    "修正して",
    "直して",
    "fix",
    "バグ",
    "bug",
    "タイポ",
    "typo",
    "軽微な",
    "ちょっとした",
)

LARGE_PROMPT_LENGTH_THRESHOLD = 200

_VALID_ENGINE_OPTIONS = {"auto", DevTaskEngine.CODEX.value, DevTaskEngine.CLAUDE.value}


@dataclass(frozen=True)
class DevTaskRoutingDecision:
    engine: DevTaskEngine
    reason: str
    overridden: bool


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def route_dev_task(prompt: str, engine_option: str) -> DevTaskRoutingDecision:
    normalized_option = str(engine_option).strip().lower()
    if normalized_option not in _VALID_ENGINE_OPTIONS:
        raise DevTaskRoutingError(f"engine_optionの指定が不正です: {engine_option!r}")

    if normalized_option in (DevTaskEngine.CODEX.value, DevTaskEngine.CLAUDE.value):
        engine = DevTaskEngine(normalized_option)
        return DevTaskRoutingDecision(engine=engine, reason="手動指定のため", overridden=True)

    has_large_keyword = _contains_keyword(prompt, LARGE_SCALE_KEYWORDS)
    has_small_keyword = _contains_keyword(prompt, SMALL_SCALE_KEYWORDS)

    if has_large_keyword and has_small_keyword:
        engine = DevTaskEngine.CLAUDE
        reason = "大規模・小規模の両方のキーワードを検出したため"
    elif has_large_keyword:
        engine = DevTaskEngine.CLAUDE
        reason = "大規模な変更を示すキーワードを検出したため"
    elif has_small_keyword:
        engine = DevTaskEngine.CODEX
        reason = "小規模な修正を示すキーワードを検出したため"
    elif len(prompt) >= LARGE_PROMPT_LENGTH_THRESHOLD:
        engine = DevTaskEngine.CLAUDE
        reason = f"プロンプトが{LARGE_PROMPT_LENGTH_THRESHOLD}文字以上のため"
    else:
        engine = DEFAULT_ENGINE
        reason = f"判定条件に一致しなかったためデフォルトの{DEFAULT_ENGINE.value}を使用"

    return DevTaskRoutingDecision(engine=engine, reason=reason, overridden=False)
