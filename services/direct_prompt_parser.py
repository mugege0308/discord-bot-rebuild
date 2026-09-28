"""Parsing helpers for /generate_ai's English direct-input mode.

The parser deliberately knows nothing about Discord, ComfyUI, or the prompt
designer.  Keeping it as a small pure function makes the copy/paste formats
easy to test without starting either AI service.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


JAPANESE_AI_INPUT_MODE = "ja_ai"
ENGLISH_DIRECT_INPUT_MODE = "en_direct"
PROMPT_IMPORT_INPUT_MODE = "prompt_import"
INPUT_MODE_LABELS = {
    JAPANESE_AI_INPUT_MODE: "日本語・AI変換",
    ENGLISH_DIRECT_INPUT_MODE: "英語・直接入力",
    PROMPT_IMPORT_INPUT_MODE: "外部Promptを整えて使う",
}


@dataclass(frozen=True)
class DirectPromptResult:
    positive: str
    negative: str
    had_section_headings: bool


_HEADING_RE = re.compile(
    r"^\s*(?P<label>positive\s+prompt|positive|negative\s+prompt|negative|"
    r"ポジティブプロンプト|ポジティブ|ネガティブプロンプト|ネガティブ)"
    r"\s*[:：](?:\s*(?P<inline>.*))?\s*$",
    re.IGNORECASE,
)
_OTHER_SECTION_RE = re.compile(
    r"^\s*(?:summary|assumptions|explanation|description|notes?|"
    r"説明|概要|要約|前提|仮定|注記|備考)\s*[:：].*$",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^\s*(?:```|~~~)(?:[A-Za-z0-9_+-]+)?\s*$")


def _heading_kind(line: str) -> tuple[str, str] | None:
    match = _HEADING_RE.match(unicodedata.normalize("NFKC", line))
    if match is None:
        return None
    label = " ".join(match.group("label").casefold().split())
    if label in {"positive", "positive prompt", "ポジティブ", "ポジティブプロンプト"}:
        return "positive", match.group("inline") or ""
    return "negative", match.group("inline") or ""


def _clean_section(value: str) -> str:
    lines = []
    for line in value.splitlines():
        if _CODE_FENCE_RE.match(line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def parse_direct_prompt(text: str, default_negative: str) -> DirectPromptResult:
    """Extract Positive/Negative sections from a single pasted input.

    A recognized heading is only accepted at the beginning of a line.  When
    no recognized heading exists, the entire cleaned input becomes Positive.
    Missing or empty Negative uses ``default_negative``; an empty Positive is
    intentionally returned to the caller so it can present a user-facing
    validation error before generation starts.
    """

    source = str(text or "")
    lines = source.splitlines()
    has_section_headings = any(_heading_kind(line) is not None for line in lines)
    if not has_section_headings:
        return DirectPromptResult(
            positive=_clean_section(source),
            negative=_clean_section(default_negative),
            had_section_headings=False,
        )

    sections: dict[str, list[str]] = {"positive": [], "negative": []}
    current: str | None = None
    for line in lines:
        heading = _heading_kind(line)
        if heading is not None:
            current, inline = heading
            if inline:
                sections[current].append(inline)
            continue
        if current is not None and _OTHER_SECTION_RE.match(unicodedata.normalize("NFKC", line)):
            current = None
            continue
        if current is not None:
            sections[current].append(line)

    positive = _clean_section("\n".join(sections["positive"]))
    negative = _clean_section("\n".join(sections["negative"]))
    return DirectPromptResult(
        positive=positive,
        negative=negative or _clean_section(default_negative),
        had_section_headings=True,
    )
