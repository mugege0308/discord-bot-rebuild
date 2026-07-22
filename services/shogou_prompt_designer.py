import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.image_memory import build_image_memory_prompt_context
from services.memory import append_memory_to_system_prompt
from services.prompt_generator import (
    PromptGeneratorError,
    _post_chat_completion,
)


logger = logging.getLogger("secretary-bot.shogou_prompt_designer")

INSTRUCTION_FILE = Path("shogou_chan_prompt_designer_instruction.md")
REQUIRED_KEYS = ("positive", "negative", "summary", "assumptions")


@dataclass(frozen=True)
class ShogouPromptResult:
    positive: str = ""
    negative: str = ""
    summary: str = ""
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    blocked_reason: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reason)


def load_shogou_instruction() -> str:
    if not INSTRUCTION_FILE.exists():
        raise PromptGeneratorError("shogou_chan_prompt_designer_instruction.md が見つかりません。")
    return INSTRUCTION_FILE.read_text(encoding="utf-8")


def _strip_think_blocks(text: str) -> str:
    # Only removes matched <think>...</think> pairs. If a </think> is never
    # found, the regex simply does not match, so the original text is
    # returned unchanged instead of being truncated at the opening tag.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_shogou_prompt_response(raw_response: str) -> ShogouPromptResult:
    cleaned = _strip_think_blocks(raw_response)
    cleaned = _strip_code_fence(cleaned)
    payload = _extract_json_object(cleaned)
    if payload is None:
        logger.warning("Shogou prompt JSON extraction failed.")
        return ShogouPromptResult(blocked_reason="LM Studioの返答からJSONを取り出せませんでした。")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        logger.warning("Shogou prompt response missing keys: %s", missing)
        return ShogouPromptResult(blocked_reason=f"必須キーが不足しています: {', '.join(missing)}")

    positive_raw = payload["positive"]
    negative_raw = payload["negative"]
    summary_raw = payload["summary"]
    assumptions_raw = payload["assumptions"]

    if not isinstance(positive_raw, str):
        return ShogouPromptResult(blocked_reason="positive が文字列ではありませんでした。")
    if not isinstance(negative_raw, str):
        return ShogouPromptResult(blocked_reason="negative が文字列ではありませんでした。")
    if not isinstance(summary_raw, str):
        return ShogouPromptResult(blocked_reason="summary が文字列ではありませんでした。")
    if not isinstance(assumptions_raw, list):
        return ShogouPromptResult(blocked_reason="assumptions が配列ではありませんでした。")

    positive = positive_raw.strip()
    negative = negative_raw.strip()
    if not positive:
        return ShogouPromptResult(blocked_reason="positive が空でした。")
    if not negative:
        return ShogouPromptResult(blocked_reason="negative が空でした。")

    assumptions = tuple(str(item).strip() for item in assumptions_raw if str(item).strip())

    logger.info(
        "Shogou prompt response parsed: positive_length=%s negative_length=%s summary_length=%s assumptions_count=%s",
        len(positive),
        len(negative),
        len(summary_raw.strip()),
        len(assumptions),
    )
    return ShogouPromptResult(positive=positive, negative=negative, summary=summary_raw.strip(), assumptions=assumptions)


def generate_shogou_prompt_result(
    user_input: str,
    user_id: int | None = None,
    reference_id: int | None = None,
) -> ShogouPromptResult:
    text = user_input.strip()
    if not text:
        raise PromptGeneratorError("内容を入力してください。例: `/generate_ai request:雨の日の小さな喫茶店`")
    if len(text) > 1000:
        raise PromptGeneratorError("少し長すぎます。1000文字以内にしてください。")

    logger.info("Shogou system prompt generation start: user_id=%s input_length=%s", user_id, len(text))
    system_prompt = append_memory_to_system_prompt(load_shogou_instruction(), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    logger.info("Shogou system prompt generation complete: user_id=%s length=%s", user_id, len(system_prompt))

    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Shogou postprocess start: raw_content_length=%s", len(raw_content))
    result = parse_shogou_prompt_response(raw_content)
    logger.info(
        "Shogou postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result
