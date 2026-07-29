import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from services.character_tag_dictionary import CharacterTagResolution, resolve_all_character_tags
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


def _build_known_character_context(resolutions: Sequence[CharacterTagResolution]) -> str:
    lines = ["Known character:"]
    for resolution in resolutions:
        lines.append(f"- name: {resolution.display_name} ({resolution.series_name})")
        lines.append(f"- character tag: {resolution.character_tag}")
        lines.append(f"- series tag: {resolution.series_tag}")
    lines.append("- these tags must appear near the beginning of positive, right after the count tag (1girl/1boy/1other)")
    lines.append("- do not translate, split, abbreviate, replace, or generalize these tags")
    lines.append("- the name/series text above is identification context only; never add it as a separate tag itself")
    if len(resolutions) > 1:
        lines.append(
            "- multiple known characters were detected in this request; keep them as distinct, separate "
            "characters and do not merge them into one"
        )
    return "\n".join(lines)


def _append_known_character_context(system_prompt: str, text: str) -> str:
    resolutions = resolve_all_character_tags(text)
    if not resolutions:
        return system_prompt
    for resolution in resolutions:
        logger.debug(
            "Character tag resolved: input_character=%s character_tag=%s series_tag=%s match_type=%s",
            resolution.matched_character_alias,
            resolution.character_tag,
            resolution.series_tag,
            resolution.match_type,
        )
    return system_prompt + "\n\n" + _build_known_character_context(resolutions)


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
    system_prompt = _append_known_character_context(system_prompt, text)
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


def _build_composition_variation_instruction(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
) -> str:
    return (
        "\n\n---\n"
        "これは既存の生成結果をもとにした「構図を変えて再生成」リクエストです。"
        "以下の元の生成条件をもとに、キャラクター・衣装・雰囲気・重要な属性、"
        "および元の色彩・明るさ・光の方向性（色温度や逆光の強さなど）はそのまま維持しつつ、"
        "構図・カメラ角度・ポーズ・画面内配置だけを変更した新しいpositive/negative/summary/assumptionsを作成してください。\n\n"
        f"元のリクエスト: {parent_request_text}\n"
        f"元のPositive: {parent_positive}\n"
        f"元のNegative: {parent_negative}\n"
        f"元のSummary: {parent_summary}\n"
        f"元のAssumptions: {json.dumps(list(parent_assumptions), ensure_ascii=False)}\n"
    )


def _build_custom_instruction_variation_instruction(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    custom_instruction: str,
) -> str:
    return (
        "\n\n---\n"
        "これは既存の生成結果をもとにした「追加指示付き再生成」リクエストです。"
        "元の生成方針をできるだけ維持し、追加指示に関係する部分のみを変更してください。"
        "追加指示と矛盾しない限り、元の色彩・ライティング・構図・雰囲気・衣装などは維持してください。\n\n"
        f"元のリクエスト: {parent_request_text}\n"
        f"元のPositive: {parent_positive}\n"
        f"元のNegative: {parent_negative}\n"
        f"元のSummary: {parent_summary}\n"
        f"元のAssumptions: {json.dumps(list(parent_assumptions), ensure_ascii=False)}\n\n"
        f"今回の追加指示: {custom_instruction}\n"
    )


def generate_shogou_custom_instruction_variation(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    custom_instruction: str,
    user_id: int | None = None,
    reference_id: int | None = None,
) -> ShogouPromptResult:
    """"指示を追加して再生成": reuses the same shogou instruction/system
    prompt as a normal /generate_ai call, but appends the parent request's
    positive/negative/summary/assumptions plus the user's own free-text
    instruction (from the regenerate modal), with an explicit instruction to
    keep the original generation policy and only change what the new
    instruction targets.
    """
    instruction_text = custom_instruction.strip()
    if not instruction_text:
        raise PromptGeneratorError("追加指示を入力してください。")
    if len(instruction_text) > 1000:
        raise PromptGeneratorError("追加指示は1000文字以内にしてください。")

    text = parent_request_text.strip() or "(元のリクエスト文なし)"
    logger.info("Shogou custom instruction variation generation start: user_id=%s", user_id)
    system_prompt = append_memory_to_system_prompt(load_shogou_instruction(), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    system_prompt = _append_known_character_context(system_prompt, text)
    system_prompt += _build_custom_instruction_variation_instruction(
        parent_request_text, parent_positive, parent_negative, parent_summary, parent_assumptions, instruction_text
    )
    logger.info(
        "Shogou custom instruction variation system prompt complete: user_id=%s length=%s", user_id, len(system_prompt)
    )

    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Shogou custom instruction variation postprocess start: raw_content_length=%s", len(raw_content))
    result = parse_shogou_prompt_response(raw_content)
    logger.info(
        "Shogou custom instruction variation postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result


def generate_shogou_composition_variation(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    user_id: int | None = None,
    reference_id: int | None = None,
) -> ShogouPromptResult:
    """Phase 1C "change_composition" regeneration: reuses the same shogou
    instruction/system prompt as a normal /generate_ai call, but appends the
    parent request's positive/negative/summary/assumptions plus an explicit
    instruction to keep character/clothing/mood and only vary composition.
    """
    text = parent_request_text.strip() or "(元のリクエスト文なし)"
    logger.info("Shogou composition variation generation start: user_id=%s", user_id)
    system_prompt = append_memory_to_system_prompt(load_shogou_instruction(), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    system_prompt = _append_known_character_context(system_prompt, text)
    system_prompt += _build_composition_variation_instruction(
        parent_request_text, parent_positive, parent_negative, parent_summary, parent_assumptions
    )
    logger.info("Shogou composition variation system prompt complete: user_id=%s length=%s", user_id, len(system_prompt))

    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Shogou composition variation postprocess start: raw_content_length=%s", len(raw_content))
    result = parse_shogou_prompt_response(raw_content)
    logger.info(
        "Shogou composition variation postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result
