import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from sd_prompt_template_generator import (
    PromptTemplateResult,
    build_discord_prompt_message,
    format_llm_prompt_response,
)
from services.memory import append_memory_to_system_prompt
from services.image_memory import build_image_memory_prompt_context


logger = logging.getLogger("secretary-bot.prompt_generator")

INSTRUCTION_FILE = Path("fixed_template_llm_instruction.md")
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MODEL = "local-model"
DEFAULT_PROMPT_DESIGNER_MAX_TOKENS = 512
TIMEOUT_MESSAGE = "LM Studioから応答がありませんでした"


class PromptGeneratorError(Exception):
    pass


class PromptGeneratorTimeout(PromptGeneratorError):
    pass


class PromptDesignerNativeUnavailable(PromptGeneratorError):
    """The LM Studio native Prompt Designer endpoint cannot be used."""


def _get_lmstudio_api_url() -> str:
    return (
        os.getenv("LMSTUDIO_API_URL", "").strip()
        or os.getenv("LM_STUDIO_API_URL", "").strip()
        or os.getenv("LOCAL_LLM_API_URL", "").strip()
    )


def _get_lmstudio_fallback_model() -> str:
    return os.getenv("LMSTUDIO_MODEL", "").strip() or os.getenv("LM_STUDIO_MODEL", "").strip() or DEFAULT_MODEL


def _get_lmstudio_text_model() -> str:
    return os.getenv("LMSTUDIO_TEXT_MODEL", "").strip() or _get_lmstudio_fallback_model()


def _get_lmstudio_vision_model() -> str:
    return os.getenv("LMSTUDIO_VISION_MODEL", "").strip() or _get_lmstudio_fallback_model()


def _get_lmstudio_model() -> str:
    return _get_lmstudio_text_model()


def _safe_url_label(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "configured-url"
    return f"{parsed.scheme}://{parsed.netloc}"


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def load_system_prompt() -> str:
    if not INSTRUCTION_FILE.exists():
        raise PromptGeneratorError("fixed_template_llm_instruction.md が見つかりません。")
    return INSTRUCTION_FILE.read_text(encoding="utf-8")


def _post_chat_completion(
    system_prompt: str,
    user_input: str,
    temperature: float = 0.35,
    *,
    max_tokens: int = 1200,
) -> str:
    api_url = _get_lmstudio_api_url()
    if not api_url:
        raise PromptGeneratorError("LMSTUDIO_API_URL が未設定です。`.env` に LM Studio のURLを設定してください。")

    model = _get_lmstudio_text_model()
    request_url = _chat_completions_url(api_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info(
        "LM Studio request start: url=%s model=%s system_prompt_length=%s user_input_length=%s timeout=%s",
        _safe_url_label(api_url),
        model,
        len(system_prompt),
        len(user_input),
        DEFAULT_TIMEOUT_SECONDS,
    )
    started_at = time.monotonic()
    try:
        response = requests.post(request_url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started_at
        logger.info("LM Studio response received: status=%s elapsed=%.2fs", response.status_code, elapsed)
        response.raise_for_status()
        response_json = response.json()
    except requests.Timeout as exc:
        logger.exception("LM Studio request timed out.")
        raise PromptGeneratorTimeout(TIMEOUT_MESSAGE) from exc
    except requests.RequestException as exc:
        logger.exception("LM Studio request failed.")
        raise PromptGeneratorError("LM Studioに接続できませんでした。起動状態とURLを確認してください。") from exc
    except ValueError as exc:
        logger.exception("LM Studio response JSON parse failed.")
        raise PromptGeneratorError("LM Studioの返答JSONを読み取れませんでした。") from exc

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        logger.error("LM Studio response had no choices: %r", response_json)
        raise PromptGeneratorError("LM Studioの返答に choices がありませんでした。")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        logger.error("LM Studio response message format invalid: %r", response_json)
        raise PromptGeneratorError("LM Studioの返答形式を読み取れませんでした。")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        logger.error("LM Studio response content empty: %r", response_json)
        raise PromptGeneratorError("LM Studioの返答が空でした。")
    return content


def _native_chat_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[:-3].rstrip("/")
    return f"{stripped}/api/v1/chat"


def _extract_native_prompt_text(response_json: object) -> str:
    if not isinstance(response_json, dict):
        raise PromptDesignerNativeUnavailable("native response is not an object")
    output = response_json.get("output")
    if not isinstance(output, list):
        raise PromptDesignerNativeUnavailable("native response has no output array")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
    raise PromptDesignerNativeUnavailable("native response has no assistant message")


def _post_native_prompt_designer_completion(
    system_prompt: str,
    user_input: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    api_url = _get_lmstudio_api_url()
    if not api_url:
        raise PromptDesignerNativeUnavailable("LM Studio URL is not configured")

    request_url = _native_chat_url(api_url)
    payload: dict[str, Any] = {
        "model": _get_lmstudio_text_model(),
        "system_prompt": system_prompt,
        "input": user_input,
        "reasoning": "off",
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": False,
    }
    logger.info(
        "Prompt Designer native request start: url=%s model=%s reasoning=off "
        "system_prompt_length=%s user_input_length=%s max_output_tokens=%s timeout=%s",
        _safe_url_label(api_url),
        payload["model"],
        len(system_prompt),
        len(user_input),
        max_tokens,
        DEFAULT_TIMEOUT_SECONDS,
    )
    started_at = time.monotonic()
    try:
        response = requests.post(request_url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started_at
        response.raise_for_status()
        response_json = response.json()
        content = _extract_native_prompt_text(response_json)
        logger.info("Prompt Designer native response received: status=%s elapsed=%.2fs", response.status_code, elapsed)
        return content
    except (requests.Timeout, requests.RequestException, ValueError, TypeError, PromptDesignerNativeUnavailable) as exc:
        logger.warning(
            "Prompt Designer native request unavailable; fallback is allowed: error_type=%s",
            type(exc).__name__,
        )
        raise PromptDesignerNativeUnavailable from exc


def post_prompt_designer_completion(
    system_prompt: str,
    user_input: str,
    temperature: float = 0.35,
    *,
    response_format: object | None = None,
    max_tokens: int = DEFAULT_PROMPT_DESIGNER_MAX_TOKENS,
) -> str:
    """Call LM Studio's native reasoning-controlled endpoint for Prompt Designer.

    ``response_format`` is accepted for compatibility with the existing
    resolver callback contract.  The native endpoint returns assistant text;
    the existing Prompt Designer JSON parser remains the source of truth.
    A valid native assistant message is never resent merely because its JSON
    content later fails Prompt Designer validation.
    """

    del response_format
    try:
        return _post_native_prompt_designer_completion(
            system_prompt,
            user_input,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except PromptDesignerNativeUnavailable as exc:
        logger.warning(
            "Prompt Designer falling back to OpenAI-compatible LM Studio endpoint: error_type=%s",
            type(exc).__name__,
        )
        return _post_chat_completion(
            system_prompt,
            user_input,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def generate_sd_prompt_result(user_input: str, user_id: int | None = None, reference_id: int | None = None) -> PromptTemplateResult:
    text = user_input.strip()
    if not text:
        raise PromptGeneratorError("内容を入力してください。例: `/prompt text:雨の日の小さな喫茶店`")
    if len(text) > 1000:
        raise PromptGeneratorError("少し長すぎます。1000文字以内にしてください。")

    logger.info("System prompt generation start: mode=prompt user_id=%s input_length=%s", user_id, len(text))
    system_prompt = append_memory_to_system_prompt(load_system_prompt(), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    logger.info("System prompt generation complete: mode=prompt user_id=%s length=%s", user_id, len(system_prompt))
    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Prompt postprocess start: raw_content_length=%s", len(raw_content))
    result = format_llm_prompt_response(raw_content, source_text=text)
    logger.info(
        "Prompt postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive_prompt),
        len(result.negative_prompt),
        result.blocked,
    )
    return result


def generate_sd_prompt_message(user_input: str, user_id: int | None = None) -> str:
    return build_discord_prompt_message(generate_sd_prompt_result(user_input, user_id))


def generate_chat_message(user_input: str, user_id: int | None = None) -> str:
    text = user_input.strip()
    if not text:
        raise PromptGeneratorError("話したい内容を入力してください。")
    if len(text) > 1500:
        raise PromptGeneratorError("少し長すぎます。1500文字以内にしてください。")

    base_system_prompt = (
        "You are 初号ちゃん, a concise and gentle Japanese assistant. "
        "Answer in Japanese. If user memory is provided, actively use it when relevant."
    )
    logger.info("System prompt generation start: mode=chat user_id=%s input_length=%s", user_id, len(text))
    system_prompt = append_memory_to_system_prompt(base_system_prompt, user_id, text, mode="chat")
    logger.info("System prompt generation complete: mode=chat user_id=%s length=%s", user_id, len(system_prompt))
    return _post_chat_completion(system_prompt, text, temperature=0.6)
