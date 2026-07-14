import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from services.prompt_generator import (
    DEFAULT_TIMEOUT_SECONDS,
    PromptGeneratorError,
    PromptGeneratorTimeout,
    _get_lmstudio_api_url,
    _safe_url_label,
)


logger = logging.getLogger("secretary-bot.image_analyzer")
LOG_FILE = Path("data/logs/image_analyze.log")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
TIMEOUT_MESSAGE = "LM Studioから応答がありませんでした。"

IMAGE_ANALYSIS_SYSTEM_PROMPT = """
あなたはStable Diffusion向けの参考画像解析アシスタントです。
画像から明確に確認できる要素だけを、Stable Diffusionで使える短い英語のDanbooru風タグにしてください。
不明な要素を推測で補わず、該当要素がなければ空配列にしてください。人物の年齢は断定せず、必要なら 1girl または woman 程度にしてください。
説明、Markdown、コードフェンスを付けず、必ず次のJSONオブジェクトだけを返してください。
{"person":[],"hair":[],"eyes":[],"expression":[],"outfit":[],"background":[],"composition":[],"lighting_mood":[],"sd_tags":[]}
すべての値は英語タグの配列にしてください。
""".strip()

TAG_KEYS = ("person", "hair", "eyes", "expression", "outfit", "background", "composition", "lighting_mood", "sd_tags")
TAG_REPLACEMENTS = {"golden blonde hair": "blonde hair", "bob hairstyle": "bob cut", "portrait composition": "portrait", "front facing": "looking at viewer"}
DISPLAY_LABELS = {"person": "人物", "hair": "髪", "eyes": "目", "expression": "表情", "outfit": "服装", "background": "背景", "composition": "構図", "lighting_mood": "光・雰囲気"}


@dataclass
class ImageAnalysisResult:
    analysis_text: str
    tags: dict[str, list[str]]
    structured: bool = True


def _normalize_tag(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    tag = re.sub(r"\s+", " ", value.strip().lower()).strip(" ,.;")
    tag = TAG_REPLACEMENTS.get(tag, tag)
    if len(tag) > 60 or "\n" in tag or tag.count(" ") > 7:
        return ""
    return tag


def _normalize_tags(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        tag = _normalize_tag(value)
        if tag and tag not in result:
            result.append(tag)
    return result


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_image_analysis_response(content: str) -> ImageAnalysisResult:
    value = _extract_json_object(content)
    if value is None:
        logger.warning("Structured image analysis JSON parse failed; using legacy text fallback.")
        return ImageAnalysisResult(content.strip(), {key: [] for key in TAG_KEYS}, structured=False)
    tags = {key: _normalize_tags(value.get(key)) for key in TAG_KEYS}
    lines = [f"{DISPLAY_LABELS[key]}: {', '.join(tags[key])}" for key in DISPLAY_LABELS if tags[key]]
    summary = "画像から確認できた要素:\n" + ("\n".join(lines) if lines else "明確な要素を抽出できませんでした。")
    return ImageAnalysisResult(summary, tags)


def _ensure_file_logging() -> None:
    if any(getattr(handler, "_image_analyze_file_handler", False) for handler in logger.handlers):
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler._image_analyze_file_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


_ensure_file_logging()


def _truncate(value: str, limit: int = 2000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...(truncated)"


def _get_image_analysis_model() -> str:
    model = os.getenv("LMSTUDIO_VISION_MODEL", "").strip()
    fallback = os.getenv("LMSTUDIO_MODEL", "").strip() or os.getenv("LM_STUDIO_MODEL", "").strip()
    selected = model or fallback
    if not selected:
        raise PromptGeneratorError("Visionモデル未設定: .env に LMSTUDIO_VISION_MODEL を設定してね。")
    return selected


def _http_error_message(status_code: int, response_text: str) -> str:
    body = _truncate(response_text, 700) or "(empty response)"
    if status_code == 400:
        return f"HTTP 400: LM Studioがリクエストを受け取れませんでした。\n{body}"
    if status_code == 404:
        return "HTTP 404: LM StudioのAPIエンドポイントが見つかりません。LMSTUDIO_API_URL が /v1 付きか確認してね。"
    return f"HTTP {status_code}: LM Studioからエラーが返りました。\n{body}"


def _native_chat_url(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1/chat"):
        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    if path.endswith("/api/v1"):
        path = path + "/chat"
        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    path = path.rstrip("/") + "/api/v1/chat"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _normalize_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def detect_image_content_type(image_bytes: bytes, fallback_content_type: str | None = None) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    normalized = _normalize_content_type(fallback_content_type)
    if normalized in SUPPORTED_CONTENT_TYPES:
        return normalized
    return None


def validate_image_bytes(image_bytes: bytes, content_type: str | None) -> None:
    if not image_bytes:
        raise PromptGeneratorError("画像データを取得できませんでした。もう一度添付して試してね。")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise PromptGeneratorError("画像が少し大きいみたい。8MB以内の画像で試してね。")
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise PromptGeneratorError("JPEG / PNG / WebP の画像を添付してね。")


def validate_image_attachment(content_type: str | None, size: int) -> None:
    normalized = _normalize_content_type(content_type)
    if normalized not in SUPPORTED_CONTENT_TYPES:
        raise PromptGeneratorError("JPEG / PNG / WebP の画像を添付してね。")
    if size > MAX_IMAGE_BYTES:
        raise PromptGeneratorError("画像が少し大きいみたい。8MB以内の画像で試してね。")


def _data_url(image_bytes: bytes, content_type: str) -> str:
    content_type = _normalize_content_type(content_type) or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _content_parts_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("value")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def _extract_native_response_text(response_json: Any) -> str:
    if isinstance(response_json, str):
        return response_json.strip()
    if not isinstance(response_json, dict):
        return ""

    output = response_json.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            text = _content_parts_to_text(item.get("content") or item.get("text"))
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    message = response_json.get("message")
    if isinstance(message, dict):
        text = _content_parts_to_text(message.get("content"))
        if text:
            return text

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = _content_parts_to_text(message.get("content"))
                if text:
                    return text
            text = _content_parts_to_text(first.get("content") or first.get("text"))
            if text:
                return text

    return ""


def analyze_image_bytes(image_bytes: bytes, content_type: str, user_note: str = "") -> ImageAnalysisResult:
    detected_content_type = detect_image_content_type(image_bytes, content_type)
    validate_image_bytes(image_bytes, detected_content_type)
    api_url = _get_lmstudio_api_url()
    if not api_url:
        raise PromptGeneratorError("LMSTUDIO_API_URL が未設定です。.env にLM StudioのURLを設定してね。")

    model = _get_image_analysis_model()
    request_url = _native_chat_url(api_url)
    user_text = (
        "この参考画像を、指定されたJSON形式のStable Diffusion向けタグへ分解してください。"
    )
    if user_note.strip():
        user_text += f"\n補足: {user_note.strip()}"

    image_data_url = _data_url(image_bytes, detected_content_type or "image/png")
    payload: dict[str, Any] = {
        "model": model,
        "system_prompt": IMAGE_ANALYSIS_SYSTEM_PROMPT,
        "input": [
            {"type": "text", "content": user_text},
            {"type": "image", "data_url": image_data_url},
        ],
        "stream": False,
        "store": False,
    }

    logger.info(
        "LM Studio native image analysis request start: api_url=%s vision_model=%s post_url=%s image_bytes=%s content_type=%s data_url_prefix=%s data_url_length=%s timeout=%s",
        _safe_url_label(api_url),
        model,
        request_url,
        len(image_bytes),
        detected_content_type,
        image_data_url[:32],
        len(image_data_url),
        DEFAULT_TIMEOUT_SECONDS,
    )
    started_at = time.monotonic()
    try:
        response = requests.post(request_url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started_at
        logger.info(
            "LM Studio native image analysis response received before raise: status=%s elapsed=%.2fs response_text=%s",
            response.status_code,
            elapsed,
            _truncate(response.text, 4000),
        )
        response.raise_for_status()
        response_json = response.json()
    except requests.Timeout as exc:
        logger.exception("LM Studio native image analysis timed out.")
        raise PromptGeneratorTimeout(TIMEOUT_MESSAGE) from exc
    except requests.ConnectionError as exc:
        logger.exception("LM Studio native image analysis connection failed: post_url=%s vision_model=%s", request_url, model)
        raise PromptGeneratorError("接続失敗: LM Studio APIに接続できませんでした。起動状態とURLを確認してね。") from exc
    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else 0
        response_text = response.text if response is not None else ""
        logger.exception(
            "LM Studio native image analysis HTTP error: status=%s response_text=%s",
            status_code,
            _truncate(response_text, 4000),
        )
        raise PromptGeneratorError(_http_error_message(status_code, response_text)) from exc
    except requests.RequestException as exc:
        logger.exception("LM Studio native image analysis request failed.")
        raise PromptGeneratorError("接続失敗: LM Studioへのリクエストに失敗しました。URLと起動状態を確認してね。") from exc
    except ValueError as exc:
        logger.exception("LM Studio native image analysis JSON parse failed.")
        raise PromptGeneratorError("レスポンス解析失敗: LM Studioの返答JSONを読み取れませんでした。") from exc

    content = _extract_native_response_text(response_json)
    if not content:
        logger.error("LM Studio native image analysis content empty or unknown format: %r", response_json)
        raise PromptGeneratorError("レスポンス解析失敗: LM Studio native APIの返答本文を取り出せませんでした。")
    return parse_image_analysis_response(content)


def build_image_memory_text(analysis: ImageAnalysisResult | str, name: str = "") -> str:
    if isinstance(analysis, ImageAnalysisResult):
        all_tags = analysis.tags.get("sd_tags") or [tag for key in TAG_KEYS[:-1] for tag in analysis.tags[key]]
        text = ", ".join(all_tags) or analysis.analysis_text
    else:
        text = analysis.strip()
    label = f"画像参考メモ「{name}」" if name.strip() else "画像参考メモ"
    prefix = f"{label}: "
    available = max(500 - len(prefix), 0)
    if len(text) > available:
        text = text[: max(available - 3, 0)].rstrip() + "..."
    return prefix + text
