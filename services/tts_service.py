from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp


logger = logging.getLogger("secretary-bot.tts")

DEFAULT_VOICEVOX_API_URL = "http://127.0.0.1:50021"
DEFAULT_VOICEVOX_SPEAKER_ID = 1
DEFAULT_VOICEVOX_TIMEOUT_SECONDS = 30.0
DEFAULT_VC_SAY_MAX_LENGTH = 200
ABSOLUTE_VC_SAY_MAX_LENGTH = 200
VOICE_TEMP_DIR = Path("data") / "voice_temp"
LOCAL_VOICEVOX_HOSTS = {"127.0.0.1", "localhost", "::1"}


class TTSServiceError(RuntimeError):
    """Base error for local text-to-speech failures."""


class VoicevoxConfigError(TTSServiceError):
    """Raised when the VOICEVOX configuration is unsafe or invalid."""


class VoicevoxUnavailableError(TTSServiceError):
    """Raised when the local VOICEVOX API cannot be reached."""


class VoicevoxResponseError(TTSServiceError):
    """Raised when VOICEVOX returns an invalid or failed response."""


@dataclass(frozen=True)
class VoicevoxConfig:
    api_url: str
    speaker_id: int
    timeout_seconds: float


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r. Using default %s.", name, raw_value, default)
        value = default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r. Using default %s.", name, raw_value, default)
        value = default
    return max(minimum, min(value, maximum))


def get_vc_say_max_length() -> int:
    return _env_int(
        "VC_SAY_MAX_LENGTH",
        DEFAULT_VC_SAY_MAX_LENGTH,
        1,
        ABSOLUTE_VC_SAY_MAX_LENGTH,
    )


def validate_voicevox_api_url(raw_url: str) -> str:
    value = raw_url.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise VoicevoxConfigError("VOICEVOX_API_URLの形式が正しくありません。") from exc

    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "http" or hostname not in LOCAL_VOICEVOX_HOSTS:
        raise VoicevoxConfigError(
            "VOICEVOX_API_URLはlocalhost、127.0.0.1、または::1のHTTP URLだけ使用できます。"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VoicevoxConfigError("VOICEVOX_API_URLに認証情報・クエリ・フラグメントは指定できません。")
    if port is not None and not 1 <= port <= 65535:
        raise VoicevoxConfigError("VOICEVOX_API_URLのポート番号が正しくありません。")
    return value


def load_voicevox_config() -> VoicevoxConfig:
    api_url = validate_voicevox_api_url(os.getenv("VOICEVOX_API_URL", DEFAULT_VOICEVOX_API_URL))
    speaker_id = _env_int("VOICEVOX_SPEAKER_ID", DEFAULT_VOICEVOX_SPEAKER_ID, 0, 2_147_483_647)
    timeout_seconds = _env_float(
        "VOICEVOX_REQUEST_TIMEOUT_SECONDS",
        DEFAULT_VOICEVOX_TIMEOUT_SECONDS,
        1.0,
        120.0,
    )
    return VoicevoxConfig(api_url=api_url, speaker_id=speaker_id, timeout_seconds=timeout_seconds)


def remove_temp_audio(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary voice file: %s", path, exc_info=True)


async def _response_excerpt(response: aiohttp.ClientResponse) -> str:
    try:
        return (await response.text())[:500]
    except Exception:
        return "<response body unavailable>"


class TTSService:
    def __init__(self, config: VoicevoxConfig | None = None, temp_dir: Path = VOICE_TEMP_DIR) -> None:
        self._config = config
        self._temp_dir = temp_dir

    def _current_config(self) -> VoicevoxConfig:
        return self._config or load_voicevox_config()

    async def synthesize_to_wav(self, text: str) -> Path:
        config = self._current_config()
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        query_url = f"{config.api_url}/audio_query"
        synthesis_url = f"{config.api_url}/synthesis"

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    query_url,
                    params={"text": text, "speaker": config.speaker_id},
                ) as response:
                    logger.info("VOICEVOX audio_query status=%s", response.status)
                    if response.status < 200 or response.status >= 300:
                        detail = await _response_excerpt(response)
                        logger.error("VOICEVOX audio_query failed status=%s body=%s", response.status, detail)
                        raise VoicevoxResponseError(
                            f"VOICEVOXの音声クエリ作成に失敗しました（HTTP {response.status}）。"
                        )
                    try:
                        audio_query = await response.json(content_type=None)
                    except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as exc:
                        logger.exception("VOICEVOX audio_query returned invalid JSON.")
                        raise VoicevoxResponseError("VOICEVOXの音声クエリJSONを解析できませんでした。") from exc

                if not isinstance(audio_query, dict):
                    raise VoicevoxResponseError("VOICEVOXの音声クエリ形式が正しくありません。")

                async with session.post(
                    synthesis_url,
                    params={"speaker": config.speaker_id},
                    json=audio_query,
                ) as response:
                    logger.info("VOICEVOX synthesis status=%s", response.status)
                    if response.status < 200 or response.status >= 300:
                        detail = await _response_excerpt(response)
                        logger.error("VOICEVOX synthesis failed status=%s body=%s", response.status, detail)
                        raise VoicevoxResponseError(
                            f"VOICEVOXの音声合成に失敗しました（HTTP {response.status}）。"
                        )
                    wav_data = await response.read()
        except asyncio.TimeoutError as exc:
            logger.warning("VOICEVOX request timed out after %.1f seconds.", config.timeout_seconds)
            raise VoicevoxUnavailableError("VOICEVOXへの接続がタイムアウトしました。起動状態を確認してね。") from exc
        except aiohttp.ClientConnectionError as exc:
            logger.warning("Could not connect to local VOICEVOX API: %s", exc)
            raise VoicevoxUnavailableError("VOICEVOXへ接続できません。VOICEVOXを起動してから試してね。") from exc
        except aiohttp.ClientError as exc:
            logger.exception("VOICEVOX HTTP request failed.")
            raise VoicevoxUnavailableError("VOICEVOXとの通信に失敗しました。起動状態を確認してね。") from exc

        if not wav_data:
            raise VoicevoxResponseError("VOICEVOXから空の音声データが返されました。")

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, raw_path = tempfile.mkstemp(prefix="voice_", suffix=".wav", dir=self._temp_dir)
        os.close(file_descriptor)
        path = Path(raw_path)
        try:
            await asyncio.to_thread(path.write_bytes, wav_data)
        except Exception:
            remove_temp_audio(path)
            raise

        logger.info("Created temporary VOICEVOX wav: path=%s bytes=%s", path, len(wav_data))
        return path
