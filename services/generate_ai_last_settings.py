"""Persistent per-user, per-model preferences for /generate_ai.

Only successful ordinary /generate_ai deliveries write this file.  Prompt
text, characters, seeds, and reference image bytes deliberately never enter
this store.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.image_composition import (
    DEFAULT_COMPOSITION_PRESET,
    DEFAULT_SIZE_KEY,
    get_image_size_spec,
    validate_composition_preset,
)


DATA_FILE = Path("data") / "generate_ai_last_settings.json"
SCHEMA_VERSION = 1
ALLOWED_RATINGS = {"auto", "safe", "sensitive", "nsfw", "explicit"}
ALLOWED_LORA_TYPES = {"character", "style", "additional"}
ALLOWED_REFERENCE_MODES = {"none", "composition", "pose"}
ALLOWED_INPUT_MODES = {"ja_ai", "en_direct", "prompt_import"}
DEFAULT_INPUT_MODE = "ja_ai"


class GenerateAiLastSettingsError(RuntimeError):
    """Raised when the settings file cannot be read, validated, or saved."""


@dataclass(frozen=True)
class GenerateAiLastSettings:
    style_key: str | None
    lora_strength: float | None
    rating: str
    reference_strength: float
    selection_ai: bool
    candidate_count: int
    size_key: str = DEFAULT_SIZE_KEY
    composition_preset: str = DEFAULT_COMPOSITION_PRESET
    loras: tuple[dict[str, Any], ...] = ()
    reference_mode: str = "none"
    input_mode: str = DEFAULT_INPUT_MODE


_settings_lock = threading.RLock()


def _clean_optional_style(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GenerateAiLastSettingsError("style_key must be a string or null")
    return value.strip() or None


def _clean_optional_strength(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerateAiLastSettingsError(f"{field} must be a number or null")
    cleaned = float(value)
    if not math.isfinite(cleaned) or not 0.0 <= cleaned <= 1.0:
        raise GenerateAiLastSettingsError(f"{field} must be between 0.0 and 1.0")
    return cleaned


def _clean_loras(value: object) -> tuple[dict[str, Any], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise GenerateAiLastSettingsError("loras must be an array")
    if len(value) > 3:
        raise GenerateAiLastSettingsError("loras must contain at most three items")
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise GenerateAiLastSettingsError("each lora must be an object")
        key = item.get("key")
        lora_type = item.get("lora_type")
        if not isinstance(key, str) or not key.strip():
            raise GenerateAiLastSettingsError("each lora must contain a key")
        if not isinstance(lora_type, str) or lora_type.strip().casefold() not in ALLOWED_LORA_TYPES:
            raise GenerateAiLastSettingsError("each lora has an invalid lora_type")
        normalized_key = key.strip().casefold()
        if normalized_key in seen:
            raise GenerateAiLastSettingsError("loras must not contain duplicates")
        seen.add(normalized_key)
        strength = _clean_optional_strength(item.get("strength"), "lora strength")
        if strength is None:
            raise GenerateAiLastSettingsError("each lora must contain a strength")
        copied = dict(item)
        copied["key"] = normalized_key
        copied["lora_type"] = lora_type.strip().casefold()
        copied["strength"] = strength
        cleaned.append(copied)
    return tuple(cleaned)


def _settings_from_object(value: object) -> GenerateAiLastSettings:
    if not isinstance(value, dict):
        raise GenerateAiLastSettingsError("model settings must be a JSON object")
    rating = value.get("rating", "auto")
    if not isinstance(rating, str) or rating.strip().casefold() not in ALLOWED_RATINGS:
        raise GenerateAiLastSettingsError("rating is invalid")
    reference_strength = _clean_optional_strength(
        value.get("reference_strength", 0.5), "reference_strength"
    )
    if reference_strength is None:
        reference_strength = 0.5
    selection_ai = value.get("selection_ai", True)
    if not isinstance(selection_ai, bool):
        raise GenerateAiLastSettingsError("selection_ai must be a boolean")
    candidate_count = value.get("candidate_count", 5)
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise GenerateAiLastSettingsError("candidate_count must be an integer")
    if not 1 <= candidate_count <= 20:
        raise GenerateAiLastSettingsError("candidate_count must be between 1 and 20")
    try:
        size_key = get_image_size_spec(value.get("size_key", DEFAULT_SIZE_KEY)).key
        composition_preset = validate_composition_preset(
            value.get("composition_preset", DEFAULT_COMPOSITION_PRESET)
        )
        loras = _clean_loras(value.get("loras", []))
    except ValueError as exc:
        raise GenerateAiLastSettingsError(str(exc)) from exc
    reference_mode = value.get("reference_mode", "none")
    if not isinstance(reference_mode, str) or reference_mode.strip().casefold() not in ALLOWED_REFERENCE_MODES:
        raise GenerateAiLastSettingsError("reference_mode is invalid")
    input_mode = value.get("input_mode", DEFAULT_INPUT_MODE)
    if not isinstance(input_mode, str) or input_mode.strip().casefold() not in ALLOWED_INPUT_MODES:
        raise GenerateAiLastSettingsError("input_mode is invalid")
    return GenerateAiLastSettings(
        style_key=_clean_optional_style(value.get("style_key")),
        lora_strength=_clean_optional_strength(value.get("lora_strength"), "lora_strength"),
        rating=rating.strip().casefold(),
        reference_strength=reference_strength,
        selection_ai=selection_ai,
        candidate_count=candidate_count,
        size_key=size_key,
        composition_preset=composition_preset,
        loras=loras,
        reference_mode=reference_mode.strip().casefold(),
        input_mode=input_mode.strip().casefold(),
    )


def _empty_payload() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "users": {}}


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerateAiLastSettingsError(f"failed to load generate_ai settings from {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("users"), dict):
        raise GenerateAiLastSettingsError("generate_ai settings JSON must contain a users object")
    version = payload.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise GenerateAiLastSettingsError(f"unsupported generate_ai settings version: {version}")
    return payload


def _write_payload(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        raise GenerateAiLastSettingsError(f"failed to save generate_ai settings to {path}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def get_last_model_key(user_id: int, path: Path | None = None) -> str | None:
    source = path or DATA_FILE
    with _settings_lock:
        payload = _load_payload(source)
        user = payload["users"].get(str(user_id))
        if not isinstance(user, dict):
            return None
        model_key = user.get("last_model_key")
        return model_key.strip() if isinstance(model_key, str) and model_key.strip() else None


def get_last_settings(
    user_id: int,
    model_key: str,
    path: Path | None = None,
) -> GenerateAiLastSettings | None:
    source = path or DATA_FILE
    with _settings_lock:
        payload = _load_payload(source)
        user = payload["users"].get(str(user_id))
        models = user.get("models") if isinstance(user, dict) else None
        value = models.get(model_key) if isinstance(models, dict) else None
        return None if value is None else _settings_from_object(value)


def save_last_settings(
    user_id: int,
    model_key: str,
    settings: GenerateAiLastSettings,
    path: Path | None = None,
) -> None:
    destination = path or DATA_FILE
    cleaned_model_key = model_key.strip()
    if not cleaned_model_key:
        raise GenerateAiLastSettingsError("model_key must not be empty")
    validated = _settings_from_object(asdict(settings))
    with _settings_lock:
        payload = _load_payload(destination)
        users = payload["users"]
        user = users.setdefault(str(user_id), {"last_model_key": None, "models": {}})
        if not isinstance(user, dict):
            raise GenerateAiLastSettingsError("user settings must be a JSON object")
        models = user.setdefault("models", {})
        if not isinstance(models, dict):
            raise GenerateAiLastSettingsError("user models must be a JSON object")
        value = asdict(validated)
        value["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        models[cleaned_model_key] = value
        user["last_model_key"] = cleaned_model_key
        _write_payload(payload, destination)


def reset_last_settings(
    user_id: int,
    model_key: str | None = None,
    path: Path | None = None,
) -> int:
    destination = path or DATA_FILE
    with _settings_lock:
        payload = _load_payload(destination)
        users = payload["users"]
        user_key = str(user_id)
        user = users.get(user_key)
        if not isinstance(user, dict):
            return 0
        models = user.get("models")
        if not isinstance(models, dict):
            models = {}
        if model_key is None:
            removed = len(models)
            users.pop(user_key, None)
        else:
            removed = 1 if models.pop(model_key, None) is not None else 0
            if removed and user.get("last_model_key") == model_key:
                user["last_model_key"] = None
            if not models:
                users.pop(user_key, None)
        if removed:
            _write_payload(payload, destination)
        return removed


def get_last_batch_settings(user_id: int, path: Path | None = None) -> dict[str, Any] | None:
    """Return the user's last bulk-generation settings.

    Bulk settings live in the same user settings document as ``/generate_ai``
    settings.  LoRA values are stored as catalog references/snapshots and are
    resolved through ``sd_control`` when a job actually starts; this avoids a
    second LoRA catalog while still allowing a removed catalog entry to fail
    safely.
    """

    source = path or DATA_FILE
    with _settings_lock:
        payload = _load_payload(source)
        user = payload["users"].get(str(user_id))
        value = user.get("batch") if isinstance(user, dict) else None
        if not isinstance(value, dict):
            return None
        # Return a detached JSON-compatible copy so callers cannot mutate the
        # in-memory object used by a concurrent command.
        return json.loads(json.dumps(value, ensure_ascii=False))


def save_last_batch_settings(
    user_id: int,
    settings: dict[str, Any],
    path: Path | None = None,
) -> None:
    """Persist the last bulk-generation settings in the existing settings file."""

    if not isinstance(settings, dict):
        raise GenerateAiLastSettingsError("batch settings must be an object")
    try:
        copied = json.loads(json.dumps(settings, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise GenerateAiLastSettingsError("batch settings must be JSON serializable") from exc
    destination = path or DATA_FILE
    with _settings_lock:
        payload = _load_payload(destination)
        users = payload["users"]
        user = users.setdefault(str(user_id), {"last_model_key": None, "models": {}})
        if not isinstance(user, dict):
            raise GenerateAiLastSettingsError("user settings must be an object")
        copied["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        user["batch"] = copied
        _write_payload(payload, destination)
