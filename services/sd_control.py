import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin

import requests
from PIL import Image

from services.pc_control import get_shutdown_delay_seconds, is_windows, schedule_shutdown
from services.comfy_lora_resolver import resolve_comfy_lora_names
from services.lora_catalog import (
    COMFY_STYLES,
    ComfyStyle,
    build_comfy_styles,
    get_comfy_lora_choice_specs,
    get_comfy_style,
    get_comfy_style_choice_specs,
    model_family_compatible,
    refresh_comfy_styles,
)
from services.lora_training import (
    LORA_TYPE_ADDITIONAL,
    LORA_TYPE_CHARACTER,
    LORA_TYPE_LABELS,
    LORA_TYPE_STYLE,
)
from services.pose_reference import (
    DEFAULT_POSE_END_PERCENT,
    DEFAULT_POSE_FIT_MODE,
    DEFAULT_POSE_START_PERCENT,
    DEFAULT_POSE_STRENGTH,
    POSE_ANIMEPOSE_CONFIDENCE,
    POSE_ANIMEPOSE_MASK_GATE,
    POSE_ANIMEPOSE_MASK_TOLERANCE,
    POSE_ANIMEPOSE_MIN_COMPLETED_CHAINS,
    POSE_ANIMEPOSE_MIN_VALID_BODY_KEYPOINTS,
    POSE_BBOX_DETECTOR_NAME,
    POSE_CONTROLNET_NAME,
    POSE_PREPROCESSOR_NAME,
    POSE_REFERENCE_PROFILE_FULL_BODY,
    POSE_REFERENCE_PROFILE_UNKNOWN,
    POSE_REFERENCE_PROFILE_UPPER_BODY,
    POSE_SDPOSE_CHECKPOINT_NAME,
    POSE_SDPOSE_DRAW_THRESHOLD,
    POSE_SDPOSE_MIN_COMPLETED_CHAINS,
    POSE_SDPOSE_MIN_DRAWN_BODY_KEYPOINTS,
    REFERENCE_MODE_POSE,
    animepose_keypoint_summary,
    build_animepose_preflight_workflow,
    build_dwpose_preflight_workflow,
    build_sdpose_preflight_workflow,
    classify_pose_reference_profile,
    evaluate_pose_quality,
    extract_json_candidates,
    pose_fit_plan,
    sdpose_keypoint_summary,
)
from services.weather import JST
from services.runtime_paths import PROJECT_ROOT, project_path, resolve_project_path


logger = logging.getLogger("secretary-bot.sd_control")

DATA_DIR = project_path("data")
OUTPUT_DIR = DATA_DIR / "sd_outputs"
POSE_DEBUG_DIR = DATA_DIR / "pose_debug"
FAVORITES_DIR = DATA_DIR / "sd_favorites"
LATEST_FILE = DATA_DIR / "sd_latest.json"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, bad anatomy, extra fingers, missing fingers, "
    "bad hands, blurry, jpeg artifacts, watermark, text, logo"
)
DEFAULT_COMFY_WORKFLOW_FILE = project_path("workflows", "comfy_workflow.json")
LORA_COMFY_WORKFLOW_FILE = project_path("workflows", "comfy_workflow_lora.json")
REFERENCE_COMFY_WORKFLOW_FILE = project_path("workflows", "comfy_workflow_ipadapter.json")
DEFAULT_REFERENCE_STRENGTH = 0.5
REFERENCE_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
COMFY_CHECKPOINTS = {
    "janku": "JANKUTrainedChenkinNoobai_v777.safetensors",
    "janku_adult": "JANKUTrainedChenkinNoobai_v777.safetensors",
    "animagine": "animagineXL40Opt_clear.safetensors",
    "animagine_adult": "animagineXL40Opt_clear.safetensors",
}
COMFY_MODEL_FAMILIES = {
    "janku": "sdxl",
    "janku_adult": "sdxl",
    "animagine": "sdxl",
    "animagine_adult": "sdxl",
}
COMFY_POLL_INTERVAL_SECONDS = 2
COMFY_TIMEOUT_SECONDS = 600
STARTUP_STATUS_WAIT_SECONDS = 5
COMFY_PAYLOAD_LOG_LIMIT = 20000
COMFY_ERROR_TEXT_LIMIT = 800

_generate_lock = threading.Lock()
_shutdown_after_done = False


# Compatibility names for the pose helpers that were historically kept in
# sd_control.py. The implementation now lives in services.pose_reference so
# the generation backend remains the only consumer of the quality policy.
_pose_fit_plan = pose_fit_plan
_sdpose_keypoint_summary = sdpose_keypoint_summary
_animepose_keypoint_summary = animepose_keypoint_summary
_classify_pose_reference_profile = classify_pose_reference_profile


def _build_pose_preflight_workflow(image_name: str, options: object):
    return build_dwpose_preflight_workflow(
        image_name,
        detect_hand=bool(getattr(options, "pose_detect_hand", True)),
        detect_face=bool(getattr(options, "pose_detect_face", True)),
    )


def _build_sdpose_preflight_workflow(image_name: str, filename_prefix: str):
    return build_sdpose_preflight_workflow(image_name, filename_prefix)


def _build_animepose_preflight_workflow(image_name: str):
    return build_animepose_preflight_workflow(image_name)


def _sdpose_detection_is_valid(
    summary: dict[str, Any] | None,
    skeleton_stats: dict[str, int | float | bool] | None,
) -> bool:
    return bool(
        evaluate_pose_quality(summary, skeleton_stats, backend="sdpose").get(
            "pose_quality_pass", False
        )
    )


def _animepose_detection_is_valid(
    summary: dict[str, Any] | None,
    skeleton_stats: dict[str, int | float | bool] | None,
) -> bool:
    return bool(
        evaluate_pose_quality(summary, skeleton_stats, backend="animepose").get(
            "pose_quality_pass", False
        )
    )


@dataclass(frozen=True)
class LoraSelection:
    """One validated LoRA selection for the generation backend."""

    key: str
    lora_name: str
    lora_type: str
    display_name: str
    trigger: str
    strength: float
    model_family: str | None = None


@dataclass(frozen=True)
class SdGenerationOptions:
    prompt: str
    negative_prompt: str | None = None
    model_key: str | None = None
    model_family: str | None = None
    filename_prefix: str | None = None
    model_selection_explicit: bool = False
    preserve_prompt_text: bool = False
    count: int = 1
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "Euler a"
    seed: int = -1
    style_key: str | None = None
    lora_strength: float | None = None
    loras: tuple[LoraSelection, ...] = ()
    reference_image_bytes: bytes | None = None
    reference_image_content_type: str | None = None
    reference_strength: float | None = None
    reference_mode: str = "none"
    pose_reference_bytes: bytes | None = None
    pose_reference_content_type: str | None = None
    pose_reference_width: int | None = None
    pose_reference_height: int | None = None
    pose_strength: float = DEFAULT_POSE_STRENGTH
    pose_start_percent: float = DEFAULT_POSE_START_PERCENT
    pose_end_percent: float = DEFAULT_POSE_END_PERCENT
    pose_fit_mode: str = DEFAULT_POSE_FIT_MODE
    pose_detect_hand: bool = True
    pose_detect_face: bool = True
    pose_debug_enabled: bool = True
    # Optional metadata written beside generated images for the independent
    # Gallery consumer.  Generation remains fully functional when unset.
    gallery_output_root: Path | None = None
    gallery_metadata: dict[str, Any] | None = None
    # Only the fields named here are written back to the ComfyUI workflow.
    # Normal /generate_ai options keep the workflow's existing defaults.
    explicit_generation_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GeneratedImage:
    """One generated image plus the generation conditions actually used to
    produce it. All condition fields except path/batch_index are Optional -
    when a value cannot be safely read back from the workflow that was
    actually queued (or, for the webui backend, is simply not knowable),
    None is stored rather than a guessed or .env-derived value.
    """

    path: Path
    batch_index: int
    seed: int | None = None
    checkpoint_name: str | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    workflow_file: str | None = None
    workflow_hash: str | None = None
    pose_backend: str | None = None
    pose_reference_profile: str | None = None
    pose_profile_reason: str | None = None
    pose_reference_debug_path: Path | None = None


def _now_text() -> str:
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def _api_url(path: str) -> str:
    base_url = os.getenv("SD_API_URL", "http://127.0.0.1:8188").strip().rstrip("/") + "/"
    return urljoin(base_url, path.lstrip("/"))


def _backend() -> str:
    return os.getenv("SD_BACKEND", "comfy").strip().lower() or "comfy"


def is_comfy_backend() -> bool:
    return _backend() == "comfy"


def _max_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s: %s. Using default %s.", name, raw, default)
        return default


def _max_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s: %s. Using default %s.", name, raw, default)
        return default


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def is_generation_running() -> bool:
    return _generate_lock.locked()


def _load_latest_paths() -> list[Path]:
    if not LATEST_FILE.exists():
        return []

    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load latest SD image state.")
        return []

    if not isinstance(data, dict):
        return []

    raw_paths = data.get("paths", [])
    if not isinstance(raw_paths, list):
        return []

    paths = []
    for raw_path in raw_paths:
        path = resolve_project_path(str(raw_path), str(raw_path))
        if path.exists() and path.is_file():
            paths.append(path)
    return paths


def _save_latest_paths(paths: list[Path]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "paths": [str(path) for path in paths],
    }
    LATEST_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_latest_image_paths() -> list[Path]:
    return _load_latest_paths()


def get_negative_prompt(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    return os.getenv("SD_DEFAULT_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT).strip() or DEFAULT_NEGATIVE_PROMPT


def _resolve_generation_model_family(
    model_key: str | None,
    model_family: str | None,
) -> str | None:
    if isinstance(model_family, str) and model_family.strip():
        return model_family.strip().casefold()
    if isinstance(model_key, str):
        return COMFY_MODEL_FAMILIES.get(model_key.strip().casefold())
    # The repository's default workflow is Animagine XL / SDXL.  Keep the
    # default explicit so a known incompatible LoRA is rejected even when an
    # older caller does not populate model_family.
    return "sdxl"


def _validate_lora_strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("LoRA strengthは0.0～1.0の数値で指定してください。")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("LoRA strengthは0.0～1.0の範囲で指定してください。")
    return resolved


def _validate_catalog_lora_file(lora: ComfyStyle) -> None:
    # Managed catalog entries point back to the training output file.  Reject
    # a stale entry before queueing; built-in legacy entries intentionally have
    # no source_path because ComfyUI resolves them from its configured model
    # root at execution time.
    if lora.source_path is not None and not lora.source_path.is_file():
        raise ValueError(f"LoRAファイルが見つかりません: {lora.display_name}")


def resolve_style_selection(
    style_key: str | None,
    strength: float | None,
    *,
    model_family: str | None = None,
) -> tuple[str | None, ComfyStyle | None, float | None]:
    """Resolve the legacy style slot through the shared LoRA catalog."""

    normalized_key = style_key.strip().casefold() if isinstance(style_key, str) else None
    normalized_key = normalized_key or None
    if normalized_key is None:
        if strength is not None:
            raise ValueError("LoRA strengthを指定するにはstyleを選択してください。")
        return None, None, None

    style = get_comfy_style(normalized_key)
    if style is None:
        raise ValueError(f"未登録のstyleです: {normalized_key}")
    if style.load_error:
        raise ValueError(style.load_error)
    _validate_catalog_lora_file(style)
    if style.lora_type != LORA_TYPE_STYLE:
        raise ValueError(f"LoRA「{style.display_name}」はstyle枠では選択できません。")
    if not model_family_compatible(style.model_family, model_family):
        raise ValueError(
            f"LoRA「{style.display_name}」はモデル系統「{model_family}」に対応していません。"
        )

    resolved_strength = style.default_strength if strength is None else strength
    return normalized_key, style, _validate_lora_strength(resolved_strength)


def _resolve_one_lora_selection(
    key: str | None,
    strength: float | None,
    expected_type: str,
    *,
    model_family: str | None,
) -> LoraSelection | None:
    normalized_key = key.strip().casefold() if isinstance(key, str) else None
    normalized_key = normalized_key or None
    if normalized_key is None:
        if strength is not None:
            raise ValueError("LoRA強度を指定するには、そのLoRAを選択してください。")
        return None

    lora = get_comfy_style(normalized_key)
    if lora is None:
        raise ValueError(f"未登録のLoRAです: {normalized_key}")
    if lora.load_error:
        raise ValueError(lora.load_error)
    _validate_catalog_lora_file(lora)
    if lora.lora_type != expected_type:
        raise ValueError(f"LoRA「{lora.display_name}」は{expected_type}枠では選択できません。")
    if not model_family_compatible(lora.model_family, model_family):
        raise ValueError(
            f"LoRA「{lora.display_name}」はモデル系統「{model_family}」に対応していません。"
        )

    resolved_strength = lora.default_strength if strength is None else strength
    return LoraSelection(
        key=normalized_key,
        lora_name=lora.lora_name,
        lora_type=lora.lora_type,
        display_name=lora.display_name,
        trigger=lora.trigger,
        strength=_validate_lora_strength(resolved_strength),
        model_family=lora.model_family,
    )


def resolve_lora_selections(
    *,
    character_key: str | None = None,
    character_strength: float | None = None,
    style_key: str | None = None,
    style_strength: float | None = None,
    additional_key: str | None = None,
    additional_strength: float | None = None,
    model_family: str | None = None,
) -> tuple[LoraSelection, ...]:
    """Resolve the typed LoRA slots in deterministic order."""

    requested_keys = [
        key.strip().casefold()
        for key in (character_key, style_key, additional_key)
        if isinstance(key, str) and key.strip()
    ]
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError("同じLoRAを複数の枠へ選択することはできません。")

    selections = tuple(
        selection
        for selection in (
            _resolve_one_lora_selection(
                character_key,
                character_strength,
                LORA_TYPE_CHARACTER,
                model_family=model_family,
            ),
            _resolve_one_lora_selection(
                style_key,
                style_strength,
                LORA_TYPE_STYLE,
                model_family=model_family,
            ),
            _resolve_one_lora_selection(
                additional_key,
                additional_strength,
                LORA_TYPE_ADDITIONAL,
                model_family=model_family,
            ),
        )
        if selection is not None
    )
    if len({selection.key for selection in selections}) != len(selections):
        raise ValueError("同じLoRAを複数の枠へ選択することはできません。")
    return selections


def resolve_saved_lora_selections(
    raw_selections: object,
    *,
    model_family: str | None = None,
) -> tuple[LoraSelection, ...]:
    """Validate replayable selection snapshots without trusting their paths."""

    if raw_selections in (None, (), []):
        return ()
    if not isinstance(raw_selections, (list, tuple)):
        raise ValueError("保存済みLoRA構成が不正です。")

    selections: list[LoraSelection] = []
    for raw in raw_selections:
        if isinstance(raw, LoraSelection):
            key = raw.key
            strength = raw.strength
            lora_type = raw.lora_type
        elif isinstance(raw, dict):
            key = raw.get("key")
            strength = raw.get("strength")
            lora_type = raw.get("lora_type")
        else:
            raise ValueError("保存済みLoRA構成が不正です。")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("保存済みLoRAのkeyが不正です。")
        current = get_comfy_style(key)
        if current is None:
            raise ValueError(f"未登録のLoRAです: {key.strip()}")
        expected_type = str(lora_type or current.lora_type)
        selection = _resolve_one_lora_selection(
            key,
            strength,
            expected_type,
            model_family=model_family,
        )
        if selection is not None:
            selections.append(selection)
    if len({selection.key for selection in selections}) != len(selections):
        raise ValueError("同じLoRAを複数指定することはできません。")
    return tuple(selections)


def effective_lora_selections(options: SdGenerationOptions) -> tuple[LoraSelection, ...]:
    model_family = _resolve_generation_model_family(options.model_key, options.model_family)
    if options.loras:
        return resolve_saved_lora_selections(options.loras, model_family=model_family)
    if options.style_key is None and options.lora_strength is None:
        return ()
    style_key, _, style_strength = resolve_style_selection(
        options.style_key,
        options.lora_strength,
        model_family=model_family,
    )
    return resolve_lora_selections(
        style_key=style_key,
        style_strength=style_strength,
        model_family=model_family,
    )


def lora_selections_to_dicts(selections: tuple[LoraSelection, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "key": selection.key,
            "lora_name": selection.lora_name,
            "lora_type": selection.lora_type,
            "display_name": selection.display_name,
            "trigger": selection.trigger,
            "strength": selection.strength,
            "model_family": selection.model_family,
        }
        for selection in selections
    )


def apply_lora_triggers(positive: str, selections: tuple[LoraSelection, ...]) -> str:
    """Prepend missing triggers while preserving atomic and weighted tags."""

    missing: list[str] = []
    for selection in selections:
        trigger = selection.trigger.strip()
        if not trigger:
            continue
        pattern = re.compile(r"(?<![\w-])" + re.escape(trigger) + r"(?![\w-])", re.IGNORECASE)
        if not pattern.search(positive):
            missing.append(trigger)
    if not missing:
        return positive
    body = positive.strip()
    return ", ".join([*missing, body]) if body else ", ".join(missing)


def get_lora_status_text(selections: tuple[LoraSelection, ...]) -> str:
    if not selections:
        return "LoRA: なし"
    lines = ["LoRA:"]
    for selection in selections:
        trigger = selection.trigger or "トリガーなし"
        label = LORA_TYPE_LABELS.get(selection.lora_type, selection.lora_type)
        lines.append(
            f"- {selection.display_name}（{label}） / strength {selection.strength:g} / trigger {trigger}"
        )
    return "\n".join(lines)


def get_style_status_text(style_key: str | None, strength: float | None) -> str:
    _, style, resolved_strength = resolve_style_selection(style_key, strength)
    if style is None or resolved_strength is None:
        return "Style: Default"
    return f"Style: {style.display_name}\nLoRA strength: {resolved_strength:g}"


def resolve_reference_strength(reference_used: bool, strength: float | None) -> float | None:
    if not reference_used:
        if strength is not None:
            raise ValueError("reference_strengthを指定するにはreference_imageを添付してください。")
        return None
    resolved = DEFAULT_REFERENCE_STRENGTH if strength is None else strength
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise ValueError("reference_strengthは0.0～1.0の数値で指定してください。")
    resolved = float(resolved)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("reference_strengthは0.0～1.0の範囲で指定してください。")
    return resolved


def get_reference_status_text(reference_used: bool, strength: float | None) -> str:
    resolved = resolve_reference_strength(reference_used, strength)
    if resolved is None:
        return "Reference image: Not used"
    return f"Reference image: Used\nReference strength: {resolved:g}"


def validate_options(options: SdGenerationOptions) -> SdGenerationOptions:
    max_count = _max_int("SD_MAX_COUNT", 10)
    max_width = _max_int("SD_MAX_WIDTH", 1024)
    max_height = _max_int("SD_MAX_HEIGHT", 1024)
    max_steps = _max_int("SD_MAX_STEPS", 60)
    max_cfg = _max_float("SD_MAX_CFG", 20.0)
    allowed_generation_fields = {"width", "height", "steps", "cfg", "sampler"}
    explicit_generation_fields = frozenset(options.explicit_generation_fields)
    unknown_generation_fields = explicit_generation_fields - allowed_generation_fields
    if unknown_generation_fields:
        raise ValueError("explicit_generation_fieldsに未対応の項目があります。")

    prompt = options.prompt if options.preserve_prompt_text else options.prompt.strip()
    if not prompt.strip():
        raise ValueError("prompt is empty.")

    if options.preserve_prompt_text:
        negative_prompt = options.negative_prompt if options.negative_prompt is not None else ""
    else:
        negative_prompt = get_negative_prompt(options.negative_prompt)

    resolved_model_family = _resolve_generation_model_family(options.model_key, options.model_family)
    style_key, _, lora_strength = resolve_style_selection(
        options.style_key,
        options.lora_strength,
        model_family=resolved_model_family,
    )
    loras = effective_lora_selections(options)
    reference_used = options.reference_image_bytes is not None
    reference_strength = resolve_reference_strength(reference_used, options.reference_strength)
    reference_content_type = None
    if reference_used:
        if not isinstance(options.reference_image_bytes, bytes) or not options.reference_image_bytes:
            raise ValueError("reference_imageの画像データが空です。")
        reference_content_type = (options.reference_image_content_type or "").split(";", 1)[0].strip().lower()
        if reference_content_type not in REFERENCE_IMAGE_EXTENSIONS:
            raise ValueError("reference_imageはJPEG / PNG / WebPを指定してください。")

    reference_mode = str(options.reference_mode or "none").strip().casefold()
    if reference_mode not in {"none", "composition", REFERENCE_MODE_POSE}:
        raise ValueError("reference_modeはnone / composition / poseのいずれかで指定してください。")
    pose_used = options.pose_reference_bytes is not None
    if reference_mode == REFERENCE_MODE_POSE and not pose_used:
        raise ValueError("pose reference画像が指定されていません。")
    if reference_mode != REFERENCE_MODE_POSE and pose_used:
        raise ValueError("pose reference画像はposeモードでのみ指定できます。")
    pose_content_type = None
    if pose_used:
        if not isinstance(options.pose_reference_bytes, bytes) or not options.pose_reference_bytes:
            raise ValueError("pose referenceの画像データが空です。")
        pose_content_type = (options.pose_reference_content_type or "").split(";", 1)[0].strip().lower()
        if pose_content_type not in REFERENCE_IMAGE_EXTENSIONS:
            raise ValueError("pose referenceはJPEG / PNG / WebPを指定してください。")
        if isinstance(options.pose_strength, bool) or not isinstance(options.pose_strength, (int, float)):
            raise ValueError("pose_strengthは0.0～1.0の数値で指定してください。")
        if not math.isfinite(float(options.pose_strength)) or not 0.0 <= float(options.pose_strength) <= 1.0:
            raise ValueError("pose_strengthは0.0～1.0の範囲で指定してください。")
        for name, value in (("pose_start_percent", options.pose_start_percent), ("pose_end_percent", options.pose_end_percent)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name}は0.0～1.0の数値で指定してください。")
        if options.pose_start_percent > options.pose_end_percent:
            raise ValueError("pose_start_percentはpose_end_percent以下にしてください。")
        if options.pose_fit_mode not in {"contain", "cover", "stretch"}:
            raise ValueError("pose_fit_modeはcontain / cover / stretchのいずれかで指定してください。")

    return SdGenerationOptions(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_key=options.model_key,
        model_family=resolved_model_family,
        filename_prefix=options.filename_prefix,
        model_selection_explicit=options.model_selection_explicit,
        preserve_prompt_text=options.preserve_prompt_text,
        count=_clamp_int(options.count, 1, max_count),
        width=_clamp_int(options.width, 256, max_width),
        height=_clamp_int(options.height, 256, max_height),
        steps=_clamp_int(options.steps, 1, max_steps),
        cfg=_clamp_float(options.cfg, 1.0, max_cfg),
        sampler=options.sampler.strip() or "Euler a",
        seed=options.seed if options.seed is not None else -1,
        style_key=style_key,
        lora_strength=lora_strength,
        loras=loras,
        reference_image_bytes=options.reference_image_bytes,
        reference_image_content_type=reference_content_type,
        reference_strength=reference_strength,
        reference_mode=reference_mode,
        pose_reference_bytes=options.pose_reference_bytes,
        pose_reference_content_type=pose_content_type,
        pose_reference_width=options.pose_reference_width,
        pose_reference_height=options.pose_reference_height,
        pose_strength=float(options.pose_strength),
        pose_start_percent=float(options.pose_start_percent),
        pose_end_percent=float(options.pose_end_percent),
        pose_fit_mode=options.pose_fit_mode,
        pose_detect_hand=bool(options.pose_detect_hand),
        pose_detect_face=bool(options.pose_detect_face),
        pose_debug_enabled=bool(options.pose_debug_enabled),
        gallery_output_root=options.gallery_output_root,
        gallery_metadata=(None if options.gallery_metadata is None else dict(options.gallery_metadata)),
        explicit_generation_fields=explicit_generation_fields,
    )


def is_api_available() -> bool:
    try:
        if _backend() == "comfy":
            response = requests.get(_api_url("/system_stats"), timeout=3)
        else:
            response = requests.get(_api_url("/sdapi/v1/progress"), timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False


def is_comfy_queue_empty() -> bool:
    """Return True only when ComfyUI reports no running or pending prompts."""

    if _backend() != "comfy":
        return False
    try:
        response = requests.get(_api_url("/queue"), timeout=3)
        response.raise_for_status()
        payload = response.json()
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("queue_running"), list)
            and isinstance(payload.get("queue_pending"), list)
            and not payload["queue_running"]
            and not payload["queue_pending"]
        )
    except (requests.RequestException, ValueError, TypeError):
        return False


def _start_file_env_name() -> str:
    return "COMFYUI_START_FILE" if _backend() == "comfy" else "SD_START_FILE"


def _start_file_missing_message(env_name: str) -> str:
    if env_name == "COMFYUI_START_FILE":
        return "COMFYUI_START_FILE が未設定です。ComfyUIを起動する .bat / .cmd / .exe の固定パスを .env に設定してください。"
    return "SD_START_FILE が未設定です。起動用 .bat / .cmd / .exe の固定パスを .env に設定してください。"


def _launch_fixed_start_file(path: Path, cwd: Path) -> None:
    if path.suffix.lower() in {".bat", ".cmd"}:
        subprocess.Popen(["cmd.exe", "/c", str(path)], cwd=str(cwd), shell=False)
    else:
        subprocess.Popen([str(path)], cwd=str(cwd), shell=False)


def _start_sd_app_impl() -> str:
    if is_api_available():
        return "画像生成APIはすでに起動しています。"

    env_name = _start_file_env_name()
    start_file = os.getenv(env_name, "").strip().strip('"')
    if not start_file:
        return _start_file_missing_message(env_name)

    path = Path(start_file)
    if not path.exists() or not path.is_file():
        return f"{env_name} の起動ファイルが見つかりません: {path}"

    if path.suffix.lower() not in {".bat", ".cmd", ".exe"}:
        return f"{env_name} は .bat / .cmd / .exe の固定パスだけに対応しています。"

    cwd_text = os.getenv("SD_START_CWD", "").strip().strip('"')
    cwd = Path(cwd_text) if cwd_text else path.parent
    if not cwd.exists() or not cwd.is_dir():
        return f"起動時の作業フォルダが見つかりません: {cwd}"

    try:
        _launch_fixed_start_file(path, cwd)
    except Exception:
        logger.exception("Failed to start SD app: backend=%s env=%s path=%s cwd=%s", _backend(), env_name, path, cwd)
        return "画像生成APIの起動に失敗しました。Botのログを確認してください。"

    logger.info("Started SD app: backend=%s env=%s path=%s cwd=%s", _backend(), env_name, path, cwd)
    time.sleep(STARTUP_STATUS_WAIT_SECONDS)
    if is_api_available():
        return "画像生成APIを起動しました。疎通確認もできています。"

    return "起動ファイルを実行しました。APIはまだ応答していません。少し待ってから /sd_status を確認してください。"


def start_sd_app() -> str:
    return _start_sd_app_impl()


def _legacy_start_sd_app_unreachable() -> str:
    if is_api_available():
        return "画像生成APIは起動済みだよ。"

    start_file = os.getenv("SD_START_FILE", "").strip().strip('"')
    if not start_file:
        return "SD_START_FILE が未設定だよ。Stability MatrixからComfyUIを手動で起動してね。"

    path = Path(start_file)
    if not path.exists() or not path.is_file():
        return f"起動ファイルが見つからないよ: {path}"

    if path.suffix.lower() not in {".bat", ".cmd", ".exe"}:
        return "SD_START_FILE は .bat / .cmd / .exe の固定パスだけ対応しているよ。"

    cwd_text = os.getenv("SD_START_CWD", "").strip().strip('"')
    cwd = Path(cwd_text) if cwd_text else path.parent
    if not cwd.exists() or not cwd.is_dir():
        return f"作業フォルダが見つからないよ: {cwd}"

    try:
        if path.suffix.lower() in {".bat", ".cmd"}:
            subprocess.Popen(["cmd.exe", "/c", str(path)], cwd=str(cwd), shell=False)
        else:
            subprocess.Popen([str(path)], cwd=str(cwd), shell=False)
    except Exception:
        logger.exception("Failed to start SD app.")
        return "画像生成UIの起動に失敗したよ。Botのログを確認してね。"

    return "画像生成UIを起動したよ。APIが使えるようになったら /sd_status で確認してね。"


def get_status_message() -> str:
    backend = _backend()
    available = is_api_available()
    lines = [
        "画像生成の状態だよ。",
        f"backend: {backend}",
        f"api: {'起動中' if available else '未起動または未応答'}",
        f"bot_generation_lock: {'生成中' if is_generation_running() else '待機中'}",
    ]

    if backend == "comfy":
        workflow_file = _get_comfy_workflow_file()
        resolved_workflow_file = _resolve_project_path(workflow_file)
        lines.append(f"workflow: {workflow_file}")
        lines.append(f"workflow_exists: {'yes' if resolved_workflow_file.exists() else 'no'}")
    elif available:
        try:
            response = requests.get(_api_url("/sdapi/v1/progress"), params={"skip_current_image": "true"}, timeout=5)
            response.raise_for_status()
            data = response.json()
            progress = float(data.get("progress", 0.0)) * 100
            eta = data.get("eta_relative")
            lines.append(f"webui_progress: {progress:.1f}%")
            if isinstance(eta, (int, float)) and eta > 0:
                lines.append(f"eta_seconds: {eta:.1f}")
        except requests.RequestException:
            logger.exception("Failed to fetch SD progress.")
            lines.append("webui_progress: 取得失敗")

    return "\n".join(lines)


def _decode_and_save_webui_images(images: list[Any]) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _now_text()
    paths: list[Path] = []

    for index, raw_image in enumerate(images, start=1):
        if not isinstance(raw_image, str):
            continue
        image_data = raw_image.split(",", 1)[-1]
        image_bytes = base64.b64decode(image_data)
        output_path = OUTPUT_DIR / f"{timestamp}_{index:02d}.png"
        output_path.write_bytes(image_bytes)
        paths.append(output_path)

    _save_latest_paths(paths)
    return paths


def _get_comfy_workflow_file(
    reference_enabled: bool = False,
    loras: Sequence[LoraSelection] = (),
) -> Path:
    if reference_enabled:
        if loras:
            raise ValueError("LoRAとReference imageの同時適用には対応していません。")
        return REFERENCE_COMFY_WORKFLOW_FILE
    if loras:
        return LORA_COMFY_WORKFLOW_FILE
    raw_value = os.getenv("COMFY_WORKFLOW_FILE", "").strip().strip('"')
    return Path(raw_value) if raw_value else DEFAULT_COMFY_WORKFLOW_FILE


def _resolve_project_path(path: Path) -> Path:
    return resolve_project_path(path, path)


def _get_node_id(name: str) -> str:
    return os.getenv(name, "").strip()


def _load_comfy_workflow(
    reference_enabled: bool = False,
    loras: Sequence[LoraSelection] = (),
) -> dict[str, Any]:
    workflow_file = _resolve_project_path(_get_comfy_workflow_file(reference_enabled, loras))
    if not workflow_file.exists() or not workflow_file.is_file():
        raise RuntimeError(f"ComfyUI workflow file が見つからないよ: {workflow_file}")

    try:
        workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ComfyUI workflow file のJSONを読み取れなかったよ: {workflow_file}") from exc

    if not isinstance(workflow, dict):
        raise RuntimeError("ComfyUI workflow はAPI形式のJSONオブジェクトにしてね。")
    _validate_comfy_workflow(workflow, workflow_file)
    return workflow


def _validate_comfy_workflow(workflow: dict[str, Any], workflow_file: Path) -> None:
    if not workflow:
        raise RuntimeError(f"ComfyUI workflow が空です: {workflow_file}")

    invalid_nodes: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            invalid_nodes.append(str(node_id))
            continue
        if not isinstance(node.get("class_type"), str) or not node.get("class_type"):
            invalid_nodes.append(str(node_id))
            continue
        if not isinstance(node.get("inputs"), dict):
            invalid_nodes.append(str(node_id))

    if invalid_nodes:
        raise RuntimeError(
            "ComfyUI workflow がAPI形式として正しくない可能性があります。"
            f" class_type / inputs を確認してください: {', '.join(invalid_nodes[:10])}"
        )

    save_nodes = _find_save_image_nodes(workflow)
    if not save_nodes:
        raise RuntimeError("ComfyUI workflow に SaveImage ノードがありません。")

    positive_id, negative_id = _find_prompt_text_nodes(workflow)
    if not positive_id or not negative_id:
        raise RuntimeError("ComfyUI workflow の Positive / Negative CLIPTextEncode を検出できません。")

    logger.info(
        "ComfyUI workflow validated: file=%s nodes=%s positive=%s negative=%s save_nodes=%s",
        workflow_file,
        len(workflow),
        positive_id,
        negative_id,
        save_nodes,
    )


def _node_inputs(workflow: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else None


def _node_class(workflow: dict[str, Any], node_id: str) -> str:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return ""
    return str(node.get("class_type", ""))


def _linked_node_id(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    return None


def _is_clip_text_node(workflow: dict[str, Any], node_id: str) -> bool:
    inputs = _node_inputs(workflow, node_id)
    return _node_class(workflow, node_id) == "CLIPTextEncode" and inputs is not None and "text" in inputs


def _find_sampler_node(workflow: dict[str, Any]) -> str | None:
    preferred_classes = {
        "KSampler",
        "KSamplerAdvanced",
        "SamplerCustom",
        "SamplerCustomAdvanced",
    }
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type in preferred_classes and ("positive" in inputs or "negative" in inputs):
            return str(node_id)

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "positive" in inputs and "negative" in inputs:
            return str(node_id)
    return None


def _resolve_clip_text_node(workflow: dict[str, Any], start_node_id: str | None, visited: set[str] | None = None) -> str | None:
    if not start_node_id:
        return None
    visited = visited or set()
    if start_node_id in visited:
        return None
    visited.add(start_node_id)

    if _is_clip_text_node(workflow, start_node_id):
        return start_node_id

    inputs = _node_inputs(workflow, start_node_id)
    if not inputs:
        return None

    for value in inputs.values():
        linked_id = _linked_node_id(value)
        if linked_id is None:
            continue
        found = _resolve_clip_text_node(workflow, linked_id, visited)
        if found:
            return found
    return None


def _find_prompt_text_nodes(workflow: dict[str, Any]) -> tuple[str | None, str | None]:
    sampler_id = _find_sampler_node(workflow)
    if sampler_id:
        sampler_inputs = _node_inputs(workflow, sampler_id) or {}
        positive_id = _resolve_clip_text_node(workflow, _linked_node_id(sampler_inputs.get("positive")))
        negative_id = _resolve_clip_text_node(workflow, _linked_node_id(sampler_inputs.get("negative")))
        if positive_id or negative_id:
            logger.info(
                "ComfyUI prompt nodes detected from sampler: sampler=%s positive=%s negative=%s",
                sampler_id,
                positive_id,
                negative_id,
            )
            return positive_id, negative_id

    clip_nodes = [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and _is_clip_text_node(workflow, str(node_id))
    ]
    logger.warning("Falling back to CLIPTextEncode order detection: nodes=%s", clip_nodes)
    positive_id = clip_nodes[0] if len(clip_nodes) >= 1 else None
    negative_id = clip_nodes[1] if len(clip_nodes) >= 2 else None
    return positive_id, negative_id


def _find_seed_input(workflow: dict[str, Any]) -> tuple[str | None, str | None]:
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if node.get("class_type") == "RandomNoise" and isinstance(inputs, dict) and "noise_seed" in inputs:
            return str(node_id), "noise_seed"

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "seed" in inputs:
            return str(node_id), "seed"

    return None, None


def _find_save_image_nodes(workflow: dict[str, Any]) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "SaveImage"
    ]


def _find_unique_node_by_class(workflow: dict[str, Any], class_type: str) -> str | None:
    node_ids = [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == class_type
    ]
    if len(node_ids) > 1:
        raise RuntimeError(f"ComfyUI workflow に{class_type}ノードが複数あります。")
    return node_ids[0] if node_ids else None


def _apply_explicit_generation_settings(
    workflow: dict[str, Any], options: SdGenerationOptions
) -> None:
    fields = options.explicit_generation_fields
    if not fields:
        return

    latent_fields = fields & {"width", "height"}
    if latent_fields:
        latent_node_id = _find_unique_node_by_class(workflow, "EmptyLatentImage")
        if latent_node_id is None:
            raise RuntimeError("明示された生成サイズに対応するEmptyLatentImageノードが見つかりません。")
        latent_inputs = _node_inputs(workflow, latent_node_id)
        if latent_inputs is None:
            raise RuntimeError("EmptyLatentImageノードのinputsが不正です。")
        for field in latent_fields:
            if field not in latent_inputs:
                raise RuntimeError(f"EmptyLatentImageノードに{field}入力がありません。")
        if "width" in latent_fields:
            _set_node_input(workflow, latent_node_id, "width", options.width)
        if "height" in latent_fields:
            _set_node_input(workflow, latent_node_id, "height", options.height)

    sampler_fields = fields & {"steps", "cfg", "sampler"}
    if sampler_fields:
        sampler_node_id = _find_unique_node_by_class(workflow, "KSampler")
        if sampler_node_id is None:
            raise RuntimeError("明示された生成設定に対応するKSamplerノードが見つかりません。")
        sampler_inputs = _node_inputs(workflow, sampler_node_id)
        if sampler_inputs is None:
            raise RuntimeError("KSamplerノードのinputsが不正です。")
        input_names = {"sampler": "sampler_name"}
        for field in sampler_fields:
            input_name = input_names.get(field, field)
            if input_name not in sampler_inputs:
                raise RuntimeError(f"KSamplerノードに{input_name}入力がありません。")
        if "steps" in sampler_fields:
            _set_node_input(workflow, sampler_node_id, "steps", options.steps)
        if "cfg" in sampler_fields:
            _set_node_input(workflow, sampler_node_id, "cfg", options.cfg)
        if "sampler" in sampler_fields:
            _set_node_input(workflow, sampler_node_id, "sampler_name", options.sampler)


def _extract_checkpoint_name(workflow: dict[str, Any]) -> str | None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) != "CheckpointLoaderSimple":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        ckpt_name = inputs.get("ckpt_name")
        if isinstance(ckpt_name, str) and ckpt_name:
            return ckpt_name
    return None


def _extract_sampler_settings(
    workflow: dict[str, Any], sampler_node_id: str | None
) -> tuple[int | None, float | None, str | None, str | None]:
    if not sampler_node_id:
        return None, None, None, None
    inputs = _node_inputs(workflow, sampler_node_id) or {}

    steps = inputs.get("steps")
    steps = steps if isinstance(steps, int) else None

    cfg = inputs.get("cfg")
    cfg = float(cfg) if isinstance(cfg, (int, float)) else None

    sampler_name = inputs.get("sampler_name")
    sampler_name = sampler_name if isinstance(sampler_name, str) and sampler_name else None

    scheduler = inputs.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, str) and scheduler else None

    return steps, cfg, sampler_name, scheduler


def _extract_latent_dimensions(workflow: dict[str, Any]) -> tuple[int | None, int | None]:
    # Prefer an exact EmptyLatentImage match (what this project's workflow
    # actually uses); only fall back to a looser "class name contains
    # Latent, and both width/height are present" match if no exact match is
    # found, so we don't misattribute dimensions from an unrelated node.
    fallback_match: tuple[int, int] | None = None
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        width = inputs.get("width")
        height = inputs.get("height")
        if not (isinstance(width, int) and isinstance(height, int)):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type == "EmptyLatentImage":
            return width, height
        if fallback_match is None and "Latent" in class_type:
            fallback_match = (width, height)
    return fallback_match if fallback_match is not None else (None, None)


def _safe_workflow_file_hash(workflow_file: Path) -> str | None:
    workflow_file = _resolve_project_path(workflow_file)
    try:
        data = workflow_file.read_bytes()
    except OSError:
        logger.warning("Could not read workflow file to compute its hash: %s", workflow_file)
        return None
    return hashlib.sha256(data).hexdigest()


def _set_node_input(workflow: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise RuntimeError(f"ComfyUI workflow の node {node_id} が見つからないよ。")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"ComfyUI workflow の node {node_id} に inputs がないよ。")
    inputs[key] = value


def _resolve_comfy_checkpoint(model_key: str) -> str:
    try:
        return COMFY_CHECKPOINTS[model_key]
    except KeyError as exc:
        raise ValueError("指定された画像生成モデルには対応していません。") from exc


def _find_checkpoint_loader_node_id(workflow: dict[str, Any]) -> str:
    node_ids = [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "CheckpointLoaderSimple"
    ]
    if not node_ids:
        raise RuntimeError(
            "ComfyUI workflow に CheckpointLoaderSimple ノードが見つかりません。"
            "モデルを選択するにはチェックポイント読込ノードが1個必要です。"
        )
    if len(node_ids) > 1:
        raise RuntimeError(
            "ComfyUI workflow に CheckpointLoaderSimple ノードが複数あります。"
            "変更対象を特定できないためモデルを適用できません。"
        )
    return node_ids[0]


def _find_lora_loader_node_ids(workflow: dict[str, Any]) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "LoraLoader"
    ]


def _new_workflow_node_id(workflow: dict[str, Any]) -> str:
    numeric_ids = [int(str(node_id)) for node_id in workflow if str(node_id).isdigit()]
    candidate = max(numeric_ids, default=0) + 1
    while str(candidate) in workflow:
        candidate += 1
    return str(candidate)


def _replace_workflow_reference(
    workflow: dict[str, Any],
    old_reference: list[Any],
    new_reference: list[Any],
    *,
    excluded_node_ids: set[str],
) -> None:
    for node_id, node in workflow.items():
        if str(node_id) in excluded_node_ids or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in tuple(inputs.items()):
            if value == old_reference:
                inputs[input_name] = list(new_reference)


def _configure_lora_loaders(
    workflow: dict[str, Any],
    selections: Sequence[LoraSelection],
    checkpoint_node_id: str,
) -> None:
    """Create a serial MODEL+CLIP LoraLoader chain for the selected LoRAs."""

    if not selections:
        return
    # Catalog names are filesystem-facing values.  ComfyUI validates the
    # loader input against /object_info, so resolve every selection before
    # mutating the workflow and preserve the exact accepted API strings.
    resolved_lora_names = resolve_comfy_lora_names(
        selection.lora_name for selection in selections
    )
    existing_ids = _find_lora_loader_node_ids(workflow)
    loader_ids = existing_ids[:1]
    while len(loader_ids) < len(selections):
        loader_id = _new_workflow_node_id(workflow)
        workflow[loader_id] = {
            "inputs": {},
            "class_type": "LoraLoader",
            "_meta": {"title": "Generated LoRA Loader"},
        }
        loader_ids.append(loader_id)

    excluded_node_ids = set(loader_ids)
    base_model = [str(checkpoint_node_id), 0]
    base_clip = [str(checkpoint_node_id), 1]

    # A supplied LoRA workflow already has a loader connected to downstream
    # consumers.  Repoint those consumers to the final loader before writing
    # the chain inputs; this also handles a workflow with one placeholder
    # loader and a request for multiple LoRAs.
    for old_loader_id in existing_ids:
        _replace_workflow_reference(
            workflow,
            [old_loader_id, 0],
            [loader_ids[-1], 0],
            excluded_node_ids=excluded_node_ids,
        )
        _replace_workflow_reference(
            workflow,
            [old_loader_id, 1],
            [loader_ids[-1], 1],
            excluded_node_ids=excluded_node_ids,
        )
    _replace_workflow_reference(
        workflow,
        base_model,
        [loader_ids[0], 0],
        excluded_node_ids=excluded_node_ids,
    )
    _replace_workflow_reference(
        workflow,
        base_clip,
        [loader_ids[0], 1],
        excluded_node_ids=excluded_node_ids,
    )

    previous_model = base_model
    previous_clip = base_clip
    for loader_id, selection, resolved_name in zip(loader_ids, selections, resolved_lora_names):
        inputs = _node_inputs(workflow, loader_id)
        if inputs is None:
            raise RuntimeError(f"LoRA Loaderノードのinputsが不正です: {loader_id}")
        inputs.update(
            {
                "lora_name": resolved_name,
                "strength_model": selection.strength,
                "strength_clip": selection.strength,
                "model": list(previous_model),
                "clip": list(previous_clip),
            }
        )
        workflow[loader_id]["class_type"] = "LoraLoader"
        previous_model = [loader_id, 0]
        previous_clip = [loader_id, 1]

    if loader_ids[0] != loader_ids[-1]:
        _replace_workflow_reference(
            workflow,
            [loader_ids[0], 0],
            [loader_ids[-1], 0],
            excluded_node_ids=excluded_node_ids,
        )
        _replace_workflow_reference(
            workflow,
            [loader_ids[0], 1],
            [loader_ids[-1], 1],
            excluded_node_ids=excluded_node_ids,
        )


def _image_dimensions_from_bytes(image_bytes: bytes) -> tuple[int, int] | None:
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            return int(image.width), int(image.height)
    except (OSError, ValueError, SyntaxError):
        return None


def _configure_pose_controlnet(
    workflow: dict[str, Any],
    options: SdGenerationOptions,
    *,
    pose_reference_name: str,
    pose_control_image_name: str | None,
    pose_backend: str,
    positive_node_id: str,
    negative_node_id: str,
) -> dict[str, str]:
    """Attach the existing fit -> OpenPose ControlNet graph in memory only."""

    sampler_node_id = _find_sampler_node(workflow)
    if sampler_node_id is None:
        raise RuntimeError("ポーズ参照workflowにSamplerノードが見つかりません。")
    sampler_inputs = _node_inputs(workflow, sampler_node_id)
    if sampler_inputs is None:
        raise RuntimeError("ポーズ参照workflowのSampler入力が不正です。")
    target_width, target_height = _extract_latent_dimensions(workflow)
    if target_width is None or target_height is None:
        raise RuntimeError("pose ControlNetへ渡す生成canvasの幅・高さを取得できませんでした。")
    source_dimensions = _image_dimensions_from_bytes(options.pose_reference_bytes or b"")
    if source_dimensions is None:
        source_dimensions = (
            options.pose_reference_width or target_width,
            options.pose_reference_height or target_height,
        )
    fit = pose_fit_plan(
        source_dimensions[0],
        source_dimensions[1],
        target_width,
        target_height,
        options.pose_fit_mode,
    )

    def new_id() -> str:
        return _new_workflow_node_id(workflow)

    load_id = new_id()
    workflow[load_id] = {
        "inputs": {"image": pose_control_image_name or pose_reference_name},
        "class_type": "LoadImage",
        "_meta": {"title": f"{pose_backend} pose skeleton input"},
    }
    resize_id = new_id()
    workflow[resize_id] = {
        "inputs": {
            "image": [load_id, 0], "upscale_method": "lanczos",
            "width": int(fit["resize_width"]), "height": int(fit["resize_height"]),
            "crop": "disabled",
        },
        "class_type": "ImageScale",
        "_meta": {"title": f"Fit {pose_backend} to generation canvas"},
    }
    control_image_id = resize_id
    canvas_id = ""
    composite_id = ""
    crop_id = ""
    if options.pose_fit_mode == "contain":
        canvas_id = new_id()
        workflow[canvas_id] = {
            "inputs": {"width": target_width, "height": target_height, "batch_size": 1, "color": 0},
            "class_type": "EmptyImage",
            "_meta": {"title": "Black pose ControlNet canvas"},
        }
        composite_id = new_id()
        workflow[composite_id] = {
            "inputs": {
                "destination": [canvas_id, 0], "source": [resize_id, 0],
                "x": int(fit["left"]), "y": int(fit["top"]), "resize_source": False,
            },
            "class_type": "ImageCompositeMasked",
            "_meta": {"title": f"Pad {pose_backend} with black"},
        }
        control_image_id = composite_id
    elif options.pose_fit_mode == "cover":
        crop_id = new_id()
        workflow[crop_id] = {
            "inputs": {
                "image": [resize_id, 0], "upscale_method": "lanczos",
                "width": target_width, "height": target_height, "crop": "center",
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Center crop pose to generation canvas"},
        }
        control_image_id = crop_id

    loader_id = new_id()
    workflow[loader_id] = {
        "inputs": {"control_net_name": POSE_CONTROLNET_NAME},
        "class_type": "ControlNetLoader",
        "_meta": {"title": "SDXL OpenPose ControlNet"},
    }
    apply_id = new_id()
    workflow[apply_id] = {
        "inputs": {
            "positive": [positive_node_id, 0], "negative": [negative_node_id, 0],
            "control_net": [loader_id, 0], "image": [control_image_id, 0],
            "strength": options.pose_strength,
            "start_percent": options.pose_start_percent,
            "end_percent": options.pose_end_percent,
        },
        "class_type": "ControlNetApplyAdvanced",
        "_meta": {"title": "Apply OpenPose ControlNet"},
    }
    sampler_inputs["positive"] = [apply_id, 0]
    sampler_inputs["negative"] = [apply_id, 1]
    return {
        "load_node_id": load_id, "resize_node_id": resize_id,
        "canvas_node_id": canvas_id, "composite_node_id": composite_id,
        "crop_node_id": crop_id, "control_image_node_id": control_image_id,
        "control_loader_node_id": loader_id, "apply_node_id": apply_id,
        "sampler_node_id": sampler_node_id,
    }


def _apply_comfy_options(
    workflow: dict[str, Any],
    options: SdGenerationOptions,
    reference_image_name: str | None = None,
    *,
    pose_reference_name: str | None = None,
    pose_control_image_name: str | None = None,
    pose_backend: str | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    loras = effective_lora_selections(options)
    checkpoint_name = None
    checkpoint_node_id = None
    if options.model_key is not None:
        checkpoint_name = _resolve_comfy_checkpoint(options.model_key)
        checkpoint_node_id = _find_checkpoint_loader_node_id(workflow)
    elif loras:
        checkpoint_node_id = _find_checkpoint_loader_node_id(workflow)

    save_nodes = _find_save_image_nodes(workflow)
    if not save_nodes:
        raise RuntimeError("ComfyUI workflow に SaveImage ノードが見つかりません。完成画像を保存するWorkflowを指定してください。")
    if options.filename_prefix is not None:
        if len(save_nodes) != 1:
            raise RuntimeError(
                "ComfyUI workflow に SaveImage ノードが複数あります。"
                "モデル別の保存先を安全に特定できないため filename_prefix を適用できません。"
            )
        _set_node_input(workflow, save_nodes[0], "filename_prefix", options.filename_prefix)

    env_positive_node_id = _get_node_id("COMFY_POSITIVE_NODE_ID")
    env_negative_node_id = _get_node_id("COMFY_NEGATIVE_NODE_ID")
    detected_positive_node_id, detected_negative_node_id = _find_prompt_text_nodes(workflow)
    positive_node_id = env_positive_node_id or detected_positive_node_id
    negative_node_id = env_negative_node_id or detected_negative_node_id

    if not positive_node_id or not negative_node_id:
        raise RuntimeError(
            "Positive/Negative用のCLIPTextEncodeノードを自動検出できませんでした。"
            "KSamplerのpositive/negative接続を確認してください。"
        )

    if checkpoint_node_id is not None and checkpoint_name is not None:
        _set_node_input(workflow, checkpoint_node_id, "ckpt_name", checkpoint_name)

    positive_prompt = apply_lora_triggers(options.prompt, loras)
    if loras:
        if checkpoint_node_id is None:
            checkpoint_node_id = _find_checkpoint_loader_node_id(workflow)
        _configure_lora_loaders(workflow, loras, checkpoint_node_id)
        logger.info(
            "LoRA generation integration applied: keys=%s strengths=%s model_family=%s",
            [selection.key for selection in loras],
            [selection.strength for selection in loras],
            options.model_family,
        )

    _apply_explicit_generation_settings(workflow, options)

    _set_node_input(workflow, positive_node_id, "text", positive_prompt)
    _set_node_input(workflow, negative_node_id, "text", options.negative_prompt or "")

    if options.reference_image_bytes is not None:
        load_image_nodes = [
            str(node_id)
            for node_id, node in workflow.items()
            if isinstance(node, dict) and str(node.get("class_type", "")) == "LoadImage"
        ]
        adapter_nodes = [
            str(node_id)
            for node_id, node in workflow.items()
            if isinstance(node, dict) and str(node.get("class_type", "")) == "IPAdapterAdvanced"
        ]
        if len(load_image_nodes) != 1 or len(adapter_nodes) != 1:
            raise RuntimeError("参考画像workflowにはLoadImageとIPAdapterAdvancedが各1個必要です。")
        if not reference_image_name:
            raise RuntimeError("ComfyUIへアップロードした参考画像名がありません。")
        _set_node_input(workflow, load_image_nodes[0], "image", reference_image_name)
        _set_node_input(workflow, adapter_nodes[0], "weight", options.reference_strength)

    if options.pose_reference_bytes is not None:
        if not pose_reference_name:
            raise RuntimeError("ComfyUIへアップロードしたpose reference名がありません。")
        _configure_pose_controlnet(
            workflow,
            options,
            pose_reference_name=pose_reference_name,
            pose_control_image_name=pose_control_image_name,
            pose_backend=pose_backend or "dwpose",
            positive_node_id=positive_node_id,
            negative_node_id=negative_node_id,
        )

    env_seed_node_id = _get_node_id("COMFY_SEED_NODE_ID")
    seed_node_id, seed_key = _find_seed_input(workflow)
    if env_seed_node_id:
        seed_node_id = env_seed_node_id
        seed_key = "noise_seed" if _node_class(workflow, seed_node_id) == "RandomNoise" else "seed"

    if seed_node_id and seed_key:
        seed_value = options.seed if isinstance(options.seed, int) and options.seed >= 0 else random.randint(0, 2**32 - 1)
        _set_node_input(workflow, seed_node_id, seed_key, seed_value)
        logger.info(
            "ComfyUI workflow inputs applied: positive=%s negative=%s seed=%s.%s save_nodes=%s",
            positive_node_id,
            negative_node_id,
            seed_node_id,
            seed_key,
            save_nodes,
        )
    else:
        logger.warning("ComfyUI seed input was not found. Workflow seed was left unchanged.")

    return workflow, seed_node_id, seed_key


def _queue_comfy_prompt(workflow: dict[str, Any], client_id: str) -> str:
    response = requests.post(_api_url("/prompt"), json={"prompt": workflow, "client_id": client_id}, timeout=30)
    response.raise_for_status()
    data = response.json()
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI APIから prompt_id が返らなかったよ。")
    return prompt_id


def _upload_comfy_reference_image(image_bytes: bytes, content_type: str) -> str:
    extension = REFERENCE_IMAGE_EXTENSIONS[content_type]
    filename = f"discord_reference_{hashlib.sha256(image_bytes).hexdigest()[:20]}{extension}"
    try:
        response = requests.post(
            _api_url("/upload/image"),
            files={"image": (filename, image_bytes, content_type)},
            data={"type": "input", "overwrite": "true"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Failed to upload reference image to ComfyUI.")
        raise RuntimeError("参考画像をComfyUIへアップロードできませんでした。") from exc
    uploaded_name = data.get("name")
    if not isinstance(uploaded_name, str) or not uploaded_name:
        raise RuntimeError("ComfyUIの参考画像アップロード結果にファイル名がありません。")
    return uploaded_name


def _short_error_text(text: str) -> str:
    clean_text = " ".join((text or "").split())
    if not clean_text:
        return "レスポンス本文は空でした。"
    return clean_text[:COMFY_ERROR_TEXT_LIMIT]


def _queue_comfy_prompt_with_logging(workflow: dict[str, Any], client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    payload_text = json.dumps(payload, ensure_ascii=False)
    logger.info(
        "ComfyUI /prompt request: url=%s payload_length=%s payload=%s",
        _api_url("/prompt"),
        len(payload_text),
        payload_text[:COMFY_PAYLOAD_LOG_LIMIT],
    )

    try:
        response = requests.post(_api_url("/prompt"), json=payload, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text if response is not None else ""
        logger.error(
            "ComfyUI /prompt HTTP error: status=%s response=%s",
            getattr(response, "status_code", "unknown"),
            response_text,
        )
        raise RuntimeError(
            "ComfyUI APIがリクエストを受け付けませんでした。"
            f" HTTP {getattr(response, 'status_code', 'unknown')}: {_short_error_text(response_text)}"
        ) from exc
    except requests.RequestException as exc:
        logger.exception("ComfyUI /prompt request failed.")
        raise RuntimeError(f"ComfyUI APIへの接続に失敗しました: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("ComfyUI /prompt returned non-JSON response: %s", response.text)
        raise RuntimeError("ComfyUI APIからJSONではない応答が返りました。") from exc

    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        logger.error("ComfyUI /prompt response did not include prompt_id: %s", data)
        raise RuntimeError("ComfyUI APIから prompt_id が返りませんでした。")
    logger.info("ComfyUI /prompt accepted: prompt_id=%s", prompt_id)
    return prompt_id


def _wait_comfy_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(_api_url(f"/history/{prompt_id}"), timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and prompt_id in data:
            history = data[prompt_id]
            if isinstance(history, dict):
                return history
        time.sleep(COMFY_POLL_INTERVAL_SECONDS)
    raise RuntimeError("ComfyUIの生成完了待ちがタイムアウトしたよ。")


def _pose_history_execution_error(history: dict[str, Any]) -> str | None:
    """Return a concise ComfyUI execution error, including node context."""

    status = history.get("status") if isinstance(history, dict) else None
    status_str = status.get("status_str") if isinstance(status, dict) else None
    messages = status.get("messages") if isinstance(status, dict) else None
    if isinstance(status_str, str) and status_str.casefold() in {
        "error", "execution_error", "execution_interrupted"
    }:
        details: list[str] = [f"status={status_str}"]
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, (list, tuple)) or not item:
                    continue
                kind = str(item[0])
                payload = item[1] if len(item) > 1 else {}
                if not isinstance(payload, dict):
                    details.append(kind)
                    continue
                node_id = payload.get("node_id") or payload.get("node") or "?"
                node_type = payload.get("node_type") or payload.get("class_type") or "?"
                message = payload.get("exception_message") or payload.get("message") or "?"
                trace = payload.get("traceback")
                details.append(f"{kind} node_id={node_id} node_type={node_type} message={message}")
                if trace:
                    details.append(f"traceback={trace}")
        return "ComfyUI pose execution error: " + " | ".join(details)[:1200]

    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            kind = str(item[0]).casefold()
            if kind not in {"execution_error", "execution_interrupted"}:
                continue
            payload = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            node_id = payload.get("node_id") or payload.get("node") or "?"
            node_type = payload.get("node_type") or payload.get("class_type") or "?"
            message = payload.get("exception_message") or payload.get("message") or "?"
            return (
                f"ComfyUI pose execution error: kind={kind} node_id={node_id} "
                f"node_type={node_type} message={message}"
            )
    return None


def _image_pixel_stats_from_bytes(image_bytes: bytes) -> dict[str, int | float | bool]:
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        grayscale = image.convert("L")
        values = list(grayscale.getdata())
        minimum = min(values) if values else 0
        maximum = max(values) if values else 0
        mean = sum(values) / len(values) if values else 0.0
        return {
            "width": int(image.width), "height": int(image.height),
            "min_pixel": minimum, "max_pixel": maximum, "mean_pixel": mean,
            "complete_black": not values or maximum == 0,
        }


def _download_pose_preview(history: dict[str, Any], node_id: str, *, backend: str) -> Path | None:
    outputs = history.get("outputs") if isinstance(history, dict) else None
    output = outputs.get(str(node_id)) if isinstance(outputs, dict) else None
    images = output.get("images") if isinstance(output, dict) else None
    if not isinstance(images, list):
        return None
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            continue
        try:
            response = requests.get(
                _api_url("/view"),
                params={
                    "filename": item["filename"],
                    "type": item.get("type", "output"),
                    "subfolder": item.get("subfolder", ""),
                },
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Failed to download %s pose skeleton preview.", backend)
            return None
        POSE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        path = POSE_DEBUG_DIR / f"{backend}_{uuid.uuid4().hex[:12]}.png"
        path.write_bytes(response.content)
        return path
    return None


def _pose_metadata_from_history(history: dict[str, Any], node_id: str) -> list[Any]:
    outputs = history.get("outputs") if isinstance(history, dict) else None
    output = outputs.get(str(node_id)) if isinstance(outputs, dict) else None
    if not isinstance(output, dict):
        return []
    values: list[Any] = []
    for key in ("openpose_json", "text", "json", "metadata"):
        value = output.get(key)
        if value is not None:
            values.extend(extract_json_candidates(value))
    return values


def _read_pose_keypoint_file(filename_prefix: str) -> list[Any]:
    roots: list[Path] = []
    for env_name in ("COMFY_OUTPUT_DIR", "COMFYUI_OUTPUT_DIR"):
        value = os.getenv(env_name, "").strip().strip('"')
        if value:
            roots.append(Path(value))
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob(f"{filename_prefix}_*.json"):
                try:
                    return extract_json_candidates(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        except OSError:
            continue
    return []


def _prepare_pose_detection_image(image_bytes: bytes) -> bytes:
    """Flatten alpha/transparency for the DWPose retry without cv2."""

    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        output = BytesIO()
        background.convert("RGB").save(output, format="PNG")
        return output.getvalue()


def _run_pose_backend_preflight(
    image_name: str,
    *,
    backend: str,
    options: SdGenerationOptions,
    client_id: str,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    if backend == "dwpose":
        workflow, _, preview_id, metadata_id = build_dwpose_preflight_workflow(
            image_name,
            detect_hand=options.pose_detect_hand,
            detect_face=options.pose_detect_face,
        )
    elif backend == "sdpose":
        workflow, _, preview_id, metadata_id = build_sdpose_preflight_workflow(
            image_name,
            filename_prefix or f"discord_sdpose_{uuid.uuid4().hex[:12]}",
        )
    elif backend == "animepose":
        workflow, _, preview_id, metadata_id = build_animepose_preflight_workflow(image_name)
    else:
        raise ValueError(f"unknown pose backend: {backend}")

    result: dict[str, Any] = {"backend": backend, "success": False, "input": image_name}
    try:
        prompt_id = _queue_comfy_prompt_with_logging(workflow, client_id)
        history = _wait_comfy_history(prompt_id)
        execution_error = _pose_history_execution_error(history)
        if execution_error:
            raise RuntimeError(execution_error)
        preview_path = _download_pose_preview(history, preview_id, backend=backend)
        skeleton_stats = _image_pixel_stats_from_bytes(preview_path.read_bytes()) if preview_path else None
        candidates = _pose_metadata_from_history(history, metadata_id)
        if not candidates and backend == "sdpose":
            candidates = _read_pose_keypoint_file(filename_prefix or "")
        if not candidates:
            raise RuntimeError(f"{backend} keypoint metadata was not returned by ComfyUI")
        raw = candidates[0]
        if backend == "animepose":
            metadata = raw if isinstance(raw, dict) else {}
            batch = metadata.get("batch") if isinstance(metadata, dict) else None
            rows = batch[0].get("keypoints") if isinstance(batch, list) and batch and isinstance(batch[0], dict) else metadata.get("keypoints") if isinstance(metadata, dict) else None
            image_size = batch[0].get("image_size") if isinstance(batch, list) and batch and isinstance(batch[0], dict) else None
            summary = animepose_keypoint_summary(rows, image_size=image_size)
        else:
            summary = sdpose_keypoint_summary(raw, POSE_SDPOSE_DRAW_THRESHOLD)
        evaluation = evaluate_pose_quality(summary, skeleton_stats, backend=backend)
        if backend == "dwpose":
            evaluation["lower_limb_evidence_available"] = bool(candidates)
            evaluation["full_body_pass"] = bool(evaluation.get("full_body_pass"))
            evaluation["success"] = bool(evaluation.get("full_body_pass")) and preview_path is not None
        else:
            evaluation["success"] = bool(evaluation.get("pose_quality_pass")) and preview_path is not None
        result.update(evaluation)
        result["prompt_id"] = prompt_id
        result["skeleton_path"] = preview_path
        result["skeleton_stats"] = skeleton_stats
        result["keypoint_summary"] = summary
        result["profile"], result["profile_reason"] = classify_pose_reference_profile(result, backend=backend)
        if not result["success"]:
            result["failure_reason"] = result.get("failure_reason") or "pose quality gate failed"
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Pose preflight failed: backend=%s input=%s", backend, image_name)
    logger.info(
        "Pose preflight result: backend=%s success=%s profile=%s reason=%s people=%s "
        "valid=%s chains=%s torso=%s leg_evidence=%s full_body_pass=%s "
        "upper_body_quality_pass=%s detected=%s hip_normalized_y=%s arm_chains=%s",
        backend, result.get("success"), result.get("profile"), result.get("profile_reason"),
        result.get("people", "unknown"), result.get("valid_body_keypoints", result.get("body_keypoints_valid_gt0", "unknown")),
        result.get("completed_chain_count", "unknown"), result.get("torso_complete", False),
        result.get("full_body_leg_chain_evidence", False), result.get("full_body_pass", False),
        result.get("upper_body_quality_pass", False), result.get("detected", "unknown"),
        result.get("hip_normalized_y", "unknown"), result.get("upper_body_arm_chains", "unknown"),
    )
    return result


def _run_pose_pipeline(
    pose_reference_name: str,
    pose_reference_bytes: bytes,
    *,
    options: SdGenerationOptions,
    client_id: str,
) -> dict[str, Any]:
    """Run DWPose -> preprocessed DWPose -> SDPose -> AnimePose."""

    attempts: list[dict[str, Any]] = []
    dwpose = _run_pose_backend_preflight(
        pose_reference_name, backend="dwpose", options=options, client_id=client_id
    )
    attempts.append(dwpose)
    if dwpose.get("success"):
        return dwpose

    try:
        prepared_bytes = _prepare_pose_detection_image(pose_reference_bytes)
        prepared_name = _upload_comfy_reference_image(prepared_bytes, "image/png")
        prepared = _run_pose_backend_preflight(
            prepared_name,
            backend="dwpose",
            options=options,
            client_id=client_id,
        )
        prepared["preprocessed_fallback"] = True
        attempts.append(prepared)
        if prepared.get("success"):
            return prepared
    except Exception as exc:
        logger.exception("DWPose preprocessing fallback failed: %s", exc)

    sdpose = _run_pose_backend_preflight(
        pose_reference_name,
        backend="sdpose",
        options=options,
        client_id=client_id,
        filename_prefix=f"discord_sdpose_{uuid.uuid4().hex[:12]}",
    )
    attempts.append(sdpose)
    if sdpose.get("success"):
        return sdpose

    animepose = _run_pose_backend_preflight(
        pose_reference_name,
        backend="animepose",
        options=options,
        client_id=client_id,
    )
    attempts.append(animepose)
    if animepose.get("success"):
        return animepose

    reasons = "; ".join(
        f"{item.get('backend', 'unknown')}: {item.get('failure_reason', 'quality gate failed')}"
        for item in attempts
    )
    raise RuntimeError(f"Pose referenceの骨格品質判定に失敗しました。未検証の骨格は使用しません。 {reasons[:700]}")


def _write_gallery_metadata_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Write the small sidecar contract consumed by the optional Gallery.

    Sidecars are best effort: they must never turn a successfully downloaded
    image into a failed Discord generation.
    """

    sidecar = path.with_name(path.name + ".metadata.json")
    try:
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        logger.warning("Gallery metadata sidecar could not be written: %s", sidecar, exc_info=True)


def _resolve_gallery_output_path(
    output_root: Path | None,
    filename: str,
    subfolder: str,
    image_type: str,
) -> Path | None:
    """Resolve an authoritative ComfyUI output path without escaping root."""

    if output_root is None or image_type != "output":
        return None
    try:
        root = output_root.expanduser().resolve()
        candidate = (root / subfolder / filename).resolve()
        candidate.relative_to(root)
        return candidate if candidate.is_file() else None
    except (OSError, ValueError):
        return None


def _download_comfy_images(
    history: dict[str, Any],
    batch_index: int,
    *,
    gallery_output_root: Path | None = None,
    gallery_metadata: dict[str, Any] | None = None,
    prompt_id: str | None = None,
    actual_seed: int | None = None,
    pose_backend: str | None = None,
    pose_reference_profile: str | None = None,
    pose_profile_reason: str | None = None,
) -> list[Path]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("ComfyUI history に outputs がなかったよ。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _now_text()
    paths: list[Path] = []

    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        images = output.get("images")
        if not isinstance(images, list):
            continue
        for image_index, image in enumerate(images, start=1):
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            image_type = image.get("type", "output")
            subfolder = image.get("subfolder", "")
            if not isinstance(filename, str) or not filename:
                continue

            response = requests.get(
                _api_url("/view"),
                params={"filename": filename, "type": image_type, "subfolder": subfolder},
                timeout=60,
            )
            response.raise_for_status()
            suffix = Path(filename).suffix or ".png"
            output_path = OUTPUT_DIR / f"{timestamp}_{batch_index:02d}_{len(paths) + image_index:02d}{suffix}"
            output_path.write_bytes(response.content)
            if gallery_metadata is not None:
                payload = dict(gallery_metadata)
                payload.setdefault("job_id", prompt_id or "")
                payload.setdefault("prompt_id", prompt_id or "")
                payload.setdefault("source_image", filename)
                payload.setdefault("batch_index", batch_index)
                payload.setdefault("generated_at", datetime.now(JST).isoformat(timespec="seconds"))
                if actual_seed is not None:
                    payload["actual_seed"] = actual_seed
                    payload["seed"] = actual_seed
                if pose_backend:
                    payload["pose_backend"] = pose_backend
                    payload["pose_reference_profile"] = pose_reference_profile or "unknown"
                    payload["pose_profile_reason"] = pose_profile_reason or ""
                _write_gallery_metadata_sidecar(output_path, payload)
                source_path = _resolve_gallery_output_path(
                    gallery_output_root, filename, subfolder, str(image_type)
                )
                if source_path is not None:
                    _write_gallery_metadata_sidecar(source_path, payload)
            paths.append(output_path)

    if not paths:
        raise RuntimeError("ComfyUIの生成画像を取得できなかったよ。SaveImageノードがあるWorkflowか確認してね。")
    return paths


def _generate_webui_images(options: SdGenerationOptions) -> list[Path]:
    payload = {
        "prompt": options.prompt,
        "negative_prompt": options.negative_prompt,
        "batch_size": options.count,
        "n_iter": 1,
        "width": options.width,
        "height": options.height,
        "steps": options.steps,
        "cfg_scale": options.cfg,
        "sampler_name": options.sampler,
        "seed": options.seed,
    }
    response = requests.post(_api_url("/sdapi/v1/txt2img"), json=payload, timeout=None)
    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])
    if not isinstance(images, list) or not images:
        raise RuntimeError("画像生成APIから画像が返りませんでした。")

    paths = _decode_and_save_webui_images(images)
    if not paths:
        raise RuntimeError("生成画像の保存に失敗しました。")
    return paths


def _generate_comfy_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    reference_enabled = options.reference_image_bytes is not None
    pose_enabled = options.pose_reference_bytes is not None
    loras = effective_lora_selections(options)
    workflow_file = _get_comfy_workflow_file(reference_enabled, loras)
    workflow_file_text = str(workflow_file)
    workflow_hash = _safe_workflow_file_hash(workflow_file)

    reference_image_name = None
    if reference_enabled:
        reference_image_name = _upload_comfy_reference_image(
            options.reference_image_bytes or b"",
            options.reference_image_content_type or "image/png",
        )

    pose_reference_name = None
    pose_control_image_name = None
    pose_backend = None
    pose_result: dict[str, Any] | None = None
    client_id = str(uuid.uuid4())
    if pose_enabled:
        pose_reference_name = _upload_comfy_reference_image(
            options.pose_reference_bytes or b"",
            options.pose_reference_content_type or "image/png",
        )
        pose_result = _run_pose_pipeline(
            pose_reference_name,
            options.pose_reference_bytes or b"",
            options=options,
            client_id=client_id,
        )
        pose_backend = str(pose_result.get("backend"))
        skeleton_path = pose_result.get("skeleton_path")
        if not isinstance(skeleton_path, Path) or not skeleton_path.is_file():
            raise RuntimeError("品質判定を通過したpose skeleton画像がありません。")
        pose_control_image_name = _upload_comfy_reference_image(
            skeleton_path.read_bytes(), "image/png"
        )

    images: list[GeneratedImage] = []
    candidate_failures: list[Exception] = []
    for batch_index in range(1, options.count + 1):
        try:
            workflow, seed_node_id, seed_key = _apply_comfy_options(
                _load_comfy_workflow(reference_enabled, loras),
                options,
                reference_image_name,
                pose_reference_name=pose_reference_name,
                pose_control_image_name=pose_control_image_name,
                pose_backend=pose_backend,
            )

            seed_value: int | None = None
            if seed_node_id and seed_key:
                seed_inputs = _node_inputs(workflow, seed_node_id) or {}
                raw_seed = seed_inputs.get(seed_key)
                seed_value = raw_seed if isinstance(raw_seed, int) else None

            checkpoint_name = _extract_checkpoint_name(workflow)
            sampler_node_id = _find_sampler_node(workflow)
            steps, cfg, sampler_name, scheduler = _extract_sampler_settings(workflow, sampler_node_id)
            width, height = _extract_latent_dimensions(workflow)

            prompt_id = _queue_comfy_prompt_with_logging(workflow, client_id)
            history = _wait_comfy_history(prompt_id)
            paths = _download_comfy_images(
                history,
                batch_index,
                gallery_output_root=options.gallery_output_root,
                gallery_metadata=options.gallery_metadata,
                prompt_id=prompt_id,
                actual_seed=seed_value,
                pose_backend=pose_backend,
                pose_reference_profile=(None if pose_result is None else pose_result.get("profile")),
                pose_profile_reason=(None if pose_result is None else pose_result.get("profile_reason")),
            )

            for path in paths:
                images.append(
                    GeneratedImage(
                        path=path,
                        batch_index=batch_index,
                        seed=seed_value,
                        checkpoint_name=checkpoint_name,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        steps=steps,
                        cfg=cfg,
                        width=width,
                        height=height,
                        workflow_file=workflow_file_text,
                        workflow_hash=workflow_hash,
                        pose_backend=pose_backend,
                        pose_reference_profile=(
                            None if pose_result is None else pose_result.get("profile")
                        ),
                        pose_profile_reason=(
                            None if pose_result is None else pose_result.get("profile_reason")
                        ),
                        pose_reference_debug_path=(
                            None if pose_result is None else pose_result.get("skeleton_path")
                        ),
                    )
                )
        except Exception as exc:
            # Candidate failures must not discard successful siblings. Preserve
            # the historical single-image behavior by re-raising immediately
            # when there is no candidate fan-out.
            if options.count <= 1:
                raise
            candidate_failures.append(exc)
            logger.exception(
                "ComfyUI candidate failed; continuing with remaining candidates: batch_index=%s/%s",
                batch_index,
                options.count,
            )

    if not images and candidate_failures:
        raise RuntimeError("すべての候補画像の生成に失敗しました。") from candidate_failures[-1]
    if candidate_failures:
        logger.warning(
            "ComfyUI generation completed with partial candidate failures: failed=%s succeeded=%s",
            len(candidate_failures),
            len(images),
        )
    return images


def _generate_webui_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    # The webui backend has no ComfyUI-style workflow file/checkpoint-loader
    # node to read back, and the actual per-image seed used by the webui API
    # is not currently parsed out of its response - those fields are left as
    # None (see GeneratedImage's docstring) rather than guessed. The
    # remaining fields reflect the request options actually sent, unchanged.
    paths = _generate_webui_images(options)
    if options.gallery_metadata is not None:
        for index, path in enumerate(paths, start=1):
            payload = dict(options.gallery_metadata)
            payload.setdefault("batch_index", index)
            _write_gallery_metadata_sidecar(path, payload)
    return [
        GeneratedImage(
            path=path,
            batch_index=index,
            seed=None,
            checkpoint_name=None,
            sampler_name=options.sampler or None,
            scheduler=None,
            steps=options.steps,
            cfg=options.cfg,
            width=options.width,
            height=options.height,
            workflow_file=None,
            workflow_hash=None,
        )
        for index, path in enumerate(paths, start=1)
    ]


def generate_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    global _shutdown_after_done

    if not _generate_lock.acquire(blocking=False):
        raise RuntimeError("すでに画像生成中だよ。完了してからもう一度実行してね。")

    safe_options = validate_options(options)
    try:
        backend = _backend()
        if backend != "comfy" and safe_options.model_key is not None and safe_options.model_selection_explicit:
            raise RuntimeError("モデルを明示選択できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。")
        if backend != "comfy" and safe_options.reference_image_bytes is not None:
            raise RuntimeError("reference_imageを指定できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。")
        if backend != "comfy" and safe_options.pose_reference_bytes is not None:
            raise RuntimeError("pose_referenceを指定できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。")
        if backend == "comfy":
            images = _generate_comfy_images_with_metadata(safe_options)
        elif backend == "webui":
            images = _generate_webui_images_with_metadata(safe_options)
        else:
            raise RuntimeError("SD_BACKEND は comfy または webui を指定してね。")

        _save_latest_paths([image.path for image in images])

        if _shutdown_after_done:
            _shutdown_after_done = False
            if is_windows():
                schedule_shutdown(get_shutdown_delay_seconds(), description="画像生成完了後のシャットダウン")
            else:
                logger.warning("Shutdown-after-done was requested, but OS is not Windows.")

        return images
    finally:
        _generate_lock.release()


def generate_images(options: SdGenerationOptions) -> list[Path]:
    return [image.path for image in generate_images_with_metadata(options)]


def set_shutdown_after_done(enabled: bool = True) -> str:
    global _shutdown_after_done
    _shutdown_after_done = enabled
    if enabled:
        return "画像生成完了後にPCをシャットダウンするよう予約したよ。次の生成完了後に実行するね。"
    return "画像生成完了後シャットダウンの予約を解除したよ。"


def save_favorites(*, all_images: bool = True, index: int = 1) -> list[Path]:
    latest_paths = get_latest_image_paths()
    if not latest_paths:
        raise RuntimeError("保存できる最新画像がないよ。")

    if all_images:
        targets = latest_paths
    else:
        if index < 1 or index > len(latest_paths):
            raise RuntimeError("指定した番号の最新画像が見つからないよ。")
        targets = [latest_paths[index - 1]]

    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for path in targets:
        destination = FAVORITES_DIR / path.name
        shutil.copy2(path, destination)
        saved_paths.append(destination)
    return saved_paths
