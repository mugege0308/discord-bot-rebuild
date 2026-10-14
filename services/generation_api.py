"""Local-only HTTP adapter for the shared generation service.

The API in this module is deliberately an adapter, not another generation
pipeline.  It validates a small public request shape, converts it to the
Discord-independent :class:`GenerationRequest`, and delegates prompt and
ComfyUI work to :class:`GenerationService`.

This module does not start a server on import.  ``create_generation_app`` is
used by the standalone launcher and by tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
from aiohttp import ContentTypeError, web
import requests

from services.appearance_resolver import get_appearance_resolver_status
from services.character_tag_dictionary import list_characters
from services.danbooru_tag_search import (
    TAG_SEARCH_CATEGORIES,
    TAG_SEARCH_DEFAULT_CATEGORY,
    TAG_SEARCH_DEFAULT_LIMIT,
    TAG_SEARCH_MAX_LIMIT,
    TAG_SEARCH_MAX_QUERY_LENGTH,
    TagSearchError,
    search_tag_candidates,
)
from services.danbooru_tag_store import DanbooruTagStore
from services.generation_service import GenerationRequest, GenerationService, PreparedGeneration
from services.generation_reference_assets import (
    MAX_REFERENCE_IMAGE_BYTES,
    load_reference_asset,
    save_reference_asset,
)
from services.generation_variation import (
    ALLOWED_BULK_GENERATION_COUNTS,
    MAX_GENERATION_COUNT,
    VARIATION_AUTO,
    VARIATION_OFF,
    merge_variation_tags,
    plan_variation_slots,
    plan_bulk_variation_slots,
)
from services.gallery_control import get_gallery_images_root
from services.image_composition import IMAGE_SIZE_SPECS
from services.multi_person_framing import (
    MULTI_PERSON_FRAMING_CHOICES,
    MULTI_PERSON_FRAMING_LABELS,
    validate_multi_person_framing,
)
from services.image_generation_profiles import (
    QWEN_IMAGE_ENGINE,
    enabled_image_generation_profiles,
    get_image_generation_profile,
    is_image_generation_profile_enabled,
)
from services.interaction_profiles import (
    InteractionProfileError,
    interaction_metadata,
    list_phase1_profiles,
    resolve_phase1_interaction,
)
from services.lora_catalog import COMFY_STYLES, model_family_compatible
from services.lora_training import LORA_TYPES, normalize_lora_type
from services.prompt_generator import PromptGeneratorError, _get_lmstudio_api_url
from services.sd_control import OUTPUT_DIR, is_api_available, resolve_saved_lora_selections
from services.sd_control import DEFAULT_REFERENCE_STRENGTH
from services.sd_control import is_comfy_queue_empty
from services.pose_reference import DEFAULT_POSE_STRENGTH
from services.shogou_prompt_designer import PROMPT_RATING_CHOICES
from services.image_selection import select_generated_images
from services.overnight_shutdown import (
    DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS,
    build_shutdown_command,
    schedule_overnight_shutdown,
)
from services.overnight_generation import (
    OVERNIGHT_ACTIVE_STATES,
    OVERNIGHT_MAX_CONSECUTIVE_INFRA_FAILURES,
    OVERNIGHT_MAX_RETRIES,
    OVERNIGHT_ROUND_SIZE,
    OVERNIGHT_TERMINAL_STATES,
    OVERNIGHT_TOTALS,
    OvernightJobError,
    OvernightJobStore,
    overnight_response,
    utc_now_iso,
)


logger = logging.getLogger("secretary-bot.generation_api")

DEFAULT_GENERATION_API_HOST = "127.0.0.1"
DEFAULT_GENERATION_API_PORT = 8787
DEFAULT_MAX_BODY_BYTES = 64 * 1024
DEFAULT_REFERENCE_UPLOAD_BYTES = MAX_REFERENCE_IMAGE_BYTES
MAX_TEXT_LENGTH = 4000
MAX_EXPLICIT_TAGS = 100
MAX_LORAS = 3
BULK_SELECTION_MODES = frozenset({"none", "best1", "best4"})
_UNSET = object()

_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "character_preset",
        "freeform_character",
        "explicit_tags",
        "appearance_text",
        "request_text",
        "interaction_id",
        "model_key",
        "mode",
        "rating",
        "loras",
        "size",
        "multi_person_framing",
        "count",
        "variation",
        "variation_offset",
        "seed",
        "reference_mode",
        "reference_image_id",
        "reference_strength",
        "reference_fit",
        # Accepted so edit clients receive a specific single-reference error.
        "reference_image_ids",
    }
)
_ALLOWED_LORA_KEYS = frozenset({"key", "lora_type", "strength"})
_ABSOLUTE_PATH_RE = re.compile(r"(?:(?:[A-Za-z]:)?[\\/])(?:[^\s,;]+[\\/])+[^\s,;]+")
GENERATION_API_APP_KEY = web.AppKey("generation_api", object)


class GenerationApiError(ValueError):
    """A client-visible, sanitized request validation error."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class GenerationDependencyError(GenerationApiError):
    """A safe error raised when an Overnight dependency is offline."""


@dataclass(frozen=True)
class GenerationApiConfig:
    """Configuration for the standalone local API.

    Only loopback binding is accepted.  The API intentionally has no CORS
    setting because browsers are expected to call a future Gallery proxy,
    rather than this service directly.
    """

    host: str = DEFAULT_GENERATION_API_HOST
    port: int = DEFAULT_GENERATION_API_PORT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    reference_upload_bytes: int = DEFAULT_REFERENCE_UPLOAD_BYTES

    def __post_init__(self) -> None:
        host = str(self.host or "").strip().casefold()
        if host == "localhost":
            host = DEFAULT_GENERATION_API_HOST
        if host != DEFAULT_GENERATION_API_HOST:
            raise ValueError("Generation APIは127.0.0.1にのみbindできます。")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("Generation API portが不正です。")
        if isinstance(self.max_body_bytes, bool) or not isinstance(self.max_body_bytes, int) or self.max_body_bytes < 1024:
            raise ValueError("Generation APIのbody上限が不正です。")
        if (
            isinstance(self.reference_upload_bytes, bool)
            or not isinstance(self.reference_upload_bytes, int)
            or self.reference_upload_bytes < 1024
            or self.reference_upload_bytes > MAX_REFERENCE_IMAGE_BYTES
        ):
            raise ValueError("参考画像uploadの上限が不正です。")
        object.__setattr__(self, "host", host)


@dataclass
class _Job:
    job_id: str
    request: GenerationRequest
    state: str = "queued"
    progress: float = 0.0
    error_summary: str | None = None
    output_metadata: dict[str, Any] | None = None
    total_count: int = 1
    completed_count: int = 0
    failed_count: int = 0
    variation: str = VARIATION_OFF
    output_items: list[dict[str, Any]] = field(default_factory=list)


class _JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}

    def create(self, request: GenerationRequest) -> _Job:
        with self._lock:
            job = _Job(job_id=secrets.token_urlsafe(12), request=request)
            self._jobs[job.job_id] = job
            return job

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)


@dataclass
class _BulkJob:
    bulk_id: str
    request: GenerationRequest
    requested_total: int
    selection_mode: str
    variation: str = VARIATION_OFF
    state: str = "queued"
    progress: float = 0.0
    current_index: int = 0
    completed_count: int = 0
    failed_count: int = 0
    error_summary: str | None = None
    errors: list[str] = field(default_factory=list)
    output_items: list[dict[str, Any]] = field(default_factory=list)
    selected_items: list[dict[str, Any]] = field(default_factory=list)
    selection_state: str = "not_requested"
    selection_error: str | None = None
    stop_requested: bool = False
    pause_requested: bool = False
    metadata_context: dict[str, Any] = field(default_factory=dict)
    seed_offset: int = 0
    requested_seed_override: object = _UNSET
    generated_paths: list[Path] = field(default_factory=list, repr=False)


class _BulkJobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, _BulkJob] = {}

    def create(
        self,
        request: GenerationRequest,
        total: int,
        selection_mode: str,
        variation: str,
        *,
        metadata_context: Mapping[str, Any] | None = None,
        seed_offset: int = 0,
        requested_seed_override: object = _UNSET,
    ) -> _BulkJob:
        with self._lock:
            if any(job.state in {"queued", "resolving", "generating", "selecting"} for job in self._jobs.values()):
                raise GenerationApiError("大量生成は同時に1件だけ実行できます。")
            job = _BulkJob(
                bulk_id=secrets.token_urlsafe(12),
                request=request,
                requested_total=total,
                selection_mode=selection_mode,
                variation=variation,
                metadata_context=dict(metadata_context or {}),
                seed_offset=int(seed_offset),
                requested_seed_override=requested_seed_override,
            )
            self._jobs[job.bulk_id] = job
            return job

    def get(self, bulk_id: str) -> _BulkJob | None:
        with self._lock:
            return self._jobs.get(bulk_id)

    def update(self, bulk_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[bulk_id]
            for key, value in values.items():
                setattr(job, key, value)


def _is_configured(*names: str) -> bool:
    return any(os.getenv(name, "").strip() for name in names)


def _json_error(message: str, status: int = 400, *, code: str | None = None) -> web.Response:
    payload: dict[str, object] = {"error": message}
    if code:
        payload["code"] = code
    return web.json_response(payload, status=status)


def _lm_studio_ready() -> bool:
    """Probe the endpoint used by the current Prompt Designer path.

    This is deliberately a small liveness check.  It never sends a prompt,
    never logs the configured URL, and accepts the two LM Studio API layouts
    used by the project.
    """

    base_url = _get_lmstudio_api_url().strip().rstrip("/")
    if not base_url:
        return False
    candidates = [
        f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models",
        f"{base_url}/api/v1/models" if not base_url.endswith("/api/v1") else f"{base_url}/models",
    ]
    for candidate in dict.fromkeys(candidates):
        try:
            response = requests.get(candidate, timeout=2)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            return True
    return False


def _clean_text(value: object, field_name: str, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise GenerationApiError(f"{field_name}は文字列で指定してください。")
    value = value.strip()
    if len(value) > MAX_TEXT_LENGTH:
        raise GenerationApiError(f"{field_name}が長すぎます。")
    return value or None


def _parse_interaction_id(value: object) -> str | None:
    """Validate the optional Phase 1 interaction id at the API boundary."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise GenerationApiError(
            "interaction_idは文字列で指定してください。",
            code="interaction_profile_schema_invalid",
        )
    interaction_id = value.strip()
    if not interaction_id:
        raise GenerationApiError(
            "interaction_idが見つかりません。",
            code="interaction_not_found",
        )
    try:
        resolve_phase1_interaction(interaction_id)
    except InteractionProfileError as exc:
        raise GenerationApiError(str(exc), code=exc.code) from exc
    return interaction_id


def _parse_explicit_tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise GenerationApiError("explicit_tagsは配列で指定してください。")
    if len(value) > MAX_EXPLICIT_TAGS:
        raise GenerationApiError("explicit_tagsが多すぎます。")
    result: list[str] = []
    for item in value:
        tag = _clean_text(item, "explicit_tags", allow_none=False)
        if tag:
            result.append(tag)
    return tuple(result)


def _parse_seed(value: object) -> int:
    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenerationApiError("seedは整数で指定してください。")
    if value < -1:
        raise GenerationApiError("seedは-1または0以上で指定してください。")
    return value


def _parse_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_GENERATION_COUNT:
        raise GenerationApiError(f"countは1～{MAX_GENERATION_COUNT}枚で指定してください。")
    return value


def _parse_variation(value: object, *, count: int) -> str:
    if value is None:
        return VARIATION_OFF
    if not isinstance(value, str) or value.casefold() not in {VARIATION_OFF, VARIATION_AUTO}:
        raise GenerationApiError("variationはoffまたはautoで指定してください。")
    mode = value.casefold()
    return VARIATION_OFF if count == 1 else mode


def _parse_variation_offset(value: object, *, count: int) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationApiError("variation_offsetが不正です。")
    if value + count > MAX_GENERATION_COUNT:
        raise GenerationApiError(
            f"variation_offsetとcountの合計は{MAX_GENERATION_COUNT}以下にしてください。"
        )
    return value


def _parse_reference(
    payload: Mapping[str, object],
    *,
    reference_upload_bytes: int,
) -> dict[str, object] | None:
    mode_value = payload.get("reference_mode", "none")
    if not isinstance(mode_value, str):
        raise GenerationApiError("reference_modeが不正です。")
    mode = mode_value.casefold().strip()
    if mode not in {"none", "composition", "pose", "edit"}:
        raise GenerationApiError("reference_modeはnone / composition / pose / editのいずれかで指定してください。")
    image_id = payload.get("reference_image_id")
    strength = payload.get("reference_strength")
    fit_value = payload.get("reference_fit", "contain")
    if not isinstance(fit_value, str) or fit_value.casefold() not in {"contain", "cover", "stretch"}:
        raise GenerationApiError("reference_fitはcontain / cover / stretchのいずれかで指定してください。")
    fit = fit_value.casefold()
    if strength is not None:
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 1.0:
            raise GenerationApiError("reference_strengthは0.0～1.0で指定してください。")
        strength = float(strength)
    if mode == "none":
        if image_id is not None:
            raise GenerationApiError("reference_mode=noneでは参考画像を指定できません。")
        return None
    if not isinstance(image_id, str) or not image_id.strip():
        raise GenerationApiError("reference_image_idを指定してください。")
    try:
        asset = load_reference_asset(image_id.strip())
    except (OSError, ValueError) as exc:
        raise GenerationApiError("指定された参考画像を利用できません。") from exc
    if len(asset.image_bytes) > reference_upload_bytes:
        raise GenerationApiError("参考画像が大きすぎます。")
    if strength is None:
        strength = DEFAULT_POSE_STRENGTH if mode == "pose" else DEFAULT_REFERENCE_STRENGTH
    # The mapping is internal to the adapter/service boundary.  The HTTP
    # request and metadata retain only the opaque id, never a path.
    return {
        "mode": mode,
        "image_id": asset.asset_id,
        "image_bytes": asset.image_bytes,
        "content_type": asset.content_type,
        "width": asset.width,
        "height": asset.height,
        "strength": strength,
        "fit": fit,
    }


def _parse_loras(value: object, *, model_family: str | None) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise GenerationApiError("lorasは配列で指定してください。")
    if len(value) > MAX_LORAS:
        raise GenerationApiError("LoRAは最大3件まで指定できます。")

    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) - _ALLOWED_LORA_KEYS:
            raise GenerationApiError("LoRAにはkey、lora_type、strengthだけ指定できます。")
        key = item.get("key")
        lora_type = item.get("lora_type")
        if not isinstance(key, str) or not key.strip():
            raise GenerationApiError("LoRA keyが不正です。")
        if not isinstance(lora_type, str):
            raise GenerationApiError("LoRA lora_typeが不正です。")
        try:
            normalized_type = normalize_lora_type(lora_type)
        except (TypeError, ValueError) as exc:
            raise GenerationApiError("LoRA lora_typeが不正です。") from exc
        strength = item.get("strength")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise GenerationApiError("LoRA strengthが不正です。")
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise GenerationApiError("LoRA strengthは0.0～1.0で指定してください。")
        result.append({"key": key.strip(), "lora_type": normalized_type, "strength": strength})

    try:
        resolve_saved_lora_selections(result, model_family=model_family)
    except (ValueError, TypeError) as exc:
        raise GenerationApiError("指定されたLoRAを利用できません。") from exc
    return tuple(result)


def _build_request(
    payload: Mapping[str, object],
    *,
    reference_upload_bytes: int = MAX_REFERENCE_IMAGE_BYTES,
) -> GenerationRequest:
    unknown = set(payload) - _ALLOWED_REQUEST_KEYS
    if unknown:
        raise GenerationApiError("サポートされていない生成項目が含まれています。")

    interaction_id = _parse_interaction_id(payload.get("interaction_id"))

    model_key = payload.get("model_key", "animagine")
    if not isinstance(model_key, str) or not model_key.strip():
        raise GenerationApiError("model_keyが不正です。")
    try:
        profile = get_image_generation_profile(model_key.strip())
    except ValueError as exc:
        raise GenerationApiError("指定されたモデルには対応していません。") from exc
    if not is_image_generation_profile_enabled(profile):
        raise GenerationApiError(f"{profile.display_name}は現在無効です。")

    mode_value = payload.get("mode", "t2i")
    if not isinstance(mode_value, str) or mode_value.casefold().strip() not in {"t2i", "edit"}:
        raise GenerationApiError("modeはt2iまたはeditで指定してください。")
    mode = mode_value.casefold().strip()
    if mode == "edit" and not profile.supports_edit:
        raise GenerationApiError("edit modeはQwen-Image 2.1でのみ利用できます。")

    rating = payload.get("rating", "auto")
    if not isinstance(rating, str) or rating.casefold() not in PROMPT_RATING_CHOICES:
        raise GenerationApiError("指定されたratingには対応していません。")
    rating = rating.casefold()
    if not profile.is_adult and rating not in {"auto", "safe"}:
        raise GenerationApiError("このratingはAdultモデルでのみ使用できます。")

    size = payload.get("size", profile.default_size)
    if not isinstance(size, str) or size.casefold() not in profile.supported_sizes:
        raise GenerationApiError("指定されたsizeには対応していません。")
    size = size.casefold()
    base_size_spec = IMAGE_SIZE_SPECS[size]
    if size == profile.default_size and profile.width and profile.height:
        size_spec = replace(base_size_spec, width=profile.width, height=profile.height)
    else:
        size_spec = base_size_spec

    try:
        multi_person_framing = validate_multi_person_framing(payload.get("multi_person_framing", "auto"))
    except ValueError as exc:
        raise GenerationApiError(str(exc)) from exc

    count = _parse_count(payload.get("count", 1))
    variation = _parse_variation(payload.get("variation"), count=count)
    variation_offset = _parse_variation_offset(payload.get("variation_offset"), count=count)
    if not profile.supports_bulk and count != 1:
        raise GenerationApiError(f"{profile.display_name}では複数枚生成は未対応です。")
    if not profile.supports_lora and payload.get("loras"):
        raise GenerationApiError(f"{profile.display_name}ではLoRAは未対応です。")
    reference_mode_value = str(payload.get("reference_mode") or "none").casefold().strip()
    if mode == "edit":
        if payload.get("reference_image_ids") is not None:
            raise GenerationApiError("Qwen-Image editでは参考画像は1枚だけ指定できます。")
        if reference_mode_value == "none" and payload.get("reference_image_id"):
            reference_mode_value = "edit"
        if reference_mode_value != "edit":
            raise GenerationApiError("Qwen-Image editではreference_mode=editと参考画像1枚が必要です。")
    elif not profile.supports_reference and (
        payload.get("reference_image_id") or reference_mode_value != "none"
    ):
        raise GenerationApiError(f"{profile.display_name}ではReference / Poseは未対応です。")
    if profile.engine == QWEN_IMAGE_ENGINE and variation != VARIATION_OFF:
        raise GenerationApiError(f"{profile.display_name}ではvariationは未対応です。")
    reference_payload = dict(payload)
    if mode == "edit" and "reference_mode" not in reference_payload:
        reference_payload["reference_mode"] = "edit"
    reference = _parse_reference(reference_payload, reference_upload_bytes=reference_upload_bytes)

    loras = _parse_loras(payload.get("loras"), model_family=profile.model_family)
    if reference is not None and loras:
        # Preserve the existing sd_control workflow contract: the IP-Adapter
        # reference workflow and LoRA workflow are not combined implicitly.
        raise GenerationApiError("Reference imageとLoRAの同時適用には対応していません。")
    generation_request = GenerationRequest(
        character_preset=_clean_text(payload.get("character_preset"), "character_preset"),
        freeform_character=_clean_text(payload.get("freeform_character"), "freeform_character"),
        explicit_tags=_parse_explicit_tags(payload.get("explicit_tags")),
        appearance_text=_clean_text(payload.get("appearance_text"), "appearance_text"),
        request_text=_clean_text(payload.get("request_text"), "request_text"),
        interaction_id=interaction_id,
        model_key=model_key.strip(),
        mode=mode,
        rating=rating,
        loras=loras,
        size_preset=size,
        multi_person_framing=multi_person_framing,
        width=size_spec.width,
        height=size_spec.height,
        explicit_generation_fields=frozenset({"width", "height"}),
        steps=int(profile.default_steps if profile.default_steps is not None else 20),
        cfg=float(profile.default_cfg if profile.default_cfg is not None else 7.0),
        sampler=str(profile.default_sampler or "Euler a"),
        count=count,
        variation=variation,
        variation_offset=variation_offset,
        seed=_parse_seed(payload.get("seed")),
        source="generation_api",
        reference=reference,
    )
    try:
        generation_request.validate()
    except PromptGeneratorError as exc:
        raise GenerationApiError(str(exc)) from exc
    return generation_request


def _build_overnight_child_request(
    payload: Mapping[str, object],
    *,
    reference_upload_bytes: int = MAX_REFERENCE_IMAGE_BYTES,
) -> GenerationRequest:
    """Build a count=1 child while preserving the parent's variation mode."""

    request = _build_request(payload, reference_upload_bytes=reference_upload_bytes)
    overnight_variation = _parse_variation(payload.get("variation"), count=2)
    return replace(request, count=1, variation=overnight_variation, variation_offset=0)


def _safe_error_summary(exc: BaseException) -> str:
    """Return a short client-safe error without traceback or absolute paths."""

    text = str(exc).strip()
    text = _ABSOLUTE_PATH_RE.sub("<path>", text)
    if not text or "traceback" in text.casefold():
        return "生成に失敗しました。"
    return text[:240]


def _safe_image_metadata(result: object) -> dict[str, Any]:
    images = tuple(getattr(result, "images", ()) or ())
    rows: list[dict[str, Any]] = []
    for image in images:
        path = getattr(image, "path", None)
        rows.append(
            {
                "batch_index": getattr(image, "batch_index", None),
                "seed": getattr(image, "seed", None),
                "filename": Path(path).name if path is not None else None,
            }
        )
    return {
        "image_count": len(rows),
        "images": rows,
        "image_identifiers": [row["filename"] for row in rows if row.get("filename")],
        "requested_seed": getattr(result, "requested_seed", None),
        "actual_seeds": [row["seed"] for row in rows],
    }


def _build_gallery_metadata(
    job_id: str,
    request: GenerationRequest,
    prepared: PreparedGeneration,
    *,
    variation_index: int = 1,
    variation_tags: tuple[str, ...] = (),
    metadata_overrides: Mapping[str, Any] | None = None,
    requested_seed_override: object = _UNSET,
) -> dict[str, Any]:
    """Build the sidecar contract without exposing local paths or secrets."""

    options = prepared.options
    loras = [
        {
            "key": str(selection.key),
            "display_name": str(selection.display_name),
            "lora_type": str(selection.lora_type),
            "strength": float(selection.strength),
            "trigger": str(selection.trigger or ""),
        }
        for selection in options.loras
    ]
    tag_provenance = [
        {
            "tag": str(item.tag),
            "source": str(item.source),
            "category": str(item.category or ""),
        }
        for item in prepared.resolved.provenance
        if str(item.tag or "").strip()
    ]
    suppressed_tags = [
        item.to_dict()
        for item in prepared.resolved.suppressed_tags
        if str(item.tag or "").strip()
    ]
    prompt_conflicts = [
        item.to_dict()
        for item in prepared.resolved.prompt_conflicts
        if str(item.category or "").strip()
    ]
    metadata = {
        "source": "generation_api",
        "job_id": job_id,
        "parent_job_id": job_id,
        "variation_index": variation_index,
        "variation_tags": list(variation_tags),
        "variation": request.variation,
        "requested_variation": request.variation,
        "variation_offset": request.variation_offset,
        "model": str(options.model_key or request.model_key),
        "mode": str(getattr(request, "mode", "t2i") or "t2i"),
        "engine": str(getattr(options, "engine", "sdxl_tag") or "sdxl_tag"),
        "workflow": str(getattr(options, "workflow_key", "") or ""),
        "workflow_hash": str(getattr(options, "workflow_hash", "") or ""),
        "width": int(getattr(options, "width", request.width)),
        "height": int(getattr(options, "height", request.height)),
        "steps": int(getattr(options, "steps", request.steps)),
        "sampler": str(getattr(options, "sampler", request.sampler) or ""),
        "scheduler": str(getattr(options, "scheduler", "normal") or ""),
        "cfg": float(getattr(options, "cfg", request.cfg)),
        "model_components": dict(getattr(options, "model_components", {}) or {}),
        "rating": request.rating,
        "positive": options.prompt,
        "prompt": options.prompt,
        "negative": options.negative_prompt,
        "negative_prompt": options.negative_prompt,
        "requested_seed": request.seed,
        "loras": loras,
        "size": {
            "preset": request.size_preset or "",
            "width": request.width,
            "height": request.height,
        },
        "character_preset": request.character_preset or "",
        "freeform_character": request.freeform_character or "",
        "explicit_tags": list(request.explicit_tags),
        "appearance_text": request.appearance_text or "",
        "request_text": request.request_text or "",
        "reference_mode": str((request.reference or {}).get("mode") or "none"),
        "reference_image_count": 1 if request.mode == "edit" and request.reference else 0,
        "reference_image_id": str((request.reference or {}).get("image_id") or ""),
        "reference_strength": (request.reference or {}).get("strength"),
        "reference_fit": str((request.reference or {}).get("fit") or "contain"),
        "tag_provenance": tag_provenance,
        "suppressed_tags": suppressed_tags,
        "prompt_conflicts": prompt_conflicts,
        "people_semantics": prepared.resolved.people_semantics.to_dict(),
        "multi_person_framing_requested": prepared.resolved.multi_person_framing_requested,
        "multi_person_framing_effective": prepared.resolved.multi_person_framing_effective,
        "framing_semantics": prepared.resolved.framing_semantics.to_dict(),
        "multi_person_framing_source": prepared.resolved.multi_person_framing_source,
    }
    if requested_seed_override is not _UNSET:
        metadata["requested_seed"] = requested_seed_override
    if metadata_overrides:
        metadata.update(dict(metadata_overrides))
    interaction = interaction_metadata(request.interaction_id)
    if interaction is not None:
        metadata["interaction"] = interaction
    return metadata


def _job_response(job: _Job) -> dict[str, Any]:
    output = dict(job.output_metadata or {})
    output.setdefault("image_count", len(job.output_items))
    output.setdefault("images", list(job.output_items))
    output.setdefault("image_identifiers", [item.get("filename") for item in job.output_items if item.get("filename")])
    output.setdefault("gallery_job_id", job.job_id)
    output["total_count"] = job.total_count
    output["completed_count"] = job.completed_count
    output["failed_count"] = job.failed_count
    return {
        "job_id": job.job_id,
        "state": job.state,
        "progress": job.progress,
        "error_summary": job.error_summary,
        "total_count": job.total_count,
        "completed_count": job.completed_count,
        "failed_count": job.failed_count,
        "current_index": job.completed_count + job.failed_count,
        "variation_offset": job.request.variation_offset,
        "variation": job.variation,
        "output": output,
    }


def _bulk_response(job: _BulkJob) -> dict[str, Any]:
    completed = int(job.completed_count)
    failed = int(job.failed_count)
    return {
        "bulk_id": job.bulk_id,
        "job_id": job.bulk_id,
        "state": job.state,
        "progress": job.progress,
        "requested_total": job.requested_total,
        "total_count": job.requested_total,
        "current_index": job.current_index,
        "completed_count": completed,
        "failed_count": failed,
        "remaining_count": max(job.requested_total - completed - failed, 0),
        "stop_requested": job.stop_requested,
        "pause_requested": job.pause_requested,
        "selection_mode": job.selection_mode,
        "variation": job.variation,
        "selection_state": job.selection_state,
        "selection_error": job.selection_error,
        "error_summary": job.error_summary,
        "errors": job.errors[:4],
        "output": {
            "image_count": len(job.output_items),
            "images": list(job.output_items),
            "image_identifiers": [item.get("filename") for item in job.output_items if item.get("filename")],
            "selected_images": list(job.selected_items),
            "gallery_job_id": job.bulk_id,
        },
    }


class GenerationApi:
    """aiohttp adapter with an in-memory, non-blocking job lifecycle."""

    def __init__(
        self,
        *,
        service: GenerationService | None = None,
        config: GenerationApiConfig | None = None,
        tag_store: DanbooruTagStore | None = None,
        tag_interpreter: object | None = None,
        overnight_db_path: str | Path | None = None,
    ) -> None:
        self.config = config or GenerationApiConfig()
        self.service = service or GenerationService()
        self.tag_store = tag_store or DanbooruTagStore()
        self.tag_interpreter = tag_interpreter
        self.jobs = _JobStore()
        self.bulk_jobs = _BulkJobStore()
        self.overnight_jobs = OvernightJobStore(overnight_db_path)
        self._overnight_task: asyncio.Task[None] | None = None
        self._generation_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self.app = web.Application(
            client_max_size=max(self.config.max_body_bytes, self.config.reference_upload_bytes)
        )
        self.app.add_routes(
            [
                web.get("/api/generation/health", self.health),
                web.get("/api/generation/options", self.options),
                web.get("/api/generation/interaction-profiles", self.interaction_profiles),
                web.get("/api/generation/tags/search", self.search_tags),
                web.post("/api/generation/references", self.upload_reference),
                web.post("/api/generation/jobs", self.create_job),
                web.get("/api/generation/jobs/{job_id}", self.get_job),
                web.post("/api/generation/bulk-jobs", self.create_bulk_job),
                web.get("/api/generation/bulk-jobs/{bulk_id}", self.get_bulk_job),
                web.post("/api/generation/bulk-jobs/{bulk_id}/stop", self.stop_bulk_job),
                web.post("/api/generation/overnight-jobs", self.create_overnight_job),
                web.get("/api/generation/overnight-jobs/{overnight_id}", self.get_overnight_job),
                web.post("/api/generation/overnight-jobs/{overnight_id}/pause", self.pause_overnight_job),
                web.post("/api/generation/overnight-jobs/{overnight_id}/resume", self.resume_overnight_job),
                web.post("/api/generation/overnight-jobs/{overnight_id}/stop", self.stop_overnight_job),
                web.post("/api/generation/overnight-jobs/{overnight_id}/reselect", self.reselect_overnight_job),
            ]
        )
        self.app.on_cleanup.append(self._cleanup)
        self.app.on_startup.append(self._recover_overnight_jobs)
        self.app[GENERATION_API_APP_KEY] = self

    async def _recover_overnight_jobs(self, _app: web.Application) -> None:
        """Require manual recovery after a Backend restart.

        A process restart must never silently replay an uncertain slot. Jobs
        with a durable in-flight marker are therefore kept in ``paused_error``
        so an explicit stop is required; jobs that were between slots become
        manually resumable ``paused`` jobs.
        """

        for record in self.overnight_jobs.list_active():
            state = str(record.get("state") or "")
            if state == "paused":
                continue
            timing = dict(record.get("timing") or {})
            timing["backend_restart_recovered_at"] = utc_now_iso()
            if record.get("in_flight"):
                timing["backend_restart_recovery_mode"] = "paused_error_in_flight"
                self.overnight_jobs.update(
                    record["job_id"],
                    state="paused_error",
                    selection_state="paused_error",
                    error_summary="Backend再起動前に実行中slotがあり、重複生成を避けるため停止しました。",
                    timing=timing,
                )
                logger.warning(
                    "Overnight job paused after restart with in-flight slot: overnight_id=%s",
                    record["job_id"],
                )
                continue
            timing["backend_restart_recovery_mode"] = "paused_manual_resume"
            self.overnight_jobs.update(
                record["job_id"],
                state="paused",
                selection_state="paused",
                pause_requested=True,
                error_summary="Backend再起動後は自動再開せず、手動再開を待機しています。",
                timing=timing,
            )
            logger.warning(
                "Overnight job paused after restart; manual resume required: overnight_id=%s",
                record["job_id"],
            )

    async def _cleanup(self, _app: web.Application) -> None:
        if self._overnight_task is not None:
            self._overnight_task.cancel()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def health(self, _request: web.Request) -> web.Response:
        appearance_status = get_appearance_resolver_status()
        return web.json_response(
            {
                "service_ready": True,
                "lm_studio_configured": _is_configured(
                    "LMSTUDIO_API_URL", "LM_STUDIO_API_URL", "LOCAL_LLM_API_URL"
                ),
                "comfyui_url_configured": _is_configured("SD_API_URL"),
                "appearance_resolver": appearance_status,
            }
        )

    async def options(self, _request: web.Request) -> web.Response:
        models = [
            {
                "key": profile.model_key,
                "display_name": profile.display_name,
                "adult": profile.is_adult,
                "model_family": profile.model_family,
                "engine": profile.engine,
                "supported_sizes": list(profile.supported_sizes),
                "capabilities": {
                    "lora": profile.supports_lora,
                    "reference": profile.supports_reference,
                    "pose": profile.supports_pose,
                    "bulk": profile.supports_bulk,
                    "overnight": profile.supports_overnight,
                },
            }
            for profile in enabled_image_generation_profiles()
        ]
        lora_catalog: dict[str, list[dict[str, object]]] = {lora_type: [] for lora_type in LORA_TYPES}
        for key, style in sorted(COMFY_STYLES.items(), key=lambda item: item[1].display_name.casefold()):
            if style.lora_type not in lora_catalog or not model_family_compatible(style.model_family, "sdxl"):
                continue
            lora_catalog[style.lora_type].append(
                {
                    "key": key,
                    "display_name": style.display_name,
                    "default_strength": style.default_strength,
                }
            )
        character_presets = [
            {
                "key": entry.character_tag,
                "display_name": entry.display_name,
                "series": entry.series_name,
            }
            for entry in list_characters()
        ]
        return web.json_response(
            {
                "models": models,
                "ratings": list(PROMPT_RATING_CHOICES),
                "sizes": [
                    {"key": spec.key, "label": spec.label, "width": spec.width, "height": spec.height}
                    for spec in IMAGE_SIZE_SPECS.values()
                ],
                "model_sizes": {
                    profile.model_key: [
                        {
                            "key": size_key,
                            "label": IMAGE_SIZE_SPECS[size_key].label,
                            "width": (
                                profile.width
                                if size_key == profile.default_size and profile.width
                                else IMAGE_SIZE_SPECS[size_key].width
                            ),
                            "height": (
                                profile.height
                                if size_key == profile.default_size and profile.height
                                else IMAGE_SIZE_SPECS[size_key].height
                            ),
                        }
                        for size_key in profile.supported_sizes
                    ]
                    for profile in enabled_image_generation_profiles()
                },
                "multi_person_framing": [
                    {"key": key, "label": MULTI_PERSON_FRAMING_LABELS[key]}
                    for key in MULTI_PERSON_FRAMING_CHOICES
                ],
                "character_presets": character_presets,
                "loras": lora_catalog,
                "reference": {
                    "modes": ["none", "composition", "pose"],
                    "fit_modes": ["contain", "cover", "stretch"],
                    "default_strengths": {
                        "composition": DEFAULT_REFERENCE_STRENGTH,
                        "pose": DEFAULT_POSE_STRENGTH,
                    },
                    "max_upload_bytes": self.config.reference_upload_bytes,
                },
            }
        )

    async def interaction_profiles(self, request: web.Request) -> web.Response:
        """Return only Phase 1 profiles safe for Gallery selection."""

        released = request.query.get("released", "production_candidate")
        return web.json_response(list_phase1_profiles(released=released))

    async def search_tags(self, request: web.Request) -> web.Response:
        query = request.query.get("q", "")
        if len(query) > TAG_SEARCH_MAX_QUERY_LENGTH:
            return _json_error("検索語が長すぎます。", 400)
        raw_limit = request.query.get("limit", str(TAG_SEARCH_DEFAULT_LIMIT))
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return _json_error("limitが不正です。", 400)
        if limit < 1:
            return _json_error("limitは1以上で指定してください。", 400)
        category = request.query.get("category", TAG_SEARCH_DEFAULT_CATEGORY).strip().casefold()
        if category not in TAG_SEARCH_CATEGORIES:
            return _json_error("categoryが不正です。", 400)
        try:
            result = await asyncio.to_thread(
                search_tag_candidates,
                query,
                limit=min(limit, TAG_SEARCH_MAX_LIMIT),
                category=category,
                store=self.tag_store,
                interpreter=self.tag_interpreter,
            )
        except TagSearchError as exc:
            return _json_error(str(exc), 400)
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            logger.warning("Danbooru tag search unavailable: error_type=%s", type(exc).__name__)
            return _json_error("タグ検索を利用できません。", 503)
        return web.json_response(result)

    def _overnight_is_active(self) -> bool:
        task_active = self._overnight_task is not None and not self._overnight_task.done()
        return task_active or bool(self.overnight_jobs.list_active())

    def _reject_if_overnight_active(self) -> None:
        if self._overnight_is_active():
            raise GenerationApiError("夜間生成実行中は通常生成と大量生成を開始できません。")

    @staticmethod
    def _overnight_requires_lm_studio(_snapshot: Mapping[str, object]) -> bool:
        """Return whether the current Overnight execution path resolves prompts.

        Overnight currently stores the public request fields and resolves them
        inside ``_run_overnight_job`` before the first slot.  There is no
        accepted resolved-prompt snapshot field, so LM Studio is required for
        every current Overnight request.  This explicit predicate keeps the
        exception point clear if a future trusted snapshot path is added.
        """

        return True

    async def _preflight_overnight_dependencies(self, snapshot: Mapping[str, object]) -> None:
        """Reject an obviously unstartable Overnight request before persistence."""

        comfy_ready = await asyncio.to_thread(is_api_available)
        if not comfy_ready:
            raise GenerationDependencyError("ComfyUIが停止しています。")
        if self._overnight_requires_lm_studio(snapshot):
            lm_ready = await asyncio.to_thread(_lm_studio_ready)
            if not lm_ready:
                raise GenerationDependencyError("LM Studioが停止しています。")

    @staticmethod
    def _shutdown_requested(record: Mapping[str, Any]) -> bool:
        timing = record.get("timing") or {}
        return bool(isinstance(timing, Mapping) and timing.get("shutdown_after_completion"))

    @staticmethod
    def _overnight_shutdown_eligible(
        record: Mapping[str, Any], *, queue_empty: bool, gallery_reflected: bool
    ) -> tuple[bool, str]:
        """Prove strict terminal success before scheduling any power action."""

        if record.get("state") != "completed":
            return False, "job_state_not_completed"
        if record.get("selection_state") != "completed":
            return False, "final_selection_not_completed"
        if int(record.get("generated_count") or 0) < int(record.get("total_count") or 0):
            return False, "generation_incomplete"
        if int(record.get("failed_count") or 0) != 0:
            return False, "generation_failed"
        if record.get("in_flight"):
            return False, "in_flight_slot_present"
        if not record.get("final_selected"):
            return False, "final_selection_empty"
        rounds = list(record.get("rounds") or [])
        if len(rounds) != int(record.get("total_rounds") or 0):
            return False, "round_records_incomplete"
        completed_slots = 0
        for round_data in rounds:
            if not isinstance(round_data, Mapping):
                return False, "round_record_invalid"
            if round_data.get("selection_state") != "completed":
                return False, "round_selection_incomplete"
            slots = list(round_data.get("slots") or [])
            if not slots or any(slot.get("status") != "completed" for slot in slots):
                return False, "generated_slots_incomplete"
            completed_slots += len(slots)
        if completed_slots != int(record.get("total_count") or 0):
            return False, "generated_slots_count_mismatch"
        if not gallery_reflected:
            return False, "gallery_reflection_incomplete"
        if not queue_empty:
            return False, "comfyui_queue_not_empty"
        return True, ""

    def _overnight_gallery_reflected(self, record: Mapping[str, Any]) -> bool:
        """Verify every completed slot still resolves inside the Gallery root."""

        gallery_root = get_gallery_images_root()
        if gallery_root is None:
            return False
        try:
            root = gallery_root.expanduser().resolve()
        except OSError:
            return False
        for round_data in record.get("rounds") or []:
            for slot in round_data.get("slots") or []:
                if slot.get("status") != "completed":
                    return False
                filename = (slot.get("image") or {}).get("filename")
                path = self._overnight_path_for_filename(filename)
                if path is None:
                    return False
                try:
                    path.resolve().relative_to(root)
                except (OSError, ValueError):
                    return False
        return True

    def _set_overnight_phase(self, overnight_id: str, phase: str) -> None:
        """Persist a bounded phase marker for diagnosing unattended stalls."""

        record = self.overnight_jobs.get(overnight_id)
        if record is None:
            return
        timing = dict(record.get("timing") or {})
        timing["worker_phase"] = phase
        timing["worker_phase_at"] = utc_now_iso()
        self.overnight_jobs.update(overnight_id, timing=timing)

    async def _maybe_schedule_overnight_shutdown(self, overnight_id: str) -> None:
        """Schedule at most one shutdown after a durable successful completion."""

        record = self.overnight_jobs.get(overnight_id)
        if record is None or not self._shutdown_requested(record):
            return
        timing = dict(record.get("timing") or {})
        timing.setdefault("shutdown_job_id", overnight_id)
        if timing.get("shutdown_scheduled_at") or timing.get("shutdown_status") == "scheduled":
            return
        queue_empty = await asyncio.to_thread(is_comfy_queue_empty)
        eligible, reason = self._overnight_shutdown_eligible(
            record,
            queue_empty=queue_empty,
            gallery_reflected=self._overnight_gallery_reflected(record),
        )
        if not eligible:
            timing["shutdown_status"] = "blocked"
            timing["shutdown_blocked_reason"] = reason
            timing["shutdown_failure_reason"] = reason
            self.overnight_jobs.update(overnight_id, timing=timing)
            logger.warning(
                "Overnight shutdown not scheduled: job_id=%s phase=eligibility reason=%s",
                overnight_id,
                reason,
            )
            return
        try:
            shutdown_metadata = await asyncio.to_thread(
                schedule_overnight_shutdown,
                DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - shutdown must never change job success
            attempted_at = timing.get("shutdown_attempted_at") or utc_now_iso()
            timing["shutdown_status"] = "failed"
            timing["shutdown_job_id"] = overnight_id
            timing["shutdown_attempted_at"] = attempted_at
            timing["shutdown_command"] = build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS)
            timing["shutdown_failed_at"] = utc_now_iso()
            timing["shutdown_failure_reason"] = f"{type(exc).__name__}: {exc}"
            self.overnight_jobs.update(overnight_id, timing=timing)
            logger.exception("Could not schedule Overnight shutdown: job_id=%s phase=command", overnight_id)
            return
        timing.update(shutdown_metadata)
        timing["shutdown_job_id"] = overnight_id
        self.overnight_jobs.update(overnight_id, timing=timing)
        if timing.get("shutdown_status") == "scheduled":
            logger.warning(
                "Overnight shutdown requested: job_id=%s attempted_at=%s scheduled_at=%s return_code=%s",
                overnight_id,
                timing.get("shutdown_attempted_at", ""),
                timing.get("shutdown_scheduled_at", ""),
                timing.get("shutdown_return_code"),
            )
        else:
            logger.error(
                "Overnight shutdown failed: job_id=%s phase=command return_code=%s reason=%s",
                overnight_id,
                timing.get("shutdown_return_code"),
                timing.get("shutdown_failure_reason", ""),
            )

    async def create_job(self, request: web.Request) -> web.Response:
        try:
            if request.content_length is not None and request.content_length > self.config.max_body_bytes:
                return _json_error("リクエストが大きすぎます。", 413)
            try:
                payload = await request.json(loads=json.loads)
            except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise GenerationApiError("JSON bodyが不正です。") from exc
            if not isinstance(payload, dict):
                raise GenerationApiError("JSON objectを指定してください。")
            generation_request = _build_request(
                payload,
                reference_upload_bytes=self.config.reference_upload_bytes,
            )
            self._reject_if_overnight_active()
        except GenerationApiError as exc:
            status = 409 if "夜間生成実行中" in str(exc) else 400
            return _json_error(str(exc), status, code=exc.code)
        except web.HTTPRequestEntityTooLarge:
            return _json_error("リクエストが大きすぎます。", 413)

        job = self.jobs.create(generation_request)
        job.total_count = generation_request.count
        job.variation = generation_request.variation
        task = asyncio.create_task(self._run_job(job.job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return web.json_response(_job_response(job), status=202)

    async def create_bulk_job(self, request: web.Request) -> web.Response:
        try:
            if request.content_length is not None and request.content_length > self.config.max_body_bytes:
                return _json_error("リクエストが大きすぎます。", 413)
            try:
                payload = await request.json(loads=json.loads)
            except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise GenerationApiError("JSON bodyが不正です。") from exc
            if not isinstance(payload, dict):
                raise GenerationApiError("JSON objectを指定してください。")
            try:
                batch_profile = get_image_generation_profile(str(payload.get("model_key", "animagine")))
            except ValueError:
                batch_profile = None
            if batch_profile is not None and not batch_profile.supports_bulk:
                raise GenerationApiError(f"{batch_profile.display_name}ではBulkは未対応です。")
            raw_total = payload.get("total_count")
            if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total not in ALLOWED_BULK_GENERATION_COUNTS:
                raise GenerationApiError("大量生成は16、24、32枚だけ指定できます。")
            selection_mode = payload.get("selection_mode", "none")
            if not isinstance(selection_mode, str) or selection_mode not in BULK_SELECTION_MODES:
                raise GenerationApiError("選別モードが不正です。")
            # The inner request is deliberately count=1, but bulk variation
            # must retain the caller's mode instead of being normalized to
            # ``off`` by the normal one-image parser.
            bulk_variation = _parse_variation(payload.get("variation"), count=2)
            generation_payload = dict(payload)
            generation_payload.pop("total_count", None)
            generation_payload.pop("selection_mode", None)
            # The normal request remains a one-image request inside each
            # sequential bulk slot.  The parent total is tracked separately.
            generation_payload["count"] = 1
            generation_payload["variation_offset"] = 0
            generation_request = _build_request(
                generation_payload,
                reference_upload_bytes=self.config.reference_upload_bytes,
            )
            self._reject_if_overnight_active()
            job = self.bulk_jobs.create(generation_request, raw_total, selection_mode, bulk_variation)
        except GenerationApiError as exc:
            status = 409 if "同時に1件" in str(exc) or "夜間生成実行中" in str(exc) else 400
            return _json_error(str(exc), status, code=exc.code)
        except web.HTTPRequestEntityTooLarge:
            return _json_error("リクエストが大きすぎます。", 413)

        task = asyncio.create_task(self._run_bulk_job(job.bulk_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return web.json_response(_bulk_response(job), status=202)

    async def upload_reference(self, request: web.Request) -> web.Response:
        """Accept one validated image and return only its opaque asset id."""

        if request.content_length is not None and request.content_length > self.config.reference_upload_bytes:
            return _json_error("参考画像が大きすぎます。", 413)
        if not request.content_type.startswith("multipart/"):
            return _json_error("multipart/form-dataが必要です。", 415)
        try:
            reader = await request.multipart()
            part = await reader.next()
            if part is None or part.name != "file":
                raise GenerationApiError("fileフィールドに参考画像を指定してください。")
            content_type = part.headers.get("Content-Type", "")
            chunks: list[bytes] = []
            total = 0
            while not part.at_eof():
                chunk = await part.read_chunk(size=min(1024 * 1024, self.config.reference_upload_bytes + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.config.reference_upload_bytes:
                    break
            image_bytes = b"".join(chunks)
            if len(image_bytes) > self.config.reference_upload_bytes:
                raise GenerationApiError("参考画像が大きすぎます。")
            if await reader.next() is not None:
                raise GenerationApiError("参考画像は1枚だけ指定してください。")
            asset = save_reference_asset(image_bytes, content_type)
        except GenerationApiError as exc:
            return _json_error(str(exc), 400)
        except web.HTTPRequestEntityTooLarge:
            return _json_error("参考画像が大きすぎます。", 413)
        except (OSError, ValueError) as exc:
            return _json_error(str(exc), 400)
        return web.json_response(
            {
                "reference_image_id": asset.asset_id,
                "content_type": asset.content_type,
                "width": asset.width,
                "height": asset.height,
            },
            status=201,
        )

    async def get_job(self, request: web.Request) -> web.Response:
        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return _json_error("jobが見つかりません。", 404)
        return web.json_response(_job_response(job))

    async def get_bulk_job(self, request: web.Request) -> web.Response:
        job = self.bulk_jobs.get(request.match_info["bulk_id"])
        if job is None:
            return _json_error("大量生成jobが見つかりません。", 404)
        return web.json_response(_bulk_response(job))

    async def stop_bulk_job(self, request: web.Request) -> web.Response:
        job = self.bulk_jobs.get(request.match_info["bulk_id"])
        if job is None:
            return _json_error("大量生成jobが見つかりません。", 404)
        if job.state in {"completed", "failed", "stopped"}:
            return web.json_response(_bulk_response(job))
        self.bulk_jobs.update(job.bulk_id, stop_requested=True)
        return web.json_response(_bulk_response(job), status=202)

    def _start_overnight_task(self, overnight_id: str) -> None:
        if self._overnight_task is not None and not self._overnight_task.done():
            return
        task = asyncio.create_task(self._run_overnight_job(overnight_id))
        self._overnight_task = task
        self._tasks.add(task)

        def _finished(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if self._overnight_task is completed:
                self._overnight_task = None

        task.add_done_callback(_finished)

    def _start_overnight_reselection_task(self, overnight_id: str) -> None:
        if self._overnight_task is not None and not self._overnight_task.done():
            return
        task = asyncio.create_task(self._recover_overnight_selection(overnight_id))
        self._overnight_task = task
        self._tasks.add(task)

        def _finished(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if self._overnight_task is completed:
                self._overnight_task = None

        task.add_done_callback(_finished)

    async def create_overnight_job(self, request: web.Request) -> web.Response:
        try:
            if request.content_length is not None and request.content_length > self.config.max_body_bytes:
                return _json_error("リクエストが大きすぎます。", 413)
            payload = await request.json(loads=json.loads)
            if not isinstance(payload, dict):
                raise GenerationApiError("JSON objectを指定してください。")
            try:
                overnight_profile = get_image_generation_profile(str(payload.get("model_key", "animagine")))
            except ValueError:
                overnight_profile = None
            if overnight_profile is not None and not overnight_profile.supports_overnight:
                raise GenerationApiError(f"{overnight_profile.display_name}ではOvernightは未対応です。")
            total_count = payload.get("total_count")
            if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count not in OVERNIGHT_TOTALS:
                raise GenerationApiError("夜間生成は128、256、512枚だけ指定できます。")
            shutdown_after_completion = payload.get("shutdown_after_completion", False)
            if not isinstance(shutdown_after_completion, bool):
                raise GenerationApiError("shutdown_after_completionはtrueまたはfalseで指定してください。")
            self._reject_if_overnight_active()
            snapshot = dict(payload)
            snapshot.pop("total_count", None)
            snapshot.pop("selection_mode", None)
            snapshot.pop("shutdown_after_completion", None)
            snapshot["count"] = 1
            snapshot["variation_offset"] = 0
            _build_request(snapshot, reference_upload_bytes=self.config.reference_upload_bytes)
            await self._preflight_overnight_dependencies(snapshot)
            record = self.overnight_jobs.create(
                snapshot,
                total_count,
                shutdown_after_completion=shutdown_after_completion,
            )
        except GenerationDependencyError as exc:
            return _json_error(str(exc), 503)
        except (GenerationApiError, OvernightJobError) as exc:
            status = 409 if "同時に1件" in str(exc) or "実行中" in str(exc) else 400
            return _json_error(str(exc), status)
        except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return _json_error("JSON bodyが不正です。", 400)
        except web.HTTPRequestEntityTooLarge:
            return _json_error("リクエストが大きすぎます。", 413)
        self._start_overnight_task(record["job_id"])
        return web.json_response(overnight_response(record), status=202)

    async def get_overnight_job(self, request: web.Request) -> web.Response:
        record = self.overnight_jobs.get(request.match_info["overnight_id"])
        if record is None:
            return _json_error("夜間生成jobが見つかりません。", 404)
        return web.json_response(overnight_response(record))

    async def pause_overnight_job(self, request: web.Request) -> web.Response:
        try:
            record = self.overnight_jobs.get(request.match_info["overnight_id"])
            if record is None:
                return _json_error("夜間生成jobが見つかりません。", 404)
            if record["state"] in OVERNIGHT_TERMINAL_STATES:
                return web.json_response(overnight_response(record))
            state = "paused" if self._overnight_task is None or self._overnight_task.done() else record["state"]
            timing = dict(record.get("timing") or {})
            timing.setdefault("pause_requested_at", utc_now_iso())
            record = self.overnight_jobs.update(
                record["job_id"], pause_requested=True, state=state, timing=timing,
            )
            return web.json_response(overnight_response(record), status=202)
        except OvernightJobError as exc:
            return _json_error(str(exc), 404)

    async def resume_overnight_job(self, request: web.Request) -> web.Response:
        try:
            record = self.overnight_jobs.get(request.match_info["overnight_id"])
            if record is None:
                return _json_error("夜間生成jobが見つかりません。", 404)
            if record["state"] in OVERNIGHT_TERMINAL_STATES:
                return web.json_response(overnight_response(record))
            if self._overnight_task is not None and not self._overnight_task.done():
                return _json_error("夜間生成はすでに実行中です。", 409)
            timing = dict(record.get("timing") or {})
            timing["last_resumed_at"] = utc_now_iso()
            timing["resume_count"] = int(timing.get("resume_count") or 0) + 1
            record = self.overnight_jobs.update(
                record["job_id"], pause_requested=False, stop_requested=False, state="running", timing=timing,
            )
            self._start_overnight_task(record["job_id"])
            return web.json_response(overnight_response(record), status=202)
        except OvernightJobError as exc:
            return _json_error(str(exc), 404)

    async def stop_overnight_job(self, request: web.Request) -> web.Response:
        try:
            record = self.overnight_jobs.get(request.match_info["overnight_id"])
            if record is None:
                return _json_error("夜間生成jobが見つかりません。", 404)
            if record["state"] in OVERNIGHT_TERMINAL_STATES:
                return web.json_response(overnight_response(record))
            state = "stopping" if self._overnight_task is not None and not self._overnight_task.done() else "stopped"
            timing = dict(record.get("timing") or {})
            timing.setdefault("stop_requested_at", utc_now_iso())
            if (
                timing.get("shutdown_after_completion")
                and not timing.get("shutdown_scheduled_at")
                and not timing.get("shutdown_cancelled_at")
            ):
                timing["shutdown_status"] = "cancelled"
                timing["shutdown_cancelled_at"] = timing["stop_requested_at"]
                timing["shutdown_failure_reason"] = "job_stop_requested"
            if state == "stopped":
                timing.setdefault("stopped_at", utc_now_iso())
            record = self.overnight_jobs.update(
                record["job_id"], stop_requested=True, state=state,
                timing=timing,
                finished_at=utc_now_iso() if state == "stopped" else record.get("finished_at") or "",
            )
            return web.json_response(overnight_response(record), status=202)
        except OvernightJobError as exc:
            return _json_error(str(exc), 404)

    async def reselect_overnight_job(self, request: web.Request) -> web.Response:
        """Re-run selection for an existing terminal job without generation."""

        record = self.overnight_jobs.get(request.match_info["overnight_id"])
        if record is None:
            return _json_error("夜間生成jobが見つかりません。", 404)
        generated_count = int(record.get("generated_count") or 0)
        total_count = int(record.get("total_count") or 0)
        fully_generated = total_count > 0 and generated_count >= total_count and not record.get("in_flight")
        if record["state"] not in OVERNIGHT_TERMINAL_STATES and not (
            record["state"] == "paused" and fully_generated
        ):
            return _json_error("実行中の夜間生成は再選別できません。", 409)
        if generated_count < 1:
            return _json_error("再選別できる生成画像がありません。", 400)
        if self._overnight_task is not None and not self._overnight_task.done():
            return _json_error("夜間生成の別処理が実行中です。", 409)
        timing = dict(record.get("timing") or {})
        timing["selection_recovery_requested_at"] = utc_now_iso()
        record = self.overnight_jobs.update(
            record["job_id"],
            state="selecting",
            selection_state="recovery_queued",
            error_summary="",
            timing=timing,
        )
        self._start_overnight_reselection_task(record["job_id"])
        return web.json_response(overnight_response(record), status=202)

    @staticmethod
    def _overnight_error_retryable(exc: BaseException) -> bool:
        text = str(exc).casefold()
        deterministic = (
            "不正", "invalid", "reference", "lora", "model", "モデル", "設定", "prompt",
            "seed", "workflow file", "見つかりません",
        )
        return not any(token in text for token in deterministic)

    @staticmethod
    def _overnight_round(record: Mapping[str, Any], round_index: int) -> dict[str, Any]:
        for item in record.get("rounds") or []:
            if isinstance(item, dict) and int(item.get("round_index") or 0) == round_index:
                return item
        return {
            "round_index": round_index,
            "generation_started_at": "",
            "generation_finished_at": "",
            "selection_started_at": "",
            "selection_finished_at": "",
            "selection_state": "pending",
            "selection_calls": 0,
            "slots": [],
            "selected": [],
            "error": "",
        }

    @staticmethod
    def _overnight_counts(rounds: list[dict[str, Any]]) -> tuple[int, int]:
        generated = 0
        failed = 0
        for round_data in rounds:
            for slot in round_data.get("slots") or []:
                if slot.get("status") == "completed":
                    generated += 1
                elif slot.get("status") == "failed":
                    failed += 1
        return generated, failed

    @staticmethod
    def _overnight_slot(round_data: dict[str, Any], slot_index: int) -> dict[str, Any] | None:
        for slot in round_data.get("slots") or []:
            if int(slot.get("slot_in_round") or 0) == slot_index:
                return slot
        return None

    @staticmethod
    def _overnight_path_for_filename(filename: object) -> Path | None:
        clean = Path(str(filename or "")).name
        if not clean or clean != str(filename):
            return None
        roots: list[Path] = []
        gallery_root = get_gallery_images_root()
        if gallery_root is not None:
            roots.append(gallery_root)
        # GeneratedImage.path is an internal runtime path. The selector must
        # be able to recover it after a restart, while browser responses still
        # expose only safe filenames. Keep the lookup project-local and never
        # accept a caller-supplied path.
        roots.append(OUTPUT_DIR)
        seen: set[Path] = set()
        for raw_root in roots:
            try:
                root = raw_root.expanduser().resolve()
            except OSError:
                continue
            if root in seen:
                continue
            seen.add(root)
            path = (root / clean).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.is_file():
                return path
        return None

    @classmethod
    def _overnight_paths(cls, rows: list[dict[str, Any]]) -> list[Path]:
        paths: list[Path] = []
        for row in rows:
            item = row.get("image") if isinstance(row, dict) else None
            path = cls._overnight_path_for_filename((item or {}).get("filename"))
            if path is not None:
                paths.append(path)
        return paths

    @staticmethod
    def _selected_rows(
        selected_paths: list[Path],
        source_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_name = {
            str(((row.get("image") or row).get("filename"))): row
            for row in source_rows
            if ((row.get("image") or row).get("filename"))
        }
        rows: list[dict[str, Any]] = []
        for rank, selected_path in enumerate(selected_paths, start=1):
            name = selected_path.name
            source = by_name.get(name)
            if source is None:
                matches = [row for key, row in by_name.items() if name.endswith("_" + key)]
                source = matches[0] if len(matches) == 1 else None
            if source is None:
                continue
            image = dict(source.get("image") or source)
            rows.append(
                {
                    "filename": str(image.get("filename") or ""),
                    "rank": rank,
                    "variation_index": (
                        source.get("variation_index")
                        if source.get("variation_index") is not None
                        else image.get("variation_index")
                    ),
                    "global_index": (
                        source.get("global_index")
                        if source.get("global_index") is not None
                        else image.get("global_index")
                    ),
                }
            )
        return rows

    @staticmethod
    def _add_selection_identity(rows: list[dict[str, Any]], overnight_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["overnight_job_id"] = overnight_id
            global_index = item.get("global_index")
            if isinstance(global_index, int) and global_index > 0:
                item["selection_identity"] = f"{overnight_id}:{global_index}"
            result.append(item)
        return result

    async def _select_overnight_paths(
        self,
        paths: list[Path],
        *,
        diagnostic_context: str = "overnight",
    ) -> tuple[list[Path], int, str | None]:
        """Use the existing selector, chunking only when its input is large."""

        if not paths:
            return [], 0, "選別対象の画像がありません。"
        if len(paths) <= 32:
            result = await asyncio.to_thread(
                select_generated_images,
                paths,
                return_count=4,
                diagnostic_context=diagnostic_context,
            )
            if not result.used_selection:
                return [], 1, _safe_error_summary(result.error or result.message or "画像選別AIを利用できませんでした。")
            return list(result.selected_paths), 1, None
        finalists: list[Path] = []
        calls = 0
        for start in range(0, len(paths), 32):
            selected, used_calls, error = await self._select_overnight_paths(
                paths[start : start + 32],
                diagnostic_context=f"{diagnostic_context}-chunk-{start // 32 + 1}",
            )
            calls += used_calls
            if error:
                return [], calls, error
            finalists.extend(selected)
        selected, used_calls, error = await self._select_overnight_paths(
            finalists,
            diagnostic_context=f"{diagnostic_context}-final",
        )
        return selected, calls + used_calls, error

    async def _recover_overnight_selection(self, overnight_id: str) -> None:
        """Re-select existing overnight images without submitting generation jobs."""

        try:
            async with self._generation_lock:
                record = self.overnight_jobs.get(overnight_id)
                if record is None:
                    return
                rounds = [dict(item) for item in (record.get("rounds") or []) if isinstance(item, dict)]
                timing = dict(record.get("timing") or {})
                timing.setdefault("selection_recovery_started_at", utc_now_iso())
                recovery_rounds: list[int] = []
                self.overnight_jobs.update(
                    overnight_id,
                    state="selecting",
                    selection_state="recovery_selecting",
                    timing=timing,
                    rounds=rounds,
                )

                for round_data in rounds:
                    if round_data.get("selection_state") != "failed":
                        continue
                    round_index = int(round_data.get("round_index") or 0)
                    if round_index < 1:
                        continue
                    recovery_rounds.append(round_index)
                    completed_rows = [
                        slot
                        for slot in (round_data.get("slots") or [])
                        if slot.get("status") == "completed" and isinstance(slot.get("image"), dict)
                    ]
                    paths = self._overnight_paths(completed_rows)
                    round_data["selection_started_at"] = utc_now_iso()
                    round_data["selection_state"] = "recovery_running"
                    self.overnight_jobs.update(overnight_id, rounds=rounds)
                    selected_paths, calls, error = await self._select_overnight_paths(
                        paths,
                        diagnostic_context=f"overnight-{overnight_id}-round-{round_index}-recovery",
                    )
                    round_data["selection_finished_at"] = utc_now_iso()
                    round_data["selection_calls"] = int(round_data.get("selection_calls") or 0) + calls
                    round_data["selected"] = self._add_selection_identity(
                        self._selected_rows(selected_paths, completed_rows), overnight_id
                    )
                    round_data["selection_state"] = "completed" if not error else "failed"
                    round_data["error"] = error or ""
                    self.overnight_jobs.update(overnight_id, rounds=rounds, error_summary=error or "")

                record = self.overnight_jobs.get(overnight_id) or record
                candidates: list[dict[str, Any]] = []
                for round_data in record.get("rounds") or []:
                    selected_rows = list(round_data.get("selected") or [])
                    if not selected_rows and round_data.get("selection_state") == "failed":
                        selected_rows = [
                            dict(slot.get("image") or {})
                            for slot in (round_data.get("slots") or [])
                            if slot.get("status") == "completed" and isinstance(slot.get("image"), dict)
                        ]
                    candidates.extend(selected_rows)
                candidate_paths = self._overnight_paths([{"image": row} for row in candidates])
                timing["final_selection_started_at"] = utc_now_iso()
                self.overnight_jobs.update(
                    overnight_id,
                    state="selecting",
                    selection_state="final_recovery_selecting",
                    timing=timing,
                )
                final_paths, final_calls, final_error = await self._select_overnight_paths(
                    candidate_paths,
                    diagnostic_context=f"overnight-{overnight_id}-final-recovery",
                )
                timing["final_selection_finished_at"] = utc_now_iso()
                timing["selection_recovery_finished_at"] = utc_now_iso()
                timing["selection_recovery_rounds"] = recovery_rounds
                timing["selection_recovery_calls"] = sum(
                    int(item.get("selection_calls") or 0) for item in (record.get("rounds") or [])
                ) + final_calls
                final_selected = list(record.get("final_selected") or [])
                if not final_error:
                    final_selected = self._add_selection_identity(
                        self._selected_rows(final_paths[:4], [{"image": row} for row in candidates]),
                        overnight_id,
                    )
                rounds_complete = all(
                    item.get("selection_state") == "completed"
                    for item in (record.get("rounds") or [])
                    if item.get("generation_finished_at")
                )
                final_state = "completed" if not final_error and rounds_complete else "completed_selection_failed"
                self.overnight_jobs.update(
                    overnight_id,
                    state=final_state,
                    selection_state="completed" if not final_error and rounds_complete else "failed",
                    final_selected=final_selected,
                    error_summary=final_error or "",
                    timing=timing,
                    finished_at=utc_now_iso(),
                    in_flight={},
                )
                if final_state == "completed":
                    await self._maybe_schedule_overnight_shutdown(overnight_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - durable recovery boundary
            logger.exception("Overnight selection recovery failed: overnight_id=%s", overnight_id)
            try:
                self.overnight_jobs.update(
                    overnight_id,
                    state="completed_selection_failed",
                    selection_state="failed",
                    error_summary=_safe_error_summary(exc),
                    in_flight={},
                )
            except OvernightJobError:
                logger.exception("Could not persist overnight selection recovery failure: overnight_id=%s", overnight_id)

    async def _run_overnight_job(self, overnight_id: str) -> None:
        record = self.overnight_jobs.get(overnight_id)
        if record is None or record["state"] in OVERNIGHT_TERMINAL_STATES:
            return
        try:
            self._set_overnight_phase(overnight_id, "starting")
            if record.get("in_flight"):
                timing = dict(record.get("timing") or {})
                timing.setdefault("backend_restart_recovered_at", utc_now_iso())
                timing.setdefault("backend_restart_recovery_mode", "paused_error_in_flight")
                self.overnight_jobs.update(
                    overnight_id,
                    state="paused_error",
                    error_summary="Backend再起動前に実行中slotがあり、重複生成を避けるため停止しました。",
                    selection_state="paused_error",
                    timing=timing,
                )
                return
            request_payload = dict(record.get("request") or {})
            request = _build_overnight_child_request(
                request_payload,
                reference_upload_bytes=self.config.reference_upload_bytes,
            )
            self._set_overnight_phase(overnight_id, "resolving_prompt")
            resolved = await asyncio.to_thread(self.service.resolve_request, request)
            slots = plan_bulk_variation_slots(
                OVERNIGHT_ROUND_SIZE,
                request.variation,
                explicit_tags=request.explicit_tags,
                resolved_prompt_tags=resolved.prompt_result.resolved_prompt_tags,
                resolved_danbooru_tags=resolved.prompt_result.danbooru_resolved_prompt_tags,
                request_text=request.request_text or "",
                reference_mode=str((request.reference or {}).get("mode") or "none"),
                requires_multiple_people=resolved.people_semantics.requires_multiple_people,
                multi_person_framing=request.multi_person_framing,
            )
            rounds = list(record.get("rounds") or [])
            used_seeds = {
                int((slot.get("image") or {}).get("seed"))
                for round_data in rounds
                for slot in (round_data.get("slots") or [])
                if slot.get("status") == "completed" and (slot.get("image") or {}).get("seed") is not None
            }
            timing = dict(record.get("timing") or {})
            timing.setdefault("overnight_started_at", record.get("started_at") or utc_now_iso())
            consecutive_infra_failures = 0

            async with self._generation_lock:
                self.overnight_jobs.update(overnight_id, state="running", timing=timing)
                for round_index in range(int(record["current_round"]), int(record["total_rounds"]) + 1):
                    record = self.overnight_jobs.get(overnight_id) or record
                    if record.get("stop_requested"):
                        timing.update(dict(record.get("timing") or {}))
                        timing["stopped_at"] = utc_now_iso()
                        self.overnight_jobs.update(
                            overnight_id, state="stopped", finished_at=timing["stopped_at"], timing=timing, in_flight={},
                        )
                        return
                    if record.get("pause_requested"):
                        self.overnight_jobs.update(overnight_id, state="paused")
                        return
                    round_data = self._overnight_round(record, round_index)
                    if not round_data.get("generation_started_at"):
                        round_data["generation_started_at"] = utc_now_iso()
                    rounds = [item for item in rounds if int(item.get("round_index") or 0) != round_index]
                    rounds.append(round_data)
                    rounds.sort(key=lambda item: int(item.get("round_index") or 0))
                    self.overnight_jobs.update(
                        overnight_id,
                        current_round=round_index,
                        current_slot=max(1, int(record.get("current_slot") or 1)),
                        state="running",
                        selection_state=f"round_{round_index}_generating",
                        rounds=rounds,
                    )

                    start_slot = max(1, int(record.get("current_slot") or 1)) if round_index == int(record.get("current_round")) else 1
                    for slot in slots:
                        if slot.index < start_slot or self._overnight_slot(round_data, slot.index) is not None:
                            continue
                        record = self.overnight_jobs.get(overnight_id) or record
                        if record.get("stop_requested"):
                            timing.update(dict(record.get("timing") or {}))
                            timing["stopped_at"] = utc_now_iso()
                            self.overnight_jobs.update(
                                overnight_id, state="stopped", finished_at=timing["stopped_at"], timing=timing, in_flight={},
                            )
                            return
                        if record.get("pause_requested"):
                            self.overnight_jobs.update(
                                overnight_id, state="paused", current_round=round_index,
                                current_slot=slot.index, rounds=rounds,
                            )
                            return
                        global_index = (round_index - 1) * OVERNIGHT_ROUND_SIZE + slot.index
                        if request.seed < 0:
                            seed = secrets.randbelow(2**32)
                            while seed in used_seeds:
                                seed = secrets.randbelow(2**32)
                        else:
                            seed = (request.seed + global_index - 1) % (2**32)
                        variation_request = replace(request, seed=seed, count=1, variation_offset=0)
                        in_flight = {
                            "round_index": round_index,
                            "slot_in_round": slot.index,
                            "global_index": global_index,
                            "seed": seed,
                            "started_at": utc_now_iso(),
                        }
                        self.overnight_jobs.update(
                            overnight_id,
                            state="running",
                            current_round=round_index,
                            current_slot=slot.index,
                            in_flight=in_flight,
                        )
                        self._set_overnight_phase(
                            overnight_id,
                            f"round_{round_index}_slot_{slot.index}_preparing",
                        )
                        last_error: BaseException | None = None
                        result = None
                        for attempt in range(OVERNIGHT_MAX_RETRIES + 1):
                            try:
                                variation_resolved = replace(resolved, request=variation_request)
                                variation_positive = merge_variation_tags(
                                    resolved.prompt_result.positive,
                                    slot.tags,
                                )
                                prepared = await asyncio.to_thread(
                                    self.service.prepare_generation,
                                    variation_request,
                                    resolved=variation_resolved,
                                    positive_override=variation_positive,
                                    variation_tags=slot.tags,
                                )
                                if not isinstance(prepared, PreparedGeneration):
                                    raise RuntimeError("生成準備結果が不正です。")
                                self._set_overnight_phase(
                                    overnight_id,
                                    f"round_{round_index}_slot_{slot.index}_submitting",
                                )
                                metadata = {
                                    "overnight_job_id": overnight_id,
                                    "round_index": round_index,
                                    "slot_in_round": slot.index,
                                    "global_index": global_index,
                                    "round_size": OVERNIGHT_ROUND_SIZE,
                                    "total_count": int(record["total_count"]),
                                }
                                gallery_metadata = _build_gallery_metadata(
                                    overnight_id,
                                    variation_request,
                                    prepared,
                                    variation_index=slot.index,
                                    variation_tags=slot.tags,
                                    metadata_overrides=metadata,
                                    requested_seed_override=request.seed,
                                )
                                prepared = replace(
                                    prepared,
                                    options=replace(
                                        prepared.options,
                                        gallery_output_root=get_gallery_images_root(),
                                        gallery_metadata=gallery_metadata,
                                    ),
                                )
                                result = await asyncio.to_thread(
                                    self.service.submit_generation,
                                    variation_request,
                                    prepared=prepared,
                                )
                                safe_result = _safe_image_metadata(result)
                                items = safe_result["images"]
                                if len(items) != 1:
                                    raise RuntimeError("夜間生成の1slotが1枚を返しませんでした。")
                                break
                            except Exception as exc:  # noqa: BLE001 - bounded retry boundary
                                last_error = exc
                                if attempt >= OVERNIGHT_MAX_RETRIES or not self._overnight_error_retryable(exc):
                                    break
                                await asyncio.sleep(5 if attempt == 0 else 15)
                        if result is None:
                            error = _safe_error_summary(last_error or RuntimeError("画像生成に失敗しました。"))
                            slot_row = {
                                "slot_in_round": slot.index,
                                "global_index": global_index,
                                "variation_index": slot.index,
                                "status": "failed",
                                "attempts": (OVERNIGHT_MAX_RETRIES + 1 if last_error else 1),
                                "error": error,
                            }
                            round_data.setdefault("slots", []).append(slot_row)
                            record = self.overnight_jobs.get(overnight_id) or record
                            rounds = [item for item in rounds if int(item.get("round_index") or 0) != round_index]
                            rounds.append(round_data)
                            rounds.sort(key=lambda item: int(item.get("round_index") or 0))
                            generated, failed = self._overnight_counts(rounds)
                            consecutive_infra_failures += 1
                            errors = list(record.get("errors") or [])
                            errors.append(error)
                            if consecutive_infra_failures >= OVERNIGHT_MAX_CONSECUTIVE_INFRA_FAILURES:
                                self.overnight_jobs.update(
                                    overnight_id,
                                    state="paused_error",
                                    selection_state="paused_error",
                                    error_summary="インフラエラーが連続したため夜間生成を一時停止しました。",
                                    errors=errors[-20:],
                                    generated_count=generated,
                                    failed_count=failed,
                                    rounds=rounds,
                                    current_round=round_index,
                                    current_slot=slot.index + 1,
                                    in_flight={},
                                )
                                return
                            self.overnight_jobs.update(
                                overnight_id,
                                errors=errors[-20:], generated_count=generated, failed_count=failed,
                                rounds=rounds, current_round=round_index, current_slot=slot.index + 1,
                                in_flight={}, error_summary=error,
                            )
                            continue

                        consecutive_infra_failures = 0
                        items = _safe_image_metadata(result)["images"]
                        image = dict(items[0])
                        actual_seed = image.get("seed")
                        if actual_seed is not None:
                            used_seeds.add(int(actual_seed))
                        slot_row = {
                            "slot_in_round": slot.index,
                            "global_index": global_index,
                            "variation_index": slot.index,
                            "variation_tags": list(slot.tags),
                            "status": "completed",
                            "attempts": 1,
                            "image": image,
                        }
                        round_data.setdefault("slots", []).append(slot_row)
                        record = self.overnight_jobs.get(overnight_id) or record
                        rounds = [item for item in rounds if int(item.get("round_index") or 0) != round_index]
                        rounds.append(round_data)
                        rounds.sort(key=lambda item: int(item.get("round_index") or 0))
                        generated, failed = self._overnight_counts(rounds)
                        self.overnight_jobs.update(
                            overnight_id,
                            generated_count=generated,
                            failed_count=failed,
                            rounds=rounds,
                            current_round=round_index,
                            current_slot=slot.index + 1,
                            in_flight={},
                            error_summary="",
                        )
                        self._set_overnight_phase(
                            overnight_id,
                            f"round_{round_index}_slot_{slot.index}_completed",
                        )

                    round_data["generation_finished_at"] = utc_now_iso()
                    round_paths = self._overnight_paths(
                        [slot for slot in round_data.get("slots") or [] if slot.get("status") == "completed"]
                    )
                    round_data["selection_started_at"] = utc_now_iso()
                    round_data["selection_state"] = "running"
                    self.overnight_jobs.update(
                        overnight_id, state="selecting", selection_state=f"round_{round_index}_selecting", rounds=rounds,
                    )
                    selected_paths, selection_calls, selection_error = await self._select_overnight_paths(
                        round_paths,
                        diagnostic_context=f"overnight-{overnight_id}-round-{round_index}",
                    )
                    round_data["selection_finished_at"] = utc_now_iso()
                    round_data["selection_calls"] = selection_calls
                    source_rows = [slot for slot in round_data.get("slots") or [] if slot.get("status") == "completed"]
                    round_data["selected"] = self._add_selection_identity(
                        self._selected_rows(selected_paths, source_rows), overnight_id
                    )
                    round_data["selection_state"] = "completed" if not selection_error else "failed"
                    round_data["error"] = selection_error or ""
                    rounds = [item for item in rounds if int(item.get("round_index") or 0) != round_index]
                    rounds.append(round_data)
                    rounds.sort(key=lambda item: int(item.get("round_index") or 0))
                    self.overnight_jobs.update(
                        overnight_id,
                        state="running",
                        selection_state="rounds_running" if not selection_error else "round_selection_failed",
                        rounds=rounds,
                        current_round=round_index + 1,
                        current_slot=1,
                        error_summary=selection_error or "",
                    )
                    record = self.overnight_jobs.get(overnight_id) or record

                record = self.overnight_jobs.get(overnight_id) or record
                candidates = []
                for round_data in record.get("rounds") or []:
                    selected_rows = list(round_data.get("selected") or [])
                    # A failed per-round selector must not discard the images
                    # already generated.  Keep that round's completed rows as
                    # final-selection candidates; the final selector may still
                    # choose a Best4 from the retained images.
                    if not selected_rows and round_data.get("selection_state") == "failed":
                        selected_rows = [
                            dict(slot.get("image") or {})
                            for slot in (round_data.get("slots") or [])
                            if slot.get("status") == "completed" and isinstance(slot.get("image"), dict)
                        ]
                    candidates.extend(selected_rows)
                candidate_paths = self._overnight_paths([{"image": row} for row in candidates])
                timing["final_selection_started_at"] = utc_now_iso()
                self.overnight_jobs.update(
                    overnight_id, state="selecting", selection_state="final_selecting", timing=timing,
                )
                final_paths, final_calls, final_error = await self._select_overnight_paths(
                    candidate_paths,
                    diagnostic_context=f"overnight-{overnight_id}-final",
                )
                timing["final_selection_finished_at"] = utc_now_iso()
                timing["selector_calls"] = sum(
                    int(item.get("selection_calls") or 0) for item in (record.get("rounds") or [])
                ) + final_calls
                final_source_rows = [{"image": row} for row in candidates]
                final_selected = self._add_selection_identity(
                    self._selected_rows(final_paths[:4], final_source_rows), overnight_id
                )
                finished = utc_now_iso()
                final_state = "completed_selection_failed" if final_error else "completed"
                self.overnight_jobs.update(
                    overnight_id,
                    state=final_state,
                    selection_state="failed" if final_error else "completed",
                    final_selected=final_selected,
                    error_summary=final_error or "",
                    timing=timing,
                    finished_at=finished,
                    in_flight={},
                )
                # Persist the completed state first.  The shutdown guard then
                # re-reads that durable record and independently checks the
                # final selection, all rounds, and the ComfyUI queue.
                if final_state == "completed":
                    await self._maybe_schedule_overnight_shutdown(overnight_id)
        except asyncio.CancelledError:
            logger.warning("Overnight worker cancelled: overnight_id=%s", overnight_id)
            try:
                current = self.overnight_jobs.get(overnight_id)
                if current is not None and current.get("state") in OVERNIGHT_ACTIVE_STATES:
                    timing = dict(current.get("timing") or {})
                    timing["worker_cancelled_at"] = utc_now_iso()
                    timing["worker_phase"] = "cancelled"
                    self.overnight_jobs.update(
                        overnight_id,
                        state="paused_error",
                        selection_state="paused_error",
                        error_summary="Overnight workerが中断されたため、重複生成防止のため手動復旧が必要です。",
                        timing=timing,
                        in_flight=current.get("in_flight") or {},
                    )
            except Exception:  # noqa: BLE001 - preserve cancellation semantics
                logger.exception("Could not persist Overnight cancellation: overnight_id=%s", overnight_id)
            raise
        except Exception as exc:  # noqa: BLE001 - durable worker boundary
            logger.exception("Overnight generation failed: overnight_id=%s", overnight_id)
            try:
                current = self.overnight_jobs.get(overnight_id)
                timing = dict((current or {}).get("timing") or {})
                timing["worker_exception_at"] = utc_now_iso()
                timing["worker_exception_type"] = type(exc).__name__
                timing["worker_phase"] = "failed"
                self.overnight_jobs.update(
                    overnight_id,
                    state="failed",
                    error_summary=_safe_error_summary(exc),
                    finished_at=utc_now_iso(),
                    in_flight={},
                    timing=timing,
                )
            except OvernightJobError:
                logger.exception("Could not persist overnight failure: overnight_id=%s", overnight_id)

    async def _run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        try:
            # Keep generation serial with the existing sd_control lock and
            # make a later request visibly queued instead of blocking HTTP.
            async with self._generation_lock:
                self.jobs.update(job_id, state="resolving", progress=0.05)
                if job.total_count == 1:
                    prepared = await asyncio.to_thread(self.service.prepare_generation, job.request)
                    if isinstance(prepared, PreparedGeneration):
                        gallery_metadata = _build_gallery_metadata(job_id, job.request, prepared)
                        prepared = replace(
                            prepared,
                            options=replace(
                                prepared.options,
                                gallery_output_root=get_gallery_images_root(),
                                gallery_metadata=gallery_metadata,
                            ),
                        )
                    self.jobs.update(job_id, state="generating", progress=0.5)
                    result = await asyncio.to_thread(
                        self.service.submit_generation,
                        job.request,
                        prepared=prepared,
                    )
                    items = _safe_image_metadata(result)["images"]
                    self.jobs.update(
                        job_id,
                        state="completed",
                        progress=1.0,
                        completed_count=len(items),
                        output_items=items,
                        output_metadata={
                            **_safe_image_metadata(result),
                            "gallery_job_id": job_id,
                            "total_count": 1,
                            "completed_count": len(items),
                            "failed_count": 0,
                        },
                    )
                    return

                resolution_request = replace(job.request, count=1)
                resolved = await asyncio.to_thread(self.service.resolve_request, resolution_request)
                slots = plan_variation_slots(
                    job.total_count,
                    job.variation,
                    start_index=resolution_request.variation_offset,
                    explicit_tags=resolution_request.explicit_tags,
                    resolved_prompt_tags=resolved.prompt_result.resolved_prompt_tags,
                    resolved_danbooru_tags=resolved.prompt_result.danbooru_resolved_prompt_tags,
                    request_text=resolution_request.request_text or "",
                    reference_mode=str((resolution_request.reference or {}).get("mode") or "none"),
                    requires_multiple_people=resolved.people_semantics.requires_multiple_people,
                    multi_person_framing=resolution_request.multi_person_framing,
                )
                outputs: list[dict[str, Any]] = []
                errors: list[str] = []
                for slot in slots:
                    current_index = len(outputs) + len(errors) + 1
                    self.jobs.update(
                        job_id,
                        state="generating",
                        progress=(current_index - 1) / job.total_count,
                    )
                    variation_request = replace(
                        resolution_request,
                        seed=(
                            -1
                            if resolution_request.seed < 0
                            else (resolution_request.seed + slot.index - 1) % (2**32)
                        ),
                    )
                    variation_resolved = replace(resolved, request=variation_request)
                    variation_positive = merge_variation_tags(
                        resolved.prompt_result.positive,
                        slot.tags,
                    )
                    try:
                        prepared = await asyncio.to_thread(
                            self.service.prepare_generation,
                            variation_request,
                            resolved=variation_resolved,
                            positive_override=variation_positive,
                            variation_tags=slot.tags,
                        )
                        if not isinstance(prepared, PreparedGeneration):
                            raise RuntimeError("生成準備結果が不正です。")
                        gallery_metadata = _build_gallery_metadata(
                            job_id,
                            variation_request,
                            prepared,
                            variation_index=slot.index,
                            variation_tags=slot.tags,
                        )
                        prepared = replace(
                            prepared,
                            options=replace(
                                prepared.options,
                                gallery_output_root=get_gallery_images_root(),
                                gallery_metadata=gallery_metadata,
                            ),
                        )
                        result = await asyncio.to_thread(
                            self.service.submit_generation,
                            variation_request,
                            prepared=prepared,
                        )
                        items = _safe_image_metadata(result)["images"]
                        if not items:
                            raise RuntimeError("画像が生成されませんでした。")
                        for item in items:
                            item["variation_index"] = slot.index
                            item["variation_tags"] = list(slot.tags)
                        outputs.extend(items)
                        self.jobs.update(
                            job_id,
                            completed_count=len(outputs),
                            failed_count=len(errors),
                            output_items=outputs,
                        )
                    except Exception as exc:  # noqa: BLE001 - one slot must not hide other results
                        logger.exception("Generation variation failed: job_id=%s index=%s", job_id, slot.index)
                        errors.append(_safe_error_summary(exc))
                        self.jobs.update(
                            job_id,
                            completed_count=len(outputs),
                            failed_count=len(errors),
                            error_summary=errors[-1],
                            output_items=outputs,
                        )
                final_state = "completed" if outputs else "failed"
                self.jobs.update(
                    job_id,
                    state=final_state,
                    progress=1.0,
                    completed_count=len(outputs),
                    failed_count=len(errors),
                    output_items=outputs,
                    output_metadata={
                        "image_count": len(outputs),
                        "images": outputs,
                        "image_identifiers": [item.get("filename") for item in outputs if item.get("filename")],
                        "gallery_job_id": job_id,
                        "total_count": job.total_count,
                        "completed_count": len(outputs),
                        "failed_count": len(errors),
                        "errors": errors[:4],
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - worker boundary must be sanitized
            logger.exception("Generation API job failed: job_id=%s", job_id)
            self.jobs.update(
                job_id,
                state="failed",
                progress=1.0,
                error_summary=_safe_error_summary(exc),
            )

    async def _run_bulk_job(self, bulk_id: str) -> None:
        job = self.bulk_jobs.get(bulk_id)
        if job is None:
            return
        try:
            async with self._generation_lock:
                self.bulk_jobs.update(bulk_id, state="resolving", progress=0.02)
                resolution_request = replace(job.request, count=1, variation_offset=0)
                resolved = await asyncio.to_thread(self.service.resolve_request, resolution_request)
                slots = plan_bulk_variation_slots(
                    job.requested_total,
                    job.variation,
                    explicit_tags=resolution_request.explicit_tags,
                    resolved_prompt_tags=resolved.prompt_result.resolved_prompt_tags,
                    resolved_danbooru_tags=resolved.prompt_result.danbooru_resolved_prompt_tags,
                    request_text=resolution_request.request_text or "",
                    reference_mode=str((resolution_request.reference or {}).get("mode") or "none"),
                    requires_multiple_people=resolved.people_semantics.requires_multiple_people,
                    multi_person_framing=resolution_request.multi_person_framing,
                )
                outputs: list[dict[str, Any]] = []
                errors: list[str] = []
                for slot in slots:
                    current_index = len(outputs) + len(errors) + 1
                    if job.stop_requested:
                        break
                    if job.pause_requested:
                        self.bulk_jobs.update(
                            bulk_id,
                            state="paused",
                            current_index=current_index,
                            progress=(current_index - 1) / job.requested_total,
                        )
                        return
                    self.bulk_jobs.update(
                        bulk_id,
                        state="generating",
                        current_index=current_index,
                        progress=(current_index - 1) / job.requested_total,
                    )
                    variation_request = replace(
                        resolution_request,
                        variation=job.variation,
                        seed=(
                            -1
                            if resolution_request.seed < 0
                            else (resolution_request.seed + job.seed_offset + slot.index - 1) % (2**32)
                        ),
                    )
                    variation_resolved = replace(resolved, request=variation_request)
                    variation_positive = merge_variation_tags(
                        resolved.prompt_result.positive,
                        slot.tags,
                    )
                    try:
                        prepared = await asyncio.to_thread(
                            self.service.prepare_generation,
                            variation_request,
                            resolved=variation_resolved,
                            positive_override=variation_positive,
                            variation_tags=slot.tags,
                        )
                        if not isinstance(prepared, PreparedGeneration):
                            raise RuntimeError("生成準備結果が不正です。")
                        gallery_metadata = _build_gallery_metadata(
                            bulk_id,
                            variation_request,
                            prepared,
                            variation_index=slot.index,
                            variation_tags=slot.tags,
                            metadata_overrides=job.metadata_context,
                            requested_seed_override=job.requested_seed_override,
                        )
                        prepared = replace(
                            prepared,
                            options=replace(
                                prepared.options,
                                gallery_output_root=get_gallery_images_root(),
                                gallery_metadata=gallery_metadata,
                            ),
                        )
                        result = await asyncio.to_thread(
                            self.service.submit_generation,
                            variation_request,
                            prepared=prepared,
                        )
                        safe_result = _safe_image_metadata(result)
                        items = safe_result["images"]
                        result_images = tuple(getattr(result, "images", ()) or ())
                        if not items:
                            raise RuntimeError("画像が生成されませんでした。")
                        for item, result_image in zip(items, result_images):
                            item["variation_index"] = slot.index
                            item["variation_tags"] = list(slot.tags)
                            image_path = getattr(result_image, "path", None)
                            if image_path is not None:
                                job.generated_paths.append(Path(image_path).resolve())
                        outputs.extend(items)
                        self.bulk_jobs.update(
                            bulk_id,
                            completed_count=len(outputs),
                            failed_count=len(errors),
                            current_index=current_index,
                            output_items=outputs,
                        )
                    except Exception as exc:  # noqa: BLE001 - continue remaining slots
                        logger.exception("Bulk generation variation failed: bulk_id=%s index=%s", bulk_id, slot.index)
                        errors.append(_safe_error_summary(exc))
                        self.bulk_jobs.update(
                            bulk_id,
                            completed_count=len(outputs),
                            failed_count=len(errors),
                            current_index=current_index,
                            error_summary=errors[-1],
                            errors=errors,
                            output_items=outputs,
                        )

                if job.pause_requested and not job.stop_requested:
                    self.bulk_jobs.update(
                        bulk_id,
                        state="paused",
                        progress=1.0,
                        current_index=len(outputs) + len(errors),
                        completed_count=len(outputs),
                        failed_count=len(errors),
                        errors=errors,
                        output_items=outputs,
                    )
                    return

                selection_state = "not_requested"
                selected_items: list[dict[str, Any]] = []
                selection_error = None
                if job.selection_mode != "none" and job.generated_paths:
                    selection_state = "running"
                    self.bulk_jobs.update(bulk_id, state="selecting", selection_state=selection_state)
                    try:
                        selection = await asyncio.to_thread(
                            select_generated_images,
                            list(job.generated_paths),
                            return_count=1 if job.selection_mode == "best1" else 4,
                        )
                        if selection.used_selection:
                            source_by_name = {path.name: path for path in job.generated_paths}
                            selected_names: list[str] = []
                            for rank, selected_path in enumerate(selection.selected_paths, start=1):
                                selected_name = selected_path.name
                                source_path = source_by_name.get(selected_name)
                                if source_path is None:
                                    suffix_matches = [
                                        path
                                        for path in job.generated_paths
                                        if selected_name.endswith("_" + path.name)
                                    ]
                                    if len(suffix_matches) == 1:
                                        source_path = suffix_matches[0]
                                        selected_name = source_path.name
                                selected_names.append(selected_name)
                                score = selection.candidate_scores.get(source_path) if source_path else None
                                row: dict[str, Any] = {"filename": selected_name, "rank": rank}
                                matched_item = next(
                                    (item for item in outputs if item.get("filename") == selected_name),
                                    None,
                                )
                                if matched_item is not None and matched_item.get("variation_index") is not None:
                                    row["variation_index"] = matched_item["variation_index"]
                                if score is not None:
                                    row["score"] = score
                                selected_items.append(row)
                            selected_set = set(selected_names)
                            for item in outputs:
                                if item.get("filename") in selected_set:
                                    item["selected"] = True
                            selection_state = "completed"
                        else:
                            selection_error = _safe_error_summary(
                                selection.error or selection.message or "画像選別AIを利用できませんでした。"
                            )
                            selection_state = "failed"
                    except Exception as exc:  # noqa: BLE001 - selection never fails generation
                        logger.exception("Bulk image selection failed: bulk_id=%s", bulk_id)
                        selection_error = _safe_error_summary(exc)
                        selection_state = "failed"

                final_state = "stopped" if job.stop_requested else ("completed" if outputs else "failed")
                self.bulk_jobs.update(
                    bulk_id,
                    state=final_state,
                    progress=1.0,
                    current_index=len(outputs) + len(errors),
                    completed_count=len(outputs),
                    failed_count=len(errors),
                    errors=errors,
                    output_items=outputs,
                    selected_items=selected_items,
                    selection_state=selection_state,
                    selection_error=selection_error,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - worker boundary is sanitized
            logger.exception("Generation API bulk job failed: bulk_id=%s", bulk_id)
            self.bulk_jobs.update(
                bulk_id,
                state="failed",
                progress=1.0,
                error_summary=_safe_error_summary(exc),
                selection_state="failed" if job.selection_mode != "none" else "not_requested",
            )


def create_generation_app(
    *,
    service: GenerationService | None = None,
    config: GenerationApiConfig | None = None,
    tag_store: DanbooruTagStore | None = None,
    tag_interpreter: object | None = None,
    overnight_db_path: str | Path | None = None,
) -> web.Application:
    """Build the app without binding a socket; suitable for a launcher/tests."""

    return GenerationApi(
        service=service,
        config=config,
        tag_store=tag_store,
        tag_interpreter=tag_interpreter,
        overnight_db_path=overnight_db_path,
    ).app


def run_generation_api(
    *,
    service: GenerationService | None = None,
    config: GenerationApiConfig | None = None,
) -> None:
    """Run the standalone loopback server."""

    api_config = config or GenerationApiConfig()
    app = create_generation_app(service=service, config=api_config)
    web.run_app(app, host=api_config.host, port=api_config.port)


__all__ = [
    "DEFAULT_GENERATION_API_HOST",
    "DEFAULT_GENERATION_API_PORT",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_REFERENCE_UPLOAD_BYTES",
    "GenerationApi",
    "GenerationApiConfig",
    "GenerationApiError",
    "create_generation_app",
    "run_generation_api",
]
