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
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from aiohttp import ContentTypeError, web

from services.appearance_resolver import get_appearance_resolver_status
from services.character_tag_dictionary import list_characters
from services.generation_service import GenerationRequest, GenerationService, PreparedGeneration
from services.generation_reference_assets import (
    MAX_REFERENCE_IMAGE_BYTES,
    load_reference_asset,
    save_reference_asset,
)
from services.generation_variation import (
    MAX_GENERATION_COUNT,
    VARIATION_AUTO,
    VARIATION_OFF,
    merge_variation_tags,
    plan_variation_slots,
)
from services.gallery_control import get_gallery_images_root
from services.image_composition import IMAGE_SIZE_SPECS
from services.image_generation_profiles import (
    IMAGE_GENERATION_PROFILES,
    get_image_generation_profile,
)
from services.lora_catalog import COMFY_STYLES, model_family_compatible
from services.lora_training import LORA_TYPES, normalize_lora_type
from services.prompt_generator import PromptGeneratorError
from services.sd_control import resolve_saved_lora_selections
from services.sd_control import DEFAULT_REFERENCE_STRENGTH
from services.pose_reference import DEFAULT_POSE_STRENGTH
from services.shogou_prompt_designer import PROMPT_RATING_CHOICES


logger = logging.getLogger("secretary-bot.generation_api")

DEFAULT_GENERATION_API_HOST = "127.0.0.1"
DEFAULT_GENERATION_API_PORT = 8787
DEFAULT_MAX_BODY_BYTES = 64 * 1024
DEFAULT_REFERENCE_UPLOAD_BYTES = MAX_REFERENCE_IMAGE_BYTES
MAX_TEXT_LENGTH = 4000
MAX_EXPLICIT_TAGS = 100
MAX_LORAS = 3

_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "character_preset",
        "freeform_character",
        "explicit_tags",
        "appearance_text",
        "request_text",
        "model_key",
        "rating",
        "loras",
        "size",
        "count",
        "variation",
        "seed",
        "reference_mode",
        "reference_image_id",
        "reference_strength",
        "reference_fit",
    }
)
_ALLOWED_LORA_KEYS = frozenset({"key", "lora_type", "strength"})
_ABSOLUTE_PATH_RE = re.compile(r"(?:(?:[A-Za-z]:)?[\\/])(?:[^\s,;]+[\\/])+[^\s,;]+")
GENERATION_API_APP_KEY = web.AppKey("generation_api", object)


class GenerationApiError(ValueError):
    """A client-visible, sanitized request validation error."""


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


def _is_configured(*names: str) -> bool:
    return any(os.getenv(name, "").strip() for name in names)


def _json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _clean_text(value: object, field_name: str, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise GenerationApiError(f"{field_name}は文字列で指定してください。")
    value = value.strip()
    if len(value) > MAX_TEXT_LENGTH:
        raise GenerationApiError(f"{field_name}が長すぎます。")
    return value or None


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


def _parse_reference(
    payload: Mapping[str, object],
    *,
    reference_upload_bytes: int,
) -> dict[str, object] | None:
    mode_value = payload.get("reference_mode", "none")
    if not isinstance(mode_value, str):
        raise GenerationApiError("reference_modeが不正です。")
    mode = mode_value.casefold().strip()
    if mode not in {"none", "composition", "pose"}:
        raise GenerationApiError("reference_modeはnone / composition / poseのいずれかで指定してください。")
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

    model_key = payload.get("model_key", "animagine")
    if not isinstance(model_key, str) or not model_key.strip():
        raise GenerationApiError("model_keyが不正です。")
    try:
        profile = get_image_generation_profile(model_key.strip())
    except ValueError as exc:
        raise GenerationApiError("指定されたモデルには対応していません。") from exc

    rating = payload.get("rating", "auto")
    if not isinstance(rating, str) or rating.casefold() not in PROMPT_RATING_CHOICES:
        raise GenerationApiError("指定されたratingには対応していません。")
    rating = rating.casefold()
    if not profile.is_adult and rating not in {"auto", "safe"}:
        raise GenerationApiError("このratingはAdultモデルでのみ使用できます。")

    size = payload.get("size", "square")
    if not isinstance(size, str) or size.casefold() not in IMAGE_SIZE_SPECS:
        raise GenerationApiError("指定されたsizeには対応していません。")
    size = size.casefold()
    size_spec = IMAGE_SIZE_SPECS[size]

    count = _parse_count(payload.get("count", 1))
    variation = _parse_variation(payload.get("variation"), count=count)
    reference = _parse_reference(payload, reference_upload_bytes=reference_upload_bytes)

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
        model_key=model_key.strip(),
        rating=rating,
        loras=loras,
        size_preset=size,
        width=size_spec.width,
        height=size_spec.height,
        count=count,
        variation=variation,
        seed=_parse_seed(payload.get("seed")),
        source="generation_api",
        reference=reference,
    )
    try:
        generation_request.validate()
    except PromptGeneratorError as exc:
        raise GenerationApiError(str(exc)) from exc
    return generation_request


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
    return {
        "source": "generation_api",
        "job_id": job_id,
        "parent_job_id": job_id,
        "variation_index": variation_index,
        "variation_tags": list(variation_tags),
        "variation": request.variation,
        "model": str(options.model_key or request.model_key),
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
        "reference_image_id": str((request.reference or {}).get("image_id") or ""),
        "reference_strength": (request.reference or {}).get("strength"),
        "reference_fit": str((request.reference or {}).get("fit") or "contain"),
    }


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
        "variation": job.variation,
        "output": output,
    }


class GenerationApi:
    """aiohttp adapter with an in-memory, non-blocking job lifecycle."""

    def __init__(
        self,
        *,
        service: GenerationService | None = None,
        config: GenerationApiConfig | None = None,
    ) -> None:
        self.config = config or GenerationApiConfig()
        self.service = service or GenerationService()
        self.jobs = _JobStore()
        self._generation_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self.app = web.Application(
            client_max_size=max(self.config.max_body_bytes, self.config.reference_upload_bytes)
        )
        self.app.add_routes(
            [
                web.get("/api/generation/health", self.health),
                web.get("/api/generation/options", self.options),
                web.post("/api/generation/references", self.upload_reference),
                web.post("/api/generation/jobs", self.create_job),
                web.get("/api/generation/jobs/{job_id}", self.get_job),
            ]
        )
        self.app.on_cleanup.append(self._cleanup)
        self.app[GENERATION_API_APP_KEY] = self

    async def _cleanup(self, _app: web.Application) -> None:
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
            }
            for profile in IMAGE_GENERATION_PROFILES.values()
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
        except GenerationApiError as exc:
            return _json_error(str(exc), 400)
        except web.HTTPRequestEntityTooLarge:
            return _json_error("リクエストが大きすぎます。", 413)

        job = self.jobs.create(generation_request)
        job.total_count = generation_request.count
        job.variation = generation_request.variation
        task = asyncio.create_task(self._run_job(job.job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return web.json_response(_job_response(job), status=202)

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
                    explicit_tags=resolution_request.explicit_tags,
                    resolved_prompt_tags=resolved.prompt_result.resolved_prompt_tags,
                    resolved_danbooru_tags=resolved.prompt_result.danbooru_resolved_prompt_tags,
                    reference_mode=str((resolution_request.reference or {}).get("mode") or "none"),
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


def create_generation_app(
    *,
    service: GenerationService | None = None,
    config: GenerationApiConfig | None = None,
) -> web.Application:
    """Build the app without binding a socket; suitable for a launcher/tests."""

    return GenerationApi(service=service, config=config).app


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
