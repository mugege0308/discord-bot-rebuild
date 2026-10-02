"""Adapt one persisted Batch image to the normal generation backend.

The Batch lifecycle service deliberately owns only durable job state.  This
module is the narrow bridge used by a future Batch worker: it resolves one
image's prompt, model, LoRA snapshot, seed, and metadata into the existing
``SdGenerationOptions`` contract, then optionally invokes the existing
single-generation function.  It does not start a worker, select candidates,
send Discord messages, or enqueue Gallery work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from services.batch_generation import (
    BatchGenerationRequest,
    BatchSpec,
    manager as batch_manager,
)
from services.generation_service import GenerationRequest, GenerationService
from services.image_generation_profiles import get_image_generation_profile
from services.gallery_control import get_gallery_images_root
from services.sd_control import (
    GeneratedImage,
    SdGenerationOptions,
    generate_images_with_metadata,
    lora_selections_to_dicts,
    resolve_saved_lora_selections,
)
from services.shogou_prompt_designer import ShogouPromptResult, validate_requested_prompt_rating


MAX_SEED = 2**32


class BatchImageConfigurationError(ValueError):
    """A persisted Batch setting cannot produce one valid image request."""

    def __init__(
        self,
        message: str,
        *,
        job_id: str = "",
        batch_id: str = "",
        variation_index: int = 0,
        image_index: int = 0,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.batch_id = batch_id
        self.variation_index = variation_index
        self.image_index = image_index


class BatchImageGenerationError(RuntimeError):
    """Generation failed for one image and can be handled by a future worker."""

    def __init__(
        self,
        message: str,
        *,
        job_id: str,
        batch_id: str,
        variation_index: int,
        image_index: int,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.batch_id = batch_id
        self.variation_index = variation_index
        self.image_index = image_index
        self.cause = cause


@dataclass(frozen=True)
class BatchImageContext:
    """Identifiers attached to one generated image."""

    job_id: str
    batch_id: str
    variation_index: int
    image_index: int
    image_serial: int


def _context(
    spec: BatchSpec,
    *,
    image_index: int,
    image_serial: int | None,
    request: BatchGenerationRequest,
) -> BatchImageContext:
    resolved_serial = image_serial
    if resolved_serial is None:
        resolved_serial = (
            max(0, spec.variation_index - 1) * request.effective_images_per_variation
            + image_index
            - 1
        )
    if isinstance(resolved_serial, bool) or not isinstance(resolved_serial, int) or resolved_serial < 0:
        raise BatchImageConfigurationError(
            "image_serialが不正です。",
            job_id=spec.job_id,
            batch_id=spec.batch_id,
            variation_index=spec.variation_index,
            image_index=image_index,
        )
    return BatchImageContext(
        job_id=str(spec.job_id or ""),
        batch_id=str(spec.batch_id or ""),
        variation_index=spec.variation_index,
        image_index=image_index,
        image_serial=resolved_serial,
    )


def resolve_batch_image_seed(
    request: BatchGenerationRequest,
    *,
    image_serial: int,
) -> int:
    """Resolve the seed rule used by the legacy Batch worker.

    ``-1`` is intentionally retained so the existing backend chooses a fresh
    seed for each call.  A fixed non-negative seed is advanced by the global
    image serial and wrapped to the uint32 range.
    """

    if isinstance(image_serial, bool) or not isinstance(image_serial, int) or image_serial < 0:
        raise ValueError("image_serialが不正です。")
    raw_seed = request.generation_config.get("seed", -1)
    if raw_seed is None:
        raw_seed = -1
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, float)):
        raise ValueError("seedが不正です。")
    numeric_seed = float(raw_seed)
    if not math.isfinite(numeric_seed) or int(numeric_seed) != numeric_seed:
        raise ValueError("seedが不正です。")
    configured_seed = int(numeric_seed)
    if configured_seed < 0:
        return -1
    return (configured_seed + image_serial) % MAX_SEED


def _config_int(
    config: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}が不正です。")
    return value


def _config_float(
    config: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}が不正です。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{name}が不正です。")
    return resolved


def _wrap_configuration_error(
    error: BaseException,
    *,
    context: BatchImageContext,
) -> BatchImageConfigurationError:
    return BatchImageConfigurationError(
        str(error) or "Batch画像設定が不正です。",
        job_id=context.job_id,
        batch_id=context.batch_id,
        variation_index=context.variation_index,
        image_index=context.image_index,
    )


def build_batch_image_options(
    request: BatchGenerationRequest,
    spec: BatchSpec,
    *,
    image_index: int = 1,
    image_serial: int | None = None,
    image_index_in_batch: int | None = None,
) -> SdGenerationOptions:
    """Build exactly one normal-generation input from a Batch snapshot.

    ``spec.prompt`` and ``spec.negative_prompt`` are already the per-variation
    prompt values.  This function only performs deterministic backend-level
    resolution; it does not invoke Prompt Designer or candidate selection.
    """

    if image_index_in_batch is not None:
        if image_index != 1 and image_index != image_index_in_batch:
            raise ValueError("image_indexとimage_index_in_batchが一致しません。")
        image_index = image_index_in_batch

    try:
        if not isinstance(request, BatchGenerationRequest):
            raise ValueError("Batch requestが不正です。")
        if not isinstance(spec, BatchSpec):
            raise ValueError("Batch specが不正です。")
        if isinstance(image_index, bool) or not isinstance(image_index, int) or image_index < 1:
            raise ValueError("image_indexは1以上の整数で指定してください。")
        if spec.requested_count and image_index > spec.requested_count:
            raise ValueError("image_indexがbatchの枚数を超えています。")
        if isinstance(spec.variation_index, bool) or not isinstance(spec.variation_index, int) or spec.variation_index < 1:
            raise ValueError("variation_indexは1以上の整数で指定してください。")
        if not str(spec.job_id or "").strip() or not str(spec.batch_id or "").strip():
            raise ValueError("job_idとbatch_idは必須です。")

        context = _context(
            spec,
            image_index=image_index,
            image_serial=image_serial,
            request=request,
        )
        profile = get_image_generation_profile(request.model_key.strip())
        resolved_rating = validate_requested_prompt_rating(
            request.rating,
            is_adult=profile.is_adult,
        )

        if spec.checkpoint and spec.checkpoint != profile.checkpoint:
            raise ValueError("保存済みBatchのcheckpointがmodel profileと一致しません。")

        raw_loras = spec.loras if spec.loras else request.loras
        loras = resolve_saved_lora_selections(
            raw_loras,
            model_family=profile.model_family,
        )

        positive = str(spec.prompt or request.base_prompt or "").strip()
        if not positive:
            explicit_parts = [
                str(request.character or "").strip(),
                str(request.request_text or "").strip(),
            ]
            positive = ", ".join(part for part in explicit_parts if part)
        if not positive:
            raise ValueError("Batch promptが空です。")
        negative = str(spec.negative_prompt or request.base_negative_prompt or "")
        config = request.generation_config
        if not isinstance(config, dict):
            raise ValueError("generation_configが不正です。")

        seed = resolve_batch_image_seed(request, image_serial=context.image_serial)
        width = _config_int(config, "width", 512, minimum=256)
        height = _config_int(config, "height", 512, minimum=256)
        steps = _config_int(config, "steps", 20)
        cfg = _config_float(config, "cfg", 7.0, minimum=0.0)
        sampler = config.get("sampler", "Euler a")
        if not isinstance(sampler, str) or not sampler.strip():
            raise ValueError("samplerが不正です。")

        metadata = {
            "batch": True,
            "batch_id": context.batch_id,
            "job_id": context.job_id,
            "variation_index": context.variation_index,
            "image_index": context.image_index,
            "image_index_in_batch": context.image_index,
            "image_serial": context.image_serial,
            "variation_count": request.effective_variation_count,
            "character": request.character,
            "character_name": request.character,
            "request_text": request.request_text,
            "model_key": request.model_key,
            "model_family": profile.model_family,
            "rating": resolved_rating,
            "seed": seed,
            "selection_ai": False,
            "reference_mode": "none",
            "pose_reference": False,
            "loras": list(lora_selections_to_dicts(loras)),
        }
        generation_request = GenerationRequest(
            freeform_character=request.character,
            request_text=request.request_text,
            model_key=profile.model_key,
            rating=request.rating,
            loras=loras,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler=sampler.strip(),
            count=1,
            seed=seed,
            requester_id=request.requested_by,
            source="discord.batch",
            model_selection_explicit=True,
            explicit_generation_fields=frozenset(
                field
                for field in ("width", "height", "steps", "cfg", "sampler")
                if field in config
            ),
        )
        prompt_snapshot = ShogouPromptResult(positive=positive, negative=negative)
        prepared = GenerationService().prepare_generation(
            generation_request,
            prompt_result=prompt_snapshot,
            resolved_loras=loras,
            gallery_output_root=get_gallery_images_root(),
            gallery_metadata=metadata,
        )
        return prepared.options
    except BatchImageConfigurationError:
        raise
    except Exception as exc:
        if "context" not in locals():
            context = BatchImageContext(
                job_id=str(getattr(spec, "job_id", "")),
                batch_id=str(getattr(spec, "batch_id", "")),
                variation_index=int(getattr(spec, "variation_index", 0) or 0),
                image_index=image_index,
                image_serial=max(0, int(image_serial or 0)),
            )
        raise _wrap_configuration_error(exc, context=context) from exc


def build_batch_image_options_for_job(
    job_id: str,
    requested_by: int,
    spec: BatchSpec,
    *,
    image_index: int = 1,
    image_serial: int | None = None,
) -> SdGenerationOptions:
    """Load an owner-checked persisted request and adapt one Batch image."""

    request = batch_manager.get_request_for_user(job_id, requested_by)
    if request is None:  # Defensive; the manager raises for a missing job.
        raise BatchImageConfigurationError("指定されたBatch jobが見つかりません。", job_id=job_id)
    if spec.job_id != job_id:
        raise BatchImageConfigurationError(
            "Batch specのjob_idが一致しません。",
            job_id=job_id,
            batch_id=spec.batch_id,
            variation_index=spec.variation_index,
            image_index=image_index,
        )
    return build_batch_image_options(
        request,
        spec,
        image_index=image_index,
        image_serial=image_serial,
    )


def generate_one_batch_image(
    request: BatchGenerationRequest,
    spec: BatchSpec,
    *,
    image_index: int = 1,
    image_serial: int | None = None,
    generator: Callable[[SdGenerationOptions], list[GeneratedImage]] = generate_images_with_metadata,
) -> GeneratedImage:
    """Invoke the existing backend once and return exactly one image.

    The callback is injectable so tests and a future worker can use a mock or
    a controlled backend without changing the normal ``/generate_ai`` path.
    """

    options = build_batch_image_options(
        request,
        spec,
        image_index=image_index,
        image_serial=image_serial,
    )
    try:
        images = list(generator(options))
        if len(images) != 1:
            raise RuntimeError(f"Batch単画像生成が{len(images)}枚を返しました。")
        return images[0]
    except BatchImageGenerationError:
        raise
    except Exception as exc:
        raise BatchImageGenerationError(
            str(exc) or "Batch画像の生成に失敗しました。",
            job_id=str(spec.job_id),
            batch_id=str(spec.batch_id),
            variation_index=spec.variation_index,
            image_index=image_index,
            cause=exc,
        ) from exc


# Descriptive aliases for a future worker integration; they intentionally do
# not start any worker today.
prepare_batch_image_options = build_batch_image_options
generate_batch_image = generate_one_batch_image
