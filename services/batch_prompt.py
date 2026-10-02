"""Prepare the prompt snapshot persisted for a Batch job.

Batch workers must consume a stable, already-resolved prompt.  The command
layer therefore prepares it once when the Batch specs are created and leaves
the worker/adapter responsible only for generation-specific resolution such as
LoRA trigger application.
"""

from __future__ import annotations

from dataclasses import replace

from services.batch_generation import BatchGenerationRequest
from services.generation_service import GenerationRequest, GenerationService
from services.image_generation_profiles import get_image_generation_profile
from services.prompt_generator import PromptGeneratorError
from services.sd_control import get_negative_prompt
from services.shogou_prompt_designer import (
    ShogouPromptResult,
    generate_shogou_prompt_result,
    validate_requested_prompt_rating,
)


def _batch_prompt_input(request: BatchGenerationRequest) -> str:
    return ", ".join(
        value
        for value in (
            str(request.character or "").strip(),
            str(request.request_text or "").strip(),
        )
        if value
    )


def prepare_batch_prompt(request: BatchGenerationRequest) -> ShogouPromptResult:
    """Return the single prompt snapshot used by every image in a Batch.

    A non-empty ``base_prompt`` is already the final result from the normal
    ``/generate_ai`` path, so it is preserved rather than sent through the
    Prompt Designer a second time.  Standalone Batch requests use the same
    Prompt Designer and post-processing as normal generation, including the
    verified-character and confirmed-appearance handling enabled by the
    generic Danbooru resolver.
    """

    profile = get_image_generation_profile(request.model_key.strip())
    validate_requested_prompt_rating(request.rating, is_adult=profile.is_adult)

    existing_positive = str(request.base_prompt or "").strip()
    if existing_positive:
        negative = str(request.base_negative_prompt or "").strip() or get_negative_prompt(None)
        return ShogouPromptResult(
            positive=existing_positive,
            negative=negative,
            summary="既存の生成経路で解決済みのBatch prompt",
        )

    prompt_input = _batch_prompt_input(request)
    if not prompt_input:
        raise PromptGeneratorError("characterまたはrequest_textを指定してください。")

    generation_request = GenerationRequest(
        freeform_character=request.character,
        request_text=request.request_text,
        model_key=profile.model_key,
        rating=request.rating,
        requester_id=request.requested_by,
        source="discord.batch",
        enable_generic_danbooru=True,
    )
    generation_service = GenerationService()
    result = generation_service.resolve_request(
        generation_request,
        prompt_resolver=generate_shogou_prompt_result,
    ).prompt_result
    if result.character_confirmation_required or result.character_unresolved_reason:
        # Batch ``character`` is not a preset-only field.  The generic
        # resolver may abstain when the text is not a verified character (or
        # when it finds an unconfirmed candidate), but that must not prevent
        # ordinary free-form subjects such as "かわいい女の子" from reaching
        # the normal Prompt Designer.  A dedicated preset command can keep
        # its stricter validation; Batch has no such explicit-preset flag.
        generation_request = replace(
            generation_request,
            enable_generic_danbooru=False,
        )
        result = generation_service.resolve_request(
            generation_request,
            prompt_resolver=generate_shogou_prompt_result,
        ).prompt_result
    if result.blocked:
        raise PromptGeneratorError(result.blocked_reason or "プロンプトを生成できませんでした。")
    if not result.positive.strip():
        raise PromptGeneratorError("生成用Positive promptが空です。")

    negative = result.negative.strip() or get_negative_prompt(None)
    if negative != result.negative:
        return replace(result, negative=negative)
    return result
