"""Discord commands for the persisted, sequential Batch worker.

The command layer only resolves user input, persists a small set of Batch
specs, and hands the job to :mod:`services.batch_generation_worker`.  Prompt
construction and image generation remain in the existing Batch adapter.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import discord
from discord import app_commands

from services.batch_generation import (
    MAX_IMAGES_PER_PROMPT,
    BatchGenerationRequest,
    BatchJobError,
    BatchJobNotFoundError,
    BatchJobOwnershipError,
    build_variation_patterns,
    manager,
    recover_incomplete_jobs as recover_persisted_jobs,
)
from services.batch_generation_worker import BatchGenerationWorker
from services.generate_ai_last_settings import (
    get_last_batch_settings,
    get_last_model_key,
    get_last_settings,
    save_last_batch_settings,
)
from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.lora_catalog import get_comfy_lora_choice_specs
from services.pc_control import is_pc_control_allowed
from services.prompt_generator import PromptGeneratorError
from services.sd_control import (
    LoraSelection,
    lora_selections_to_dicts,
    resolve_lora_selections,
    resolve_saved_lora_selections,
)
from services.shogou_prompt_designer import validate_requested_prompt_rating


logger = logging.getLogger("secretary-bot.generate_batch_commands")

DENIED_MESSAGE = "この操作は、PC_CONTROL_ALLOWED_USER_IDS で許可されたユーザーだけが実行できます。"
NO_LORA_VALUE = "__none__"
DEFAULT_TOTAL_COUNT = 6
DEFAULT_IMAGES_PER_VARIATION = 1
MODEL_CHOICES = [
    app_commands.Choice(name="Animagine XL 4.0 Clear", value="animagine"),
    app_commands.Choice(name="Animagine XL 4.0 Clear Adult", value="animagine_adult"),
    app_commands.Choice(name="JANKU v7.77", value="janku"),
    app_commands.Choice(name="JANKU v7.77 Adult", value="janku_adult"),
]
RATING_CHOICES = [
    app_commands.Choice(name="Auto（既存設定）", value="auto"),
    app_commands.Choice(name="Safe", value="safe"),
    app_commands.Choice(name="Sensitive", value="sensitive"),
    app_commands.Choice(name="NSFW", value="nsfw"),
    app_commands.Choice(name="Explicit", value="explicit"),
]


def _lora_choices(lora_type: str) -> list[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name="なし（無効）", value=NO_LORA_VALUE),
        *(
            app_commands.Choice(name=name, value=key)
            for name, key in get_comfy_lora_choice_specs(lora_type, model_family="sdxl")[:24]
        ),
    ]


CHARACTER_LORA_CHOICES = _lora_choices("character")
STYLE_LORA_CHOICES = _lora_choices("style")
ADDITIONAL_LORA_CHOICES = _lora_choices("additional")


def _choice_value(choice: app_commands.Choice[str] | None) -> tuple[bool, str | None]:
    if choice is None:
        return False, None
    return True, None if choice.value == NO_LORA_VALUE else choice.value


def _lora_by_type(selections: tuple[LoraSelection, ...]) -> dict[str, LoraSelection]:
    return {selection.lora_type: selection for selection in selections}


def _last_ai_loras(user_id: int, model_key: str) -> tuple[LoraSelection, ...]:
    try:
        previous = get_last_settings(user_id, model_key)
        if previous is None:
            return ()
        return resolve_saved_lora_selections(
            previous.loras,
            model_family=get_image_generation_profile(model_key).model_family,
        )
    except (TypeError, ValueError, OSError):
        logger.warning("Stored /generate_ai LoRA settings are unavailable.", exc_info=True)
        return ()


def _resolve_model_and_loras(
    user_id: int,
    previous: dict[str, Any],
    model: app_commands.Choice[str] | None,
    character_lora: app_commands.Choice[str] | None,
    style_lora: app_commands.Choice[str] | None,
    additional_lora: app_commands.Choice[str] | None,
) -> tuple[str, tuple[LoraSelection, ...]]:
    """Resolve explicit > last Batch > /generate_ai > default settings."""

    if model is not None:
        model_key = model.value
    else:
        model_key = str(previous.get("model_key") or "").strip()
        if not model_key:
            model_key = str(get_last_model_key(user_id) or "").strip()
        model_key = model_key or DEFAULT_IMAGE_GENERATION_MODEL_KEY
    profile = get_image_generation_profile(model_key)
    model_family = profile.model_family

    try:
        stored_loras = resolve_saved_lora_selections(
            previous.get("loras") or [], model_family=model_family
        )
    except (TypeError, ValueError):
        logger.warning("Stored Batch LoRA settings are unavailable; using /generate_ai settings.", exc_info=True)
        stored_loras = ()
    ai_loras = _last_ai_loras(user_id, profile.model_key)
    saved_by_type = _lora_by_type(stored_loras or ai_loras)

    explicit_character, character_key = _choice_value(character_lora)
    explicit_style, style_key = _choice_value(style_lora)
    explicit_additional, additional_key = _choice_value(additional_lora)

    def fallback(lora_type: str) -> tuple[str | None, float | None]:
        saved = saved_by_type.get(lora_type)
        return (saved.key, saved.strength) if saved else (None, None)

    character_key, character_strength = (
        (character_key, None) if explicit_character else fallback("character")
    )
    style_key, style_strength = (
        (style_key, None) if explicit_style else fallback("style")
    )
    additional_key, additional_strength = (
        (additional_key, None) if explicit_additional else fallback("additional")
    )

    if stored_loras and not (explicit_character or explicit_style or explicit_additional):
        return profile.model_key, stored_loras
    return profile.model_key, resolve_lora_selections(
        character_key=character_key,
        character_strength=character_strength,
        style_key=style_key,
        style_strength=style_strength,
        additional_key=additional_key,
        additional_strength=additional_strength,
        model_family=model_family,
    )


def _resolve_rating(
    user_id: int,
    model_key: str,
    previous: dict[str, Any],
    choice: app_commands.Choice[str] | None,
) -> str:
    if choice is not None:
        requested = choice.value
        explicit = True
    else:
        requested = str(previous.get("rating") or "").strip()
        if not requested:
            previous_ai = get_last_settings(user_id, model_key)
            requested = previous_ai.rating if previous_ai is not None else "auto"
        explicit = False
    profile = get_image_generation_profile(model_key)
    try:
        return validate_requested_prompt_rating(requested, is_adult=profile.is_adult)
    except PromptGeneratorError as error:
        if explicit:
            raise ValueError(str(error)) from error
        return validate_requested_prompt_rating("auto", is_adult=profile.is_adult)


def _base_prompt(character: str, request_text: str) -> str:
    values = [str(character or "").strip(), str(request_text or "").strip()]
    return ", ".join(value for value in values if value)


def _build_specs(job_id: str, request: BatchGenerationRequest, *, checkpoint: str) -> None:
    """Persist the variation specs consumed by the 3C-11c worker."""

    base_prompt = _base_prompt(request.character, request.request_text) or request.base_prompt.strip()
    if not base_prompt:
        raise ValueError("characterまたはrequest_textを指定してください。")
    variation_count = request.effective_variation_count if request.variation_enabled else 1
    patterns = build_variation_patterns(variation_count) if request.variation_enabled else ((),)
    remaining = request.total_count
    for variation_index in range(1, variation_count + 1):
        if remaining <= 0:
            break
        count = min(request.effective_images_per_variation, remaining)
        suffix = ", ".join(patterns[variation_index - 1]) if request.variation_enabled else ""
        prompt = f"{base_prompt}, {suffix}" if suffix else base_prompt
        manager.create_batch(
            job_id,
            request.requested_by,
            prompt=prompt,
            negative_prompt=request.base_negative_prompt,
            character=request.character,
            loras=request.loras,
            checkpoint=checkpoint,
            requested_count=count,
            variation_index=variation_index,
        )
        remaining -= count


def _save_last_batch_request(request: BatchGenerationRequest) -> None:
    try:
        save_last_batch_settings(
            request.requested_by,
            {
                "character": request.character,
                "model_key": request.model_key,
                "loras": list(lora_selections_to_dicts(request.loras)),
                "request_text": request.request_text,
                "total_count": request.total_count,
                "variation_enabled": request.variation_enabled,
                "variation_count": request.effective_variation_count,
                "images_per_variation": request.effective_images_per_variation,
                "rating": request.rating,
            },
        )
    except Exception:
        logger.warning("Could not save last Batch settings.", exc_info=True)


def _format_job(job: dict[str, Any], *, heading: str = "大量生成ジョブ") -> str:
    status = str(job.get("status") or "unknown")
    completed = int(job.get("completed_count") or 0)
    failed = int(job.get("failed_count") or 0)
    total = int(job.get("total_count") or 0)
    lines = [
        heading,
        f"job_id: `{job.get('job_id', '-')}`",
        f"状態: **{status}**",
        f"進捗: {completed} / {total}枚",
        f"失敗: {failed}",
    ]
    current = str(job.get("current_batch_id") or "")
    if current:
        lines.append(f"現在のbatch: `{current}`")
    if status == "cancelled":
        lines.append("現在処理中の1枚が終わった後、未処理分を停止しました。")
    error = str(job.get("error") or "").strip()
    if error:
        lines.append(f"エラー: {error[:360]}")
    return "\n".join(lines)


_worker = BatchGenerationWorker()
_worker_tasks: dict[str, asyncio.Task[object]] = {}


def _task_done(job_id: str, task: asyncio.Task[object]) -> None:
    _worker_tasks.pop(job_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Batch worker task failed: job=%s error=%s", job_id, error, exc_info=error)


def start_job(job_id: str, requested_by: int) -> bool:
    """Schedule one persisted job once; the worker claims it atomically."""

    existing = _worker_tasks.get(job_id)
    if existing is not None and not existing.done():
        return False
    task = asyncio.create_task(
        asyncio.to_thread(_worker.run_job, job_id, requested_by),
        name=f"batch-generation-command-{job_id}",
    )
    _worker_tasks[job_id] = task
    task.add_done_callback(lambda completed: _task_done(job_id, completed))
    return True


def recover_incomplete_jobs() -> int:
    return recover_persisted_jobs()


async def shutdown() -> None:
    """Stop command-launched workers after the current image returns."""

    _worker.shutdown()
    tasks = list(_worker_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _worker_tasks.clear()


def _resolve_target(job_id: str, user_id: int) -> str | None:
    return job_id.strip() or manager.latest_job_id(user_id)


def setup(client: discord.Client) -> None:
    if client.tree.get_command("generate_batch") is not None:
        logger.warning("generate_batch commands are already registered; skipping duplicate setup.")
        return
    batch_group = app_commands.Group(
        name="generate_batch",
        description="複数枚の画像生成ジョブを管理します。",
    )

    async def reject_if_unavailable(interaction: discord.Interaction) -> bool:
        if is_pc_control_allowed(interaction.user.id):
            return False
        await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
        return True

    @batch_group.command(name="create", description="複数枚を順番に生成するBatch jobを作成します。")
    @app_commands.choices(
        model=MODEL_CHOICES,
        character_lora=CHARACTER_LORA_CHOICES,
        style_lora=STYLE_LORA_CHOICES,
        additional_lora=ADDITIONAL_LORA_CHOICES,
        rating=RATING_CHOICES,
    )
    @app_commands.describe(
        character="キャラクター名または生成対象",
        total_count="生成する総枚数（1～100、未指定時は前回値または6）",
        images_per_prompt=f"1 variationあたりの枚数（1～{MAX_IMAGES_PER_PROMPT}）",
        variation_count="variationの種類数",
        images_per_variation=f"各variationの枚数（1～{MAX_IMAGES_PER_PROMPT}）",
        request_text="追加のprompt。未指定時は前回値",
        model="未指定時は前回Batch→/generate_ai→既定値",
        character_lora="未指定時は前回値。なしで解除",
        style_lora="未指定時は前回値。なしで解除",
        additional_lora="未指定時は前回値。なしで解除",
        rating="未指定時は前回Batch→/generate_ai→Auto",
        variation_enabled="variationを使うか",
    )
    async def generate_batch_create(
        interaction: discord.Interaction,
        character: str,
        total_count: int | None = None,
        images_per_prompt: int | None = None,
        variation_count: int | None = None,
        images_per_variation: int | None = None,
        request_text: str | None = None,
        model: app_commands.Choice[str] | None = None,
        character_lora: app_commands.Choice[str] | None = None,
        style_lora: app_commands.Choice[str] | None = None,
        additional_lora: app_commands.Choice[str] | None = None,
        rating: app_commands.Choice[str] | None = None,
        variation_enabled: bool | None = None,
    ) -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            previous = get_last_batch_settings(interaction.user.id) or {}
            model_key, loras = await asyncio.to_thread(
                _resolve_model_and_loras,
                interaction.user.id,
                previous,
                model,
                character_lora,
                style_lora,
                additional_lora,
            )
            resolved_rating = await asyncio.to_thread(
                _resolve_rating, interaction.user.id, model_key, previous, rating
            )
            if variation_count is not None or images_per_variation is not None:
                if variation_count is None or images_per_variation is None:
                    raise ValueError("variation_countとimages_per_variationは両方指定してください。")
                resolved_variations = variation_count
                resolved_images = images_per_variation
                resolved_total = variation_count * images_per_variation
            else:
                resolved_total = total_count if total_count is not None else int(previous.get("total_count") or DEFAULT_TOTAL_COUNT)
                resolved_images = images_per_prompt if images_per_prompt is not None else int(previous.get("images_per_variation") or DEFAULT_IMAGES_PER_VARIATION)
                resolved_variations = int(previous["variation_count"]) if total_count is None and images_per_prompt is None and previous.get("variation_count") else None
            resolved_variation = variation_enabled if variation_enabled is not None else bool(previous.get("variation_enabled", True))
            effective_text = request_text if request_text is not None else str(previous.get("request_text") or "")
            request = BatchGenerationRequest(
                requested_by=interaction.user.id,
                character=character.strip(),
                total_count=resolved_total,
                images_per_prompt=resolved_images,
                images_per_variation=resolved_images,
                variation_count=resolved_variations,
                model_key=model_key,
                loras=loras,
                request_text=effective_text,
                variation_enabled=resolved_variation,
                rating=resolved_rating,
            )
            job_id = await asyncio.to_thread(manager.create_job, request)
            try:
                profile = get_image_generation_profile(model_key)
                await asyncio.to_thread(_build_specs, job_id, request, checkpoint=profile.checkpoint)
            except Exception:
                # A job without runnable specs must not remain queued.
                try:
                    manager.finish_job(job_id, interaction.user.id, status="failed", error="Batch specの保存に失敗しました。")
                except Exception:
                    logger.exception("Failed to close incomplete Batch job=%s", job_id)
                raise
            _save_last_batch_request(request)
            start_job(job_id, interaction.user.id)
            job = await asyncio.to_thread(manager.get_job_for_user, job_id, interaction.user.id)
        except (ValueError, BatchJobError, PromptGeneratorError) as error:
            await interaction.followup.send(f"大量生成ジョブを作成できませんでした。\n{error}", ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /generate_batch create failure.")
            await interaction.followup.send("大量生成ジョブの作成に失敗しました。Botログを確認してください。", ephemeral=True)
            return
        await interaction.followup.send(_format_job(job, heading="大量生成ジョブを受け付けました"), ephemeral=True)

    @batch_group.command(name="defaults", description="Batchの前回設定を明示的に保存します。")
    @app_commands.choices(
        model=MODEL_CHOICES,
        character_lora=CHARACTER_LORA_CHOICES,
        style_lora=STYLE_LORA_CHOICES,
        additional_lora=ADDITIONAL_LORA_CHOICES,
    )
    @app_commands.describe(
        character="次回Batchで使うキャラクターまたは生成対象",
        model="既定モデル",
        character_lora="既定character LoRA。なしで解除",
        style_lora="既定style LoRA。なしで解除",
        additional_lora="既定additional LoRA。なしで解除",
    )
    async def generate_batch_defaults(
        interaction: discord.Interaction,
        character: str,
        model: app_commands.Choice[str] | None = None,
        character_lora: app_commands.Choice[str] | None = None,
        style_lora: app_commands.Choice[str] | None = None,
        additional_lora: app_commands.Choice[str] | None = None,
    ) -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            previous = get_last_batch_settings(interaction.user.id) or {}
            model_key, loras = await asyncio.to_thread(
                _resolve_model_and_loras,
                interaction.user.id,
                previous,
                model,
                character_lora,
                style_lora,
                additional_lora,
            )
            save_last_batch_settings(
                interaction.user.id,
                {
                    **previous,
                    "character": character.strip(),
                    "model_key": model_key,
                    "loras": list(lora_selections_to_dicts(loras)),
                },
            )
        except (ValueError, BatchJobError) as error:
            await interaction.followup.send(f"Batchの既定設定を保存できませんでした。\n{error}", ephemeral=True)
            return
        await interaction.followup.send("Batchの既定設定を保存しました。次回createで未指定項目に利用します。", ephemeral=True)

    @batch_group.command(name="repeat", description="自分の前回Batch jobを同じ設定で再実行します。")
    @app_commands.describe(job_id="省略時は自分の最新job")
    async def generate_batch_repeat(interaction: discord.Interaction, job_id: str = "") -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        target = _resolve_target(job_id, interaction.user.id)
        if not target:
            await interaction.followup.send("繰り返せるBatch jobがありません。", ephemeral=True)
            return
        try:
            request = await asyncio.to_thread(manager.get_request_for_user, target, interaction.user.id)
            new_job_id = await asyncio.to_thread(manager.repeat_job, target, interaction.user.id)
            repeated = await asyncio.to_thread(manager.get_request_for_user, new_job_id, interaction.user.id)
            profile = get_image_generation_profile(repeated.model_key)
            try:
                await asyncio.to_thread(_build_specs, new_job_id, repeated, checkpoint=profile.checkpoint)
            except Exception:
                try:
                    manager.finish_job(
                        new_job_id,
                        interaction.user.id,
                        status="failed",
                        error="Batch specの保存に失敗しました。",
                    )
                except Exception:
                    logger.exception("Failed to close incomplete repeated Batch job=%s", new_job_id)
                raise
            _save_last_batch_request(request)
            start_job(new_job_id, interaction.user.id)
            job = await asyncio.to_thread(manager.get_job_for_user, new_job_id, interaction.user.id)
        except (ValueError, BatchJobError) as error:
            await interaction.followup.send(f"再生成を開始できませんでした。\n{error}", ephemeral=True)
            return
        await interaction.followup.send(_format_job(job, heading="同じ設定のBatchを受け付けました"), ephemeral=True)

    @batch_group.command(name="status", description="自分のBatch jobの進捗を表示します。")
    @app_commands.describe(job_id="省略時は自分の最新job")
    async def generate_batch_status(interaction: discord.Interaction, job_id: str = "") -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        target = _resolve_target(job_id, interaction.user.id)
        if not target:
            await interaction.followup.send("確認できるBatch jobがありません。", ephemeral=True)
            return
        try:
            job = await asyncio.to_thread(manager.get_job_for_user, target, interaction.user.id)
        except (BatchJobNotFoundError, BatchJobOwnershipError) as error:
            await interaction.followup.send(str(error), ephemeral=True)
            return
        await interaction.followup.send(_format_job(job), ephemeral=True)

    @batch_group.command(name="cancel", description="現在の画像完了後に未処理Batchを停止します。")
    @app_commands.describe(job_id="省略時は自分の最新job")
    async def generate_batch_cancel(interaction: discord.Interaction, job_id: str = "") -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        target = _resolve_target(job_id, interaction.user.id)
        if not target:
            await interaction.followup.send("キャンセルできるBatch jobがありません。", ephemeral=True)
            return
        try:
            accepted, message = await asyncio.to_thread(manager.cancel_job, target, interaction.user.id)
            if accepted:
                _worker.request_stop(target)
            await interaction.followup.send(f"job_id `{target}`: {message}", ephemeral=True)
        except (BatchJobNotFoundError, BatchJobOwnershipError) as error:
            await interaction.followup.send(str(error), ephemeral=True)

    client.tree.add_command(batch_group)
    logger.info("Registered generate_batch create/defaults/repeat/status/cancel commands.")
