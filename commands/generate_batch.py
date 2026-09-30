"""Discord commands for the persisted, sequential Batch worker.

The command layer only resolves user input, persists a small set of Batch
specs, and hands the job to :mod:`services.batch_generation_worker`.  Prompt
construction and image generation remain in the existing Batch adapter.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from services.gallery_control import (
    GalleryStatus,
    gallery_job_url,
    get_gallery_job_summary,
    get_gallery_status,
    start_gallery,
)
from services.gallery_review_queue import (
    enqueue_gallery_review,
    shutdown_gallery_review_queue,
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
PROGRESS_POLL_SECONDS = 5
PROGRESS_EDIT_INTERVAL_SECONDS = 10
# Gallery's async summary endpoint can return the last committed index while
# a background filesystem scan is still running.  This wait belongs only to
# the optional Gallery follow-up; the Batch completion notice is published
# before it starts.
GALLERY_SUMMARY_WAIT_SECONDS = 45.0
GALLERY_SUMMARY_INITIAL_BACKOFF_SECONDS = 1.0
GALLERY_SUMMARY_MAX_BACKOFF_SECONDS = 5.0


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
                "auto_review_enabled": request.auto_review_enabled,
                "rating": request.rating,
            },
        )
    except Exception:
        logger.warning("Could not save last Batch settings.", exc_info=True)


def _format_job(
    job: dict[str, Any],
    *,
    heading: str = "大量生成ジョブ",
    summary: dict[str, Any] | None = None,
    gallery_error: str = "",
) -> str:
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
    review_status = str(job.get("review_status") or "not_requested")
    if review_status != "not_requested":
        review_job_id = str(job.get("review_job_id") or "")
        suffix = f" ({review_job_id})" if review_job_id else ""
        lines.append(f"Gallery AIレビュー: {review_status}{suffix}")
    if summary is not None:
        lines.append(
            "Gallery: "
            f"{int(summary.get('total') or 0)}枚 / "
            f"AI評価済み {int(summary.get('ai_scored') or 0)}枚"
        )
    if status == "cancelled":
        lines.append("現在処理中の1枚が終わった後、未処理分を停止しました。")
    error = str(job.get("error") or "").strip()
    if error:
        lines.append(f"エラー: {error[:360]}")
    review_error = str(job.get("review_error") or "").strip()
    if review_error:
        lines.append(f"Galleryレビュー: {review_error[:360]}")
    if gallery_error:
        lines.append(f"Galleryリンク/summaryを取得できませんでした。/gallery open を試してください。")
    return "\n".join(lines)


_worker = BatchGenerationWorker()
_worker_tasks: dict[str, asyncio.Task[object]] = {}
_review_tasks: dict[str, asyncio.Task[object]] = {}
_watch_tasks: dict[str, asyncio.Task[object]] = {}
_gallery_followup_tasks: dict[str, asyncio.Task[object]] = {}
_shutting_down = False


def _review_task_done(job_id: str, task: asyncio.Task[object]) -> None:
    _review_tasks.pop(job_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Batch Gallery review task failed: job=%s error=%s", job_id, error, exc_info=error)


def _watch_task_done(job_id: str, task: asyncio.Task[object]) -> None:
    _watch_tasks.pop(job_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Batch progress watcher failed: job=%s error=%s", job_id, error, exc_info=error)


def _gallery_followup_task_done(job_id: str, task: asyncio.Task[object]) -> None:
    _gallery_followup_tasks.pop(job_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Batch Gallery follow-up failed: job=%s error=%s",
            job_id,
            error,
            exc_info=error,
        )


def _generation_task_done(job_id: str, requested_by: int, task: asyncio.Task[object]) -> None:
    _worker_tasks.pop(job_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Batch worker task failed: job=%s error=%s", job_id, error, exc_info=error)
        return
    if _shutting_down:
        return
    review_task = asyncio.create_task(
        _post_generation_review(job_id, requested_by),
        name=f"batch-gallery-review-{job_id}",
    )
    _review_tasks[job_id] = review_task
    review_task.add_done_callback(lambda completed: _review_task_done(job_id, completed))


async def _run_generation_job(job_id: str, requested_by: int) -> object:
    return await asyncio.to_thread(_worker.run_job, job_id, requested_by)


def start_job(job_id: str, requested_by: int) -> bool:
    """Schedule one persisted job once; the worker claims it atomically."""

    existing = _worker_tasks.get(job_id)
    if existing is not None and not existing.done():
        return False
    task = asyncio.create_task(
        _run_generation_job(job_id, requested_by),
        name=f"batch-generation-command-{job_id}",
    )
    _worker_tasks[job_id] = task
    task.add_done_callback(
        lambda completed: _generation_task_done(job_id, requested_by, completed)
    )
    return True


def recover_incomplete_jobs() -> int:
    return recover_persisted_jobs()


async def shutdown() -> None:
    """Stop command-launched workers after the current image returns."""

    global _shutting_down
    _shutting_down = True
    _worker.shutdown()
    for task in list(_watch_tasks.values()):
        task.cancel()
    watch_tasks = list(_watch_tasks.values())
    if watch_tasks:
        await asyncio.gather(*watch_tasks, return_exceptions=True)
    generation_tasks = list(_worker_tasks.values())
    if generation_tasks:
        await asyncio.gather(*generation_tasks, return_exceptions=True)
    for task in list(_review_tasks.values()):
        task.cancel()
    review_tasks = list(_review_tasks.values())
    if review_tasks:
        await asyncio.gather(*review_tasks, return_exceptions=True)
    for task in list(_gallery_followup_tasks.values()):
        task.cancel()
    gallery_followup_tasks = list(_gallery_followup_tasks.values())
    if gallery_followup_tasks:
        await asyncio.gather(*gallery_followup_tasks, return_exceptions=True)
    await shutdown_gallery_review_queue()
    _worker_tasks.clear()
    _review_tasks.clear()
    _watch_tasks.clear()
    _gallery_followup_tasks.clear()


def _review_state_callback(job_id: str, requested_by: int):
    def callback(state: str, details: dict[str, Any]) -> None:
        normalized = "running" if state == "starting" else state
        try:
            manager.update_review_status(
                job_id,
                requested_by,
                status=normalized,
                review_job_id=str(details.get("review_job_id") or ""),
                error=str(details.get("error") or "") if normalized == "failed" else "",
            )
        except Exception:
            logger.warning("Could not persist Batch Gallery review state: job=%s state=%s", job_id, state, exc_info=True)

    return callback


async def _post_generation_review(job_id: str, requested_by: int) -> None:
    """Start optional Gallery review after generation without changing job state."""

    try:
        job = await asyncio.to_thread(manager.get_job_for_user, job_id, requested_by)
        if not job.get("auto_review_enabled") or int(job.get("completed_count") or 0) <= 0:
            if job.get("auto_review_enabled") and int(job.get("completed_count") or 0) <= 0:
                await asyncio.to_thread(
                    manager.update_review_status,
                    job_id,
                    requested_by,
                    status="failed",
                    error="AI評価対象の生成画像がありません。",
                )
            return
        expected = int(job["completed_count"])
        await asyncio.to_thread(
            manager.update_review_status,
            job_id,
            requested_by,
            status="queued",
        )
        future = enqueue_gallery_review(
            job_id,
            expected,
            on_state=_review_state_callback(job_id, requested_by),
        )
        result = await future
        if str(result.get("status") or "") == "completed":
            await asyncio.to_thread(
                manager.update_review_status,
                job_id,
                requested_by,
                status="completed",
                review_job_id=str(result.get("review_job_id") or ""),
            )
        else:
            await asyncio.to_thread(
                manager.update_review_status,
                job_id,
                requested_by,
                status="failed",
                review_job_id=str(result.get("review_job_id") or ""),
                error=str(result.get("error") or "Gallery AIレビューに失敗しました。"),
            )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning("Batch Gallery review was skipped: job=%s error=%s", job_id, error, exc_info=True)
        try:
            await asyncio.to_thread(
                manager.update_review_status,
                job_id,
                requested_by,
                status="failed",
                error=str(error),
            )
        except Exception:
            logger.warning("Could not persist Batch Gallery review failure: job=%s", job_id, exc_info=True)


class BatchGalleryView(discord.ui.View):
    def __init__(self, owner_id: int, job_id: str, url: str) -> None:
        super().__init__(timeout=24 * 60 * 60)
        self.owner_id = owner_id
        self.job_id = job_id
        self.add_item(
            discord.ui.Button(
                label="GalleryでBatchを見る",
                style=discord.ButtonStyle.link,
                url=gallery_job_url(url, job_id),
            )
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "このボタンは実行者本人だけが使えます。", ephemeral=True
            )
            return False
        return True


def _build_gallery_view(
    owner_id: int,
    job_id: str,
    status: GalleryStatus | None,
) -> BatchGalleryView | None:
    if status is None:
        return None
    urls = status.urls
    url = urls.tailscale[0] if urls.tailscale else urls.local
    return BatchGalleryView(owner_id, job_id, url) if url else None


async def _terminal_gallery_context(
    job_id: str,
    *,
    start: bool,
    expected_count: int = 0,
) -> tuple[GalleryStatus | None, dict[str, Any] | None, str]:
    status: GalleryStatus | None = None
    error_text = ""
    try:
        if start:
            started = await asyncio.to_thread(start_gallery)
            status = started.status
        else:
            status = await asyncio.to_thread(get_gallery_status)
    except Exception as error:
        error_text = str(error)
        logger.warning("Could not prepare Gallery link for Batch job=%s: %s", job_id, error)
        try:
            status = await asyncio.to_thread(get_gallery_status)
        except Exception:
            logger.warning("Could not read Gallery status for Batch job=%s", job_id, exc_info=True)
    summary, summary_error = await _get_gallery_summary_for_batch(
        job_id,
        expected_count=expected_count,
    )
    if summary_error:
        error_text = error_text or summary_error
    return status, summary, error_text


async def _get_gallery_summary_for_batch(
    job_id: str,
    *,
    expected_count: int,
) -> tuple[dict[str, Any] | None, str]:
    """Read a Batch summary without making Gallery part of Batch completion.

    ``/api/images/summary`` is intentionally backed by Gallery's last indexed
    state.  A successful response with ``total=0`` therefore means
    "not indexed yet" when the Batch already produced images; it is not a
    valid terminal summary.  Retry only this optional lookup for a bounded
    period.  A Batch with no successful images is never made to wait for an
    index that cannot appear.
    """

    expected = max(int(expected_count or 0), 0)
    deadline = time.monotonic() + GALLERY_SUMMARY_WAIT_SECONDS
    delay = GALLERY_SUMMARY_INITIAL_BACKOFF_SECONDS
    last_error = ""

    while True:
        try:
            summary = await asyncio.to_thread(get_gallery_job_summary, job_id)
            total = int(summary.get("total") or 0)
            if expected == 0 or total >= expected:
                return summary, ""
            # The endpoint responded successfully, but its committed index
            # is behind the Batch result.  Do not expose this stale summary as
            # a completed Gallery result.
            last_error = f"Gallery索引の反映待ちです ({total}/{expected})。"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            last_error = str(error)

        # A genuinely empty Batch is terminal for this best-effort follow-up.
        # Successful Batches with missing index entries get only the bounded
        # async-sync grace period, including transient Gallery errors.
        if expected == 0:
            logger.warning("Could not read Gallery summary for Batch job=%s: %s", job_id, last_error)
            return None, last_error

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Gallery summary did not reach %s images before timeout: job=%s last=%s",
                expected,
                job_id,
                last_error,
            )
            return None, last_error or "Gallery summaryの反映待ちがタイムアウトしました。"

        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, GALLERY_SUMMARY_MAX_BACKOFF_SECONDS)


async def _send_gallery_followup(
    interaction: discord.Interaction,
    job_id: str,
) -> None:
    """Send optional Gallery information after the terminal Batch notice.

    This task is deliberately one-shot.  Gallery review ingestion/retry stays
    in ``gallery_review_queue``; this task only attempts to add a link/summary
    without delaying or changing the already-published generation result.
    """

    try:
        job = await asyncio.to_thread(
            manager.get_job_for_user,
            job_id,
            interaction.user.id,
        )
        status, summary, gallery_error = await _terminal_gallery_context(
            job_id,
            start=int(job.get("completed_count") or 0) > 0,
            expected_count=int(job.get("completed_count") or 0),
        )
        if status is None and summary is None:
            if gallery_error:
                logger.warning(
                    "Gallery follow-up unavailable: job=%s error=%s",
                    job_id,
                    gallery_error,
                )
            return
        view = _build_gallery_view(interaction.user.id, job_id, status)
        await interaction.followup.send(
            _format_job(
                job,
                heading="Gallery情報",
                summary=summary,
                gallery_error=gallery_error,
            ),
            view=view,
            ephemeral=True,
            wait=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Could not publish Batch Gallery follow-up: job=%s",
            job_id,
            exc_info=True,
        )


def _schedule_gallery_followup(
    interaction: discord.Interaction,
    job_id: str,
) -> None:
    if _shutting_down:
        return
    existing = _gallery_followup_tasks.get(job_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _send_gallery_followup(interaction, job_id),
        name=f"batch-gallery-followup-{job_id}",
    )
    _gallery_followup_tasks[job_id] = task
    task.add_done_callback(
        lambda completed: _gallery_followup_task_done(job_id, completed)
    )


async def _publish_progress(
    interaction: discord.Interaction,
    message: discord.Message | None,
    content: str,
    *,
    view: discord.ui.View | None = None,
) -> discord.Message | None:
    if message is not None:
        try:
            await message.edit(content=content, view=view)
            return message
        except Exception:
            logger.warning("Could not edit Batch progress message.", exc_info=True)
    try:
        return await interaction.followup.send(content, view=view, ephemeral=True, wait=True)
    except Exception:
        logger.warning("Could not send Batch progress message.", exc_info=True)
        return message


async def _watch_job(
    interaction: discord.Interaction,
    job_id: str,
    progress_message: discord.Message | None,
) -> None:
    """Edit one acknowledgement message and publish one terminal summary."""

    last_signature = ""
    last_edit = 0.0
    completion_published = False
    try:
        for _ in range(7200):
            await asyncio.sleep(PROGRESS_POLL_SECONDS)
            job = await asyncio.to_thread(
                manager.get_job_for_user, job_id, interaction.user.id
            )
            status = str(job.get("status") or "unknown")
            terminal = status in {"completed", "failed", "cancelled"}
            review_status = str(job.get("review_status") or "not_requested")
            review_pending = (
                bool(job.get("auto_review_enabled"))
                and int(job.get("completed_count") or 0) > 0
                and review_status in {
                    "not_requested",
                    "queued",
                    "starting",
                    "running",
                }
            )
            signature = ":".join(
                str(job.get(key) or "")
                for key in (
                    "status",
                    "completed_count",
                    "failed_count",
                    "current_batch_id",
                    "review_status",
                    "review_job_id",
                )
            )
            now = time.monotonic()
            if terminal:
                heading = "大量生成完了"
                if status == "failed":
                    heading = "大量生成失敗"
                elif status == "cancelled":
                    heading = "大量生成をキャンセルしました"
                if not completion_published:
                    progress_message = await _publish_progress(
                        interaction,
                        progress_message,
                        _format_job(job, heading=heading),
                    )
                    completion_published = True
                if review_pending:
                    # The result is already announced.  Wait only for the
                    # existing review queue's terminal state before asking
                    # for optional Gallery context; no Gallery API call is
                    # made from this polling loop.
                    await asyncio.sleep(PROGRESS_POLL_SECONDS)
                    continue
                _schedule_gallery_followup(interaction, job_id)
                return
            if signature != last_signature and now - last_edit >= PROGRESS_EDIT_INTERVAL_SECONDS:
                heading = "大量生成中"
                if terminal and review_pending:
                    heading = "生成完了・Gallery AIレビュー中"
                progress_message = await _publish_progress(
                    interaction,
                    progress_message,
                    _format_job(job, heading=heading),
                )
                last_signature = signature
                last_edit = now
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Batch progress watcher stopped: job=%s", job_id, exc_info=True)
    finally:
        _watch_tasks.pop(job_id, None)


def _start_progress_watch(
    interaction: discord.Interaction,
    job_id: str,
    progress_message: discord.Message | None,
) -> None:
    existing = _watch_tasks.get(job_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _watch_job(interaction, job_id, progress_message),
        name=f"batch-progress-{job_id}",
    )
    _watch_tasks[job_id] = task
    task.add_done_callback(lambda completed: _watch_task_done(job_id, completed))


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
        auto_review_enabled="生成完了後、このjobだけGallery AIレビュー。未指定時は前回値",
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
        auto_review_enabled: bool | None = None,
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
            resolved_auto_review = (
                auto_review_enabled
                if auto_review_enabled is not None
                else bool(previous.get("auto_review_enabled", False))
            )
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
                auto_review_enabled=resolved_auto_review,
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
        progress_message = await interaction.followup.send(
            _format_job(job, heading="大量生成ジョブを受け付けました"),
            ephemeral=True,
            wait=True,
        )
        _start_progress_watch(interaction, job_id, progress_message)

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
        progress_message = await interaction.followup.send(
            _format_job(job, heading="同じ設定のBatchを受け付けました"),
            ephemeral=True,
            wait=True,
        )
        _start_progress_watch(interaction, new_job_id, progress_message)

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
        gallery_status = None
        summary = None
        gallery_error = ""
        if str(job.get("status") or "") in {"completed", "failed", "cancelled"}:
            gallery_status, summary, gallery_error = await _terminal_gallery_context(
                target,
                start=False,
                expected_count=int(job.get("completed_count") or 0),
            )
        view = _build_gallery_view(interaction.user.id, target, gallery_status)
        await interaction.followup.send(
            _format_job(job, summary=summary, gallery_error=gallery_error),
            view=view,
            ephemeral=True,
        )

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
