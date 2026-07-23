import asyncio
import logging
from pathlib import Path

import discord
from discord import app_commands

from sd_prompt_template_generator import build_discord_prompt_message
from services.generation_history import (
    GenerationRequestRecord,
    SavedGeneration,
    resolve_image_ids_for_paths,
    save_generation_history_detailed,
    set_image_discord_message,
    upsert_image_rating,
)
from services.image_selection import (
    get_selection_generate_count,
    is_image_selection_available,
    select_generated_images,
)
from services.pc_control import is_pc_control_allowed
from services.prompt_generator import PromptGeneratorError, PromptGeneratorTimeout, generate_sd_prompt_result
from services.sd_control import (
    GeneratedImage,
    SdGenerationOptions,
    generate_images,
    generate_images_with_metadata,
    get_negative_prompt,
)
from services.shogou_prompt_designer import ShogouPromptResult, generate_shogou_prompt_result


logger = logging.getLogger("secretary-bot.prompt")

SD_DENIED_MESSAGE = "画像生成は、許可されたユーザーだけが実行できます。"
ONLY_OWNER_MESSAGE = "この画像生成ボタンは、/prompt を実行した本人だけが押せます。"
GENERATING_MESSAGE = "画像生成中です。少し待っててね。"
RAW_PROMPT_MAX_LENGTH = 4000
SHOGOU_MAX_ASSUMPTIONS_DISPLAY = 5
SHOGOU_ASSUMPTION_ITEM_LIMIT = 200
SHOGOU_SUMMARY_LIMIT = 300
SHOGOU_MESSAGE_LIMIT = 1900


# Discord allows at most 10 attachments per message.
MAX_SEND_IMAGE_FILES = 10
RATING_ORDER = ("like", "meh", "off_target")
RATING_VIEW_TIMEOUT_SECONDS = 3600
RATING_LABELS = {"like": "好き", "meh": "微妙", "off_target": "想定と違う"}
RATING_EMOJI = {"like": "👍", "meh": "👎", "off_target": "🎯"}
# 3 buttons per image, so 1 image per action row (3 buttons/row, within
# Discord's 5-per-row limit). With Discord's 5-row cap, only the first 5
# rateable images (in attachment order) get buttons; the rest are still
# sent, just without rating buttons.
RATING_IMAGES_PER_ROW = 1
MAX_RATEABLE_IMAGES = 5 * RATING_IMAGES_PER_ROW


class ImageRatingButton(discord.ui.Button):
    def __init__(self, *, image_id: int, rating: str, display_index: int, row: int) -> None:
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label=f"{RATING_EMOJI[rating]} {RATING_LABELS[rating]} {display_index}",
            custom_id=f"genai_rate:{image_id}:{rating}",
            row=row,
        )
        self.image_id = image_id
        self.rating = rating

    async def callback(self, interaction: discord.Interaction) -> None:
        # Anyone may press: image_ratings is keyed by (image_id, user_id), so
        # each presser's rating is stored as their own, and pressing again
        # replaces only that user's previous value.
        try:
            stored = await asyncio.to_thread(
                upsert_image_rating, self.image_id, interaction.user.id, self.rating
            )
        except Exception:
            logger.exception("Unexpected error while saving image rating: image_id=%s", self.image_id)
            stored = False

        if stored:
            text = f"{RATING_LABELS[self.rating]}として保存しました。"
        else:
            # Never surface the DB error itself to Discord.
            text = "評価を保存できませんでした。時間をおいてもう一度試してね。"
        try:
            await interaction.response.send_message(text, ephemeral=True)
        except Exception:
            logger.exception("Failed to reply to rating button: image_id=%s", self.image_id)


class ImageRatingView(discord.ui.View):
    """Buttons for rating the images /generate_ai actually sent.

    Not a persistent view: after RATING_VIEW_TIMEOUT_SECONDS (or a bot
    restart) the buttons stop working. Restoring them is out of scope for
    Phase 1B; the generated images and saved history are unaffected.
    """

    def __init__(self, rateable_images: list[tuple[int, int]]) -> None:
        super().__init__(timeout=RATING_VIEW_TIMEOUT_SECONDS)
        self.message: discord.Message | None = None
        self.rateable_images = rateable_images
        for position, (display_index, image_id) in enumerate(rateable_images[:MAX_RATEABLE_IMAGES]):
            row = position // RATING_IMAGES_PER_ROW
            for rating in RATING_ORDER:
                self.add_item(
                    ImageRatingButton(
                        image_id=image_id,
                        rating=rating,
                        display_index=display_index,
                        row=row,
                    )
                )

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True
        if self.message is None:
            return
        try:
            await self.message.edit(view=self)
        except Exception:
            # The message may be gone or no longer editable; the already
            # delivered images and stored ratings still stand.
            logger.info("Could not disable rating buttons after timeout.", exc_info=True)


async def _safe_followup_send(
    interaction: discord.Interaction,
    *,
    label: str,
    content: str | None = None,
    files: list[discord.File] | None = None,
    view: discord.ui.View | None = None,
    ephemeral: bool = True,
    wait: bool = False,
) -> "discord.Message | None":
    logger.info(
        "Discord followup.send before: label=%s content_length=%s file_count=%s has_view=%s ephemeral=%s",
        label,
        len(content or ""),
        len(files or []),
        view is not None,
        ephemeral,
    )
    try:
        kwargs = {
            "ephemeral": ephemeral,
            "allowed_mentions": discord.AllowedMentions.none(),
        }
        if content is not None:
            kwargs["content"] = content
        if files is not None:
            kwargs["files"] = files
        if view is not None:
            kwargs["view"] = view
        # wait=True is only requested by callers that need the resulting
        # Message back; existing callers keep the previous fire-and-forget
        # behavior untouched.
        if wait:
            kwargs["wait"] = True
        return await interaction.followup.send(**kwargs)
    except Exception:
        logger.exception("Discord followup.send failed: label=%s", label)
        return None
        raise
    logger.info("Discord followup.send after: label=%s", label)


async def _send_error(interaction: discord.Interaction, message: str, *, label: str) -> None:
    try:
        await _safe_followup_send(interaction, label=label, content=message, ephemeral=True)
    except Exception:
        logger.exception("Failed to send error response: label=%s", label)


async def _send_image_files(
    interaction: discord.Interaction,
    message: str,
    paths: list[Path],
    *,
    ephemeral: bool = True,
    view: discord.ui.View | None = None,
    wait: bool = False,
) -> "discord.Message | None":
    # view/wait are optional additions for /generate_ai's rating buttons.
    # Callers that omit them (/generate, /generate_raw, /character_generate)
    # behave exactly as before and can keep ignoring the return value.
    files: list[discord.File] = []
    try:
        for path in list(paths)[:MAX_SEND_IMAGE_FILES]:
            files.append(discord.File(str(path), filename=path.name))
        return await _safe_followup_send(
            interaction,
            label="prompt.image_files",
            content=message,
            files=files,
            ephemeral=ephemeral,
            view=view,
            wait=wait,
        )
    finally:
        for file in files:
            file.close()


def _selection_message(selection, send_count: int, generated_count: int) -> str:
    if generated_count <= 1:
        return "画像を生成しました。"
    if selection.used_selection:
        return selection.message or f"画像選別AIで上位{send_count}枚を選びました。"
    if selection.error:
        return f"画像選別AIは使えませんでした: {selection.error}\n通常の生成画像を返すね。"
    if selection.message:
        return f"{selection.message}\n通常の生成画像を返すね。"
    return f"画像を{generated_count}枚生成しました。"


def _truncate_prompt(value: str, limit: int = 900) -> str:
    text = value.strip()
    if len(text) <= limit:
        truncated = text
    else:
        truncated = text[:limit].rstrip() + "..."
    return truncated.replace("```", "``\u200b`")


def _build_generated_image_message(
    positive_prompt: str,
    negative_prompt: str,
    selection,
    send_count: int,
    generated_count: int,
) -> str:
    selection_text = _selection_message(selection, send_count, generated_count)
    return (
        f"{selection_text}\n\n"
        "Positive:\n"
        f"```text\n{_truncate_prompt(positive_prompt)}\n```\n"
        "Negative:\n"
        f"```text\n{_truncate_prompt(negative_prompt)}\n```"
    )


def _build_generation_options(
    positive_prompt: str,
    negative_prompt: str,
    *,
    preserve_prompt_text: bool = False,
) -> SdGenerationOptions:
    selection_available = is_image_selection_available()
    count = get_selection_generate_count() if selection_available else 1
    logger.info(
        "Prompt image generation count decided: count=%s selection_available=%s",
        count,
        selection_available,
    )
    return SdGenerationOptions(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        preserve_prompt_text=preserve_prompt_text,
        count=count,
        width=512,
        height=512,
        steps=20,
        cfg=7.0,
        sampler="Euler a",
        seed=-1,
    )


async def _generate_images_from_prompts(
    interaction: discord.Interaction,
    *,
    positive_prompt: str,
    negative_prompt: str,
    label_prefix: str,
    preserve_prompt_text: bool = False,
    result_ephemeral: bool = True,
) -> None:
    options = _build_generation_options(
        positive_prompt,
        negative_prompt,
        preserve_prompt_text=preserve_prompt_text,
    )
    try:
        paths = await asyncio.to_thread(generate_images, options)
    except Exception as exc:
        logger.exception("Failed to generate image: label=%s", label_prefix)
        error_text = str(exc).strip() or "unknown error"
        await _send_error(
            interaction,
            f"画像生成に失敗しました: {error_text[:300]}",
            label=f"{label_prefix}.generate.error",
        )
        return

    selection = await asyncio.to_thread(select_generated_images, paths)
    send_paths = selection.selected_paths or paths
    message = _build_generated_image_message(
        positive_prompt,
        negative_prompt,
        selection,
        len(send_paths),
        len(paths),
    )
    await _send_image_files(interaction, message, send_paths, ephemeral=result_ephemeral)


def _format_shogou_assumptions(assumptions: tuple[str, ...]) -> str:
    if not assumptions:
        return "（なし）"
    shown = assumptions[:SHOGOU_MAX_ASSUMPTIONS_DISPLAY]
    lines = [f"- {_truncate_prompt(item, limit=SHOGOU_ASSUMPTION_ITEM_LIMIT)}" for item in shown]
    remaining = len(assumptions) - len(shown)
    if remaining > 0:
        lines.append(f"...(他{remaining}件省略)")
    return "\n".join(lines)


def _build_shogou_generated_image_message(
    result: ShogouPromptResult,
    selection,
    send_count: int,
    generated_count: int,
) -> str:
    selection_text = _selection_message(selection, send_count, generated_count)
    message = (
        f"{selection_text}\n\n"
        "Summary:\n"
        f"{_truncate_prompt(result.summary, limit=SHOGOU_SUMMARY_LIMIT)}\n\n"
        "Assumptions:\n"
        f"{_format_shogou_assumptions(result.assumptions)}\n\n"
        "Positive:\n"
        f"```text\n{_truncate_prompt(result.positive)}\n```\n"
        "Negative:\n"
        f"```text\n{_truncate_prompt(result.negative)}\n```"
    )
    if len(message) > SHOGOU_MESSAGE_LIMIT:
        message = message[: SHOGOU_MESSAGE_LIMIT - 3].rstrip() + "..."
    return message


async def _save_generation_history_safely(
    *,
    user_id: int,
    request_text: str,
    reference_id: int | None,
    result: ShogouPromptResult,
    generated_images: list[GeneratedImage],
    selection,
) -> SavedGeneration | None:
    # Best-effort only: a history-save failure must never block the Discord
    # result from being sent, and must never crash the bot. Any exception
    # here (including one raised while building the record itself, before
    # save_generation_history_detailed's own internal try/except is even
    # reached) is caught and logged.
    try:
        workflow_file = generated_images[0].workflow_file if generated_images else None
        workflow_hash = generated_images[0].workflow_hash if generated_images else None
        request_record = GenerationRequestRecord(
            user_id=user_id,
            request_text=request_text,
            reference_id=reference_id,
            positive=result.positive,
            negative=result.negative,
            summary=result.summary,
            assumptions=result.assumptions,
            workflow_file=workflow_file,
            workflow_hash=workflow_hash,
            selection_used=selection.used_selection,
            selection_message=selection.message or selection.error or "",
        )
        saved = await asyncio.to_thread(
            save_generation_history_detailed,
            request_record,
            generated_images,
            selected_paths=selection.selected_paths,
        )
        if saved is None:
            logger.error(
                "Generation history save reported failure for /generate_ai "
                "(see services.generation_history logs); continuing without blocking Discord delivery."
            )
        return saved
    except Exception:
        logger.exception(
            "Unexpected error while saving generation history; continuing without blocking Discord delivery."
        )
        return None


async def _build_rating_view(
    saved: SavedGeneration | None,
    generated_images: list[GeneratedImage],
    send_paths: list[Path],
) -> tuple["ImageRatingView | None", list[int]]:
    """Builds the rating buttons for the images actually sent to Discord.

    Only sent images get buttons - anything Selection AI dropped is never
    rateable. display_index follows the attachment order so the button label
    matches what the user sees, even when one image fails to map.
    """
    if saved is None or not saved.image_ids:
        return None, []
    try:
        matched = await asyncio.to_thread(
            resolve_image_ids_for_paths,
            generated_images,
            saved.image_ids,
            send_paths[:MAX_SEND_IMAGE_FILES],
        )
    except Exception:
        logger.exception("Failed to map sent images to history rows; skipping rating buttons.")
        return None, []

    rateable = [
        (display_index, image_id)
        for display_index, image_id in enumerate(matched, start=1)
        if image_id is not None
    ]
    if not rateable:
        logger.warning("No sent image could be matched to a history row; rating buttons skipped.")
        return None, []
    return ImageRatingView(rateable), [image_id for _, image_id in rateable]


async def _store_discord_message_safely(
    image_ids: list[int],
    sent_message: "discord.Message | None",
) -> None:
    # Bookkeeping only: a failure here must not affect the delivered images
    # or the rating buttons that are already live.
    if sent_message is None or not image_ids:
        return
    try:
        channel = getattr(sent_message, "channel", None)
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            logger.info("Sent message has no channel id; skipping Discord message bookkeeping.")
            return
        await asyncio.to_thread(
            set_image_discord_message, image_ids, sent_message.id, channel_id
        )
    except Exception:
        logger.exception("Failed to store Discord message info for rated images.")


async def _generate_images_from_shogou_result(
    interaction: discord.Interaction,
    result: ShogouPromptResult,
    *,
    label_prefix: str,
    request_text: str = "",
    reference_id: int | None = None,
    result_ephemeral: bool = False,
) -> None:
    # Mirrors _generate_images_from_prompts' preserve_prompt_text=True path exactly,
    # but keeps its own message builder so /generate_raw's behavior stays untouched.
    options = _build_generation_options(
        result.positive,
        result.negative,
        preserve_prompt_text=True,
    )
    try:
        generated_images = await asyncio.to_thread(generate_images_with_metadata, options)
    except Exception as exc:
        logger.exception("Failed to generate image: label=%s", label_prefix)
        error_text = str(exc).strip() or "unknown error"
        await _send_error(
            interaction,
            f"画像生成に失敗しました: {error_text[:300]}",
            label=f"{label_prefix}.generate.error",
        )
        return

    paths = [image.path for image in generated_images]
    selection = await asyncio.to_thread(select_generated_images, paths)
    send_paths = selection.selected_paths or paths
    message = _build_shogou_generated_image_message(
        result,
        selection,
        len(send_paths),
        len(paths),
    )

    saved = await _save_generation_history_safely(
        user_id=interaction.user.id,
        request_text=request_text,
        reference_id=reference_id,
        result=result,
        generated_images=generated_images,
        selection=selection,
    )

    view, rated_image_ids = await _build_rating_view(saved, generated_images, send_paths)
    sent_message = await _send_image_files(
        interaction,
        message,
        send_paths,
        ephemeral=result_ephemeral,
        view=view,
        # Only ask Discord for the Message back when we have something to
        # record against it.
        wait=view is not None,
    )
    if view is not None:
        view.message = sent_message
        await _store_discord_message_safely(rated_image_ids, sent_message)


async def _build_shogou_prompt_result(
    interaction: discord.Interaction,
    text: str,
    *,
    label_prefix: str,
    reference_id: int | None = None,
) -> ShogouPromptResult | None:
    try:
        result = await asyncio.to_thread(generate_shogou_prompt_result, text, interaction.user.id, reference_id)
        logger.info(
            "%s result received: positive_length=%s negative_length=%s blocked=%s",
            label_prefix,
            len(result.positive),
            len(result.negative),
            result.blocked,
        )
        return result
    except PromptGeneratorTimeout:
        logger.exception("%s timed out.", label_prefix)
        await _send_error(interaction, "LM Studioから応答がありませんでした。", label=f"{label_prefix}.timeout")
        return None
    except PromptGeneratorError as exc:
        logger.exception("%s failed.", label_prefix)
        await _send_error(interaction, str(exc), label=f"{label_prefix}.generator_error")
        return None
    except Exception:
        logger.exception("Unexpected %s error.", label_prefix)
        await _send_error(
            interaction,
            "プロンプト生成に失敗しました。LM Studioの起動状態を確認してください。",
            label=f"{label_prefix}.unexpected_error",
        )
        return None


class PromptImageGenerateView(discord.ui.View):
    def __init__(self, owner_id: int, positive_prompt: str, negative_prompt: str) -> None:
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return False
        return True

    @discord.ui.button(label="画像生成", style=discord.ButtonStyle.primary)
    async def generate_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied prompt image generation by user_id=%s", interaction.user.id)
            await interaction.response.send_message(SD_DENIED_MESSAGE, ephemeral=True)
            return

        await interaction.response.send_message(GENERATING_MESSAGE, ephemeral=True)
        await _generate_images_from_prompts(
            interaction,
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            label_prefix="prompt.button",
        )


async def _build_prompt_result(interaction: discord.Interaction, text: str, *, label_prefix: str, reference_id: int | None = None):
    try:
        result = await asyncio.to_thread(generate_sd_prompt_result, text, interaction.user.id, reference_id)
        logger.info(
            "%s result received: positive_length=%s negative_length=%s blocked=%s",
            label_prefix,
            len(result.positive_prompt),
            len(result.negative_prompt),
            result.blocked,
        )
        return result
    except PromptGeneratorTimeout:
        logger.exception("%s timed out.", label_prefix)
        await _send_error(interaction, "LM Studioから応答がありませんでした。", label=f"{label_prefix}.timeout")
        return None
    except PromptGeneratorError as exc:
        logger.exception("%s failed.", label_prefix)
        await _send_error(interaction, str(exc), label=f"{label_prefix}.generator_error")
        return None
    except Exception:
        logger.exception("Unexpected %s error.", label_prefix)
        await _send_error(
            interaction,
            "プロンプト生成に失敗しました。LM Studioの起動状態を確認してください。",
            label=f"{label_prefix}.unexpected_error",
        )
        return None


def setup(client: discord.Client) -> None:
    @client.tree.command(name="prompt", description="日本語入力からStable Diffusion用プロンプトを作ります。")
    @app_commands.describe(text="作りたい画像の内容。日本語で入力できます。", reference_id="全タグを使う画像参考メモID")
    async def slash_prompt(interaction: discord.Interaction, text: str, reference_id: int | None = None) -> None:
        await interaction.response.defer(ephemeral=True)
        result = await _build_prompt_result(interaction, text, label_prefix="prompt", reference_id=reference_id)
        if result is None:
            return

        message = build_discord_prompt_message(result)
        logger.info("/prompt Discord message built: message_length=%s", len(message))

        if not result.blocked and result.positive_prompt and result.negative_prompt:
            logger.info("Prompt view creation start: user_id=%s", interaction.user.id)
            view = PromptImageGenerateView(interaction.user.id, result.positive_prompt, result.negative_prompt)
            logger.info("Prompt view creation complete: user_id=%s view=%s", interaction.user.id, type(view).__name__)
            await _safe_followup_send(
                interaction,
                label="prompt.result_with_view",
                content=message,
                view=view,
                ephemeral=True,
            )
            return

        logger.info("Prompt response without view: blocked=%s", result.blocked)
        await _safe_followup_send(
            interaction,
            label="prompt.result_without_view",
            content=message,
            ephemeral=True,
        )

    @client.tree.command(name="generate", description="日本語入力からプロンプト生成と画像生成をまとめて実行します。")
    @app_commands.describe(text="作りたい画像の内容。日本語で入力できます。", reference_id="全タグを使う画像参考メモID")
    async def slash_generate(interaction: discord.Interaction, text: str, reference_id: int | None = None) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /generate by user_id=%s", interaction.user.id)
            await _send_error(interaction, SD_DENIED_MESSAGE, label="generate.denied")
            return

        result = await _build_prompt_result(interaction, text, label_prefix="generate", reference_id=reference_id)
        if result is None:
            return

        if result.blocked or not result.positive_prompt or not result.negative_prompt:
            message = build_discord_prompt_message(result)
            await _safe_followup_send(
                interaction,
                label="generate.prompt_blocked",
                content=message,
                ephemeral=True,
            )
            return

        await _safe_followup_send(
            interaction,
            label="generate.started",
            content=GENERATING_MESSAGE,
            ephemeral=True,
        )
        await _generate_images_from_prompts(
            interaction,
            positive_prompt=result.positive_prompt,
            negative_prompt=result.negative_prompt,
            label_prefix="generate",
        )

    @client.tree.command(name="generate_raw", description="入力したPositive / Negativeをそのまま使って画像生成します。")
    @app_commands.describe(
        positive="ComfyUIへそのまま渡すPositive。必須です。",
        negative="ComfyUIへそのまま渡すNegative。省略できます。",
        use_default_negative="negative省略時に既定Negativeを使うか。falseなら空欄です。",
        public="生成結果をチャンネルへ公開するか。既定はtrueです。",
    )
    async def slash_generate_raw(
        interaction: discord.Interaction,
        positive: str,
        negative: str | None = None,
        use_default_negative: bool = True,
        public: bool = True,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /generate_raw by user_id=%s", interaction.user.id)
            await _send_error(interaction, SD_DENIED_MESSAGE, label="generate_raw.denied")
            return

        if not positive.strip():
            await _send_error(
                interaction,
                "Positiveは必須です。空白以外の内容を入力してください。",
                label="generate_raw.positive_empty",
            )
            return
        if len(positive) > RAW_PROMPT_MAX_LENGTH:
            await _send_error(
                interaction,
                f"Positiveは{RAW_PROMPT_MAX_LENGTH}文字以内で入力してください。",
                label="generate_raw.positive_too_long",
            )
            return
        if negative is not None and len(negative) > RAW_PROMPT_MAX_LENGTH:
            await _send_error(
                interaction,
                f"Negativeは{RAW_PROMPT_MAX_LENGTH}文字以内で入力してください。",
                label="generate_raw.negative_too_long",
            )
            return

        effective_negative = negative
        if effective_negative is None:
            effective_negative = get_negative_prompt(None) if use_default_negative else ""

        await _safe_followup_send(
            interaction,
            label="generate_raw.started",
            content=GENERATING_MESSAGE,
            ephemeral=True,
        )
        await _generate_images_from_prompts(
            interaction,
            positive_prompt=positive,
            negative_prompt=effective_negative,
            label_prefix="generate_raw",
            preserve_prompt_text=True,
            result_ephemeral=not public,
        )

    @client.tree.command(
        name="generate_ai",
        description="日本語の要望からshogou-chan-prompt-designerでプロンプト生成し画像生成します。",
    )
    @app_commands.describe(
        request="作りたい画像の日本語要望。",
        reference_id="任意。全タグを使う画像参考メモID",
    )
    async def slash_generate_ai(
        interaction: discord.Interaction,
        request: str,
        reference_id: int | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /generate_ai by user_id=%s", interaction.user.id)
            await _send_error(interaction, SD_DENIED_MESSAGE, label="generate_ai.denied")
            return

        result = await _build_shogou_prompt_result(interaction, request, label_prefix="generate_ai", reference_id=reference_id)
        if result is None:
            return

        if result.blocked:
            await _send_error(interaction, result.blocked_reason, label="generate_ai.blocked")
            return

        await _safe_followup_send(
            interaction,
            label="generate_ai.started",
            content=GENERATING_MESSAGE,
            ephemeral=True,
        )
        await _generate_images_from_shogou_result(
            interaction,
            result,
            label_prefix="generate_ai",
            request_text=request,
            reference_id=reference_id,
        )

    logger.info("Registered prompt slash commands: prompt, generate, generate_raw, generate_ai")
