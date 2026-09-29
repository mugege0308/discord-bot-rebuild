import asyncio
from dataclasses import replace
import logging
from pathlib import Path

import discord
from discord import app_commands

from sd_prompt_template_generator import build_discord_prompt_message
from services.direct_prompt_parser import (
    ENGLISH_DIRECT_INPUT_MODE,
    INPUT_MODE_LABELS,
    JAPANESE_AI_INPUT_MODE,
    PROMPT_IMPORT_INPUT_MODE,
    DirectPromptResult,
    parse_direct_prompt,
)
from services.prompt_import import PromptImportResult, resolve_prompt_import
from services.generation_history import (
    MIN_GROUP_SIZE_FOR_TAG_COMPARISON,
    MIN_RATED_FOR_TREND,
    NORMAL_GENERATION_LABEL,
    GenerationRequestDetail,
    GenerationRequestRecord,
    GenerationStats,
    SavedGeneration,
    TagTrendEntry,
    get_generation_request,
    get_generation_stats,
    resolve_image_ids_for_paths,
    save_generation_history_detailed,
    set_image_discord_message,
    upsert_image_rating,
)
from services.image_selection import (
    ImageSelectionResult,
    get_selection_generate_count,
    is_image_selection_available,
    select_generated_images,
)
from services.image_analyzer import MAX_IMAGE_BYTES, detect_image_content_type, validate_image_bytes
from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.danbooru_character_resolver import DanbooruCharacterResolution
from services.verified_character_registry import VerifiedCharacterRegistry
from services.pc_control import is_pc_control_allowed
from services.prompt_generator import PromptGeneratorError, PromptGeneratorTimeout, generate_sd_prompt_result
from services.sd_control import (
    GeneratedImage,
    SdGenerationOptions,
    generate_images,
    generate_images_with_metadata,
    get_negative_prompt,
    get_reference_status_text,
    is_comfy_backend,
    resolve_reference_strength,
)
from services.shogou_prompt_designer import (
    PROMPT_RATING_AUTO,
    ShogouPromptResult,
    generate_shogou_composition_variation,
    generate_shogou_custom_instruction_variation,
    generate_shogou_prompt_result,
)


logger = logging.getLogger("secretary-bot.prompt")

SD_DENIED_MESSAGE = "画像生成は、許可されたユーザーだけが実行できます。"
ONLY_OWNER_MESSAGE = "この画像生成ボタンは、/prompt を実行した本人だけが押せます。"
GENERATING_MESSAGE = "画像生成中です。少し待っててね。"
RAW_PROMPT_MAX_LENGTH = 4000
SHOGOU_MAX_ASSUMPTIONS_DISPLAY = 5
SHOGOU_ASSUMPTION_ITEM_LIMIT = 200
SHOGOU_SUMMARY_LIMIT = 300
SHOGOU_MESSAGE_LIMIT = 1900
GENERATE_AI_MODEL_CHOICES = [
    app_commands.Choice(name="Animagine XL 4.0 Clear", value="animagine"),
    app_commands.Choice(name="JANKU v7.77", value="janku"),
]
GENERATE_AI_PROMPT_RATING_CHOICES = [
    app_commands.Choice(name="Auto", value=PROMPT_RATING_AUTO),
    app_commands.Choice(name="Safe", value="safe"),
    app_commands.Choice(name="Sensitive", value="sensitive"),
    app_commands.Choice(name="NSFW", value="nsfw"),
    app_commands.Choice(name="Explicit", value="explicit"),
]
GENERATE_AI_INPUT_MODE_CHOICES = [
    app_commands.Choice(name=INPUT_MODE_LABELS[JAPANESE_AI_INPUT_MODE], value=JAPANESE_AI_INPUT_MODE),
    app_commands.Choice(name=INPUT_MODE_LABELS[ENGLISH_DIRECT_INPUT_MODE], value=ENGLISH_DIRECT_INPUT_MODE),
    app_commands.Choice(name=INPUT_MODE_LABELS[PROMPT_IMPORT_INPUT_MODE], value=PROMPT_IMPORT_INPUT_MODE),
]


# Discord allows at most 10 attachments per message.
MAX_SEND_IMAGE_FILES = 10
RATING_ORDER = ("favorite", "like", "meh", "off_target")
RATING_VIEW_TIMEOUT_SECONDS = 3600
RATING_LABELS = {"favorite": "お気に入り", "like": "好き", "meh": "微妙", "off_target": "想定と違う"}
RATING_EMOJI = {"favorite": "⭐", "like": "👍", "meh": "👎", "off_target": "🎯"}
# 4 buttons per image, so 1 image per action row (4 buttons/row, within
# Discord's 5-per-row limit). Discord allows at most 5 action rows per
# message, and Phase 1C reserves the last one (REGENERATE_ROW) for the
# regenerate buttons, so only 4 rows - the first 4 rateable images in
# attachment order - are left for rating buttons; the rest are still sent,
# just without rating buttons.
RATING_IMAGES_PER_ROW = 1
MAX_RATEABLE_IMAGES = 4 * RATING_IMAGES_PER_ROW

# Phase 1C: regenerate the parent generation_request instead of rating one
# of its images. Always on the row after the last possible rating row, so
# together they never exceed Discord's 5-row cap.
REGENERATION_TYPES = ("same_prompt", "change_composition")
REGENERATE_ROW = MAX_RATEABLE_IMAGES // RATING_IMAGES_PER_ROW
REGENERATE_LABELS = {
    "same_prompt": "同じ条件で再生成",
    "change_composition": "構図を変えて再生成",
    "custom_instruction": "指示を追加して再生成",
}
REGENERATE_EMOJI = {"same_prompt": "🔁", "change_composition": "🎥", "custom_instruction": "✏️"}
REGENERATE_ONLY_OWNER_MESSAGE = "この再生成ボタンは、元の生成を実行した本人だけが押せます。"
REGENERATE_ALREADY_RUNNING_MESSAGE = "すでに再生成を実行中です。終わってからもう一度試してね。"
REGENERATE_STARTED_MESSAGE = "再生成を開始しました。"
REGENERATE_FAILED_MESSAGE = "再生成に失敗しました。時間をおいてもう一度試してね。"
REGENERATE_PARENT_MISSING_MESSAGE = "元の生成情報が見つかりませんでした。履歴が削除されたか、DBが利用できない可能性があります。"

# 指示を追加して再生成: opens a Modal instead of firing immediately, so this
# type is deliberately kept out of REGENERATION_TYPES (which drives the
# immediate-action RegenerateButton loop) and handled by its own button/modal
# classes below.
CUSTOM_INSTRUCTION_TYPE = "custom_instruction"
CUSTOM_INSTRUCTION_MAX_LENGTH = 1000
CUSTOM_INSTRUCTION_MODAL_TITLE = "追加指示を入力"
CUSTOM_INSTRUCTION_INPUT_LABEL = "追加指示"
CUSTOM_INSTRUCTION_EMPTY_MESSAGE = "追加指示を入力してください。"

# Phase 2A: /generate_ai_stats. Display-only labels; the underlying values
# ("normal"/"same_prompt"/"change_composition"/"custom_instruction") come
# from generation_history.
STATS_REGENERATION_LABELS = {
    NORMAL_GENERATION_LABEL: "通常生成",
    "same_prompt": "同じ条件で再生成",
    "change_composition": "構図を変えて再生成",
    "custom_instruction": "指示を追加して再生成",
}
STATS_RATING_ICON_ORDER = ("favorite", "like", "meh", "off_target")
STATS_NO_HISTORY_MESSAGE = "まだ /generate_ai の生成履歴がありません。/generate_ai で画像を生成してみてね。"
STATS_NO_RATINGS_NOTE = "まだあなた自身の評価がありません。生成した画像に評価ボタン（⭐👍👎🎯）を押すと、ここに集計されるよ。"
STATS_LOW_SAMPLE_NOTE = "まだ評価数が少ないため、傾向判断には不十分です。"
STATS_TOP_TAG_LIMIT = 3
STATS_FAILED_MESSAGE = "評価統計の集計に失敗しました。時間をおいてもう一度試してね。"
LAST_REFERENCE_MISSING_MESSAGE = "再利用できる参考画像がありません。先にreference_imageを添付して生成してください。"

# In-process only (not persisted, not restored after a bot restart): guards
# against double-clicking a regenerate button while its background task is
# still running. Keyed by the parent generation_requests.id being
# regenerated from.
_active_regenerations: set[int] = set()

# In-process only: the most recently supplied /generate_ai reference image,
# isolated by Discord user ID. It intentionally resets when the bot restarts.
_last_reference_images: dict[int, tuple[bytes, str]] = {}


async def _disable_view_message_safely(
    view: discord.ui.View, message: "discord.Message | None", *, log_label: str
) -> None:
    """Disables every child of `view` and tries to reflect that on the
    already-sent Discord message, on a timeout.

    `message` (when a View that timed out has one) was obtained from
    `interaction.followup.send(..., wait=True)`, so editing it directly goes
    through the original interaction's webhook route - and by the time
    RATING_VIEW_TIMEOUT_SECONDS (1 hour) has elapsed, that webhook token is
    long expired, which raises `401 Unauthorized (error code: 50027 Invalid
    Webhook Token)`. Re-fetching the message through its channel instead uses
    the normal bot-token message-edit route, which has no such expiry.

    Safe to call more than once (re-disabling already-disabled children and
    re-fetching/re-editing the message are both harmless no-ops the second
    time). Never raises: a message that is gone, no longer accessible, or -
    for an ephemeral message, which was never a normal channel message to
    begin with - simply not fetchable this way, is logged as a warning and
    otherwise ignored. The already-delivered content and any stored ratings
    are unaffected either way.
    """
    for child in view.children:
        child.disabled = True
    if message is None:
        return
    try:
        fetched_message = await message.channel.fetch_message(message.id)
        await fetched_message.edit(view=view)
    except Exception:
        logger.warning(
            "Could not disable %s view buttons after timeout (message deleted, inaccessible, "
            "or the original interaction webhook token expired): message_id=%s",
            log_label,
            message.id,
            exc_info=True,
        )


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
        await _disable_view_message_safely(self, self.message, log_label="rating")


class RegenerateButton(discord.ui.Button):
    """Regenerates the whole generation_request (not one image) that this
    message's images came from. Phase 1C: same_prompt reuses the saved
    positive/negative as-is; change_composition sends the parent's
    request/positive/negative/summary/assumptions back through the Prompt
    Designer with an instruction to vary only the composition.
    """

    def __init__(self, *, request_id: int, regeneration_type: str, owner_id: int, row: int) -> None:
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label=f"{REGENERATE_EMOJI[regeneration_type]} {REGENERATE_LABELS[regeneration_type]}",
            custom_id=f"genai_regen:{request_id}:{regeneration_type}",
            row=row,
        )
        self.request_id = request_id
        self.regeneration_type = regeneration_type
        self.owner_id = owner_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            try:
                await interaction.response.send_message(REGENERATE_ONLY_OWNER_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to regenerate button permission check: request_id=%s", self.request_id)
            return

        # Anti-spam: only one regeneration per parent request_id at a time.
        # This is an in-process guard only (no persistence, no cross-restart
        # tracking), which is enough to stop rapid double-clicks from
        # queuing duplicate ComfyUI/LM Studio work.
        if self.request_id in _active_regenerations:
            try:
                await interaction.response.send_message(REGENERATE_ALREADY_RUNNING_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to regenerate button busy check: request_id=%s", self.request_id)
            return

        _active_regenerations.add(self.request_id)
        try:
            await interaction.response.send_message(REGENERATE_STARTED_MESSAGE, ephemeral=True)
        except Exception:
            logger.exception("Failed to acknowledge regenerate button: request_id=%s", self.request_id)
            _active_regenerations.discard(self.request_id)
            return

        # Fire-and-forget: the ephemeral ack above is the Discord-visible
        # response; the actual regeneration follows the same
        # defer-then-followup pattern as the slash commands and reports its
        # own errors via _send_error. _active_regenerations is cleared by
        # _run_regeneration's finally block once this finishes either way.
        asyncio.create_task(_run_regeneration(interaction, self.request_id, self.regeneration_type))


class CustomInstructionModal(discord.ui.Modal):
    """「✏️ 指示を追加して再生成」: collects the free-text instruction before
    starting the same background regeneration pipeline the other regenerate
    buttons use. Opened by CustomInstructionRegenerateButton, which already
    performed the owner/already-running checks before showing this modal.
    """

    def __init__(self, request_id: int, owner_id: int) -> None:
        super().__init__(title=CUSTOM_INSTRUCTION_MODAL_TITLE)
        self.request_id = request_id
        self.owner_id = owner_id
        self.instruction = discord.ui.TextInput(
            label=CUSTOM_INSTRUCTION_INPUT_LABEL,
            style=discord.TextStyle.paragraph,
            max_length=CUSTOM_INSTRUCTION_MAX_LENGTH,
            required=True,
        )
        self.add_item(self.instruction)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            try:
                await interaction.response.send_message(REGENERATE_ONLY_OWNER_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to custom instruction modal permission check: request_id=%s", self.request_id)
            return

        instruction_text = str(self.instruction.value).strip()
        if not instruction_text:
            try:
                await interaction.response.send_message(CUSTOM_INSTRUCTION_EMPTY_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to empty custom instruction: request_id=%s", self.request_id)
            return

        # Same anti-spam guard as RegenerateButton.callback, checked again
        # here since opening the modal itself does not reserve request_id -
        # only actually starting the regeneration does.
        if self.request_id in _active_regenerations:
            try:
                await interaction.response.send_message(REGENERATE_ALREADY_RUNNING_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to custom instruction busy check: request_id=%s", self.request_id)
            return

        _active_regenerations.add(self.request_id)
        try:
            await interaction.response.send_message(REGENERATE_STARTED_MESSAGE, ephemeral=True)
        except Exception:
            logger.exception("Failed to acknowledge custom instruction modal: request_id=%s", self.request_id)
            _active_regenerations.discard(self.request_id)
            return

        asyncio.create_task(
            _run_regeneration(
                interaction,
                self.request_id,
                CUSTOM_INSTRUCTION_TYPE,
                custom_instruction=instruction_text,
            )
        )


class CustomInstructionRegenerateButton(discord.ui.Button):
    """Opens CustomInstructionModal instead of starting the regeneration
    immediately - the only regenerate button that needs input from the user
    first."""

    def __init__(self, *, request_id: int, owner_id: int, row: int) -> None:
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label=f"{REGENERATE_EMOJI[CUSTOM_INSTRUCTION_TYPE]} {REGENERATE_LABELS[CUSTOM_INSTRUCTION_TYPE]}",
            custom_id=f"genai_regen:{request_id}:{CUSTOM_INSTRUCTION_TYPE}",
            row=row,
        )
        self.request_id = request_id
        self.owner_id = owner_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            try:
                await interaction.response.send_message(REGENERATE_ONLY_OWNER_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to regenerate button permission check: request_id=%s", self.request_id)
            return

        if self.request_id in _active_regenerations:
            try:
                await interaction.response.send_message(REGENERATE_ALREADY_RUNNING_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to regenerate button busy check: request_id=%s", self.request_id)
            return

        try:
            await interaction.response.send_modal(CustomInstructionModal(self.request_id, self.owner_id))
        except Exception:
            logger.exception("Failed to open custom instruction modal: request_id=%s", self.request_id)


def _build_regenerate_buttons(request_id: int, owner_id: int) -> list[discord.ui.Button]:
    buttons: list[discord.ui.Button] = [
        RegenerateButton(
            request_id=request_id,
            regeneration_type=regeneration_type,
            owner_id=owner_id,
            row=REGENERATE_ROW,
        )
        for regeneration_type in REGENERATION_TYPES
    ]
    buttons.append(
        CustomInstructionRegenerateButton(request_id=request_id, owner_id=owner_id, row=REGENERATE_ROW)
    )
    return buttons


async def _safe_followup_send(
    interaction: discord.Interaction,
    *,
    label: str,
    content: str | None = None,
    embed: discord.Embed | None = None,
    files: list[discord.File] | None = None,
    view: discord.ui.View | None = None,
    ephemeral: bool = True,
    wait: bool = False,
) -> "discord.Message | None":
    logger.info(
        "Discord followup.send before: label=%s content_length=%s has_embed=%s file_count=%s has_view=%s ephemeral=%s",
        label,
        len(content or ""),
        embed is not None,
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
        if embed is not None:
            kwargs["embed"] = embed
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
    model_key: str | None = None,
    filename_prefix: str | None = None,
    model_selection_explicit: bool = False,
    reference_image_bytes: bytes | None = None,
    reference_image_content_type: str | None = None,
    reference_strength: float | None = None,
) -> SdGenerationOptions:
    selection_available = is_image_selection_available()
    count = 1 if reference_image_bytes is not None else (
        get_selection_generate_count() if selection_available else 1
    )
    logger.info(
        "Prompt image generation count decided: count=%s selection_available=%s",
        count,
        selection_available,
    )
    return SdGenerationOptions(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        model_key=model_key,
        filename_prefix=filename_prefix,
        model_selection_explicit=model_selection_explicit,
        preserve_prompt_text=preserve_prompt_text,
        count=count,
        width=512,
        height=512,
        steps=20,
        cfg=7.0,
        sampler="Euler a",
        seed=-1,
        reference_image_bytes=reference_image_bytes,
        reference_image_content_type=reference_image_content_type,
        reference_strength=reference_strength,
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
    *,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    reference_used: bool = False,
    reference_strength: float | None = None,
) -> str:
    profile = get_image_generation_profile(model_key)
    selection_text = _selection_message(selection, send_count, generated_count)
    reference_text = get_reference_status_text(reference_used, reference_strength)
    message = (
        f"{selection_text}\n\n"
        f"Model: {profile.display_name}\n\n"
        f"{reference_text}\n\n"
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
    parent_request_id: int | None = None,
    regeneration_type: str | None = None,
    custom_instruction: str | None = None,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
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
            parent_request_id=parent_request_id,
            regeneration_type=regeneration_type,
            custom_instruction=custom_instruction,
            model_key=model_key,
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


async def _build_result_view(
    saved: SavedGeneration | None,
    generated_images: list[GeneratedImage],
    send_paths: list[Path],
    owner_id: int,
) -> tuple["ImageRatingView | None", list[int]]:
    """Combines the rating buttons (per sent image) with the regenerate
    buttons (per request) into the single View a Discord message can carry.

    Regenerate buttons only need a request_id, so they are offered whenever
    the history save succeeded - even if no individual image could be
    matched for rating (_build_rating_view returning None for that reason
    alone must not also hide the regenerate buttons).
    """
    if saved is None:
        return None, []
    view, rated_image_ids = await _build_rating_view(saved, generated_images, send_paths)
    if view is None:
        view = ImageRatingView([])
    for button in _build_regenerate_buttons(saved.request_id, owner_id):
        view.add_item(button)
    return view, rated_image_ids


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
    parent_request_id: int | None = None,
    regeneration_type: str | None = None,
    custom_instruction: str | None = None,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    model_selection_explicit: bool = False,
    reference_image_bytes: bytes | None = None,
    reference_image_content_type: str | None = None,
    reference_strength: float | None = None,
) -> None:
    # Mirrors _generate_images_from_prompts' preserve_prompt_text=True path exactly,
    # but keeps its own message builder so /generate_raw's behavior stays untouched.
    profile = get_image_generation_profile(model_key)
    options = _build_generation_options(
        result.positive,
        result.negative,
        preserve_prompt_text=True,
        model_key=profile.model_key,
        filename_prefix=profile.filename_prefix,
        model_selection_explicit=model_selection_explicit,
        reference_image_bytes=reference_image_bytes,
        reference_image_content_type=reference_image_content_type,
        reference_strength=reference_strength,
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
    if reference_image_bytes is not None and len(paths) == 1:
        selection = ImageSelectionResult(selected_paths=[], used_selection=False)
    else:
        selection = await asyncio.to_thread(select_generated_images, paths)
    send_paths = selection.selected_paths or paths
    message = _build_shogou_generated_image_message(
        result,
        selection,
        len(send_paths),
        len(paths),
        model_key=profile.model_key,
        reference_used=reference_image_bytes is not None,
        reference_strength=reference_strength,
    )

    saved = await _save_generation_history_safely(
        user_id=interaction.user.id,
        request_text=request_text,
        reference_id=reference_id,
        result=result,
        generated_images=generated_images,
        selection=selection,
        parent_request_id=parent_request_id,
        regeneration_type=regeneration_type,
        custom_instruction=custom_instruction,
        model_key=profile.model_key,
    )

    view, rated_image_ids = await _build_result_view(saved, generated_images, send_paths, interaction.user.id)
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
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
) -> ShogouPromptResult | None:
    try:
        result = await asyncio.to_thread(
            generate_shogou_prompt_result,
            text,
            interaction.user.id,
            reference_id,
            model_key,
            enable_generic_danbooru=(label_prefix == "generate_ai"),
        )
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


async def _build_shogou_regeneration_result(
    interaction: discord.Interaction,
    parent: GenerationRequestDetail,
    *,
    label_prefix: str,
) -> ShogouPromptResult | None:
    """Phase 1C "change_composition": mirrors _build_shogou_prompt_result's
    error handling exactly, but calls generate_shogou_composition_variation
    instead of the plain shogou call.
    """
    try:
        result = await asyncio.to_thread(
            generate_shogou_composition_variation,
            parent.request_text,
            parent.positive,
            parent.negative,
            parent.summary,
            parent.assumptions,
            interaction.user.id,
            parent.reference_id,
            parent.model_key,
        )
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


async def _build_shogou_custom_instruction_result(
    interaction: discord.Interaction,
    parent: GenerationRequestDetail,
    custom_instruction: str,
    *,
    label_prefix: str,
) -> ShogouPromptResult | None:
    """「指示を追加して再生成」: mirrors _build_shogou_regeneration_result's
    error handling exactly, but calls generate_shogou_custom_instruction_variation
    with the user's free-text instruction alongside the parent fields.
    """
    try:
        result = await asyncio.to_thread(
            generate_shogou_custom_instruction_variation,
            parent.request_text,
            parent.positive,
            parent.negative,
            parent.summary,
            parent.assumptions,
            custom_instruction,
            interaction.user.id,
            parent.reference_id,
            parent.model_key,
        )
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


async def _run_regeneration(
    interaction: discord.Interaction,
    request_id: int,
    regeneration_type: str,
    *,
    custom_instruction: str | None = None,
) -> None:
    """Runs one regeneration in the background, after RegenerateButton (or
    CustomInstructionModal, for custom_instruction) has already sent the
    ephemeral "started" reply.

    Any failure here (missing parent, LM Studio error, generation error,
    or anything unexpected) is reported to the user as a short generic
    message via _send_error - never a raw exception - while the full detail
    goes to logger.exception. _active_regenerations is always cleared here,
    regardless of outcome, so a failed regeneration does not permanently
    lock out further attempts.
    """
    try:
        parent = await asyncio.to_thread(get_generation_request, request_id)
        if parent is None:
            await _send_error(
                interaction,
                REGENERATE_PARENT_MISSING_MESSAGE,
                label="generate_ai.regen.parent_missing",
            )
            return

        if regeneration_type == "same_prompt":
            # No Prompt Designer call: reuse the saved positive/negative
            # exactly. generate_images_with_metadata still randomizes the
            # seed on every call, and checkpoint/sampler/steps/cfg/width/
            # height come from the same workflow file either way, so "same
            # condition, new seed" falls out of the existing generation
            # path without any extra bookkeeping here.
            result = ShogouPromptResult(
                positive=parent.positive,
                negative=parent.negative,
                summary=parent.summary,
                assumptions=parent.assumptions,
            )
        elif regeneration_type == "change_composition":
            result = await _build_shogou_regeneration_result(interaction, parent, label_prefix="generate_ai.regen")
            if result is None:
                return
            if result.blocked:
                await _send_error(interaction, result.blocked_reason, label="generate_ai.regen.blocked")
                return
        elif regeneration_type == CUSTOM_INSTRUCTION_TYPE:
            result = await _build_shogou_custom_instruction_result(
                interaction, parent, custom_instruction or "", label_prefix="generate_ai.regen"
            )
            if result is None:
                return
            if result.blocked:
                await _send_error(interaction, result.blocked_reason, label="generate_ai.regen.blocked")
                return
        else:
            logger.error("Unknown regeneration_type encountered: %r request_id=%s", regeneration_type, request_id)
            return

        await _generate_images_from_shogou_result(
            interaction,
            result,
            label_prefix="generate_ai.regen",
            request_text=parent.request_text,
            reference_id=parent.reference_id,
            parent_request_id=request_id,
            regeneration_type=regeneration_type,
            custom_instruction=custom_instruction if regeneration_type == CUSTOM_INSTRUCTION_TYPE else None,
            model_key=parent.model_key,
            model_selection_explicit=parent.model_key != DEFAULT_IMAGE_GENERATION_MODEL_KEY,
        )
    except Exception:
        logger.exception(
            "Unexpected error during regeneration: request_id=%s regeneration_type=%s",
            request_id,
            regeneration_type,
        )
        await _send_error(interaction, REGENERATE_FAILED_MESSAGE, label="generate_ai.regen.unexpected_error")
    finally:
        _active_regenerations.discard(request_id)


def _format_percentage(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "-"
    return f"{numerator / denominator * 100:.1f}%"


def _format_rating_line(ratings) -> str:
    icons = {"favorite": "⭐", "like": "👍", "meh": "👎", "off_target": "🎯"}
    counts = {name: getattr(ratings, name) for name in STATS_RATING_ICON_ORDER}
    return " / ".join(f"{icons[name]} {counts[name]}" for name in STATS_RATING_ICON_ORDER)


def _select_top_tag_phrases(
    tag_trend: tuple[TagTrendEntry, ...],
    good_sample_count: int,
    low_sample_count: int,
    limit: int = STATS_TOP_TAG_LIMIT,
) -> tuple[list[str], list[str]]:
    """Ranks watched phrases by how much more often they show up in the
    caller's good (favorite/like) ratings versus low (meh/off_target)
    ratings, as a share of each group's own sample size (so a small group
    is not penalized just for having fewer total ratings).

    Only meaningful once both groups meet MIN_GROUP_SIZE_FOR_TAG_COMPARISON -
    callers are expected to check that before calling this.
    """
    scored: list[tuple[str, float]] = []
    for entry in tag_trend:
        good_ratio = entry.good_count / good_sample_count if good_sample_count else 0.0
        low_ratio = entry.low_count / low_sample_count if low_sample_count else 0.0
        scored.append((entry.phrase, good_ratio - low_ratio))

    good_leaning = sorted((item for item in scored if item[1] > 0), key=lambda item: item[1], reverse=True)
    low_leaning = sorted((item for item in scored if item[1] < 0), key=lambda item: item[1])
    return (
        [phrase for phrase, _ in good_leaning[:limit]],
        [phrase for phrase, _ in low_leaning[:limit]],
    )


def _build_generation_stats_embed(stats: GenerationStats) -> discord.Embed:
    embed = discord.Embed(title="/generate_ai 評価統計", color=discord.Color.blurple())

    if stats.total_requests == 0:
        embed.description = STATS_NO_HISTORY_MESSAGE
        return embed

    embed.add_field(
        name="全体件数",
        value=(
            f"生成リクエスト: {stats.total_requests}件 / 生成画像: {stats.total_images}枚\n"
            f"評価済み: {stats.rated_images}枚 / 未評価: {stats.unrated_images}枚"
        ),
        inline=False,
    )

    if stats.rated_images == 0:
        embed.add_field(name="評価別件数", value=STATS_NO_RATINGS_NOTE, inline=False)
        return embed

    embed.add_field(
        name="評価別件数",
        value=(
            f"{_format_rating_line(stats.ratings_all_time)}\n"
            f"好評価率: {_format_percentage(stats.ratings_all_time.good, stats.ratings_all_time.total)}"
            f"（低評価: {_format_percentage(stats.ratings_all_time.low, stats.ratings_all_time.total)}）"
        ),
        inline=False,
    )

    recent_lines = [
        f"直近7日: {stats.ratings_7d.total}件（{_format_rating_line(stats.ratings_7d)}）",
        f"直近30日: {stats.ratings_30d.total}件（{_format_rating_line(stats.ratings_30d)}）",
        f"全期間: {stats.ratings_all_time.total}件",
    ]
    if stats.rated_images < MIN_RATED_FOR_TREND:
        recent_lines.append(f"※{STATS_LOW_SAMPLE_NOTE}")
    embed.add_field(name="直近の傾向", value="\n".join(recent_lines), inline=False)

    regeneration_lines = []
    for entry in stats.regeneration_breakdown:
        label = STATS_REGENERATION_LABELS.get(entry.regeneration_type, entry.regeneration_type)
        line = f"{label}: {entry.request_count}件（評価 {_format_rating_line(entry.ratings)}）"
        if entry.ratings.total < MIN_GROUP_SIZE_FOR_TAG_COMPARISON:
            line += "（件数少なめ）"
        regeneration_lines.append(line)
    if regeneration_lines:
        embed.add_field(name="再生成傾向", value="\n".join(regeneration_lines), inline=False)

    if (
        stats.rated_images >= MIN_RATED_FOR_TREND
        and stats.good_sample_count >= MIN_GROUP_SIZE_FOR_TAG_COMPARISON
        and stats.low_sample_count >= MIN_GROUP_SIZE_FOR_TAG_COMPARISON
    ):
        good_phrases, low_phrases = _select_top_tag_phrases(
            stats.tag_trend, stats.good_sample_count, stats.low_sample_count
        )
        trend_lines = []
        if good_phrases:
            trend_lines.append(f"好評価で多い: {', '.join(good_phrases)}")
        if low_phrases:
            trend_lines.append(f"低評価で多い: {', '.join(low_phrases)}")
        if not trend_lines:
            trend_lines.append("好評価・低評価の間で目立った傾向差は見つかりませんでした。")
        trend_value = "\n".join(trend_lines)
    else:
        trend_value = STATS_LOW_SAMPLE_NOTE
    embed.add_field(name="Prompt Designerのテキスト傾向", value=trend_value, inline=False)

    return embed


class GenerateAiDanbooruCharacterConfirmationView(discord.ui.View):
    """Requester-only first-use confirmation for an unregistered character."""

    def __init__(self, owner_id: int, resolution: DanbooruCharacterResolution, continue_generation) -> None:
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.resolution = resolution
        self.registry = VerifiedCharacterRegistry()
        self.continue_generation = continue_generation
        candidate = resolution.candidate_character_tags[0] if resolution.candidate_character_tags else None
        if candidate is None:
            raise ValueError("character confirmation requires a Danbooru candidate")
        self.candidate = candidate
        self.add_item(discord.ui.Button(label="このキャラで登録", style=discord.ButtonStyle.success, custom_id="genai_char_confirm"))
        self.add_item(discord.ui.Button(label="登録しない", style=discord.ButtonStyle.secondary, custom_id="genai_char_cancel"))
        self.children[0].callback = self._confirm
        self.children[1].callback = self._cancel

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("この確認画面を操作できるのは依頼者だけです。", ephemeral=True)
            return False
        return True

    def message_text(self) -> str:
        return (
            "未登録のキャラクター候補が見つかりました。画像生成はまだ開始していません。\n"
            f"入力: `{self.resolution.input_character_name}`\n"
            f"Danbooru候補: `{self.candidate.name}`\n"
            f"Prompt tag: `{self.candidate.prompt_tag}`\n\n"
            "正しい候補なら「このキャラで登録」を押してください。"
        )

    async def _confirm(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            saved = await asyncio.to_thread(
                self.registry.register_verified_candidate,
                self.candidate,
                input_alias=self.resolution.input_character_name,
                copyright_tag=self.resolution.selected_copyright_tag,
                context_aliases=self.resolution.generated_work_queries,
                display_name=self.resolution.input_character_name,
            )
            confirmed = self.resolution.with_registry_confirmation(
                character=saved,
                saved_alias=self.resolution.input_character_name,
            )
            await interaction.edit_original_response(
                content=f"`{saved.prompt_tag}` を確認済み登録しました。生成を続行します。",
                view=None,
            )
            await self.continue_generation(confirmed)
        except Exception:
            logger.exception("Failed to register Danbooru character candidate")
            await interaction.edit_original_response(
                content="キャラクター登録に失敗しました。画像生成は開始していません。",
                view=None,
            )

    async def _cancel(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(
            content="キャラクター登録をキャンセルしました。画像生成は開始していません。",
            view=None,
        )

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True


async def _resume_generate_ai_after_character_confirmation(
    interaction: discord.Interaction,
    result: ShogouPromptResult,
    confirmed_resolution: DanbooruCharacterResolution,
    *,
    request: str,
    reference_id: int | None,
    profile,
    model,
    reference_image_bytes: bytes | None,
    reference_image_content_type: str | None,
    resolved_reference_strength: float | None,
) -> None:
    confirmed_result = replace(
        result,
        danbooru_character_resolution=confirmed_resolution,
        danbooru_character_resolved_tag_ids=confirmed_resolution.selected_tag_ids,
        danbooru_character_resolved_tags=confirmed_resolution.selected_prompt_tags,
        character_confirmation_required=False,
        character_unresolved_reason="",
    )
    await _safe_followup_send(
        interaction,
        label="generate_ai.started_after_character_confirmation",
        content=GENERATING_MESSAGE,
        ephemeral=True,
    )
    await _generate_images_from_shogou_result(
        interaction,
        confirmed_result,
        label_prefix="generate_ai",
        request_text=request,
        reference_id=reference_id,
        model_key=profile.model_key,
        model_selection_explicit=model is not None,
        reference_image_bytes=reference_image_bytes,
        reference_image_content_type=reference_image_content_type,
        reference_strength=resolved_reference_strength,
    )


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
    @app_commands.choices(model=GENERATE_AI_MODEL_CHOICES)
    @app_commands.describe(
        request="作りたい画像の日本語要望。",
        model="画像生成モデル。未指定時はAnimagine XL 4.0 Clearを使います。",
        reference_id="任意。全タグを使う画像参考メモID",
        reference_image="任意。同じキャラの顔・髪・雰囲気を参照する画像1枚。",
        reference_strength="参考画像の強さ（0.0～1.0）。未指定時は0.5です。",
        use_last_reference="直前に使用した参考画像を再利用します。reference_image指定時はそちらを優先します。",
    )
    async def slash_generate_ai(
        interaction: discord.Interaction,
        request: str,
        model: app_commands.Choice[str] | None = None,
        reference_id: int | None = None,
        reference_image: discord.Attachment | None = None,
        reference_strength: float | None = None,
        use_last_reference: bool = False,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /generate_ai by user_id=%s", interaction.user.id)
            await _send_error(interaction, SD_DENIED_MESSAGE, label="generate_ai.denied")
            return

        cached_reference = (
            _last_reference_images.get(interaction.user.id)
            if reference_image is None and use_last_reference
            else None
        )
        if reference_image is None and use_last_reference and cached_reference is None:
            await _send_error(
                interaction,
                LAST_REFERENCE_MISSING_MESSAGE,
                label="generate_ai.last_reference_missing",
            )
            return

        reference_used = reference_image is not None or cached_reference is not None
        if reference_used and not is_comfy_backend():
            await _send_error(
                interaction,
                "reference_imageを指定できるのはComfyUIバックエンドだけです。",
                label="generate_ai.reference_backend_unsupported",
            )
            return

        try:
            resolved_reference_strength = resolve_reference_strength(
                reference_used, reference_strength
            )
        except ValueError as exc:
            await _send_error(interaction, str(exc), label="generate_ai.reference_strength_invalid")
            return

        reference_image_bytes, reference_image_content_type = cached_reference or (None, None)
        if reference_image is not None:
            if reference_image.size > MAX_IMAGE_BYTES:
                await _send_error(
                    interaction,
                    "参考画像は8MB以内のJPEG / PNG / WebPを指定してください。",
                    label="generate_ai.reference_image_too_large",
                )
                return
            try:
                reference_image_bytes = await reference_image.read()
                reference_image_content_type = detect_image_content_type(
                    reference_image_bytes, reference_image.content_type
                )
                validate_image_bytes(reference_image_bytes, reference_image_content_type)
            except PromptGeneratorError as exc:
                await _send_error(interaction, str(exc), label="generate_ai.reference_image_invalid")
                return
            except Exception:
                logger.exception("Failed to read /generate_ai reference image.")
                await _send_error(
                    interaction,
                    "参考画像を読み込めませんでした。もう一度添付して試してください。",
                    label="generate_ai.reference_image_read_failed",
                )
                return
            _last_reference_images[interaction.user.id] = (
                reference_image_bytes,
                reference_image_content_type,
            )

        try:
            profile = get_image_generation_profile(model.value if model is not None else None)
        except ValueError as exc:
            await _send_error(interaction, str(exc), label="generate_ai.model_invalid")
            return
        if model is not None and not is_comfy_backend():
            await _send_error(
                interaction,
                "モデルを明示選択できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。",
                label="generate_ai.model_backend_unsupported",
            )
            return

        result = await _build_shogou_prompt_result(
            interaction,
            request,
            label_prefix="generate_ai",
            reference_id=reference_id,
            model_key=profile.model_key,
        )
        if result is None:
            return

        if result.character_confirmation_required and result.danbooru_character_resolution is not None:
            async def continue_after_character_confirmation(confirmed_resolution: DanbooruCharacterResolution) -> None:
                await _resume_generate_ai_after_character_confirmation(
                    interaction,
                    result,
                    confirmed_resolution,
                    request=request,
                    reference_id=reference_id,
                    profile=profile,
                    model=model,
                    reference_image_bytes=reference_image_bytes,
                    reference_image_content_type=reference_image_content_type,
                    resolved_reference_strength=resolved_reference_strength,
                )

            confirmation_view = GenerateAiDanbooruCharacterConfirmationView(
                interaction.user.id,
                result.danbooru_character_resolution,
                continue_after_character_confirmation,
            )
            await _safe_followup_send(
                interaction,
                label="generate_ai.character_confirmation",
                content=confirmation_view.message_text(),
                view=confirmation_view,
                ephemeral=True,
            )
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
            model_key=profile.model_key,
            model_selection_explicit=model is not None,
            reference_image_bytes=reference_image_bytes,
            reference_image_content_type=reference_image_content_type,
            reference_strength=resolved_reference_strength,
        )

    @client.tree.command(
        name="generate_ai_stats",
        description="/generate_ai の自分の評価履歴を集計して表示します。",
    )
    async def slash_generate_ai_stats(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /generate_ai_stats by user_id=%s", interaction.user.id)
            await _send_error(interaction, SD_DENIED_MESSAGE, label="generate_ai_stats.denied")
            return

        try:
            stats = await asyncio.to_thread(get_generation_stats, interaction.user.id)
        except Exception:
            logger.exception("Unexpected error while building /generate_ai_stats for user_id=%s", interaction.user.id)
            stats = None

        if stats is None:
            await _send_error(interaction, STATS_FAILED_MESSAGE, label="generate_ai_stats.failed")
            return

        embed = _build_generation_stats_embed(stats)
        await _safe_followup_send(interaction, label="generate_ai_stats.result", embed=embed, ephemeral=True)

    logger.info("Registered prompt slash commands: prompt, generate, generate_raw, generate_ai, generate_ai_stats")
