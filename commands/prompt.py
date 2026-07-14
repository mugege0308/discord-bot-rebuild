import asyncio
import logging
from pathlib import Path

import discord
from discord import app_commands

from sd_prompt_template_generator import build_discord_prompt_message
from services.image_selection import (
    get_selection_generate_count,
    is_image_selection_available,
    select_generated_images,
)
from services.pc_control import is_pc_control_allowed
from services.prompt_generator import PromptGeneratorError, PromptGeneratorTimeout, generate_sd_prompt_result
from services.sd_control import SdGenerationOptions, generate_images


logger = logging.getLogger("secretary-bot.prompt")

SD_DENIED_MESSAGE = "画像生成は、許可されたユーザーだけが実行できます。"
ONLY_OWNER_MESSAGE = "この画像生成ボタンは、/prompt を実行した本人だけが押せます。"
GENERATING_MESSAGE = "画像生成中です。少し待っててね。"


async def _safe_followup_send(
    interaction: discord.Interaction,
    *,
    label: str,
    content: str | None = None,
    files: list[discord.File] | None = None,
    view: discord.ui.View | None = None,
    ephemeral: bool = True,
) -> None:
    logger.info(
        "Discord followup.send before: label=%s content_length=%s file_count=%s has_view=%s ephemeral=%s",
        label,
        len(content or ""),
        len(files or []),
        view is not None,
        ephemeral,
    )
    try:
        kwargs = {"ephemeral": ephemeral}
        if content is not None:
            kwargs["content"] = content
        if files is not None:
            kwargs["files"] = files
        if view is not None:
            kwargs["view"] = view
        await interaction.followup.send(**kwargs)
    except Exception:
        logger.exception("Discord followup.send failed: label=%s", label)
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
) -> None:
    files: list[discord.File] = []
    try:
        for path in list(paths)[:10]:
            files.append(discord.File(str(path), filename=path.name))
        await _safe_followup_send(
            interaction,
            label="prompt.image_files",
            content=message,
            files=files,
            ephemeral=ephemeral,
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
        return text
    return text[:limit].rstrip() + "..."


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


def _build_generation_options(positive_prompt: str, negative_prompt: str) -> SdGenerationOptions:
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
) -> None:
    options = _build_generation_options(positive_prompt, negative_prompt)
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
    await _send_image_files(interaction, message, send_paths, ephemeral=True)


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
