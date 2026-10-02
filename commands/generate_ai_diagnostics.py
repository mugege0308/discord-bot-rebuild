"""Public diagnostics and settings commands for ``/generate_ai``.

The generation pipeline owns the debug record and last-settings storage. This
module only exposes those existing service APIs through Discord commands; it
does not create a second generation path.
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands

from commands.help import split_discord_message
from commands.prompt import GENERATE_AI_MODEL_CHOICES
from services.generate_ai_debug import (
    format_generate_ai_debug,
    format_generate_ai_debug_detail_chunks,
    get_latest_generate_ai_debug,
)
from services.generate_ai_last_settings import reset_last_settings
from services.pc_control import is_pc_control_allowed


logger = logging.getLogger("secretary-bot.generate_ai_diagnostics")
DENIED_MESSAGE = "このコマンドは、許可されたユーザーだけが実行できます。"


async def _send_chunks(
    interaction: discord.Interaction,
    chunks: list[str],
    *,
    label: str,
) -> None:
    safe_chunks = [part for chunk in chunks for part in split_discord_message(chunk)] or [""]
    await interaction.response.send_message(
        safe_chunks[0],
        ephemeral=True,
        allowed_mentions=discord.AllowedMentions.none(),
    )
    for index, chunk in enumerate(safe_chunks[1:], start=1):
        try:
            await interaction.followup.send(
                chunk,
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            logger.exception("Failed to send %s followup part=%s", label, index)
            return


def setup(client: discord.Client) -> None:
    @client.tree.command(
        name="generate_ai_debug",
        description="自分の直近 /generate_ai の簡略診断を表示します。",
    )
    async def slash_generate_ai_debug(interaction: discord.Interaction) -> None:
        record = get_latest_generate_ai_debug(interaction.user.id)
        message = (
            "直近の /generate_ai 診断情報はありません。"
            if record is None
            else format_generate_ai_debug(record)
        )
        await _send_chunks(interaction, [message], label="generate_ai_debug")

    @client.tree.command(
        name="generate_ai_debug_detail",
        description="自分の直近 /generate_ai のstage別Positive/Negativeを表示します。",
    )
    async def slash_generate_ai_debug_detail(interaction: discord.Interaction) -> None:
        record = get_latest_generate_ai_debug(interaction.user.id)
        chunks = (
            ["直近の /generate_ai 詳細診断情報はありません。"]
            if record is None
            else format_generate_ai_debug_detail_chunks(record)
        )
        await _send_chunks(interaction, chunks, label="generate_ai_debug_detail")

    @client.tree.command(
        name="generate_ai_settings_reset",
        description="自分の /generate_ai 前回設定をモデル別または全モデル分リセットします。",
    )
    @app_commands.choices(model=GENERATE_AI_MODEL_CHOICES)
    @app_commands.describe(model="指定モデルだけをリセット。未指定なら全モデル分をリセットします。")
    async def slash_generate_ai_settings_reset(
        interaction: discord.Interaction,
        model: app_commands.Choice[str] | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send(DENIED_MESSAGE, ephemeral=True)
            return
        try:
            removed = await asyncio.to_thread(
                reset_last_settings,
                interaction.user.id,
                None if model is None else model.value,
            )
        except Exception:
            logger.exception("Failed to reset /generate_ai settings: user_id=%s", interaction.user.id)
            await interaction.followup.send(
                "前回設定をリセットできませんでした。設定ファイルを確認してください。",
                ephemeral=True,
            )
            return

        target = "全モデル" if model is None else model.name
        message = (
            f"{target}の前回設定をリセットしました。"
            if removed
            else f"{target}に保存済みの前回設定はありません。"
        )
        await interaction.followup.send(message, ephemeral=True)

    logger.info(
        "Registered generate_ai diagnostics commands: "
        "generate_ai_debug, generate_ai_debug_detail, generate_ai_settings_reset"
    )
