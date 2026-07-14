import asyncio
import logging

import discord

from services.claude_control import ClaudeControlError, get_claude_status, start_claude_code
from services.pc_control import is_pc_control_allowed


logger = logging.getLogger("secretary-bot.claude")
DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できます。"


def setup(client: discord.Client) -> None:
    @client.tree.command(name="claude_start", description="自宅PCでClaude Codeを起動します。")
    async def slash_claude_start(interaction: discord.Interaction) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /claude_start by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            result = await asyncio.to_thread(start_claude_code)
        except ClaudeControlError as exc:
            logger.exception("/claude_start failed for user_id=%s", interaction.user.id)
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /claude_start failure for user_id=%s", interaction.user.id)
            await interaction.followup.send("Claude Codeを起動できませんでした。Botのログを確認してください。", ephemeral=True)
            return
        if result.already_running:
            await interaction.followup.send("Claude Codeはすでに起動しています。", ephemeral=True)
        else:
            await interaction.followup.send("Claude Codeを起動しました。", ephemeral=True)

    @client.tree.command(name="claude_status", description="Claude Codeの起動状態を確認します。")
    async def slash_claude_status(interaction: discord.Interaction) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /claude_status by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            status = await asyncio.to_thread(get_claude_status)
        except Exception:
            logger.exception("Unexpected /claude_status failure for user_id=%s", interaction.user.id)
            await interaction.followup.send("Claude Codeの状態を確認できませんでした。Botのログを確認してください。", ephemeral=True)
            return
        message = "Claude Codeは起動中です。" if status.running else "Claude Codeは停止しています。"
        await interaction.followup.send(message, ephemeral=True)

    logger.info("Registered Claude Code slash commands: claude_start, claude_status")
