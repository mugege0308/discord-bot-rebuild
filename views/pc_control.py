import asyncio
import logging

import discord

from services.pc_control import is_pc_control_allowed, is_windows, schedule_shutdown


logger = logging.getLogger("secretary-bot.pc_control_view")

ONLY_REQUESTER_MESSAGE = "この確認ボタンは、シャットダウンを要求した本人だけが押せるよ。"
NOT_ALLOWED_MESSAGE = "この操作は、許可されたユーザーだけが実行できるよ。"
SHUTDOWN_FAILED_MESSAGE = "シャットダウン予約に失敗したみたい。Botのログを確認してね。"
WINDOWS_ONLY_MESSAGE = "PCシャットダウン機能はWindowsでだけ実行できるよ。"


class PcShutdownConfirmView(discord.ui.View):
    def __init__(
        self,
        requester_id: int,
        delay_seconds: int,
        success_message: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(timeout=120)
        self.requester_id = requester_id
        self.delay_seconds = delay_seconds
        self.success_message = success_message
        self.description = description

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(ONLY_REQUESTER_MESSAGE, ephemeral=True)
            return False
        return True

    @discord.ui.button(label="実行する", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied PC shutdown confirmation by user_id=%s", interaction.user.id)
            await interaction.response.edit_message(content=NOT_ALLOWED_MESSAGE, view=None)
            return

        if not is_windows():
            logger.warning("Denied PC shutdown confirmation because OS is not Windows. user_id=%s", interaction.user.id)
            await interaction.response.edit_message(content=WINDOWS_ONLY_MESSAGE, view=None)
            return

        try:
            await asyncio.to_thread(schedule_shutdown, self.delay_seconds, interaction.user.id, self.description)
        except Exception:
            logger.exception("Failed to schedule PC shutdown by user_id=%s", interaction.user.id)
            await interaction.response.edit_message(content=SHUTDOWN_FAILED_MESSAGE, view=None)
            return

        logger.warning(
            "PC shutdown scheduled by user_id=%s delay_seconds=%s",
            interaction.user.id,
            self.delay_seconds,
        )
        message = self.success_message or (
            f"シャットダウンを予約したよ。\n{self.delay_seconds}秒後にPCを終了するね。\n\n"
            "必要なら /pc_shutdown_cancel で取り消せるよ。"
        )
        await interaction.response.edit_message(content=message, view=None)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        logger.info("PC shutdown confirmation cancelled by user_id=%s", interaction.user.id)
        await interaction.response.edit_message(content="キャンセルしたよ。", view=None)
