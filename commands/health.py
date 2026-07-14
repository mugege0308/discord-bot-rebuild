import asyncio
import logging
from datetime import datetime, timedelta

import discord
from discord.ext import tasks

from services.health_check import (
    build_health_check_report,
    get_health_check_channel_id,
    get_health_check_interval_minutes,
    is_periodic_health_check_enabled,
)
from services.pc_control import is_pc_control_allowed
from services.weather import JST


logger = logging.getLogger("secretary-bot.health")

DENIED_MESSAGE = "この点検は、許可されたユーザーだけが実行できるよ。"


def setup(client: discord.Client) -> None:
    @client.tree.command(name="health_check", description="Botとローカル環境の状態を点検します。")
    async def slash_health_check(interaction: discord.Interaction) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied /health_check by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            report = await asyncio.to_thread(build_health_check_report)
        except Exception:
            logger.exception("Failed to build health check report.")
            await interaction.followup.send("状態点検に失敗したみたい。Botのログを確認してね。", ephemeral=True)
            return

        await interaction.followup.send(report, ephemeral=True)

    last_sent_at: datetime | None = None

    @tasks.loop(minutes=1)
    async def health_check_loop() -> None:
        nonlocal last_sent_at

        if not is_periodic_health_check_enabled():
            return

        now = datetime.now(JST)
        interval = get_health_check_interval_minutes()
        if last_sent_at is not None and now - last_sent_at < timedelta(minutes=interval):
            return

        channel_id = get_health_check_channel_id()
        if channel_id is None:
            return

        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            if not isinstance(channel, discord.abc.Messageable):
                logger.warning("Health check channel is not messageable: %s", channel_id)
                return

            report = await asyncio.to_thread(build_health_check_report)
            await channel.send(report)
            last_sent_at = now
            logger.info("Health check notification sent to channel_id=%s", channel_id)
        except Exception:
            logger.exception("Failed to send periodic health check notification.")

    client.health_check_loop = health_check_loop
