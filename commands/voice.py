from __future__ import annotations

import logging

import discord
from discord import app_commands

from services.tts_service import (
    TTSService,
    TTSServiceError,
    VoicevoxUnavailableError,
    get_vc_say_max_length,
    remove_temp_audio,
)
from services.voice_service import (
    VoiceConnectionError,
    VoiceJoinResult,
    VoiceNotConnectedError,
    VoiceService,
    VoiceServiceError,
)


logger = logging.getLogger("secretary-bot.voice-command")
GUILD_ONLY_MESSAGE = "このコマンドはサーバー内で使ってね。"
USER_NOT_IN_VC_MESSAGE = "先に読み上げ先のボイスチャンネルへ参加してね。"

tts_service = TTSService()
voice_service = VoiceService()


def _user_voice_channel(interaction: discord.Interaction) -> discord.abc.Connectable | None:
    voice_state = getattr(interaction.user, "voice", None)
    return getattr(voice_state, "channel", None)


async def cleanup(client: discord.Client) -> None:
    await voice_service.close_all(client.voice_clients)


def setup(client: discord.Client) -> None:
    @client.tree.command(name="vc_join", description="参加中のボイスチャンネルへ初号ちゃんを呼びます。")
    async def slash_vc_join(interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return

        channel = _user_voice_channel(interaction)
        if channel is None:
            await interaction.response.send_message(USER_NOT_IN_VC_MESSAGE, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            result = await voice_service.join_voice_channel(interaction.guild, channel)
        except VoiceConnectionError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /vc_join failure: guild_id=%s user_id=%s", interaction.guild.id, interaction.user.id)
            await interaction.followup.send("ボイスチャンネルへの接続に失敗しました。ログを確認してね。", ephemeral=True)
            return

        if result is VoiceJoinResult.ALREADY_CONNECTED:
            message = "初号ちゃんはすでに同じボイスチャンネルにいるよ。"
        elif result is VoiceJoinResult.MOVED:
            message = "あなたのボイスチャンネルへ移動したよ。"
        else:
            message = "ボイスチャンネルへ参加したよ。"
        await interaction.followup.send(message, ephemeral=True)

    @client.tree.command(name="vc_leave", description="初号ちゃんをボイスチャンネルから退出させます。")
    async def slash_vc_leave(interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            stopped = await voice_service.leave_guild(interaction.guild)
        except Exception:
            logger.exception("Unexpected /vc_leave failure: guild_id=%s user_id=%s", interaction.guild.id, interaction.user.id)
            await interaction.followup.send("VCの終了処理に失敗しました。ログを確認してね。", ephemeral=True)
            return

        message = "VCから退出し、読み上げキューを破棄したよ。" if stopped else "初号ちゃんはVCに接続していないよ。"
        await interaction.followup.send(message, ephemeral=True)

    @client.tree.command(name="vc_say", description="VOICEVOXで文章を読み上げます。")
    @app_commands.describe(text="読み上げる文章。最大200文字")
    async def slash_vc_say(interaction: discord.Interaction, text: str) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return

        normalized_text = text.strip()
        max_length = get_vc_say_max_length()
        if not normalized_text:
            await interaction.response.send_message("読み上げる文章を入力してね。", ephemeral=True)
            return
        if len(normalized_text) > max_length:
            await interaction.response.send_message(
                f"読み上げ文章は{max_length}文字以内にしてね（現在{len(normalized_text)}文字）。",
                ephemeral=True,
            )
            return

        if not voice_service.is_connected(interaction.guild) and _user_voice_channel(interaction) is None:
            await interaction.response.send_message(USER_NOT_IN_VC_MESSAGE, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        wav_path = None
        try:
            wav_path = await tts_service.synthesize_to_wav(normalized_text)

            if not voice_service.is_connected(interaction.guild):
                channel = _user_voice_channel(interaction)
                if channel is None:
                    raise VoiceNotConnectedError("音声合成中にVCから退出したため、自動接続できませんでした。")
                await voice_service.join_voice_channel(interaction.guild, channel)

            queue_size = await voice_service.enqueue(interaction.guild, wav_path)
            wav_path = None  # Ownership moved to VoiceService.
        except VoicevoxUnavailableError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except (TTSServiceError, VoiceServiceError) as exc:
            logger.warning("/vc_say failed: guild_id=%s user_id=%s error=%s", interaction.guild.id, interaction.user.id, exc)
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /vc_say failure: guild_id=%s user_id=%s", interaction.guild.id, interaction.user.id)
            await interaction.followup.send("音声の合成または再生準備に失敗しました。ログを確認してね。", ephemeral=True)
            return
        finally:
            remove_temp_audio(wav_path)

        await interaction.followup.send(f"読み上げキューに追加したよ（待機キュー: {queue_size}件）。", ephemeral=True)

    async def on_voice_state_update(
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if client.user is None or member.id != client.user.id:
            return
        if before.channel is not None and after.channel is None:
            await voice_service.handle_manual_disconnect(member.guild.id)

    client.event(on_voice_state_update)
    logger.info("Registered voice slash commands: vc_join, vc_leave, vc_say")
