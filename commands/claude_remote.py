import logging
from datetime import datetime

import discord

from services.claude_remote import (
    ERROR_MESSAGES,
    ClaudeRemoteError,
    ClaudeRemoteManager,
    ClaudeRemoteStatus,
    claude_remote_manager,
)
from services.pc_control import is_pc_control_allowed


logger = logging.getLogger("secretary-bot.claude_remote_commands")

DENIED_MESSAGE = "この操作は、PC_CONTROL_ALLOWED_USER_IDS で許可されたユーザーだけが実行できます。"
DISABLED_MESSAGE = ERROR_MESSAGES["disabled"]
LOST_LINK_GUIDANCE = (
    "Bot再起動などで接続URLが失われた場合は、Claudeアプリまたは "
    "https://claude.ai/code のCode一覧から、設定したセッション名を探してください。"
)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}時間{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def _format_started_at(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_status(status: ClaudeRemoteStatus) -> str:
    state = "停止中"
    if status.stopping:
        state = "停止処理中"
    elif status.running:
        state = "起動中"
    return "\n".join(
        [
            f"状態: {state}",
            f"PID: {status.pid if status.pid is not None else '-'}",
            f"起動時刻: {_format_started_at(status.started_at)}",
            f"経過時間: {_format_duration(status.elapsed_seconds)}",
            f"対象プロジェクト: {status.project_dir if status.project_dir is not None else '-'}",
            f"接続URL: {'取得済み' if status.url_available else '未取得'}",
            f"終了コード: {status.exit_code if status.exit_code is not None else '-'}",
            f"直近エラー: {status.last_error or 'なし'}",
        ]
    )


def setup(client: discord.Client, manager: ClaudeRemoteManager = claude_remote_manager) -> None:
    async def reject_if_unavailable(interaction: discord.Interaction) -> bool:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied Claude Remote Control command by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return True
        if not manager.get_config().enabled:
            await interaction.response.send_message(DISABLED_MESSAGE, ephemeral=True)
            return True
        return False

    @client.tree.command(name="claude_remote_start", description="Claude Code Remote Controlを起動します。")
    async def slash_claude_remote_start(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            result = await manager.start()
        except ClaudeRemoteError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /claude_remote_start failure (output omitted).")
            await interaction.followup.send(ERROR_MESSAGES["start_failed"], ephemeral=True)
            return

        if result.already_running:
            message = f"Remote Controlはすでに起動中です（PID: {result.pid}）。"
            if result.url:
                message += f"\n接続URL: {result.url}"
            else:
                message += "\n接続URLはまだ取得できていません。"
        elif result.timed_out:
            message = ERROR_MESSAGES["start_timeout"]
        else:
            message = f"Remote Controlを起動しました（PID: {result.pid}）。"
            if result.url:
                message += f"\n接続URL: {result.url}\nこのURLは公開チャンネルへ貼らないでください。"
        await interaction.followup.send(message, ephemeral=True)

    @client.tree.command(name="claude_remote_status", description="Claude Code Remote Controlの状態を確認します。")
    async def slash_claude_remote_status(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return
        status = await manager.status()
        await interaction.response.send_message(format_status(status), ephemeral=True)

    @client.tree.command(name="claude_remote_link", description="Claude Code Remote Controlの接続URLを再表示します。")
    async def slash_claude_remote_link(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return
        status = await manager.status()
        if status.running and status.url:
            message = f"接続URL: {status.url}\nこのURLは公開チャンネルへ貼らないでください。"
        elif status.running:
            message = "Remote Controlは起動中ですが、接続URLはまだ取得できていません。少し待って再実行してください。"
        else:
            message = f"Remote Controlは停止中で、接続URLは保持していません。\n{LOST_LINK_GUIDANCE}"
        await interaction.response.send_message(message, ephemeral=True)

    @client.tree.command(name="claude_remote_stop", description="Claude Code Remote Controlを安全に停止します。")
    async def slash_claude_remote_stop(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            result = await manager.stop()
        except ClaudeRemoteError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /claude_remote_stop failure (output omitted).")
            await interaction.followup.send(ERROR_MESSAGES["stop_failed"], ephemeral=True)
            return
        if result.already_stopped:
            message = "Remote Controlはすでに停止しています。"
        elif result.used_kill:
            message = "Remote Controlを強制終了しました。"
        elif result.used_terminate:
            message = "通常終了しなかったためterminateで停止しました。"
        else:
            message = "Remote Controlを安全に停止しました。"
        await interaction.followup.send(message, ephemeral=True)

    logger.info(
        "Registered Claude Remote Control commands: claude_remote_start, claude_remote_status, claude_remote_link, claude_remote_stop"
    )
