import asyncio
import logging
import os
import re
from collections import Counter
from datetime import datetime

import discord
from discord import app_commands
from dotenv import load_dotenv

import daily_log
from commands import health
from commands import blue_archive
from commands import character_generate
from commands import claude
from commands import claude_remote
from commands import dev_task
from commands import image_analyze
from commands import memory
from commands import prompt
from commands import voice
from commands.help import handle_help
from commands.memo import handle_memo
from commands.ping import handle_ping
from commands.today import build_today_summary
from commands.todo import handle_todo_add, handle_todo_delete
from services.double_translate import (
    build_start_message,
    build_status_message,
    should_translate_message,
    start_double_translate,
    stop_double_translate,
)
from services.bot_control import launch_bot_start_file
from services.claude_remote import claude_remote_manager
from services.dev_task_store import mark_running_tasks_failed_on_startup
from services.jockie import (
    build_jockie_shuffle_response,
    build_saved_jockie_shuffle_response,
    delete_playlist,
    is_youtube_playlist_url,
    list_playlists,
    save_playlist,
)
from services.nisa import get_nisa_message
from services.pc_control import (
    cancel_shutdown,
    format_shutdown_status_message,
    get_shutdown_delay_seconds,
    is_pc_control_allowed,
    is_windows,
    seconds_from_minutes,
    seconds_until_time,
)
from services.image_selection import select_generated_images
from services.sd_control import (
    SdGenerationOptions,
    generate_images,
    get_latest_image_paths,
    get_status_message,
    save_favorites,
    set_shutdown_after_done,
    start_sd_app,
)
from services.translator import translate_ja_to_en_to_ja
from services.weather import (
    EVENING_NOTIFICATION,
    JST,
    MORNING_NOTIFICATION,
    get_startup_weather_skip_reason,
    get_startup_weather_notification_kind,
    get_startup_weather_message,
    get_weather_channel_id,
    get_weather_message,
    mark_weather_notified,
)
from views.panel import OperationPanel, PANEL_MESSAGE
from views.pc_control import PcShutdownConfirmView


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("secretary-bot")

WEATHER_ERROR_MESSAGE = "ごめんね、天気情報を取れなかったみたい。少し時間をおいて試してね。"
TRANSLATE_ERROR_MESSAGE = "翻訳に失敗したみたい。少し時間をおいて試してね。"
PC_CONTROL_DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できるよ。"
PC_CONTROL_WINDOWS_ONLY_MESSAGE = "PCシャットダウン機能はWindowsでだけ実行できるよ。"
PC_SHUTDOWN_CANCEL_FAILED_MESSAGE = "シャットダウン予約のキャンセルに失敗したみたい。Botのログを確認してね。"
GUILD_ONLY_MESSAGE = "このコマンドはサーバー内で使ってね。"
REQUIRED_SLASH_COMMANDS = {
    "ba_damage",
    "ba_tl_cost",
    "generate_raw",
    "generate_ai",
    "image_memory_list",
    "image_memory_show",
    "image_memory_rename",
    "image_memory_rate",
    "image_memory_use",
    "image_memory_delete",
    "claude_start",
    "claude_status",
    "claude_remote_start",
    "claude_remote_status",
    "claude_remote_link",
    "claude_remote_stop",
    "dev_task",
    "dev_task_status",
    "dev_task_stop",
    "vc_join",
    "vc_leave",
    "vc_say",
}


def _app_command_names(commands) -> list[str]:
    return sorted(str(getattr(command, "qualified_name", command.name)) for command in commands)


def _log_app_command_names(scope: str, commands, *, check_required: bool = True) -> None:
    names = _app_command_names(commands)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    logger.info("Slash commands %s: count=%s names=%s", scope, len(names), ", ".join(names))
    if duplicates:
        logger.warning("Duplicate slash command names %s: %s", scope, ", ".join(duplicates))
    if not check_required:
        return
    missing = sorted(REQUIRED_SLASH_COMMANDS - set(names))
    if missing:
        logger.warning("Required slash commands missing from %s: %s", scope, ", ".join(missing))
    else:
        logger.info("Required slash commands included in %s.", scope)


def _prepare_guild_only_commands(tree: app_commands.CommandTree, guild: discord.Object) -> list[str]:
    global_commands = list(tree.get_commands())
    delete_target_names = _app_command_names(global_commands)
    logger.info("Global slash commands scheduled for deletion: count=%s names=%s", len(delete_target_names), ", ".join(delete_target_names))

    tree.clear_commands(guild=guild)
    for command in global_commands:
        tree.add_command(command, guild=guild)

    tree.clear_commands(guild=None)
    return delete_target_names


def extract_command(content: str, bot_user_id: int) -> str | None:
    """Return the command after a leading bot mention, or None if not addressed."""
    mention_pattern = rf"^<@!?{bot_user_id}>\s*(.*)$"
    match = re.match(mention_pattern, content)
    if match is None:
        return None

    return match.group(1).strip()


class SecretaryClient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._dev_task_recovery_lock = asyncio.Lock()
        self._dev_task_startup_recovery_done = False

    async def close(self) -> None:
        try:
            await claude_remote_manager.cleanup()
        except Exception:
            logger.exception("Failed to clean up Claude Remote Control during Bot shutdown.")
        try:
            await voice.cleanup(self)
        except Exception:
            logger.exception("Failed to clean up Discord voice state during Bot shutdown.")
        await super().close()

    async def setup_hook(self) -> None:
        guild_id = os.getenv("DISCORD_GUILD_ID")
        try:
            _log_app_command_names("registered before sync (global)", self.tree.get_commands())
            if guild_id:
                guild = discord.Object(id=int(guild_id))
                _prepare_guild_only_commands(self.tree, guild)
                _log_app_command_names("prepared for guild sync", self.tree.get_commands(guild=guild))

                global_after_clear = await self.tree.sync()
                _log_app_command_names("global after clear sync", global_after_clear, check_required=False)
                logger.info("Cleared global slash commands. Remaining global count=%s.", len(global_after_clear))

                synced = await self.tree.sync(guild=guild)
                _log_app_command_names(f"synced to guild {guild_id}", synced)
                logger.info("Synced %s slash commands to guild %s.", len(synced), guild_id)
                return

            synced = await self.tree.sync()
            _log_app_command_names("synced globally", synced)
            logger.info("Synced %s global slash commands.", len(synced))
        except ValueError:
            logger.error("DISCORD_GUILD_ID must be a number: %s", guild_id)
        except discord.HTTPException:
            logger.exception("Failed to sync slash commands.")


client = SecretaryClient()
daily_log.setup(client)
health.setup(client)
blue_archive.setup(client)
claude.setup(client)
claude_remote.setup(client)
dev_task.setup(client)
character_generate.setup(client)
image_analyze.setup(client)
memory.setup(client)
prompt.setup(client)
voice.setup(client)


def bot_mention() -> str:
    if client.user is None:
        return "@bot"
    return client.user.mention


async def send_startup_weather_notification() -> None:
    channel_id = get_weather_channel_id()
    if channel_id is None:
        logger.info("天気通知: WEATHER_CHANNEL_ID が未設定です")
        return

    now = datetime.now(JST)
    notification_kind = get_startup_weather_notification_kind(now)
    skip_reason = get_startup_weather_skip_reason(now)
    if skip_reason == "before_hour":
        logger.info("天気通知: 6時前なので送信しません")
        return
    if notification_kind == EVENING_NOTIFICATION:
        logger.info("天気通知: 18時以降なので夕方通知を確認します")
    if skip_reason == "morning_already_notified":
        logger.info("天気通知: 朝通知は送信済みです")
        return
    if skip_reason == "evening_already_notified":
        logger.info("天気通知: 夕方通知は送信済みです")
        return

    try:
        message = await asyncio.to_thread(get_startup_weather_message, notification_kind or MORNING_NOTIFICATION)
    except Exception:
        logger.exception("天気通知: API取得に失敗しました")
        return

    try:
        channel = client.get_channel(channel_id)
        if channel is None:
            channel = await client.fetch_channel(channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            logger.error("天気通知: 送信先チャンネルにメッセージを送れません channel_id=%s", channel_id)
            return

        await channel.send(message)
        mark_weather_notified(notification_kind or MORNING_NOTIFICATION, now.date().isoformat())
        if notification_kind == EVENING_NOTIFICATION:
            logger.info("天気通知: 夕方通知を送信しました")
        else:
            logger.info("天気通知: 朝通知を送信しました")
    except Exception:
        logger.exception("天気通知: 送信に失敗しました")


@client.event
async def on_ready() -> None:
    if client.user is None:
        logger.info("Bot is ready, but client.user is not available yet.")
        return

    logger.info("Logged in as %s (ID: %s)", client.user, client.user.id)
    daily_report_loop = getattr(client, "daily_report_loop", None)
    if daily_report_loop is not None and not daily_report_loop.is_running():
        daily_report_loop.start()
    health_check_loop = getattr(client, "health_check_loop", None)
    if health_check_loop is not None and not health_check_loop.is_running():
        health_check_loop.start()

    # on_ready() can fire again after a mere reconnect (not just a genuine
    # process restart), so this recovery pass - which fails any dev task
    # still "running" - must only ever run once per process lifetime.
    # Otherwise a reconnect while a dummy execution is legitimately in
    # progress would incorrectly mark it failed.
    #
    # The lock serializes concurrent on_ready() firings so at most one
    # recovery attempt ever runs at a time; the "done" flag is only set
    # AFTER mark_running_tasks_failed_on_startup() actually succeeds, so a
    # failed attempt (e.g. a transient disk error) is retried on the next
    # on_ready() instead of being permanently skipped for the rest of the
    # process.
    if not client._dev_task_startup_recovery_done:
        async with client._dev_task_recovery_lock:
            if not client._dev_task_startup_recovery_done:
                try:
                    changed_count = await asyncio.to_thread(mark_running_tasks_failed_on_startup)
                    client._dev_task_startup_recovery_done = True
                    if changed_count:
                        logger.warning("Marked %s dev task(s) as failed after Bot restart.", changed_count)
                except Exception:
                    logger.exception("Failed to run dev task startup recovery.")

    await send_startup_weather_notification()


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    if client.user is None:
        return

    command_text = extract_command(message.content, client.user.id)
    if command_text is not None:
        mention = client.user.mention
        await message.channel.send(PANEL_MESSAGE, view=OperationPanel(message.author.id, mention))
        return

    should_translate, reason = should_translate_message(message.author.id, message.content)
    if reason is not None:
        await message.channel.send(reason)
        return

    if not should_translate:
        return

    try:
        translated = await asyncio.to_thread(translate_ja_to_en_to_ja, message.content.strip())
        if translated:
            await message.channel.send(translated)
        else:
            await message.channel.send(TRANSLATE_ERROR_MESSAGE)
    except Exception:
        logger.exception("Failed to double translate message.")
        await message.channel.send(TRANSLATE_ERROR_MESSAGE)


@client.tree.command(name="help", description="使える機能を表示します。")
async def slash_help(interaction: discord.Interaction) -> None:
    response = handle_help(bot_mention(), user_id=interaction.user.id)
    await interaction.response.send_message(response)


@client.tree.command(name="ping", description="Botが起きているか確認します。")
async def slash_ping(interaction: discord.Interaction) -> None:
    response = handle_ping(bot_mention(), user_id=interaction.user.id)
    await interaction.response.send_message(response)


@client.tree.command(name="memo", description="メモを追加します。")
@app_commands.describe(content="保存するメモ内容")
async def slash_memo(interaction: discord.Interaction, content: str) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = handle_memo(bot_mention(), content, interaction.user.id)
    try:
        record_id = await asyncio.to_thread(
            daily_log.add_record,
            interaction.guild.id,
            interaction.channel_id or 0,
            interaction.user.id,
            "memo",
            content,
        )
        response = f"{response}\n記録ID: #{record_id}"
    except ValueError:
        response = "メモ内容を入力してね。"
    except Exception:
        logger.exception("Failed to save /memo to daily_log records.")
        response = f"{response}\nただし、日次レポート用DBへの保存に失敗したみたい。"

    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="memo_list", description="自分のメモ一覧を表示します。")
async def slash_memo_list(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = await asyncio.to_thread(daily_log.list_recent_records, interaction.guild.id, "memo", 10)
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="memo_delete", description="指定した記録IDのメモを削除します。")
@app_commands.describe(number="削除するメモの記録ID")
async def slash_memo_delete(interaction: discord.Interaction, number: int) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = await asyncio.to_thread(daily_log.delete_record, interaction.guild.id, number)
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="todo_add", description="ToDoを追加します。")
@app_commands.describe(content="追加するToDo内容")
async def slash_todo_add(interaction: discord.Interaction, content: str) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = handle_todo_add(content, interaction.user.id)
    try:
        record_id = await asyncio.to_thread(
            daily_log.add_record,
            interaction.guild.id,
            interaction.channel_id or 0,
            interaction.user.id,
            "todo",
            content,
        )
        response = f"{response}\nTODO ID: #{record_id}"
    except ValueError:
        response = "ToDoの内容を入力してね。"
    except Exception:
        logger.exception("Failed to save /todo_add to daily_log records.")
        response = f"{response}\nただし、日次レポート用DBへの保存に失敗したみたい。"
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="todo_list", description="自分の未完了ToDo一覧を表示します。")
async def slash_todo_list(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = await asyncio.to_thread(daily_log.format_pending_todos, interaction.guild.id)
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="todo_done", description="指定したIDのTODOを完了にします。")
@app_commands.describe(todo_id="完了にするTODO ID")
async def slash_todo_done(interaction: discord.Interaction, todo_id: int) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
        return

    response = await asyncio.to_thread(daily_log.mark_todo_done, interaction.guild.id, todo_id)
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="todo_delete", description="指定した番号のToDoを削除します。")
@app_commands.describe(number="削除するToDo番号")
async def slash_todo_delete(interaction: discord.Interaction, number: int) -> None:
    response = handle_todo_delete(number, interaction.user.id)
    await interaction.response.send_message(response, ephemeral=True)


@client.tree.command(name="today", description="今日の天気、ToDo、メモ状況をまとめて表示します。")
async def slash_today(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    response = await asyncio.to_thread(build_today_summary, interaction.user.id)
    await interaction.followup.send(response, ephemeral=True)


@client.tree.command(name="pc_shutdown", description="許可ユーザーだけがPCシャットダウンの確認画面を表示できます。")
async def slash_pc_shutdown(interaction: discord.Interaction) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /pc_shutdown by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    if not is_windows():
        logger.warning("Denied /pc_shutdown because OS is not Windows. user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
        return

    delay_seconds = get_shutdown_delay_seconds()
    logger.warning("PC shutdown confirmation requested by user_id=%s delay_seconds=%s", interaction.user.id, delay_seconds)
    await interaction.response.send_message(
        "\n".join(
            [
                f"PCを{delay_seconds}秒後にシャットダウンするよ。",
                "本当に実行してもだいじょうぶ？",
            ]
        ),
        view=PcShutdownConfirmView(interaction.user.id, delay_seconds, description=f"{delay_seconds}秒後のシャットダウン"),
        ephemeral=True,
    )



@client.tree.command(name="pc_shutdown_in", description="指定した分数後にPCシャットダウンの確認画面を表示します。")
@app_commands.describe(minutes="何分後にシャットダウンするか。1〜1440分")
async def slash_pc_shutdown_in(interaction: discord.Interaction, minutes: int) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /pc_shutdown_in by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    if not is_windows():
        logger.warning("Denied /pc_shutdown_in because OS is not Windows. user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
        return

    try:
        delay_seconds = seconds_from_minutes(minutes)
    except ValueError:
        await interaction.response.send_message("minutes は 1〜1440 の範囲で指定してね。", ephemeral=True)
        return

    logger.warning("PC timed shutdown confirmation requested by user_id=%s minutes=%s delay_seconds=%s", interaction.user.id, minutes, delay_seconds)
    await interaction.response.send_message(
        "\n".join(
            [
                f"{minutes}分後にPCをシャットダウンする予約をするよ。",
                "これは大きめの操作だから、ほんとうに実行するか確認させてね。",
            ]
        ),
        view=PcShutdownConfirmView(
            interaction.user.id,
            delay_seconds,
            f"シャットダウンを予約したよ。\n{minutes}分後にPCを終了するね。\n\n必要なら /pc_shutdown_cancel で取り消せるよ。",
            f"{minutes}分後のシャットダウン",
        ),
        ephemeral=True,
    )


@client.tree.command(name="pc_shutdown_at", description="指定した時刻にPCシャットダウンの確認画面を表示します。")
@app_commands.describe(time="時刻。例: 23:30, 2330, 730, 23時30分, 23時")
async def slash_pc_shutdown_at(interaction: discord.Interaction, time: str) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /pc_shutdown_at by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    if not is_windows():
        logger.warning("Denied /pc_shutdown_at because OS is not Windows. user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
        return

    try:
        delay_seconds, target = seconds_until_time(time)
    except ValueError:
        await interaction.response.send_message(
            "時刻の形が少し分からなかったみたい。\n`2330` や `23:30`、`23時30分` の形で入れてみてね。",
            ephemeral=True,
        )
        return

    target_label = target.strftime("%H:%M")
    day_label = "今日" if delay_seconds < 24 * 60 * 60 and target.date() == datetime.now(JST).date() else "翌日"
    logger.warning(
        "PC shutdown-at confirmation requested by user_id=%s time=%s delay_seconds=%s target=%s",
        interaction.user.id,
        time,
        delay_seconds,
        target.isoformat(),
    )
    await interaction.response.send_message(
        "\n".join(
            [
                f"{day_label}の{target_label}にPCをシャットダウンする予約をするよ。",
                "これは大きめの操作だから、ほんとうに実行するか確認させてね。",
            ]
        ),
        view=PcShutdownConfirmView(
            interaction.user.id,
            delay_seconds,
            f"シャットダウンを予約したよ。\n{day_label}の{target_label}ごろにPCを終了するね。\n\n必要なら /pc_shutdown_cancel で取り消せるよ。",
            f"{day_label}の{target_label}ごろのシャットダウン",
        ),
        ephemeral=True,
    )


@client.tree.command(name="pc_shutdown_status", description="Botが最後に保存したPCシャットダウン予約を表示します。")
async def slash_pc_shutdown_status(interaction: discord.Interaction) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /pc_shutdown_status by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.send_message(format_shutdown_status_message(), ephemeral=True)


@client.tree.command(name="pc_shutdown_cancel", description="許可ユーザーだけが予約済みシャットダウンをキャンセルできます。")
async def slash_pc_shutdown_cancel(interaction: discord.Interaction) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /pc_shutdown_cancel by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    if not is_windows():
        logger.warning("Denied /pc_shutdown_cancel because OS is not Windows. user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
        return

    try:
        await asyncio.to_thread(cancel_shutdown)
    except Exception:
        logger.exception("Failed to cancel PC shutdown by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_SHUTDOWN_CANCEL_FAILED_MESSAGE, ephemeral=True)
        return

    logger.warning("PC shutdown cancelled by user_id=%s", interaction.user.id)
    await interaction.response.send_message("予約済みシャットダウンをキャンセルしたよ。", ephemeral=True)


@client.tree.command(name="bot_restart", description="許可ユーザーだけがBot自身を再起動できます。")
async def slash_bot_restart(interaction: discord.Interaction) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /bot_restart by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    try:
        start_file = await asyncio.to_thread(launch_bot_start_file)
    except Exception as exc:
        logger.exception("Failed to launch bot restart file by user_id=%s", interaction.user.id)
        await interaction.response.send_message(
            f"Bot再起動用ファイルの起動に失敗しました: {str(exc)[:300]}",
            ephemeral=True,
        )
        return

    logger.warning("Bot restart requested by user_id=%s start_file=%s", interaction.user.id, start_file)
    await interaction.response.send_message("Botを再起動します。", ephemeral=True)
    await asyncio.sleep(1)
    try:
        await client.close()
    finally:
        os._exit(0)



def _sd_allowed(interaction: discord.Interaction) -> bool:
    return is_pc_control_allowed(interaction.user.id)


async def _send_sd_files(interaction: discord.Interaction, message: str, paths) -> None:
    files = []
    try:
        for path in list(paths)[:10]:
            files.append(discord.File(str(path), filename=path.name))
        await interaction.followup.send(message, files=files)
    finally:
        for file in files:
            file.close()


def _sd_selection_message(selection, send_count: int) -> str:
    if selection.used_selection:
        return selection.message or f"画像選別AIで上位{send_count}枚を選びました。"
    if selection.error:
        return f"画像選別AIは使えませんでした: {selection.error}\n通常の生成画像を返します。"
    if selection.message:
        return f"{selection.message}\n通常の生成画像を返します。"
    return f"画像を{send_count}枚生成しました。"


async def _select_and_send_sd_files(interaction: discord.Interaction, paths) -> None:
    selection = await asyncio.to_thread(select_generated_images, list(paths))
    send_paths = selection.selected_paths or paths
    await _send_sd_files(interaction, _sd_selection_message(selection, len(send_paths)), send_paths)


@client.tree.command(name="sd_start", description="ローカル画像生成UIを起動します。")
async def slash_sd_start(interaction: discord.Interaction) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_start by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    response = await asyncio.to_thread(start_sd_app)
    await interaction.followup.send(response, ephemeral=True)


@client.tree.command(name="sd_status", description="ローカル画像生成APIの状態を確認します。")
async def slash_sd_status(interaction: discord.Interaction) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_status by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    response = await asyncio.to_thread(get_status_message)
    await interaction.followup.send(response, ephemeral=True)


@client.tree.command(name="sd_generate", description="ローカルStable Diffusion WebUIで画像生成します。")
@app_commands.describe(
    prompt="生成プロンプト",
    negative_prompt="ネガティブプロンプト。未指定なら既定テンプレートを使います。",
    count="生成枚数。最大10枚",
    width="画像幅",
    height="画像高さ",
    steps="ステップ数",
    cfg="CFG Scale",
    sampler="サンプラー名",
    seed="シード。-1ならランダム",
)
async def slash_sd_generate(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str = "",
    count: int = 1,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    sampler: str = "Euler a",
    seed: int = -1,
) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_generate by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer()
    options = SdGenerationOptions(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        count=count,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        seed=seed,
    )

    try:
        paths = await asyncio.to_thread(generate_images, options)
    except Exception as exc:
        logger.exception("Failed to generate SD images.")
        await interaction.followup.send(f"画像生成に失敗しました: {exc}")
        return

    await _select_and_send_sd_files(interaction, paths)


@client.tree.command(name="sd_latest", description="最新の生成画像をDiscordに送ります。")
async def slash_sd_latest(interaction: discord.Interaction) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_latest by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer()
    paths = await asyncio.to_thread(get_latest_image_paths)
    if not paths:
        await interaction.followup.send("最新生成画像はまだありません。")
        return

    await _send_sd_files(interaction, "最新生成画像です。", paths)


@client.tree.command(name="sd_save_favorite", description="最新生成画像をfavoritesフォルダにコピーします。")
@app_commands.describe(all_images="trueなら最新画像群すべてを保存します。", index="all_images=falseのとき保存する画像番号")
async def slash_sd_save_favorite(interaction: discord.Interaction, all_images: bool = True, index: int = 1) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_save_favorite by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    try:
        saved_paths = await asyncio.to_thread(save_favorites, all_images=all_images, index=index)
    except Exception as exc:
        logger.exception("Failed to save SD favorites.")
        await interaction.followup.send(f"お気に入り保存に失敗しました: {exc}", ephemeral=True)
        return

    await interaction.followup.send(f"favorites に{len(saved_paths)}枚保存しました。", ephemeral=True)


@client.tree.command(name="sd_shutdown_after_done", description="次の画像生成完了後にPCをシャットダウンします。")
@app_commands.describe(enabled="trueで予約、falseで解除")
async def slash_sd_shutdown_after_done(interaction: discord.Interaction, enabled: bool = True) -> None:
    if not _sd_allowed(interaction):
        logger.warning("Denied /sd_shutdown_after_done by user_id=%s", interaction.user.id)
        await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
        return

    response = await asyncio.to_thread(set_shutdown_after_done, enabled)
    await interaction.response.send_message(response, ephemeral=True)
@client.tree.command(name="weather", description="今日または明日の天気を表示します。")
@app_commands.describe(day="今日または明日")
@app_commands.choices(
    day=[
        app_commands.Choice(name="今日", value="今日"),
        app_commands.Choice(name="明日", value="明日"),
    ]
)
async def slash_weather(interaction: discord.Interaction, day: str = "今日") -> None:
    await interaction.response.defer()
    try:
        response = await asyncio.to_thread(get_weather_message, day)
        await interaction.followup.send(response)
    except Exception:
        logger.exception("Failed to handle /weather.")
        await interaction.followup.send(WEATHER_ERROR_MESSAGE)


@client.tree.command(name="nisa", description="NISAのざっくり目安を表示します。")
async def slash_nisa(interaction: discord.Interaction) -> None:
    try:
        await interaction.response.send_message(get_nisa_message())
    except Exception:
        logger.exception("Failed to handle /nisa.")
        await interaction.response.send_message("ごめんね、NISAの目安を表示できなかったみたい。少し時間をおいて試してね。")


@client.tree.command(name="double_translate_start", description="指定した分数だけ二重翻訳モードを開始します。")
@app_commands.describe(minutes="二重翻訳モードを有効にする分数")
async def slash_double_translate_start(interaction: discord.Interaction, minutes: int) -> None:
    safe_minutes = max(1, min(minutes, 120))
    start_double_translate(interaction.user.id, safe_minutes)
    await interaction.response.send_message(build_start_message(safe_minutes), ephemeral=True)


@client.tree.command(name="double_translate_stop", description="二重翻訳モードを終了します。")
async def slash_double_translate_stop(interaction: discord.Interaction) -> None:
    stop_double_translate(interaction.user.id)
    await interaction.response.send_message("二重翻訳モードを止めたよ。\nおつかれさま。", ephemeral=True)


@client.tree.command(name="double_translate_status", description="二重翻訳モードの状態を表示します。")
async def slash_double_translate_status(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(build_status_message(interaction.user.id), ephemeral=True)


@client.tree.command(name="jockie_shuffle", description="Jockie Music用のプレイリストシャッフル再生命令を作ります。")
@app_commands.describe(
    url="YouTube再生リストURL",
    prefix="Jockie Musicのプレフィックス。省略時は m!",
)
async def slash_jockie_shuffle(interaction: discord.Interaction, url: str, prefix: str = "m!") -> None:
    if not is_youtube_playlist_url(url):
        await interaction.response.send_message(
            "YouTubeの再生リストURLを指定してね。`playlist?list=...` が含まれるURLが対象だよ。",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(build_jockie_shuffle_response(url, prefix))


@client.tree.command(name="playlist_save", description="YouTube再生リストURLを名前付きで保存します。")
@app_commands.describe(name="保存名", url="YouTube再生リストURL")
async def slash_playlist_save(interaction: discord.Interaction, name: str, url: str) -> None:
    await interaction.response.send_message(save_playlist(interaction.user.id, name, url), ephemeral=True)


@client.tree.command(name="playlist_list", description="保存済みの再生リスト一覧を表示します。")
async def slash_playlist_list(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(list_playlists(interaction.user.id), ephemeral=True)


@client.tree.command(name="playlist_delete", description="保存済みの再生リストを削除します。")
@app_commands.describe(name="保存名")
async def slash_playlist_delete(interaction: discord.Interaction, name: str) -> None:
    await interaction.response.send_message(delete_playlist(interaction.user.id, name), ephemeral=True)


@client.tree.command(name="jockie_shuffle_saved", description="保存済み再生リストからJockie用コマンドを作ります。")
@app_commands.describe(
    name="保存名",
    prefix="Jockie Musicのプレフィックス。省略時は m!",
)
async def slash_jockie_shuffle_saved(interaction: discord.Interaction, name: str, prefix: str = "m!") -> None:
    await interaction.response.send_message(build_saved_jockie_shuffle_response(interaction.user.id, name, prefix))


def main() -> None:
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Create .env and set your bot token.")

    client.run(token)


if __name__ == "__main__":
    main()


