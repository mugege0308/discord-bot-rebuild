import asyncio
import logging

import discord

import daily_log
from commands.help import handle_help
from commands.today import build_today_summary
from services.double_translate import (
    build_start_message,
    build_status_message,
    start_double_translate,
    stop_double_translate,
)
from services.jockie import build_jockie_shuffle_response, get_user_playlists
from services.nisa import get_nisa_message
from services.pc_control import cancel_shutdown, format_shutdown_status_message, get_shutdown_delay_seconds, is_pc_control_allowed, is_windows, seconds_until_time
from services.sd_control import get_latest_image_paths, get_status_message, start_sd_app
from services.weather import get_weather_message
from views.pc_control import PcShutdownConfirmView


logger = logging.getLogger("secretary-bot.panel")

PANEL_MESSAGE = "初号ちゃんだよ。\n何をしよっか？"
ONLY_OWNER_MESSAGE = "このパネルは呼び出した本人用だよ。"
GUILD_ONLY_MESSAGE = "この操作はサーバー内で使ってね。"
WEATHER_ERROR_MESSAGE = "ごめんね、天気情報を取れなかったみたい。少し時間をおいて試してね。"
NISA_ERROR_MESSAGE = "ごめんね、NISAの目安を表示できなかったみたい。少し時間をおいて試してね。"
TODAY_ERROR_MESSAGE = "ごめんね、今日のまとめを表示できなかったみたい。少し時間をおいて試してね。"
RECORD_ERROR_MESSAGE = "ごめんね、記録管理の操作に失敗したみたい。少し時間をおいて試してね。"
PC_CONTROL_DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できるよ。"
PC_CONTROL_WINDOWS_ONLY_MESSAGE = "PCシャットダウン機能はWindowsでだけ実行できるよ。"
PC_SHUTDOWN_CANCEL_FAILED_MESSAGE = "シャットダウン予約の取り消しに失敗したみたい。Botのログを確認してね。"


class OwnerOnlyView(discord.ui.View):
    def __init__(self, owner_id: int, bot_mention: str, *, timeout: float = 180) -> None:
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.bot_mention = bot_mention

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return False
        return True


class OperationPanel(OwnerOnlyView):
    @discord.ui.button(label="今日", style=discord.ButtonStyle.primary)
    async def today_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message("今日まわりを見るメニューだよ。", view=TodayPanel(self.owner_id, self.bot_mention), ephemeral=True)

    @discord.ui.button(label="記録", style=discord.ButtonStyle.primary)
    async def records_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message("メモやToDoを見るメニューだよ。", view=RecordsPanel(self.owner_id, self.bot_mention), ephemeral=True)

    @discord.ui.button(label="便利", style=discord.ButtonStyle.secondary)
    async def utility_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message("小さな便利機能のメニューだよ。", view=UtilityPanel(self.owner_id, self.bot_mention), ephemeral=True)

    @discord.ui.button(label="音楽", style=discord.ButtonStyle.secondary)
    async def music_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message("Jockie Music補助メニューだよ。", view=MusicPanel(self.owner_id, self.bot_mention), ephemeral=True)

    @discord.ui.button(label="PC操作", style=discord.ButtonStyle.secondary)
    async def pc_control_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        message = "PC操作メニューだよ。\n許可されたユーザーだけが実行できるよ。"
        await interaction.response.send_message(message, view=PcControlPanel(self.owner_id, self.bot_mention), ephemeral=True)

    @discord.ui.button(label="ヘルプ", style=discord.ButtonStyle.secondary)
    async def help_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message(handle_help(self.bot_mention), ephemeral=True)


class TodayPanel(OwnerOnlyView):
    async def send_weather(self, interaction: discord.Interaction, day: str, *, umbrella_only: bool = False) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            message = await asyncio.to_thread(get_weather_message, day, umbrella_only=umbrella_only)
            await interaction.followup.send(message, ephemeral=True)
        except Exception:
            logger.exception("Failed to handle weather panel button.")
            await interaction.followup.send(WEATHER_ERROR_MESSAGE, ephemeral=True)

    @discord.ui.button(label="今日のまとめ", style=discord.ButtonStyle.primary)
    async def today_summary_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            message = await asyncio.to_thread(build_today_summary, interaction.user.id)
            await interaction.followup.send(message, ephemeral=True)
        except Exception:
            logger.exception("Failed to handle today summary panel button.")
            await interaction.followup.send(TODAY_ERROR_MESSAGE, ephemeral=True)

    @discord.ui.button(label="今日の天気", style=discord.ButtonStyle.primary)
    async def today_weather_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.send_weather(interaction, "今日")

    @discord.ui.button(label="明日の天気", style=discord.ButtonStyle.primary)
    async def tomorrow_weather_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.send_weather(interaction, "明日")

    @discord.ui.button(label="傘いる？", style=discord.ButtonStyle.secondary)
    async def umbrella_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.send_weather(interaction, "今日", umbrella_only=True)


class RecordAddModal(discord.ui.Modal):
    def __init__(self, owner_id: int, record_type: str, title: str) -> None:
        super().__init__(title=title)
        self.owner_id = owner_id
        self.record_type = record_type
        self.content = discord.ui.TextInput(
            label="内容",
            style=discord.TextStyle.paragraph,
            max_length=1000,
            required=True,
        )
        self.add_item(self.content)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return

        try:
            record_id = await asyncio.to_thread(
                daily_log.add_record,
                interaction.guild.id,
                interaction.channel_id or 0,
                interaction.user.id,
                self.record_type,
                str(self.content.value),
            )
        except Exception:
            logger.exception("Failed to add record from panel modal.")
            await interaction.response.send_message(RECORD_ERROR_MESSAGE, ephemeral=True)
            return

        label = {"memo": "メモ", "todo": "TODO", "progress": "進捗"}.get(self.record_type, "記録")
        await interaction.response.send_message(f"{label}を保存したよ。ID: #{record_id}", ephemeral=True)


class DailyReportSettingModal(discord.ui.Modal):
    def __init__(self, owner_id: int) -> None:
        super().__init__(title="日次レポート設定")
        self.owner_id = owner_id
        self.hour = discord.ui.TextInput(label="時", default="7", max_length=2, required=True)
        self.minute = discord.ui.TextInput(label="分", default="0", max_length=2, required=True)
        self.add_item(self.hour)
        self.add_item(self.minute)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return

        try:
            hour = int(str(self.hour.value).strip())
            minute = int(str(self.minute.value).strip())
        except ValueError:
            await interaction.response.send_message("時刻は数字で入力してね。", ephemeral=True)
            return

        response = await asyncio.to_thread(
            daily_log.set_daily_report,
            interaction.guild.id,
            interaction.channel_id or 0,
            hour,
            minute,
        )
        await interaction.response.send_message(response, ephemeral=True)


class RecordsPanel(OwnerOnlyView):
    async def _guild_id_or_reply(self, interaction: discord.Interaction) -> int | None:
        if interaction.guild is None:
            await interaction.response.send_message(GUILD_ONLY_MESSAGE, ephemeral=True)
            return None
        return interaction.guild.id

    @discord.ui.button(label="メモ一覧", style=discord.ButtonStyle.primary)
    async def memo_list_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        guild_id = await self._guild_id_or_reply(interaction)
        if guild_id is None:
            return
        response = await asyncio.to_thread(daily_log.list_recent_records, guild_id, "memo", 10)
        await interaction.response.send_message(response, ephemeral=True)

    @discord.ui.button(label="メモの使い方", style=discord.ButtonStyle.secondary)
    async def memo_usage_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        message = "\n".join(
            [
                "メモはこの方法で使えるよ。",
                "- /memo content:内容",
                "- /memo_list",
                "- /memo_delete number:記録ID",
                "",
                "記録パネルでは、入力画面からメモ追加もできるよ。",
            ]
        )
        await interaction.response.send_message(message, ephemeral=True)

    @discord.ui.button(label="ToDo一覧", style=discord.ButtonStyle.primary)
    async def todo_list_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        guild_id = await self._guild_id_or_reply(interaction)
        if guild_id is None:
            return
        response = await asyncio.to_thread(daily_log.format_pending_todos, guild_id)
        await interaction.response.send_message(response, ephemeral=True)

    @discord.ui.button(label="ToDoの使い方", style=discord.ButtonStyle.secondary)
    async def todo_usage_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        message = "\n".join(
            [
                "ToDoはこの方法で使えるよ。",
                "- /todo content:内容",
                "- /todo_add content:内容",
                "- /todo_list",
                "- /todo_done todo_id:ID",
                "- /todo_delete number:番号",
                "",
                "記録パネルでは、入力画面からTODO追加もできるよ。",
            ]
        )
        await interaction.response.send_message(message, ephemeral=True)

    @discord.ui.button(label="メモ追加", style=discord.ButtonStyle.secondary)
    async def memo_add_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(RecordAddModal(self.owner_id, "memo", "メモ追加"))

    @discord.ui.button(label="TODO追加", style=discord.ButtonStyle.secondary)
    async def todo_add_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(RecordAddModal(self.owner_id, "todo", "TODO追加"))

    @discord.ui.button(label="進捗追加", style=discord.ButtonStyle.secondary)
    async def progress_add_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(RecordAddModal(self.owner_id, "progress", "進捗追加"))

    @discord.ui.button(label="最近の記録", style=discord.ButtonStyle.secondary)
    async def recent_records_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        guild_id = await self._guild_id_or_reply(interaction)
        if guild_id is None:
            return
        response = await asyncio.to_thread(daily_log.list_recent_records, guild_id, "all", 10)
        await interaction.response.send_message(response, ephemeral=True)

    @discord.ui.button(label="日次レポート", style=discord.ButtonStyle.secondary)
    async def daily_report_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        guild_id = await self._guild_id_or_reply(interaction)
        if guild_id is None:
            return
        response = await asyncio.to_thread(daily_log.build_daily_report, guild_id, daily_log.yesterday_jst_text())
        await interaction.response.send_message(response, ephemeral=True)

    @discord.ui.button(label="日次設定", style=discord.ButtonStyle.secondary)
    async def daily_report_setting_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(DailyReportSettingModal(self.owner_id))

    @discord.ui.button(label="日次停止", style=discord.ButtonStyle.danger)
    async def daily_report_off_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        guild_id = await self._guild_id_or_reply(interaction)
        if guild_id is None:
            return
        response = await asyncio.to_thread(daily_log.disable_daily_report, guild_id)
        await interaction.response.send_message(response, ephemeral=True)


class UtilityPanel(OwnerOnlyView):
    @discord.ui.button(label="二重翻訳 5分開始", style=discord.ButtonStyle.primary)
    async def start_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        start_double_translate(interaction.user.id, 5)
        await interaction.response.send_message(build_start_message(5), ephemeral=True)

    @discord.ui.button(label="二重翻訳 状態", style=discord.ButtonStyle.secondary)
    async def status_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message(build_status_message(interaction.user.id), ephemeral=True)

    @discord.ui.button(label="二重翻訳 停止", style=discord.ButtonStyle.danger)
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        stop_double_translate(interaction.user.id)
        await interaction.response.send_message("二重翻訳モードを止めたよ。\nおつかれさま。", ephemeral=True)

    @discord.ui.button(label="NISA目安", style=discord.ButtonStyle.secondary)
    async def nisa_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            await interaction.followup.send(get_nisa_message(), ephemeral=True)
        except Exception:
            logger.exception("Failed to handle NISA panel button.")
            await interaction.followup.send(NISA_ERROR_MESSAGE, ephemeral=True)


class MusicPanel(OwnerOnlyView):
    @discord.ui.button(label="保存済み一覧", style=discord.ButtonStyle.primary)
    async def playlist_list_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        playlists = get_user_playlists(interaction.user.id)
        if not playlists:
            message = "\n".join(
                [
                    "保存済みの再生リストはまだなさそうだよ。",
                    "",
                    "よく使う再生リストは、このコマンドで保存できるよ。",
                    "/playlist_save name:名前 url:YouTube再生リストURL",
                ]
            )
            await interaction.response.send_message(message, ephemeral=True)
            return

        await interaction.response.send_message(
            "保存済み再生リストだよ。\n下から再生したいリストを選んでね。",
            view=SavedPlaylistPanel(self.owner_id, self.bot_mention, playlists),
            ephemeral=True,
        )

    @discord.ui.button(label="使い方", style=discord.ButtonStyle.secondary)
    async def usage_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        message = "\n".join(
            [
                "Jockie Music補助機能だよ。",
                "- /jockie_shuffle url:再生リストURL",
                "- /playlist_save name:名前 url:再生リストURL",
                "- /playlist_list",
                "- /playlist_delete name:名前",
                "- /jockie_shuffle_saved name:名前",
                "",
                "初号ちゃんがJockie Musicを直接操作するのではなく、コピー用コマンドを作るよ。",
            ]
        )
        await interaction.response.send_message(message, ephemeral=True)


class SavedPlaylistPanel(OwnerOnlyView):
    def __init__(self, owner_id: int, bot_mention: str, playlists: dict[str, str]) -> None:
        super().__init__(owner_id, bot_mention)
        self.add_item(SavedPlaylistSelect(owner_id, playlists))


class SavedPlaylistSelect(discord.ui.Select):
    def __init__(self, owner_id: int, playlists: dict[str, str]) -> None:
        self.owner_id = owner_id
        self.playlist_names = list(playlists.keys())[:25]
        options = [
            discord.SelectOption(label=name[:100], value=str(index))
            for index, name in enumerate(self.playlist_names)
        ]
        super().__init__(
            placeholder="保存済み再生リストを選ぶ",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return

        playlists = get_user_playlists(interaction.user.id)
        selected_index = int(self.values[0])
        name = self.playlist_names[selected_index]
        url = playlists.get(name)
        if url is None:
            await interaction.response.send_message("その再生リストは見つからなかったよ。/playlist_list で確認してね。", ephemeral=True)
            return

        await interaction.response.send_message(build_jockie_shuffle_response(url), ephemeral=True)


class TimedShutdownModal(discord.ui.Modal):
    def __init__(self, owner_id: int) -> None:
        super().__init__(title="時間指定シャットダウン")
        self.owner_id = owner_id
        self.time = discord.ui.TextInput(
            label="終了時刻",
            placeholder="例: 2330 / 23:30 / 23時30分 / 23時",
            max_length=20,
            required=True,
        )
        self.add_item(self.time)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return

        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied timed PC shutdown modal by user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
            return

        if not is_windows():
            logger.warning("Denied timed PC shutdown modal because OS is not Windows. user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
            return

        try:
            delay_seconds, target = seconds_until_time(str(self.time.value))
        except ValueError:
            await interaction.response.send_message(
                "時刻の形が少し分からなかったみたい。\n`2330` や `23:30`、`23時30分` の形で入れてみてね。",
                ephemeral=True,
            )
            return

        target_label = target.strftime("%H:%M")
        day_label = "今日" if target.date() == discord.utils.utcnow().astimezone(target.tzinfo).date() else "翌日"
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


class PcControlPanel(OwnerOnlyView):
    @discord.ui.button(label="シャットダウン", style=discord.ButtonStyle.danger)
    async def shutdown_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied PC shutdown panel button by user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
            return

        if not is_windows():
            logger.warning("Denied PC shutdown panel button because OS is not Windows. user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
            return

        delay_seconds = get_shutdown_delay_seconds()
        logger.warning(
            "PC shutdown confirmation requested from panel by user_id=%s delay_seconds=%s",
            interaction.user.id,
            delay_seconds,
        )
        await interaction.response.send_message(
            "\n".join(
                [
                    f"PCを{delay_seconds}秒後にシャットダウンするよ。",
                    "本当に実行してもだいじょうぶ？",
                ]
            ),
            view=PcShutdownConfirmView(
                interaction.user.id,
                delay_seconds,
                description=f"{delay_seconds}秒後のシャットダウン",
            ),
            ephemeral=True,
        )

    @discord.ui.button(label="時間指定シャットダウン", style=discord.ButtonStyle.secondary)
    async def timed_shutdown_usage_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(TimedShutdownModal(self.owner_id))

    @discord.ui.button(label="予約状況", style=discord.ButtonStyle.secondary)
    async def shutdown_status_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied PC shutdown status panel button by user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
            return

        await interaction.response.send_message(format_shutdown_status_message(), ephemeral=True)

    @discord.ui.button(label="シャットダウン取消", style=discord.ButtonStyle.secondary)
    async def shutdown_cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            logger.warning("Denied PC shutdown cancel panel button by user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_DENIED_MESSAGE, ephemeral=True)
            return

        if not is_windows():
            logger.warning("Denied PC shutdown cancel panel button because OS is not Windows. user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_CONTROL_WINDOWS_ONLY_MESSAGE, ephemeral=True)
            return

        try:
            await asyncio.to_thread(cancel_shutdown)
        except Exception:
            logger.exception("Failed to cancel PC shutdown from panel by user_id=%s", interaction.user.id)
            await interaction.response.send_message(PC_SHUTDOWN_CANCEL_FAILED_MESSAGE, ephemeral=True)
            return

        logger.warning("PC shutdown cancelled from panel by user_id=%s", interaction.user.id)
        await interaction.response.send_message("予約済みシャットダウンを取り消したよ。", ephemeral=True)

    @discord.ui.button(label="使い方", style=discord.ButtonStyle.secondary)
    async def usage_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        message = "\n".join(
            [
                "PC操作はこの方法で使えるよ。",
                "- /pc_shutdown",
                "- /pc_shutdown_status",
                "- /pc_shutdown_cancel",
                "- PC操作パネルからも実行可能",
                "",
                "注意:",
                "- 許可ユーザーだけが実行できるよ",
                "- Botが起動しているPCだけを操作できるよ",
                "- シャットダウンは確認ボタンを押すまで実行されないよ",
            ]
        )
        await interaction.response.send_message(message, ephemeral=True)


