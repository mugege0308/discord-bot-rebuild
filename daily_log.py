import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import tasks

from services.weather import JST


logger = logging.getLogger("secretary-bot.daily_log")

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "daily_log.db"
VALID_RECORD_TYPES = {"memo", "todo", "progress"}
VALID_LOG_KINDS = {"memo", "todo", "progress", "all"}


def now_jst() -> datetime:
    return datetime.now(JST)


def today_jst_text() -> str:
    return now_jst().date().isoformat()


def yesterday_jst_text() -> str:
    return (now_jst().date() - timedelta(days=1)).isoformat()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                done_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_settings (
                guild_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL,
                hour INTEGER NOT NULL DEFAULT 7,
                minute INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                last_posted_for TEXT
            )
            """
        )
        connection.commit()


def _connect() -> sqlite3.Connection:
    init_db()
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def add_record(guild_id: int, channel_id: int, user_id: int, record_type: str, content: str) -> int:
    record_content = content.strip()
    if record_type not in VALID_RECORD_TYPES:
        raise ValueError(f"Invalid record type: {record_type}")
    if not record_content:
        raise ValueError("Record content is empty.")

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO records (guild_id, channel_id, user_id, type, content, done, created_at, done_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, NULL)
            """,
            (guild_id, channel_id, user_id, record_type, record_content, now_jst().isoformat(timespec="seconds")),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_pending_todos(guild_id: int, *, limit: int = 30) -> list[sqlite3.Row]:
    safe_limit = max(1, min(limit, 30))
    with _connect() as connection:
        return list(
            connection.execute(
                """
                SELECT id, content, user_id, created_at
                FROM records
                WHERE guild_id = ? AND type = 'todo' AND done = 0
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (guild_id, safe_limit),
            )
        )


def format_pending_todos(guild_id: int) -> str:
    rows = list_pending_todos(guild_id)
    if not rows:
        return "今のところ、やることは残ってなさそうだよ。\nえらいね。"

    lines = ["未完了TODOだよ。", ""]
    lines.extend(f"#{row['id']} {row['content']}" for row in rows)
    return "\n".join(lines)


def mark_todo_done(guild_id: int, todo_id: int) -> str:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, content
            FROM records
            WHERE guild_id = ? AND id = ? AND type = 'todo' AND done = 0
            """,
            (guild_id, todo_id),
        ).fetchone()
        if row is None:
            return "そのIDの未完了TODOは見つからなかったよ。"

        connection.execute(
            "UPDATE records SET done = 1, done_at = ? WHERE id = ?",
            (now_jst().isoformat(timespec="seconds"), todo_id),
        )
        connection.commit()

    return f"#{todo_id} 「{row['content']}」を完了にしたよ。\nおつかれさま。"


def delete_record(guild_id: int, record_id: int) -> str:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM records WHERE guild_id = ? AND id = ?",
            (guild_id, record_id),
        )
        connection.commit()

    if cursor.rowcount == 0:
        return "そのIDの記録は見つからなかったよ。"
    return f"#{record_id} の記録を削除したよ。"


def list_recent_records(guild_id: int, kind: str = "all", limit: int = 10) -> str:
    record_kind = kind.lower().strip()
    if record_kind not in VALID_LOG_KINDS:
        return "kind は memo / todo / progress / all のどれかを指定してね。"

    safe_limit = max(1, min(limit, 30))
    params: list[Any] = [guild_id]
    where = "guild_id = ?"
    if record_kind != "all":
        where += " AND type = ?"
        params.append(record_kind)
    params.append(safe_limit)

    with _connect() as connection:
        rows = list(
            connection.execute(
                f"""
                SELECT id, type, content, done, created_at
                FROM records
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            )
        )

    if not rows:
        return "記録はまだなさそうだよ。"

    lines = ["最近の記録だよ。", ""]
    for row in rows:
        status = " 完了" if row["type"] == "todo" and row["done"] else ""
        lines.append(f"#{row['id']} [{row['type']}{status}] {row['content']}")
    return "\n".join(lines)


def set_daily_report(guild_id: int, channel_id: int, hour: int, minute: int) -> str:
    safe_hour = max(0, min(hour, 23))
    safe_minute = max(0, min(minute, 59))
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO daily_settings (guild_id, channel_id, hour, minute, enabled, last_posted_for)
            VALUES (?, ?, ?, ?, 1, NULL)
            ON CONFLICT(guild_id) DO UPDATE SET
                channel_id = excluded.channel_id,
                hour = excluded.hour,
                minute = excluded.minute,
                enabled = 1
            """,
            (guild_id, channel_id, safe_hour, safe_minute),
        )
        connection.commit()

    return f"日次レポートを <#{channel_id}> に {safe_hour:02d}:{safe_minute:02d} 投稿するようにしたよ。"


def disable_daily_report(guild_id: int) -> str:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO daily_settings (guild_id, channel_id, hour, minute, enabled, last_posted_for)
            VALUES (?, 0, 7, 0, 0, NULL)
            ON CONFLICT(guild_id) DO UPDATE SET enabled = 0
            """,
            (guild_id,),
        )
        connection.commit()

    return "日次レポートの自動投稿を止めておいたよ。"


def _rows_for_date(guild_id: int, target_date: str, record_type: str) -> list[sqlite3.Row]:
    with _connect() as connection:
        return list(
            connection.execute(
                """
                SELECT id, content, done
                FROM records
                WHERE guild_id = ? AND type = ? AND substr(created_at, 1, 10) = ?
                ORDER BY created_at ASC, id ASC
                """,
                (guild_id, record_type, target_date),
            )
        )


def _pending_todos_for_report(guild_id: int) -> list[sqlite3.Row]:
    return list_pending_todos(guild_id, limit=30)


def _format_section(rows: list[sqlite3.Row], empty_message: str) -> list[str]:
    if not rows:
        return [empty_message]
    return [f"- #{row['id']} {row['content']}" for row in rows]


def build_daily_report(guild_id: int, target_date: str) -> str:
    progress_rows = _rows_for_date(guild_id, target_date, "progress")
    memo_rows = _rows_for_date(guild_id, target_date, "memo")
    todo_rows = _rows_for_date(guild_id, target_date, "todo")
    pending_todos = _pending_todos_for_report(guild_id)

    summary = (
        f"進捗が{len(progress_rows)}件、メモが{len(memo_rows)}件、TODOが{len(todo_rows)}件あったよ。"
    )
    if pending_todos:
        summary += " 未完了TODOも残っているから、次はそこから見ていくとよさそうかな。"
    else:
        summary += " 未完了TODOはなさそうだよ。ちゃんと進んでいて、えらいね。"

    lines = [
        f"📘 {target_date} の進行内容",
        "",
        "✅ 進捗",
        *_format_section(progress_rows, "- 進捗の記録はなさそうだよ。"),
        "",
        "📝 メモ",
        *_format_section(memo_rows, "- メモはなさそうだよ。"),
        "",
        "📌 昨日追加されたTODO",
        *_format_section(todo_rows, "- 昨日追加されたTODOはなさそうだよ。"),
        "",
        "📍 未完了TODO",
        *_format_section(pending_todos, "- 未完了TODOはなさそうだよ。"),
        "",
        "🤖 まとめ",
        summary,
    ]
    return "\n".join(lines)


def get_due_daily_settings(hour: int, minute: int, target_date: str) -> list[sqlite3.Row]:
    with _connect() as connection:
        return list(
            connection.execute(
                """
                SELECT guild_id, channel_id
                FROM daily_settings
                WHERE enabled = 1
                  AND hour = ?
                  AND minute = ?
                  AND channel_id > 0
                  AND (last_posted_for IS NULL OR last_posted_for != ?)
                """,
                (hour, minute, target_date),
            )
        )


def mark_daily_report_posted(guild_id: int, target_date: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE daily_settings SET last_posted_for = ? WHERE guild_id = ?",
            (target_date, guild_id),
        )
        connection.commit()


def _guild_channel_user(interaction: discord.Interaction) -> tuple[int, int, int] | None:
    if interaction.guild is None:
        return None
    return interaction.guild.id, interaction.channel_id or 0, interaction.user.id


def _guild_id(interaction: discord.Interaction) -> int | None:
    if interaction.guild is None:
        return None
    return interaction.guild.id


async def _send_guild_only_message(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("このコマンドはサーバー内で使ってね。", ephemeral=True)


async def post_due_daily_reports(client: discord.Client) -> None:
    now = now_jst()
    target_date = (now.date() - timedelta(days=1)).isoformat()
    settings = get_due_daily_settings(now.hour, now.minute, target_date)
    for setting in settings:
        guild_id = int(setting["guild_id"])
        channel_id = int(setting["channel_id"])
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            if not isinstance(channel, discord.abc.Messageable):
                logger.error("Daily report channel is not messageable. guild_id=%s channel_id=%s", guild_id, channel_id)
                continue

            report = await asyncio.to_thread(build_daily_report, guild_id, target_date)
            await channel.send(report)
            mark_daily_report_posted(guild_id, target_date)
            logger.info("Posted daily report. guild_id=%s target_date=%s", guild_id, target_date)
        except Exception:
            logger.exception("Failed to post daily report. guild_id=%s channel_id=%s", guild_id, channel_id)


def setup(client: discord.Client) -> None:
    init_db()

    @client.tree.command(name="todo", description="TODOを記録します。")
    @app_commands.describe(content="TODO内容")
    async def slash_todo(interaction: discord.Interaction, content: str) -> None:
        ids = _guild_channel_user(interaction)
        if ids is None:
            await _send_guild_only_message(interaction)
            return
        guild_id, channel_id, user_id = ids
        try:
            record_id = await asyncio.to_thread(add_record, guild_id, channel_id, user_id, "todo", content)
        except ValueError:
            await interaction.response.send_message("TODO内容を入力してね。", ephemeral=True)
            return
        await interaction.response.send_message(f"TODOに入れておいたよ。ID: #{record_id}", ephemeral=True)

    @client.tree.command(name="progress", description="今日の進捗を記録します。")
    @app_commands.describe(content="進捗内容")
    async def slash_progress(interaction: discord.Interaction, content: str) -> None:
        ids = _guild_channel_user(interaction)
        if ids is None:
            await _send_guild_only_message(interaction)
            return
        guild_id, channel_id, user_id = ids
        try:
            record_id = await asyncio.to_thread(add_record, guild_id, channel_id, user_id, "progress", content)
        except ValueError:
            await interaction.response.send_message("進捗内容を入力してね。", ephemeral=True)
            return
        await interaction.response.send_message(f"進捗を記録しておいたよ。ID: #{record_id}", ephemeral=True)

    @client.tree.command(name="log_list", description="最近の記録を表示します。")
    @app_commands.describe(kind="memo / todo / progress / all", limit="表示件数。最大30件")
    @app_commands.choices(
        kind=[
            app_commands.Choice(name="memo", value="memo"),
            app_commands.Choice(name="todo", value="todo"),
            app_commands.Choice(name="progress", value="progress"),
            app_commands.Choice(name="all", value="all"),
        ]
    )
    async def slash_log_list(interaction: discord.Interaction, kind: str = "all", limit: int = 10) -> None:
        guild_id = _guild_id(interaction)
        if guild_id is None:
            await _send_guild_only_message(interaction)
            return
        response = await asyncio.to_thread(list_recent_records, guild_id, kind, limit)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="log_delete", description="指定IDの記録を削除します。")
    @app_commands.describe(record_id="削除する記録ID")
    async def slash_log_delete(interaction: discord.Interaction, record_id: int) -> None:
        guild_id = _guild_id(interaction)
        if guild_id is None:
            await _send_guild_only_message(interaction)
            return
        response = await asyncio.to_thread(delete_record, guild_id, record_id)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="daily_report_set", description="日次レポートの投稿先と時刻を設定します。")
    @app_commands.describe(channel="投稿先チャンネル", hour="投稿時。0-23", minute="投稿分。0-59")
    async def slash_daily_report_set(
        interaction: discord.Interaction,
        channel: discord.TextChannel,
        hour: int = 7,
        minute: int = 0,
    ) -> None:
        guild_id = _guild_id(interaction)
        if guild_id is None:
            await _send_guild_only_message(interaction)
            return
        response = await asyncio.to_thread(set_daily_report, guild_id, channel.id, hour, minute)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="daily_report_off", description="日次レポートの自動投稿を停止します。")
    async def slash_daily_report_off(interaction: discord.Interaction) -> None:
        guild_id = _guild_id(interaction)
        if guild_id is None:
            await _send_guild_only_message(interaction)
            return
        response = await asyncio.to_thread(disable_daily_report, guild_id)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="daily_report_now", description="指定日のレポートを今すぐ投稿します。")
    @app_commands.describe(date="YYYY-MM-DD。未指定なら昨日分")
    async def slash_daily_report_now(interaction: discord.Interaction, date: str = "") -> None:
        guild_id = _guild_id(interaction)
        if guild_id is None:
            await _send_guild_only_message(interaction)
            return

        target_date = date.strip() or yesterday_jst_text()
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            await interaction.response.send_message("date は YYYY-MM-DD 形式で指定してね。", ephemeral=True)
            return

        await interaction.response.defer()
        response = await asyncio.to_thread(build_daily_report, guild_id, target_date)
        await interaction.followup.send(response)

    @tasks.loop(minutes=1)
    async def daily_report_loop() -> None:
        await post_due_daily_reports(client)

    @daily_report_loop.before_loop
    async def before_daily_report_loop() -> None:
        await client.wait_until_ready()

    client.daily_report_loop = daily_report_loop  # type: ignore[attr-defined]
