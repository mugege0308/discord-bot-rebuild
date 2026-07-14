import logging

from commands.memo import get_memo_count
from commands.todo import format_pending_todos_for_today
from services.weather import get_weather_message


logger = logging.getLogger("secretary-bot.today")


def _build_memo_summary(user_id: int) -> str:
    memo_count = get_memo_count(user_id)
    if memo_count == 0:
        return "今のところ、メモはなさそうだよ。"

    return f"メモは{memo_count}件あるよ。\n見返すときは /memo_list を使ってね。"


def build_today_summary(user_id: int) -> str:
    try:
        weather_message = get_weather_message("今日")
    except Exception:
        logger.exception("Failed to fetch weather for /today.")
        weather_message = "天気情報を取れなかったみたい。少し時間をおいて試してね。"

    return "\n".join(
        [
            "今日の確認だよ。",
            "",
            "【天気】",
            weather_message,
            "",
            "【ToDo】",
            format_pending_todos_for_today(user_id),
            "",
            "【メモ】",
            _build_memo_summary(user_id),
            "",
            "NISA目安を見たいときは /nisa で確認できるよ。",
            "",
            "今日も無理しすぎないでね。",
        ]
    )
