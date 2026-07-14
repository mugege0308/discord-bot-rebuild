from datetime import datetime, timedelta, timezone


MAX_TRANSLATE_LENGTH = 500
JST = timezone(timedelta(hours=9), name="JST")

_active_until_by_user_id: dict[int, datetime] = {}


def now_jst() -> datetime:
    return datetime.now(JST)


def start_double_translate(user_id: int, minutes: int) -> datetime:
    safe_minutes = max(1, min(minutes, 120))
    end_time = now_jst() + timedelta(minutes=safe_minutes)
    _active_until_by_user_id[user_id] = end_time
    return end_time


def stop_double_translate(user_id: int) -> bool:
    return _active_until_by_user_id.pop(user_id, None) is not None


def get_double_translate_end_time(user_id: int) -> datetime | None:
    end_time = _active_until_by_user_id.get(user_id)
    if end_time is None:
        return None

    if end_time <= now_jst():
        _active_until_by_user_id.pop(user_id, None)
        return None

    return end_time


def is_double_translate_active(user_id: int) -> bool:
    return get_double_translate_end_time(user_id) is not None


def remaining_minutes(user_id: int) -> int | None:
    end_time = get_double_translate_end_time(user_id)
    if end_time is None:
        return None

    remaining_seconds = max(0, int((end_time - now_jst()).total_seconds()))
    return max(1, (remaining_seconds + 59) // 60)


def build_start_message(minutes: int) -> str:
    return "\n".join(
        [
            f"{minutes}分間、二重翻訳モードにしたよ。",
            "",
            "この間のあなたの発言を、",
            "日本語 → 英語 → 日本語",
            "にして返すね。",
            "",
            "ちょっと変な言い方になっても、ゆるしてね。",
        ]
    )


def build_status_message(user_id: int) -> str:
    minutes = remaining_minutes(user_id)
    if minutes is None:
        return "今は二重翻訳モードではなさそうだよ。"

    return f"二重翻訳モード中だよ。残りは約{minutes}分くらいかな。"


def should_translate_message(user_id: int, content: str) -> tuple[bool, str | None]:
    if not is_double_translate_active(user_id):
        return False, None

    text = content.strip()
    if not text:
        return False, None

    if len(text) > MAX_TRANSLATE_LENGTH:
        return False, "文章が少し長いみたい。500文字以内に短くしてから試してね。"

    return True, None
