import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


logger = logging.getLogger("secretary-bot.weather")

DATA_DIR = Path("data")
WEATHER_STATE_FILE = DATA_DIR / "weather_state.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

TODAY = "today"
TOMORROW = "tomorrow"
MORNING_NOTIFICATION = "morning"
EVENING_NOTIFICATION = "evening"


@dataclass(frozen=True)
class WeatherRequest:
    day: str = TODAY
    umbrella_only: bool = False


def get_jst_timezone():
    try:
        return ZoneInfo("Asia/Tokyo")
    except ZoneInfoNotFoundError:
        logger.warning("Asia/Tokyo timezone data was not found. Falling back to UTC+09:00.")
        return timezone(timedelta(hours=9), name="JST")


JST = get_jst_timezone()

WEATHER_CODE_MESSAGES = {
    0: "よく晴れそうだよ。",
    1: "おおむね晴れそうだよ。",
    2: "少し雲が出そうだよ。",
    3: "くもり空になりそうだよ。",
    45: "霧が出るかもしれないね。",
    48: "霧が出るかもしれないね。",
    51: "少し雨が降るかもしれないね。",
    53: "雨が降る時間がありそうだよ。",
    55: "しっかり雨が降るかもしれないね。",
    61: "雨が降る時間がありそうだよ。",
    63: "しっかり雨が降るかもしれないね。",
    65: "強い雨に気をつけてね。",
    80: "にわか雨がありそうだよ。",
    81: "にわか雨がありそうだよ。",
    82: "強いにわか雨に気をつけてね。",
    95: "雷雨に気をつけてね。",
}


def get_weather_channel_id() -> int | None:
    raw_channel_id = os.getenv("WEATHER_CHANNEL_ID", "").strip()
    if not raw_channel_id:
        return None

    try:
        return int(raw_channel_id)
    except ValueError:
        logger.error("WEATHER_CHANNEL_ID must be a number: %s", raw_channel_id)
        return None


def get_notify_after_hour() -> int:
    raw_hour = os.getenv("WEATHER_NOTIFY_AFTER_HOUR", "6").strip()
    try:
        hour = int(raw_hour)
    except ValueError:
        logger.error("WEATHER_NOTIFY_AFTER_HOUR must be a number: %s", raw_hour)
        return 6

    if hour < 0 or hour > 23:
        logger.error("WEATHER_NOTIFY_AFTER_HOUR must be between 0 and 23: %s", raw_hour)
        return 6

    return hour


def get_evening_notify_after_hour() -> int:
    raw_hour = os.getenv("WEATHER_EVENING_NOTIFY_AFTER_HOUR", "18").strip()
    try:
        hour = int(raw_hour)
    except ValueError:
        logger.error("WEATHER_EVENING_NOTIFY_AFTER_HOUR must be a number: %s", raw_hour)
        return 18

    if hour < 0 or hour > 23:
        logger.error("WEATHER_EVENING_NOTIFY_AFTER_HOUR must be between 0 and 23: %s", raw_hour)
        return 18

    return hour


def get_startup_weather_notification_kind(now: datetime | None = None) -> str | None:
    now = now or datetime.now(JST)
    morning_hour = get_notify_after_hour()
    evening_hour = get_evening_notify_after_hour()

    if now.hour < morning_hour:
        return None
    if now.hour >= evening_hour:
        return EVENING_NOTIFICATION

    return MORNING_NOTIFICATION


def get_startup_weather_skip_reason(now: datetime | None = None) -> str | None:
    now = now or datetime.now(JST)
    kind = get_startup_weather_notification_kind(now)
    if kind is None:
        return "before_hour"

    if get_last_notified_date(kind) == now.date().isoformat():
        return f"{kind}_already_notified"

    return None


def should_notify_weather(now: datetime | None = None) -> bool:
    return get_startup_weather_skip_reason(now) is None


def _load_weather_state() -> dict:
    if not WEATHER_STATE_FILE.exists():
        return {}

    try:
        with WEATHER_STATE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        logger.exception("Failed to decode weather state: %s", WEATHER_STATE_FILE)
        return {}
    except OSError:
        logger.exception("Failed to read weather state: %s", WEATHER_STATE_FILE)
        return {}

    if not isinstance(data, dict):
        logger.error("Weather state is not a dictionary: %s", WEATHER_STATE_FILE)
        return {}

    return data


def get_last_notified_date(kind: str = MORNING_NOTIFICATION) -> str | None:
    data = _load_weather_state()
    key = "evening_last_notified_date" if kind == EVENING_NOTIFICATION else "morning_last_notified_date"
    value = data.get(key)
    if isinstance(value, str):
        return value

    # Backward compatibility with the old one-notification-per-day state.
    if kind == MORNING_NOTIFICATION:
        old_value = data.get("last_notified_date")
        if isinstance(old_value, str):
            return old_value

    return None


def mark_weather_notified(kind: str = MORNING_NOTIFICATION, date_text: str | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    notified_date = date_text or datetime.now(JST).date().isoformat()
    data = _load_weather_state()
    key = "evening_last_notified_date" if kind == EVENING_NOTIFICATION else "morning_last_notified_date"
    data[key] = notified_date
    with WEATHER_STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_weather_day(day: str | None) -> str:
    if not day:
        return TODAY

    normalized = day.strip().lower()
    if normalized in {"明日", "あした", "tomorrow", TOMORROW}:
        return TOMORROW

    return TODAY


def day_to_offset(day: str) -> int:
    return 1 if normalize_weather_day(day) == TOMORROW else 0


def day_to_label(day: str) -> str:
    return "明日" if normalize_weather_day(day) == TOMORROW else "今日"


def umbrella_note(precipitation_probability: int, precipitation_sum: float) -> str:
    if precipitation_probability >= 40 or precipitation_sum > 0:
        return "折りたたみ傘、持っていくと安心だよ。"

    return "雨の心配は少なそうだよ。"


def weather_note(weather_code: int, precipitation_probability: int, precipitation_sum: float) -> str:
    note = WEATHER_CODE_MESSAGES.get(weather_code)
    umbrella = umbrella_note(precipitation_probability, precipitation_sum)
    if note and umbrella != "雨の心配は少なそうだよ。":
        return f"{note}\n{umbrella}"
    if note:
        return note
    return umbrella


def fetch_weather(day: str = TODAY) -> dict[str, float | int | str]:
    import requests

    day = normalize_weather_day(day)
    latitude = float(os.getenv("WEATHER_LATITUDE", "35.0116"))
    longitude = float(os.getenv("WEATHER_LONGITUDE", "135.7681"))

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "Asia/Tokyo",
        "forecast_days": 2,
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
            ]
        ),
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    daily = data["daily"]
    index = day_to_offset(day)

    return {
        "date": daily["time"][index],
        "day": day,
        "weather_code": daily["weather_code"][index],
        "temperature_max": daily["temperature_2m_max"][index],
        "temperature_min": daily["temperature_2m_min"][index],
        "precipitation_sum": daily["precipitation_sum"][index],
        "precipitation_probability": daily["precipitation_probability_max"][index],
    }


def build_weather_message(
    weather: dict[str, float | int | str],
    *,
    include_greeting: bool = False,
    umbrella_only: bool = False,
) -> str:
    location_name = os.getenv("WEATHER_LOCATION_NAME", "京都").strip() or "京都"
    day = normalize_weather_day(str(weather.get("day", TODAY)))
    label = day_to_label(day)
    weather_code = int(weather["weather_code"])
    precipitation_probability = int(weather["precipitation_probability"])
    precipitation_sum = float(weather["precipitation_sum"])

    if umbrella_only:
        has_rain = umbrella_note(precipitation_probability, precipitation_sum) != "雨の心配は少なそうだよ。"
        first_line = f"{label}は少し雨の可能性があるかも。" if has_rain else f"{label}は雨の心配は少なそうだよ。"
        return "\n".join(
            [
                first_line,
                f"降水確率は{precipitation_probability}%、降水量は{precipitation_sum}mmだよ。",
                umbrella_note(precipitation_probability, precipitation_sum),
            ]
        )

    lines = []
    if include_greeting:
        lines.extend(["おはよう。", f"{label}の{location_name}の天気を見ておくね。"])
    else:
        lines.append(f"{label}の{location_name}の天気だよ。")

    lines.extend(
        [
            "",
            f"最高気温: {weather['temperature_max']}℃",
            f"最低気温: {weather['temperature_min']}℃",
            f"降水確率: {precipitation_probability}%",
            f"降水量: {precipitation_sum}mm",
            "",
            weather_note(weather_code, precipitation_probability, precipitation_sum),
        ]
    )

    if include_greeting:
        lines.extend(["", "今日も無理しすぎないでね。"])

    return "\n".join(lines)


def build_startup_weather_message(weather: dict[str, float | int | str], kind: str) -> str:
    if kind == EVENING_NOTIFICATION:
        base_message = build_weather_message(weather)
        return "\n".join(
            [
                "おつかれさま。",
                "今日も一日、ちゃんとがんばったね。",
                "",
                base_message,
                "",
                "今夜はゆっくり休んでね。",
            ]
        )

    return build_weather_message(weather, include_greeting=True)


def get_weather_message(
    day: str = TODAY,
    *,
    include_greeting: bool = False,
    umbrella_only: bool = False,
) -> str:
    return build_weather_message(
        fetch_weather(day),
        include_greeting=include_greeting,
        umbrella_only=umbrella_only,
    )


def get_startup_weather_message(kind: str) -> str:
    day = TOMORROW if kind == EVENING_NOTIFICATION else TODAY
    return build_startup_weather_message(fetch_weather(day), kind)


def get_today_weather_message() -> str:
    return get_weather_message(TODAY, include_greeting=True)


def parse_weather_request(message: str) -> WeatherRequest | None:
    text = message.strip().replace("　", " ").lower()
    if not text or len(text) > 50:
        return None

    has_weather_word = "天気" in text
    has_rain_word = "雨" in text or "傘" in text
    mentions_tomorrow = "明日" in text or "あした" in text
    short_tomorrow_question = mentions_tomorrow and len(text) <= 12 and (
        "?" in text or "？" in text or text.rstrip("？") in {"明日は", "あしたは"}
    )

    if not (has_weather_word or has_rain_word or short_tomorrow_question):
        return None

    day = TOMORROW if mentions_tomorrow else TODAY
    return WeatherRequest(day=day, umbrella_only=has_rain_word)
