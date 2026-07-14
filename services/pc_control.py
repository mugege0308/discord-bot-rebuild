import json
import logging
import os
import platform
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from services.weather import JST


logger = logging.getLogger("secretary-bot.pc_control")

DATA_DIR = Path("data")
SHUTDOWN_STATE_FILE = DATA_DIR / "pc_shutdown_state.json"
DEFAULT_SHUTDOWN_DELAY_SECONDS = 60
MIN_SHUTDOWN_DELAY_SECONDS = 10
MAX_SHUTDOWN_DELAY_SECONDS = 3600
MIN_TIMED_SHUTDOWN_MINUTES = 1
MAX_TIMED_SHUTDOWN_MINUTES = 1440
MAX_SCHEDULED_SHUTDOWN_SECONDS = 86400
MIN_AT_SHUTDOWN_SECONDS = 60


def get_allowed_user_ids() -> set[int]:
    raw_value = os.getenv("PC_CONTROL_ALLOWED_USER_IDS", "")
    allowed_user_ids: set[int] = set()

    for raw_user_id in raw_value.split(","):
        user_id_text = raw_user_id.strip()
        if not user_id_text:
            continue

        try:
            allowed_user_ids.add(int(user_id_text))
        except ValueError:
            logger.warning("Invalid PC_CONTROL_ALLOWED_USER_IDS entry: %s", user_id_text)

    return allowed_user_ids


def is_pc_control_allowed(user_id: int) -> bool:
    return user_id in get_allowed_user_ids()


def get_shutdown_delay_seconds() -> int:
    raw_value = os.getenv("PC_SHUTDOWN_DELAY_SECONDS", str(DEFAULT_SHUTDOWN_DELAY_SECONDS)).strip()

    try:
        delay_seconds = int(raw_value)
    except ValueError:
        logger.warning("Invalid PC_SHUTDOWN_DELAY_SECONDS: %s", raw_value)
        return DEFAULT_SHUTDOWN_DELAY_SECONDS

    if delay_seconds < MIN_SHUTDOWN_DELAY_SECONDS or delay_seconds > MAX_SHUTDOWN_DELAY_SECONDS:
        logger.warning(
            "PC_SHUTDOWN_DELAY_SECONDS out of range: %s. Using default %s.",
            delay_seconds,
            DEFAULT_SHUTDOWN_DELAY_SECONDS,
        )
        return DEFAULT_SHUTDOWN_DELAY_SECONDS

    return delay_seconds


def seconds_from_minutes(minutes: int) -> int:
    if minutes < MIN_TIMED_SHUTDOWN_MINUTES or minutes > MAX_TIMED_SHUTDOWN_MINUTES:
        raise ValueError("minutes must be between 1 and 1440.")
    return minutes * 60


def parse_shutdown_time_input(text: str, now: datetime | None = None) -> datetime:
    text = text.strip()
    if not text:
        raise ValueError("time is empty.")

    colon_match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    japanese_match = re.fullmatch(r"(\d{1,2})\s*時\s*(?:(\d{1,2})\s*分?)?", text)
    digit_match = re.fullmatch(r"\d{3,4}", text)

    if colon_match is not None:
        hour = int(colon_match.group(1))
        minute = int(colon_match.group(2))
    elif japanese_match is not None:
        hour = int(japanese_match.group(1))
        minute = int(japanese_match.group(2) or 0)
    elif digit_match is not None:
        if len(text) == 3:
            hour = int(text[0])
            minute = int(text[1:])
        else:
            hour = int(text[:2])
            minute = int(text[2:])
    else:
        raise ValueError("time format is invalid.")

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time is out of range.")

    current = now or datetime.now(JST)
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)

    return target


def seconds_until_time(time_text: str, now: datetime | None = None) -> tuple[int, datetime]:
    current = now or datetime.now(JST)
    target = parse_shutdown_time_input(time_text, current)
    delay_seconds = int((target - current).total_seconds())
    if delay_seconds < MIN_AT_SHUTDOWN_SECONDS:
        raise ValueError("time is too close.")
    if delay_seconds > MAX_SCHEDULED_SHUTDOWN_SECONDS:
        raise ValueError("time must be within 24 hours.")

    return delay_seconds, target


def is_windows() -> bool:
    return platform.system() == "Windows"


def _write_shutdown_state(state: dict[str, object]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SHUTDOWN_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_shutdown_state() -> None:
    try:
        SHUTDOWN_STATE_FILE.unlink(missing_ok=True)
    except OSError:
        logger.exception("Failed to delete PC shutdown state file.")


def load_shutdown_state() -> dict[str, object] | None:
    if not SHUTDOWN_STATE_FILE.exists():
        return None
    try:
        value = json.loads(SHUTDOWN_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read PC shutdown state file.")
        return None
    if not isinstance(value, dict):
        return None
    return value


def format_shutdown_status_message() -> str:
    state = load_shutdown_state()
    if state is None:
        return "Botが最後に覚えているシャットダウン予約はないよ。"

    target_at_text = str(state.get("target_at", ""))
    scheduled_at_text = str(state.get("scheduled_at", ""))
    description = str(state.get("description") or "シャットダウン予約")
    delay_seconds = state.get("delay_seconds")
    requested_by = state.get("requested_by_user_id")

    lines = [
        "Botが最後に予約したシャットダウン内容だよ。",
        "",
        f"内容: {description}",
    ]
    if delay_seconds is not None:
        lines.append(f"予約秒数: {delay_seconds}秒")
    if target_at_text:
        lines.append(f"予定時刻: {target_at_text}")
        try:
            target_at = datetime.fromisoformat(target_at_text)
            if datetime.now(JST) >= target_at:
                lines.append("状態: 予定時刻は過ぎているよ。Windows側で実行済み、または予約が消えている可能性があるよ。")
            else:
                lines.append("状態: Botの記録上は予約中だよ。")
        except ValueError:
            lines.append("状態: 予定時刻の形式を読み取れなかったよ。")
    if scheduled_at_text:
        lines.append(f"Botが記録した時刻: {scheduled_at_text}")
    if requested_by is not None:
        lines.append(f"実行ユーザーID: {requested_by}")

    lines.extend(
        [
            "",
            "注意: これはWindows側の予約を直接検出した結果ではなく、Botが最後に保存した予約内容だよ。",
        ]
    )
    return "\n".join(lines)


def schedule_shutdown(
    delay_seconds: int,
    requested_by_user_id: int | None = None,
    description: str | None = None,
) -> None:
    if not is_windows():
        raise RuntimeError("PC shutdown is supported only on Windows.")

    if delay_seconds < MIN_SHUTDOWN_DELAY_SECONDS or delay_seconds > MAX_SCHEDULED_SHUTDOWN_SECONDS:
        raise ValueError("delay_seconds must be between 10 and 86400.")

    now = datetime.now(JST)
    target_at = now + timedelta(seconds=int(delay_seconds))
    command = ["shutdown", "/s", "/t", str(int(delay_seconds))]
    logger.warning("Scheduling PC shutdown with fixed command: %s", command)
    subprocess.run(command, check=True, shell=False, timeout=10)
    _write_shutdown_state(
        {
            "scheduled_at": now.isoformat(timespec="seconds"),
            "target_at": target_at.isoformat(timespec="seconds"),
            "delay_seconds": int(delay_seconds),
            "requested_by_user_id": requested_by_user_id,
            "description": description or f"{int(delay_seconds)}秒後のシャットダウン",
        }
    )


def cancel_shutdown() -> None:
    if not is_windows():
        raise RuntimeError("PC shutdown cancellation is supported only on Windows.")

    command = ["shutdown", "/a"]
    logger.warning("Cancelling scheduled PC shutdown with fixed command: %s", command)
    try:
        subprocess.run(command, check=True, shell=False, timeout=10)
    finally:
        clear_shutdown_state()
