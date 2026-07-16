from __future__ import annotations

import logging
import os

from services.dev_task_store import MAX_DEV_TASK_TIMEOUT_SECONDS, MAX_SNOWFLAKE_ID


logger = logging.getLogger("secretary-bot.dev_task_config")

DEFAULT_DEV_TASK_TIMEOUT_SECONDS = 1800
DEFAULT_DEV_TASK_BASE_BRANCH = "master"


def _parse_snowflake_id(value_text: str) -> int | None:
    """Parse `value_text` as a Discord snowflake id: a strict base-10
    integer in (0, MAX_SNOWFLAKE_ID] - the same bounds
    services.dev_task_store enforces for every user/guild/channel id field.
    Returns None (never raises) for anything malformed or out of range, so
    every caller here can log-and-skip/log-and-fall-back uniformly.
    """
    try:
        parsed = int(value_text)
    except ValueError:
        return None
    if parsed <= 0 or parsed > MAX_SNOWFLAKE_ID:
        return None
    return parsed


def get_dev_task_allowed_user_ids() -> set[int]:
    """DEV_TASK_ALLOWED_USER_IDS is deliberately its own env var, entirely
    separate from PC_CONTROL_ALLOWED_USER_IDS: being allowed to shut down the
    PC does not imply being allowed to create git worktrees and run dev
    tasks, and vice versa.
    """
    raw_value = os.getenv("DEV_TASK_ALLOWED_USER_IDS", "")
    allowed_user_ids: set[int] = set()

    for raw_user_id in raw_value.split(","):
        user_id_text = raw_user_id.strip()
        if not user_id_text:
            continue

        parsed_user_id = _parse_snowflake_id(user_id_text)
        if parsed_user_id is None:
            logger.warning("Invalid DEV_TASK_ALLOWED_USER_IDS entry: %s", user_id_text)
            continue
        allowed_user_ids.add(parsed_user_id)

    return allowed_user_ids


def is_dev_task_allowed(user_id: int) -> bool:
    return user_id in get_dev_task_allowed_user_ids()


def get_dev_task_timeout_seconds() -> int:
    raw_value = os.getenv("DEV_TASK_TIMEOUT_SECONDS", str(DEFAULT_DEV_TASK_TIMEOUT_SECONDS)).strip()

    try:
        timeout_seconds = int(raw_value)
    except ValueError:
        logger.warning("Invalid DEV_TASK_TIMEOUT_SECONDS: %s", raw_value)
        return DEFAULT_DEV_TASK_TIMEOUT_SECONDS

    if timeout_seconds < 1 or timeout_seconds > MAX_DEV_TASK_TIMEOUT_SECONDS:
        logger.warning(
            "DEV_TASK_TIMEOUT_SECONDS out of range: %s. Using default %s.",
            timeout_seconds,
            DEFAULT_DEV_TASK_TIMEOUT_SECONDS,
        )
        return DEFAULT_DEV_TASK_TIMEOUT_SECONDS

    return timeout_seconds


def get_dev_task_result_channel_id() -> int | None:
    raw_value = os.getenv("DEV_TASK_RESULT_CHANNEL_ID", "").strip()
    if not raw_value:
        return None

    parsed_channel_id = _parse_snowflake_id(raw_value)
    if parsed_channel_id is None:
        logger.warning("Invalid DEV_TASK_RESULT_CHANNEL_ID: %s", raw_value)
        return None
    return parsed_channel_id


def get_dev_task_default_base_branch() -> str:
    raw_value = os.getenv("DEV_TASK_DEFAULT_BASE_BRANCH", "").strip()
    return raw_value or DEFAULT_DEV_TASK_BASE_BRANCH
