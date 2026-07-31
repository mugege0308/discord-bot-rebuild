"""Permission gate for character-dictionary admin/diagnostic commands.

CHARACTER_ADMIN_ALLOWED_USER_IDS is deliberately its own env var, mirroring
services/pc_control.py's PC_CONTROL_ALLOWED_USER_IDS and services/
dev_task_config.py's DEV_TASK_ALLOWED_USER_IDS: being allowed to control the
PC or run dev tasks does not imply being allowed to see other users'
generation-history request text, and vice versa.

Currently used only by /character_missing (which can surface other users'
raw request text - see services/character_missing_finder.py). /character_info
/character_list/character_search/character_validate are read-only over the
dictionary itself (no per-user history), so they intentionally do not gate
on this.
"""

import logging
import os


logger = logging.getLogger("secretary-bot.character_dictionary_admin")


def get_character_admin_allowed_user_ids() -> set[int]:
    raw_value = os.getenv("CHARACTER_ADMIN_ALLOWED_USER_IDS", "")
    allowed_user_ids: set[int] = set()

    for raw_user_id in raw_value.split(","):
        user_id_text = raw_user_id.strip()
        if not user_id_text:
            continue

        try:
            allowed_user_ids.add(int(user_id_text))
        except ValueError:
            logger.warning("Invalid CHARACTER_ADMIN_ALLOWED_USER_IDS entry: %s", user_id_text)

    return allowed_user_ids


def is_character_admin_allowed(user_id: int) -> bool:
    return user_id in get_character_admin_allowed_user_ids()
