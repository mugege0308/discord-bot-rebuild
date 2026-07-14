import json
import logging
from pathlib import Path


logger = logging.getLogger("secretary-bot.memo")

DATA_DIR = Path("data")
MEMOS_FILE = DATA_DIR / "memos.json"


def _load_memos() -> dict[str, list[str]]:
    if not MEMOS_FILE.exists():
        return {}

    try:
        with MEMOS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        logger.exception("Failed to decode memo data: %s", MEMOS_FILE)
        return {}

    if not isinstance(data, dict):
        logger.error("Memo data is not a dictionary: %s", MEMOS_FILE)
        return {}

    memos: dict[str, list[str]] = {}
    for user_id, items in data.items():
        if isinstance(items, list):
            memos[str(user_id)] = [str(item) for item in items]

    return memos


def _save_memos(memos: dict[str, list[str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MEMOS_FILE.open("w", encoding="utf-8") as file:
        json.dump(memos, file, ensure_ascii=False, indent=2)


def get_memo_count(user_id: int) -> int:
    return len(_load_memos().get(str(user_id), []))


def handle_memo(bot_mention: str, command_args: str = "", user_id: int | None = None) -> str:
    memo_text = command_args.strip()
    if not memo_text:
        return f"メモする内容を教えてね。例: `{bot_mention} メモ 給料日は25日`"

    if user_id is None:
        logger.error("Cannot save memo because user_id is None.")
        return "ごめんね、メモを保存できなかったみたい。"

    memos = _load_memos()
    user_key = str(user_id)
    memos.setdefault(user_key, []).append(memo_text)
    _save_memos(memos)

    return "メモしておいたよ。\nあとで一緒に見返そうね。"


def handle_memo_list(bot_mention: str, command_args: str = "", user_id: int | None = None) -> str:
    if user_id is None:
        logger.error("Cannot list memos because user_id is None.")
        return "ごめんね、メモを表示できなかったみたい。"

    memos = _load_memos()
    user_memos = memos.get(str(user_id), [])
    if not user_memos:
        return "今のところ、メモはなさそうだよ。"

    return "\n".join(f"{index}. {memo}" for index, memo in enumerate(user_memos, start=1))


def handle_memo_delete(bot_mention: str, command_args: str = "", user_id: int | None = None) -> str:
    if user_id is None:
        logger.error("Cannot delete memo because user_id is None.")
        return "ごめんね、メモを削除できなかったみたい。"

    try:
        memo_number = int(command_args.strip())
    except ValueError:
        return f"削除するメモ番号を教えてね。例: `{bot_mention} メモ削除 1`"

    memos = _load_memos()
    user_key = str(user_id)
    user_memos = memos.get(user_key, [])

    if memo_number < 1 or memo_number > len(user_memos):
        return "その番号のメモは見つからなかったよ。メモ一覧で番号を確認してみてね。"

    del user_memos[memo_number - 1]
    memos[user_key] = user_memos
    _save_memos(memos)

    return "指定のメモを消しておいたよ。"
