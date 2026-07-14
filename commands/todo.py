import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from services.weather import JST


logger = logging.getLogger("secretary-bot.todo")

DATA_DIR = Path("data")
TODOS_FILE = DATA_DIR / "todos.json"


def _now_text() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _normalize_todo(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    content = str(item.get("content", "")).strip()
    if not content:
        return None

    return {
        "content": content,
        "done": bool(item.get("done", False)),
        "created_at": item.get("created_at") if isinstance(item.get("created_at"), str) else _now_text(),
        "done_at": item.get("done_at") if isinstance(item.get("done_at"), str) else None,
    }


def _load_todos() -> dict[str, list[dict[str, Any]]]:
    if not TODOS_FILE.exists():
        return {}

    try:
        with TODOS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        logger.exception("Failed to decode todo data: %s", TODOS_FILE)
        return {}

    if not isinstance(data, dict):
        logger.error("Todo data is not a dictionary: %s", TODOS_FILE)
        return {}

    todos: dict[str, list[dict[str, Any]]] = {}
    for user_id, items in data.items():
        if not isinstance(items, list):
            continue

        normalized_items = []
        for item in items:
            normalized = _normalize_todo(item)
            if normalized is not None:
                normalized_items.append(normalized)

        todos[str(user_id)] = normalized_items

    return todos


def _save_todos(todos: dict[str, list[dict[str, Any]]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TODOS_FILE.open("w", encoding="utf-8") as file:
        json.dump(todos, file, ensure_ascii=False, indent=2)


def get_pending_todos(user_id: int) -> list[dict[str, Any]]:
    return [todo for todo in _load_todos().get(str(user_id), []) if not todo.get("done")]


def format_pending_todos(user_id: int) -> str:
    pending_todos = get_pending_todos(user_id)
    if not pending_todos:
        return "今のところ、やることは残ってなさそうだよ。\nえらいね。"

    lines = ["今のToDoだよ。", ""]
    lines.extend(f"{index}. {todo['content']}" for index, todo in enumerate(pending_todos, start=1))
    return "\n".join(lines)


def format_pending_todos_for_today(user_id: int) -> str:
    pending_todos = get_pending_todos(user_id)
    if not pending_todos:
        return "今のところ、やることは残ってなさそうだよ。\nえらいね。"

    return "\n".join(f"{index}. {todo['content']}" for index, todo in enumerate(pending_todos, start=1))


def handle_todo_add(content: str, user_id: int | None = None) -> str:
    todo_content = content.strip()
    if not todo_content:
        return "ToDoの内容を入力してね。"

    if user_id is None:
        logger.error("Cannot add todo because user_id is None.")
        return "ごめんね、ToDoを追加できなかったみたい。"

    todos = _load_todos()
    user_key = str(user_id)
    todos.setdefault(user_key, []).append(
        {
            "content": todo_content,
            "done": False,
            "created_at": _now_text(),
            "done_at": None,
        }
    )
    _save_todos(todos)

    return "ToDoに入れておいたよ。\nひとつずつ片づけていこ。"


def handle_todo_list(user_id: int | None = None) -> str:
    if user_id is None:
        logger.error("Cannot list todos because user_id is None.")
        return "ごめんね、ToDoを表示できなかったみたい。"

    return format_pending_todos(user_id)


def _find_pending_todo_index(todos: list[dict[str, Any]], number: int) -> int | None:
    pending_number = 0
    for index, todo in enumerate(todos):
        if todo.get("done"):
            continue

        pending_number += 1
        if pending_number == number:
            return index

    return None


def handle_todo_done(number: int, user_id: int | None = None) -> str:
    if user_id is None:
        logger.error("Cannot mark todo done because user_id is None.")
        return "ごめんね、ToDoを完了にできなかったみたい。"

    todos = _load_todos()
    user_key = str(user_id)
    user_todos = todos.get(user_key, [])
    todo_index = _find_pending_todo_index(user_todos, number)

    if todo_index is None:
        return "その番号のToDoは見つからなかったよ。"

    user_todos[todo_index]["done"] = True
    user_todos[todo_index]["done_at"] = _now_text()
    todos[user_key] = user_todos
    _save_todos(todos)

    return f"「{user_todos[todo_index]['content']}」を完了にしたよ。\nおつかれさま。"


def handle_todo_delete(number: int, user_id: int | None = None) -> str:
    if user_id is None:
        logger.error("Cannot delete todo because user_id is None.")
        return "ごめんね、ToDoを削除できなかったみたい。"

    todos = _load_todos()
    user_key = str(user_id)
    user_todos = todos.get(user_key, [])
    todo_index = _find_pending_todo_index(user_todos, number)

    if todo_index is None:
        return "その番号のToDoは見つからなかったよ。"

    del user_todos[todo_index]
    todos[user_key] = user_todos
    _save_todos(todos)

    return "指定のToDoを消しておいたよ。"
