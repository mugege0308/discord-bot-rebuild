import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse


logger = logging.getLogger("secretary-bot.jockie")

DATA_DIR = Path("data")
PLAYLISTS_FILE = DATA_DIR / "playlists.json"


def is_youtube_playlist_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    query = parse_qs(parsed.query)

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return "list" in query

    if host == "youtu.be":
        return "list" in query

    return False


def normalize_prefix(prefix: str | None) -> str:
    value = (prefix or "m!").strip()
    return value or "m!"


def build_jockie_shuffle_command(url: str, prefix: str | None = None) -> str:
    return f"{normalize_prefix(prefix)}play {url.strip()} --shuffle"


def build_jockie_shuffle_response(url: str, prefix: str | None = None) -> str:
    command = build_jockie_shuffle_command(url, prefix)
    fallback_play = f"{normalize_prefix(prefix)}play {url.strip()}"
    fallback_shuffle = f"{normalize_prefix(prefix)}shuffle"
    return "\n".join(
        [
            "Jockie Music用のコマンドを作ったよ。",
            "",
            "下のコマンドをコピーして送ってね。",
            "",
            "```text",
            command,
            "```",
            "",
            "うまくいかないときは、こっちも試してみてね。",
            "",
            "```text",
            fallback_play,
            fallback_shuffle,
            "```",
        ]
    )


def _load_playlists() -> dict[str, dict[str, str]]:
    if not PLAYLISTS_FILE.exists():
        return {}

    try:
        with PLAYLISTS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        logger.exception("Failed to decode playlists data: %s", PLAYLISTS_FILE)
        return {}

    if not isinstance(data, dict):
        logger.error("Playlists data is not a dictionary: %s", PLAYLISTS_FILE)
        return {}

    playlists: dict[str, dict[str, str]] = {}
    for user_id, user_playlists in data.items():
        if not isinstance(user_playlists, dict):
            continue
        playlists[str(user_id)] = {
            str(name): str(url)
            for name, url in user_playlists.items()
            if str(name).strip() and str(url).strip()
        }

    return playlists


def _save_playlists(playlists: dict[str, dict[str, str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PLAYLISTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(playlists, file, ensure_ascii=False, indent=2)


def save_playlist(user_id: int, name: str, url: str) -> str:
    playlist_name = name.strip()
    playlist_url = url.strip()

    if not playlist_name:
        return "保存名を入力してね。"

    if not is_youtube_playlist_url(playlist_url):
        return "YouTubeの再生リストURLを指定してね。`playlist?list=...` が含まれるURLが対象だよ。"

    playlists = _load_playlists()
    user_key = str(user_id)
    playlists.setdefault(user_key, {})[playlist_name] = playlist_url
    _save_playlists(playlists)

    return f"「{playlist_name}」として再生リストを保存したよ。"


def list_playlists(user_id: int) -> str:
    playlists = _load_playlists().get(str(user_id), {})
    if not playlists:
        return "保存済みの再生リストはまだなさそうだよ。"

    lines = ["保存済み再生リストだよ。"]
    lines.extend(f"{index}. {name}" for index, name in enumerate(playlists.keys(), start=1))
    return "\n".join(lines)


def get_user_playlists(user_id: int) -> dict[str, str]:
    return dict(_load_playlists().get(str(user_id), {}))


def delete_playlist(user_id: int, name: str) -> str:
    playlist_name = name.strip()
    if not playlist_name:
        return "削除する保存名を入力してね。"

    playlists = _load_playlists()
    user_key = str(user_id)
    user_playlists = playlists.get(user_key, {})

    if playlist_name not in user_playlists:
        return f"「{playlist_name}」は保存されていなさそうだよ。"

    del user_playlists[playlist_name]
    playlists[user_key] = user_playlists
    _save_playlists(playlists)

    return f"「{playlist_name}」を削除したよ。"


def build_saved_jockie_shuffle_response(user_id: int, name: str, prefix: str | None = None) -> str:
    playlist_name = name.strip()
    if not playlist_name:
        return "保存名を入力してね。"

    playlists = _load_playlists().get(str(user_id), {})
    url = playlists.get(playlist_name)
    if url is None:
        return f"「{playlist_name}」は保存されていなさそうだよ。/playlist_list で確認できるよ。"

    return build_jockie_shuffle_response(url, prefix)
