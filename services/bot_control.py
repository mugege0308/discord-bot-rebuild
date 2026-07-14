import logging
import os
import subprocess
from pathlib import Path


logger = logging.getLogger("secretary-bot.bot_control")

ALLOWED_START_SUFFIXES = {".bat", ".cmd", ".exe"}


def get_bot_start_file() -> Path:
    raw_value = os.getenv("BOT_START_FILE", "").strip().strip('"')
    if not raw_value:
        raise RuntimeError("BOT_START_FILE が未設定です。Bot起動用 .bat / .cmd / .exe のフルパスを .env に設定してください。")
    return Path(raw_value)


def validate_bot_start_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"BOT_START_FILE のファイルが見つかりません: {path}")
    if path.suffix.lower() not in ALLOWED_START_SUFFIXES:
        raise RuntimeError("BOT_START_FILE は .bat / .cmd / .exe だけ指定できます。")


def launch_bot_start_file() -> Path:
    path = get_bot_start_file()
    validate_bot_start_file(path)
    cwd = path.parent

    logger.warning("Launching bot restart file: path=%s cwd=%s", path, cwd)
    if path.suffix.lower() in {".bat", ".cmd"}:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "Shogou-chan Bot", str(path)],
            cwd=str(cwd),
            shell=False,
        )
    else:
        subprocess.Popen([str(path)], cwd=str(cwd), shell=False)

    return path
