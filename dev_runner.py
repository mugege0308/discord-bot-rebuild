import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_FILE = PROJECT_ROOT / "main.py"
EXCLUDED_DIRS = {"data", "__pycache__", ".git", ".venv", "venv"}
DEBOUNCE_SECONDS = 1.0
STOP_TIMEOUT_SECONDS = 10


class BotProcess:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        print("Botを起動します: python main.py", flush=True)
        self.process = subprocess.Popen([sys.executable, str(MAIN_FILE)], cwd=PROJECT_ROOT)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return

        print("Botを停止します...", flush=True)
        self.process.terminate()
        try:
            self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print("Botが終了しないため強制終了します。", flush=True)
            self.process.kill()
            self.process.wait(timeout=STOP_TIMEOUT_SECONDS)

    def restart(self) -> None:
        self.stop()
        self.start()


class PythonChangeHandler(FileSystemEventHandler):
    def __init__(self, bot_process: BotProcess) -> None:
        self.bot_process = bot_process
        self.last_restart_at = 0.0

    def on_any_event(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return

        path = Path(event.src_path)
        if not should_watch(path):
            return

        now = time.monotonic()
        if now - self.last_restart_at < DEBOUNCE_SECONDS:
            return

        self.last_restart_at = now
        relative_path = path.relative_to(PROJECT_ROOT)
        print(f"変更を検知しました: {relative_path}", flush=True)
        self.bot_process.restart()


def should_watch(path: Path) -> bool:
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False

    if path.suffix != ".py":
        return False

    return not any(part in EXCLUDED_DIRS for part in relative_path.parts)


def main() -> None:
    if not MAIN_FILE.exists():
        raise RuntimeError(f"main.py が見つかりません: {MAIN_FILE}")

    bot_process = BotProcess()
    event_handler = PythonChangeHandler(bot_process)
    observer = Observer()
    observer.schedule(event_handler, str(PROJECT_ROOT), recursive=True)

    print("開発用自動再起動ランナーを開始します。終了するには Ctrl+C を押してください。", flush=True)
    bot_process.start()
    observer.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("終了します...", flush=True)
    finally:
        observer.stop()
        observer.join()
        bot_process.stop()


if __name__ == "__main__":
    main()
