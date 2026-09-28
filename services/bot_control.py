import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


logger = logging.getLogger("secretary-bot.bot_control")

ALLOWED_START_SUFFIXES = {".bat", ".cmd", ".exe"}
_restart_state_lock = threading.Lock()
_restart_requested = False


class BotAlreadyRunningError(RuntimeError):
    """別のBotプロセスがインスタンスロックを保持している。"""


class BotInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._handle = None

    def acquire(self) -> "BotInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as exc:
            handle.close()
            raise BotAlreadyRunningError(f"Bot instance lock is already held: {self.path}") from exc
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "BotInstanceLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


def get_bot_instance_lock_path() -> Path:
    configured = os.getenv("BOT_INSTANCE_LOCK_FILE", "").strip().strip('"')
    return (Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "bot_instance.lock").resolve()


def acquire_bot_instance_lock() -> BotInstanceLock:
    return BotInstanceLock(get_bot_instance_lock_path()).acquire()


def claim_restart_request() -> bool:
    global _restart_requested
    with _restart_state_lock:
        if _restart_requested:
            return False
        _restart_requested = True
        return True


def release_restart_request() -> None:
    global _restart_requested
    with _restart_state_lock:
        _restart_requested = False


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


@dataclass(frozen=True)
class BotRestartLaunch:
    """The detached restart monitor accepted before the current Bot exits."""

    start_file: Path
    monitor_pid: int
    restart_log: Path


def get_bot_restart_start_file() -> Path:
    """Return the non-interactive launcher used only by /bot_restart."""

    configured = os.getenv("BOT_RESTART_START_FILE", "").strip().strip('"')
    if configured:
        return Path(configured)
    return get_bot_start_file().with_name("start_bot_restart.bat")


def get_bot_restart_log_path(project_dir: Path | None = None) -> Path:
    configured = os.getenv("BOT_RESTART_LOG_FILE", "").strip().strip('"')
    if configured:
        return Path(configured).resolve()
    root = project_dir or Path(__file__).resolve().parents[1]
    return (root / "logs" / "bot_restart.log").resolve()


def _restart_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _append_restart_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_restart_timestamp()} {message}\n")
        handle.flush()


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _detached_creation_flags() -> int:
    return (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


def launch_bot_restart(*, wait_for_pid: int) -> BotRestartLaunch:
    """Start the detached restart monitor before the current Bot exits.

    The monitor waits for ``wait_for_pid`` to release the instance lock, then
    starts the dedicated no-pause launcher and records startup progress.  A
    Popen failure is raised synchronously so the caller can keep the current
    Bot alive.
    """

    if wait_for_pid <= 0:
        raise ValueError("wait_for_pid must be a positive process ID")

    start_file = get_bot_restart_start_file().resolve()
    validate_bot_start_file(start_file)
    cwd = start_file.parent
    restart_log = get_bot_restart_log_path(cwd)
    restart_log.parent.mkdir(parents=True, exist_ok=True)
    monitor_script = Path(__file__).with_name("bot_restart_launcher.py").resolve()
    if not monitor_script.is_file():
        raise RuntimeError(f"再起動monitorが見つかりません: {monitor_script}")

    command = [
        sys.executable,
        str(monitor_script),
        "--old-pid",
        str(wait_for_pid),
        "--start-file",
        str(start_file),
        "--project-dir",
        str(cwd),
        "--restart-log",
        str(restart_log),
        "--runtime-log",
        str(cwd / "logs" / "bot_runtime.log"),
        "--stdout-log",
        str(cwd / "logs" / "bot_restart.stdout.log"),
        "--stderr-log",
        str(cwd / "logs" / "bot_restart.stderr.log"),
    ]
    command_text = subprocess.list2cmdline(command)
    _append_restart_log(
        restart_log,
        f"restart_request timestamp={_restart_timestamp()} old_pid={wait_for_pid} "
        f"launcher_command={command_text}",
    )

    log_handle = restart_log.open("a", encoding="utf-8")
    try:
        monitor = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            creationflags=_detached_creation_flags(),
        )
    except Exception as exc:
        _append_restart_log(
            restart_log,
            f"launcher_process_start success=false error={type(exc).__name__}:{exc}",
        )
        raise
    finally:
        log_handle.close()

    _append_restart_log(
        restart_log,
        f"launcher_process_start success=true monitor_pid={monitor.pid} old_pid={wait_for_pid}",
    )
    logger.warning(
        "Bot restart monitor started: start_file=%s monitor_pid=%s old_pid=%s restart_log=%s",
        start_file,
        monitor.pid,
        wait_for_pid,
        restart_log,
    )
    return BotRestartLaunch(
        start_file=start_file,
        monitor_pid=monitor.pid,
        restart_log=restart_log,
    )


def _launch_after_pid_exits(path: Path, cwd: Path, wait_for_pid: int) -> None:
    quoted_path = _powershell_quote(str(path))
    quoted_cwd = _powershell_quote(str(cwd))
    if path.suffix.lower() in {".bat", ".cmd"}:
        start_command = (
            f"Start-Process -FilePath 'cmd.exe' -ArgumentList '/c',{quoted_path} "
            f"-WorkingDirectory {quoted_cwd} -WindowStyle Hidden"
        )
    else:
        start_command = f"Start-Process -FilePath {quoted_path} -WorkingDirectory {quoted_cwd} -WindowStyle Hidden"
    command = (
        f"$p=Get-Process -Id {int(wait_for_pid)} -ErrorAction SilentlyContinue; "
        "if($null -ne $p){$p.WaitForExit()}; "
        f"{start_command}"
    )
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_detached_creation_flags(),
    )


def launch_bot_start_file(*, wait_for_pid: int | None = None) -> Path:
    path = get_bot_start_file()
    validate_bot_start_file(path)
    cwd = path.parent

    logger.warning("Launching bot restart file: path=%s cwd=%s wait_for_pid=%s", path, cwd, wait_for_pid)
    if wait_for_pid is not None:
        _launch_after_pid_exits(path, cwd, wait_for_pid)
    elif path.suffix.lower() in {".bat", ".cmd"}:
        subprocess.Popen(
            ["cmd.exe", "/c", str(path)],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_detached_creation_flags(),
        )
    else:
        subprocess.Popen(
            [str(path)],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_detached_creation_flags(),
        )

    return path
