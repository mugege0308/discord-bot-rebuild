import ctypes
import json
import logging
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("secretary-bot.claude_control")

STATE_FILE = Path("data/claude_process_state.json")
ALLOWED_START_SUFFIXES = {".bat", ".cmd"}
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
WINDOWS_EPOCH_TICKS = 116444736000000000
TICKS_PER_SECOND = 10_000_000
PROCESS_TIME_TOLERANCE_SECONDS = 2.0

_start_lock = threading.Lock()


class ClaudeControlError(RuntimeError):
    """An error whose message is safe to show in Discord."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    process_name: str
    creation_time: float | None


@dataclass(frozen=True)
class ClaudeProcessStatus:
    running: bool
    pid: int | None = None


@dataclass(frozen=True)
class ClaudeLaunchSpec:
    start_file: Path
    args: tuple[str, ...]
    cwd: Path
    creationflags: int


@dataclass(frozen=True)
class ClaudeStartResult:
    started: bool
    already_running: bool
    pid: int | None = None


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def _safe_start_error(message: str, *, path: Path | None = None) -> ClaudeControlError:
    if path is not None:
        logger.error("Claude start-file validation failed: path=%s reason=%s", path, message)
    else:
        logger.error("Claude start-file validation failed: reason=%s", message)
    return ClaudeControlError(message)


def get_claude_start_file() -> Path:
    raw_value = os.getenv("CLAUDE_START_FILE", "").strip().strip('"')
    if not raw_value:
        raise _safe_start_error("Claude Codeの起動設定がありません。管理者に確認してください。")
    return Path(raw_value)


def validate_claude_start_file(path: Path) -> Path:
    if platform.system() != "Windows":
        raise _safe_start_error("Claude Codeの起動機能はWindows専用です。")
    if not path.is_absolute():
        raise _safe_start_error("Claude Codeの起動設定が正しくありません。管理者に確認してください。", path=path)
    if path.suffix.lower() not in ALLOWED_START_SUFFIXES:
        raise _safe_start_error("Claude Codeの起動設定が正しくありません。管理者に確認してください。", path=path)
    if not path.is_file():
        raise _safe_start_error("Claude Codeの起動ファイルが見つかりません。管理者に確認してください。", path=path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        logger.exception("Failed to resolve Claude start file: path=%s", path)
        raise ClaudeControlError("Claude Codeの起動ファイルを確認できませんでした。管理者に確認してください。") from exc
    if resolved.suffix.lower() not in ALLOWED_START_SUFFIXES:
        raise _safe_start_error("Claude Codeの起動設定が正しくありません。管理者に確認してください。", path=resolved)
    return resolved


def _get_command_processor() -> Path:
    raw_value = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe").strip().strip('"')
    path = Path(raw_value)
    if not path.is_absolute() or path.name.lower() != "cmd.exe" or not path.is_file():
        logger.error("Invalid COMSPEC for Claude launch: %s", path)
        raise ClaudeControlError("Windowsのコマンド実行環境を確認できませんでした。")
    return path.resolve()


def build_claude_launch_spec() -> ClaudeLaunchSpec:
    start_file = validate_claude_start_file(get_claude_start_file())
    command_processor = _get_command_processor()
    return ClaudeLaunchSpec(
        start_file=start_file,
        args=(str(command_processor), "/d", "/c", "call", str(start_file)),
        cwd=start_file.parent,
        creationflags=CREATE_NEW_CONSOLE,
    )


def _filetime_to_unix(value: FILETIME) -> float:
    ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
    return (ticks - WINDOWS_EPOCH_TICKS) / TICKS_PER_SECOND


def _inspect_process(pid: int) -> ProcessIdentity | None:
    if platform.system() != "Windows" or pid <= 0:
        return None
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
        kernel32.GetProcessTimes.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != STILL_ACTIVE:
                return None

            buffer_size = wintypes.DWORD(32768)
            image_buffer = ctypes.create_unicode_buffer(buffer_size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, image_buffer, ctypes.byref(buffer_size)):
                return None

            creation = FILETIME()
            exit_time = FILETIME()
            kernel_time = FILETIME()
            user_time = FILETIME()
            creation_time = None
            if kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                creation_time = _filetime_to_unix(creation)
            return ProcessIdentity(pid=pid, process_name=Path(image_buffer.value).name.lower(), creation_time=creation_time)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        logger.exception("Failed to inspect Claude process identity: pid=%s", pid)
        return None


def _write_state_atomic(payload: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, STATE_FILE)


def _load_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read Claude process state JSON: path=%s", STATE_FILE)
        return None
    return value if isinstance(value, dict) else None


def _clear_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        logger.exception("Failed to clear Claude process state: path=%s", STATE_FILE)


def _state_matches_process(state: dict[str, Any], identity: ProcessIdentity) -> bool:
    stored_name = str(state.get("process_name") or "").strip().lower()
    if stored_name and stored_name != identity.process_name:
        logger.warning("Claude PID identity mismatch: pid=%s stored_name=%s actual_name=%s", identity.pid, stored_name, identity.process_name)
        return False
    try:
        stored_creation = float(state.get("process_creation_time"))
    except (TypeError, ValueError):
        stored_creation = None
    if stored_creation is not None and identity.creation_time is not None:
        if abs(stored_creation - identity.creation_time) > PROCESS_TIME_TOLERANCE_SECONDS:
            logger.warning(
                "Claude PID creation-time mismatch: pid=%s stored=%s actual=%s",
                identity.pid,
                stored_creation,
                identity.creation_time,
            )
            return False
    return True


def get_claude_status() -> ClaudeProcessStatus:
    state = _load_state()
    if not state:
        return ClaudeProcessStatus(running=False)
    try:
        pid = int(state.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    identity = _inspect_process(pid)
    if identity is None or not _state_matches_process(state, identity):
        _clear_state()
        return ClaudeProcessStatus(running=False)
    return ClaudeProcessStatus(running=True, pid=pid)


def start_claude_code() -> ClaudeStartResult:
    with _start_lock:
        current = get_claude_status()
        if current.running:
            return ClaudeStartResult(started=False, already_running=True, pid=current.pid)

        spec = build_claude_launch_spec()
        logger.info("Launching Claude Code: start_file=%s cwd=%s", spec.start_file, spec.cwd)
        try:
            process = subprocess.Popen(
                list(spec.args),
                cwd=str(spec.cwd),
                shell=False,
                creationflags=spec.creationflags,
            )
        except Exception as exc:
            logger.exception("Failed to launch Claude Code: start_file=%s", spec.start_file)
            raise ClaudeControlError("Claude Codeを起動できませんでした。Botのログを確認してください。") from exc

        identity = _inspect_process(process.pid)
        payload: dict[str, Any] = {
            "pid": process.pid,
            "process_name": identity.process_name if identity else "cmd.exe",
            "process_creation_time": identity.creation_time if identity else None,
            "launched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "launcher_file_name": spec.start_file.name,
        }
        try:
            _write_state_atomic(payload)
        except Exception:
            logger.exception("Claude Code started but process state could not be saved: pid=%s", process.pid)
            raise ClaudeControlError("Claude Codeは起動しましたが、状態を保存できませんでした。Botのログを確認してください。")
        logger.info("Claude Code launch recorded: pid=%s process_name=%s", process.pid, payload["process_name"])
        return ClaudeStartResult(started=True, already_running=False, pid=process.pid)
