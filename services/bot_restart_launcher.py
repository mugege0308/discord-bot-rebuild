"""Detached, observable launcher used by ``/bot_restart`` on Windows."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROCESS_START_RE = re.compile(r"runtime_event=process_start\s+pid=(\d+)")
LOGIN_RE = re.compile(r"runtime_event=discord_login_success\b.*?pid=(\d+)")
SYNC_TEXT = "Slash Command sync succeeded"
SYNC_SKIPPED_TEXT = "Skipped Slash Command sync:"
PROCESS_STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258

STARTUP_TIMEOUT_SECONDS = 20.0
MONITOR_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.25
TAIL_LIMIT = 4000


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_timestamp()} {message}\n")
        handle.flush()


def _tail(path: Path, limit: int = TAIL_LIMIT) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<read_failed {type(exc).__name__}:{exc}>"
    return text[-limit:] if len(text) > limit else text


def _log_output_tail(restart_log: Path, label: str, path: Path) -> None:
    append_log(restart_log, f"{label}_file={path}")
    content = _tail(path).strip()
    if not content:
        append_log(restart_log, f"{label}_tail=<empty>")
        return
    for line in content.splitlines():
        append_log(restart_log, f"{label}_tail={line}")


def _runtime_log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_runtime_delta(path: Path, offset: int) -> tuple[list[str], int]:
    try:
        data = path.read_bytes()
    except OSError:
        return [], offset
    if len(data) < offset:
        offset = 0
    chunk = data[offset:]
    if not chunk:
        return [], len(data)
    return chunk.decode("utf-8", errors="replace").splitlines(), len(data)


def parse_runtime_markers(
    lines: list[str],
    new_pid: int | None,
    *,
    sync_detected: bool = False,
    login_detected: bool = False,
) -> tuple[int | None, str | None, bool, bool]:
    """Extract restart-relevant markers from appended runtime-log lines."""

    startup_line: str | None = None
    for line in lines:
        process_match = PROCESS_START_RE.search(line)
        if process_match is not None and new_pid is None:
            new_pid = int(process_match.group(1))
            startup_line = line
            continue
        if new_pid is None:
            continue
        if SYNC_TEXT in line or SYNC_SKIPPED_TEXT in line:
            sync_detected = True
        login_match = LOGIN_RE.search(line)
        if login_match is not None and int(login_match.group(1)) == new_pid:
            login_detected = True
    return new_pid, startup_line, sync_detected, login_detected


def _windows_process_handle(pid: int):
    if os.name != "nt" or pid <= 0:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x00100000 | 0x00001000, False, pid)
    return kernel32, handle


def wait_for_process_exit(pid: int, timeout_seconds: float) -> bool:
    """Wait for a process handle without polling the old Bot's event loop."""

    if os.name == "nt":
        result = _windows_process_handle(pid)
        if result is None:
            return True
        kernel32, handle = result
        if not handle:
            return True
        try:
            wait_result = kernel32.WaitForSingleObject(handle, int(timeout_seconds * 1000))
            return wait_result == WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = _windows_process_handle(pid)
        if result is None:
            return False
        kernel32, handle = result
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == PROCESS_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_restart_process(
    start_file: Path,
    project_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            ["cmd.exe", "/d", "/c", str(start_file)],
            cwd=str(project_dir),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=(
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ),
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


def run_restart(
    *,
    old_pid: int,
    start_file: Path,
    project_dir: Path,
    restart_log: Path,
    runtime_log: Path,
    stdout_log: Path,
    stderr_log: Path,
    startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    monitor_timeout: float = MONITOR_TIMEOUT_SECONDS,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> int:
    append_log(
        restart_log,
        f"restart_monitor_start old_pid={old_pid} start_file={start_file} "
        f"runtime_log={runtime_log}",
    )
    if not wait_for_process_exit(old_pid, startup_timeout):
        append_log(
            restart_log,
            f"old_pid_exit detected=false old_pid={old_pid} timeout_seconds={startup_timeout}",
        )
        _log_output_tail(restart_log, "launcher_stdout", stdout_log)
        _log_output_tail(restart_log, "launcher_stderr", stderr_log)
        append_log(restart_log, "final_restart_result=failed:old_pid_exit_timeout")
        return 1

    append_log(restart_log, f"old_pid_exit detected=true old_pid={old_pid}")
    append_log(
        restart_log,
        f"launcher_command=cmd.exe /d /c {start_file} project_dir={project_dir}",
    )
    # Take the baseline before starting the replacement process.  Otherwise a
    # very fast startup can append process_start between Popen() and the first
    # read, making the monitor report a false timeout.
    runtime_offset = _runtime_log_size(runtime_log)
    try:
        launcher = start_restart_process(start_file, project_dir, stdout_log, stderr_log)
    except Exception as exc:
        append_log(
            restart_log,
            f"launcher_process_start success=false error={type(exc).__name__}:{exc}",
        )
        _log_output_tail(restart_log, "launcher_stdout", stdout_log)
        _log_output_tail(restart_log, "launcher_stderr", stderr_log)
        append_log(restart_log, "final_restart_result=failed:launcher_start")
        return 1

    append_log(restart_log, f"launcher_process_start success=true pid={launcher.pid}")
    monitor_started = time.monotonic()
    startup_deadline = monitor_started + startup_timeout
    monitor_deadline = monitor_started + monitor_timeout
    new_pid: int | None = None
    startup_line: str | None = None
    sync_detected = False
    login_detected = False
    process_start_timeout_logged = False
    process_exit_logged = False

    while True:
        lines, runtime_offset = read_runtime_delta(runtime_log, runtime_offset)
        new_pid, detected_line, sync_detected, login_detected = parse_runtime_markers(
            lines,
            new_pid,
            sync_detected=sync_detected,
            login_detected=login_detected,
        )
        if detected_line is not None and startup_line is None:
            startup_line = detected_line
            append_log(
                restart_log,
                f"new_process_start detected=true new_pid={new_pid} startup_log={startup_line}",
            )

        now = time.monotonic()
        if new_pid is None and now >= startup_deadline:
            append_log(
                restart_log,
                f"new_process_start detected=false timeout_seconds={startup_timeout}",
            )
            process_start_timeout_logged = True
            break
        if new_pid is not None and not process_is_alive(new_pid):
            if not process_exit_logged:
                append_log(restart_log, f"new_process_exit detected=true new_pid={new_pid}")
                process_exit_logged = True
            break
        if now >= monitor_deadline:
            break
        time.sleep(poll_interval)

    alive_after_monitor = new_pid is not None and process_is_alive(new_pid)
    append_log(
        restart_log,
        f"sync_detected={sync_detected} login_detected={login_detected} "
        f"new_pid_alive_after_{int(monitor_timeout)}s={alive_after_monitor}",
    )
    if startup_line is None and not process_start_timeout_logged:
        append_log(
            restart_log,
            f"new_process_start detected=false timeout_seconds={startup_timeout}",
        )

    _log_output_tail(restart_log, "launcher_stdout", stdout_log)
    _log_output_tail(restart_log, "launcher_stderr", stderr_log)
    if new_pid is None:
        result = "failed:process_start_timeout"
    elif not alive_after_monitor:
        result = "failed:new_process_exited"
    elif not sync_detected or not login_detected:
        result = "failed:startup_incomplete"
    else:
        result = "success"
    append_log(restart_log, f"final_restart_result={result} new_pid={new_pid}")
    return 0 if result == "success" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--restart-log", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_restart(
        old_pid=args.old_pid,
        start_file=args.start_file,
        project_dir=args.project_dir,
        restart_log=args.restart_log,
        runtime_log=args.runtime_log,
        stdout_log=args.stdout_log,
        stderr_log=args.stderr_log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
